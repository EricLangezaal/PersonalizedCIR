import sys
import os
# so we can succesfully import pcir
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('/', 1)[0])

import argparse
import gc
import pickle
import logging

import torch
import toml
from torch import nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

from pcir.data_structure import StreamingDataset, EmbeddingCache
from pcir.models import load_model

torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)

def GetProcessingFn(args, query=False):
    def fn(vals, i):
        passage_len, passage = vals
        max_len = 64 if query else args.max_seq_length

        pad_len = max(0, max_len - passage_len)
        token_type_ids = ([0] if query else [1]) * passage_len + [0] * pad_len
        attention_mask = [1] * passage_len + [0] * pad_len

        passage_collection = [(i, passage, attention_mask, token_type_ids)]

        query2id_tensor = torch.tensor(
            [f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor(
            [f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor(
            [f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor(
            [f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(
            all_input_ids_a,
            all_attention_mask_a,
            all_token_type_ids_a,
            query2id_tensor)

        return [ts for ts in dataset]

    return fn



import torch

def InferenceEmbeddingFromStreamDataLoader(
    args,
    model,
    train_dataloader,
    is_query_inference=True,
):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = max(1, args.n_gpu) * args.per_gpu_eval_batch_size

    # Inference!
    logging.info("***** Running ANN Embedding Inference *****")
    logging.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    tmp_n = 0
    expect_per_block_passage_num = 2500000  # Adjust if necessary
    block_size = expect_per_block_passage_num // eval_batch_size  # 1000
    block_id = 0
    total_write_passages = 0

    # Compute total number of steps
    total_steps = (args.total_passages + eval_batch_size - 1) // eval_batch_size

    # GPU device properties for memory monitoring
    device = torch.cuda.get_device_properties(0)
    total_memory = device.total_memory / 1024 / 1024 / 1024  # Convert to GB
    logging.info(f"Total GPU memory: {total_memory:.2f}GB")

    for batch in tqdm(train_dataloader,
                      desc="Inferencing",
                      disable=args.disable_tqdm,
                      total=total_steps,
                      position=0,
                      leave=True):

        idxs = batch[3].detach().numpy()  # [#B]
        batch = tuple(t.to(args.device) for t in batch)
        
        # Memory monitoring before processing
        logging.info("Memory usage before batch processing:")
        logging.info("Allocated: %fGB", torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
        logging.info("Reserved: %fGB", torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
        logging.info("Max Reserved: %fGB", torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long()
            }
            embs = model(inputs["input_ids"], inputs["attention_mask"])

        embs = embs.detach().cpu().numpy()

        # check for multi-chunk output for long sequence
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)

        tmp_n += 1
        if tmp_n % 500 == 0:
            logging.info("Processed {} batches...".format(tmp_n))

        if tmp_n % block_size == 0:
            embedding = np.concatenate(embedding, axis=0)
            embedding2id = np.concatenate(embedding2id, axis=0)
            emb_block_path = os.path.join(args.data_output_path, "passage_emb_block_{}.pb".format(block_id))
            with open(emb_block_path, 'wb') as handle:
                pickle.dump(embedding, handle, protocol=4)
            embid_block_path = os.path.join(args.data_output_path, "passage_embid_block_{}.pb".format(block_id))
            with open(embid_block_path, 'wb') as handle:
                pickle.dump(embedding2id, handle, protocol=4)
            total_write_passages += len(embedding)
            block_id += 1

            logging.info("Written {} passages so far...".format(total_write_passages))
            embedding = []
            embedding2id = []
            gc.collect()

        # Memory monitoring after processing each batch
        logging.info("Memory usage after batch processing:")
        logging.info("Allocated: %fGB", torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
        logging.info("Reserved: %fGB", torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
        logging.info("Max Reserved: %fGB", torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)

    if len(embedding) > 0:
        embedding = np.concatenate(embedding, axis=0)
        embedding2id = np.concatenate(embedding2id, axis=0)

        emb_block_path = os.path.join(args.data_output_path, "passage_emb_block_{}.pb".format(block_id))
        embid_block_path = os.path.join(args.data_output_path, "passage_embid_block_{}.pb".format(block_id))
        with open(emb_block_path, 'wb') as handle:
            pickle.dump(embedding, handle, protocol=4)
        with open(embid_block_path, 'wb') as handle:
            pickle.dump(embedding2id, handle, protocol=4)
        total_write_passages += len(embedding)
        block_id += 1

    logging.info("Total passages written: {}".format(total_write_passages))


# streaming inference
def StreamInferenceDoc(args,
                       model,
                       fn,
                       prefix,
                       f,
                       is_query_inference=True,
                       merge=True):
    inference_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(inference_dataset,
                                      batch_size=inference_batch_size)


    if args.local_rank != -1:
        dist.barrier()  # directory created

    InferenceEmbeddingFromStreamDataLoader(
        args,
        model,
        inference_dataloader,
        is_query_inference=is_query_inference,
        )



    logging.info("merging embeddings")


def generate_new_ann(args):

    _, model = load_model(args.model_type + "_Passage", args.pretrained_passage_encoder)
    model = model.to(args.device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids = list(range(args.n_gpu)))

    merge = False

    logging.info("***** inference of passages *****")
    passage_collection_path = os.path.join(args.tokenized_passage_collection_dir_path,
                                           "passages")
    passage_cache = EmbeddingCache(passage_collection_path)
    with passage_cache as emb:
        StreamInferenceDoc(
            args,
            model,
            GetProcessingFn(args, query=False),
            "passage_",
            emb,
            is_query_inference=False,
            merge=merge)
    logging.info("***** Done passage inference *****")



def ann_data_gen(args):

    logging.info("start generate ann data")
    generate_new_ann(args)

    if args.local_rank != -1:
        dist.barrier()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True)

    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.data_output_path)
    
    return args

def main():
    args = get_args()
    ann_data_gen(args)


if __name__ == "__main__":
    main()


# python gen_doc_embeddings.py --config gen_doc_embeddings.toml
