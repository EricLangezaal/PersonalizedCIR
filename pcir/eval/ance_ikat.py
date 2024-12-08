import logging
import argparse
import time
import copy
import pickle
import os
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
import faiss
import torch
import toml
import numpy as np
import pytrec_eval
from transformers import RobertaConfig, RobertaTokenizer

from models import ANCE
# NOTE: originally this was import "test_cast23_rewrite" but that does not exist
from data_structure import ANCERewriteDataset, pad_seq_ids_with_mask
from pcir.utils import set_seed, get_output_path_trec, calculate_trec_res_NDCG

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)

def load_config(file_path, year):
    config_dict = toml.load(file_path)["test_config"]
    # Replace placeholders with actual values
    for key, value in config_dict.items():
        if isinstance(value, str) and "{YEAR}" in value:
            config_dict[key] = value.replace("{YEAR}", year)
    return argparse.Namespace(**config_dict)

'''
Test process, perform dense retrieval on collection (e.g., MS MARCO):
1. get args
2. establish index with Faiss on GPU for fast dense retrieval
3. load the model, build the test query dataset/dataloader, and get the query embeddings. 
4. iteratively searched on each passage block one by one to got the retrieved scores and passge ids for each query.
5. merge the results on all pasage blocks
6. output the result
'''

def build_faiss_index(args):
    logging.info("Building index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.n_gpu
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(768)  
    index = None
    if args.use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    return index


def search_one_by_one_with_faiss(args, passge_embeddings_dir, index, query_embeddings, topN):
    merged_candidate_matrix = None
    if args.passage_block_num < 0:
        # automaticall get the number of passage blocks
        for filename in os.listdir(passge_embeddings_dir):
            try:
                args.passage_block_num = max(args.passage_block_num, int(filename.split(".")[1]) + 1)
            except:
                continue
        print("Automatically detect that the number of doc blocks is: {}".format(args.passage_block_num))
    for block_id in range(args.passage_block_num):
        logging.info("Loading passage block " + str(block_id))
        passage_embedding = None
        passage_embedding2id = None
        
        with open(os.path.join(passge_embeddings_dir, "passage_emb_block_{}.pb".format(block_id)), 'rb') as handle:
            passage_embedding = pickle.load(handle)
        with open(os.path.join(passge_embeddings_dir, "passage_embid_block_{}.pb".format(block_id)), 'rb') as handle:
            passage_embedding2id = pickle.load(handle)
            if isinstance(passage_embedding2id, list):
                passage_embedding2id = np.array(passage_embedding2id)
        
        logging.info('passage embedding shape: ' + str(passage_embedding.shape))
        logging.info("query embedding shape: " + str(query_embeddings.shape))

        passage_embeddings = np.array_split(passage_embedding, args.num_split_block)
        passage_embedding2ids = np.array_split(passage_embedding2id, args.num_split_block)
        for split_idx in range(len(passage_embeddings)):
            passage_embedding = passage_embeddings[split_idx]
            passage_embedding2id = passage_embedding2ids[split_idx]
            
            logging.info("Adding block {} split {} into index...".format(block_id, split_idx))
            index.add(passage_embedding)
            
            # ann search
            tb = time.time()
            D, I = index.search(query_embeddings, topN)
            elapse = time.time() - tb
            logging.info({
                'time cost': elapse,
                'query num': query_embeddings.shape[0],
                'time cost per query': elapse / query_embeddings.shape[0]
            })

            candidate_id_matrix = passage_embedding2id[I] # passage_idx -> passage_id
            D = D.tolist()
            candidate_id_matrix = candidate_id_matrix.tolist()
            candidate_matrix = []

            for score_list, passage_list in zip(D, candidate_id_matrix):
                candidate_matrix.append([])
                for score, passage in zip(score_list, passage_list):
                    candidate_matrix[-1].append((score, passage))
                assert len(candidate_matrix[-1]) == len(passage_list)
            assert len(candidate_matrix) == I.shape[0]

            index.reset()
            del passage_embedding
            del passage_embedding2id

            if merged_candidate_matrix == None:
                merged_candidate_matrix = candidate_matrix
                continue
            
            # merge
            merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
            merged_candidate_matrix = []
            for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                            candidate_matrix):
                p1, p2 = 0, 0
                merged_candidate_matrix.append([])
                while p1 < topN and p2 < topN:
                    if merged_list[p1][0] >= cur_list[p2][0]:
                        merged_candidate_matrix[-1].append(merged_list[p1])
                        p1 += 1
                    else:
                        merged_candidate_matrix[-1].append(cur_list[p2])
                        p2 += 1
                while p1 < topN:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                while p2 < topN:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1

    merged_D, merged_I = [], []

    for merged_list in merged_candidate_matrix:
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)

    logging.info(merged_D.shape)
    logging.info(merged_I.shape)

    return merged_D, merged_I


def get_test_query_embedding(args):
    set_seed(args.seed)

    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)
    model = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)

    # Set model to evaluation mode
    model.eval()

    # Test dataset/dataloader
    args.batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    logging.info("Building test dataset...")
    test_dataset = ANCERewriteDataset(args, tokenizer, args.test_file_path, split_rewrite=args.split_rewrite)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=test_dataset.get_collate_fn(args, split_rewrite=args.split_rewrite)
    )

    logging.info("Generating query embeddings for testing...")
    model.zero_grad()

    embeddings = []
    embedding2id = []

    if args.split_rewrite:
        logging.info("Splitting and averaging rewrite utterances with batch processing...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, disable=args.disable_tqdm, desc="Generating embeddings")):
                bt_sample_ids = batch["bt_sample_id"]  # List of sample IDs
                bt_rewrite = batch["bt_rewrite"]        # List of lists of token IDs

                # Lists to hold all queries and their corresponding sample indices
                all_queries = []
                query_sample_mapping = []  # Maps each query to its sample index

                for sample_idx, rewrite_queries in enumerate(bt_rewrite):
                    for query in rewrite_queries:
                        all_queries.append(query)
                        query_sample_mapping.append(sample_idx)

                if not all_queries:
                    logging.warning(f"Batch {batch_idx + 1}: No queries found.")
                    continue

                # Determine the maximum query length in this batch for padding
                max_query_length = max(len(q) for q in all_queries)
                logging.debug(f"Batch {batch_idx + 1}: Maximum query length is {max_query_length}.")

                # Pad all queries to the maximum length
                padded_queries = []
                attention_masks = []
                for q_idx, q in enumerate(all_queries):
                    padded, mask = pad_seq_ids_with_mask(q, max_length=max_query_length)
                    padded_queries.append(padded)
                    attention_masks.append(mask)
                    logging.debug(f"Batch {batch_idx + 1}, Query {q_idx + 1}: Padded query to length {max_query_length}.")

                # Convert lists to tensors
                input_ids = torch.tensor(padded_queries).to(args.device)          # Shape: [total_queries, max_query_length]
                attention_mask = torch.tensor(attention_masks).to(args.device)    # Shape: [total_queries, max_query_length]

                # Forward pass through the model to get embeddings
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logging.debug(f"Batch {batch_idx + 1}: outputs type: {type(outputs)}")
                logging.debug(f"Batch {batch_idx + 1}: outputs shape: {outputs.shape}")

                # Assuming that the model returns a single tensor, compute the mean of the embeddings
                query_embeddings = outputs.cpu().numpy()  # Shape: [total_queries, hidden_size]

                logging.debug(f"Batch {batch_idx + 1}: Obtained query embeddings with shape {query_embeddings.shape}.")

                # Initialize a list to accumulate embeddings per sample
                sample_embeddings = [[] for _ in range(len(bt_sample_ids))]

                # Assign embeddings to their respective samples
                for q_idx, sample_idx in enumerate(query_sample_mapping):
                    sample_embeddings[sample_idx].append(query_embeddings[q_idx])

                # Compute mean embeddings per sample
                for sample_idx, sample_id in enumerate(bt_sample_ids):
                    if sample_embeddings[sample_idx]:
                        mean_embedding = np.mean(sample_embeddings[sample_idx], axis=0)
                        # Optionally, normalize the embedding
                        # norm = np.linalg.norm(mean_embedding)
                        # if norm > 0:
                        #     mean_embedding /= norm
                        embeddings.append(mean_embedding)
                        logging.debug(f"Sample {batch_idx * args.batch_size + sample_idx + 1}: Computed mean embedding.")
                    else:
                        # If no embeddings were obtained, append a zero vector
                        hidden_size = model.config.hidden_size
                        embeddings.append(np.zeros(hidden_size))
                        logging.warning(f"Sample {batch_idx * args.batch_size + sample_idx + 1}: No embeddings obtained; appended zero vector.")

                    # Append sample ID
                    embedding2id.append(sample_id)

                logging.info(f"Processed batch {batch_idx + 1}/{len(test_loader)}: {len(bt_sample_ids)} embeddings added.")

    else:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, disable=args.disable_tqdm, desc="Generating embeddings")):
                bt_sample_ids = batch["bt_sample_id"]  # List of sample IDs
                # Determine the test type and prepare inputs
                if args.test_type == "rewrite":
                    input_ids = batch["bt_rewrite"].to(args.device)
                    input_masks = batch["bt_rewrite_mask"].to(args.device)
                elif args.test_type == "reformulate":
                    input_ids = batch["bt_rewrite"].to(args.device)
                    input_masks = batch["bt_rewrite_mask"].to(args.device)
                elif args.test_type == "convq":
                    input_ids = batch["bt_conv_q"].to(args.device)
                    input_masks = batch["bt_conv_q_mask"].to(args.device)
                elif args.test_type == "convqa":
                    input_ids = batch["bt_conv_qa"].to(args.device)
                    input_masks = batch["bt_conv_qa_mask"].to(args.device)
                elif args.test_type == "convqp":
                    input_ids = batch["bt_conv_qp"].to(args.device)
                    input_masks = batch["bt_conv_qp_mask"].to(args.device)
                else:
                    raise ValueError("test type:{}, has not been implemented.".format(args.test_type))
                
                # Forward pass through the model to get embeddings
                query_embs = model(input_ids, input_masks)
                query_embs = query_embs.detach().cpu().numpy()
                embeddings.append(query_embs)
                embedding2id.extend(bt_sample_ids)

                logging.info(f"Processed batch {batch_idx + 1}/{len(test_loader)}: {query_embs.shape[0]} embeddings added.")


    # Concatenate all embeddings into a single NumPy array
    if args.split_rewrite:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.concatenate(embeddings, axis=0)
    torch.cuda.empty_cache()

    logging.info(f"Total embeddings generated: {embeddings.shape}")

    return embeddings, embedding2id




def output_test_res(query_embedding2id,
                    retrieved_scores_mat,  # score matrix: test_query_num x top_k
                    retrieved_pid_mat,     # pid matrix: test_query_num x top_k (indices)
                    offset2pid,            # mapping from indices to passage IDs
                    args):

    qids_to_ranked_candidate_passages = {}
    topN = args.top_k

    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]

        top_ann_indices = retrieved_pid_mat[query_idx]
        top_ann_scores = retrieved_scores_mat[query_idx]

        selected_ann_indices = top_ann_indices[:topN]
        selected_ann_scores = top_ann_scores[:topN]

        if query_id not in qids_to_ranked_candidate_passages:
            qids_to_ranked_candidate_passages[query_id] = []

        rank = 0
        for idx, score in zip(selected_ann_indices, selected_ann_scores):
            pred_pid = offset2pid[idx]  # Map index to actual passage ID
            if pred_pid not in seen_pid:
                qids_to_ranked_candidate_passages[query_id].append((pred_pid, score))
                seen_pid.add(pred_pid)
                rank += 1
                if rank >= topN:
                    break

    # Write to TREC file
    logging.info('begin to write the output...')

    with open(args.output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            # Adjust query ID format if necessary
            if "cast21" in args.test_file_path:
                qid = qid.replace("-", "_")
            for rank, (pid, score) in enumerate(passages, start=1):
                g.write(f"{qid} Q0 {pid} {rank} {score} ance\n")
    logging.info("output file write ok at {}".format(args.output_trec_file))
    

def gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id):
    # Load offset2pid mapping
    with open(args.passage_offset2pid_path, "rb") as f:
        offset2pid = pickle.load(f)
    
    # Retrieve scores and passage indices
    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(
                                                        args,
                                                        args.passage_embeddings_dir_path, 
                                                        index, 
                                                        query_embeddings, 
                                                        args.top_k) 
    
    # Use offset2pid to map indices to passage IDs
    output_test_res(query_embedding2id,
                    retrieved_scores_mat,
                    retrieved_pid_mat,
                    offset2pid,
                    args)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run dense retrieval with a specified TOML configuration file.")
    parser.add_argument("--config", type=str, default="pcir/eval/test_ikat_config.toml", help="Path to the TOML configuration file")
    parser.add_argument("--test_file_path", type=str, default=None, help="Path to the test file")
    parser.add_argument("--split_rewrite", action='store_true', help="Enable splitting and averaging of 'rewrite_utt_text' into multiple queries")
    parser.add_argument("--year", type=str, default="2023", help="Year of the dataset")
    parser.add_argument("--automatic_method",action="store_true", help="Whether to use automatic method")
    args = parser.parse_args()
    # args.output_trec_file = get_output_path_trec("ance_", args)
    return args

def main():
    # Get the TOML file path from the command-line argument
    cl_args = parse_arguments()
    # Load configuration from the specified TOML file
    args = load_config(cl_args.config, cl_args.year)
    args.test_file_path = cl_args.test_file_path
    args.split_rewrite = cl_args.split_rewrite
    args.year = cl_args.year
    args.automatic_method = cl_args.automatic_method

    args.output_trec_file = get_output_path_trec("ance_", args)
    set_seed(args.seed)
    
    # Use GPU if specified
    args.device = torch.device("cuda:0" if args.use_gpu else "cpu")
    
    logging.info("---------------------The configuration is:---------------------")
    logging.info(args)
    
    # Build the index and perform retrieval operations
    index = build_faiss_index(args)
    query_embeddings, query_embedding2id = get_test_query_embedding(args)
    gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id)

    print("="*200)
    print("full set results")
    calculate_trec_res_NDCG(args.output_trec_file, args.trec_gold_qrel_file_path, args.rel_threshold, args.automatic_method, year=args.year)
    print("Subset results")
    calculate_trec_res_NDCG(args.output_trec_file, args.trec_gold_qrel_file_path, args.rel_threshold, args.automatic_method, True, year=args.year)
    print("Corrected subset results")
    calculate_trec_res_NDCG(args.output_trec_file, args.trec_gold_qrel_file_path, args.rel_threshold, args.automatic_method, True, True, year=args.year)



if __name__ == '__main__':
    main()
