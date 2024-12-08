import sys
import os
# so we can find pcir module
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('/', 1)[0])

import pickle
import argparse
import json
import gzip
from multiprocessing import Process

import torch
import toml
from torch.utils.data import TensorDataset
import numpy as np

from pcir.models import load_model
from pcir.data_structure import EmbeddingCache

torch.multiprocessing.set_sharing_strategy('file_system')


def pad_input_ids(input_ids, max_length, pad_on_left=False, pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids


def numbered_byte_file_generator(base_path, file_no, record_size):
    for i in range(file_no):
        with open('{}_split{}'.format(base_path, i), 'rb') as f:
            while True:
                b = f.read(record_size)
                if not b:
                    # eof
                    break
                yield b


def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):
    tokenizer, _ = load_model(args.model_type + "_Passage", args.pretrained_passage_encoder)

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f, \
            open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        first_line = False  # tsv with first line
        for idx, line in enumerate(in_f):
            if idx % num_process != i or first_line:
                first_line = False
                continue
            try:
                res = line_fn(args, line, tokenizer)
            except ValueError:
                print("Bad passage.")
            else:
                out_f.write(res)


def multi_file_process(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(target=tokenize_to_file,
                    args=(
                        args,
                        i,
                        num_process,
                        in_path,
                        out_path,
                        line_fn,
                    ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def preprocess(args):
    pid2offset = {}
    offset2pid = []
    in_passage_path = args.raw_collection_path

    out_passage_path = os.path.join(
        args.data_output_path,
        "passages",
    )

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return

    out_line_count = 0

    print('start passage file split processing')
    multi_file_process(
        args,
        32,
        in_passage_path,
        out_passage_path,
        PassagePreprocessingFn)

    print('start merging splits')
    with open(out_passage_path, 'wb') as f:
        for idx, record in enumerate(numbered_byte_file_generator(
                out_passage_path, 32, 64 + 4 + args.max_seq_length * 4)):  # Adjust size for `p_id`
            p_id = record[:64].rstrip(b'\x00').decode('utf-8')  # Decode `p_id` from bytes
            f.write(record[64:])  # Write the rest (excluding `p_id` bytes)
            pid2offset[p_id] = idx
            offset2pid.append(p_id)
            if idx < 3:
                print(str(idx) + " " + str(p_id))
            out_line_count += 1

    print("Total lines written: " + str(out_line_count))
    meta = {
        'type': 'int32',
        'total_number': out_line_count,
        'embedding_size': args.max_seq_length}
    with open(out_passage_path + "_meta", 'w') as f:
        json.dump(meta, f)
    embedding_cache = EmbeddingCache(out_passage_path)
    print("First line")
    with embedding_cache as emb:
        print(emb[0])

    pid2offset_path = os.path.join(
        args.data_output_path,
        "pid2offset.pickle",
    )
    offset2pid_path = os.path.join(
        args.data_output_path,
        "offset2pid.pickle",
    )
    with open(pid2offset_path, 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4)
    with open(offset2pid_path, 'wb') as handle:
        pickle.dump(offset2pid, handle, protocol=4)
    print("done saving pid2offset")


def PassagePreprocessingFn(args, line, tokenizer, title=False):
    line = line.strip()
    if not line:  # empty line
        raise ValueError

    ext = args.raw_collection_path[args.raw_collection_path.rfind("."):]
    passage = None
    if ext == ".jsonl":
        obj = json.loads(line)
        # `p_id` as a string
        p_id = obj["id"]
        # Use `contents` instead of `text`
        p_text = obj["contents"].rstrip()
        # `p_title` is empty because our collection has no titles
        # p_title = ""

        full_text = p_text[:args.max_doc_character]

        passage = tokenizer.encode(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_seq_length,
        )

    elif ext == ".tsv":
        try:
            line_arr = line.split('\t')
            p_id = line_arr[0]  # Keep `p_id` as a string
            if title:
                p_text = line_arr[2].rstrip().replace(' [SEP] ', ' ') + ' ' + line_arr[1].rstrip()
            else:
                p_text = line_arr[1].rstrip()
        except IndexError:  # split error
            raise ValueError  # empty passage
        else:
            full_text = p_text[:args.max_doc_character]
            passage = tokenizer.encode(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_seq_length,
            )

    else:
        raise TypeError("Unrecognized file type")

    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length)

    # Convert `p_id` to bytes
    p_id_bytes = p_id.encode('utf-8')
    p_id_bytes = p_id_bytes.ljust(64, b'\x00')[:64]  # Pad or truncate to 64 bytes

    return p_id_bytes + passage_len.to_bytes(4, 'big') + np.array(input_id_b, np.int32).tobytes()


def QueryPreprocessingFn(args, line, tokenizer):
    line_arr = line.split('\t')
    q_id = int(line_arr[0])

    passage = tokenizer.encode(
        line_arr[1].rstrip(),
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_query_length)
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)

    return q_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + np.array(input_id_b, np.int32).tobytes()


def GetProcessingFn(args, query=False):
    def fn(vals, i):
        passage_len, passage = vals
        max_len = args.max_query_length if query else args.max_seq_length

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


def GetTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = line_arr[1]  # Keep as a string
        neg_pids = line_arr[2].split(',')

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(
            args, query=True)(
            query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(
            args, query=False)(
            passage_cache[pos_pid], pos_pid)[0]

        pos_label = torch.tensor(1, dtype=torch.long)
        neg_label = torch.tensor(0, dtype=torch.long)

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(
                args, query=False)(
                passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2], pos_label)
            yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2], neg_label)

    return fn


def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = line_arr[1]  # Keep as a string
        neg_pids = line_arr[2].split(',')

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(
            args, query=True)(
            query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(
            args, query=False)(
            passage_cache[pos_pid], pos_pid)[0]

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(
                args, query=False)(
                passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2],
                   neg_data[0], neg_data[1], neg_data[2])

    return fn


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)

    # output dir check
    os.makedirs(args.data_output_path, exist_ok=True)

    return args


def main():
    args = get_args()
    preprocess(args)


if __name__ == '__main__':
    main()
