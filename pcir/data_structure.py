"""
NOTE: authors forgot to add this. We retrieved it from another repo of theirs at:
https://github.com/fengranMark/ConvRelExpand/blob/main/scripts/data_structure.py
We are still not sure if this is the supposed class, but at least the input/output arguments now match.
"""
import json
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
from torch.utils.data import IterableDataset
import torch.distributed as dist
import re

from pcir.utils import parse_relevant_ids, is_relevant, get_assessed_turn_ids

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)


class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                # print("yielding record")
                # print(rec)
                yield rec


class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(
                self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".
                format(key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number



@dataclass
class RewriteSample:
    sample_id: str
    rewrite: list


class ANCERewriteDataset(Dataset):
    def __init__(self, args, query_tokenizer, filename, split_rewrite=False):
        """
        Initializes the dataset by loading and processing the data.

        Args:
        - args: Argument parser with necessary attributes.
        - query_tokenizer: Tokenizer to encode queries.
        - filename (str): Path to the JSONL file.
        - split_rewrite (bool): Whether to split 'rewrite_utt_text' into multiple queries.
        """
        self.examples = []

        relevant_ids = get_assessed_turn_ids(year=args.year)

        with open(filename, 'r') as f:
            data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)

        logging.info("Loading {} data file...".format(filename))

        for i in trange(n):
            data[i] = json.loads(data[i])
            if 'id' in data[i]:
                sample_id = data[i]['id']
            else:
                sample_id = data[i]['sample_id']

            if not is_relevant(sample_id, relevant_ids):
                continue

            if 'output' in data[i]:
                rewrite_text = data[i]['output']
                rewrite_tokens = query_tokenizer.encode(rewrite_text, add_special_tokens=True)
                rewrite = rewrite_tokens
            elif 'rewrite_utt_text' in data[i]:
                rewrite_text = data[i]['rewrite_utt_text']
                if split_rewrite:
                    rewrite = self.split_and_tokenize_rewrite(rewrite_text, query_tokenizer)
                    if not rewrite:
                        logging.warning(f"Line {i+1} (Sample ID: {sample_id}): Failed to process 'rewrite_utt_text'.")
                        continue
                else:
                    rewrite_tokens = query_tokenizer.encode(rewrite_text, add_special_tokens=True)
                    rewrite = rewrite_tokens

            else:
                rewrite_text = data[i]['oracle_utt_text']
                rewrite_tokens = query_tokenizer.encode(rewrite_text, add_special_tokens=True)
                rewrite = rewrite_tokens


            self.examples.append(RewriteSample(sample_id, rewrite)) 


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def split_and_tokenize_rewrite(self, rewrite_text, tokenizer, max_queries=5):
        """
        Splits the rewrite_utt_text into up to max_queries distinct queries and tokenizes each.

        Args:
        - rewrite_text (str): The original rewrite_utt_text.
        - tokenizer: The tokenizer to encode queries.
        - max_queries (int): Maximum number of queries to process.

        Returns:
        - List[List[int]]: A list containing lists of token IDs for each query.
        """
        # Step 1: Remove quotation marks
        text = rewrite_text.replace('"', '').replace("“", "").replace("”", "")

        # Step 2: Remove numbering (e.g., "1. ", "2. ")
        text = re.sub(r'\d+\.\s*', '', text)

        # Step 3: Replace newlines with commas to standardize delimiters
        text = text.replace('\n', ', ')

        # Step 4: Split based on question marks followed by optional commas and whitespace
        if '?' in text:
            queries = re.split(r'\?,?\s*', text)
            queries = [q.strip() + '?' for q in queries if q.strip()]
        else:
            # Split based on commas if no question marks
            queries = [q.strip() for q in text.split(',') if q.strip()]
            # Optionally, append question marks to comma-separated queries
            # Uncomment the following line if you want to add question marks
            # queries = [q + '?' if not q.endswith('?') else q for q in queries]

        # Step 5: Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
            if len(unique_queries) == max_queries:
                break

        if not unique_queries:
            return None  # No valid queries found

        # Step 6: Tokenize each query
        tokenized_queries = []
        for query in unique_queries:
            token_ids = tokenizer.encode(query, add_special_tokens=True)
            tokenized_queries.append(token_ids)

        return tokenized_queries

    @staticmethod
    def get_collate_fn(args, split_rewrite=False):
        """
        Returns a collate function for the DataLoader.

        Args:
        - args: Argument parser with necessary attributes.
        - split_rewrite (bool): Whether rewrites are split into multiple queries.

        Returns:
        - function: Collate function.
        """

        def collate_fn(batch: list):
            """
            Collate function to handle batches with single or multiple tokenized rewrites.

            Args:
            - batch (list): List of RewriteSample instances.

            Returns:
            - dict: Dictionary containing batched data.
            """
            collated_dict = {
                "bt_sample_id": [],
                "bt_rewrite": [],  # List of lists of token IDs or List[List[List[int]]]
            }

            bt_sample_id = []
            bt_rewrite = []

            for example in batch:
                bt_sample_id.append(example.sample_id)
                bt_rewrite.append(example.rewrite)  # This is a list of token ID lists

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_rewrite"] = bt_rewrite  # List[List[List[int]]] if split_rewrite=True

            if not split_rewrite:
                # When not splitting, bt_rewrite is List[List[int]]
                # Pad and create masks
                bt_rewrite_padded = []
                bt_rewrite_mask = []
                for rewrite in bt_rewrite:
                    if not isinstance(rewrite, list):
                        logging.error(f"Expected rewrite to be a list of token IDs, got {type(rewrite)}")
                        raise TypeError(f"Expected rewrite to be a list of token IDs, got {type(rewrite)}")
                    padded, mask = pad_seq_ids_with_mask(rewrite, max_length=args.max_concat_length)
                    bt_rewrite_padded.append(padded)
                    bt_rewrite_mask.append(mask)

                collated_dict["bt_rewrite"] = bt_rewrite_padded
                collated_dict["bt_rewrite_mask"] = bt_rewrite_mask

                # Convert to tensors
                for key in ["bt_rewrite", "bt_rewrite_mask"]:
                    if not isinstance(collated_dict[key], list):
                        logging.error(f"collated_dict[{key}] is not a list: {type(collated_dict[key])}")
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)

            else:
                # When splitting, bt_rewrite is List[List[List[int]]]
                # Handle variable number of queries and variable query lengths
                # One approach is to flatten all queries and keep track of sample IDs
                # Alternatively, process them as lists in the embedding function
                # Here, we'll keep them as lists for flexibility
                pass  # No tensor conversion

            return collated_dict

        return collate_fn
    

def pad_seq_ids_with_mask(input_ids,
                            max_length,
                            pad_on_left=False,
                            pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    attention_mask = []

    if padding_length <= 0:
        input_ids = input_ids[-max_length:]
        attention_mask = [1] * max_length
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return input_ids, attention_mask
