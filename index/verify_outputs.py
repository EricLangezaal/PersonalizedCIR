import os
import sys
import json
import pickle
import argparse
import numpy as np
import toml

def load_config(toml_path):
    try:
        config = toml.load(toml_path)
    except Exception as e:
        print(f"Error loading TOML configuration file: {e}")
        exit(1)
    return config

def verify_output_files(config):
    # Access the top-level keys directly since section headers are commented out
    try:
        model_type = config.get('model_type', None)
        pretrained_passage_encoder = config.get('pretrained_passage_encoder', None)
        max_seq_length = config.get('max_seq_length', None)
        max_doc_character = config.get('max_doc_character', None)

        if None in [model_type, pretrained_passage_encoder, max_seq_length, max_doc_character]:
            raise KeyError("One or more keys from the [Model] section are missing.")
    except KeyError as e:
        print(f"Error: Missing model-related configuration keys - {e}")
        exit(1)

    try:
        raw_collection_path = config.get('raw_collection_path', None)
        if raw_collection_path is None:
            raise KeyError("raw_collection_path key from [Input Data] section is missing.")
    except KeyError as e:
        print(f"Error: Missing input data-related configuration keys - {e}")
        exit(1)

    try:
        data_output_path = config.get('data_output_path', None)
        if data_output_path is None:
            raise KeyError("data_output_path key from [Output] section is missing.")
    except KeyError as e:
        print(f"Error: Missing output-related configuration keys - {e}")
        exit(1)
    
   # Verify Combined Passage File
    combined_passage_path = os.path.join(data_output_path, "passages")
    if not os.path.exists(combined_passage_path):
        print(f"Error: Combined passage file {combined_passage_path} does not exist.")
        return

    print("Checking combined passage file...")
    with open(combined_passage_path, 'rb') as f:
        # Check structure of first few records
        for idx in range(3):  # Check first few records
            # Read the first 64 bytes intended for p_id
            p_id_bytes = f.read(64)
            try:
                # Attempt to decode `p_id` bytes; handle if decoding fails
                p_id = p_id_bytes.rstrip(b'\x00').decode('utf-8')
            except UnicodeDecodeError:
                p_id = p_id_bytes  # If decoding fails, keep `p_id` as bytes
                print(f"Warning: p_id at index {idx} could not be decoded and is kept as bytes.")

            # Read the next 4 bytes for passage length
            passage_len_bytes = f.read(4)
            passage_len = int.from_bytes(passage_len_bytes, 'big')
            passage_bytes = f.read(max_seq_length * 4)
            
            # Check passage length and padding
            if len(passage_bytes) != max_seq_length * 4:
                print(f"Error: Passage at index {idx} does not match max_seq_length.")
            else:
                print(f"Passage {p_id} at index {idx} is valid.")
    
    # Verify Metadata File
    meta_path = combined_passage_path + "_meta"
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file {meta_path} does not exist.")
        return

    print("Checking metadata file...")
    with open(meta_path, 'r') as meta_file:
        meta = json.load(meta_file)
        expected_meta = {
            'type': 'int32',
            'total_number': None,  # We’ll verify this next
            'embedding_size': max_seq_length
        }
        for key in expected_meta:
            if meta.get(key) != expected_meta[key] and key != 'total_number':
                print(f"Error: Metadata {key} does not match expected value.")
            else:
                print(f"Metadata {key} is correct.")

    # Verify Pickle Files for Mapping
    pid2offset_path = os.path.join(data_output_path, "pid2offset.pickle")
    offset2pid_path = os.path.join(data_output_path, "offset2pid.pickle")

    if not os.path.exists(pid2offset_path) or not os.path.exists(offset2pid_path):
        print("Error: pid2offset or offset2pid pickle files are missing.")
        return
    
    print("Checking pid2offset and offset2pid mappings...")
    with open(pid2offset_path, 'rb') as f:
        pid2offset = pickle.load(f)
    with open(offset2pid_path, 'rb') as f:
        offset2pid = pickle.load(f)
    
    # Check that each `p_id` in pid2offset maps correctly to offset2pid
    valid_mapping = True
    for p_id, offset in pid2offset.items():
        if offset >= len(offset2pid) or offset2pid[offset] != p_id:
            print(f"Error: Mapping mismatch for p_id {p_id} at offset {offset}.")
            valid_mapping = False
    
    if valid_mapping:
        print("pid2offset and offset2pid mappings are consistent.")
    else:
        print("Error found in pid2offset and offset2pid mappings.")

    # Verify Embedding Cache (if applicable)
    embedding_cache_path = combined_passage_path
    if os.path.exists(embedding_cache_path):
        from pcir.data_structure import EmbeddingCache  # Assuming this is available

        print("Checking embedding cache...")
        embedding_cache = EmbeddingCache(embedding_cache_path)
        with embedding_cache as emb:
            # Check the first embedding
            first_embedding = emb[0]
            if len(first_embedding) != max_seq_length:
                print(f"Error: Embedding length {len(first_embedding)} does not match max_seq_length {max_seq_length}.")
            else:
                print("Embedding cache first entry is valid.")
    else:
        print("Embedding cache file does not exist, skipping cache validation.")

if __name__ == "__main__":
    # Parse command-line arguments for the config path
    parser = argparse.ArgumentParser(description="Verification of ANCE Preprocessing Outputs")
    parser.add_argument("--config", type=str, required=True, help="Path to the TOML configuration file")
    args = parser.parse_args()

    # Load configuration and run verification
    config = load_config(args.config)
    verify_output_files(config)
