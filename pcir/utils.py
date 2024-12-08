"""
# NOTE: This was missing, added parts from:
# https://github.com/fengranMark/ConvRelExpand/blob/main/scripts/utils.py
Most code is our addition to properly extract duplicated code,
or to make hardcoded parts dynamic.
"""
from typing import Union
import json
import os
from collections import defaultdict
import random

import logging
import numpy as np
import random
import re
import pytrec_eval
import torch
import transformers
from openai import OpenAI

        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_sample_ids(output_path):
    """
    Loads sample id's to check if there are sample-id's left to process
    Useful for: 
    (1) Check if there are no samples skipped
    (2) avoid reprocessing if job / openAI interrupts
    """
    processed_sample_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as outfile:
            for line in outfile:
                try:
                    data = json.loads(line)
                    sample_id = data.get('sample_id')
                    if sample_id:
                        processed_sample_ids.add(sample_id)
                except json.JSONDecodeError:
                    print("Skipping")
    return processed_sample_ids


def get_assessed_turn_ids(path=None, year="2023"):
    """
    Get the ids of the turns that have been assessed
    We checked this to be the same as the hardcoded list they used originally.
    """
    if path is None:
        path = f"data/{year}-qrels.all-turns.txt"
    idxs = set()
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                idxs.add(parts[0].replace("_", "-"))
    return idxs


def has_used_provenance(ptkb_provenance, provenance_seen, number):
    for item in ptkb_provenance:
        if item in provenance_seen[number]:
            return True  
    return False 



def get_relevant_assessed_turn_ids(set176, new_ptkb=True, file=None, year="2023"):
    """
    Get relevant assessed turn ids based on provenance filtering.
    """
    if file is None:
        file = f"data/{year}_test_topics_flattened.jsonl"
    data_list = []
    relevant_entries = []
    provenance_seen = {}
    with open(file) as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    if not data_list:
        raise ValueError("Data list is empty.")
    
    # Optionally, remove any strict ordering checks
    # Previously: if data_list[0].get('sample_id') != '9-1-1':
    #              raise ValueError("Data list is not sorted. List should start with sample_id '9-1-1'.")
    # Removed to accommodate different sample_id formats

    for data in data_list:
        # number is conversation like 9-1 or 0
        number = str(data.get('number', ''))
        # sample_id is turn in conversation like 9-1-1 or 0-1
        sample_id = data.get('sample_id', '')
        ptkb_provenance = data.get('ptkb_provenance', [])
        
        # new_ptkb for modified methodology
        if new_ptkb:
            # Keep history of used provenance for each conversation
            if number not in provenance_seen:
                provenance_seen[number] = set()
            # Check if any provenance_ptkb item in current turn was already used in conversation
            if has_used_provenance(ptkb_provenance, provenance_seen, number):
                continue
            provenance_seen[number].update(ptkb_provenance)
        if sample_id not in set176:
            continue
        # Always filter out entries with empty ptkb_provenance list
        if not ptkb_provenance:
            continue
        relevant_entries.append(data['sample_id'])
    print(f'Found {len(relevant_entries)} relevant turns.')
    return relevant_entries


def parse_relevant_ids(relevant_ptkb_path='data/{YEAR}_test_topics_flattened.jsonl', subset=False, new_ptkb=False, verbose=True, year="2023"):
    relevant_ids = get_assessed_turn_ids(year=year)  
    if subset:
        if verbose:
            print("Filtering empty provenance is ON")
            print(f"Corrected methodology filtering is {'ON' if new_ptkb else 'OFF'}")
        relevant_ids = get_relevant_assessed_turn_ids(relevant_ids, new_ptkb=new_ptkb, file=relevant_ptkb_path, year=year)
    elif verbose:
        print("No filter applied to assessed 176 entries.")
    return relevant_ids

def is_relevant(qid, relevant_ids):
    if qid in relevant_ids:
        return True
    
    if len(qid.split("-")) == 4:
        return qid[:qid.rfind("-")] in relevant_ids
    return False

def ensure_unique_path(file_path):
    """
    No overwriting, add counter to filename if it already exists
    """
    base_name, ext = os.path.splitext(file_path)
    counter = 1  
    new_path = f"{base_name}_run{counter}{ext}"
    
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_name}_run{counter}{ext}"
    
    return new_path

def load_provenance_dict(file_path, type='LLM'):
    provenance_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            sample_id = data.get('sample_id', '')
            if type == 'LLM' or type == 'STR':
                select = data.get(f'LLM_select', '')
            elif type == 'None':
                select = []
            elif type == 'All':
                select = list(data.get(f'ptkb', '').keys())
            elif type == 'human':
                select = data.get(f'ptkb_provenance', '')
            provenance_dict[sample_id] = select
    return provenance_dict



def demonstrate(shot, random_examples=False, seed=None):
    """
    Generates a few-shot demonstration for the model.
    
    Parameters:
    - shot (int): Number of examples to use in the demonstration (1, 3, or 5).
    - random_examples (bool): Whether to select random examples instead of hardcoded ones.
    - seed (int): Random seed for reproducibility when selecting random examples.

    Returns:
    - str: A formatted demonstration text with few-shot examples.
    """
    # Hardcoded examples when `random_examples=False`
    hardcoded_demo = {
        1: ["1-1"],
        3: ["1-1", "1-2", "2-1"],
        5: ["1-1", "1-2", "2-1", "2-2", "7-1"]
    }
    
    demo_text = ''
    with open('data/2023_train_topics.json', 'r') as file:
        data = json.load(file)

        if random_examples:
            # Select random samples based on the provided seed
            if seed is not None:
                random.seed(seed)
            
            # Gather all available "number" fields
            available_numbers = [entry["number"] for entry in data]
            
            # Ensure we don't exceed the available numbers
            if shot > len(available_numbers):
                raise ValueError(f"Requested shot size ({shot}) exceeds available examples ({len(available_numbers)}).")

            selected_numbers = random.sample(available_numbers, shot)
        else:
            # Use hardcoded numbers for demonstration
            if shot not in hardcoded_demo:
                raise ValueError(f"Invalid shot size. Must be one of {list(hardcoded_demo.keys())}.")
            selected_numbers = hardcoded_demo[shot]

        # Generate the demonstration text from the selected numbers
        i = 0
        for entry in data:
            if entry['number'] in selected_numbers:
                turns = entry["turns"]
                demo_question = [i["utterance"] for i in turns]
                demo_rewrite = [i["resolved_utterance"] for i in turns]
                demo_response = [i["response"] for i in turns]
                demo_ptkb_prov = [i["ptkb_provenance"] for i in turns]
                qra = ''
                for q, rw, rp, prov in zip(demo_question, demo_rewrite, demo_response, demo_ptkb_prov):
                    qra += (
                        f"Question: {q}\n"
                        f"provenance: {str(prov)}\n"
                        f"Rewrite: {rw}\n"
                        f"Response: {rp}\n\n"
                    )
                i += 1
                demo_text += (
                    f"# Example {i}\n\n"
                    f"User's information: {str(entry['ptkb'])}\n\n"
                    f"{qra}"
                )
    
    return demo_text


def extract_numbers_from_dict(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            value_str = ' '.join(map(str, value))
        else:
            # Ensure the value is a string
            value_str = str(value)
        # Use regex to find all numbers in the value
        numbers = re.findall(r'\d+', value_str)
        output_dict[key] = numbers
    return output_dict

def get_output_path_trec(prefix, args):
    out = prefix
    if prefix == "ance_":
        test_file_path = args.test_file_path
        out += test_file_path.split("/")[-1].replace(".jsonl", "")
        if args.split_rewrite:
            out += "_split"
    else:
        test_file_path = args.input_query_path
        out += test_file_path.split("/")[-1].replace(".jsonl", "")
    if hasattr(args, "query_type"):
        out += "_qtype_" + args.query_type
    if hasattr(args, "test_type"):
        out += "_ttype_" + args.test_type
    if args.output_id:
        out += "_" + args.output_id
    
    # if 2023 is in the path, save it in the 2023 folder
    if "2023" in test_file_path:
        # check if the folder exists and build otherwise
        if not os.path.exists("data/results/trec_files/2023"):
            os.makedirs("data/results/trec_files/2023")
        out = f"data/results/trec_files/2023/{out}.trec"
    elif "2024" in test_file_path:
        if not os.path.exists("data/results/trec_files/2024"):
            os.makedirs("data/results/trec_files/2024")
        out = f"data/results/trec_files/2024/{out}.trec"
    else:
        out = f"data/results/trec_files/{out}.trec"
    return out

def get_avg_eval_results(res1, res2 = None):
    map_list = [v['map'] for v in res1.values()]
    mrr_list = [v['recip_rank'] for v in res1.values()]
    recall_20_list = [v['recall_20'] for v in res1.values()]
    recall_1000_list = [v['recall_1000'] for v in res1.values()]
    precision_20_list = [v['P_20'] for v in res1.values()]

    res2 = res2 if res2 else res1
    ndcg_3_list = [v['ndcg_cut_3'] for v in res2.values()]
    ndcg_5_list = [v['ndcg_cut_5'] for v in res2.values()]
    ndcg_1000_list = [v['ndcg_cut_1000'] for v in res2.values()]

    res = {
            "MRR": float(round(np.average(mrr_list)*100, 5)),
            "NDCG@3": float(round(np.average(ndcg_3_list)*100, 5)),
            "NDCG@5": float(round(np.average(ndcg_5_list)*100, 5)),
            "NDCG@1000": float(round(np.average(ndcg_1000_list)*100, 5)),
            "Precision@20": float(round(np.average(precision_20_list)*100, 5)),
            "Recall@20": float(round(np.average(recall_20_list)*100, 5)),
            "Recall@1000": float(round(np.average(recall_1000_list)*100, 5)),
            "MAP": float(round(np.average(map_list)*100, 5)),
    }
    return res

def calculate_trec_res_NDCG(run_file, qrel_file, rel_threshold, automatic_method=False, subset=False, new_ptkb=False, relevant_ptkb_path=None, year="2023"):

    if relevant_ptkb_path is None:
        relevant_ptkb_path = f"data/{year}_test_topics_flattened.jsonl"
    
    relevant_ids = parse_relevant_ids(relevant_ptkb_path, subset, new_ptkb, year=year)
    
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = defaultdict(dict)
    qrels_ndcg = defaultdict(dict)
    
    # Load ground truth relevance judgments
    for line in qrel_data:
        line = line.strip().split()
        query = line[0].replace('_', '-')
        passage = line[2]
        rel = int(line[3])

        # For NDCG
        qrels_ndcg[query][passage] = rel
        # For MAP, MRR, Recall
        qrels[query][passage] = int(rel >= rel_threshold)

    eval_general = lambda: pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.1000", "P_20"})
    eval_ndcg = lambda: pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3", "ndcg_cut.5", "ndcg_cut.1000"})

    with open(run_file, 'r') as f:
        run_data = f.readlines()

    if automatic_method:
        run_per_ptkb = defaultdict(lambda: defaultdict(dict))
        for line in run_data:
            line = line.split()
            query = line[0]
            ptkb = query.split("-")[-1]
            qid = query[:query.rfind("-")]
            if qid not in relevant_ids:
                continue
            pktb_run = run_per_ptkb[ptkb]
            passage = line[2]
            rel = float(line[4])
            pktb_run[qid][passage] = rel

        # Evaluate runs
        result_dict = defaultdict(lambda: defaultdict(dict))
        for ptkb, run in run_per_ptkb.items():
            # Compute general metrics
            res_general = eval_general().evaluate(run)
            for qid, metrics in res_general.items():
                result_dict[qid][ptkb].update(metrics)
            # Compute NDCG metrics
            res_ndcg = eval_ndcg().evaluate(run)
            for qid, metrics in res_ndcg.items():
                result_dict[qid][ptkb].update(metrics)

        best_result_dict_baseline = {}
        best_result_dict_always_ptkb = {}

        for qid, pktb_dict in result_dict.items():
            # method 1: Using "no ptkb" (index 0) can be selected as it can give the highest NDCG@3
            best_metric_baseline = max(
                pktb_dict.values(),
                key=lambda metrics: metrics.get("ndcg_cut_3", 0)
            )
            best_result_dict_baseline[qid] = best_metric_baseline

            # method 2: always select a ptkb even if it's worse than using "no ptkb"
            always_ptkb = {ptkb: metrics for ptkb, metrics in pktb_dict.items() if ptkb != "0"}
            best_metric_always_ptkb = max(
                always_ptkb.values(),
                key=lambda metrics: metrics.get("ndcg_cut_3", 0)
            )
            best_result_dict_always_ptkb[qid] = best_metric_always_ptkb

        total_result_baseline = get_avg_eval_results(best_result_dict_baseline)
        total_result_always_ptkb = get_avg_eval_results(best_result_dict_always_ptkb)

        logging.info("Possibly no ptkb results: %s", total_result_baseline)
        logging.info("Always a ptkb query results: %s", total_result_always_ptkb)

        return total_result_baseline, total_result_always_ptkb

    else:
        # Evaluate without automatic method
        runs = defaultdict(dict)
        for line in run_data:
            line = line.split()
            query = line[0]
            if query not in relevant_ids:
                continue
            passage = line[2]
            rel = float(line[4])
            runs[query][passage] = rel

        res1 = eval_general().evaluate(runs)
        res2 = eval_ndcg().evaluate(runs)
        total_result = get_avg_eval_results(res1, res2)

        logging.info("---------------------Evaluation results:---------------------")
        logging.info(total_result)
        return total_result


def get_run_files(base_name, directory, num_runs=5, suffix=''):
    run_files = []
    for i in range(1, num_runs + 1):
        run_file = os.path.join(directory, f"{base_name}_run{i}{suffix}.trec")
        if not os.path.exists(run_file):
            logging.warning(f"Run file does not exist: {run_file}")
            return None
        run_files.append(run_file)
    return run_files

def init_llm(model_id:str) -> Union[str, transformers.Pipeline]:
    if "llama" in model_id:
        model = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        set_seed(42)
        return model
    else:
        return model_id

def query_llm(system_msg:str, content:str, model:Union[str, transformers.Pipeline]) -> str:

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": content}
    ]
    if isinstance(model, str):
        r = OpenAI().chat.completions.create(
            model=model,
            messages=messages
        )
        content = r.choices[0].message.content
    elif isinstance(model, transformers.Pipeline):
        outputs = model(messages, max_new_tokens=500)
        content = outputs[0]["generated_text"][-1]["content"]
    else:
        raise ValueError("Invalid model engine.")   
    return content