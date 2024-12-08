import json
import argparse
import os
import sys
from openai import OpenAI
from pcir.utils import get_assessed_turn_ids, demonstrate, ensure_unique_path

client = OpenAI()

def append_batch_id_mapping(folder, filename, batch_id):
    if "2024" in folder:
        mapping_file = os.path.join(folder, "batch_id_mapping_SAR_2024.json")
    else:
        mapping_file = os.path.join(folder, "batch_id_mapping_SAR_2023.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            batch_id_mapping = json.load(f)
    else:
        batch_id_mapping = {}
    batch_id_mapping[filename] = batch_id
    with open(mapping_file, "w") as f:
        json.dump(batch_id_mapping, f, indent=4)

def prepare_batch_file(input_path, batch_file_path, model_engine, shot, random_examples, seed):
    if shot in [1, 3, 5]:
        d = demonstrate(shot, random_examples, seed)
    elif shot == 0:
        d = ''
    else:
        print("Error: --shot must be 0, 1, 3 or 5.")
        sys.exit(1)

    if "2024_test_topics" in input_path:
        assessed_turns = get_assessed_turn_ids(path="data/2024-qrels.trec")
    else:
        assessed_turns = get_assessed_turn_ids(path="data/2023-qrels.all-turns.txt")
    batch_requests = []
    with open(input_path) as infile:
        for line in infile:
            data = json.loads(line)
            sample_id = data.get("sample_id", "")

            if sample_id not in assessed_turns:
                continue

            ptkb = data.get('ptkb', '')
            cur_utt = data.get('cur_utt_text','')
            cur_resp = data.get('cur_response_text','')
            conv = data.get('ctx_utts_text','')
            response_list = data.get('response', '')
            provenance = data.get('ptkb_provenance','')
            r_utterance = data.get('oracle_utt_text','')
            history_response = data.get('history_response','')
            turn = data.get('number', '')
            last_response = data.get('last_response','')
        
            if conv == []:
                conv = cur_utt
            YourTask = '#Your Task (only user\'s information, questions and the response are given):\n\n' + 'User\'s information: ' + str(ptkb) + '\n\n'
            for q,a in zip(conv,history_response):
                YourTask += f'Question: {q}\nResponse: {a}\n\n'
            YourTask += 'Question: ' + cur_utt + '\n\n'

            prompt = f"For an information-seeking dialog, please help reformulate the question " \
                     "into a rewrite that can fully express the user's information needs without the need for context, "\
                     "but also generate an informative response to answer the question. "\
                     "You can generate a rewrite and response based on the user's personal information."
            if d:
                prompt += f" I will provide you with some examples:\n {d}\n\n"
            prompt += YourTask + \
                    "(Now, you should give me the rewrite and an informative response of the **Current Question** based on the **Context**. "\
                    "Please provide a complete informative response, but keep it under 200 words. "\
                    "The output format should always be: Provenance: $The user information number you are using\nRewrite: $Rewrite\nResponse: $Response. Go ahead!)"


            batch_requests.append({
                "custom_id": sample_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_engine,
                    "messages": [
                        {"role": "system", "content": "you are a query rewriter and knowledge selector"},
                        {"role": "user", "content": prompt}
                    ]
                }
            })

    batch_file_path, _ = ensure_unique_path(batch_file_path)
    with open(batch_file_path, "w") as outfile:
        for request in batch_requests:
            json.dump(request, outfile)
            outfile.write("\n")
    return batch_file_path



def upload_and_submit_batch(batch_file_path, folder):
    """
    Uploads a batch file to OpenAI and submits it for processing.
    """
    with open(batch_file_path, "rb") as batch_file:
        file_response = client.files.create(file=batch_file, purpose="batch")
    print(f"Batch file uploaded with ID: {file_response.id}")

    batch_response = client.batches.create(
        input_file_id=file_response.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Batch processing for select_reformulate_X_shot"},
    )
    filename = os.path.basename(batch_file_path)
    append_batch_id_mapping(folder, filename, batch_response.id)
    return batch_response.id

def main():
    parser = argparse.ArgumentParser(description="Prepare and submit a batched API job for select_reformulate_X_shot.")
    parser.add_argument("--input_path", type=str, default="data/2023_test_topics_flattened.jsonl", help="Input test file.")
    parser.add_argument("--shot", type=int, choices=[0, 1, 3, 5], default=0, help="Number of few-shot examples.")
    parser.add_argument("--batch_file_path", type=str, default=None, help="Optional path for batch file.")
    parser.add_argument("--model_engine", type=str, default="gpt-4o-mini", help="GPT model engine.")
    parser.add_argument("--random_examples", action='store_true', help='Select random samples for few-shot examples')
    parser.add_argument("--seed", type=int, default=None, help='Random seed for reproducibility when selecting random examples')
    args = parser.parse_args()

    if "2024_test_topics" in args.input_path:
        folder = f"data/batch/{args.model_engine}/2024/"
    else:
        folder = f"data/batch/{args.model_engine}/2023/"
    if not os.path.exists(folder):
        os.makedirs(folder) 

    if args.batch_file_path is None:
        file_name = f"batch_SAR_{args.shot}shot.jsonl"
        batch_file_path = os.path.join(folder, file_name)
    else:
        batch_file_path = args.batch_file_path

    batch_file_path = prepare_batch_file(
        input_path=args.input_path,
        batch_file_path=batch_file_path,
        model_engine=args.model_engine,
        shot=args.shot,
        random_examples=args.random_examples,
        seed=args.seed
    )

    print(f"Batch file created at: {batch_file_path}")
    batch_id = upload_and_submit_batch(batch_file_path, folder)
    print(f"Batch submitted with ID: {batch_id}")

if __name__ == '__main__':
    main()
