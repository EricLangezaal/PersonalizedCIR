import json
import argparse
import os
import sys
from openai import OpenAI
from pcir.utils import get_assessed_turn_ids, ensure_unique_path

client = OpenAI()

def append_batch_id_mapping(folder, filename, batch_id):
    if "2024" in folder:
        mapping_file = os.path.join(folder, "batch_id_mapping_2024.json")
    else:
        mapping_file = os.path.join(folder, "batch_id_mapping_2023.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            batch_id_mapping = json.load(f)
    else:
        batch_id_mapping = {}
    batch_id_mapping[filename] = batch_id
    with open(mapping_file, "w") as f:
        json.dump(batch_id_mapping, f, indent=4)


def prepare_batch_file(input_path, batch_file_path, model_engine):

    batch_file_path, _ = ensure_unique_path(batch_file_path)
    
    if "2024_test_topics" in input_path:
        assessed_turns = get_assessed_turn_ids(path="data/2024-qrels.trec")
    else:
        assessed_turns = get_assessed_turn_ids(path="data/2023-qrels.all-turns.txt")
    
    batch_requests = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            sample_id = data.get("sample_id", "")
    
            if sample_id not in assessed_turns:
                continue
    
            ptkb = data.get("ptkb", {})
            cur_utt = data.get("cur_utt_text", "")
            conv = data.get("ctx_utts_text", [])
            history_response = data.get("history_response", [])
    
            for i, v in ptkb.items():
                if not conv:
                    conv = [cur_utt]
                YourTask = '#Your Task:\n\n' + 'User\'s personal information: ' + v + '\n\n'
                for q, a in zip(conv, history_response):
                    YourTask += f'Question: {q}\nResponse: {a}\n\n'
                YourTask += 'Current Question: ' + cur_utt + '\n\n'
    
                prompt = (
                    "For an information-seeking dialog, please help reformulate the question "
                    "into rewrite that can fully express the user's information needs without the need of context, "
                    "but also generate an informative response to answer the question. "
                    "You can generate a rewrite and response based on user's personal information. " + YourTask +
                    "(Now, you should give me the rewrite and an informative response of the **Current Question** based on the **Context**. "
                    "Please provide a complete informative response, but keep it under 200 words. "
                    "The output format should always be: Rewrite: $Rewrite\nResponse: $Response. Go ahead!)"
                )
    
                batch_requests.append({
                    "custom_id": f"{sample_id}-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_engine,
                        "messages": [
                            {"role": "system", "content": "You are a query rewriter and knowledge selector."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                })
    
    if not batch_requests:
        print("No samples found for batching.")
        sys.exit(0)
    
    with open(batch_file_path, "w", encoding="utf-8") as outfile:
        for request in batch_requests:
            json.dump(request, outfile)
            outfile.write("\n")
    return batch_file_path

def upload_and_submit_batch(batch_file_path, folder):
    with open(batch_file_path, "rb") as batch_file:
        file_response = client.files.create(file=batch_file, purpose="batch")
    print(f"Batch file uploaded with ID: {file_response.id}")
    
    batch_response = client.batches.create(
        input_file_id=file_response.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Batch processing automatic"},
    )
    filename = os.path.basename(batch_file_path)
    append_batch_id_mapping(folder, filename, batch_response.id)
    print(f"Batch submitted with ID: {batch_response.id}")
    return batch_response.id

def main():
    parser = argparse.ArgumentParser(description="Prepare and submit a batched API job.")
    parser.add_argument("--input_path", type=str, default="data/2023_test_topics_flattened.jsonl", help="Input test file")
    parser.add_argument("--batch_file_path", type=str, default=None, help="Optional path for batch file")
    parser.add_argument("--model_engine", type=str, default="gpt-4o-mini", help="gpt engine.")
    args = parser.parse_args()
    
    if "2024_test_topics" in args.input_path:
        folder = f"data/batch/{args.model_engine}/2024/"
    else:
        folder = f"data/batch/{args.model_engine}/2023/"
    os.makedirs(folder, exist_ok=True)
    
    if args.batch_file_path is None:
        file_name = f"batch_automatic_0shot.jsonl"
        batch_file_path = os.path.join(folder, file_name)
    else:
        batch_file_path = args.batch_file_path
    
    batch_file_path = prepare_batch_file(
        input_path=args.input_path,
        batch_file_path=batch_file_path,
        model_engine=args.model_engine
    )
    print(f"Batch file created at: {batch_file_path}")
    batch_id = upload_and_submit_batch(batch_file_path, folder)
    print(f"Batch submitted with ID: {batch_id}")

if __name__ == '__main__':
    main()
