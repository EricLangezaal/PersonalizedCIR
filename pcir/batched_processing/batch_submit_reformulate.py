import json
import argparse
import os
import sys
from openai import OpenAI
from pcir.utils import get_assessed_turn_ids, load_provenance_dict, ensure_unique_path, extract_numbers_from_dict

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

def prepare_batch_file(input_path, batch_file_path, model_engine, prompt_type, shot, annotation):
    """
    Prepares batch file for 50% discount
    """
    batch_file_path, run_number = ensure_unique_path(batch_file_path)
    
    if "2024_test_topics" in input_path:
        assessed_turns = get_assessed_turn_ids(path="data/2024-qrels.trec")
    else:
        assessed_turns = get_assessed_turn_ids(path="data/2023-qrels.all-turns.txt")

    print(assessed_turns)
    print(len(assessed_turns))
    year_folder = "2024" if "2024_test_topics" in input_path else "2023"
    if annotation == 'LLM':
        provenance_file = f"data/batch/{model_engine}/{year_folder}/processed/batch_LLM_select_ptkb_{shot}shot_run{run_number}.jsonl"
        print(f'Using {provenance_file}')
    else:
        # human, None and All can be derived from test data provencance
        provenance_file = input_path

    if not os.path.exists(provenance_file):
        print(f"Provenance file {provenance_file} not found.")
        sys.exit(1)

    provenance_dict = load_provenance_dict(provenance_file, type=annotation)
    if annotation in ['LLM', 'STR']:
        provenance_dict = extract_numbers_from_dict(provenance_dict)
    batch_requests = []
    with open(input_path) as infile:
        for line in infile:
            data = json.loads(line)
            sample_id = data.get("sample_id", "")

            if sample_id not in assessed_turns:
                continue

            ptkb = data.get("ptkb", {})
            cur_utt = data.get("cur_utt_text", "")
            conv = data.get("ctx_utts_text", [])
            history_response = data.get("history_response", [])
            provenance = provenance_dict.get(sample_id, [])

            print("USED PROVENANCE KEYS", provenance)
            if conv == []:
                conv = cur_utt
            ptkb_lable = ''
            for i in provenance:
                ptkb_item = ptkb.get(str(i), '')
                if ptkb_item:
                    ptkb_lable += ptkb_item + '\n'

            if not ptkb_lable.strip():
                ptkb_lable = 'None'
            print("LABEL", ptkb_lable)
            YourTask = ''
            for q,a in zip(conv,history_response):
                YourTask += f'Question: {q}\nResponse: {a}\n\n'
            YourTask += 'Current Question: ' + cur_utt 
        
            if prompt_type == 1:
                prompt = f"### Instruction: I will give you a conversation between a user and a system. "\
                "Also will give you some background information about the user. "\
                "You should answer the last utterance of the user based on user background information. "\
                "Please remember that your answer to the last question the user shouldn't be more than 200 words.\n"\
                f"### Background information about the user: \n{ptkb_lable}\n "\
                f"### Conversation: \n{YourTask}\n### Response: \n"\
                "### Can you generate the unique queries that can be used for retrieving your previous answer to the user? "\
                "(Please write queries in one line and don't generate more than 5 queries)\n"\
                "### Queries: \n(Now, you should give me the Response and the Queries. "\
                "The output format should always be: Response: $Response Queries: $Queries. Go ahead!)" 

            if prompt_type == 2:
                prompt = f"For an information-seeking dialog, please help reformulate the question "\
                    "into a rewrite that can fully express the user's information needs without the need for context, "\
                    "but also generate an informative response to answer the question. "\
                    "You can generate a rewrite and response based on the user's personal information(if any).\n"\
                    f"#YourTask: \n User's personal information:\n{ptkb_lable}\n{YourTask}\n"\
                    "(Now, you should give me the rewrite and an informative response of the **Current Question** based on the **Context**. "\
                    "Please provide a complete informative response, but keep it under 200 words. "\
                    "The output format should always be: Rewrite: $Rewrite\nResponse: $Response. Go ahead!)" 

            #print(prompt)

            batch_requests.append({
                "custom_id": sample_id,
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
        metadata={"description": "Batch processing for reformulation"},
    )
    filename = os.path.basename(batch_file_path)
    append_batch_id_mapping(folder, filename, batch_response.id)
    return batch_response.id




def main():
    parser = argparse.ArgumentParser(description="Prepare and submit a batched API job.")
    parser.add_argument("--input_path", type=str, default="data/2023_test_topics_flattened.jsonl", help="input test file ikat")
    parser.add_argument("--shot", type=int, default=0, help="num examples")
    parser.add_argument("--annotation", type=str, default="human", help="Annotation type (None, All, human, automatic or LLM).")
    parser.add_argument("--batch_file_path", type=str, default=None, help="Optional path for batch file")
    parser.add_argument("--model_engine", type=str, default="gpt-4o-mini", help="GPT model engine.")
    parser.add_argument("--prompt_type", type=int, choices=[1, 2], default=1, help="Prompt type.")
    args = parser.parse_args()

    if "2024_test_topics" in args.input_path:
        folder = f"data/batch/{args.model_engine}/2024/"
    else:
        folder = f"data/batch/{args.model_engine}/2023/"
    if not os.path.exists(folder):
        os.makedirs(folder) 

    if args.batch_file_path is None:
        file_name = f"batch_{args.annotation}_{args.shot}shot_prompt{args.prompt_type}.jsonl"
        batch_file_path = os.path.join(folder, file_name)
    else:
        batch_file_path = args.batch_file_path

    batch_file_path = prepare_batch_file(
        input_path=args.input_path,
        batch_file_path=batch_file_path,
        model_engine=args.model_engine,
        prompt_type=args.prompt_type,
        shot=args.shot,
        annotation=args.annotation,
    )

    print(f"Batch file created at: {batch_file_path}")
    batch_id = upload_and_submit_batch(batch_file_path, folder)
    print(f"Batch submitted with ID: {batch_id}")

if __name__ == '__main__':
    main()
