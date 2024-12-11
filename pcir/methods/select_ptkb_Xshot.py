import json
import time
import re
import argparse
from pcir.utils import load_processed_sample_ids, get_assessed_turn_ids, demonstrate, query_llm, init_llm, ensure_unique_path
import sys
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

client = OpenAI()

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def process_prompt(model, prompt):
    cnv = query_llm("you are a good information selector", prompt, model)
    match_prov = re.search(r'Provenance:\s*(.*)', cnv, re.DOTALL)

    if not match_prov:
        raise ValueError("No matching provenance in response.")

    return match_prov.group(1)

def main():
    args = get_args()
    if args.shot in [1, 3, 5]:
        d = demonstrate(args.shot, args.random_examples, args.seed)
    elif args.shot == 0:
        d = ''
    else:
        print("Error: --shot must be 0, 1, 3 or 5.")
        sys.exit(1)

    model = init_llm(args.llm_model, args.seed)

    processed_sample_ids = load_processed_sample_ids(args.output_path)
    set_176 = get_assessed_turn_ids()


    with open(args.input_path) as f:
        for line_num, line in enumerate(f):
            if line_num < 0:
                continue
            data = json.loads(line)
            sample_id = data.get('sample_id','')
            if sample_id not in set_176:
                continue

            # Skip if sample_id is already processed
            if sample_id in processed_sample_ids:
                print(f"Skipping already processed sample_id: {sample_id}")
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
            YourTask = "#Your Task (only user's information, questions and the responses are given):\n\n" \
                       f"User's information: {str(ptkb)}\n\n"
            for q, a in zip(conv, history_response):
                YourTask += f"Question: {q}\nResponse: {a}\n\n"
            YourTask += f"Question: {cur_utt}\n\n"

            prompt = "For an information-seeking dialog, please select the user information that will help answer this question (if there is any).\n"
            if d:
                prompt += f"I will provide you with some examples:\n {d}\n\n"
            prompt += YourTask + \
                      "(Now you need to give me a list of the serial numbers you have chosen. " \
                      "The output format should always be: Provenance: $The user information number you have selected."

            print(sample_id)
            print(prompt)
            try:
                match_prov = process_prompt(model, prompt)
                print(f"Sample ID: {sample_id} --> SUCCESS")
                print(f"Provenance: {match_prov}")

                data['LLM_select'] = match_prov

            except Exception as e:
                print(f"Sample ID: {sample_id} --> FAILED after retries")
                print(f"Error: {e}")
                data['LLM_select'] = 'None'


            with open(args.output_path, 'a+', encoding='utf-8') as outfile:
                json.dump(data, outfile)
                outfile.write('\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/2023_test_topics_flattened.jsonl")
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--random_examples', action='store_true', help='Select random samples for few-shot examples')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--llm_model', type=str, default="gpt-3.5-turbo-16k")
    args = parser.parse_args()

    # Set output_path dynamically based on shot
    if args.output_path is None:
        llm_part = "" if args.llm_model == "gpt-3.5-turbo-16k" else "_" + args.llm_model.split("/")[0]
        args.output_path = f"data/results/2023_test_LLM_select_{args.shot}shot{llm_part}_run{args.seed}.jsonl"

    print("Output path:", args.output_path)
    return args


if __name__ == '__main__':
    main()
