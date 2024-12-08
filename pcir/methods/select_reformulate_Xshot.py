import json
import time
import re
import argparse
import os
import sys

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
import os
from pcir.utils import load_processed_sample_ids, get_assessed_turn_ids, demonstrate, query_llm, init_llm

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def process_prompt(model, prompt):
    cnv = query_llm("you are a query rewriter and knowledge selector", prompt, model)
    print('-' * 200)
    print("Content of cnv:", cnv)
    
    #match_prov = re.search(r'Provenance:(.*?)Rewrite:', cnv, re.DOTALL)
    match_rewrite = re.search(r'Rewrite:(.*?)Response:', cnv, re.DOTALL)
    match_response = re.search(r'Response:\s*(.*)', cnv, re.DOTALL)

    # retry if not all 3 have content
    if not (match_rewrite and match_response):
        raise ValueError("Failed to match required outputs in response.")

    return match_rewrite.group(1), match_response.group(1)

def main():
    args = get_args()

    if args.shot in [1, 3, 5]:
        d = demonstrate(args.shot, args.random_examples, args.seed)
    elif args.shot == 0:
        d = ''
    else:
        print("Error: --shot must be 0, 1, 3 or 5.")
        sys.exit(1)

    model = init_llm(args.llm_model)
    set_176 = get_assessed_turn_ids()

    processed_sample_ids = load_processed_sample_ids(args.output_path)


    with open(args.input_path) as f:
        for line_num,line in enumerate(f):
            if line_num < 0 :
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

            print(sample_id)
            print(prompt)
            try:
                rewrite, response = process_prompt(model, prompt)
                
                print(f"Sample ID: {sample_id} --> SUCCESS")
                print(f"Rewrite: {rewrite}")
                print(f"Response: {response}")

                data['rewrite_utt_text'] = rewrite
                data['response_utt_text'] = response

            except Exception as e:
                print(f"Sample ID: {sample_id} --> FAILED after retries")
                print(f"Error: {e}")
                continue
        
            with open(args.output_path, 'a+') as outfile:
                    json.dump(data, outfile)
                    outfile.write('\n')
            time.sleep(0.1)  

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
        args.output_path = f"data/results/2023_test_SAR_{args.shot}shot{llm_part}.jsonl"

    print("Output path:", args.output_path)
    return args


if __name__ == '__main__':
    main()

