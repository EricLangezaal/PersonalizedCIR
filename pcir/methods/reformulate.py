import json
import time
import re
import argparse
import sys
import os

from openai import OpenAI, RateLimitError
from pcir.utils import (
    get_assessed_turn_ids, 
    load_provenance_dict, 
    load_processed_sample_ids, 
    extract_numbers_from_dict,
    query_llm,
    init_llm,
)
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2), reraise=True)
def process_prompt(model, prompt, prompt_type):
    try:
        cnv = query_llm("you are a query rewriter and knowledge selector", prompt, model)
        print("CNV:", cnv)

        if prompt_type == 1:
            match_rewrite = re.search(r'Queries:\s*(.*)', cnv, re.DOTALL)
            match_response = re.search(r'Response:(.*?)Queries:', cnv, re.DOTALL)
        elif prompt_type == 2:
            match_rewrite = re.search(r'Rewrite:\s*(.*?)Response:', cnv, re.DOTALL)
            match_response = re.search(r'Response:\s*(.*)', cnv, re.DOTALL)

        if not (match_rewrite and match_response):
            raise ValueError("Failed to match required outputs in response.")

        return match_rewrite.group(1).strip(), match_response.group(1).strip()

    except RateLimitError as e:
        print("Rate limit error:", e)
        time.sleep(60) 
        raise

    except RetryError as e:
        print("RetryError after max attempts:", e)
        return "retried reached {}".format(e)

    except Exception as e:
        print("other error:", e)
        raise


def main():
    args = get_args()
    model = init_llm(args.llm_model, args.seed)

    set_176 = get_assessed_turn_ids()
    processed_sample_ids = load_processed_sample_ids(args.output_path)

    if args.annotation in ['STR', 'LLM']:
        llm_part = "" if args.llm_model == "gpt-3.5-turbo-16k" else "_" + args.llm_model.split("/")[0]
        provenance_file = f"data/results/2023_test_{args.annotation}_select_{args.shot}shot{llm_part}_run{args.seed}.jsonl"
    else:
        # human, None and All can be derived from test data provencance
        provenance_file = args.input_path

    if not os.path.exists(provenance_file):
        print(f"Provenance file {provenance_file} not found.")
        sys.exit(1)

    provenance_dict = load_provenance_dict(provenance_file, type=args.annotation)
    if args.annotation in ['LLM', 'STR']:
        provenance_dict = extract_numbers_from_dict(provenance_dict)
    with open(args.input_path) as f:
        for line in f:
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
            conv = data.get('ctx_utts_text','')
            history_response = data.get('history_response','')

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
        
            if args.prompt_type == 1:
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

            if args.prompt_type == 2:
                prompt = f"For an information-seeking dialog, please help reformulate the question "\
                    "into a rewrite that can fully express the user's information needs without the need for context, "\
                    "but also generate an informative response to answer the question. "\
                    "You can generate a rewrite and response based on the user's personal information(if any).\n"\
                    f"#YourTask: \n User's personal information:\n{ptkb_lable}\n{YourTask}\n"\
                    "(Now, you should give me the rewrite and an informative response of the **Current Question** based on the **Context**. "\
                    "Please provide a complete informative response, but keep it under 200 words. "\
                    "The output format should always be: Rewrite: $Rewrite\nResponse: $Response. Go ahead!)" 

            print(prompt)

            try:
                rewrite, response = process_prompt(model, prompt, args.prompt_type)
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/2023_test_topics_flattened.jsonl")
    parser.add_argument('--shot', type=int, default=0)
    parser.add_argument('--annotation', type=str, default="LLM", choices=["LLM", "STR", "human", "None", "All"])
    parser.add_argument('--prompt_type', type=int, default=1)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--llm_model', type=str, default="gpt-3.5-turbo-16k")
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.output_path is None:
        annotation = args.annotation.replace("LLM", "STR")
        llm_part = "" if args.llm_model == "gpt-3.5-turbo-16k" else "_" + args.llm_model.split("/")[0]
        args.output_path = f"data/results/2023_test_{annotation}_{args.shot}shot_prompt_type{args.prompt_type}{llm_part}_run{args.seed}.jsonl"

    print("Output path:", args.output_path)
    return args


        

if __name__ == '__main__':
    main()

