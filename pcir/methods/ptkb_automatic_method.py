import json
import time
import re
import argparse

from pcir.utils import get_assessed_turn_ids, init_llm, query_llm

def main():
    args = get_args()

    model = init_llm(args.llm_model)
    set_176 = get_assessed_turn_ids()

    with open(args.input_path) as f:
        for line in f:
            data = json.loads(line)
            sample_id = data.get('sample_id','')
            if sample_id not in set_176:
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


            for i,v in ptkb.items():
                if conv == []:
                    conv = cur_utt
                YourTask = '#Your Task:\n\n' + 'User\'s personal information: ' + v + '\n\n'
                for q,a in zip(conv,history_response):
                    YourTask += f'Question: {q}\nResponse: {a}\n\n'
                YourTask += 'Current Question: ' + cur_utt + '\n\n'

                prompt = f"For an information-seeking dialog, please help reformulate the question "\
                        "into rewrite that can fully express the user's information needs without the need of context, "\
                        "but also generate an informative response to answer the question. "\
                        "You can generate a rewrite and response based on user's personal information. " + YourTask + \
                        "(Now, you should give me the rewrite and an informative response of the **Current Question** based on the **Context**. "\
                        "Please provide a complete informative response, but keep it under 200 words. "\
                        "The output format should always be: Rewrite: $Rewrite\nResponse: $Response. Go ahead!)"
                print(sample_id)
                print(prompt)
                cnv = query_llm("you are a query rewriter and knowledge selector", prompt, model)
            
                print('-' * 100)
                print(cnv)

                match_rewrite = re.search(r'Rewrite:(.*?)Response:', cnv,re.DOTALL)
                match_response = re.search(r'Response:\s*(.*)', cnv,re.DOTALL)
                rewrite = match_rewrite.group(1)
                response = match_response.group(1)

                data['rewrite_utt_text'] = rewrite
                data['response_utt_text'] = response
                data['ptkb_num'] = i
                data['sample_id'] = sample_id + '-' +str(i)
                print(data['sample_id'])


                with open(args.output_path, 'a+') as outfile:
                        json.dump(data, outfile)
                        outfile.write('\n')
                time.sleep(0.5)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/2023_test_topics_flattened.jsonl")
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--llm_model', type=str, default="gpt-3.5-turbo-16k")

    args = parser.parse_args()

    if args.output_path is None:
        llm_part = "" if args.llm_model == "gpt-3.5-turbo-16k" else "_" + args.llm_model.split("/")[0]
        args.output_path = f"data/results/2023_test_automatic_select_0shot{llm_part}.jsonl"
    return args


if __name__ == '__main__':
    main()

