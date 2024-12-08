import json
import argparse

def preprocess(input_file, output_file, only_needs_ptkb=False):
    temp_q = ''
    temp_a = ''
    with open(input_file) as f, open(output_file, 'w') as outfile:
        d = json.load(f)
        for conv in d:
            number = conv['number']
            title = conv['title']
            ptkb = conv['ptkb']
            for turn in conv['turns']:
                if not turn["ptkb_provenance"] and only_needs_ptkb:
                    continue
                data = {}
                turn_id = turn['turn_id']
                data['sample_id'] = str(number) + '-' + str(turn_id)
                data['number'] = number
                data['title'] = title
                data['cur_utt_text'] = turn['utterance']
                data['oracle_utt_text'] = turn['resolved_utterance']
                data['cur_response_text'] = turn['response']
                data['ptkb'] = ptkb
                data['ptkb_provenance'] = turn['ptkb_provenance']
                data['response_provenance'] = turn['response_provenance']
                if turn_id == 1:
                    ctx_utts_text = []
                    history_response = []
                else:
                    ctx_utts_text.append(temp_q)
                    history_response.append(temp_a)
                data['ctx_utts_text'] = ctx_utts_text
                data['history_response'] = history_response
                temp_q = turn['utterance']
                temp_a = turn['response']
                json.dump(data, outfile)
                outfile.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str,default="data/2023_test_topics.json")
    parser.add_argument("-o", "--output_path", type=str, default="data/2023_test_topics_flattened.jsonl")
    parser.add_argument("--only_needs_ptkb",  action="store_true")
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path, only_needs_ptkb=args.only_needs_ptkb)