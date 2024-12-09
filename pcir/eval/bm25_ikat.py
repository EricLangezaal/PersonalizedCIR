import logging
import json
from pyserini.search.lucene import LuceneSearcher
import argparse
from utils import is_relevant, calculate_trec_res_NDCG, get_assessed_turn_ids, get_output_path_trec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)

def main(args):
    query_list = []
    qid_list = []
    relevant_ids = get_assessed_turn_ids(year=args.year)

    print(f'Number of relevant turns: {len(relevant_ids)}')
    if args.query_type == "combine":
        with open(args.input_query_path, "r") as f:
            data = f.readlines()
        with open(args.input_query_path1, "r") as f:
            data1 = f.readlines()
        for (l,l1) in zip(data,data1):
            l = json.loads(l)
            l1 = json.loads(l1)
            query_id = l["sample_id"]
            if not is_relevant(query_id, relevant_ids):
                continue
            query = l["rewrite_utt_text"] + l["response_utt_text"] + l1["response_utt_text"]
            query_list.append(query)
            qid_list.append(query_id)

    else:
        with open(args.input_query_path, "r") as f:
            data = f.readlines()
        for l in data:
            l = json.loads(l)
            query_id = l["sample_id"]
            if not is_relevant(query_id, relevant_ids):
                continue
            if args.query_type == "rewrite":
                query = l['rewrite_utt_text']
            if args.query_type == "concat":
                i = 1
                query = ''
                while i <= args.query_number:
                    query += l['rewrite_utt_text']
                    i += 1
                query += l['response_utt_text']
            if args.query_type == "concat_auto":
                i = 1
                rewrite = l['rewrite_utt_text']
                response = l['response_utt_text']
                words1 = rewrite.split()
                words2 = response.split()
                n = round(len(words2)/len(words1)/args.c)
                query = ''
                while i <= n:
                    query += l['rewrite_utt_text']
                    i += 1
                query += l['response_utt_text']
            if args.query_type == "oracle":
                query = l['oracle_utt_text']
            if args.query_type == "fusion":
                query = l['fusion'] 
            if args.query_type == "response":
                query = l['response_utt_text']

            query_list.append(query)
            qid_list.append(query_id)
   
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k = args.top_k, threads = 40)
    
    with open(args.output_path, "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid,
                                                i + 1,
                                                -i - 1 + 200,
                                                item.score,
                                                "bm25"
                                                ))
                f.write('\n')

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_query_path", type=str)
    parser.add_argument("--input_query_path1", type=str, default=None)
    parser.add_argument('--output_id', type=str, default="", help="Use to give output file a unique postfix.")
    #parser.add_argument('--gold_qrel_file_path', type=str, default="data/2023-qrels.all-turns.txt")
    parser.add_argument('--index_dir_path', type=str, default="/scratch-shared/ikat23/trec_ikat_2023_passage_index")
    parser.add_argument("--top_k", type=int,  default=1000)
    parser.add_argument("--rel_threshold", type=int,  default=1)
    parser.add_argument("--bm25_k1", type=int,  default=0.82)
    parser.add_argument("--bm25_b", type=int,  default=0.68)
    parser.add_argument('--query_type', type=str, default="concat")
    parser.add_argument('--query_number', type=int, default=1)
    parser.add_argument('--c', type=float, default=1)
    parser.add_argument("--automatic_method", action='store_true', help="automatic method for query generation")
    args = parser.parse_args()

    args.output_path = get_output_path_trec("bm25_", args)

    if "2024" in args.input_query_path:
        args.year = "2024"
    elif "2023" in args.input_query_path:
        args.year = "2023"
    else:
        raise ValueError("2023 or 2024 not found in the input query path.")
    
    args.gold_qrel_file_path = f'data/{args.year}-qrels.all-turns.txt'

    return args

if __name__ == '__main__':
    args = get_args()
    print("ARGUMENTS")
    print(args)
    print('=======')
    main(args)
    print("="*200)
    print("full set results")
    calculate_trec_res_NDCG(args.output_path, args.gold_qrel_file_path, args.rel_threshold, args.automatic_method, year=args.year)
    print("Subset results")
    calculate_trec_res_NDCG(args.output_path, args.gold_qrel_file_path, args.rel_threshold, args.automatic_method, subset=True, year=args.year)
    print("Corrected subset results")
    calculate_trec_res_NDCG(args.output_path, args.gold_qrel_file_path, args.rel_threshold, args.automatic_method, subset=True, new_ptkb=True, year=args.year)

