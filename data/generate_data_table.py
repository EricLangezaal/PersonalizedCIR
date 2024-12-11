
import pandas as pd
import argparse

from pyserini.search.lucene import LuceneSearcher
from pcir.utils import get_assessed_turn_ids

def main(args):
    train_data = pd.read_json(args.train_file_path)
    test_data = pd.read_json(args.test_file_path)

    topics_train = len(train_data['title'].unique())
    topics_test = len(test_data['title'].unique())

    conversations_train = len(train_data)
    conversations_test = len(test_data)

    turns_train = train_data["turns"].apply(len).sum()
    turns_test = test_data["turns"].apply(len).sum()

    assessed_turns = get_assessed_turn_ids(args.qrels_file)

    ptkb_turns_train = train_data["turns"].apply(lambda x: len([t for t in x if t["ptkb_provenance"]])).sum()
    ptkb_turns_test = test_data["turns"].apply(lambda x: len([t for t in x if t["ptkb_provenance"]])).sum()

    num_docs = LuceneSearcher(args.passage_index).num_docs

    data = {
        'Variable': ['topics', 'conversations', 'turns', 'assessed_turns', 'total_ptkb_turns', "collection"],
        'Train': [topics_train, conversations_train, turns_train, "", ptkb_turns_train, ""],
        'Test': [topics_test, conversations_test, turns_test, len(assessed_turns), ptkb_turns_test, f"{num_docs / 1e6:.1f}M"]
    }

    print(pd.DataFrame(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", type=str, default="2023_train_topics.json")
    parser.add_argument("--test_file_path", type=str, default="2023_test_topics.json")
    parser.add_argument("--qrels_file", type=str, default="2023-qrels.all-turns.txt")
    parser.add_argument("--passage_index", type=str, default="/scratch-shared/ikat23/trec_ikat_2023_passage_index")
    args = parser.parse_args()
    main(args)

