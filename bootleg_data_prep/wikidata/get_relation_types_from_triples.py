"""
Used Wikidata extractor to get a list of triples for all qids in our dump such that the qid was the head of some relation. The tail could be any qid,
in our dump or not. (See wikidata_extractor get_all_wikipedia_triples.py)
"""
import argparse
import json
from collections import defaultdict

from tqdm import tqdm

from bootleg_data_prep.language import ENSURE_ASCII


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid_file', type=str, default='utils/param_files/pid_names_en.json', help='path to pid file')
    parser.add_argument('--kg_triples', type=str, default='', help='path to kg_triples.json')
    parser.add_argument('--type_file', type=str, default='', help='path to types.json')
    parser.add_argument('--output_file', type=str, default='', help='path to output file')
    parser.add_argument('--output_vocab_file', type=str, default='', help='path to output vocab file')
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print("Reading in PID names")
    pid_names = json.load(open(args.pid_file, "r"))

    print(f"Reading in triples")
    triples = json.load(open(args.kg_triples, "r"))
    head_trips_dict = defaultdict(set)
    pred_count = defaultdict(int)
    for head in triples:
        for pred in triples[head]:
            head_trips_dict[head].add(pred)
            pred_count[pred] += 1

    print("Mapping pids")
    rel_ids = defaultdict(int)
    rel_ids_internal = defaultdict(int)
    # Index starts at 1
    num = 1
    for pid in pred_count:
        rel_ids[pid_names.get(pid,pid)] = num
        rel_ids_internal[pid] = num
        num += 1

    # Current list of types...wanted to make sure the keys matched. This is optional
    print("Loading QIDs")
    all_type_qids = {}
    with open(args.type_file, "r") as in_f:
        all_type_qids = json.load(in_f)


    print("Dumping")
    out_file = open(args.output_file, "w")
    max_len = 0
    final_qid2type = {}
    for qid in tqdm(all_type_qids):
        types = []
        if qid in head_trips_dict:
            type_ids = head_trips_dict[qid]
            max_len = max(max_len, len(type_ids))
            # Type 0 is the "instance of" type that is not discriminative at all
            type_ids = sorted(type_ids, key=lambda x: [pred_count[x], x], reverse=True)
            type_ids = [rel_ids_internal[i] for i in type_ids]
            types = [i for i in type_ids if i != rel_ids["instance of"]]
        final_qid2type[qid] = types


    json.dump(final_qid2type, out_file, ensure_ascii=ENSURE_ASCII)

    out_file.close()
    with open(args.output_vocab_file, "w") as out_f:
        json.dump(rel_ids, out_f, ensure_ascii=ENSURE_ASCII)

    print("Max number of relation types per qid", max_len)

if __name__ == '__main__':
    main()