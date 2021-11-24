'''
This file 

1. Reads in all entity descriptions and the first sentence of every wikipedia page
2. Dumps mapping from QID to description
'''

import os, json, argparse, time
from rich.progress import track
import marisa_trie
from glob import glob
from multiprocessing import set_start_method, Pool
from pathlib import Path
import simple_wikidata_db.utils as utils

from bootleg_data_prep.language import ENSURE_ASCII


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str, default = '/lfs/raiders8/0/lorr1/wikidata', help = 'path to output directory')
    parser.add_argument('--out_dir', type = str, default = 'desc_output', help = 'path to output directory')
    parser.add_argument('--wikipedia_page_data', type = str, default = '/lfs/raiders8/0/lorr1/data/wiki_dump/alias_filtered_sentences')
    parser.add_argument('--qids', default = '/lfs/raiders8/0/lorr1/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json')
    return parser 

def main():
    start = time.time()
    args = get_arg_parser().parse_args()

    filter_qids = set()
    if len(args.qids) > 0:
        filter_qids = json.load(open(args.qids))
        if type(filter_qids) is dict:
            filter_qids = set(filter_qids.keys())
        else:
            filter_qids = set(filter_qids)
    print(f"Loaded {len(filter_qids)} qids.")

    qid2desc = {}
    all_files = list(Path(args.wikipedia_page_data).glob("wiki_*.jsonl"))
    print("Reading in first sentence for each QID")
    for file in track(all_files):
        with open(file) as in_f:
            for line in in_f:
                line = json.loads(line)
                if (len(filter_qids) <= 0 or line["qid"] in filter_qids) and (len(line["sentences"]) > 0):
                    qid2desc[line["qid"]] = line["sentences"][0]["sentence"]
    print("Reading in descriptions from wikidata")
    fdir = os.path.join(args.data, "processed_batches", "descriptions")
    entity_table_files = utils.get_batch_files(fdir)
    for f in track(entity_table_files):
        for line in open(f):
            line = json.loads(line)
            if line["qid"] not in qid2desc and (len(filter_qids) <= 0 or line["qid"] in filter_qids):
                qid2desc[line["qid"]] = line["description"]

    out_dir = os.path.join(args.data, args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    json.dump(qid2desc, open(os.path.join(out_dir, "qid2desc.json"), "w"))
    print(f"Saved to {out_dir}. Finihsed in {time.time() - start}s.")

if __name__ == "__main__":
    main()