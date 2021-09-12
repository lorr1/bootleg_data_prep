'''
This file: Aggregates QIDS from across all benchmarks. Make sure to have generated unfilterd jsonls for each of the benchmarks (see README.md in benchmarks/)

Example run command:
python3.6 -m contextual_embeddings.bootleg_data_prep.aggregate_benchmarks
'''

import argparse
import os

import ujson as json

import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.language import ensure_ascii
from bootleg_data_prep.utils import utils

BENCHMARK_DIRS = [
    'data/aida/unfiltered',
    'data/msnbc/unfiltered',
    'data/kore50/unfiltered',
    'data/rss500/unfiltered'
]

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data and runs
    parser.add_argument('--data_dir', type=str, default='data/utils/')
    parser.add_argument('--ganea_candidates', default = "contextual_embeddings/bootleg_data_prep/benchmarks/ganea/processed/cands.json")
    return parser


def get_qids_from_file(fpath):

    qids = []
    with open(fpath) as in_file: 
        for line in in_file: 
            obj = json.loads(line.strip())
            qids.extend(obj['qids'])
    return qids

def get_candidates(args):

    cand_map = json.load(open(args.ganea_candidates))
    qids = set()
    for alias, cands in cand_map.items():
        for qid, _ in cands:
            qids.add(qid)
    print(f"{len(qids)} qids from ganea candidates.")
    return qids

             
def main():
    args = get_arg_parser().parse_args()
    print(json.dumps(vars(args), indent=4))
    utils.ensure_dir(args.data_dir)
    
    all_qids = set()
    for dataset in BENCHMARK_DIRS:
        files = prep_utils.glob_files(os.path.join(dataset, '*.jsonl'))
        for file in files:
            file_qids = get_qids_from_file(file)
            for qid in file_qids:
                all_qids.add(qid)
            print(f"Extracted {len(file_qids)} from {file}. {len(all_qids)} total QIDS collected.")
    all_qids = all_qids.union(get_candidates(args))
    print(f"{len(all_qids)} Qids collected from benchmarks.")
    with open(os.path.join(args.data_dir, 'benchmark_qids.json'), 'w') as out_file: 
        json.dump(list(all_qids), out_file, ensure_ascii=ensure_ascii)

    prep_utils.save_config(args, "aggregate_benchmarks.json")
    print(f"Data saved to {args.data_dir}")
    



if __name__ == '__main__':
    main()