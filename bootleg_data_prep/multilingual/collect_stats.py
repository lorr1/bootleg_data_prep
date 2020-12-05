import argparse
import glob
import os
import time
from collections import defaultdict

import ujson as json
from jsonlines import jsonlines
from tqdm import tqdm

from bootleg.utils import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/lfs/raiders8/0/lorr1/de_bootleg/raw/train', help='Where input files are.')
    args = parser.parse_args()
    return args

def get_stats(input_files):
    stat_dict = defaultdict(lambda:defaultdict(int))
    all_qids = set()
    for file in tqdm(input_files):
        with jsonlines.open(file, "r") as in_f:
            for line_obj in in_f:
                for al, qid in zip(line_obj["aliases"], line_obj["qids"]):
                    stat_dict[al.lower()][qid] += 1
                    all_qids.add(qid)
    print(f"Found {len(all_qids)} qids")
    return dict(stat_dict)

def main():
    gl_start = time.time()
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    out_dir = os.path.join(os.path.dirname(args.input_dir), "stats")
    utils.ensure_dir(out_dir)
    all_in_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    print(f"Found {len(all_in_files)} files")
    stat_dict = get_stats(all_in_files)
    out_f = os.path.join(out_dir, "alias_qid_alldata_withaugment.json")
    with open(out_f, "w") as f:
        json.dump(stat_dict, f)
    print(f"Finished data_filter in {time.time() - gl_start} seconds.")

if __name__ == '__main__':
    main()