import argparse
import glob
import os
import time
from collections import defaultdict

import ujson as json
from jsonlines import jsonlines
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/lfs/raiders8/0/lorr1/de_bootleg/input', help='Where input files are.')
    parser.add_argument('--glob', type=str, default='*.jsonl', help='Glob to search for files.')
    parser.add_argument('--embs', type=str, default='/lfs/raiders8/0/lorr1/de_bootleg/embs_de/', help='Where embedding directory is.')
    args = parser.parse_args()
    return args

def get_sent_cooc(input_files):
    sent_cooc = defaultdict(lambda: defaultdict(int))
    for file in tqdm(input_files):
        with jsonlines.open(file, "r") as in_f:
            for line_obj in in_f:
                qids = line_obj["qids"]
                for i, q1 in enumerate(qids):
                    for j, q2 in enumerate(qids):
                        if i == j:
                            continue
                        sent_cooc[q1][q2] += 1
    return sent_cooc

def main():
    gl_start = time.time()
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    all_in_files = glob.glob(os.path.join(args.input_dir, args.glob))
    print(f"Found {len(all_in_files)} files")
    sent_cooc = get_sent_cooc(all_in_files)
    out_f = os.path.join(args.embs, "sent_cooccur_map.json")
    with open(out_f, "w") as f:
        json.dump(sent_cooc, f)
    print(f"Finished data_filter in {time.time() - gl_start} seconds.")

if __name__ == '__main__':
    main()