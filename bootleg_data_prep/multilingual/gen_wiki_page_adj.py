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
    parser.add_argument('--embs', type=str, default='/lfs/raiders8/0/lorr1/de_bootleg/embs_de/', help='Where embedding directory is.')
    args = parser.parse_args()
    return args

def get_page_adj(input_files):
    page_adj = defaultdict(set)
    for file in tqdm(input_files):
        with jsonlines.open(file, "r") as in_f:
            for line_obj in in_f:
                qids = line_obj["qids"]
                parent_qid = line_obj["parent_qid"]
                for i, q1 in enumerate(qids):
                    page_adj[parent_qid].add(q1)
    return page_adj

def write_adj(out_f, page_adj):
    with open(out_f, 'w') as out_file:
        for q1, rels in tqdm(page_adj.items()):
            for q2 in rels:
                out_file.write(f"{q1}\t{q2}\n")
    return

def main():
    gl_start = time.time()
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    all_in_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    print(f"Found {len(all_in_files)} files")
    page_adj = get_page_adj(all_in_files)
    out_f = os.path.join(args.embs, "wiki_coo_adj.txt")
    write_adj(out_f, page_adj)
    print(f"Finished data_filter in {time.time() - gl_start} seconds.")

if __name__ == '__main__':
    main()