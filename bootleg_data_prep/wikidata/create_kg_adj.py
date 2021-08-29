'''
This file 

1. Reads in all entity-entity relations
2. Dumps them out in the kg_adj format 

to run: 
python3.6 -m processor.get_rels_as

''' 

    
import os, sys, json, argparse, time, shutil

import marisa_trie
from tqdm import tqdm
from glob import glob
from multiprocessing import set_start_method, Pool

from collections import defaultdict

from processor.constants import *
from processor.utils import *

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str, default = '/lfs/raiders8/0/lorr1/wikidata', help = 'path to output directory')
    parser.add_argument('--out_dir', type = str, default = 'kg_output', help = 'path to output directory')
    parser.add_argument('--processes', type = int, default = 80, help = "Number of concurrent processes to spin off. ")
    parser.add_argument('--qids', default = '/lfs/raiders8/0/lorr1/es_bootleg/es_qids.json')
    return parser

def init_process(args):
    temp_f = args[0]
    vals = list(json.load(open(temp_f, "r")))
    global filter_qids_global
    filter_qids_global = marisa_trie.Trie(vals)

def launch_entity_table(entity_files, filter_qids, out_dir, args):
    temp_f = os.path.join(out_dir, "_temp_filter.json")
    json.dump(list(filter_qids), open(temp_f, "w"))
    print(f"Starting with {args.processes} processes")
    pool = Pool(processes = args.processes, initializer=init_process, initargs=(tuple([temp_f]),))
    messages = [(i, len(entity_files), entity_files[i], out_dir) for i in range(len(entity_files))]
    pool.map(load_entity_file, messages, chunksize=1)
    merge_and_save(out_dir)
    return

def load_entity_file(message):
    start = time.time()
    job_index, num_jobs, filename, out_dir = message
    rel_dict = {}
    #print(f"Starting {job_index} / {num_jobs}")
    for triple in jsonl_generator(filename):
        qid, property_id, value = triple['qid'], triple['property_id'], triple['value']
        if len(filter_qids_global) == 0 or (qid in filter_qids_global and value in filter_qids_global):
            a = qid
            b = value
            if not a in rel_dict:
                rel_dict[a] = {}
            if not b in rel_dict[a]:
                rel_dict[a][b] = 1
            else:
                rel_dict[a][b] += 1

    out_f = open(os.path.join(out_dir, f"_out_{job_index}.json"), "w")
    print(f"Found {len(rel_dict)}")
    json.dump(rel_dict, out_f)
    print(f"Finished {job_index} / {num_jobs}...{filename}. {time.time() - start} seconds. Saved in {out_f}.")
    return

def merge_and_save(out_dir):
    rel_dict = {}
    in_files = glob(os.path.join(out_dir, f"_out_*.json"))
    out_f = os.path.join(out_dir, "kg_adj.txt")
    for f in tqdm(in_files):
        d = json.load(open(f, 'r'))
        for a, list_of_b in d.items():
            if not a in rel_dict:
                rel_dict[a] = {}
            for b in list_of_b:
                if not b in rel_dict[a]:
                    rel_dict[a][b] = 1
                else:
                    rel_dict[a][b] += 1
    write_rels(out_f, rel_dict)

    print(f"Removing the temporary files")
    for file in in_files:
        os.remove(file)
    return

def write_rels(fpath, rel_dict):
    with open(fpath, 'w') as out_file: 
        for q1, rels in tqdm(rel_dict.items()):
            for q2 in rels: 
                out_file.write(f"{q1}\t{q2}\n")

def main():
    set_start_method('spawn')
    start = time.time()
    args = get_arg_parser().parse_args()

    out_dir = os.path.join(args.data, args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filter_qids = []
    if len(args.qids) > 0:
        filter_qids = json.load(open(args.qids))
        if type(filter_qids) is dict:
            filter_qids = list(filter_qids.keys())
    print(f"Loaded {len(filter_qids)} qids.")
    entity_table_files = get_batch_files("entity", args)
    launch_entity_table(entity_table_files, filter_qids, out_dir, args)
    print(f"Finished in {time.time() - start}")

if __name__ == "__main__":
    main()