'''
This file 

1. Reads in all entity-entity relations
2. Saves all triples for properties: P31 (instance of),  P106 (occupation)

to run: 
python3.6 -m processor.get_types

''' 

    
import os, json, argparse, time, shutil
from glob import glob

import marisa_trie
from tqdm import tqdm 
from multiprocessing import set_start_method, Pool

from collections import defaultdict, Counter

import simple_wikidata_db.utils as utils

HUMAN_QID = 'Q5'
TWIN_QID = 'Q159979'
OCCUPATION = 'P106'
INSTANCE_OF = 'P31'
SUBCLASS_OF = 'P279'
# Get instance of, subclass of, and occupation (for people)
TYPE_PIDS = {OCCUPATION, INSTANCE_OF, SUBCLASS_OF}
# We use occupation types instead of generic human types
AVOID_TYPES = {HUMAN_QID, TWIN_QID}

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str, default = '/lfs/raiders8/0/lorr1/wikidata', help = 'path to output directory')
    parser.add_argument('--out_dir', type = str, default = 'types_output', help = 'path to output directory')
    parser.add_argument('--processes', type = int, default = 80, help = "Number of concurrent processes to spin off. ")
    parser.add_argument('--qids', default = '/lfs/raiders8/0/lorr1/es_bootleg/es_qids.json')
    return parser 

def init_process(args):
    temp_f = args[0]
    vals = list(json.load(open(temp_f, "r")))
    global filter_qids_global
    filter_qids_global = marisa_trie.Trie(vals)

def launch_entity_table(entity_files, qid_to_title, filter_qids, out_dir, args):
    temp_f = os.path.join(out_dir, "_temp_filter.json")
    json.dump(list(filter_qids), open(temp_f, "w"))
    print(f"Starting with {args.processes} processes")
    pool = Pool(processes = args.processes, initializer=init_process, initargs=(tuple([temp_f]),))
    messages = [(i, len(entity_files), entity_files[i], out_dir) for i in range(len(entity_files))]
    pool.map(load_entity_file, messages, chunksize=1)
    merge_and_save(out_dir, qid_to_title)
    return

def load_entity_file(message):
    start = time.time()
    job_index, num_jobs, filename, out_dir = message
    type_dict = defaultdict(set)
    for triple in utils.jsonl_generator(filename):
        qid, property_id, value = triple['qid'], triple['property_id'], triple['value']
        if property_id in TYPE_PIDS and value not in AVOID_TYPES and (len(filter_qids_global) == 0 or qid in filter_qids_global):
            type_dict[qid].add(value)

    out_f = open(os.path.join(out_dir, f"_out_{job_index}.json"), "w")
    print(f"Found {len(type_dict)}")
    type_dict = dict(type_dict)
    # convert to list type for json serialization
    for k in list(type_dict.keys()):
        type_dict[k] = list(type_dict[k])
    json.dump(type_dict, out_f)
    print(f"Finished {job_index} / {num_jobs}...{filename}. Fetched types for {len(type_dict)} entities. {time.time() - start} seconds.")
    return dict(type_dict)

def merge_and_save(out_dir, qid_to_title):
    type_dict = defaultdict(set)
    type_freq = defaultdict(int)
    in_files = glob(os.path.join(out_dir, f"_out_*.json"))
    for f in tqdm(in_files):
        d = json.load(open(f, 'r'))
        for qid, types in d.items():
            for qtype in types:
                type_dict[qid].add(qtype)
                type_freq[qtype] += 1
    # Sort types based on most to least frequent
    sorted_typs = sort_types(type_dict, type_freq)
    write_types(out_dir, sorted_typs, qid_to_title)

    with open(os.path.join(out_dir, 'type_freqs.json'), 'w') as out_file:
        json.dump(type_freq, out_file)
    print(f"Removing the temporary files")
    for file in in_files:
        os.remove(file)

    return

# Sort types based on most to least frequent
def sort_types(type_dict, type_freq):
    sorted_types = {} # map QID to list of ordered type qids
    for qid, types in type_dict.items():
        stypes = sorted(types, key=lambda i: type_freq[i], reverse=True)
        sorted_types[qid] = stypes 
    return sorted_types

def write_types(out_dir, type_list, qid_to_title):
    index = 1
    typetitle2index = {}
    title2typeqid = {}
    typeqid2index = {}
    with open(os.path.join(out_dir, 'wikidata_types.json'), 'w') as out_file:
        new_type_list = {}
        for qid, types in type_list.items(): 
            type_indices = []
            for qtype in types:
                typetitle = qid_to_title.get(qtype, qtype)
                if not qtype in typeqid2index:
                    # New QID we haven't seen before
                    typeqid2index[qtype] = index
                    if typetitle in title2typeqid:
                        assert title2typeqid[typetitle] != qtype
                        typetitle += f"_{qtype}"
                        assert typetitle not in title2typeqid
                    title2typeqid[typetitle] = qtype
                    typetitle2index[typetitle] = index
                    index += 1
                type_indices.append(typeqid2index[qtype])
            new_type_list[qid] = type_indices
        json.dump(new_type_list, out_file)
    
    with open(os.path.join(out_dir, 'wikidatatitle_to_typeid.json'), 'w') as out_file:
        json.dump(typetitle2index, out_file)
    with open(os.path.join(out_dir, 'wikidatatitle_to_typeqid.json'), 'w') as out_file:
        json.dump(title2typeqid, out_file)
    print(f"Writtten to {out_dir}")

def read_in_wikidata_title(args):
    fdir = os.path.join(args.data, "processed_batches", "label")
    wikidata_files = utils.get_batch_files(fdir)
    id_to_title = {}
    for file in tqdm(wikidata_files, desc="Reading in wikidata files"):
        with open(file, "r") as in_f:
            for line in in_f:
                line = json.loads(line)
                id_to_title[line["qid"]] = line["label"]
    return id_to_title

def main():
    set_start_method('spawn')
    start = time.time()
    args = get_arg_parser().parse_args()

    out_dir = os.path.join(args.data, args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Loading wikidata qids to titles")
    qid_to_title = read_in_wikidata_title(args)

    print("args.qids", args.qids)
    if len(args.qids) > 0:
        filter_qids = json.load(open(args.qids))
        if type(filter_qids) is dict:
            filter_qids = set(filter_qids.keys())
        else:
            filter_qids = set(json.load(open(args.qids)))
    else:
        filter_qids = []
    print(f"Loaded {len(filter_qids)} qids.")
    fdir = os.path.join(args.data, "processed_batches", "entity")
    entity_table_files = get_batch_files(fdir)
    launch_entity_table(entity_table_files, qid_to_title, filter_qids, out_dir, args)
    print(f"Finished in {time.time() - start}")


if __name__ == "__main__":
    main()