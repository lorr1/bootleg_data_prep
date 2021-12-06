'''
This file 

1. Reads in all entity-entity relations
2. Dumps them out in the kg_adj format 

to run: 
python3.6 -m processor.get_all_wikipedia_triples

''' 

import os, json, argparse, time

import marisa_trie
from glob import glob
from multiprocessing import set_start_method, Pool

import simple_wikidata_db.utils as utils

from bootleg_data_prep.language import ENSURE_ASCII


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str, default = '/lfs/raiders8/0/lorr1/wikidata', help = 'path to output directory')
    parser.add_argument('--out_dir', type = str, default = 'kg_output', help = 'path to output directory')
    parser.add_argument('--processes', type = int, default = 80, help = "Number of concurrent processes to spin off. ")
    parser.add_argument('--qids', default = '/lfs/raiders8/0/lorr1/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json')
    return parser 

def init_process(args):
    temp_f = args[0]
    vals = list(json.load(open(temp_f, "r", encoding="utf-8")))
    global filter_qids_global
    filter_qids_global = marisa_trie.Trie(vals)

def launch_entity_table(entity_files, filter_qids, out_dir, args):
    temp_f = os.path.join(out_dir, "_temp_filter.json")
    json.dump(list(filter_qids), open(temp_f, "w", encoding='utf8'), ensure_ascii=ENSURE_ASCII)
    print(f"Starting with {args.processes} processes")
    pool = Pool(processes = args.processes, initializer=init_process, initargs=(tuple([temp_f]),))
    messages = [(i, len(entity_files), entity_files[i], out_dir) for i in range(len(entity_files))]
    pool.map(load_and_filter_triples, messages, chunksize=1)
    merge_and_save(out_dir)
    return

def merge_and_save(out_dir):
    in_files = glob(os.path.join(out_dir, f"_out_*.json"))
    out_f = os.path.join(out_dir, "kg_triples.json")
    final_triples = {}
    for f in in_files:
        triples = json.load(open(f, 'r', encoding="utf-8"))
        for qid in triples:
            if qid not in final_triples:
                final_triples[qid] = {}
            for rel in triples[qid]:
                if rel not in final_triples[qid]:
                    final_triples[qid][rel] = []
                for qid2 in triples[qid][rel]:
                    final_triples[qid][rel].append(qid2)
    with open(out_f,'w', encoding='utf8') as wfd:
        json.dump(final_triples, wfd, ensure_ascii=ENSURE_ASCII)
    print(f"Removing the temporary files")
    for file in in_files:
        os.remove(file)
    return

def load_and_filter_triples(message):
    start = time.time()
    job_index, num_jobs, filename, out_dir = message
    triples = {}
    print(f"Starting {job_index} / {num_jobs} on {filename}.")
    for triple in utils.jsonl_generator(filename):
        qid, property_id, value = triple['qid'], triple['property_id'], triple['value']
        if len(filter_qids_global) == 0 or qid in filter_qids_global: # and value in filter_qids_global:
            if qid not in triples:
                triples[qid] = {}
            if property_id not in triples[qid]:
                triples[qid][property_id] = []
            triples[qid][property_id].append(value)
    out_f = open(os.path.join(out_dir, f"_out_{job_index}.json"), "w", encoding='utf8')
    print(f"Found {len(triples)}")
    json.dump(triples, out_f, ensure_ascii=ENSURE_ASCII)
    print(f"Finished {job_index} / {num_jobs}...{filename}. {time.time() - start} seconds. Saved in {out_f}.")
    return
            
def main():
    set_start_method('forkserver')
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
        else:
            filter_qids = set(filter_qids)
    print(f"Loaded {len(filter_qids)} qids.")
    fdir = os.path.join(args.data, "processed_batches", "entity_rels")
    entity_table_files = utils.get_batch_files(fdir)
    launch_entity_table(entity_table_files, filter_qids, out_dir, args)
    print(f"Finihsed in {time.time() - start}s.")

if __name__ == "__main__":
    main()