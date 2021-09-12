'''
This file 

1. Reads in all wikidata aliases from file
2. Goes through triples and adds more aliases from a select set of properties 
3. Performs first/last name augmentation 
4. Dumps all results to file 

to run: 
python3.6 -m 

''' 

    
import os, json, argparse, time, shutil
from tqdm import tqdm 
from multiprocessing import set_start_method, Pool
from collections import defaultdict
import unicodedata

import simple_wikidata_db.utils as utils
from simple_wikidata_db.preprocess_dump import ALIAS_PROPERTIES

from bootleg_data_prep.language import stopwords, get_lnrm, ensure_ascii, HumanNameParser


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str, default = '/lfs/raiders8/0/lorr1/wikidata', help = 'path to output directory')
    parser.add_argument('--out_file', type = str, default = 'augmented_alias_map_large_uncased_1216.jsonl', help = 'path to output directory')
    parser.add_argument('--processes', type = int, default = 10, help = "Number of concurrent processes to spin off. ")
    parser.add_argument('--qids', type = str, default = "/lfs/raiders8/0/lorr1/data/utils/all_wikipedia_qids_1016.json") # Taken from Wikipedia part of Wikidata dump
    parser.add_argument('--batch_size', type = int, default = 100000)
    parser.add_argument('--stripandlower', action='store_true', help='If set, will stripandlower and strip punctuation of aliases.')
    return parser 

def get_aliases_from_table(alias_files, args):
    pool = Pool(processes = args.processes)
    messages = [(i, len(alias_files), alias_files[i]) for i in range(len(alias_files))]
    list_of_qid2alias = pool.map(load_alias_file, messages)
    return list_of_qid2alias

def load_alias_file(message): 

    job_index, num_jobs, filename  = message
    qid2alias = {}
    for triple in utils.jsonl_generator(filename):
        qid, alias = triple['qid'], triple['alias']
        if not qid in qid2alias:
            qid2alias[qid] = []
        qid2alias[qid].append(alias)

    print(f"Finished {job_index} / {num_jobs}...{filename}. Found {len(qid2alias)} entities.")
    return qid2alias

def get_aliases_from_values(value_files, args):
    pool = Pool(processes = args.processes)
    messages = [(i, len(value_files), value_files[i]) for i in range(len(value_files))]
    list_of_qid2alias = pool.map(load_value_file, messages, chunksize=1)
    return list_of_qid2alias

def load_value_file(message): 

    job_index, num_jobs, filename  = message
    qid2alias = {}
    for triple in utils.jsonl_generator(filename):
        qid, property_id, value = triple['qid'], triple['property_id'], triple['value'].lower()
        if not property_id in ALIAS_PROPERTIES:
            continue
        if not qid in qid2alias:
            qid2alias[qid] = []
        qid2alias[qid].append(value)

    print(f"Finished {job_index} / {num_jobs}...{filename}. Found {len(qid2alias)} entities.")
    return qid2alias

def get_types_and_aliases_from_entities(entity_files, args):
    pool = Pool(processes = args.processes)
    messages = [(i, len(entity_files), entity_files[i]) for i in range(len(entity_files))]
    results = pool.map(load_entity_file, messages, chunksize=1)
    list_of_qid2alias = [r['alias'] for r in results]
    list_of_qid2type = [r['type'] for r in results]
    return list_of_qid2alias, list_of_qid2type

def load_entity_file(message): 

    job_index, num_jobs, filename  = message
    qid2alias = {}
    qid2type = {}
    for triple in utils.jsonl_generator(filename):
        qid, property_id, value = triple['qid'], triple['property_id'], triple['value']
        if property_id in ALIAS_PROPERTIES:
            if not qid in qid2alias:
                qid2alias[qid] = []
            qid2alias[qid].append(value)
        if property_id == 'P31':
            qid2type[qid] = value
    print(f"Finished {job_index} / {num_jobs}...{filename}. Found {len(qid2alias)} entities.")
    return {'alias': qid2alias, 'type': qid2type}

def merge_aliases(list_of_qid2alias, stripandlower):
    qid2alias = defaultdict(set)
    all_aliases = set()
    for d in tqdm(list_of_qid2alias):
        for q, aliases in d.items():
            for a in aliases:
                before_a = a
                a = get_lnrm(a, stripandlower)
                qid2alias[q].add(a)
                all_aliases.add(a)
    for q, a in qid2alias.items():
        qid2alias[q] = list(a)
    print(f"Extracted {len(qid2alias)} QIDS and {len(all_aliases)} total aliases.")
    return qid2alias

def generate_short_long_names(qid2alias, human_qid):
    augmented_qid2alias = {}
    for qid, orig_aliases in tqdm(qid2alias.items()):
        new_aliases = set(orig_aliases)
        if qid in human_qid:
            for alias in orig_aliases:
                name = HumanNameParser(alias)
                if len(name.first) > 0 and name.first not in stopwords:
                    new_aliases.add(name.first)
                if len(name.last) > 0 and name.last not in stopwords:
                    new_aliases.add(name.last)
                print("ALIAS", alias, name.first, name.last)
        augmented_qid2alias[qid] = list(new_aliases)
    print(f"Finished augmentation. {len(qid2alias)} QIDS.")
    return augmented_qid2alias
            
def main():
    set_start_method('spawn')
    start = time.time()
    args = get_arg_parser().parse_args()
    out_file = args.out_file

    if len(os.path.dirname(out_file)) > 0 and not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    print("args.qids", args.qids)
    if len(args.qids) > 0:
        filter_qids = json.load(open(args.qids))
        if type(filter_qids) is dict:
            qid_filter = set(filter_qids.keys())
        else:
            qid_filter = set(filter_qids)
    else:
        qid_filter = set()
    print(f"Loaded {len(qid_filter)} qids.")

    fdir = os.path.join(args.data, "processed_batches", "aliases")
    alias_table_files = utils.get_batch_files(fdir)
    list_of_qid2alias = get_aliases_from_table(alias_table_files, args)

    fdir = os.path.join(args.data, "processed_batches", "entity_values")
    value_table_files = utils.get_batch_files(fdir)
    list_of_qid2alias.extend(get_aliases_from_values(value_table_files, args))

    fdir = os.path.join(args.data, "processed_batches", "entity_rels")
    entity_table_files = utils.get_batch_files(fdir)
    _, list_of_qid2types = get_types_and_aliases_from_entities(entity_table_files, args)

    print("Building human type map")
    human_qid = {}
    for d in list_of_qid2types:
        for qid, typ in d.items():
            if typ == 'Q5':
                human_qid[qid] = 1
    print(f"Found {len(human_qid)} entities of type individual.")
    qid2alias = merge_aliases(list_of_qid2alias, args.stripandlower)
    print(qid2alias['Q10993'])
    print('Q10993' in human_qid)
    qid2alias = generate_short_long_names(qid2alias, human_qid)
    print("Inverting qid2alias...")
    alias2qid = {}
    for qid, aliases in qid2alias.items():
        if len(qid_filter) > 0 and not qid in qid_filter:
            continue 
        for alias in aliases:
            if not alias in alias2qid:
                alias2qid[alias] = []
            alias2qid[alias].append(qid)
    print(f"{len(alias2qid)} aliases.")
    print(f"Saving to file {args.out_file}...")
    with open(args.out_file, "w") as out_file: 
        json.dump(alias2qid, out_file, ensure_ascii=ensure_ascii)


if __name__ == "__main__":
    main()