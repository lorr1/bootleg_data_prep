
import argparse
import os
import time
import ujson
from multiprocessing import Pool

import simple_wikidata_db.utils as utils
from language import ENSURE_ASCII

INSTANCE_OF_PROP = "P31"
HUMAN = {"Q5", "Q159979"}
GENDER = "P21"

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str, default = '/lfs/raiders8/0/lorr1/wikidata', help = 'path to output directory')
    parser.add_argument('--out_dir', type = str, default = 'persen_gender', help = 'path to output directory')
    parser.add_argument('--processes', type = int, default = 30, help = "Number of concurrent processes to spin off. ")
    return parser 


def launch_entity_table(entity_files, args):
    print(f"Starting with {args.processes} processes")
    pool = Pool(processes = args.processes)
    messages = [(i, len(entity_files), entity_files[i]) for i in range(len(entity_files))]
    res = pool.map(load_entity_file, messages, chunksize=1)
    return res

def load_entity_file(message):
    start = time.time()
    job_index, num_jobs, filename = message
    ids = set()
    id2gender = {}
    for triple in utils.jsonl_generator(filename):
        qid, property_id, value = triple['qid'], triple['property_id'], triple['value']
        if property_id == INSTANCE_OF_PROP and value in HUMAN:
            ids.add(qid)
        if property_id == GENDER:
            id2gender[qid] = value

    print(f"Finished {job_index} / {num_jobs}...{filename}. Fetched types for {len(ids)} entities and {len(id2gender)} id2gender. {time.time() - start} seconds.")
    return ids, id2gender

def main():
    start = time.time()
    args = get_arg_parser().parse_args()

    out_dir = os.path.join(args.data, args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fdir = os.path.join(args.data, "processed_batches", "entity_rels")
    external_tables = utils.get_batch_files(fdir)
    all_res = launch_entity_table(external_tables, args)

    final_ids = set()
    final_id2gender = {}
    for ids, id2gen in all_res:
        for id in ids:
            final_ids.add(id)
        for id in id2gen:
            final_id2gender[id] = id2gen[id]

    out_fpath = os.path.join(out_dir, 'person_qids.json')
    with open(out_fpath, 'w') as out_file:
        ujson.dump(list(final_ids), out_file, ensure_ascii=ENSURE_ASCII, indent=4)

    out_fpath2 = os.path.join(out_dir, 'person_gender.json')
    with open(out_fpath2, 'w') as out_file:
        ujson.dump(final_id2gender, out_file, ensure_ascii=ENSURE_ASCII, indent=4)
    print(f"Written {len(final_ids)} to {out_fpath} and person2gender to {out_fpath2} in {time.time()-start}")
    
    
if __name__ == "__main__":
    main()