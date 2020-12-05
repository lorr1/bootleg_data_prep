import argparse
from collections import defaultdict
import glob

import ujson as json
import multiprocessing
import os
import shutil
import time

from jsonlines import jsonlines
from tqdm import tqdm

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils
import bootleg_data_prep.utils.data_prep_utils as prep_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='de')
    parser.add_argument('--entity_input_dir', type=str, default='/lfs/raiders8/0/lorr1/de_bootleg/', help='Where input files are.')
    parser.add_argument('--input_dir', type=str, default='/lfs/raiders8/0/lorr1/de_bootleg/input', help='Where input files are.')
    parser.add_argument('--output_dir', type=str, default='/lfs/raiders8/0/lorr1/de_bootleg/raw', help='Where entity dump is saved.')
    parser.add_argument('--max_candidates', type=int, default=30)
    parser.add_argument('--processes', type=int, default=60)
    args = parser.parse_args()
    return args

def transform_al(alias):
    return alias.lower()

def get_num_lines(path):
    return sum(1 for line in open(path,'r'))

def load_wpid2qid(path, lang, all_qids):
    in_file = os.path.join(path, f"{lang}_wp_to_wd.jsonl")
    num_lines = get_num_lines(in_file)
    wpid2qid = {}
    print(f"Loading wpid to qid")
    with jsonlines.open(in_file, "r") as in_f:
        for line in tqdm(in_f, total=num_lines):
            # Only add qids if they are in our set
            if line["wd_id"] in all_qids:
                wpid2qid[line["wp_id"]] = line["wd_id"]
    return wpid2qid

def load_alias2qids(arg_max_candidates, lang, path):
    alias2qids_dict = defaultdict(set)
    qid2freq = {}
    qid2title = {}
    max_al_length = -1
    max_cands = -1
    in_file = os.path.join(path, f"{lang}_aliases.jsonl")
    num_lines = get_num_lines(in_file)
    print(f"Loading alias dictionary")
    with jsonlines.open(in_file, "r") as in_f:
        for line in tqdm(in_f, total=num_lines):
            all_aliases = []
            qid = line["id"]
            wikidata_aliases = [transform_al(al) for al in line["aliases"]] if "aliases" in line else []
            title = line["primary"] if "primary" in line else None
            gold_aliases = [transform_al(al) for al in line["gold"].keys()] if "gold" in line else []
            qid_freq = line["freq"] if "freq" in line else 0

            # if title is none, try to take one for the gold aliases
            if title is None:
                if len(gold_aliases) > 0:
                    title = gold_aliases[0]
                else:
                    title = qid
            assert title is not None
            assert qid not in qid2title
            qid2title[qid] = title
            qid2freq[qid] = qid_freq

            # lower title is alias
            all_aliases.append(transform_al(title))
            # each wikidata alias is alias
            for al in wikidata_aliases:
                all_aliases.append(al)
            # each gold alias is alias
            for al in gold_aliases:
                all_aliases.append(al)
            # add to alias2qids
            for al in all_aliases:
                alias2qids_dict[al].add(qid)
    
    alias2qids = {}
    print(f"Parsing aliases after loading")
    for al in tqdm(alias2qids_dict):
        qid_cands = [[q, qid2freq[q]] for q in alias2qids_dict[al]]
        qid_cands = sorted(qid_cands, key=lambda x: x[1], reverse=True)
        if len(qid_cands) > arg_max_candidates:
            qid_cands = qid_cands[:arg_max_candidates]
        alias2qids[al] = qid_cands
        max_cands = max(max_cands, len(qid_cands))
        max_al_length = max(max_al_length, len(al.split()))
    return alias2qids, qid2title, max_al_length, max_cands

def launch_subprocess_filter_data(args, out_dir, entity_symbols, files):
    temp_folder = os.path.join(out_dir, "temp_dump")
    utils.ensure_dir(temp_folder)
    print(f"Saving entity_symbol files in {temp_folder}")

    qid2title_f = os.path.join(temp_folder, "qid2title.json")
    alias2qids_f = os.path.join(temp_folder, "alias2qids.json")
    utils.dump_json_file(qid2title_f, entity_symbols.get_qid2title())
    utils.dump_json_file(alias2qids_f, entity_symbols.get_alias2qids())

    all_process_args = [tuple([i,
                               len(files),
                               args,
                               files[i],
                               out_dir,
                               qid2title_f,
                               alias2qids_f
                               ]) for i in range(len(files))]
    pool = multiprocessing.Pool(processes=args.processes)
    list_of_stats = tqdm(pool.imap(subprocess_data_filter, all_process_args, chunksize=1))
    stats = prep_utils.aggregate_list_of_dictionaries(list_of_stats)
    pool.close()
    print(f"Cleaning up {temp_folder}")
    shutil.rmtree(temp_folder)
    return stats

def subprocess_data_filter(all_args):
    i, total, args, in_filepath, out_dir, qid2title_f, alias2qids_f = all_args

    qid2title = utils.load_json_file(qid2title_f)
    alias2qids = utils.load_json_file(alias2qids_f)
    out_fname = f"out_{i}.jsonl"
    out_file = open(os.path.join(out_dir, out_fname), "w")
    print(f"Starting {i}/{total}. Reading in {in_filepath}. Outputting to {out_fname}. Outdir {out_dir}")

    statistics = {'total_mentions': 0, 'total_preserved': 0, 'total_dropped': 0, 'dropped_not_in_max_cand': 0, 'dropped_no_qid': 0}
    # load file -- should be JSONL with
    with open(in_filepath, 'r') as in_file:
        for line in in_file:
            sent_obj = json.loads(line.strip())
            statistics['total_mentions'] += len(sent_obj['qids'])
            # # LAUREL
            # for al in sent_obj['aliases']:
            #     if al not in alias2qids:
            #         print("BAD SENTENCE OBJ")
            #         print(json.dumps(sent_obj, indent=4))
            # items = list(filter(lambda x:(not args.train_in_candidates) or ((x[0] in alias2qids) and (x[1] in [y[0] for y in alias2qids[x[0]]])),
            #                     zip(sent_obj['aliases'], sent_obj['qids'], sent_obj['spans'], sent_obj['gold'])))
            # # LAUREL
            # lower case aliases
            sent_obj['aliases'] = [transform_al(al) for al in sent_obj['aliases']]
            items = list(filter(lambda x: (x[1] in [y[0] for y in alias2qids[x[0]]]),
                                zip(sent_obj['aliases'], sent_obj['qids'], sent_obj['spans'])))
            temp_len = len(items)
            statistics['dropped_not_in_max_cand'] += len(sent_obj['qids']) - temp_len
            items = list(filter(lambda x: x[1] in qid2title, items))
            statistics['total_dropped'] += len(sent_obj['qids']) - len(items)
            # there should be no difference between these
            assert temp_len - len(items) == 0
            statistics['dropped_no_qid'] += temp_len - len(items)
            if len(items) == 0:
                continue
            aliases, qids, spans = zip(*items)
            if len(aliases) > 0:
                new_sent_obj = sent_obj
                new_sent_obj['aliases'] = aliases
                new_sent_obj['qids'] = qids
                new_sent_obj['spans'] = spans
                # This is wikipedia data without augmented labels
                new_sent_obj['gold'] = [True for _ in range(len(aliases))]
                out_file.write(json.dumps(new_sent_obj) + '\n')
                # Update stats
                statistics['total_preserved'] += len(aliases)
    assert statistics['total_mentions'] == (statistics['total_dropped'] + statistics['total_preserved'])
    out_file.close()
    print(f"Finished {i}/{total}. Data written to {os.path.join(out_dir, out_fname)}.")
    return statistics

def main():
    gl_start = time.time()
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    entity_out_dir = os.path.join(args.output_dir, "entity_db/entity_mappings")
    utils.ensure_dir(entity_out_dir)

    alias2qids, qid2title, max_al_length, max_candidates = load_alias2qids(args.max_candidates, args.lang, args.entity_input_dir)
    wpid2qid = load_wpid2qid(args.entity_input_dir, args.lang, set(qid2title.keys()))

    entity_symbols = EntitySymbols(
        max_candidates=max_candidates,
        max_alias_len=max_al_length,
        alias2qids=alias2qids,
        qid2title=qid2title,
        wpid2qid=wpid2qid
    )
    entity_symbols.dump(entity_out_dir)
    print(f"Finished generating entity dump in {time.time() - gl_start} seconds.")

    all_in_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    print(f"Found {len(all_in_files)} files")

    stats = launch_subprocess_filter_data(args, args.output_dir, entity_symbols, all_in_files)
    print("="*5, "Recall Statistics", "="*5)
    print(json.dumps(stats, indent=4))
    print(f"Finished data_filter in {time.time() - gl_start} seconds.")

if __name__ == '__main__':
    main()