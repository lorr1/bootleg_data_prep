'''
This file
1. Loads alias2qid from passed entity dump
2. Loads data from file
3. Walks through mentions, and drops alias mentions which aren't in alias2qid.
4. Writes filtered sentences to file.
5. Outputs recall proportion.

To run:
python3.6 -m bootleg_data_prep.benchmarks.filter_and_compute_recall --data data/kore50 --entity_dump data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings --verbose
'''
import glob
import os
import shutil
import sys

import jsonlines
import marisa_trie
import ujson as json

sys.path.append(os.path.join(sys.path[0], "../"))
from tqdm import tqdm
import argparse
import time
from collections import defaultdict

from bootleg_data_prep.utils.classes.entity_symbols import EntitySymbols
from bootleg_data_prep.benchmarks.candidate_generators import Standard, Contextual, AIDACand
from bootleg_data_prep.utils import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/kore50/', help='Path to AIDA file')
    parser.add_argument('--entity_dump', default = 'entity_db/entity_mappings', type = str)
    parser.add_argument('--sub_dir', type=str, default='unfiltered', help='Path to AIDA file')
    parser.add_argument('--file', type=str, help='File to filter')
    parser.add_argument('--out_dir', type=str, default='filtered', help='Path to AIDA file')
    parser.add_argument('--errors_dir', type=str, default='cand_gen_filtering_errors', help='Path to errors file')
    parser.add_argument('--max_candidates', type=int, default=30, help='Maximum candidates')
    parser.add_argument('--verbose', action = 'store_true')
    parser.add_argument('--gold_given', action = 'store_true', help = "if gold QIDs are in dataset")
    parser.add_argument('--method', default='standard', choices =['standard', 'contextual', 'aida'], help = 'which candidate generator to use.')
    parser.add_argument('--expand_aliases', action='store_true', help='Whether to expand mention boundaries based on Spacy')
    parser.add_argument('--large_alias_map', default = '/dfs/scratch0/lorr1/projects/data_prep/unfiltered_data_0906/entity_db/entity_mappings/alias2qids.json', help = 'Path to large alias-to-qid map.')
    parser.add_argument('--large_title_map', default = '/dfs/scratch0/lorr1/projects/data_prep/unfiltered_data_0906/entity_db/entity_mappings/qid2title.json', help = 'Path to large qid-to-title map.')
    parser.add_argument('--wiki_pages', default = '/dfs/scratch0/lorr1/projects/data_prep/unfiltered_data_0906/', help = 'Path to wikipedia pages.')
    parser.add_argument('--aida_candidates', default = "/dfs/scratch0/lorr1/projects/bootleg-data-prep/bootleg_data_prep/benchmarks/pershina/processed/cands.json", help='Path to candidate list for AIDA')
    args = parser.parse_args()
    return args

def get_outdir(save_dir, subfolder, remove_old=False):
    out_dir = os.path.join(save_dir, subfolder)
    if remove_old and os.path.exists(out_dir):
        print(f"Deleting {out_dir}...")
        shutil.rmtree(out_dir)
    print(f"Making {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_sentences(file, args):
    sentences = []
    with jsonlines.open(file) as in_file:
        for i, obj in enumerate(in_file):
            obj['sent_idx_unq'] = i
            sentences.append(obj)
    return sentences

def write_output_file(sentences, fpath, args):
    print(f"Output to {fpath}")
    with jsonlines.open(fpath, 'w') as f:
        for sentence in sentences:
            f.write(sentence)

def write_errors(errors, out_fpath):
    with open(out_fpath, 'w') as out_file:
        for error in errors:
            out_file.write(json.dumps(error) + '\n')

def filter_and_clean(file, candidate_generator, args):
    sentences = load_sentences(file, args)
    filtered_sentences = []
    all_errors = []
    total_mentions = 0
    success_total = 0
    no_alias_exists_total = 0
    qid_not_in_cands_total = 0
    no_qid_in_dump_total = 0
    for sentence in tqdm(sentences, desc="Processing sentences"):
        total_mentions += len(sentence['aliases'])
        new_sentence, success, no_alias_exists, qid_not_in_cands, no_qid_in_dump, errors \
             = candidate_generator.filter_sentence(sentence)
        success_total += success
        no_alias_exists_total += no_alias_exists
        qid_not_in_cands_total += qid_not_in_cands
        no_qid_in_dump_total += no_qid_in_dump
        all_errors.extend(errors)
        if len([a for a in new_sentence['gold'] if a is True]) > 0:
            # new_sentence['aliases_idx'] = [i for i in range(len(new_sentence['aliases'])) if new_sentence["gold"][i] is True]
            filtered_sentences.append(new_sentence)
    print(f"{file}. Total mentions {total_mentions}. Succesfully mapped: {success_total}."
          f" Alias not found for {no_alias_exists_total} mentions. Correct QID not in candidate list for {qid_not_in_cands_total} mentions. "
          f"Correct QID not in dump for {no_qid_in_dump_total} mentions.")
    recall = {
        'total_mentions': total_mentions,
        'found_mentions': success_total,
        'alias_in_list': total_mentions - no_alias_exists_total
    }
    return filtered_sentences, all_errors, recall

def load_wikipedia_pages(args):
    # The sorting is to make it determinisitc and due to some overlapping QIDs, we get one more mention with reverse=True for RSS.
    # We should find out what QID this is and fix this, but just so folks know why the reverse=True is here.
    files = sorted(glob.glob(os.path.join(args.wiki_pages, '*.jsonl')), reverse=True)
    print(f"Loading {len(files)} wikipedia pages...")
    st = time.time()
    qid2page = {}
    for file in tqdm(files):
        with jsonlines.open(file) as in_file:
            for obj in in_file:
                qid2page[obj['qid']] = obj
    print(f"Took {time.time()-st}s to load {len(files)} files")
    return qid2page

def load_large_alias_map(args, entity_dump):
    temp_dir = "_temp"
    utils.ensure_dir(temp_dir)
    temp_f = os.path.join(temp_dir, "alias_map.pkl")
    if os.path.exists(temp_f):
        print(f"Reading alias map from {temp_f}")
        filtered_alias_map = utils.load_pickle_file(temp_f)
        return filtered_alias_map
    large_alias_to_qid_map = json.load(open(args.large_alias_map))

    # filter down and remove qids which aren't contained in the entity dump
    filtered_alias_map = {}
    for alias, qid_list in large_alias_to_qid_map.items():
        filtered_qid_list = []
        for qid, count in qid_list:
            if entity_dump.qid_exists(qid):
                filtered_qid_list.append([qid, count])
        filtered_alias_map[alias] = filtered_qid_list
    utils.dump_pickle_file(temp_f, filtered_alias_map)
    return filtered_alias_map

def load_large_title_map(args):
    return json.load(open(args.large_title_map))

def main(args):
    out_dir = get_outdir(args.data_dir, args.out_dir)
    errors_dir = get_outdir(args.data_dir, args.errors_dir)

    print(f"Candidate generator: {args.method}")
    if args.method == 'aida':
        entity_dump = EntitySymbols.load_from_cache(load_dir=args.entity_dump)
        candidate_generator = AIDACand(args, entity_dump)
    elif args.method == 'standard':
        entity_dump = EntitySymbols.load_from_cache(load_dir=args.entity_dump)
        candidate_generator = Standard(args, entity_dump)
    elif args.method == 'contextual':
        qid2page = load_wikipedia_pages(args)
        entity_dump = EntitySymbols.load_from_cache(load_dir=args.entity_dump)
        alias_map = load_large_alias_map(args, entity_dump)
        title_map = load_large_title_map(args)
        alias_tri = marisa_trie.Trie(alias_map.keys())
        candidate_generator = Contextual(args, entity_dump, qid2page, alias_map, alias_tri, title_map)

    if args.file:
        files = [args.file]
    else:
        files = glob.glob(os.path.join(args.data_dir, args.sub_dir, '*.jsonl'))
    print(f"Found files {files}")
    recall_results = {}
    for f in files:
        print(f"Filtering file {f}")
        filtered_sentences, errors, recall = filter_and_clean(f, candidate_generator, args)
        fname = os.path.basename(f)
        write_output_file(filtered_sentences, os.path.join(out_dir, fname), args)
        write_errors(errors, os.path.join(errors_dir, fname.replace('jsonl', 'txt')))
        recall_results[fname] = recall

    # save entity dump to output directory
    entity_dump_dir = os.path.join(out_dir, 'entity_db/entity_mappings')
    entity_dump._alias2qids = candidate_generator.new_alias2qid
    entity_dump.save(entity_dump_dir)
    print(f"Saved new entity dump to {entity_dump_dir}")

    result_fpath = os.path.join(errors_dir, 'recall_results.json')
    json.dump(recall_results, open(result_fpath, 'w'))
    print(f"Saved results to {recall_results}")
    return recall_results



if __name__ == '__main__':
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    main(args)
