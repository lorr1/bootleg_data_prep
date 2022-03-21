'''
This file
1. Reads in data from <data_dir>/<subfolder_name>
2. Merges ALL data together, shuffles the lines, and outputs the data into small chuns into <data_dir>/<subfolder_name>/<train/dev/test>

After this, run generate_slices.py

Example run command:

'''
import argparse
import collections
import copy
import hashlib
import os
import random
import shutil
import time

import ujson as json
from tqdm import tqdm

import bootleg_data_prep.utils.utils as utils
import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.language import ENSURE_ASCII


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=2, choices=range(1, 100), help='Percent for test and dev data; rest goes to train')
    parser.add_argument('--data_dir', type=str, default='data/wiki_dump', help='Directory for data is.')
    parser.add_argument('--subfolder_name', type=str, default="filtered_data")
    parser.add_argument('--without_weak_label', action='store_true', help='Whether to remove non-golds for training data or not. By default they are kept in.')
    parser.add_argument('--hash_keys', nargs="+", default=['parent_qid', 'parent_title'], help='What keys should be use to shuffle the data')
    parser.add_argument('--sentence_split', action='store_true', help='Whether to make the hash key be based on sentence as well as page')
    parser.add_argument('--bytes', type=str, default='10M')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

def my_hash(key):
    return int(hashlib.sha1(key).hexdigest(), 16)


def keep_only_gold_aliases(sent):
    # Remove all aliases that are not golds!
    new_sent = copy.deepcopy(sent)
    new_sent['aliases'] = []
    new_sent['spans'] = []
    new_sent['qids'] = []
    new_sent['gold'] = []
    removed = 0
    for i in range(len(sent['gold'])):
        if sent['gold'][i] is True:
            new_sent['aliases'].append(sent['aliases'][i])
            new_sent['spans'].append(sent['spans'][i])
            new_sent['qids'].append(sent['qids'][i])
            new_sent['gold'].append(sent['gold'][i])
        else:
            removed += 1
    return new_sent, removed

def unswap_gold_aliases(sent):
    # Return the aliases to the original one for gold aliases (we swap for training to increase conflict)
    for i in range(len(sent['gold'])):
        if sent['gold'][i] is True:
            if sent.get('unswap_aliases', 'aliases')[i] != sent['aliases'][i]:
                sent['aliases'][i] = sent.get('unswap_aliases', 'aliases')[i]
    return sent

def main():
    gl_start = time.time()
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    random.seed(args.seed)

    load_dir = os.path.join(args.data_dir, args.subfolder_name)
    print(f"Loading data from {load_dir}...")
    files = prep_utils.glob_files(f"{load_dir}/*")
    print(f"Loaded {len(files)} files.")

    out_dir = os.path.join(args.data_dir, f"{args.subfolder_name}")
    vars(args)['saved_data_folder'] = out_dir

    # Setup stats dir
    stats_dir = os.path.join(out_dir, f"stats")

    out_file_with = os.path.join(stats_dir, "alias_qid_traindata_withaugment.json")
    out_file_without = os.path.join(stats_dir, "alias_qid_traindata_withoutaugment.json")
    train_qidcnt_file = os.path.join(out_dir, "entity_db", "entity_mappings", "qid2cnt.json")
    print(f"Will output to {out_file_with} and {out_file_without} and counts to {train_qidcnt_file}")
    alias_qid_with = collections.defaultdict(lambda: collections.defaultdict(int))
    alias_qid_without = collections.defaultdict(lambda: collections.defaultdict(int))

    vars(args)['stats_dir'] = stats_dir
    prep_utils.save_config(args, "split_config.json")
    # Store the current file handle
    # Make the train, dev, and test subfolders
    # Get train, dev, test slits
    counters = {}
    splits = {}
    for fold in ["train", "dev", "test"]:
        key = os.path.join(out_dir, fold)
        if os.path.exists(key):
            shutil.rmtree(key)
        utils.ensure_dir(key)
        counters[key] = tuple([0, open(os.path.join(key, f"out_0.jsonl"), "w")])
        if fold == "train":
            splits[key] = list(range(2*args.split, 100))
        elif fold == "dev":
            splits[key] = list(range(args.split, 2*args.split))
        elif fold == "test":
            splits[key] = list(range(0, args.split))

    # Get file size for splits
    power = 'kmg'.find(args.bytes[-1].lower()) + 1
    file_size = int(args.bytes[:-1]) * 1024 ** power

    random.seed(args.seed)
    # Read in all data, shuffle, and split
    start = time.time()
    lines = []
    for file in tqdm(sorted(files)):
        with open(file, "r") as in_f:
            for line in in_f:
                line_strip = json.loads(line.strip())
                hash_keys_for_item = []
                for key in args.hash_keys:
                    hash_keys_for_item.append(line_strip[key])
                if args.sentence_split:
                    hash_keys_for_item = [line_strip["doc_sent_idx"]] + hash_keys_for_item
                key = str(tuple(hash_keys_for_item)).encode('utf-8')
                lines.append([my_hash(key), line])
    print(f"Read data in {time.time() - start} seconds.")
    start = time.time()
    random.shuffle(lines)
    print(f"Shuffled in {time.time() - start} seconds.")

    # If data different lengths, this will reset random here
    print(f"Starting to write out {len(lines)} lines")
    start = time.time()
    trainqid2cnt = collections.defaultdict(int)
    line_idx = 0
    total_removed = 0
    for hash_key, line in tqdm(lines):
        line = json.loads(line.strip())
        if 'gold' not in line:
            assert 'anchor' in line
            line['gold'] = line['anchor']
            del line['anchor']
        spl = hash_key % 100
        for key in splits:
            if spl in splits[key]:
                # if the split is not train, remove all non-gold aliases
                if (args.without_weak_label):
                    line, removed = keep_only_gold_aliases(line)
                    total_removed += removed
                # update train stats
                if (key == os.path.join(out_dir, "train")):
                    for alias, qid, gold in zip(line["aliases"], line["qids"], line["gold"]):
                        trainqid2cnt[qid] += 1
                        alias_qid_with[alias][qid] += 1
                        if gold:
                            alias_qid_without[alias][qid] += 1
                # use unswapped GOLD aliases for test/dev
                if (key != os.path.join(out_dir, "train")):
                    line = unswap_gold_aliases(line)
                if len(line['aliases']) == 0:
                    continue
                (idx, out_f) = counters[key]
                if out_f.tell() > file_size:
                    out_f.close()
                    idx += 1
                    out_f = open(os.path.join(key, f"out_{idx}.jsonl"), "w")
                    counters[key] = tuple([idx, out_f])
                line["sent_idx_unq"] = line_idx
                line_idx += 1
                out_f.write(json.dumps(line, sort_keys=True, ensure_ascii=ENSURE_ASCII) + "\n")

    utils.dump_json_file(out_file_with, alias_qid_with)
    utils.dump_json_file(out_file_without, alias_qid_without)
    utils.dump_json_file(train_qidcnt_file, trainqid2cnt)
    print(f"Finished writing files in {time.time() - start} seconds. Removed {total_removed} non-gold aliases from dev and test and train.")
    # Closing files
    for key in tqdm(splits):
        counters[key][1].close()
    print(f"Close files")
    print(f"Finished merge_shuff_split in {time.time() - gl_start} seconds.")
    

if __name__ == '__main__':
    main()