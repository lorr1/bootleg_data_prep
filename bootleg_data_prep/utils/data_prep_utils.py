# A random assortment of utility functions that are helpful for data prep
import glob
import shutil

import marisa_trie
import psutil
from jsonlines import jsonlines
from tqdm import tqdm
import time
from collections import defaultdict
from datetime import datetime
import os

from bootleg_data_prep.utils import utils


def print_memory():
    process = psutil.Process(os.getpid())
    print(f"{int(process.memory_info().rss)/1024**3} GB ({process.memory_percent()} %) memory used process {process}")

def load_qid_title_map(title_to_qid_fpath):
    start = time.time()
    title_to_qid = {}
    qid_to_all_titles = defaultdict(set)
    wpid_to_qid = {}
    qid_to_title = {}
    all_rows = []
    with jsonlines.open(title_to_qid_fpath, 'r') as in_file:
        for items in tqdm(in_file, total=sum([1 for _ in open(title_to_qid_fpath)])):
            # the title is the url title that may be redirected to another wikipedia page
            qid, title, wikidata_title, wikipedia_title, wpid = items['qid'], items['title'], items['wikidata_title'], items['wikipedia_title'], items['id']
            if str(qid) == "-1":
                continue
            all_rows.append([qid, title, wikidata_title, wikipedia_title, wpid])
            qid_to_all_titles[qid].add(wikidata_title)
            qid_to_all_titles[qid].add(wikipedia_title)
            qid_to_all_titles[qid].add(title)

            # We want to keep the wikipedia titles
            if wikipedia_title in title_to_qid and qid != title_to_qid[wikipedia_title]:
                print(f"Wikipedia Title {wikipedia_title} for {title_to_qid[wikipedia_title]} already exists and we want {qid}")
            title_to_qid[wikipedia_title] = qid
            # if wikipedia_title.lower() in title_to_qid and qid != title_to_qid[wikipedia_title.lower()]:
            #     print(f"Wikipedia Title {wikipedia_title.lower()} for {title_to_qid[wikipedia_title.lower()]} already exists and we want {qid}")
            # title_to_qid[wikipedia_title.lower()] = qid
            qid_to_title[qid] = wikipedia_title
            wpid_to_qid[wpid] = qid

        # The title represents a redirect. We only want to add them if the redirect title does not already point to a QID from Wikipedia.
        for item in tqdm(all_rows, desc="Adding extra titles"):
            qid, title, wikidata_title, wikipedia_title, wpid = item
            if title not in title_to_qid:
                title_to_qid[title] = qid

    print(f"Loaded title-qid map for {len(title_to_qid)} titles from {title_to_qid_fpath}. {time.time() - start} seconds.")
    return title_to_qid, qid_to_all_titles, wpid_to_qid, qid_to_title


def aggregate_list_of_nested_dictionaries(list_of_nested_dicts):
    account = {}
    for dictionary in tqdm(list_of_nested_dicts):
        for k, dictionary2 in dictionary.items():
            if not k in account:
                account[k] = {}
            for kk, vv in dictionary2.items():
                if not kk in account[k]:
                    account[k][kk] = 0
                account[k][kk] += vv
    return account

def aggregate_list_of_dictionaries(list_of_dicts):
    account = {}
    for dictionary in tqdm(list_of_dicts):
        for k, v in dictionary.items():
            if not k in account:
                account[k] = 0
            account[k] += v
    return account

def normalize_count_nested_dict(nested_dict):
    res = {}
    for key1 in nested_dict:
        total_cnt = sum(nested_dict[key1].values())
        res[key1] = {key2: cnt/total_cnt for key2, cnt in nested_dict[key1].items()}
    return res

def glob_files(path):
    files = glob.glob(path)
    return list(filter(lambda x: not os.path.isdir(x), files))

def save_config(args, filename="config.json"):
    vars(args)["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if hasattr(args, "out_subdir"):
        utils.dump_json_file(filename=os.path.join(args.data_dir, args.out_subdir, filename), contents=vars(args))
    else:
        utils.dump_json_file(filename=os.path.join(args.data_dir, filename), contents=vars(args))

def get_outfname(in_filepath, ending="jsonl"):
    # Gets basename and removes jsonl
    out_fname = os.path.splitext(os.path.basename(in_filepath))[0]
    hash_v = hash(in_filepath)
    # Removes old + hash() parts of file
    out_fname_base = out_fname.rsplit("_", maxsplit=1)[0]
    return f"{out_fname_base}_{hash_v}.{ending}"

def get_outdir(save_dir, subfolder, remove_old=False):
    out_dir = os.path.join(save_dir, subfolder)
    if remove_old and os.path.exists(out_dir):
        print(f"Deleting {out_dir}...")
        shutil.rmtree(out_dir)
    print(f"Making {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def create_single_item_trie(vocab, out_file=""):
    keys = []
    values = []
    for k in vocab:
        assert type(vocab[k]) is int
        keys.append(k)
        # Tries require list of item for the record trie
        values.append(tuple([vocab[k]]))
    fmt = "<l"
    trie = marisa_trie.RecordTrie(fmt, zip(keys, values))
    if out_file != "":
        trie.save(out_file)
    return trie