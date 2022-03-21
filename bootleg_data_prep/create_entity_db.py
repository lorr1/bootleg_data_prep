'''
This file
1. Reads in statistics from <data_dir>/<subfolder_name>_stats. Reads in data from <data_dir>/<subfolder_name>/<dev/test>
2. Uses statistics to generate alias, entity pairs for slice generation (see utils.slice_definitions.get_test_stats_dict)
3. Generates slice chunks from the dev/test data chunks
4. Merges together the slices
5. Outputs a final train.jsonl, test.jsonl, dev.jsonl in <data_dir>/<subfolder_name>_final

After this, you are done

Example run command:
python3 -m bootleg_data_prep.generate_slices --subfolder_name korealiases --processes 30 --kg_adj kg_adj.txt
'''

import argparse
from collections import defaultdict

import psutil
import json
import multiprocessing
import logging
import os
import shutil
import time

from rich.progress import track
from tqdm import tqdm

import bootleg_data_prep.utils.utils as utils
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.entity_profile import EntityProfile
from bootleg.symbols.type_symbols import TypeSymbols
from bootleg.symbols.kg_symbols import KGSymbols

# Properties that are used for types
# P31 = instance of
# P279 = subclass of
KG_PROPERTIES_TO_REMOVE = {"P31", "P279"}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/wiki_dump', help='Where files saved')
    parser.add_argument('--subfolder_name', type=str, default="filtered_data")
    parser.add_argument('--max_types', type=int, default=3)
    parser.add_argument('--max_relations', type=int, default=50)
    parser.add_argument('--ent_desc', type=str, default="qid2desc.json")
    parser.add_argument('--kg_triples', type=str, default='kg_triples.json')
    parser.add_argument('--kg_vocab', type=str, default='utils/param_files/pid_names_en.json')
    parser.add_argument('--wd_vocab', type=str, default='wikidatatitle_to_typeid.json')
    parser.add_argument('--wd_types', type=str, default='wikidata_types.json')
    args = parser.parse_args()
    return args

def create_entity_profile(args, entity_symbols):
    # Load entity desctiption and add to entity symbols
    print("Writing out descriptions")
    qid2desc_path = args.ent_desc
    if os.path.exists(qid2desc_path):
        print("Found qid2desc.json. Creating entity symbols")
        qid2desc = json.load(open(qid2desc_path))
        entity_symbols = EntitySymbols(
            alias2qids=entity_symbols.get_alias2qids_dict(),
            qid2title=entity_symbols.get_qid2title_dict(),
            qid2desc=qid2desc,
            max_candidates=entity_symbols.max_candidates
        )

    # KG relations
    kg_list = []
    kg_vocab = json.load(open(args.kg_vocab))
    all_relations = json.load(open(args.kg_triples))
    qid2relations = {}
    for head_qid in track(entity_symbols.get_all_qids(), total=len(entity_symbols.get_all_qids()), description="Filt qid2rels"):
        qid2relations[head_qid] = {}
        for rel in all_relations.get(head_qid, {}):
            if rel not in KG_PROPERTIES_TO_REMOVE:
                rel_name = kg_vocab.get(rel, rel)
                qid2relations[head_qid][rel_name] = all_relations[head_qid][rel][:args.max_relations]
    kg_symbols = KGSymbols(
        qid2relations=qid2relations,
        max_connections=args.max_relations
    )
    # Types
    qid2types = {q: [] for q in entity_symbols.get_all_qids()}
    vocab_map = json.load(open(args.wd_vocab))
    vocab_inv = {v: k for k, v in vocab_map.items()}
    with open(args.wd_types) as in_f:
        type_mappings = json.load(in_f)
        for qid in track(type_mappings, total=len(type_mappings), description="Filt types"):
            # Already has all qids inside
            if qid in qid2types:
                qid2types[qid] = [vocab_inv[i] for i in type_mappings[qid]][:args.max_types]
    type_symbols = TypeSymbols(qid2typenames=qid2types, max_types=args.max_types)
    entity_profile = EntityProfile(
        entity_symbols=entity_symbols,
        type_systems={"wiki": type_symbols},
        kg_symbols=kg_symbols
    )
    return entity_profile

def main():
    multiprocessing.set_start_method("forkserver", force=True)
    gl_start = time.time()
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    # Get load data subfolder
    load_dir = os.path.join(args.data_dir, f"{args.subfolder_name}")
    print(f"Loading data from {load_dir}...")

    # Load entity symbols
    print("Loading entity symbols...")
    start = time.time()
    entity_symbols = EntitySymbols.load_from_cache(load_dir=os.path.join(load_dir, "entity_db/entity_mappings"))
    print(
        f"Loaded entity symbols with {entity_symbols.num_entities} entities and {len(entity_symbols.get_all_aliases())} aliases. {time.time() - start} seconds.")

    entity_profile = create_entity_profile(args, entity_symbols)
    print(f"Saving entity profile to", os.path.join(load_dir, "entity_db_final"))
    entity_profile.save(os.path.join(load_dir, "entity_db_final"))

    print(f"Finished generate_slices in {time.time() - gl_start} seconds.")


if __name__ == '__main__':
    main()