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
import ujson as json
import multiprocessing
import logging
import os
import shutil
import time

import bootleg_data_prep.utils.utils as utils
from bootleg_data_prep.utils.constants import PAIR_IDX_MATTER, \
    POSSIBLE_SLICE_FOLDERS, TAILONLY, NOHEADCAND, NOSINGLE, NOCTX, TORSOONLY, HEADONLY, TOESONLY
import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.utils.classes.entity_symbols import EntitySymbols
import bootleg_data_prep.utils.slice_definitions as slice_utils
import bootleg_data_prep.utils.entity_symbols_for_signals as es_for_signals


def parse_args():
    parser = argparse.ArgumentParser()
    # Data and runs
    parser.add_argument('--topK', type=int, default=800, help="Number of most/least popular pairs to take.")
    parser.add_argument('--distP', type=int, default=10, help="Percentile of distances to quantify for close")
    parser.add_argument('--not_use_aug_stats', action='store_true', help="Use the augmented training data for popularity")
    parser.add_argument('--tail_threshold', default=10, type=int, help="Maximum number of times entity can be seen in training for the tail")
    parser.add_argument('--torso_threshold', default=1000, type=int, help="Maximum number of times entity can be seen in training for the torso")
    parser.add_argument('--emb_dir', type=str, default='embs', help='Where files saved')
    parser.add_argument('--data_dir', type=str, default='data/wiki_dump', help='Where files saved')
    parser.add_argument('--subfolder_name', type=str, default="filtered_data")
    parser.add_argument('--file_id_start', type=int, default=0, help="start file index to read of files for each folder; start idx 4 will start reading 4th file from dev and 4th from test")
    parser.add_argument('--file_id_end', type=int, default=-1, help="end file index to read (-1 for all) (exclusive)")
    parser.add_argument('--folders_to_slice', nargs='+', default=["train_prepped", "dev_prepped", "test_prepped"], help='List of folders to slice; subset of POSSIBLE_SLICE_FOLDERS.')
    parser.add_argument('--overwrite_saved_entities', action='store_true')
    parser.add_argument('--processes', type=int, default=int(35))
    parser.add_argument('--max_types', type=int, default=3)
    parser.add_argument('--max_types_rel', type=int, default=50)
    parser.add_argument('--max_relations', type=int, default=150) # 90th percentile was 138
    parser.add_argument('--kg_adj', type=str, default='kg_adj_1229.txt')
    parser.add_argument('--kg_triples', type=str, default='kg_triples_1229.txt')
    parser.add_argument('--hy_vocab', type=str, default='hyena_vocab.json')
    parser.add_argument('--hy_types', type=str, default='hyena_types_0905.json')
    parser.add_argument('--wd_vocab', type=str, default='wikidatatitles_to_typeid_1229.json')
    parser.add_argument('--wd_types', type=str, default='wikidata_types_1229.json')
    parser.add_argument('--rel_vocab', type=str, default='relation_to_typeid_1229.json')
    parser.add_argument('--rel_types', type=str, default='kg_relation_types_1229.json')
    args = parser.parse_args()
    return args

def normalize_prepped_folder(fold):
    return fold.replace("_prepped", "")

def print_memory():
    process = psutil.Process(os.getpid())
    print(f"{int(process.memory_info().rss) / 1024 ** 3} GB ({process.memory_percent()} %) memory used process {process}")

def init_process(args):
    entity_symbols_plus_f = args
    global entity_symbols_plus_global
    entity_symbols_plus_global = utils.load_pickle_file(entity_symbols_plus_f)


def launch_subprocess(args, temp_folder, entity_symbols_plus_f, test_stats_dict_f, all_files):
    print_memory()

    # Generate list of files and dictionary of slice folders for saving
    files = []
    all_test_folders = {}
    # key is dev, test, or train director

    for key in all_files:
        test_folder = prep_utils.get_outdir(key, f"folder_{args.file_id_start}_{args.file_id_end}", remove_old=True)
        all_test_folders[key] = test_folder
        for f in all_files[key]:
            files.append(tuple([f, test_folder]))
    all_process_args = [tuple([i + 1,
                               len(files),
                               args,
                               files[i],
                               test_stats_dict_f
                               ]) for i in range(len(files))]
    pool = multiprocessing.Pool(processes=args.processes, initializer=init_process, initargs=(entity_symbols_plus_f,))
    logging.info(f"Launching {args.processes} processes for {len(files)} jobs")
    st = time.time()
    res = pool.map(subprocess, all_process_args, chunksize=1)
    pool.close()
    total_remaining = defaultdict(int)
    all_slices = set()
    for r in res:
        all_slices.update(r["all_slices"])
        for key in r["filter_stats"]:
            total_remaining[key] += r["filter_stats"][key]
    logging.info(f"Finished slice generation in {time.time() - st}. Cleaning up {temp_folder}."
                 f"QIDs meeting filter criteria (only filtered if filter is set) {total_remaining}")
    return all_test_folders, total_remaining, all_slices


def subprocess(all_args):
    i, total, args, in_filepath_tupl, test_stats_dict_f = all_args
    in_filepath, test_folder = in_filepath_tupl
    test_stats_dict = utils.load_json_file(test_stats_dict_f)
    print_memory()
    print(f"Starting {i}/{total}. Reading in {in_filepath}.")

    out_fname = prep_utils.get_outfname(in_filepath)
    out_file = open(os.path.join(test_folder, out_fname), "w")

    remaining_tot = defaultdict(int)
    all_slices = set()
    with open(in_filepath, "r") as in_f:
        for line in in_f.readlines():
            sent_obj = json.loads(line.strip())
            if "gold" not in sent_obj:
                sent_obj["gold"] = sent_obj["anchor"]
                del sent_obj["anchor"]
            if len(sent_obj["spans"]) > 0 and type(sent_obj["spans"][0]) is str:
                sent_obj["spans"] = [list(map(int, s.split(":"))) for s in sent_obj["spans"]]
            slice_dict, total_remaining = generate_slice_dict(sent_obj, test_stats_dict, args.tail_threshold, args.torso_threshold)
            for key in total_remaining:
                remaining_tot[key] += total_remaining[key]
            # Remove unecessary fields
            del sent_obj["verb_unigrams"]
            del sent_obj["verb_unigrams_pos"]
            del sent_obj["verb_bigrams"]
            del sent_obj["verb_bigrams_pos"]
            del sent_obj["allowed_comma_pos"]
            del sent_obj["sentence_tokens"]
            del sent_obj["sentence_tokens_pos"]
            # Converts slice indexes into a probability over each alias
            slice_dict_to_write = make_slice_dict_probabilistic(slice_dict, len(sent_obj['aliases']))
            sent_obj["slices"] = slice_dict_to_write
            all_slices.update(set(slice_dict.keys()))
            write_sentence(out_file, sent_obj)
    out_file.close()
    return {"filter_stats": remaining_tot, "all_slices": all_slices}

def generate_slice_dict(sent_obj, test_stats_dict, tail_threshold, torso_threshold):
    slices = {}
    # This was mainly used when we remove False golds to map the slice indexes back into the original space; this is likely not needed anymore
    all_alias2pred_idx = list(range(len(sent_obj["aliases"])))
    # in case downstream
    gold_indexes = sent_obj["gold"]
    qids_to_measure = sent_obj['qids']
    aliases_to_measure = sent_obj['aliases']
    spans_to_measure = sent_obj["spans"]
    tokens = sent_obj["sentence_tokens"]

    # st = time.time()
    # This is used as a filter at the very end to remove head QIDs and to remove other QIDs that can be disambiguated via text
    non_head_cand_qids = [qid != entity_symbols_plus_global.get_qid_cands(al)[0] for al, qid in zip(aliases_to_measure, qids_to_measure)]
    non_single_cand_qids = [len(entity_symbols_plus_global.get_qid_cands(al)) > 1 for al in aliases_to_measure]
    body_part_qids = slice_utils.do_qids_meet_count(qids_to_measure, entity_symbols_plus_global, tail_threshold=tail_threshold, torso_threshold=torso_threshold)

    body_part_qids[NOSINGLE] = non_single_cand_qids
    body_part_qids[NOHEADCAND] = non_head_cand_qids

    total_remaining = {
        NOSINGLE: sum(non_single_cand_qids),
        NOHEADCAND: sum(non_head_cand_qids),
        TAILONLY: sum(body_part_qids[TAILONLY]),
        TORSOONLY: sum(body_part_qids[TORSOONLY]),
        HEADONLY: sum(body_part_qids[HEADONLY]),
        TOESONLY: sum(body_part_qids[TOESONLY])
    }
    assert len(gold_indexes) == len(aliases_to_measure)
    assert len(gold_indexes) == len(qids_to_measure)
    assert len(gold_indexes) == len(spans_to_measure)
    assert len(gold_indexes) == len(non_head_cand_qids)

    #**************************
    # SPECIFIC ALIAS-ENTITY PAIRS
    #**************************
    pairs = list(zip(aliases_to_measure, qids_to_measure))
    # st = time.time()
    for sl in PAIR_IDX_MATTER:
        # select alias_idx that are in the pairs
        alias2pred_idx = [all_alias2pred_idx[i] for i in range(len(pairs)) if pairs[i][0] in test_stats_dict[sl] and pairs[i][1] in test_stats_dict[sl][pairs[i][0]]]
        alias2pred_subslices = generate_body_slices(all_alias2pred_idx, body_part_qids, alias2pred_idx)
        slices = update_slices(slices, sl, alias2pred_subslices)

    # print(f"Time for pairs {time.time() - st}")
    # st = time.time()

    #**************************
    # UNIFORM SLICE
    #**************************
    alias2pred_subslices = generate_body_slices(all_alias2pred_idx, body_part_qids, list(range(len(all_alias2pred_idx))))
    slices = update_slices(slices, "unif", alias2pred_subslices)

    #**************************
    # NO TYPE AND KG
    #**************************
    aliases_to_predict_no_ty_no_kg = slice_utils.get_no_type_no_kg(qids_to_measure, entity_symbols_plus_global)
    alias2pred_subslices = generate_body_slices(all_alias2pred_idx, body_part_qids, aliases_to_predict_no_ty_no_kg)
    slices = update_slices(slices, "noTnoR", alias2pred_subslices)

    #**************************
    # CONSISTENCY
    #**************************
    # This returns an dictionary of aliases to use for weak and strong consistency
    min_common_types = [1, entity_symbols_plus_global.max_types]
    filter_types = set()
    alias_to_predict_type = slice_utils.exists_shared_true_type(qids_to_measure, aliases_to_measure, min_common_types, filter_types, entity_symbols_plus_global)
    alias2pred_subslices_strong = generate_body_slices(all_alias2pred_idx, body_part_qids,
                                                                alias_to_predict_type[entity_symbols_plus_global.max_types])
    alias2pred_subslices_weak = generate_body_slices(all_alias2pred_idx, body_part_qids, alias_to_predict_type[1])
    slices = update_slices(slices, "3_cons", alias2pred_subslices_strong)
    slices = update_slices(slices, "1_cons", alias2pred_subslices_weak)
    # print(f"Time for types_alias_same {time.time() - st}")
    # st = time.time()

    #**************************
    # WIKIDATA AFFORDANCE
    #**************************
    # This returns an dictionary of aliases to use for weak and strong affordance
    min_common_types = [1, entity_symbols_plus_global.max_types]
    filter_types = set()
    check_type = lambda esp, q: esp.qid_in_qid2typeid_wd(q)
    get_types = lambda esp, q: esp.get_types_wd(q)
    affordance_types = entity_symbols_plus_global.affordance_types
    if affordance_types is None:
        affordance_types = {}
    alias_to_predict_type = slice_utils.get_affordance_signal(qids_to_measure, min_common_types, tokens, filter_types,
                                                              affordance_types, check_type, get_types, entity_symbols_plus_global)
    alias2pred_subslices_strong = generate_body_slices(all_alias2pred_idx, body_part_qids,
                                                                alias_to_predict_type[entity_symbols_plus_global.max_types])
    alias2pred_subslices_weak = generate_body_slices(all_alias2pred_idx, body_part_qids, alias_to_predict_type[1])
    slices = update_slices(slices, "3_wd_aff", alias2pred_subslices_strong)
    slices = update_slices(slices, "1_wd_aff", alias2pred_subslices_weak)
    # print(f"Time for types_alias_same {time.time() - st}")
    # st = time.time()

    #**************************
    # RELATIONAL AFFORDANCE
    #**************************
    # This returns an dictionary of aliases to use for weak and strong affordance
    min_common_types = [1]
    filter_types = set()
    check_type = lambda esp, q: esp.qid_in_qid2typeid_rel(q)
    get_types = lambda esp, q: esp.get_types_rel(q)
    affordance_rel_types = entity_symbols_plus_global.affordance_rel_types
    if affordance_rel_types is None:
        affordance_rel_types = {}
    alias_to_predict_type = slice_utils.get_affordance_signal(qids_to_measure, min_common_types, tokens, filter_types,
                                                              affordance_rel_types, check_type, get_types, entity_symbols_plus_global)
    alias2pred_subslices_weak = generate_body_slices(all_alias2pred_idx, body_part_qids, alias_to_predict_type[1])
    slices = update_slices(slices, "1_rel_aff", alias2pred_subslices_weak)
    # print(f"Time for types_alias_same {time.time() - st}")
    # st = time.time()

    #**************************
    # KG SIGNAL EXISTS
    #**************************
    aliases_to_predict_kg = slice_utils.get_kg_signal(qids_to_measure, aliases_to_measure, entity_symbols_plus_global)
    alias2pred_subslices = generate_body_slices(all_alias2pred_idx, body_part_qids, aliases_to_predict_kg)
    slices = update_slices(slices, "kg_rel", alias2pred_subslices)
    return slices, total_remaining

# Generate 7 subslices for the alias_idx: ALL, NO_HEAD+TORSO, NO_HEAD+TAIL, NO_HEAD+TOES, NO_HEAD+CTX+TORSO, NO_HEAD+CTX+TAIL, NO_HEAD+CTX+TOES
def generate_body_slices(all_alias2pred_idx, body_part_qids, alias_idx):
    res_subslices = {
        "all": [], f"{HEADONLY}": [], f"{TORSOONLY}": [], f"{TAILONLY}": [], f"{TOESONLY}": [],
        f"{NOSINGLE}_all": [], f"{NOSINGLE}_{HEADONLY}": [], f"{NOSINGLE}_{TORSOONLY}": [], f"{NOSINGLE}_{TAILONLY}": [], f"{NOSINGLE}_{TOESONLY}": [],
        f"{NOHEADCAND}_all": [], f"{NOHEADCAND}_{HEADONLY}": [], f"{NOHEADCAND}_{TORSOONLY}": [], f"{NOHEADCAND}_{TAILONLY}": [], f"{NOHEADCAND}_{TOESONLY}": [],
        }
    for i in alias_idx:
        res_subslices["all"].append(all_alias2pred_idx[i])
        if body_part_qids[NOSINGLE][i] is True:
            res_subslices[f"{NOSINGLE}_all"].append(all_alias2pred_idx[i])
        if body_part_qids[NOHEADCAND][i] is True:
            res_subslices[f"{NOHEADCAND}_all"].append(all_alias2pred_idx[i])
        if body_part_qids[TOESONLY][i] is True:
            res_subslices[f"{TOESONLY}"].append(all_alias2pred_idx[i])
            if body_part_qids[NOSINGLE][i] is True:
                res_subslices[f"{NOSINGLE}_{TOESONLY}"].append(all_alias2pred_idx[i])
            if body_part_qids[NOHEADCAND][i] is True:
                res_subslices[f"{NOHEADCAND}_{TOESONLY}"].append(all_alias2pred_idx[i])
        # NOT MUTUALLY EXCLUSIVE WITH TOES
        if body_part_qids[TAILONLY][i] is True:
            res_subslices[f"{TAILONLY}"].append(all_alias2pred_idx[i])
            if body_part_qids[NOSINGLE][i] is True:
                res_subslices[f"{NOSINGLE}_{TAILONLY}"].append(all_alias2pred_idx[i])
            if body_part_qids[NOHEADCAND][i] is True:
                res_subslices[f"{NOHEADCAND}_{TAILONLY}"].append(all_alias2pred_idx[i])
        if body_part_qids[TORSOONLY][i] is True:
            res_subslices[f"{TORSOONLY}"].append(all_alias2pred_idx[i])
            if body_part_qids[NOSINGLE][i] is True:
                res_subslices[f"{NOSINGLE}_{TORSOONLY}"].append(all_alias2pred_idx[i])
            if body_part_qids[NOHEADCAND][i] is True:
                res_subslices[f"{NOHEADCAND}_{TORSOONLY}"].append(all_alias2pred_idx[i])
        if body_part_qids[HEADONLY][i] is True:
            res_subslices[f"{HEADONLY}"].append(all_alias2pred_idx[i])
            if body_part_qids[NOSINGLE][i] is True:
                res_subslices[f"{NOSINGLE}_{HEADONLY}"].append(all_alias2pred_idx[i])
            if body_part_qids[NOHEADCAND][i] is True:
                res_subslices[f"{NOHEADCAND}_{HEADONLY}"].append(all_alias2pred_idx[i])
    return res_subslices

def update_slices(slices, key, res_subslices):
    for k in res_subslices:
        subsl = res_subslices[k]
        if len(subsl) > 0:
            slices[f"{key}_{k}"] = subsl
    return slices

def make_slice_dict_probabilistic(slice_dict, num_aliases):
    new_slice_dict = {}
    for slice_name in slice_dict:
        new_slice_dict[slice_name] = {}
        for i in range(num_aliases):
            if i in slice_dict[slice_name]:
                new_slice_dict[slice_name][str(i)] = 1.0
            else:
                new_slice_dict[slice_name][str(i)] = 0.0
    return new_slice_dict

def write_sentence(out_f, sent_obj):
    out_f.write(json.dumps(sent_obj) + "\n")

def combine_data(out_file, folder, clean_up=True):
    logging.info(f"Combining files from {folder}")
    files = list(sorted(prep_utils.glob_files(f"{folder}/*")))
    if len(files) == 0:
        return
    with open(out_file, "w") as out_f:
        logging.info(f"Opening data file for writing {out_file}")
        for f in files:
            logging.info(f"Reading in file {f}")
            with open(f, 'r') as fd:
                for line in fd:
                    sent_obj = json.loads(line.strip())
                    if len(sent_obj['aliases']) > 0:
                        out_f.write(json.dumps(sent_obj) + "\n")
                # shutil.copyfileobj(fd, out_f)
    if clean_up:
        logging.info(f"Removing directory {folder}")
        shutil.rmtree(folder)
    return


def main():
    multiprocessing.set_start_method("forkserver", force=True)
    gl_start = time.time()
    args = parse_args()

    # Check folders
    for fold in args.folders_to_slice:
        assert fold in POSSIBLE_SLICE_FOLDERS, f"Folder {fold} not in {POSSIBLE_SLICE_FOLDERS}"

    # Set up logging
    formatter = logging.Formatter('%(asctime)s %(message)s')
    log_file = os.path.join(args.data_dir, f"{args.subfolder_name}_logs", f"log_{args.file_id_start}_{args.file_id_end}")
    utils.ensure_dir(os.path.join(args.data_dir, f"{args.subfolder_name}_logs"))
    open(log_file, 'a').close()
    # Create file
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logging.getLogger().addHandler(sh)
    logging.getLogger().setLevel(logging.INFO)
    print(f"Logging info to {log_file}")
    logging.info(json.dumps(vars(args), indent=4))

    # Get load data subfolder
    load_dir = os.path.join(args.data_dir, f"{args.subfolder_name}")
    logging.info(f"Loading data from {load_dir}...")

    # Split up files in the case of file ranges
    all_files = defaultdict(list)
    all_files_list = []
    file_id_start = args.file_id_start
    file_id_end = args.file_id_end
    # Read in all test, dev, and train files
    # This is over prepped data
    for fold in args.folders_to_slice:
        key = os.path.join(load_dir, fold)
        all_files_list.extend([[key, int(os.path.splitext(f)[0].split("out_")[1]), f] for f in prep_utils.glob_files(f"{key}/out_*.jsonl")])
    all_files_list = list(sorted(all_files_list, key=lambda x: (x[0], x[1])))
    if file_id_end == -1:
        file_id_end = len(all_files_list)
    # Subselect files
    files_to_run = all_files_list[file_id_start:file_id_end]
    for f in files_to_run:
        all_files[f[0]].append(f[2])

    logging.info(f"Loaded {sum(len(files) for files in all_files.values())} files.")
    # Remove old out directory if exists
    out_dir = prep_utils.get_outdir(args.data_dir, f"{args.subfolder_name}_{args.file_id_start}_{args.file_id_end}_final", remove_old=True)
    vars(args)['saved_data_folder'] = out_dir
    prep_utils.save_config(args, f"{args.subfolder_name}_gen_slices_config.json")

    # load entity symbols
    logging.info("=" * 10)
    logging.info("Loading entity symbols...")
    start = time.time()
    entity_symbols = EntitySymbols(load_dir=os.path.join(load_dir, "entity_db/entity_mappings"))
    logging.info(
        f"Loaded entity symbols with {entity_symbols.num_entities} entities and {len(entity_symbols.get_all_aliases())} aliases. {time.time() - start} seconds.")

    # Generating temp folders and dump files
    # We don't want to remove old ones so we do not use the prep_utils command
    temp_folder = prep_utils.get_outdir(args.data_dir, f"{args.subfolder_name}_temp_dump_{args.file_id_start}_{args.file_id_end}")
    logging.info(f"Will save entity_symbol files in {temp_folder}")
    # This will store our entity dump and other auxiliary pieces of information
    entity_symbols_plus_f = os.path.join(temp_folder, "entity_symbols_plus.pkl")
    # This will store slice specific metrics and alias-entity pairs for sampling
    test_stats_dict_f = os.path.join(temp_folder, "test_stats_dict.pkl")

    if not os.path.exists(entity_symbols_plus_f) or not os.path.exists(test_stats_dict_f) or args.overwrite_saved_entities:
        logging.info(f"Loading types and relations.")
        entity_symbols_plus = es_for_signals.create_entity_symbols_for_slices(args, entity_symbols)
        utils.dump_pickle_file(entity_symbols_plus_f, entity_symbols_plus)
        # Technically we load these up for create_entity_symbols, but as we reuse some of these functions in other code, we don't want to require
        # this being passed around
        # Get stats file for alias-entity counts
        alias_qid = es_for_signals.load_alias_qid(args)
        test_stats_dict = slice_utils.get_test_stats_dict(args, alias_qid, entity_symbols_plus)
        utils.dump_json_file(test_stats_dict_f, test_stats_dict)
    start = time.time()
    all_test_folders, total_remaining, all_slices = launch_subprocess(args, temp_folder, entity_symbols_plus_f, test_stats_dict_f, all_files)
    logging.info(f"Done with filtering in {time.time() - start} seconds.")
    # Add counts to config and redump; if something fails we have dump from before
    for key in total_remaining:
        vars(args)[key] = total_remaining[key]
    prep_utils.save_config(args, f"{args.subfolder_name}_gen_slices_config.json")
    # Cleaning up
    # shutil.rmtree(temp_folder)

    # Combine the output files; key will be folder
    for key in all_test_folders:
        save_file_str = os.path.join(out_dir, f"{normalize_prepped_folder(os.path.basename(key))}.jsonl")
        combine_data(save_file_str, all_test_folders[key], clean_up=True)
    # Save non-sliced data if any
    for fold in POSSIBLE_SLICE_FOLDERS:
        if fold in args.folders_to_slice:
            continue
        save_file_str = os.path.join(out_dir, f"{normalize_prepped_folder(fold)}.jsonl")
        try:
            combine_data(save_file_str, os.path.join(load_dir, fold), clean_up=False)
        except Exception as e:
            logging.info(f"Error {e} when trying to combine {fold}.")
    # Dump entities in the final folder
    entity_symbols.dump(os.path.join(out_dir, "entity_db/entity_mappings"))
    logging.info(f"Final Slices Are: {all_slices}")
    logging.info(f"Finished generate_slices in {time.time() - gl_start} seconds.")


if __name__ == '__main__':
    main()
