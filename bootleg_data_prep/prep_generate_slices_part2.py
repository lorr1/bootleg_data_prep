'''
First goes and maps the sentence tokens and bigrams/unigrams to their vocab indexes (computed in part 1). It then uses these to create a map
of the most popular words for each entity and for each wikidata fine type. These are used in slice generation for affordance slices and contextual filters.

python3 -m contextual_embeddings.bootleg_data_prep.prep_generate_slices_part2 --subfolder_name korealiases --processes 30

(May need to add --overwrite_saved_entities flag if you want to overwrite existing dumps)
'''


import argparse
import glob
import logging
import shutil

from sklearn.feature_extraction import text
import numpy as np
import jsonlines
import marisa_trie
import ujson as json
import multiprocessing
import os
import time

from collections import defaultdict
from nltk import FreqDist
from tqdm import tqdm

import bootleg_data_prep.utils.utils as utils
import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.utils.constants import TYPEWORDS, RELATIONWORDS, VOCABFILE, VOCAB
from bootleg_data_prep.utils.classes.entity_symbols import EntitySymbols
import bootleg_data_prep.utils.entity_symbols_for_signals as esfs

QIDEXISTS = "qidexist"
TYPEEXIST = "typeexist"
RELATIONEXIST = "relationexist"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/wiki_dump', help='Directory for data to be saved.')
    parser.add_argument('--subfolder_name', type=str, default='orig_wl_prn', help = 'Subdirectory to save filtered sentences.')
    parser.add_argument('--emb_dir', type=str, default='embs', help='Where files saved')
    parser.add_argument('--processes', type=int, default=int(0.8*multiprocessing.cpu_count()))
    parser.add_argument('--top_k_entity_words', type=int, default=50, help="Take top K most common words.")
    parser.add_argument('--top_k_affordance_words', type=int, default=15, help="Take top K most common words.")
    parser.add_argument('--not_use_aug_stats', action='store_true', help="Not use the augmented training data for popularity")
    parser.add_argument('--overwrite_saved_entities', action='store_true')
    parser.add_argument('--max_types', type=int, default=3)
    parser.add_argument('--max_types_rel', type=int, default=50)
    parser.add_argument('--max_relations', type=int, default=150) # 90th percentile was 138
    parser.add_argument('--kg_adj', type=str, default='kg_adj_1229.txt')
    parser.add_argument('--kg_triples', type=str, default='kg_triples_1229.txt')
    parser.add_argument('--hy_vocab', type=str, default='hyena_vocab.json')
    parser.add_argument('--hy_types', type=str, default='hyena_types_0905.json')
    parser.add_argument('--wd_vocab', type=str, default='wikidatatitle_to_typeid_1229.json')
    parser.add_argument('--wd_types', type=str, default='wikidata_types_1229.json')
    parser.add_argument('--rel_vocab', type=str, default='relation_to_typeid_1229.json')
    parser.add_argument('--rel_types', type=str, default='kg_relation_types_1229.json')
    parser.add_argument('--multilingual', action = 'store_true', help = 'If set, will not do english-based language processing (e.g. stemming).')
    parser.add_argument('--test', action = 'store_true', help = 'If set, will only generate for one file.')

    args = parser.parse_args()
    return args


def create_entity_symbols_for_slices(args, entity_symbols):
    entity_dump_dir = os.path.join(args.data_dir, f"{args.subfolder_name}", "entity_db/entity_mappings")
    tri_collection, qid2typeid_hy, qid2typeid_wd, qid2typeid_rel, rel_mapping, rel_tri, rel_vocab_inv = esfs.load_tri_collection(args, entity_symbols)
    affordance_types = esfs.init_type_words()
    affordance_rel_types = esfs.init_type_words()
    # We could load vocab if we wanted, but we don't need it for anything
    vocab = None
    qid_count = esfs.load_qid_counts(args, entity_dump_dir)
    entity_dump_for_slices = esfs.EntitySymbolsForSlice(
        entity_dump_dir=entity_dump_dir,
        entity_symbols=entity_symbols,
        qid_count=qid_count,
        tri_collection=tri_collection,
        max_types=args.max_types,
        max_types_rel=args.max_types_rel,
        max_relations=args.max_relations,
        max_candidates=entity_symbols.max_candidates,
        qid2typeid_hy=qid2typeid_hy,
        qid2typeid_wd=qid2typeid_wd,
        qid2typeid_rel=qid2typeid_rel,
        rel_mapping=rel_mapping,
        contextual_rel=rel_tri,
        contextual_rel_vocab_inv=rel_vocab_inv,
        affordance_types=affordance_types,
        affordance_rel_types=affordance_rel_types,
        vocab=vocab)
    logging.info(f"Finished building entity symbols")
    return entity_dump_for_slices

def init_process_entity(args):
    entity_symbols_plus_f = args
    global entity_symbols_plus_global
    entity_symbols_plus_global = utils.load_pickle_file(entity_symbols_plus_f)

def init_process_vocab(args):
    word_vocab_f = args
    global word_vocab_global
    word_vocab_global = marisa_trie.Trie().mmap(word_vocab_f)

def init_process_tfidf(args):
    word_vocab_f = args
    global word_vocab_global
    global word_vocab_inv_global
    word_vocab_global = marisa_trie.Trie().mmap(word_vocab_f)
    word_vocab_inv_global = {}
    for k in tqdm(word_vocab_global):
        word_vocab_inv_global[word_vocab_global[k]] = k

def launch_subprocess_translate_to_ints(args, all_files, word_vocab_f):
    files = []
    # Save for cleaning later
    out_folders = {}
    # key is dev, test, or train director
    for key in all_files:
        path = os.path.dirname(key)
        save_folder = prep_utils.get_outdir(path, f"{os.path.basename(key)}_prepped", remove_old=True)
        out_folders[key] = save_folder
        for f in all_files[key]:
            files.append(tuple([f, save_folder]))

    all_process_args = [tuple([i + 1,
                               len(files),
                               args,
                               files[i]
                               ]) for i in range(len(files))]
    pool = multiprocessing.Pool(processes=args.processes, initializer=init_process_vocab, initargs=(tuple([word_vocab_f])))
    logging.info("****************************MAPPING SENTENCES TO WORD IDXS****************************")
    logging.info("Starting pool...")
    st = time.time()
    pool.map(subprocess_translate_to_ints, all_process_args, chunksize=1)
    logging.info(f"Finished in {time.time()-st}s")
    pool.close()
    return out_folders


def subprocess_translate_to_ints(all_args):
    start_time = time.time()
    idx, total, args, file_tup = all_args
    in_filepath, out_folder = file_tup
    out_filepath = os.path.join(out_folder, os.path.basename(in_filepath))
    print(f"Starting {idx}/{total}. Reading in {in_filepath}. Ouputting to {out_filepath}")
    with jsonlines.open(in_filepath, 'r') as in_file, jsonlines.open(out_filepath, "w") as out_file:
        for sent_idx, sent_obj in enumerate(in_file):
            tokens, tokens_pos, verb_unigrams, verb_unigrams_pos, verb_bigrams, verb_bigrams_pos = prep_utils.clean_sentence_to_tokens(sent_obj["sentence"], args.multilingual, skip_verbs=True)
            allowed_comma_pos = prep_utils.extract_allowed_comma_positions(sent_obj["sentence"])
            # print(sent_obj["sentence"])
            # print(tokens)
            # print(tokens_pos)
            # print(verb_unigrams)
            # print(verb_unigrams_pos)
            # print(verb_bigrams)
            # print(verb_bigrams_pos)
            # A Trie stores a list of tuples; they are length one in our case as it's a unique id
            tokens_trans = [word_vocab_global[t] for t in tokens]
            verb_unigrams_trans = [word_vocab_global[t] for t in verb_unigrams]
            verb_bigrams_trans = [word_vocab_global[t] for t in verb_bigrams]
            sent_obj["sentence_tokens"] = tokens_trans
            sent_obj["sentence_tokens_pos"] = tokens_pos
            sent_obj["verb_unigrams"] = verb_unigrams_trans
            sent_obj["verb_unigrams_pos"] = verb_unigrams_pos
            sent_obj["verb_bigrams"] = verb_bigrams_trans
            sent_obj["verb_bigrams_pos"] = verb_bigrams_pos
            sent_obj["allowed_comma_pos"] = allowed_comma_pos
            out_file.write(sent_obj)
    print(f"Finished {idx}/{total}. Written to {out_filepath}. {time.time() - start_time} seconds.")
    return None


def launch_subprocess_get_word_bags(args, temp_out_dir, entity_symbols_plus_f, in_files):
    all_process_args = [tuple([i+1,
                               len(in_files),
                               temp_out_dir,
                               args,
                               in_files[i],
                               ]) for i in range(len(in_files))]
    pool = multiprocessing.Pool(processes=args.processes, initializer=init_process_entity, initargs=(tuple([entity_symbols_plus_f])))
    logging.info("****************************COLLECTING WORD BAGS****************************")
    logging.info("Starting pool...")
    pool.map(subprocess_get_word_bags, all_process_args, chunksize=1)
    pool.close()
    return

def subprocess_get_word_bags(all_args):
    start_time = time.time()
    idx, total, temp_outdir, args, in_filepath = all_args
    # create output files:
    out_fname_types = os.path.join(temp_outdir, os.path.splitext(os.path.basename(in_filepath))[0] + f"_{TYPEWORDS}.json")
    out_fname_relations = os.path.join(temp_outdir, os.path.splitext(os.path.basename(in_filepath))[0] + f"_{RELATIONWORDS}.json")
    out_fname_qid_exist = os.path.join(temp_outdir, os.path.splitext(os.path.basename(in_filepath))[0] + f"_{QIDEXISTS}.json")
    out_fname_type_exist = os.path.join(temp_outdir, os.path.splitext(os.path.basename(in_filepath))[0] + f"_{TYPEEXIST}.json")
    out_fname_relations_exist = os.path.join(temp_outdir, os.path.splitext(os.path.basename(in_filepath))[0] + f"_{RELATIONEXIST}.json")
    print(f"Starting {idx}/{total}. Reading in {in_filepath}. Ouputting to {out_fname_types}")
    type_words = defaultdict(list)
    rel_words = defaultdict(list)
    qid_counts = FreqDist()
    type_counts = FreqDist()
    rel_counts = FreqDist()
    with jsonlines.open(in_filepath, 'r') as in_file:
        for sent_idx, sent_obj in enumerate(in_file):
            tokens = sent_obj["sentence_tokens"]
            qid_spans = sent_obj["spans"]
            if type(qid_spans[0]) is str:
                qid_spans = [list(map(int, s.split(":"))) for s in qid_spans]
            assert len(qid_spans[0]) == 2, f"Bad Spans {sent_obj}"
            for qid, spans in zip(sent_obj["qids"], qid_spans):
                qid_counts[qid] += 1
                if entity_symbols_plus_global.qid_in_qid2typeid_wd(qid):
                    types = entity_symbols_plus_global.get_types_wd(qid)
                    # filtered_unigrams = prep_utils.filter_seq_by_span(verb_unigrams, verb_unigrams_pos, spans, threshold=5)
                    # filtered_bigrams = prep_utils.filter_seq_by_span(verb_bigrams, verb_bigrams_pos, spans, threshold=5)
                    # Update types individually
                    for typ in types:
                        type_counts[typ] += 1
                        type_words[typ].append(tokens)
                        # type_words[typ].update(filtered_unigrams)
                        # type_words[typ].update(filtered_bigrams)
                    # Update types globally
                    # type_words[tuple(sorted(types))].update(filtered_unigrams)
                    # type_words[tuple(sorted(types))].update(filtered_bigrams)
                    type_words[tuple(sorted(types))].append(tokens)
                    type_counts[tuple(sorted(types))] += 1
                if entity_symbols_plus_global.qid_in_qid2typeid_rel(qid):
                    types = entity_symbols_plus_global.get_types_rel(qid)
                    # filtered_unigrams = prep_utils.filter_seq_by_span(verb_unigrams, verb_unigrams_pos, spans, threshold=5)
                    # filtered_bigrams = prep_utils.filter_seq_by_span(verb_bigrams, verb_bigrams_pos, spans, threshold=5)
                    # Update types individually
                    for typ in types:
                        rel_counts[typ] += 1
                        rel_words[typ].append(tokens)
                        # type_words[typ].update(filtered_unigrams)
                        # type_words[typ].update(filtered_bigrams)
                    # Update types globally
                    # type_words[tuple(sorted(types))].update(filtered_unigrams)
                    # type_words[tuple(sorted(types))].update(filtered_bigrams)
                    # rel_words[tuple(sorted(types))].append(tokens)
                    # rel_counts[tuple(sorted(types))] += 1
    # Pre aggregate here to top 10*top_k (reduces error of aggregation)
    # agg_entity_words = {}
    # for key in entity_words:
    #     agg_entity_words[key] = FreqDist(dict(entity_words[key].most_common(args.top_k_entity_words*10)))
    # agg_type_words = {}
    # for key in type_words:
    #     agg_type_words[key] = FreqDist(dict(type_words[key].most_common(args.top_k_affordance_words*10)))
    # agg_rel_words = {}
    # for key in rel_words:
    #     agg_rel_words[key] = FreqDist(dict(rel_words[key].most_common(args.top_k_affordance_words*10)))
    # utils.dump_json_file(out_fname_bag, agg_entity_words)
    # utils.dump_json_file(out_fname_types, agg_type_words)
    # utils.dump_json_file(out_fname_relations, agg_rel_words)
    for k in list(rel_words.keys()):
        rel_words[k] = rel_words[k][:500]
    utils.dump_json_file(out_fname_types, type_words)
    utils.dump_json_file(out_fname_relations, rel_words)
    utils.dump_json_file(out_fname_qid_exist, qid_counts)
    utils.dump_json_file(out_fname_type_exist, type_counts)
    utils.dump_json_file(out_fname_relations_exist, rel_counts)

    print(f"Finished {idx}/{total}. Written to {out_fname_types} and {out_fname_relations}. {time.time() - start_time} seconds."
          f"Found a total of {len(type_words)} types and {len(rel_words)} relations.")
    return None


def merge_and_gen_vocab(folders, out_file):
    vocabulary = set()
    # Due to conflicts, we may get different words assigned to the same value. So, we reassing indexes here.
    # This MUST be deterministic so we can handle conflicts of word ids
    keys = sorted(list(folders))
    for folder_to_read in keys:
        assert type(folder_to_read) is str
        files = sorted(list(prep_utils.glob_files(f"{folder_to_read}/*.json")))
        print(f"Reading in files from {folder_to_read}: {files}")
        for file in tqdm(files):
            # Although we dumped a set, json loads a list
            word_dict = utils.load_json_file(file)
            assert type(word_dict) is list, f"Word list is not a dict type, it has type {type(word_dict)}"
            for word in word_dict:
                if word not in vocabulary:
                    vocabulary.add(word)
    logging.info(f"Creating vocab with len {len(vocabulary)}")
    prep_utils.create_trie(vocabulary, out_file=out_file)
    return


def aggregate_count_dict(list_of_dicts):
    """Merges list of count dictionaries that are key->count"""
    agg_dict = defaultdict(int)
    for dct in tqdm(list_of_dicts, desc="Aggregating counts"):
        for k in dct:
            agg_dict[k] += dct[k]
    return agg_dict


def aggregate_list_of_tokens(list_of_dicts):
    """Merges list of list of token dictionaries that are key->list of tokens"""
    agg_dict = defaultdict(list)
    for dct in tqdm(list_of_dicts, desc="Aggregating lists"):
        for k in dct:
            for l in dct[k]:
                agg_dict[k].append(l)
    return agg_dict


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def compute_tfidf(dct, word2id_f, top_words, temp_dir, processes, max_words_item=-1):
    tfidf_by_key = {}
    sub_dcts = []
    for key in dct:
        sub_d = {}
        if max_words_item > 0:
            sub_d[key] = dct[key][:max_words_item]
        else:
            sub_d[key] = dct[key]
        sub_dcts.append(sub_d)
    all_process_args = [[i, sub_d, top_words, temp_dir] for i, sub_d in enumerate(sub_dcts)]
    pool = multiprocessing.Pool(processes=processes, initializer=init_process_tfidf, initargs=tuple([word2id_f]))
    logging.info("****************************GENERATING TFIDF MATRICES****************************")
    logging.info("Starting pool...")
    st = time.time()
    for res in tqdm(pool.imap_unordered(compute_tfidf_process, all_process_args, chunksize=1), desc="Itearting over tfidf keys", total=len(all_process_args)):
        for k in res:
            assert k not in tfidf_by_key
            tfidf_by_key[k] = res[k]
    '''
    for file in glob(os.path.join(temp_dir, "tfidf*.json")):
        res = utils.load_json_file(file)
        for k in res:
            assert k not in tfidf_by_key
            tfidf_by_key[k] = res[k]
    '''
    logging.info(f"Finished in {time.time()-st}s")
    pool.close()
    return tfidf_by_key


def compute_tfidf_process(args):
    """Compute TFIDF for each key in the dct. Dct is key->list of list of tokens"""
    i, dct, top_words, out_dir = args
    other_stop = ["also", "first", "second"]
    tfidf_by_key = defaultdict(list)
    for k in dct:
        sentences = [" ".join([word_vocab_inv_global[idx] for idx in l]) for l in dct[k]]
        tfid_vectorizor = text.TfidfVectorizer()
        try:
            X = tfid_vectorizor.fit_transform(sentences)
            words = tfid_vectorizor.get_feature_names()
            vals = np.array(np.mean(X, axis=0).tolist()).flatten()
            # sorted_vals = np.flip(np.sort(vals))[:top_words]
            # print("MIN", sorted_vals.min(), "MAX", sorted_vals.max())
            best_words = np.array(words)[np.flip(np.argsort(vals))[:top_words]].tolist()
            best_words = [wd for wd in best_words if wd not in other_stop]
        except ValueError as e:
            print(f"VALUE ERROR {e}")
            best_words = []

        # wd may not be in word2id in the case that sklearn splits the words differently than we do (i.e. x-ray becomes x ray and we may only
        # have x-ray in our vocab
        tfidf_by_key[k] = [word_vocab_global[wd] for wd in best_words if wd in word_vocab_global]
    out_f = os.path.join(out_dir, f"tfidf_{i}.json")
    # print(f"Saving results from {i} to {out_f}")
    utils.dump_json_file(out_f, tfidf_by_key)
    return tfidf_by_key



def aggregate_word_dicts(args, list_of_dicts, key_count, top_k, min_occ=0, fraction_of_key_occ=0.0):
    """
    Merges the list of dictionaries to generate the top_k words per key.

    After grouping and summing the word occurrences per item, we
    - filter such that #word occurrences  > min_occ times
    - filter such that #word occurrences/#key occurrences > fraction_of_key_occ
    - take topK
    """
    # Aggregates dicts
    agg_dict = defaultdict(lambda: FreqDist())
    for sub_dict in tqdm(list_of_dicts):
        for item in sub_dict:
            for k in sub_dict[item]:
                agg_dict[item][k] += sub_dict[item][k]
    # Filter
    temp = []
    for item in tqdm(list(agg_dict.keys())):
        for word in list(agg_dict[item].keys()):
            word_cnt = agg_dict[item][word]
            item_cnt = key_count.get(item, 0)
            if word_cnt < min_occ:
                del agg_dict[item][word]
                continue
            if item_cnt > 0:
                temp.append(word_cnt/item_cnt)
            if item_cnt > 0 and word_cnt/item_cnt < fraction_of_key_occ:
                del agg_dict[item][word]
    # Take top K
    for item in tqdm(list(agg_dict.keys())):
        new_value = []
        for pair in agg_dict[item].most_common(top_k):
            # Json converts integer keys to strings
            assert type(pair[0]) is str and pair[0].isdigit()
            new_value.append(int(pair[0]))
        agg_dict[item] = new_value
    return agg_dict, temp


def save_as_trie(args, item_to_vocab_dict, out_dir, name, top_k):
    keys = []
    values = []
    for dict_key in tqdm(item_to_vocab_dict, desc="Saving trie"):
        # Pad with -1
        value = [-1]*top_k
        keys.append(dict_key)
        for i, word in enumerate(item_to_vocab_dict[dict_key]):
            value[i] = word
        values.append(tuple(value))
    fmt = "<" + "l"*top_k
    trie = marisa_trie.RecordTrie(fmt, zip(keys, values))
    config = {"fmt": fmt}
    logging.info(f"Saving bag of words to {out_dir}")
    utils.dump_json_file(filename=os.path.join(out_dir, "config.json"), contents=config)
    trie.save(os.path.join(out_dir, f'{name}.marisa'))
    return

def main():
    gl_start = time.time()
    multiprocessing.set_start_method("forkserver", force=True)
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    # Setup logging so we can see outputs from generate_slices.py
    formatter = logging.Formatter('%(asctime)s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logging.getLogger().addHandler(sh)
    logging.getLogger().setLevel(logging.INFO)

    # Read in all files
    load_dir = os.path.join(args.data_dir, args.subfolder_name)
    folders_to_prep = ["train", "dev", "test"]
    for fold in folders_to_prep:
        assert fold in os.listdir(load_dir), f"This method requries a {fold} folder in {load_dir}"
    # Split up files in the case of file ranges
    all_files = defaultdict(list)
    all_files_list = []
    for fold in folders_to_prep:
        key = os.path.join(load_dir, fold)
        all_files_list.extend([[key, int(os.path.splitext(f)[0].split("out_")[1]), f] for f in prep_utils.glob_files(f"{key}/out_*.jsonl")])
    all_files_list = list(sorted(all_files_list, key=lambda x: (x[0], x[1])))
    for f in all_files_list:
        # Store folder, input file
        all_files[f[0]].append(f[2])
    logging.info(f"Loaded {sum(len(files) for files in all_files.values())} files.")
    # Loaded folders from step 1 that got all the vocab
    temp_folders = []
    for fold in folders_to_prep:
        folds = [f for f in glob.glob(f"{load_dir}/{fold}*_temp") if os.path.isdir(f)]
        for f in folds:
            temp_folders.append(f)
    print("Part 1 Folders", temp_folders)
    vocab_out_dir = prep_utils.get_outdir(load_dir, os.path.join("entity_db/entity_mappings", f"entity_{VOCAB}"), remove_old=False)
    vocab_out_file = os.path.join(vocab_out_dir, f"{VOCABFILE}.marisa")
    logging.info(f"Saving vocab in {vocab_out_file}.")
    # '''
    st = time.time()
    merge_and_gen_vocab(temp_folders, vocab_out_file)
    logging.info(f"Done saving vocab in {time.time() - st}s")
    #**********************************************
    # MAP SENTENCES TO TOKEN IDS
    out_folders = launch_subprocess_translate_to_ints(args, all_files, vocab_out_file)
    # '''

    #**********************************************
    # BUILD AFFODANCES AND ENTITY WORD BAGS (ONLY OVER TRAIN DATA)
    load_dir = os.path.join(args.data_dir, args.subfolder_name)
    assert "train_prepped" in os.listdir(load_dir), f"This method requries a train folder in {load_dir}"
    logging.info(f"Loading data from {os.path.join(load_dir, 'train_prepped')}...")
    files = prep_utils.glob_files(f"{os.path.join(load_dir, 'train_prepped')}/*.jsonl")
    if args.test:
        files = files[:5]
    logging.info(f"Loaded {len(files)} files.")

    # Prep output folders
    # '''
    out_dir_types = prep_utils.get_outdir(load_dir, os.path.join("entity_db/entity_mappings", f"entity_{TYPEWORDS}"), remove_old=True)
    out_dir_rels = prep_utils.get_outdir(load_dir, os.path.join("entity_db/entity_mappings", f"entity_{RELATIONWORDS}"), remove_old=True)
    logging.info(f"Saving entity words and type words in {out_dir_types} and {out_dir_rels}")
    # There are other stats in stats, so don't remove stats
    out_dir_temp = prep_utils.get_outdir(load_dir, os.path.join("stats", "bags_temp"), remove_old=True)
    out_dir_temp_tfidf = prep_utils.get_outdir(load_dir, os.path.join("stats", "bags_temp/tfidf_res"), remove_old=True)
    vars(args)[f'saved_data_folder_{TYPEWORDS}'] = out_dir_types
    vars(args)[f'saved_data_folder_{RELATIONWORDS}'] = out_dir_rels
    vars(args)[f'save_data_folder_{VOCABFILE}'] = vocab_out_file
    prep_utils.save_config(args, f"{args.subfolder_name}_prep_generate_slices.json")
    # '''
    # load entity symbols
    # create entity dump and get qid2alias
    print("Loading entity dump")
    entity_symbols = EntitySymbols(load_dir=os.path.join(load_dir, 'entity_db/entity_mappings'))
    print(f"Loaded entity dump with {entity_symbols.num_entities} entities.")

    # Generating temp folders and dump files
    # We don't want to remove old ones so we do not use the prep_utils command (we want to reuse in generate_slices)
    # We index like generate_slices would where start id os 0 and end id is -1
    temp_folder = prep_utils.get_outdir(args.data_dir, f"{args.subfolder_name}_temp_dump_0_-1")
    entity_symbols_plus_f = os.path.join(temp_folder, "entity_symbols_plus.pkl")
    print(f"Will save entity_symbol files in {entity_symbols_plus_f}")

    # We are loading in the entity_symbol_plus from generate slices. This tries to load affordances
    # and bag of words, too, but as they are being generated here, they will be empty.
    if not os.path.exists(entity_symbols_plus_f) or args.overwrite_saved_entities:
        print(f"Loading types and relations.")
        entity_symbols_plus = create_entity_symbols_for_slices(args, entity_symbols)
        utils.dump_pickle_file(entity_symbols_plus_f, entity_symbols_plus)

    # launch subprocesses and collect outputs
    # '''
    launch_subprocess_get_word_bags(args, out_dir_temp, entity_symbols_plus_f, files)
    # '''
    # load_dir = os.path.join(args.data_dir, args.subfolder_name)
    # out_dir_word = prep_utils.get_outdir(load_dir, os.path.join("entity_db/entity_mappings", f"entity_{ENTITYWORDS}"), remove_old=False)
    # out_dir_types = prep_utils.get_outdir(load_dir, os.path.join("entity_db/entity_mappings", f"entity_{TYPEWORDS}"), remove_old=False)
    # out_dir_rels = prep_utils.get_outdir(load_dir, os.path.join("entity_db/entity_mappings", f"entity_{RELATIONWORDS}"), remove_old=False)
    # out_dir_temp = prep_utils.get_outdir(load_dir, os.path.join("stats", "bags_temp"), remove_old=False)
    # out_dir_temp_tfidf = prep_utils.get_outdir(load_dir, os.path.join("stats", "bags_temp/tfidf_res"), remove_old=True)

    # Counts of words associated with QIDs
    # Each file is mapping of qid to list of list of tokens in each sentence it appears
    # '''

    # Counts of verbs associated with types
    if os.path.exists(os.path.join(out_dir_temp, "AGG_TYPE_WORDS_FINAL.json")):
        print(f"Loading agg type words final")
        agg_type_words = utils.load_json_file(os.path.join(out_dir_temp, "AGG_TYPE_WORDS_FINAL.json"))
        print(f"Loaded {len(agg_type_words)} type words")
    else:
        type_count_files = glob.glob(f"{out_dir_temp}/*_{TYPEWORDS}.json")
        list_of_type_dicts = [utils.load_json_file(f) for f in tqdm(type_count_files, desc="Loading type files")]
        agg_type_words = aggregate_list_of_tokens(list_of_type_dicts)
        utils.dump_json_file(os.path.join(out_dir_temp, "AGG_TYPE_WORDS_FINAL.json"), agg_type_words)
    tifidf_type_words = compute_tfidf(agg_type_words, vocab_out_file, args.top_k_affordance_words, out_dir_temp_tfidf, args.processes)
    save_as_trie(args, tifidf_type_words, out_dir_types, TYPEWORDS, args.top_k_affordance_words)
    print("Done with type tfidf")
    # '''
    # Counts of words associated with relation types
    # '''
    if os.path.exists(os.path.join(out_dir_temp, "AGG_RELATION_WORDS_FINAL.json")):
        print(f"Loading agg relation words final")
        agg_rel_words = utils.load_json_file(os.path.join(out_dir_temp, "AGG_RELATION_WORDS_FINAL.json"))
    else:
        rel_type_count_files = glob.glob(f"{out_dir_temp}/*_{RELATIONWORDS}.json")
        list_of_rel_type_dicts = [utils.load_json_file(f) for f in tqdm(rel_type_count_files, desc="Loading relation files")]
        agg_rel_words = aggregate_list_of_tokens(list_of_rel_type_dicts)
        utils.dump_json_file(os.path.join(out_dir_temp, "AGG_RELATION_WORDS_FINAL.json"), agg_rel_words)
    tifidf_rel_words = compute_tfidf(agg_rel_words, vocab_out_file, args.top_k_affordance_words, out_dir_temp_tfidf, args.processes, max_words_item=15000)
    save_as_trie(args, tifidf_rel_words, out_dir_rels, RELATIONWORDS, args.top_k_affordance_words)
    # '''
    print("Done with relation tfidf")
    print(f"Done gathering counts and aggregating...")
    # remove temp
    print(f"Removing temporary files from f{out_dir_temp}")
    shutil.rmtree(out_dir_temp)
    print(f"Finished prep_generate_slices in {time.time() - gl_start} seconds.")





if __name__ == '__main__':
    main()