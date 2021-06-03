'''
This simply goes and collects all unique vocab words in the data that have been stemmed (for slice generation). This is separated out so it can
be run in parallel on multiple machines if desired.

python3 -m contextual_embeddings.bootleg_data_prep.prep_generate_slices_part1 --subfolder_name korealiases --processes 30
'''


import argparse
import logging
import multiprocessing
import os
import time
from collections import defaultdict

import jsonlines
import ujson as json

import bootleg_data_prep.utils.utils as utils
import bootleg_data_prep.utils.data_prep_utils as prep_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/wiki_dump', help='Directory for data to be saved.')
    parser.add_argument('--subfolder_name', type=str, default='orig_wl_prn', help = 'Subdirectory to save filtered sentences.')
    parser.add_argument('--processes', type=int, default=int(0.8*multiprocessing.cpu_count()))
    parser.add_argument('--file_id_start', type=int, default=0, help="start file index to read of files for each folder; start idx 4 will start reading 4th file from dev and 4th from test")
    parser.add_argument('--file_id_end', type=int, default=-1, help="end file index to read (-1 for all) (exclusive)")
    parser.add_argument('--multilingual', action = 'store_true', help = 'If set, will not do english-based language processing (e.g. stemming).')
    parser.add_argument('--test', action = 'store_true', help = 'If set, will only generate for one file.')

    args = parser.parse_args()
    return args


def launch_subprocess_gen_vocab_ints(args, all_files):
    files = []
    # key is dev, test, or train director
    for key in all_files:
        path = os.path.dirname(key)
        temp_folder = prep_utils.get_outdir(path, f"{os.path.basename(key)}_{args.file_id_start}_{args.file_id_end}_temp", remove_old=True)
        for f in all_files[key]:
            files.append(tuple([f, temp_folder]))

    all_process_args = [tuple([i + 1,
                               len(files),
                               args,
                               files[i]
                               ]) for i in range(len(files))]
    pool = multiprocessing.Pool(processes=args.processes)
    logging.info("****************************COLLECTING VOCABULARY WORDS****************************")
    logging.info("Starting pool...")
    st = time.time()
    pool.map(subprocess_gen_vocab_ints, all_process_args, chunksize=1)
    logging.info(f"Finished in {time.time()-st}s")
    pool.close()
    return

# This collects ALL WORDs in the sentences to generate a vocab mapping
# This map will need to be merged before final word->int assignments can be made
def subprocess_gen_vocab_ints(all_args):
    start_time = time.time()
    idx, total, args, file_tup = all_args
    in_filepath, temp_folder = file_tup
    # create output files:
    out_fname = os.path.join(temp_folder, os.path.splitext(os.path.basename(in_filepath))[0] + f"_vocab.json")

    print(f"Starting {idx}/{total}. Reading in {in_filepath}. Ouputting to {out_fname}")
    vocab_words = set()
    with jsonlines.open(in_filepath, 'r') as in_file:
        for sent_idx, sent_obj in enumerate(in_file):
            tokens, tokens_pos, verb_unigrams, verb_unigrams_pos, verb_bigrams, verb_bigrams_pos = prep_utils.clean_sentence_to_tokens(sent_obj["sentence"], args.multilingual)
            for t in tokens:
                if t not in vocab_words:
                    vocab_words.add(t)
            for t in verb_unigrams:
                if t not in vocab_words:
                    vocab_words.add(t)
            for t in verb_bigrams:
                if t not in vocab_words:
                    vocab_words.add(t)

    utils.dump_json_file(out_fname, list(vocab_words))
    print(f"Finished {idx}/{total}. Written to {out_fname}. {time.time() - start_time} seconds. {len(vocab_words)} words.")
    return None


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
    file_id_start = args.file_id_start
    file_id_end = args.file_id_end
    for fold in folders_to_prep:
        key = os.path.join(load_dir, fold)
        all_files_list.extend([[key, int(os.path.splitext(f)[0].split("out_")[1]), f] for f in prep_utils.glob_files(f"{key}/out_*.jsonl")])
    all_files_list = list(sorted(all_files_list, key=lambda x: (x[0], x[1])))
    if file_id_end == -1:
        file_id_end = len(all_files_list)
    files_to_run = all_files_list[file_id_start:file_id_end]
    for f in files_to_run:
        # Store folder, input file
        all_files[f[0]].append(f[2])
    logging.info(f"Loaded {sum(len(files) for files in all_files.values())} files.")
    #**********************************************
    # GENERATE VOCAB TO INT MAP (VOCAB WORDS ARE STEMMED FOR SLICES)
    launch_subprocess_gen_vocab_ints(args, all_files)
    print(f"Finished prep_generate_slices in {time.time() - gl_start} seconds.")





if __name__ == '__main__':
    main()