'''
This file
1. Reads in raw wikipedia sentences from /lfs/raiders7/0/lorr1/sentences
2. Reads in map of WPID-Title-QID from /lfs/raiders7/0/lorr1/title_to_all_ids.jsonl
3. Computes frequencies for alias-QID mentions over Wikipedia. Keeps only alias-QID mentions which occur at least args.min_frequency times
4. Merges alias-QID map with alias-QID map extracted from Wikidata
2. Saves alias-qid map as alias_to_qid_filter.json to args.data_dir

After this, run remove_bad_aliases.py

Example run command:
python3.6 -m contextual_embeddings.bootleg_data_prep.curate_aliases
'''

import argparse
import multiprocessing
import os

import ujson
import ujson as json
import numpy as np
import shutil
import time
import glob

from tqdm import tqdm

from bootleg_data_prep.language import ENSURE_ASCII
from bootleg_data_prep.utils import utils


def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='/dfs/scratch0/lorr1/bootleg/data/korealiases_title_1229', help='Where files saved')
    parser.add_argument('--out_subdir', type=str, default='incand_data', help='Where files saved')
    parser.add_argument('--alias2cands', type=str, default='dfs/scratch0/lorr1/bootleg/data/korealiases_title_1229/entity_db/entity_mappings/alias2qids.json', help='Where files saved')
    parser.add_argument('--test', action='store_true', help='If set, will only generate for one file.')
    parser.add_argument('--processes', type=int, default=int(50))
    return parser

def initializer(alias2cand_f):
    global alias2qid_global
    with open(alias2cand_f, "r") as in_f:
        alias2qid_global = ujson.load(in_f)

def filter_data(num_processes, alias2qid_f, data_file, output_file):
    st = time.time()
    # Chunk file for parallel writing
    # We do not use TemporaryFolders as the temp dir may not have enough space for large files
    create_ex_indir = os.path.join(os.path.dirname(data_file), "_bootleg_temp_indir")
    utils.ensure_dir(create_ex_indir)
    create_ex_outdir = os.path.join(os.path.dirname(data_file), "_bootleg_temp_outdir")
    utils.ensure_dir(create_ex_outdir)
    print(f"Counting lines")
    total_input = sum(1 for _ in open(data_file))
    chunk_input = int(np.ceil(total_input / num_processes))
    print(f"Chunking up {total_input} lines into subfiles of size {chunk_input} lines")
    total_input_from_chunks, input_files_dict = utils.chunk_file(data_file, create_ex_indir, chunk_input)

    # Generation input/output pairs
    input_files = list(input_files_dict.keys())
    input_file_lines = [input_files_dict[k] for k in input_files]
    output_files = [in_file_name.replace(create_ex_indir, create_ex_outdir) for in_file_name in input_files]
    assert total_input == total_input_from_chunks, f"Lengths of files {total_input} doesn't mathc {total_input_from_chunks}"
    print(f"Done chunking files")

    pool = multiprocessing.Pool(processes=num_processes, initializer=initializer, initargs=[alias2qid_f])

    input_args = list(zip(input_files, input_file_lines, output_files))

    total = 0
    for res in pool.imap(
            filter_data_hlp,
            input_args,
            chunksize=1
    ):
        total += 1

    # Merge output files to final file
    print(f"Merging output files")
    with open(output_file, 'wb') as outfile:
        for filename in glob.glob(os.path.join(create_ex_outdir, "*")):
            if filename == output_file:
                # don't want to copy the output into the output
                continue
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)
    # Remove temporary files/folders
    shutil.rmtree(create_ex_indir)
    shutil.rmtree(create_ex_outdir)


def filter_data_hlp(args):
    input_file, input_lines, output_file = args
    cache_alias2qids = {}
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in, total=input_lines, desc="Processing data"):
            line = ujson.loads(line)
            aliases = line['aliases']
            qids = line["qids"]
            spans = line["spans"]
            golds = line["gold"]
            slices = line["slices"]

            new_als, new_qids, new_spans, new_golds = [], [], [], []
            new_slices = {}
            j = 0
            for i, (al, qid) in enumerate(zip(aliases, qids)):
                if al in cache_alias2qids:
                    cands = cache_alias2qids[al]
                else:
                    cands = set(map(lambda x: x[0], alias2qid_global[al]))
                    cache_alias2qids[al] = cands
                if qid in cands:
                    new_als.append(aliases[i])
                    new_qids.append(qids[i])
                    new_spans.append(spans[i])
                    new_golds.append(golds[i])
                    for slice_name in slices:
                        if slice_name not in new_slices:
                            new_slices[slice_name] = {}
                        new_slices[slice_name][str(j)] = slices[slice_name][str(i)]
                    j += 1
            line['aliases'] = new_als
            line["qids"] = new_qids
            line["spans"] = new_spans
            line["gold"] = new_golds
            line["slices"] = new_slices
            f_out.write(ujson.dumps(line, ensure_ascii=ENSURE_ASCII) + "\n")


def main():
    gl_start = time.time()
    multiprocessing.set_start_method("spawn")
    args = get_arg_parser().parse_args()
    print(json.dumps(vars(args), indent=4))
    utils.ensure_dir(args.data_dir)

    out_dir = os.path.join(args.data_dir, args.out_subdir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Reading in files
    in_files = glob.glob(os.path.join(args.data_dir, "*.jsonl"))
    for file in in_files:
        out_file_name = os.path.basename(file)
        out_file = os.path.join(out_dir, out_file_name)
        print(f"Filtering {file} to {out_file}")
        filter_data(args.processes, args.alias2cands, file, out_file)

    print(f"Finished in {time.time()-gl_start}s")


if __name__ == '__main__':
    main()
