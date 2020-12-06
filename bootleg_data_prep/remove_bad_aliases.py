'''
This file
1. Reads in raw wikipedia sentences from /lfs/raiders7/0/lorr1/sentences
2. Reads in map of WPID-Title-QID from /lfs/raiders7/0/lorr1/title_to_all_ids.jsonl
3. Reads in alias-QID filter from <data_dir>/<alias_filter>
4. Iterates through mentions in all sentences, dropping alias-qid mentions which are not in our alias-QID map
5. Saves resulting sentences to <data_dir>/<out_subdir>
6. Lastly, it goes through and filters any QIDs in our alias-QID map that never appear in training nor in benchmark data.
    Aliases are dropped if, after QID filtering, they have no QIDs left.

After this, run add_labels_single_fun.py

Example run command:
python3.6 -m contextual_embeddings.bootleg_data_prep.remove_bad_aliases
'''
import argparse
import glob
import os
import shutil
import sys
import time
from collections import defaultdict
from multiprocessing import Queue, Process

import psutil
import ujson as json
from jsonlines import jsonlines
from tqdm import tqdm

import bootleg_data_prep.utils.utils as utils
import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.utils.classes.entity_symbols import EntitySymbols


# RAIDERS 8: python3 -m contextual_embeddings.bootleg_data_prep.remove_bad_aliases --sentence_dir /lfs/raiders8/0/lorr1/sentences_copy --title_to_qid /lfs/raiders8/0/lorr1/title_to_all_ids.jsonl
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_dir', type=str, default='/lfs/raiders10/0/lorr1/sentences_copy', help='Where files saved')
    parser.add_argument('--data_dir', type=str, default='data/wiki_dump', help='Directory for data to be saved.')
    parser.add_argument('--curate_aliases_subdir', type=str, default='curate_aliases', help='Subdirectory in data_dir')
    parser.add_argument('--out_subdir', type=str, default='alias_filtered_sentences', help='Subdirectory to save filtered sentences.')
    parser.add_argument('--title_to_qid', type=str, default='/lfs/raiders10/0/lorr1/title_to_all_ids.jsonl', help='Mapping of pageids to and titles to QIDs.')
    parser.add_argument('--alias_filter', type = str, default = 'alias_to_qid_filter.json', help = 'Path to JSON with alias filter (maps each alias to a set of appropriate QIDs). Inside curate_aliases_subdir.')
    parser.add_argument('--benchmark_qids', default = "", type =str, help = "File of list of QIDS that should be kept in entity dump. This is to ensure the trained model has entity embeddings for these.")
    parser.add_argument('--processes', type=int, default=int(50))
    parser.add_argument('--test', action = 'store_true', help = 'If set, will only generate for one file.')
    args = parser.parse_args()
    return args


def print_memory():
    process = psutil.Process(os.getpid())
    print(f"{int(process.memory_info().rss)/1024**3} GB ({process.memory_percent()}) memory used process {process}")


def launch_subprocess(args, outdir, temp_outdir, alias_qid_from_curate, title_to_qid, files):
    # dump jsons to pass
    print_memory()
    print(f"Memory of alias_qid_from_curate {sys.getsizeof(alias_qid_from_curate)/1024**3}")
    print(f"Memory of title_to_qid {sys.getsizeof(title_to_qid)/1024**3}")

    process_count = max(1, args.processes)
    maxsize = 10 * process_count

    # initialize jobs queue
    jobs_queue = Queue(maxsize=maxsize)
    print("After job_queue")
    # start worker processes
    workers = []
    for i in range(process_count):
        extractor = Process(target=extract_process,
                            args=(i, jobs_queue, len(files), args, outdir, temp_outdir, alias_qid_from_curate, title_to_qid))
        extractor.daemon = True  # only live while parent process lives
        extractor.start()
        workers.append(extractor)
    print("Mapper processes")
    # Mapper process
    for file_num, file in enumerate(files):
        job = (file_num, file)
        print(f"Mapper processes putting {file_num}")
        jobs_queue.put(job)  # goes to any available extract_process

    # signal termination
    for _ in workers:
        jobs_queue.put(None)
    # wait for workers to terminate
    for w in workers:
        w.join()
    return

def extract_process(j, jobs_queue, len_files, args, outdir, temp_outdir, alias_qid_from_curate, title_to_qid):
    print(f"Starting worker extractor {j}")
    global alias_qid_from_curate_gl
    global title_to_qid_gl

    alias_qid_from_curate_gl = alias_qid_from_curate
    title_to_qid_gl = title_to_qid
    while True:
        job = jobs_queue.get()  # job is (id, in_filepath)
        if job:
            i, in_filepath = job
            print(f"Starting job {i}")
            subprocess(i, len_files, args, outdir, temp_outdir, in_filepath)
            print(f"Finishing job {i}")
        else:
            print(f"Breaking extractor {j}")
            break


def subprocess(i, len_files, args, outdir, temp_outdir, in_filepath):
    # i, total, args, outdir, temp_outdir, in_filepath = all_args
    print(len(title_to_qid_gl), len(alias_qid_from_curate_gl))
    print_memory()
    print(f"Starting {i}/{len_files}. Reading in {in_filepath}.")
    start = time.time()

    # create output files:  
    out_fname = prep_utils.get_outfname(in_filepath)
    out_file = open(os.path.join(outdir, out_fname), "w")

    # track the local frequency of alias-to-qids
    filtered_aliases_to_qid_count = defaultdict(lambda: defaultdict(int))
    filtered_qid_count = defaultdict(int)
    discarded_counts = {'no_alias': 0, 'no_qid': 0, 'not_in_filter': 0, 'span_issue': 0, 'qid_neg_one': 0, 'len_zero_alias': 0}
    discarded_values = {'len_zero_alias': defaultdict(lambda: defaultdict(int)),
                        'no_alias': defaultdict(lambda: defaultdict(int)),
                        'no_qid': defaultdict(lambda: defaultdict(int)),
                        'not_in_filter': defaultdict(lambda: defaultdict(int)),
                        'span_issue': defaultdict(lambda: defaultdict(int)),
                        'qid_neg_one': defaultdict(lambda: defaultdict(int))}
    entities_kept = {}
    # We want to separately keep track of all wikipage QIDs because a few of them have no incoming links
    # We still want these to be augmented in the next step so must keep these in our entity dump
    wiki_page_qids = set()
    total_kept = 0
    with jsonlines.open(in_filepath, 'r') as in_file:
        for doc in in_file:
            new_doc = {
                'qid': title_to_qid_gl.get(doc['page_title'], "-1"),
                'title': doc['page_title'], 
                'sentences': []
            }
            if new_doc['qid'] != "-1":
                wiki_page_qids.add(new_doc['qid'])
            for sentence in doc['aliases']:
                sent_idx_str = 'sent_idx'
                if 'doc_sent_idx' in sentence:
                    sent_idx_str = 'doc_sent_idx'
                new_sent = {
                    'doc_sent_idx': sentence[sent_idx_str],
                    'sentence': sentence['sentence'], 
                    'aliases': [],
                    'qids': [],
                    'spans': []
                }
                num_words = len(sentence['sentence'].split(" "))
                if len(sentence['spans']) > 0 and type(sentence['spans'][0]) is str:
                    sentence['spans'] = [list(map(int, s.split(":"))) for s in sentence["spans"]]
                # Iterate through the aliases in the sentence and check that they match the critera
                for alias, title, span in zip(sentence['aliases'], sentence['titles'], sentence['spans']):
                    alias = prep_utils.get_lnrm(alias)
                    if len(alias) <= 0:
                        discarded_counts['len_zero_alias'] += 1
                        discarded_values['len_zero_alias'][alias][title] += 1
                        continue
                    if span[0] >= num_words:
                        discarded_counts['span_issue'] += 1
                        discarded_values['span_issue'][alias][title] += 1
                        continue
                    if alias not in alias_qid_from_curate_gl:
                        discarded_counts['no_alias'] += 1
                        discarded_values['no_alias'][alias][title] += 1
                        continue 
                    if title not in title_to_qid_gl:
                        discarded_counts['no_qid'] += 1
                        discarded_values['no_qid'][alias][title] += 1
                        continue 
                    qid = str(title_to_qid_gl[title])
                    if qid not in alias_qid_from_curate_gl[alias]:
                        discarded_counts['not_in_filter'] += 1
                        discarded_values['not_in_filter'][alias][title] += 1
                        continue
                    if qid == "-1":
                        discarded_counts['qid_neg_one'] += 1
                        discarded_values['qid_neg_one'][alias][title] += 1
                        continue
                        
                    entities_kept[qid] = 1
                    total_kept += 1
                    new_sent['aliases'].append(alias)
                    new_sent['qids'].append(qid)
                    new_sent['spans'].append(span)
                    filtered_aliases_to_qid_count[alias][qid] += 1
                    filtered_qid_count[qid] += 1
                new_doc['sentences'].append(new_sent)
            out_file.write(json.dumps(new_doc) + '\n')
    out_file.close()
    sum_discarded_counts = sum(discarded_counts.values())
    print(f"Finished {i}/{len_files}. Written to {out_fname}. {time.time() - start} seconds.\n"
          f"Entities kept: {len(entities_kept)}.\n"
          f"Page Ids Seen: {len(wiki_page_qids)}.\n"
          f"Aliases kept: {total_kept} ({total_kept / (total_kept + sum_discarded_counts)}%)\n"
          f"Discarded: {json.dumps(discarded_counts, indent=4)}."
    )
    utils.dump_json_file(os.path.join(temp_outdir, f"filtered_aliases_to_qid_count_{i}.json"), filtered_aliases_to_qid_count)
    utils.dump_json_file(os.path.join(temp_outdir, f"filtered_qid_count_{i}.json"), filtered_qid_count)
    utils.dump_json_file(os.path.join(temp_outdir, f"wiki_page_qids_{i}.json"), wiki_page_qids)
    utils.dump_json_file(os.path.join(temp_outdir, f"discarded_counts_{i}.json"), discarded_counts)
    utils.dump_json_file(os.path.join(temp_outdir, f"discarded_values_{i}.json"), discarded_values)
    return

def make_entity_symbol(alias2qid_from_curate, alias2qids_counts, qid_counts, qid_to_title, benchmark_qids, wiki_page_qids, args):
    alias2qids_out = {}
    all_qids = set(qid_counts.keys()).union(benchmark_qids)
    all_qids = all_qids.union(wiki_page_qids)
    if "-1" in all_qids:
        print(f"Removing -1 from all_qids...this should probably not be in the set")
        all_qids.remove("-1")
    print(f"There are {len(all_qids)} qids about to be filtered for the entity dump and {len(alias2qid_from_curate)} raw aliases")
    max_candidates = 0
    max_alias_len = 0
    for alias in tqdm(list(alias2qid_from_curate.keys())):
        alias_qids_dict = alias2qid_from_curate[alias]
        new_qids = set(alias_qids_dict.keys()).intersection(all_qids)
        if len(new_qids) == 0:
            del alias2qid_from_curate[alias]
            # print(f"Deleting alias {alias} because no QIDs in our data")
        else:
            # Give QIDs not seen count of 1
            alias2qids_out[alias] = [[qid, qid_counts.get(qid, 1)] for qid in new_qids]
            max_candidates = max(max_candidates, len(new_qids))
            max_alias_len = max(max_alias_len, len(alias.split(" ")))

    print(f"There are {len(alias2qids_out)} aliases going into the entity dump")
    # Make entity dump object
    entity_dump = EntitySymbols(
        max_candidates=max_candidates,
        max_alias_len=max_alias_len,
        alias2qids=alias2qids_out,
        qid2title=qid_to_title
    )
    out_dir = os.path.join(args.data_dir, args.out_subdir, "entity_db/entity_mappings")
    vars(args)["entity_dump_dir"] = out_dir
    entity_dump.dump(out_dir)

def main():
    gl_start = time.time()
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    utils.ensure_dir("{:s}/".format(args.data_dir))

    # Store intermediate results
    temp_outdir = prep_utils.get_outdir(args.data_dir, "_temp", remove_old=True)

    # loader folder
    load_dir = os.path.join(args.data_dir, args.curate_aliases_subdir)

    # store final results
    outdir = prep_utils.get_outdir(args.data_dir, args.out_subdir, remove_old=True)

    # load aliases_to_qid
    start = time.time()
    in_file = os.path.join(load_dir, args.alias_filter)
    # nested dict of alias: qid: score
    alias_qid_from_curate = utils.load_json_file(in_file)
    print(f"Loaded candidates for {len(alias_qid_from_curate)} aliases from {in_file}. {time.time() - start} seconds.")

    title_to_qid, qid_to_all_titles, _, qid_to_title = prep_utils.load_qid_title_map(args.title_to_qid)
    print_memory()
    # launch subprocesses
    files = glob.glob(f"{args.sentence_dir}/*/wiki_*")
    if args.test:
        files = files[:1]
    print(f"Loaded {len(files)} files from {args.sentence_dir}. Launching {args.processes} processes.")
    launch_subprocess(args, outdir, temp_outdir, alias_qid_from_curate, title_to_qid, files)

    # read in dumps
    aliases_to_qid_count_files = glob.glob(f"{temp_outdir}/filtered_aliases_to_qid_count_*")
    list_of_alias_dicts = [utils.load_json_file(f) for f in aliases_to_qid_count_files]
    qid_count_files = glob.glob(f"{temp_outdir}/filtered_qid_count_*")
    list_of_qid_dicts = [utils.load_json_file(f) for f in qid_count_files]
    wiki_page_qid_files = glob.glob(f"{temp_outdir}/wiki_page_qids_*")
    list_of_wiki_page_qid_dicts = [set(utils.load_json_file(f)) for f in wiki_page_qid_files]
    discarded_counts_files = glob.glob(f"{temp_outdir}/discarded_counts_*")
    list_of_discarded_counts_dicts = [utils.load_json_file(f) for f in discarded_counts_files]
    discarded_values_files = glob.glob(f"{temp_outdir}/discarded_values_*")
    list_of_discarded_values_dicts = [utils.load_json_file(f) for f in discarded_values_files]

    # merge outputs
    alias_to_qid_count = prep_utils.aggregate_list_of_nested_dictionaries(list_of_alias_dicts)
    qid_counts = prep_utils.aggregate_list_of_dictionaries(list_of_qid_dicts)
    wiki_page_qids = set.union(*list_of_wiki_page_qid_dicts)
    discarded_counts_stats = prep_utils.aggregate_list_of_dictionaries(list_of_discarded_counts_dicts)
    discarded_values_stats = {}
    for key in list_of_discarded_values_dicts[0]:
        discarded_values_stats[key] = prep_utils.aggregate_list_of_nested_dictionaries([lst[key] for lst in list_of_discarded_values_dicts])

    vars(args)["discarded_counts_stats"] = discarded_counts_stats

    benchmark_qids = []
    if os.path.exists(args.benchmark_qids):
        with open(args.benchmark_qids) as in_file:
            benchmark_qids = json.load(in_file)
    if len(benchmark_qids) == 0:
        print("*************************\n"*10)
        print("ZERO benchmark qids have been loaded. Did you mean this?")
    print(f"Loaded {len(benchmark_qids)} QIDS from {args.benchmark_qids}")

    make_entity_symbol(alias_qid_from_curate, alias_to_qid_count, qid_counts, qid_to_title, set(benchmark_qids), wiki_page_qids, args)

    # remove temp
    shutil.rmtree(temp_outdir)
    utils.dump_json_file(os.path.join(outdir, "discarded_bad_aliases.json"), discarded_values_stats)
    utils.dump_json_file(os.path.join(outdir, "alias_to_qid_count.json"), alias_to_qid_count)
    utils.dump_json_file(os.path.join(outdir, "qid_counts.json"), qid_counts)
    utils.dump_json_file(os.path.join(outdir, "wiki_page_qids.json"), wiki_page_qids)
    prep_utils.save_config(args, "remove_bad_aliases_config.json")
    print(f"Finished remove_bad_aliases in {time.time() - gl_start} seconds.")

if __name__ == '__main__':
    main()