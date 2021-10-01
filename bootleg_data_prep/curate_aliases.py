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
import ujson as json
import copy
import shutil
import time
from html import escape, unescape
import jsonlines
from collections import defaultdict
import glob

from tqdm import tqdm

from bootleg_data_prep.language import get_lnrm
from bootleg_data_prep.utils import utils
import bootleg_data_prep.utils.data_prep_utils as prep_utils


def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data and runs
    parser.add_argument('--sentence_dir', type=str, default='/lfs/raiders10/0/lorr1/sentences_copy', help='Where files saved')
    parser.add_argument('--data_dir', type=str, default='data/wiki_dump', help='Where files saved')
    parser.add_argument('--out_subdir', type=str, default='curate_aliases', help='Where files saved')
    parser.add_argument('--title_to_qid', type=str, default='/lfs/raiders10/0/lorr1/title_to_all_ids.jsonl')
    parser.add_argument('--wd_aliases', type=str, default=None, help='Path to directory with JSONL mapping alias to QID')
    parser.add_argument('--min_frequency', type=int, default=4, help='Minimum number of times a QID must appear with an alias')
    parser.add_argument('--strip', action='store_true', help='If set, will strip punctuation of aliases.')
    parser.add_argument('--lower', action='store_true', help='If set, will lower case all aliases.')
    parser.add_argument('--test', action='store_true', help='If set, will only generate for one file.')
    parser.add_argument('--processes', type=int, default=int(50))
    return parser


def launch_subprocess(args, temp_outdir, in_files):
    all_process_args = [tuple([i + 1,
                               len(in_files),
                               args,
                               temp_outdir,
                               in_files[i],
                               ]) for i in range(len(in_files))]
    pool = multiprocessing.Pool(processes=args.processes)
    pool.map(subprocess, all_process_args, chunksize=1)

    pool.close()
    return


def subprocess(all_args):
    """
    Each subprocess launches over a different jsonl file where each line corresponds to a Wikipedia page.
    We first collect all bolded aliases for a page (collected from the bolded words in the first few sentences).
    We then iterate through each alias in each sentence and track how frequently aliases and titles coccur.
    """
    i, total, args, out_dir, in_filepath = all_args
    hashed_outfilename = prep_utils.get_outfname(in_filepath, ending="json")
    outfilename = os.path.join(out_dir, os.path.splitext(hashed_outfilename)[0] + "_anchoraliases.json")
    boldalias_outfilename = os.path.join(out_dir, os.path.splitext(hashed_outfilename)[0] + "_boldaliases.json")
    acronymalias_outfilename = os.path.join(out_dir, os.path.splitext(hashed_outfilename)[0] + "_acronymaliases.json")
    print(f"Starting {i}/{total}. Reading in {in_filepath}. Ouputting to {outfilename}")
    aliases_to_title = defaultdict(lambda: defaultdict(int))
    boldaliases_to_title = defaultdict(lambda: defaultdict(int))
    acronymaliases_to_title = defaultdict(lambda: defaultdict(int))
    with jsonlines.open(in_filepath, 'r') as in_file:
        for page_obj in in_file:
            for alias in page_obj.get("bold_aliases", []):
                alias = get_lnrm(alias, args.strip, args.lower)
                if len(alias) > 0:
                    boldaliases_to_title[alias][page_obj["page_title"]] += 1
            for alias in page_obj.get("acronym_aliases", []):
                alias = get_lnrm(alias, args.strip, args.lower)
                if len(alias) > 0:
                    acronymaliases_to_title[alias][page_obj["page_title"]] += 1
            # aliases is a list of sentences with aliases, their gold wikipedia page title, the text, and spans
            for sentence in page_obj["aliases"]:
                pairs = zip(sentence["aliases"], sentence["titles"])
                for alias, title in pairs:
                    # normalize alias
                    alias = get_lnrm(alias, args.strip, args.lower)
                    if len(alias) > 0:
                        aliases_to_title[alias][title] += 1
    utils.dump_json_file(outfilename, aliases_to_title)
    utils.dump_json_file(boldalias_outfilename, boldaliases_to_title)
    utils.dump_json_file(acronymalias_outfilename, acronymaliases_to_title)
    return


def filter_aliases_and_convert_to_qid(anchoraliases_to_title, boldaliases_to_title, acronymaliases_to_title, title_to_qid, qid_to_all_titles, args):
    """ 
        1. We walk through each anchor alias-title pair and keep the ones that appear a minimum of two times.
        2. We union these with the bold alias-title pairs where the bold aliases appear in the first few sentences of a Wikipedia page.
        2. We union these with the acronyms for the title found in the first few sentences of a Wikipedia page.
        3. We map each title to a wikidata QID. If a title doesn't have a mapping, we discard it.

    We build a map of aliases-titles which meet our frequency requirement AND for which the title maps
    to a QID. We save these to file.    
    """
    # aliases_to_title is {alias: {title: count}}
    total_pairs_anchor = sum([len(title_dict) for title_dict in anchoraliases_to_title.values()])
    total_pairs_bold = sum([len(title_dict) for title_dict in boldaliases_to_title.values()])
    total_pairs_acronym = sum([len(title_dict) for title_dict in acronymaliases_to_title.values()])
    vars(args)["original_anchor_alias_entity_pair_count"] = total_pairs_anchor
    vars(args)["original_bold_alias_entity_pair_count"] = total_pairs_bold
    vars(args)["original_acronym_alias_entity_pair_count"] = total_pairs_acronym
    print(f"Extracted {total_pairs_anchor} anchor alias-title pairs and {total_pairs_bold} bold alias-title pairs from Wikipedia"
          f" and {total_pairs_acronym} acronym alias-title pairs from Wikipedia. Filtering anchor alias-title pairs by frequency.")

    # track important aliases/qids/etc
    filtered_aliasqid = defaultdict(lambda: defaultdict(int))  # The set of alias-QID pairs whose frequency >= args.min_frequency
    filtered_qids = defaultdict(int)  # Entity count of all QIDs remaining
    qid_unavailable = defaultdict(int)  # A dictionary of titles for which we cannot map to a QID
    unpopular_removed = defaultdict(lambda: defaultdict(int))
    for alias, title_dict in tqdm(anchoraliases_to_title.items()):
        for title_raw, count in title_dict.items():
            if count >= args.min_frequency:
                title = title_raw
                if title not in title_to_qid:
                    title = unescape(title_raw)
                if title not in title_to_qid:
                    title = escape(title_raw)
                if title in title_to_qid:
                    filtered_aliasqid[alias][title_to_qid[title]] += 1
                    filtered_qids[title_to_qid[title]] += 1
                # The ELSE happens when we link to an external non-wikipedia page, the wikipedia page doesn't have a QID (rare), or the title is a redirect
                # eg https://en.wikipedia.org/wiki/Vidyodaya_University redirects to https://en.wikipedia.org/wiki/University_of_Sri_Jayewardenepura
                else:
                    qid_unavailable[title] += count
            else:
                unpopular_removed[alias][title_raw] += count
    # union with bold aliases and acronym aliases, incrementing count again as the bold aliases link to the page they are on
    for alias_dict in [boldaliases_to_title, acronymaliases_to_title]:
        for alias, title_dict in tqdm(alias_dict.items()):
            for title_raw, count in title_dict.items():
                title = title_raw
                if title not in title_to_qid:
                    title = unescape(title_raw)
                if title not in title_to_qid:
                    title = escape(title_raw)
                if title in title_to_qid:
                    filtered_aliasqid[alias][title_to_qid[title]] += 1
                    filtered_qids[title_to_qid[title]] += 1
                else:
                    qid_unavailable[title] += count
    # we increment the count here to represent that each page links to itself; as we compute counts later, this is just for sorting
    for qid in qid_to_all_titles:
        for title in qid_to_all_titles[qid]:
            alias = get_lnrm(title, args.strip, args.lower)
            if len(alias) > 0:
                filtered_aliasqid[alias][qid] += 1
                filtered_qids[qid] += 1

    total_wikipedia_mentions = sum([v for qid_dict in anchoraliases_to_title.values() for v in qid_dict.values()])
    print(f"{total_wikipedia_mentions} total wikipedia mentions")
    total_pairs = sum([len(qid_dict) for qid_dict in filtered_aliasqid.values()])
    vars(args)["filtered_alias_qid_pair_count"] = total_pairs
    print(f"{total_pairs} alias-title pairs after filtering.")
    # Count total QIDs
    print(f"{len(filtered_qids)} unique QIDs appear in Wikipedia")
    vars(args)["filtered_qids_count"] = len(filtered_qids)
    # Count the number of popular misses
    vars(args)["unique_popular_titles_discarded"] = len(qid_unavailable)
    print(f"{len(qid_unavailable)} unique titles discarded from misses.")
    total_aliases_discarded = sum(qid_unavailable.values())
    vars(args)["total_popular_aliases_discarded"] = total_aliases_discarded
    print(f"{total_aliases_discarded} total aliases discarded from misses.")
    # Count the number of unpopular misses
    vars(args)["unique_unpopular_aliases_discarded"] = len(unpopular_removed)
    print(f"{len(unpopular_removed)} unique aliases discarded from freq.")
    total_aliases_discarded = sum([len(alias_dict) for alias_dict in unpopular_removed.values()])
    vars(args)["total_unpopular_aliase_title_pairs_discarded"] = total_aliases_discarded
    print(f"{total_aliases_discarded} total alias-title discarded from freq.")
    keys = [set(alias_dict.keys()) for alias_dict in unpopular_removed.values()]
    total_titles_discarded = len(set.union(*keys)) if keys else 0
    vars(args)["total_unpopular_titles_discarded"] = total_titles_discarded
    print(f"{total_aliases_discarded} total titles discarded from freq.")
    return filtered_aliasqid, filtered_qids, qid_unavailable, unpopular_removed


def merge_wikidata_aliases(args, aliases_to_qid, all_qids, wikidata_alias_to_qid):
    """ We merge alias-qid pairs from Wikidata with the alias-qid pairs we've 
    extracted from Wikipedia. Note that we recompute QID counts in the next step for the candidate maps.
    """

    print("Adding wikidata aliases...")
    stats = defaultdict(int)
    new_qids_from_wikidata = set()
    for wd_alias, qids in tqdm(wikidata_alias_to_qid.items()):
        wd_alias = get_lnrm(wd_alias, args.strip, args.lower)
        if len(wd_alias) <= 0:
            continue
        if wd_alias in aliases_to_qid:
            for qid in qids:
                if qid not in aliases_to_qid[wd_alias]:
                    stats["number_added_qids"] += 1
                    aliases_to_qid[wd_alias][qid] = 1
        else:
            stats["number_added_aliases"] += 1
            for qid in qids:
                if qid not in all_qids:
                    new_qids_from_wikidata.add(qid)
            aliases_to_qid[wd_alias] = {qid: 1 for qid in qids}

    print(f"{stats['number_added_qids']} number of added qid-alias pairs from wikidata")
    print(f"{stats['number_added_aliases']} number of added aliases from wikidata")
    print(f"{len(new_qids_from_wikidata)} new QIDs have been addded from {len(all_qids)}")
    vars(args)["number_added_qids_wikidata"] = stats['number_added_qids']
    vars(args)["number_added_aliases_wikidata"] = stats['number_added_aliases']
    vars(args)["number_added_qids_wikidata"] = len(new_qids_from_wikidata)
    return aliases_to_qid, list(new_qids_from_wikidata)


def main():
    gl_start = time.time()
    multiprocessing.set_start_method("spawn")
    args = get_arg_parser().parse_args()
    print(json.dumps(vars(args), indent=4))
    utils.ensure_dir(args.data_dir)

    print(f"Loading data from {args.sentence_dir}...")
    files = glob.glob(f"{args.sentence_dir}/*/wiki_*")
    if args.test:
        files = files[:1]

    # Get outdirectory
    outdir = prep_utils.get_outdir(args.data_dir, args.out_subdir, remove_old=True)

    # Store intermediate results
    temp_outdir = prep_utils.get_outdir(args.data_dir, "_temp", remove_old=True)

    print(f"Loaded {len(files)} files.")
    print(f"Launching subprocess with {args.processes} processes...")
    launch_subprocess(args, temp_outdir, files)
    print("Finished subprocesses.")

    # load wikidata qid-to-title map
    title_to_qid, qid_to_all_titles, _, _ = prep_utils.load_qid_title_map(args.title_to_qid)
    # Aggregate alias-title counts from list and filter.

    list_of_anchoraliases_to_titles = [utils.load_json_file(f) for f in glob.glob(f"{temp_outdir}/*_anchoraliases.json")]
    list_of_boldaliases_to_titles = [utils.load_json_file(f) for f in glob.glob(f"{temp_outdir}/*_boldaliases.json")]
    list_of_acronymaliases_to_titles = [utils.load_json_file(f) for f in glob.glob(f"{temp_outdir}/*_acronymaliases.json")]
    print("Aggregating list of dictionaries.")
    anchoraliases_to_title = prep_utils.aggregate_list_of_nested_dictionaries(list_of_anchoraliases_to_titles)
    boldaliases_to_title = prep_utils.aggregate_list_of_nested_dictionaries(list_of_boldaliases_to_titles)
    acronymaliases_to_title = prep_utils.aggregate_list_of_nested_dictionaries(list_of_acronymaliases_to_titles)
    # TODO: REMOVE
    al = "distributed algorithms"
    print("AL", al)
    print(boldaliases_to_title.get(al, []))
    print(acronymaliases_to_title.get(al, []))
    print(anchoraliases_to_title.get(al, []))
    # filter aliases and convert to QID
    aliases_to_qid, all_qids, qid_unavailable, unpopular_removed = filter_aliases_and_convert_to_qid(anchoraliases_to_title, boldaliases_to_title,
                                                                                                     acronymaliases_to_title,
                                                                                                     title_to_qid, qid_to_all_titles, args)
    for al in aliases_to_qid:
        assert len(al) > 0

    # load wikidata aliases; each line is json if alias: qid value
    wikidata_alias_to_qid = {}
    if args.wd_aliases:
        with jsonlines.open(args.wd_aliases) as in_f:
            for line in in_f:
                for k in line:
                    wikidata_alias_to_qid[k] = line[k]
    for al in wikidata_alias_to_qid:
        wikidata_alias_to_qid[al] = set(wikidata_alias_to_qid[al])

    # Merge aliases and QIDs
    print(f"Found {len(wikidata_alias_to_qid)} wikidata aliases. Merging with wikidata aliases")
    aliases_to_qid_merged, new_qids_from_wikidata = merge_wikidata_aliases(args, copy.deepcopy(aliases_to_qid), all_qids, wikidata_alias_to_qid)

    # remove temp
    shutil.rmtree(temp_outdir)

    # save to file
    out_file = os.path.join(outdir, "wp_anchor_aliases_to_title.json")
    utils.dump_json_file(out_file, anchoraliases_to_title)
    vars(args)["out_anchor_aliases_to_title_file"] = out_file

    out_file = os.path.join(outdir, "wp_bold_aliases_to_title.json")
    utils.dump_json_file(out_file, boldaliases_to_title)
    vars(args)["out_bold_aliases_to_title_file"] = out_file

    out_file = os.path.join(outdir, "wp_acronym_aliases_to_title.json")
    utils.dump_json_file(out_file, acronymaliases_to_title)
    vars(args)["out_acronym_aliases_to_title_file"] = out_file

    out_file = os.path.join(outdir, "alias_to_qid_filter.json")
    utils.dump_json_file(out_file, aliases_to_qid_merged)
    vars(args)["out_aliases_to_qid_filter_file"] = out_file

    out_file = os.path.join(outdir, "qids_added_wikidata.json")
    utils.dump_json_file(out_file, new_qids_from_wikidata)
    vars(args)["out_qids_added_wikidata_file"] = out_file

    out_file = os.path.join(outdir, "alias_to_qid_wikipedia.json")
    utils.dump_json_file(out_file, aliases_to_qid)
    vars(args)["out_aliases_to_qid_wikipedia_file"] = out_file

    out_file = os.path.join(outdir, "alias_with_unknown_qid.json")
    utils.dump_json_file(out_file, qid_unavailable)
    vars(args)["out_qid_unavailable_file"] = out_file

    out_file = os.path.join(outdir, "alias_title_unpopular_removed.json")
    utils.dump_json_file(out_file, unpopular_removed)
    vars(args)["out_unpopular_removed_file"] = out_file

    prep_utils.save_config(args, "curate_aliases_config.json")
    print(f"Data saved to {args.data_dir}")
    print(f"Finished curate_aliases in {time.time() - gl_start} seconds.")


if __name__ == '__main__':
    main()
