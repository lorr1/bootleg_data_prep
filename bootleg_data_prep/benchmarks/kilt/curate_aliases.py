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
from urllib.parse import quote, unquote
import html
import multiprocessing
import os
import ujson as json
import copy
import shutil
import time

import jsonlines
from collections import defaultdict
import glob

from tqdm import tqdm

from bootleg_data_prep.utils import utils
import bootleg_data_prep.utils.data_prep_utils as prep_utils

"""
Imput Example
1	{"_id": "21936854", "wikipedia_id": "21936854", "wikipedia_title": "Darrell Panizza",
     "text": "Darrell Peter Panizza (born 11 March 1959 in Bunbury, Western Australia) is a former Australian rules footballer who represented in the West Australian Football League (WAFL) and the now-defunct Woodville Football Club in the South Australian National Football League (SANFL) during the 1980s and 1990s. He also represented and coached Western Australia in interstate football.",
     "anchors": [
         {"text": "Bunbury, Western Australia", "href": "Bunbury%2C%20Western%20Australia", "source": {"paragraph_id": 1, "start": 45, "end": 71}, "start": 45, "end": 71},
         {"text": "Australian rules football", "href": "Australian%20rules%20football", "source": {"paragraph_id": 1, "start": 85, "end": 110}, "start": 85, "end": 110}, 
         {"text": "West Australian Football League", "href": "West%20Australian%20Football%20League", "source": {"paragraph_id": 1, "start": 136, "end": 167}, "start": 136, "end": 167}, 
         {"text": "Woodville Football Club", "href": "Woodville%20Football%20Club", "source": {"paragraph_id": 1, "start": 195, "end": 218}, "start": 195, "end": 218}], 
    "categories": "Claremont Football Club coaches,1959 births,Woodville Football Club players,Claremont Football Club players,Australian rules footballers from Western Australia,Living people,West Australian Football Hall of Fame inductees",
    "history": {"revid": 908631551, "timestamp": "2019-07-30T23:31:38Z", "parentid": 824151745, "pre_dump": true, "pageid": 21936854, "url": "https://en.wikipedia.org/w/index.php?title=Darrell%20Panizza&oldid=908631551"},
    "sources": [{"paragraph_id": 1, "start": 0, "end": 377}],
    "section": "Section::::Abstract"}
"""

def get_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data and runs
    parser.add_argument('--kilt_dir', type=str, default='/dfs/scratch1/lorr1/KILT/KILT_data/processed_chunks', help='Where files saved')
    parser.add_argument('--data_dir', type=str, default='/dfs/scratch1/lorr1/KILT/KILT_data/dump', help='Where files saved')
    parser.add_argument('--out_subdir', type=str, default='curate_aliases', help='Where files saved')
    parser.add_argument('--title_to_qid', type=str, default='/lfs/raiders8/0/lorr1/title_to_all_ids_temp.jsonl')
    parser.add_argument('--wd_aliases', type=str, default='/lfs/raiders8/0/lorr1/augmented_alias_map_large_0913.jsonl',
                        help='Path to directory with JSONL mapping alias to QID')
    parser.add_argument('--min_frequency', type=int, default=2, help='Minimum number of times a QID must appear with an alias')
    parser.add_argument('--test', action='store_true', help='If set, will only generate for one file.')
    parser.add_argument('--processes', type=int, default=int(32))
    return parser


def convert_title(title):
    return html.unescape(unquote(title))


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
    Each subprocess launches over a different jsonl file where each line corresponds to a Wikipedia paragraph from Kilt.
    We iterate through each alias in each sentence and track how frequently aliases and titles coccur.
    """
    i, total, args, out_dir, in_filepath = all_args
    hashed_outfilename = prep_utils.get_outfname(in_filepath, ending="json")
    outfilename = os.path.join(out_dir, os.path.splitext(hashed_outfilename)[0] + "_anchoraliases.json")
    boldalias_outfilename = os.path.join(out_dir, os.path.splitext(hashed_outfilename)[0] + "_boldaliases.json")
    print(f"Starting {i}/{total}. Reading in {in_filepath}. Ouputting to {outfilename}")
    aliases_to_title = defaultdict(lambda: defaultdict(int))
    boldaliases_to_title = defaultdict(lambda: defaultdict(int))
    with open(in_filepath, 'r', encoding="utf-8") as in_file:
        for line in in_file:
            line_idx, page_obj = line.split("\t")
            page_obj = json.loads(page_obj)
            for alias in page_obj.get("bold_aliases", []):
                alias = prep_utils.get_lnrm(alias)
                if len(alias) > 0:
                    boldaliases_to_title[alias][page_obj["page_title"]] += 1
            # aliases is a list of sentences with aliases, their gold wikipedia page title, the text, and spans
            for anchor_obj in page_obj["anchors"]:
                alias = anchor_obj["text"]
                title = anchor_obj["href"]
                # Some texts point to the subsection, demarkated by the #; We want the first part
                new_title = convert_title(title).split("#")[0].strip()
                # normalize alias
                alias = prep_utils.get_lnrm(alias)
                if len(alias) > 0:
                    aliases_to_title[alias][new_title] += 1
    utils.dump_json_file(outfilename, aliases_to_title)
    utils.dump_json_file(boldalias_outfilename, boldaliases_to_title)
    print(f"{i} Dumped {len(aliases_to_title)}")
    return


def filter_aliases_and_convert_to_qid(anchoraliases_to_title, boldaliases_to_title, title_to_qid, qid_to_all_titles, args):
    """ 
        1. We walk through each anchor alias-title pair and keep the ones that appear a minimum of two times.
        2. We union these with the bold alias-title pairs where the bold aliases appear in the first few sentences of a Wikipedia page
        3. We map each title to a wikidata QID. If a title doesn't have a mapping, we discard it.

    We build a map of aliases-titles which meet our frequency requirement AND for which the title maps
    to a QID. We save these to file.    
    """
    # aliases_to_title is {alias: {title: count}}
    total_pairs_anchor = sum([len(title_dict) for title_dict in anchoraliases_to_title.values()])
    total_pairs_bold = sum([len(title_dict) for title_dict in boldaliases_to_title.values()])
    vars(args)["original_anchor_alias_entity_pair_count"] = total_pairs_anchor
    vars(args)["original_bold_alias_entity_pair_count"] = total_pairs_bold
    print(f"Extracted {total_pairs_anchor} anchor alias-title pairs and {total_pairs_bold} bold alias-title pairs from Wikipedia. Filtering anchor alias-title pairs by frequency.")

    # track important aliases/qids/etc
    filtered_aliasqid = defaultdict(lambda: defaultdict(int))  # The set of alias-QID pairs whose frequency >= args.min_frequency
    filtered_qids = defaultdict(int)  # Entity count of all QIDs remaining
    qid_unavailable = defaultdict(int)  # A dictionary of titles for which we cannot map to a QID
    unpopular_removed = defaultdict(lambda: defaultdict(int))
    for alias, title_dict in tqdm(anchoraliases_to_title.items()):
        for title, count in title_dict.items():
            if count >= args.min_frequency:
                if title in title_to_qid:
                    filtered_aliasqid[alias][title_to_qid[title]] += 1
                    filtered_qids[title_to_qid[title]] += 1
                # The ELSE happens when we link to an external non-wikipedia page, the wikipedia page doesn't have a QID (rare), or the title is a redirect
                # eg https://en.wikipedia.org/wiki/Vidyodaya_University redirects to https://en.wikipedia.org/wiki/University_of_Sri_Jayewardenepura
                else:
                    qid_unavailable[title] += count
            else:
                unpopular_removed[alias][title] += count
    # union with bold aliases, incrementing count again as the bold aliases link to the page they are on
    for alias, title_dict in tqdm(boldaliases_to_title.items()):
        for title, count in title_dict.items():
            if title in title_to_qid:
                filtered_aliasqid[alias][title_to_qid[title]] += 1
                filtered_qids[title_to_qid[title]] += 1
            else:
                qid_unavailable[title] += count
    # we increment the count here to represent that each page links to itself; as we compute counts later, this is just for sorting
    for qid in qid_to_all_titles:
        for title in qid_to_all_titles[qid]:
            alias = prep_utils.get_lnrm(title)
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
    if len(unpopular_removed) > 0:
        total_titles_discarded = len(set.union(*[set(alias_dict.keys()) for alias_dict in unpopular_removed.values()]))
    else:
        total_titles_discarded = 0
    vars(args)["total_unpopular_titles_discarded"] = total_titles_discarded
    print(f"{total_aliases_discarded} total titles discarded from freq.")
    return filtered_aliasqid, filtered_qids, qid_unavailable, unpopular_removed


def merge_wikidata_aliases(args, aliases_to_qid, all_qids, wikidata_alias_to_qid):
    """ We merge alias-qid pairs from Wikidata with the alias-qid pairs we've 
    extracted from Wikipedia.
    """

    print("Adding wikidata aliases...")
    stats = defaultdict(int)
    new_qids_from_wikidata = set()
    for wd_alias, qids in tqdm(wikidata_alias_to_qid.items()):
        wd_alias = prep_utils.get_lnrm(wd_alias)
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
    return aliases_to_qid, new_qids_from_wikidata


def main():
    gl_start = time.time()
    multiprocessing.set_start_method("spawn")
    args = get_arg_parser().parse_args()
    print(json.dumps(vars(args), indent=4))
    utils.ensure_dir(args.data_dir)

    print(f"Loading data from {args.kilt_dir}...")
    files = glob.glob(f"{args.kilt_dir}/out_*")
    if args.test:
        files = files[:1]

    # Get outdirectory
    outdir = prep_utils.get_outdir(args.data_dir, args.out_subdir, remove_old=True)

    # Store intermediate results
    temp_outdir = prep_utils.get_outdir(args.data_dir, "_temp", remove_old=True)

    print(f"Loaded {len(files)} files.")
    print(f"Launching subprocess with {args.processes} processes...")
    launch_subprocess(args, temp_outdir, files)
    print("Finished subprocesses. Loading title maps.")

    # load wikidata qid-to-title map
    title_to_qid, qid_to_all_titles, _, _ = prep_utils.load_qid_title_map(args.title_to_qid)
    # Aggregate alias-title counts from list and filter.
    print(f'Reading in {len(glob.glob(f"{temp_outdir}/*_anchoraliases.json"))} json dumps')
    list_of_anchoraliases_to_titles = [utils.load_json_file(f) for f in glob.glob(f"{temp_outdir}/*_anchoraliases.json")]
    list_of_boldaliases_to_titles = [utils.load_json_file(f) for f in glob.glob(f"{temp_outdir}/*_boldaliases.json")]
    print("Aggregating list of dictionaries.")
    anchoraliases_to_title = prep_utils.aggregate_list_of_nested_dictionaries(list_of_anchoraliases_to_titles)
    boldaliases_to_title = prep_utils.aggregate_list_of_nested_dictionaries(list_of_boldaliases_to_titles)

    # filter aliases and convert to QID
    aliases_to_qid, all_qids, qid_unavailable, unpopular_removed = filter_aliases_and_convert_to_qid(anchoraliases_to_title, boldaliases_to_title,
                                                                                                     title_to_qid, qid_to_all_titles, args)
    for al in aliases_to_qid:
        assert len(al) > 0

    # load wikidata aliases; each line is json if alias: qid value
    wikidata_alias_to_qid = {}
    if args.wd_aliases != "":
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

    out_file = os.path.join(outdir, "alias_to_qid_filter.json")
    utils.dump_json_file(out_file, aliases_to_qid_merged)
    vars(args)["out_aliases_to_qid_filter_file"] = out_file

    out_file = os.path.join(outdir, "qids_added_wikidata.json")
    utils.dump_json_file(out_file, new_qids_from_wikidata)
    vars(args)["out_qids_added_wikidata_file"] = out_file

    out_file = os.path.join(outdir, "alias_to_qid_wikipedia.json")
    utils.dump_json_file(out_file, aliases_to_qid)
    vars(args)["out_aliases_to_qid_wikipedia_file"] = out_file

    out_file = os.path.join(outdir, "title_with_unknown_qid.json")
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
