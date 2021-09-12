"""The script queries to try to get as many redirect page titles as possible."""

import ujson as json
import argparse
import requests
import os
import time
import shutil
import html
import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.language import ensure_ascii
from bootleg_data_prep.utils import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/wiki_dump', help='Directory for data to be saved.')
    parser.add_argument('--title_to_qid', type=str, default='/lfs/raiders7/0/lorr1/title_to_all_ids.jsonl', help='Mapping of pageids to and titles to QIDs.')
    parser.add_argument('--alias_to_title_file', type = str, default = 'curate_aliases/wp_aliases_to_title.json')
    args = parser.parse_args()
    return args


def extract(api_query, still_unknown, no_wpid_list, wpid_to_qid, stats, in_f):
    try:
        res = api_query(still_unknown)
        query = res["query"]
        pages = query["pages"]
    except Exception as e:
        print(f"Error {e} \n processing request")
        print("Unknowns", still_unknown)
        print(json.dumps(res, indent=4, ensure_ascii=ensure_ascii))
        return 0

    # If len(still_unknown) > 1, the results returned from the query are out of order; we have to use mappings to get back the original title
    redirect_mapping_inv = {}
    new_items = []
    if "redirects" in query:
        redirects = query["redirects"]
        for redirect in redirects:
            redirect_mapping_inv[redirect["to"]] = redirect["from"]
    normalized_mapping_inv = {}
    if "normalized" in query:
        normalized = query["normalized"]
        for normal in normalized:
            normalized_mapping_inv[normal["to"]] = normal["from"]
    for wpid in pages:
        if int(wpid) < 0:
            stats["no_wpid"] += 1
            no_wpid_list.add(json.dumps(pages[wpid], ensure_ascii=ensure_ascii))
            continue
        title = pages[wpid]["title"]
        orig_title = title
        if title in redirect_mapping_inv:
            orig_title = redirect_mapping_inv[title]
        if orig_title in normalized_mapping_inv:
            orig_title = normalized_mapping_inv[orig_title]
        if orig_title not in still_unknown:
            if len(still_unknown) == 1:
                orig_title = still_unknown[0]
            else:
                # let's try again looping through one at a time
                return 0
        qid = "-1"
        if wpid in wpid_to_qid:
            qid = wpid_to_qid[wpid]
        else:
            stats["no_qid"] += 1
        item = {"qid": qid, "title": orig_title, "id": wpid}
        new_items.append(item)
    for item in new_items:
        in_f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
    time.sleep(1)
    return 1



def main():
    args = parse_args()
    in_file = os.path.join(args.data_dir, args.alias_to_title_file)
    print(f"Reading in {in_file}")
    in_f = utils.load_json_file(in_file)
    all_titles = sorted(list(set([html.unescape(v.strip()).strip() for titles in in_f.values() for v in titles if len(v.strip()) > 0])))


    title_to_qid, wpid_to_qid, qid_to_title = prep_utils.load_qid_title_map(args.title_to_qid, lower_case = False)
    print(f"{len(all_titles)} starting all titles and {len(title_to_qid)} titles to QIDs and {len(wpid_to_qid)} wpids to qids")

    print(f"Starting round 1 filtering by titles we already have")
    still_unknown = []
    for title in all_titles:
        if title not in title_to_qid:
            still_unknown.append(title)

    print(f"{len(still_unknown)} titles are still unknown from {len(all_titles)}")

    new_f = os.path.splitext(args.title_to_qid)[0] + "_appended.jsonl"
    shutil.copy(args.title_to_qid, new_f)
    in_f = open(new_f, "a")
    api_query = lambda x: json.loads(requests.get(r'https://en.wikipedia.org/w/api.php?action=query&titles={}&&redirects&format=json'.format("|".join(x))).text)
    no_wpid_list = set()
    final_still_unknown = set()
    stats = {'no_wpid': 0, 'no_qid': 0}
    offset = 50
    for i in range(0, len(still_unknown), offset):
        print(f"At index {i}")
        res = extract(api_query, still_unknown[i:i+offset], no_wpid_list, wpid_to_qid, stats, in_f)
        print(len(no_wpid_list))
        if res == 0:
            for j in range(i, i+offset):
                res = extract(api_query, still_unknown[i+j:i+j+1], no_wpid_list, wpid_to_qid, stats, in_f)
                if res == 0:
                    final_still_unknown.add(still_unknown[i+j:i+j+1][0])

    in_f.close()

    utils.dump_pickle_file(os.path.join(args.data_dir, "no_wpid_list.pkl"), no_wpid_list)

if __name__ == '__main__':
    main()