'''
This file 

1. Reads in wikidata and wikipedia tags and generates a list of wikipedia

{"qid": "Q6199", "title": "Anar-kisim", "id": "12", "wikidata_title": "anarchism", "wikipedia_title": "Anarchism"}
{"id": "25", "title": "Autistic traits", "qid": "Q38404", "wikidata_title": "autism", "wikipedia_title": "Autism"}
...
where title is some redirect, wikipedia, or wikidata title that at one point linked to the QID. ID is the wikipedia id.

to run: 
python3.6 -m processor.get_title_to_ids

'''
import os, argparse, time, html, json, jsonlines, copy
from glob import glob
from urllib.parse import unquote

from tqdm import tqdm

from collections import defaultdict
import simple_wikidata_db.utils as utils
import dateutil.parser

from bootleg_data_prep.language import ENSURE_ASCII


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str, default = '/lfs/raiders8/0/lorr1/wikidata', help = 'path to output directory')
    parser.add_argument('--wikipedia_xml', type = str, default = '/lfs/raiders8/0/lorr1/enwiki-20201020-pages-articles-multistream.xml', help = 'path to xml wikipedia')
    parser.add_argument('--total_wikipedia_xml_lines', type = int, default = 1132098881, help = "For reporting progress of wikipedia xml readining (optional)")
    parser.add_argument('--wikipedia_pageids', type = str, default = '/lfs/raiders8/0/lorr1/pageids/AA', help = 'path to output directory')
    parser.add_argument('--out_dir', type = str, default = '/lfs/raiders8/0/lorr1/title_mappings', help = 'path to output directory')
    return parser 


def convert_title(title):
    return html.unescape(unquote(title))

def read_in_saved_title_file(title_file, total_lines):
    wikititle2item = defaultdict(set)
    with jsonlines.open(title_file, 'r') as in_file:
        for items in tqdm(in_file, total=total_lines, desc="Reading in title file"):
            # the title is the url title that may be redirected to another wikipedia page
            qid, title, wikidata_title, wikipedia_title, wpid = items['qid'], items['title'], items['wikidata_title'], items['wikipedia_title'], items['id']
            # convert to string for hashing
            wikititle2item[wikipedia_title].add(json.dumps(items, default=str, ensure_ascii=ENSURE_ASCII))
    return wikititle2item

def merge_title_mappings(output_file, qid_to_wikidata, wikipedia_titles_to_qid, wikipedia_titles_to_wpid):
    total_lines = 0
    with open(output_file, "w") as out_f:
        for wikipedia_title, qids in tqdm(wikipedia_titles_to_qid.items(), desc="Merging dicts"):
            if len(qids) > 1:
                print(f"Wikipedia {wikipedia_title} has qid {qids}")
            wpid = wikipedia_titles_to_wpid.get(wikipedia_title, "-1")
            for qid in qids:
                if qid not in qid_to_wikidata:
                    print(f"QID {qid} not in wikidata qids")
                    wikidata_titles = {wikipedia_title.lower()}
                else:
                    wikidata_titles = qid_to_wikidata[qid]
                for wikidata_title in wikidata_titles:
                    for title in [wikipedia_title]:
                        res = {
                            "qid": qid,
                            "id": wpid,
                            "title": title,
                            "wikidata_title": wikidata_title,
                            "wikipedia_title": wikipedia_title
                        }
                        out_f.write(json.dumps(res, default=str, ensure_ascii=ENSURE_ASCII) + "\n")
                        total_lines += 1
    return total_lines

def read_in_redirects(in_filename, total_lines):
    title_map = defaultdict(set)
    with open(in_filename, "r") as in_f:
        cur_title = ""
        redirect_title = ""
        cur_timestamp = None
        for line in tqdm(in_f, total=total_lines, desc="Reading in redirects"):
            line = line.strip()
            if line.startswith("<title>"):
                cur_title = line.replace("<title>", "").replace("</title>", "")
                if cur_timestamp is None and len(redirect_title) > 0:
                    print("CUR TITLE", cur_title, "WITH", redirect_title, "HAD A NONE TIMESTAMP")
                redirect_title = ""
                cur_timestamp = None
            if line.startswith("<redirect title"):
                redirect_title = line.replace("<redirect title=\"", "").replace("\" />", "")
            if line.startswith("<timestamp"):
                cur_timestamp = dateutil.parser.isoparse(line.replace("<timestamp>", "").replace("</timestamp>", ""))
                title_map[redirect_title].add(tuple([cur_timestamp, cur_title]))
    new_dict = {}
    for k in tqdm(title_map, desc="Converting to lists"):
        new_dict[k] = list(title_map[k])
    return new_dict

def add_redirects(title_map, wikititle2item):
    wikititle2item_save = copy.deepcopy(wikititle2item)
    for good_title in tqdm(title_map, desc="Adding redirects"):
        good_title_converted = convert_title(good_title)
        if good_title not in wikititle2item_save and good_title_converted not in wikititle2item_save:
            print(f"Good {good_title} (or {good_title_converted}) not in wikititle2item")
            continue
        if good_title in wikititle2item_save:
            items = wikititle2item_save[good_title]
        else:
            items = wikititle2item_save[good_title_converted]
        for redirect_title_pair in title_map[good_title]:
            new_item = {}
            # print(good_title, "R:", redirect_title, list(wikititle2item[good_title])[0], "---", wikititle2item[good_title])
            item_to_copy = json.loads(list(items)[0])
            for k in item_to_copy:
                new_item[k] = item_to_copy[k]
            new_item["title"] = redirect_title_pair[1]
            wikititle2item[good_title].add(json.dumps(new_item, default=str, ensure_ascii=ENSURE_ASCII))
    return wikititle2item, wikititle2item_save


def read_in_wikipedia_pageids(args):
    wikipedia_files = glob(os.path.join(args.wikipedia_pageids, "*"))
    title_to_id = {}
    for file in tqdm(wikipedia_files, desc="Reading in wikipedia files"):
        with open(file, "r") as in_f:
            for line in in_f:
                line = json.loads(line)
                if line["title"] in title_to_id:
                    print(f"WARNING", line["title"], f"already has a wikipedia id of", title_to_id[line["title"]], f"You are adding", line["id"], "instead")
                title_to_id[line["title"]] = line["id"]
    return title_to_id


def read_in_wikipedia_title(args):
    fdir = os.path.join(args.data, "processed_batches", "wikipedia_links")
    wikipedia_files = utils.get_batch_files(fdir)
    title_to_id = defaultdict(set)
    for file in tqdm(wikipedia_files, desc="Reading in wikipedia files"):
        with open(file, "r") as in_f:
            for line in in_f:
                line = json.loads(line)
                title_to_id[line["wiki_title"]].add(line["qid"])
    return title_to_id

def read_in_wikidata_title(args):
    fdir = os.path.join(args.data, "processed_batches", "labels")
    wikidata_files = utils.get_batch_files(fdir)
    id_to_title = defaultdict(set)
    for file in tqdm(wikidata_files, desc="Reading in wikidata files"):
        with open(file, "r") as in_f:
            for line in in_f:
                line = json.loads(line)
                id_to_title[line["qid"]].add(line["label"])
    return id_to_title

def main():
    start = time.time()
    args = get_arg_parser().parse_args()

    out_dir = os.path.join(args.data, args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # This will NOT contain redirects from Wikidata
    output_file = os.path.join(out_dir, "title_to_all_ids_raw.jsonl")
    total_lines = 16849876

    wikipedia_titles_to_qid = read_in_wikipedia_title(args)
    wikipedia_titles_to_wpid = read_in_wikipedia_pageids(args)
    qid_to_wikidata = read_in_wikidata_title(args)

    total_lines = merge_title_mappings(output_file, qid_to_wikidata, wikipedia_titles_to_qid, wikipedia_titles_to_wpid)

    redirect_title_map = read_in_redirects(args.wikipedia_xml, args.total_wikipedia_xml_lines)
    assert len(redirect_title_map) > 0
    out_file = "temp_redicts.json"
    with open(out_file, "w") as out_f:
        json.dump(redirect_title_map, out_f, default=str, ensure_ascii=ENSURE_ASCII)
    # with open(out_file, "r") as in_f:
    #     redirect_title_map = ujson.load(in_f)

    # Add redirects from Wikidata
    wikititle2item = read_in_saved_title_file(output_file, total_lines)
    wikititle2item, wikititle2item_original = add_redirects(redirect_title_map, wikititle2item)

    output_file = os.path.join(out_dir, "title_to_all_ids.jsonl")
    with jsonlines.open(output_file, 'w') as out_file:
        for good_title in wikititle2item:
            for item in wikititle2item[good_title]:
                out_file.write(json.loads(item))

    print(f"Finished in {time.time() - start}")

if __name__ == "__main__":
    main()