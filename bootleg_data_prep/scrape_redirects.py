from collections import defaultdict
import ujson, jsonlines
from tqdm import tqdm
from urllib.parse import unquote
import html
import copy

from bootleg_data_prep.language import ENSURE_ASCII


def convert_title(title):
    return html.unescape(unquote(title))

title_file = "/lfs/raiders8/0/lorr1/title_to_all_ids.jsonl"
wikititle2item = defaultdict(set)
with jsonlines.open(title_file, 'r') as in_file:
    for items in tqdm(in_file, total=6457844):
        # the title is the url title that may be redirected to another wikipedia page
        qid, title, wikidata_title, wikipedia_title, wpid = items['qid'], items['title'], items['wikidata_title'], items['wikipedia_title'], items['id']
        # convert to string for hashing
        wikititle2item[wikipedia_title].add(ujson.dumps(items, ensure_ascii=ENSURE_ASCII))

wikititle2item_save = copy.deepcopy(wikititle2item)
in_filename = "enwiki-latest-pages-articles-multistream.xml"
title_map = defaultdict(set)
with open(in_filename, "r") as in_f:
    cur_title = ""
    redirect_title = ""
    for line in tqdm(in_f, total=1132098881):
        line = line.strip()
        # Title is the old title, redirect is what the page redirects to (good title)
        if line.startswith("<title>"):
            cur_title = line.replace("<title>", "").replace("</title>", "")
        if line.startswith("<redirect title"):
            redirect_title = line.replace("<redirect title=\"", "").replace("\" />", "")
            title_map[redirect_title].add(cur_title)


new_dict = {}
for k in tqdm(title_map):
    new_dict[k] = list(title_map[k])


out_file = "temp_redicts.json"
with open(out_file, "w") as out_f:
    ujson.dump(new_dict, out_f, ensure_ascii=ENSURE_ASCII)


for good_title in title_map:
    good_title_converted = convert_title(good_title)
    if good_title not in wikititle2item_save and good_title_converted not in wikititle2item_save:
        print(f"Good {good_title} (or {good_title_converted}) not in wikititle2item")
        continue
    if good_title in wikititle2item_save:
        items = wikititle2item_save[good_title]
    else:
        items = wikititle2item_save[good_title_converted]
    for redirect_title in title_map[good_title]:
        new_item = {}
        # print(good_title, "R:", redirect_title, list(wikititle2item[good_title])[0], "---", wikititle2item[good_title])
        item_to_copy = ujson.loads(list(items)[0])
        for k in item_to_copy:
            new_item[k] = item_to_copy[k]
        new_item["title"] = redirect_title
        wikititle2item[good_title].add(ujson.dumps(new_item, ensure_ascii=ENSURE_ASCII))



title_file2 = "/lfs/raiders8/0/lorr1/title_to_all_ids_temp.jsonl"
with jsonlines.open(title_file2, 'w') as out_file:
    for good_title in wikititle2item:
        for item in wikititle2item[good_title]:
            out_file.write(ujson.loads(item))

