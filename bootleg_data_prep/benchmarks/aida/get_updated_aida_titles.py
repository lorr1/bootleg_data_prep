'''

AIDA uses wikipedia page titles from 2004. Our Wikipedia dumps are from Fall 2019. Some pages have changed titles, been deleted, etc. We need to normalize the AIDA dataset by updating the 2004 page titles
with the 2019 versions. We have to ensure that the pages match on page ID. To do this, we build a map of 2004_title-->page_id for AIDA, and a map of page_id --> 2019_title for our 2019 dump.

We output save our results to file, so we should only need to run this once. 

''' 

import os
import sys

import ujson as json

sys.path.append(os.path.join(sys.path[0], "../"))
from bootleg.utils import utils
from nltk.tokenize import RegexpTokenizer
import argparse

tokenizer = RegexpTokenizer(r'\w+')
SUB_DELIM = "~*~"
SUPER_DELIM = "|"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pageid_dir', type=str, default='/lfs/raiders7/0/lorr1/pageidssave/', help='Directory of pageids from WikiExtractor.py.')
    parser.add_argument('--save_dir', type=str, default='data/aida/utils', help='Directory for data to be saved.')
    parser.add_argument('--aida_map', type=str, default='data/aida/AIDA-YAGO2-annotations.tsv', help = 'Map to AIDA annotations TSV')
    parser.add_argument('--redirects_map', type=str, default='data/aida/redirects_map.tsv', help = 'Map of URL redirects collected using wikipedia API')
    args = parser.parse_args()
    return args

def read_current_pages(args):
    pageid_files = []
    for direc in os.listdir(args.pageid_dir):
        for file in os.listdir(os.path.join(args.pageid_dir, direc)):
            pageid_files.append(os.path.join(args.pageid_dir, direc, file))

    pageid_to_title = {}
    title_to_pageid = {}
    for file in pageid_files:
        with open(file, "r") as reading_f:
            for line in reading_f:
                data = json.loads(line)
                pageid_to_title[data['id']] = data['title']
                title_to_pageid[data['title']] = data['id']
    return pageid_to_title, title_to_pageid

def load_aida_title_map(args):
    title_to_pageid = {}
    collisions = 0
    with open(args.aida_map, 'r') as in_file:
        for line in in_file:
            items = line.strip().split("\t")
            if len(items) > 3: 
                title = items[2].replace('http://en.wikipedia.org/wiki/', '')
                page_id = items[3]
                if title in title_to_pageid and not title_to_pageid[title] == page_id:
                    print(f"DUPLICATE ENTRY DETECTED. Attempting to replace {title_to_pageid[title]} with {page_id} for {title}")
                    collisions += 1
                title_to_pageid[title] = page_id
    print(f"{collisions} collisions detected during construction.")
    print(f"loaded {len(title_to_pageid)} title-id pairs.")
    print(f"{len(set(title_to_pageid.values()))} distinct entities.")
    return title_to_pageid

def get_redirects_map(args):

    old_title_to_new_title = {}
    with open(args.redirects_map, 'r') as in_file:

        for line in in_file:
            items = line.strip().split("\t")
            old_title = items[0].replace('http://en.wikipedia.org/wiki/', '').replace('_', ' ')
            old_title_to_new_title[old_title] = items[1]
    return old_title_to_new_title


def build_map(args):

    # load map of page_id --> title for 2019 Wikipedia Dump
    current_wpid2title, current_title2pageid = read_current_pages(args)

    # load map of title --> page_id for 2004 AIDA dataset
    aida_title_to_pageid = load_aida_title_map(args)

    # load map of title redirects. For each page in the wikipedia dataset, we used the 
    # wikimedia API to collect the title of the page it currently redirects to. We use this 
    # as a lost resort. 
    old_title_to_new_title = get_redirects_map(args)

    updated_title_map = {} # map old AIDA title to updated wikipedia titles 
    num_changes = 0 
    num_deleted = 0 
    for title, pageid in aida_title_to_pageid.items():
        
        # convert "European_Commission" -> "European Commission"

        title = title.replace('_', ' ')
        
        # 1. Check if the page id exists in 
        if pageid in current_wpid2title:
            new_title = current_wpid2title[pageid]
            updated_title_map[title] = new_title 
            if not new_title  == title: 
                num_changes += 1 

        # 2. Check if the page title exists (maybe the ID has changed, or become defunct.)
        elif title in current_title2pageid:
            # just stick with the current title 
            updated_title_map[title] = title 
        else:
            new_title = old_title_to_new_title[title]
            if new_title == title or not new_title in current_title2pageid:
                num_deleted += 1
                print(f"Couldn't find page for {title}. Page ID: {pageid}")    
            else:
                updated_title_map[title] = new_title 
            
    
    # print some interesting statistics 
    print(f"Detected {num_changes} pages which have changed titles.")
    print(f"Detected {num_deleted} titles for which pages no longer exist")

    # save to file 
    out_fpath = os.path.join(args.save_dir, 'aida_title_map.txt')
    with open(out_fpath, 'w') as out_file:
        for aida_title, new_title in updated_title_map.items():
            out_file.write(f"{aida_title}\t{new_title}\n")

    print(f"Written updated map to {out_fpath}")


def main():
    args = parse_args()
    utils.ensure_dir("{:s}/".format(args.save_dir))
    build_map(args)

if __name__ == '__main__':
    main()
