'''
python3 -m contextual_embeddings.bootleg_data_prep.benchmarks.rss500.build_rss500_dataset
'''
from urllib.parse import unquote
import ujson as json
from tqdm import tqdm
import sys
import os
import re
from collections import defaultdict
import html
import rdflib
import argparse
from nltk import word_tokenize
from jsonlines import jsonlines
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, JSON

import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg.utils import utils

NO_WIKI_ID = "-2"
manual_mappings = {
    "http://dbpedia.org/resource/The_Charleston_Gazette": "Q5084133",
    "http://dbpedia.org/resource/San_Francisco_%E2%80%93_Oakland_Bay_Bridge": "Q771404",
    "http://dbpedia.org/resource/St._Joseph_Medical_Center": "-1",
    "http://dbpedia.org/resource/Chad_Johnson_(cornerback)": "Q471833",
    "http://dbpedia.org/resource/Mohr_Davidow_Ventures": "-1",
    "http://dbpedia.org/resource/National_Venture_Capital_Association": "Q6979295",
    "http://dbpedia.org/resource/Bed_Bath_%26_Beyond": "Q813782",
    "http://dbpedia.org/resource/Russia_in_Global_Affairs": "Q3453515"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='/dfs/scratch0/lorr1/bootleg/data/rss500_take2/unfiltered', help='Path to output file')

    # Paths to supplementary files -- do not adjust
    parser.add_argument('--title_to_qid', type=str, default='/lfs/raiders7/0/lorr1/title_to_all_ids.jsonl')
    parser.add_argument('--title_to_qid_lmdb', type=str, default='/dfs/scratch0/lorr1/contextual-embeddings/title2qid_db')
    parser.add_argument('--wpid_to_qid_lmdb', type=str, default='/dfs/scratch0/lorr1/contextual-embeddings/wpid2qid_db')
    parser.add_argument('--dataset', type=str, default='/dfs/scratch0/lorr1/bootleg/bootleg-internal/bootleg_data_prep/benchmarks/rss500/raw_data/rss_500.nt', help='Path to RSS500 file')

    args = parser.parse_args()
    return args

def get_qid(url, title2qid, wpid2qid, sparql):
    if "/notInWiki/" in url:
        return NO_WIKI_ID
    if str(url) in manual_mappings:
        return manual_mappings[str(url)]
    query = "SELECT DISTINCT ?o ?p WHERE {{ <{}> ?o ?p }}"
    temp = query.format(url)
    sparql.setQuery(temp)
    results = sparql.query().convert()
    results = results['results']['bindings']
    possible_qid = "-1"
    for res in results:
        obj = res['o']
        pred = res['p']
        if obj['value'] == 'http://dbpedia.org/ontology/wikiPageID':
            wikiid = pred['value']
            qid = wpid2qid.get(str(wikiid))
            if qid is not None and qid != "-1":
                possible_qid = qid
            else:
                print(f"WPID {wikiid} not found with url {url}")
        if obj['value'] == "http://dbpedia.org/ontology/wikiPageRedirects":
            new_url = pred['value']
            print(f"Trying urls {new_url}")
            temp = query.format(new_url)
            sparql.setQuery(temp)
            results = sparql.query().convert()
            results = results['results']['bindings']
            for res in results:
                obj = res['o']
                pred = res['p']
                if obj['value'] == 'http://dbpedia.org/ontology/wikiPageID':
                    wikiid = pred['value']
                    qid = wpid2qid.get(wikiid)
                    if qid is not None and qid != "-1":
                        possible_qid = qid
                    else:
                        print(f"WPID (take2) {wikiid} not found with url {new_url}")
    if possible_qid != "-1":
        return possible_qid
    # If that doesn't work, let's try query the title directly
    def success(head, body):
        for_map = f"{head} {' '.join(body)}".strip()
        qid = title2qid.get(for_map)
        if not qid or qid == "-1":
            return False, qid
        return True, qid
    print(f"Tying to match exact title {url}")
    # remove html characters
    title = url.rsplit("/")[-1]
    title = unquote(title)
    title_parts = title.split("_")
    head = title_parts[0]

    # try the raw title first
    suc, qid = success(head, title_parts[1:])
    if suc:
        return qid
    # try normalizing it like wikipedia
    body = [t.lower() for t in title_parts[1:]]
    suc, qid = success(head, body)
    if suc:
        return qid
    # try capitalizing words in order
    for to_cap in range(len(body)):
        body[to_cap] = body[to_cap].title()
        suc, qid = success(head, body)
        if suc:
            return qid
    print(f"**TITLE {title} not found with url {url}")
    return "-1"

def num_white_spaces(sentence, char_idx):
    left = sentence[:char_idx+1]
    left_spl = left.split()
    with_sp_len = len(left)
    without_sp_len = len(''.join(left_spl))
    return with_sp_len - without_sp_len


def get_words_in_context(no_wh_sentence, sentence_spl):
    wd_idx = 0
    char_idx = 0
    to_break = False
    for i, w in enumerate(sentence_spl):
        for c in w:
            if char_idx >= len(no_wh_sentence):
                to_break = True
                break
            char_idx += 1
        if to_break:
            break
        wd_idx += 1
    return wd_idx


def convert_dump_to_jsonl(args, parsed_dict, wiki_filter):
    sentences = []
    all_keys = sorted(list(parsed_dict.keys()))
    for key in all_keys:
        # assert there is one char_span with the sentence
        assert sum(['sentence' in parsed_dict[key][i] for i in parsed_dict[key]]) == 1
        aliases = []
        qids = []
        char_spans = []
        sentence = None
        for char_span in parsed_dict[key]:
            if 'sentence' in parsed_dict[key][char_span]:
                sentence = parsed_dict[key][char_span]['sentence']
            else:
                assert 'alias' in parsed_dict[key][char_span]
                assert 'qid' in parsed_dict[key][char_span]
                assert 'start_char' in parsed_dict[key][char_span]
                assert 'end_char' in parsed_dict[key][char_span]
                char_spans.append([parsed_dict[key][char_span]['start_char'], parsed_dict[key][char_span]['end_char']])
                aliases.append(parsed_dict[key][char_span]['alias'])
                qids.append(parsed_dict[key][char_span]['qid'])
        assert sentence is not None, f'Something bad happened with {json.dumps(parsed_dict[key])}'
        sentence_spl = word_tokenize(sentence)
        spans = []
        print("KEY", key)
        print(json.dumps(parsed_dict[key], indent=4))
        for al_idx, alias in enumerate(aliases):
            print("ALIAS", alias, "VS", sentence[char_spans[al_idx][0]:char_spans[al_idx][1]])
            start_sp = get_words_in_context(sentence[:char_spans[al_idx][0]].replace(" ", ""), sentence_spl)
            end_sp = get_words_in_context(sentence[:char_spans[al_idx][1]].replace(" ", ""), sentence_spl)
            if end_sp == start_sp:
                end_sp += 1
                # Happens when some aliases are mislabelled: Japan alias for Japanese
                print("SPANS ARE EQUAL", alias, sentence, char_spans[al_idx])
            if alias.lower().replace(" ", "") != ''.join(sentence_spl[start_sp:end_sp]).lower():
                print("BAD", ' '.join(sentence_spl[start_sp:end_sp]), "VS", alias.lower(), [start_sp, end_sp])
                if not ''.join(sentence_spl[start_sp:end_sp]).lower().startswith(alias.lower()):
                    sys.exit()
            if start_sp == -1:
                print("BAD SPANS!!!")
            spans.append([start_sp, end_sp])
        spans, aliases, qids = zip(*list(sorted(zip(spans, aliases, qids),  key=lambda x: x[0][0])))
        if wiki_filter:
            if all([q == NO_WIKI_ID for q in qids]):
                print(f"Key {key} has all no wiki ids")
                continue
            spans, aliases, qids = zip(*list(filter(lambda x: x[2] != NO_WIKI_ID,zip(spans, aliases, qids))))
        aliases = [' '.join(word_tokenize(a.lower())) for a in aliases]
        sent_obj = {'sent_idx_unq': key,
                    'aliases': aliases,
                    'qids': qids,
                    'spans': spans,
                    'sentence': " ".join(sentence_spl),
                    'gold': [True for _ in range(len(aliases))]}
        print(f"Dumping {key}")
        sentences.append(sent_obj)
    return sentences


def load_and_dump_sentences(args, title2qid, wpid2qid, sparql, g):
    parsed_dict = {}
    all_qids = set()
    for stmt in tqdm(g):
        head, body, tail = stmt
        key = head.split("#")[0]
        if not key.startswith("http://aksw.org/N3/RSS-500/"):
            continue
        # gives character spans of each item
        sort_order = head.split("#")[1].split("char=")[1]
        key = int(key.rsplit('/')[-1])
        if key not in parsed_dict:
            parsed_dict[key] = defaultdict(lambda: defaultdict(str))
        part = body.split("#")[1]
        # print(key, "---", sort_order, "---", stmt)
        if part == "anchorOf":
            assert type(tail) is rdflib.term.Literal
            alias = tail.value
            assert 'alias' not in parsed_dict[key][sort_order], f"Current dict is {parsed_dict[key][sort_order]}"
            parsed_dict[key][sort_order]["alias"] = alias
        elif part == "beginIndex":
            assert type(tail) is rdflib.term.Literal
            start_char = tail.value
            assert 'start_char' not in parsed_dict[key][sort_order], f"Current dict is {parsed_dict[key][sort_order]}"
            parsed_dict[key][sort_order]["start_char"] = start_char
        elif part == "endIndex":
            assert type(tail) is rdflib.term.Literal
            end_char = tail.value
            assert 'end_char' not in parsed_dict[key][sort_order], f"Current dict is {parsed_dict[key][sort_order]}"
            parsed_dict[key][sort_order]["end_char"] = end_char
        elif part == "isString":
            assert type(tail) is rdflib.term.Literal
            sent = tail.value
            assert 'sentence' not in parsed_dict[key][sort_order], f"Current dict is {parsed_dict[key][sort_order]}"
            parsed_dict[key][sort_order]["sentence"] = sent
        elif part == "taIdentRef":
            qid = get_qid(tail, title2qid, wpid2qid, sparql)
            assert 'qid' not in parsed_dict[key][sort_order], f"Current dict is {parsed_dict[key][sort_order]}"
            parsed_dict[key][sort_order]["qid"] = qid
            all_qids.add(qid)
    utils.ensure_dir(args.out_dir)
    utils.dump_json_file(os.path.join(args.out_dir, '../pre-sentenceing.json'), parsed_dict)
    parsed_dict = utils.load_json_file(os.path.join(args.out_dir, '../pre-sentenceing.json'))
    sentences = convert_dump_to_jsonl(args, parsed_dict, True)
    print(f"Saving sentences to {os.path.join(args.out_dir, 'test.jsonl')}")
    with jsonlines.open(os.path.join(args.out_dir, "test.jsonl"), "w") as out_f:
        for sent in sentences:
            out_f.write(sent)

    sentences = convert_dump_to_jsonl(args, parsed_dict, False)
    print(f"Saving sentences to {os.path.join(args.out_dir, '../test_all.jsonl')}")
    with jsonlines.open(os.path.join(args.out_dir, "../test_all.jsonl"), "w") as out_f:
        for sent in sentences:
            out_f.write(sent)

    # dump all QIDS
    # utils.dump_json_file(os.path.join(args.out_dir, "../rss500_qids.json"), all_qids)
    return


def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    title2qid, _, wpid2qid, _ = prep_utils.load_qid_title_map(args.title_to_qid)
    # title2qid = {}
    # wpid2qid = {}
    # utils.ensure_dir(args.out_dir)
    # env1 = lmdb.open(args.title_to_qid_lmdb, map_size=107374182400, readonly=True)
    # txn_title = env1.begin()

    # env2 = lmdb.open(args.wpid_to_qid_lmdb, map_size=107374182400, readonly=True)
    # txn_wpid = env2.begin()
    # sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    g = Graph()
    g.parse(args.dataset, format="nt")
    load_and_dump_sentences(args, title2qid, wpid2qid, sparql, g)
    # env1.close()
    # env2.close()


if __name__ == '__main__':
    main()

'''
select ?s ?o ?p where {
filter( regex(str(?x), "exam" ))
}

import lmdb
import bootleg_data_prep.utils.data_prep_utils as prep_utils
env = lmdb.open('wpid2qid_db', map_size=107374182400)
txn = env.begin(write = True)
_, wpid2qid, _ = prep_utils.load_qid_title_map('/lfs/raiders7/0/lorr1/title_to_all_ids.jsonl', lower_case = False)

for wpid in wpid2qid:
    qid = wpid2qid[wpid]
    if qid == "-1" or qid == -1 or wpid == -1:
        continue
    txn.put(wpid.encode(), qid.encode())
'''