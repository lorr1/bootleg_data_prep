'''
This file

1. Extracts data from raw msnbc files 
2. Maps each wikipedia title to a QID
3. Saves data

Sample command to generate MSNBC sentence version: 
python3.6 -m contextual_embeddings.bootleg_data_prep.benchmarks.msnbc.build_msnbc_data


Sample command to generate MSNBC document version: 
python3.6 -m contextual_embeddings.bootleg_data_prep.benchmarks.msnbc.build_msnbc_data --scope document
'''
import xmltodict
import ujson as json
import sys
import os
sys.path.append(os.path.join(sys.path[0], "../"))
import re
import numpy as np
import argparse
import gzip
import shutil
import unicodedata
import requests 
import re 

import bootleg_data_prep.utils.data_prep_utils as prep_utils


TITLE_MAP = {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='data/msnbc', help='Path to file')
    parser.add_argument('--sub_dir', type=str, default='unfiltered', help='Path to file')
    parser.add_argument('--output_format', type = str, default = 'jsonl', choices = ['jsonl', 'txt'])

    # Parameters to set! Make sure the output directory reflects this. 
    parser.add_argument('--scope', type=str, default='sentence', choices=['sentence', 'document'], help = 'Scope to consider when disambiguating mentions (e.g. at the document level, at the sentence level, etc)')

    # Paths to supplementary files -- do not adjust 
    parser.add_argument('--title_to_qid', type=str, default='/lfs/raiders8/0/lorr1/title_to_all_ids.jsonl')
    parser.add_argument('--dataset', type=str, default='/dfs/scratch0/lorr1/bootleg-data-prep/bootleg_data_prep/benchmarks/msnbc/raw/', help='Path to file')
    parser.add_argument('--redirect_map', type=str, default='/dfs/scratch0/lorr1/bootleg-data-prep/bootleg_data_prep/benchmarks/msnbc/raw/title_redirect.json', help='Path to file')

    args = parser.parse_args()
    return args

def get_title_from_url(url):
    return url.replace("http://en.wikipedia.org/wiki/", "").replace("_", " ")

class Sentence:

    def __init__(self, sentence, sent_idx = None, sent_idx_unq = None):
        self.aliases = [] 
        self.spans = []
        self.gold = []
        self.qids = []
        self.sentence = sentence.strip()
        self.sent_idx = None 
        self.sent_idx_unq = None 
    
    def add_alias(self, alias, span, qid, gold = True):
        alias = prep_utils.get_lnrm(alias, strip=True, lower=True)
        self.aliases.append(alias)
        self.spans.append(span)
        self.qids.append(qid)
        self.gold.append(gold)
    
    def extend_sentence(self, sentence): 
        if len(self.sentence) == 0:
            self.sentence = sentence.strip()
        else:
            self.sentence = self.sentence + ' . ' + sentence.strip()
    
    def to_json(self): 
        #assert(self.sent_idx is not None)
        #assert(self.sent_idx_unq is not None)
        return json.dumps({
            'aliases': self.aliases, 
            'spans': self.spans, 
            'gold': self.gold,
            'qids': self.qids, 
            'sentence': self.sentence, 
            'sent_idx': self.sent_idx, 
            'sent_idx_unq': self.sent_idx_unq
        })


class QIDMapper:

    def __init__(self, title2qid, wpid_to_qid, args):
        self.args = args
        self.title2qid = title2qid # maps wikipedia page title to QID 
        self.wpid_to_qid = wpid_to_qid # maps wikipedia page ID to QID

        self.title_redirect = {}
        if os.path.exists(args.redirect_map):
            self.title_redirect = json.load(open(os.path.join(args.dataset, args.redirect_map)))

    def get_qid_from_url(self, url): 
        title = get_title_from_url(url)

        # Get QID based on title 
        if title in self.title2qid:
            return self.title2qid[title]
        else: 
            wpid = self.get_page_id(title)
            if str(wpid) in self.wpid_to_qid:
                return self.wpid_to_qid[str(wpid)]

        print(f"No QID for {title} ({url})")
        return None
    
    def get_page_id(self, title):
        
        if title in self.title_redirect: 
            return self.title_redirect[title]
        
        # get id 
        url_title = requests.utils.quote(title)
        query = requests.get(r'https://en.wikipedia.org/w/api.php?action=query&titles={}&&redirects&format=json'.format(url_title))
        try:
            data = json.loads(query.text)
            wpid = str(list(data['query']['pages'].keys())[0])
            self.title_redirect[title] = wpid
        except: 
            wpid = None 
        return wpid 


    def dump(self): 
        with open(os.path.join(self.args.dataset, self.args.redirect_map), 'w') as out_file: 
            json.dump(self.title_redirect, out_file)



def extract_references(fpath):
    with open(fpath) as in_file: 
        doc = xmltodict.parse(in_file.read())
    references = doc['ReferenceProblem']['ReferenceInstance']
    return references 

def get_text(fpath):
    with open(fpath) as in_file:
        lines = [line.replace('\n', ' ') for line in in_file]
    return lines

def remove_punctuation(sent):
    for x in [',', '.', '’', '?', '-', ';', '\'', "\"", "“", "”", "(", ")", "‘", "â€™", "`"]:
        sent = sent.replace(x, ' ')
    return sent


def get_span(alias, sent):
    alias = remove_punctuation(alias).split()
    sent = remove_punctuation(sent).split()
    for i in range(len(sent)):
        if sent[i:i + len(alias)] == alias:
            sent[i:i + len(alias)] = ['|' for _ in range(len(alias))]
            return [i, i+len(alias)], ' '.join(sent).strip()
    print(f"Error. Alias {alias} not found in sentence {sent}")
    return None, None

def create_sentences(references, text, qm):
    sentences = []
    total_len = 0 
    reference_index = 0
    for line in text: 
        if len(line) == 0:
            continue
        total_len += len(line)
        if len(line.strip()) == 0:
            continue
        line = remove_punctuation(line)
        s = Sentence(line)
        sent_to_mine = line.strip()
        if (reference_index < len(references)) and (int(references[reference_index]['Offset']) < total_len):
            while int(references[reference_index]['Offset']) < total_len:
                alias = references[reference_index]['SurfaceForm']
                #alias = split_punctuation(alias)
                wiki_url = references[reference_index]['ChosenAnnotation']
                qid = qm.get_qid_from_url(wiki_url)
                span, new_sent = get_span(alias, sent_to_mine)
                if qid is not None and span is not None:
                    assert span[1] <= len(line.split()), f"{span}, {alias}, {sent_to_mine.split()}"
                    s.add_alias(alias.lower(), span, qid)
                reference_index += 1
                sent_to_mine = new_sent
                if reference_index == len(references):
                    break
        else:
            pass
            #print(f"No alias in line: {line}")
        sentences.append(s)
        #print("\n\n")
    return sentences

def create_document(references, text, qm):
    s = Sentence("")
    total_len = 0 
    reference_index = 0
    for line in text: 
        if len(line) == 0:
            continue 
        total_len += len(line)
        if len(line.strip()) == 0:
            continue 
        line = remove_punctuation(line)
        s.extend_sentence(line)
        sent_to_mine = line.strip()
        if (reference_index < len(references)) and (int(references[reference_index]['Offset']) < total_len):
            while int(references[reference_index]['Offset']) < total_len: 
                alias = references[reference_index]['SurfaceForm']
                #alias = split_punctuation(alias)
                wiki_url = references[reference_index]['ChosenAnnotation']
                qid = qm.get_qid_from_url(wiki_url)
                span, new_sent = get_span(alias, sent_to_mine)
                if qid is not None and span is not None:
                    assert span[1] <= len(line.split()), f"{span}, {alias}, {sent_to_mine.split()}"
                    s.add_alias(alias.lower(), span, qid)
                reference_index += 1
                sent_to_mine = new_sent
                if reference_index == len(references):
                    break
        else:
            pass
            print(f"No alias in line: {line}")
        #print("\n\n")
    return [s]

def load_and_dump_sentences(args, qm):
    ''' Load sentences from fp ''' 

    documents = os.listdir(os.path.join(args.dataset, 'Problems'))
    sentences = []
    for document in sorted(documents): 
        print(f"{document}")
        references = extract_references(os.path.join(args.dataset, 'Problems', document))
        text = get_text(os.path.join(args.dataset, 'RawTextsSimpleChars_utf8', document))
        if args.scope == 'document':
            sentences.extend(create_document(references, text, qm)) 
        else:
            sentences.extend(create_sentences(references, text, qm))

    total_references = 0
    all_qids = []
    for i in range(len(sentences)):
        sentences[i].sent_idx = str(i)
        sentences[i].sent_idx_unq = str(i)
        total_references += len(sentences[i].aliases)
        all_qids.extend(sentences[i].qids)
    print(f"{len(documents)} total documents. {total_references} total references. {len(sentences)} total sentences.")
    
    out_dir = args.out_dir
    if args.scope == 'document':
        out_dir = f"{out_dir}_document"
    fdir = prep_utils.get_outdir(out_dir, args.sub_dir)
    with open(os.path.join(fdir, 'test.jsonl'), 'w') as out_file: 
        for sentence in sentences:
            out_file.write(sentence.to_json() + '\n')
    
    with open(os.path.join(out_dir, 'msnbc_qids.json'), 'w') as out_file: 
        json.dump(list(all_qids), out_file)


def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    
    title_to_qid, qid_to_all_titles, wpid_to_qid, qid_to_title = prep_utils.load_qid_title_map(args.title_to_qid)
    # title_to_qid = {}
    # wpid_to_qid = {}
    qm = QIDMapper(title_to_qid, wpid_to_qid, args)
    load_and_dump_sentences(args, qm)
    qm.dump()


if __name__ == '__main__':
    main()