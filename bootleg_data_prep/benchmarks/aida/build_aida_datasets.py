'''
This file

1. Extracts data from raw aida files
2. Maps each wikipedia title to a QID
3. Saves data

Sample command to generate aida sentence version:
python3.6 -m bootleg_data_prep.benchmarks.aida.build_aida_datasets

Sample command to generate AIDA document version:
python3.6 -m bootleg_data_prep.benchmarks.aida.build_aida_datasets --scope sentence --include_title --include_aliases_in_prefix
'''

import ujson as json
import sys
import os
sys.path.append(os.path.join(sys.path[0], "../"))
import re
import numpy as np
import argparse
import gzip
import shutil
import copy

import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.utils.constants import PREFIX_DELIM


SUB_DELIM = "~*~"
SUPER_DELIM = "|"
NEW_DOC_DELIM = '-DOCSTART-'
TITLE_MAP = {}

# Delineates the splits between the train, dev (testa), and evaluation set (testb)
DEV_START_DOC = '947testa CRICKET'
TEST_START_DOC = '1163testb SOCCER'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='data/aida', help='Path to AIDA file')
    parser.add_argument('--sub_dir', type=str, default='unfiltered', help='Path to AIDA file')

    # Parameters to set! Make sure the output directory reflects this.
    parser.add_argument('--scope', type=str, default='sentence', choices=['sentence', 'document'], help = 'Scope to consider when disambiguating mentions (e.g. at the document level, at the sentence level, etc)')
    parser.add_argument('--include_title', action='store_true', help = 'Adds title of document as prefix to sentence')
    parser.add_argument('--include_first_sentence', action = 'store_true', help = 'Adds first sentence of document as prefix to sentence.')
    parser.add_argument('--include_aliases_in_prefix', action='store_true', help = 'Whether to include the original aliases from above in the prefix.')

    # Paths to supplementary files -- do not adjust
    parser.add_argument('--title_to_qid', type=str, default='/dfs/scratch1/mleszczy/contextual-embeddings-git/title_to_all_ids.jsonl')
    parser.add_argument('--title_map', type=str, default='bootleg_data_prep/benchmarks/aida/raw_aida/redirects_map.txt', help='Path to file mapping URLs to titles')
    parser.add_argument('--dataset', type=str, default='bootleg_data_prep/benchmarks/aida/raw_aida/AIDA-YAGO2-dataset.txt', help='Path to AIDA file')
    parser.add_argument('--annotated_file', type=str, default='bootleg_data_prep/benchmarks/aida/raw_aida/AIDA-YAGO2-annotations.txt', help='Path to AIDA file')
    args = parser.parse_args()
    return args

def get_title_from_url(url):
    return url.replace("http://en.wikipedia.org/wiki/", "").replace("_", " ")

def get_aida_title_to_wpid(args):
    title2wpid = {}
    with open(args.annotated_file) as in_file:
        # line with WPID and title looks like this:
        # 2	Germany	http://en.wikipedia.org/wiki/Germany	11867	/m/0345h
        for line in in_file:
            items = line.strip().split("\t")
            if len(items) > 3:
                title = get_title_from_url(items[2])
                wpid = items[3]
                title2wpid[title] = wpid
    print(f"Loaded {len(title2wpid)} title-WPID pairs from {args.annotated_file}")
    return title2wpid

def get_title_to_new_title(args):
    title2new_title = {}
    with open(args.title_map) as in_file:
        # line looks like this:
        # http://en.wikipedia.org/wiki/Mike_Conley,_Sr.	Mike Conley Sr.
        for line in in_file:
            url, new_title = line.strip().split("\t")
            old_title = get_title_from_url(url)
            title2new_title[old_title] = new_title
    print(f"Loaded {len(title2new_title)} title redirect pairs from {args.title_map}")
    return title2new_title




class QIDMapper:

    def __init__(self, aida_to_wpid, wpid2qid, title2qid, title2new_title):

        self.aida_to_wpid =  aida_to_wpid # maps AIDA title to Wikipedia page ID
        self.wpid2qid = wpid2qid # maps wpid to QID
        self.title2qid = title2qid # maps wikipedia page title to QID
        self.title2new_title = title2new_title # maps [old] wikipedia page title to new title (frequently result of redirect)

    def get_qid_from_url(self, url):
        title = get_title_from_url(url)

        # Get QID based on WPID
        if title in self.aida_to_wpid:
            wpid = self.aida_to_wpid[title]
            if wpid in self.wpid2qid:
                return self.wpid2qid[wpid]

        # Get QID based on title
        if title in self.title2qid:
            return self.title2qid[title]

        # Get redirected title
        if title in self.title2new_title:
            title = self.title2new_title[title]
            if title in self.title2qid:
                return self.title2qid[title]

        return "None"


def is_all_upper(token):
    if len(token) == 1:
        return False
    return token[1].isupper()

class ScopeGroup:

    def __init__(self, lines, prefix, mapper, args, doc_id, alias_id):
        # is_prefix indicates whether this line has been artificially added. If true, then we should set
        # any entities we detect here to anchor = False.
        self.tokens = []
        self.entities = []
        self.mapper = mapper
        self.lines = lines
        self.prefix = prefix
        self.doc_id = doc_id
        self.alias_idx = alias_id
        assert len(lines) == len(prefix), 'the length of is_prefix and lines do not match'
        self.create_from_word_lines(args)

    def create_from_word_lines(self, args):
        i = 0
        doc_al_id_prefix = 0
        doc_al_id_not_prefix = self.alias_idx
        while i < len(self.lines):
            line = self.lines[i]
            assert len(self.lines[i]) > 0
            items = line.split("\t")
            if len(items) > 4 and (not self.prefix[i] or args.include_aliases_in_prefix):
                prefix = self.prefix[i]
                token = items[0]
                if is_all_upper(token) and token != PREFIX_DELIM:
                    token = token.lower()
                full_mention = items[2]
                wiki = items[4]
                self.tokens.append(token)
                start_span = len(self.tokens)-1
                # set j to be the lookahead index
                j = i +1
                while True:
                    # we've reached the end of the scope -- break out of the look ahead.
                    if j >= len(self.lines):
                        break
                    # check to see if the next line contains an entity mention
                    items = self.lines[j].split("\t")
                    if len(items) > 4:
                        # check that its the same entity
                        if items[1] == 'I':
                            next_token = items[0]
                            if is_all_upper(next_token) and next_token != PREFIX_DELIM:
                                next_token = next_token.lower()
                            self.tokens.append(next_token)
                            j = j + 1
                            continue
                    break
                span_end = len(self.tokens)
                i = j
                if prefix:
                    doc_al_id = doc_al_id_prefix
                    doc_al_id_prefix += 1
                else:
                    doc_al_id = doc_al_id_not_prefix
                    doc_al_id_not_prefix += 1
                self.entities.append({
                    'alias': full_mention.lower(),
                    'span_start': start_span,
                    'span_end': span_end,
                    'qid': self.mapper.get_qid_from_url(wiki),
                    'anchor': not prefix,
                    'doc_id': self.doc_id,
                    'doc_al_id': doc_al_id
                })
            else:
                token = items[0]
                if is_all_upper(token) and token != PREFIX_DELIM:
                    token = token.lower()
                self.tokens.append(token)
                i = i+ 1

    def get_sent_dict(self):
        sentence_dict = {"aliases": [], "qids": [], "spans": [], "sentence": "", "anchor": [], "doc_al_id": []}
        for entity in self.entities:
            sentence_dict['doc_id'] = entity['doc_id']
            sentence_dict['doc_al_id'].append(entity['doc_al_id'])
            sentence_dict['aliases'].append(entity['alias'])
            sentence_dict['spans'].append('%d:%d' % (entity['span_start'], entity['span_end']))
            sentence_dict['sentence'] = ' '.join(self.tokens)
            sentence_dict['qids'].append(entity['qid'])
            sentence_dict['anchor'].append(entity['anchor'])
        return sentence_dict


class Doc:

    def __init__(self, doc_id, lines, mapper, args):
        self.id = doc_id
        self.lines = lines
        self.args = args
        self.mapper = mapper

        # keep track of the alias idx in the document
        self.alias_idx = 0

        # list of samples associated with this document. If the scope is document level,
        # this will only be one dictionary. If the scope is sentence level, there will be
        # one dictionary for each sentence (which has an entity mention)
        self.items = []

        if args.scope == 'sentence':
            self.items = self.get_sentence_dicts()
        elif args.scope == 'document':
            self.items = self.get_document_dict()
        else:
            raise Exception(f"Disambiguation scope not implemented: {args.scope}")

        #if args.include_title and len(self.items) > 0:
        #    print(doc_id, self.items[0])

    def get_sentence_dicts(self):
        # Disambiguate each sentence independently
        sentence_dicts = []
        scope = []
        title_lines = []
        first_sentence_lines = []
        in_first_sentences = True
        for i, line in enumerate(self.lines):
            if len(line) == 0:
                is_prefix = [False for _ in range(len(scope))]
                if len(sentence_dicts) == 0:
                    title_lines = list(scope)
                    # We want to add the title as a sample as well with title [SEP] title
                    scope = copy.deepcopy(title_lines)
                    is_prefix = [False for _ in range(len(scope))]
                    title_lines.append(PREFIX_DELIM)

                # We take the 'first sentence' as the portion
                if len(sentence_dicts) < 3:
                    first_sentence_lines.extend(scope)
                elif len(sentence_dicts) == 3:
                    first_sentence_lines.append(PREFIX_DELIM)

                # add title to scope
                if self.args.include_title:
                    scope = title_lines + scope
                    is_prefix = [True for _ in title_lines] + is_prefix

                # add first few sentences to scope
                if self.args.include_first_sentence and len(sentence_dicts) > 3:
                    scope = first_sentence_lines + scope
                    is_prefix = [True for _ in first_sentence_lines] + is_prefix

                sg = ScopeGroup(scope, is_prefix, self.mapper, self.args, self.id, self.alias_idx)
                sd = sg.get_sent_dict()
                n_true_aliases = len([a for i, a in enumerate(sd['aliases']) if sd['anchor'][i] is True])
                self.alias_idx += n_true_aliases
                if n_true_aliases > 0:
                    sentence_dicts.append(sd)

                scope = []
            else:
                scope.append(line)
        return sentence_dicts

    def get_document_dict(self):
        # Disambiguate all sentences in a document collectively
        scope = [line for line in self.lines if len(line) > 0]
        is_prefix = [False for _ in range(len(scope))]
        sg = ScopeGroup(scope, is_prefix, self.mapper, self.args, self.id)
        sd = sg.get_sent_dict()
        return [sd]


def load_and_dump_sentences(args, qm):
    ''' Load sentences from fp '''

    train_docs = []
    dev_docs = []
    test_docs = []
    split = 'train'
    current_doc_lines = []
    current_doc_id = None
    with open(args.dataset, 'r') as in_file:
        for line in in_file:
            line = line.strip()
            if NEW_DOC_DELIM in line:
                if not current_doc_id is None:
                    doc = Doc(current_doc_id, current_doc_lines, qm, args)
                    if split == 'train':
                        train_docs.append(doc)
                    elif split == 'dev':
                        dev_docs.append(doc)
                    else:
                        test_docs.append(doc)
                    current_doc_lines = []
                current_doc_id = line.replace(NEW_DOC_DELIM, "").replace("(", "").replace(")", "").strip()
                if current_doc_id == DEV_START_DOC:
                    split = 'dev'
                elif current_doc_id == TEST_START_DOC:
                    split = 'test'
            else:
                current_doc_lines.append(line.strip())
        # When the loop ends, the last document's lines are still contained in current_doc_lines.
        # We have to create a document from them and add them to test_docs
        test_docs.append(Doc(current_doc_id, current_doc_lines, qm, args))

    # The following assertions are taken from Ganea + Hoffman, Table 2
    print("# train documents: {}".format(len(train_docs)))
    assert(len(train_docs) == 946)
    print("# dev documents: {}".format(len(dev_docs)))
    assert(len(dev_docs) == 216)
    print("# test documents: {}".format(len(test_docs)))
    assert(len(test_docs) == 231)

    # create output directory, if it doesn't exist
    out_dir = args.out_dir
    if args.scope == 'document':
        out_dir = f"{out_dir}_document"

    if args.include_title:
        out_dir = f"{out_dir}_title"

    if args.include_first_sentence:
        #raise Exception("BE CAREFUL ABOUT USING THIS. THE FIRST SENTENCE OF THE ARTICLE IS INCONSISTENT. PLEASE MANUALLY REMOVE THIS WARNING.")
        out_dir = f"{out_dir}_first_sentence"

    if args.include_aliases_in_prefix:
        out_dir = f"{out_dir}_include_aliases_in_prefix"

    out_dir = prep_utils.get_outdir(out_dir, args.sub_dir)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving data to {out_dir}")
    all_qids = set()
    train = []
    out_path = os.path.join(out_dir, 'train.jsonl')
    with open(out_path, 'w') as out_file:
        sentence_counter = 0
        entity_mention_counter = 0
        anchor_counter = 0
        for doc in train_docs:
            for d in doc.items:
                entity_mention_counter += len(d['qids'])
                anchor_counter += sum(d['anchor'])
                sentence_counter += 1
                for qid in d['qids']:
                    all_qids.add(qid)
                if len(d['qids']) > 0:
                    out_file.write(json.dumps(d) + "\n")
                    train.append(d)
                else:
                    print(f"No entities found in document {doc.id} (TRAIN)")
        print(f"TRAIN: {sentence_counter} sentences. {entity_mention_counter} entity mentions. {anchor_counter} to be evaluated.")

    out_path = os.path.join(out_dir, 'dev_unif.jsonl')
    dev = []
    with open(out_path, 'w') as out_file:
        sentence_counter = 0
        entity_mention_counter = 0
        anchor_counter = 0
        for doc in dev_docs:
            for d in doc.items:
                for qid in d['qids']:
                    all_qids.add(qid)

                sentence_counter += 1
                entity_mention_counter += len(d['qids'])
                anchor_counter += sum(d['anchor'])
                # some documents do not have any entity mentions!
                # we ignore them.
                if len(d['qids']) > 0:
                    out_file.write(json.dumps(d) + "\n")
                    dev.append(d)
                else:
                    print(f"No entities found in document {doc.id} (DEV)")
        print(f"DEV: {sentence_counter} sentences. {entity_mention_counter} entity mentions. {anchor_counter} to be evaluated.")

    out_path = os.path.join(out_dir, 'test.jsonl')
    test = []
    with open(out_path, 'w') as out_file:
        sentence_counter = 0
        entity_mention_counter = 0
        anchor_counter = 0
        for doc in test_docs:
            for d in doc.items:
                for qid in d['qids']:
                    all_qids.add(qid)
                sentence_counter += 1
                entity_mention_counter += len(d['qids'])
                anchor_counter += sum(d['anchor'])
                # some documents do not have any entity mentions!
                # we ignore them.
                if len(d['qids']) > 0:
                    out_file.write(json.dumps(d) + "\n")
                    test.append(d)
                else:
                    print(f"No entities found in document {doc.id} (TEST)")
        print(f"TEST: {sentence_counter} sentences. {entity_mention_counter} entity mentions. {anchor_counter} to be evaluated.")


    print(f"{len(all_qids)} total QIDS across train/dev/test")

    # dump all QIDS
    with open(os.path.join(args.out_dir, 'aida_qids.json'), 'w') as out_file:
        json.dump(list(all_qids), out_file)



def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(json.dumps(vars(args), indent=4))

    aida_to_wpid = get_aida_title_to_wpid(args)
    title2new_title = get_title_to_new_title(args)
    title_to_qid, qid_to_all_titles, wpid_to_qid, qid_to_title = prep_utils.load_qid_title_map(args.title_to_qid)

    qm = QIDMapper(aida_to_wpid, wpid_to_qid, title_to_qid, title2new_title)
    load_and_dump_sentences(args, qm)

if __name__ == '__main__':
    main()
