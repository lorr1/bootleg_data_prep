'''
This file
Extract and format sentences from Kore50 data. Also dumps out list of QIDs reference in Kore50 data. 
'''

import os
import sys

import ujson as json

sys.path.append(os.path.join(sys.path[0], "../../../"))
import argparse

import bootleg_data_prep.utils.data_prep_utils as prep_utils


SUB_DELIM = "~*~"
SUPER_DELIM = "|"
NEW_DOC_DELIM = '-DOCSTART-' 
TITLE_MAP = {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='/dfs/scratch0/lorr1/bootleg/data/benchmarks/kore50_tmp/')
    parser.add_argument('--sub_dir', type=str, default='unfiltered')
    parser.add_argument('--output_format', type = str, default = 'jsonl', choices = ['jsonl', 'txt'])

    # Parameters to set! Make sure the output directory reflects this. 
    parser.add_argument('--scope', type=str, default='sentence', choices=['sentence', 'document'], help = 'Scope to consider when disambiguating mentions (e.g. at the document level, at the sentence level, etc)')

    # Paths to supplementary files -- do not adjust 
    parser.add_argument('--title_to_qid', type=str, default='/dfs/scratch0/lorr1/bootleg/data/wikidata_mappings/title_to_all_ids.jsonl')
    parser.add_argument('--dataset', type=str, default='/dfs/scratch0/lorr1/bootleg/bootleg-internal/bootleg_data_prep/benchmarks/kore50/raw_data/kore50_en.txt', help='Path to Kore50 file')
    parser.add_argument('--langid', type=str, default='en', help='Language id. "en" for english, ...')

    args = parser.parse_args()
    return args

def get_title_from_url(url, langid):
    return url.replace(f'http://{langid}.wikipedia.org/wiki/', "").replace("_", " ")

def get_aida_title_to_wpid(args): 
    title2wpid = {}
    langid = args.langid
    with open(args.annotated_file) as in_file:
        # line with WPID and title looks like this: 
        # 2	Germany	http://en.wikipedia.org/wiki/Germany	11867	/m/0345h
        for line in in_file: 
            items = line.strip().split("\t")
            if len(items) > 3:
                title = get_title_from_url(items[2], langid)
                wpid = items[3]
                title2wpid[title] = wpid 
    print(f"Loaded {len(title2wpid)} title-WPID pairs from {args.annotated_file}")
    return title2wpid 



class QIDMapper:

    def __init__(self, title2qid):
        
        self.title2qid = title2qid # maps wikipedia page title to QID 

    def get_qid_from_url(self, url, langid):
        title = get_title_from_url(url, langid)

        # Get QID based on title 
        if title in self.title2qid:
            return self.title2qid[title]

        print(f"No QID for {title}.")
        return "None"

def convert_lines_to_json(lines, mapper, langid):
    sentence_dict = {"aliases": [], "qids": [], "spans": [], "sentence": "", "gold": []}

    i = 0 
    tokens = []
    cur_span = 0
    while i < len(lines):
        line = lines[i]
        line_items = line.split("\t")
        print("CUR SPAN", cur_span, line_items[0], line_items)
        if len(line_items) >= 4 and not line_items[3] == '--NME--':
            token = line_items[0]
            full_mention = line_items[2]
            print("FULL MENTION", full_mention)
            wiki = line_items[3]
            tokens.append(token)
            start_span = cur_span
            print("TOK", token, start_span)
            span_len = len(line_items[0].split())
            # set j to be the lookahead index 
            j = i + 1
            while True: 
                # we've reached the end of the scope -- break out of the look ahead.
                if j >= len(lines):
                    break
                # check to see if the next line contains an entity mention
                items = lines[j].split("\t")
                if len(items) == 4:
                    # check that its the same entity 
                    if items[1] == 'I':
                        next_token = items[0]
                        tokens.append(next_token)
                        span_len += len(next_token.split())
                        j = j + 1
                        continue 
                break
            span_end = start_span + span_len
            print(tokens, "-", start_span, "len", span_len, "end", span_end, "cur", cur_span)
            i = j
            sentence_dict['aliases'].append(full_mention.lower())
            sentence_dict['spans'].append([start_span,span_end])
            sentence_dict['gold'].append(True)
            if len(line_items) == 5:
                sentence_dict['qids'].append(line_items[4])
            else:
                sentence_dict['qids'].append(mapper.get_qid_from_url(wiki, langid))
            # end_span is +1 from actually token index
            cur_span = span_end
        else:
            tokens.append(line_items[0])
            cur_span += len(line_items[0].split())
            i = i + 1
    sentence_dict['sentence'] = ' '.join(tokens)
    print(sentence_dict['sentence'], "***", sentence_dict['aliases'], "***", sentence_dict['spans'])
    for sp in sentence_dict["spans"]:
        sp_end = sp[-1]
        assert sp_end <= len(sentence_dict["sentence"].split())
    return sentence_dict


def write_output_file(sentences, args):

    out_dir = prep_utils.get_outdir(args.out_dir, args.sub_dir)
    if args.output_format == 'jsonl':
        out_file = open(os.path.join(out_dir, 'test.jsonl'), 'w')
    else: 
        out_file = open(os.path.join(out_dir, 'test.txt'), 'w')

    all_qids = set()
    none_count = 0    
    for i, sentence in enumerate(sentences):
        if args.output_format == 'jsonl':
            out_file.write(json.dumps(sentence) + "\n")
        else: 
            #print(sentence)
            aliases_to_predict = '~*~'.join([str(i) for i in range(len(sentence['qids']))])
            aliases = '~*~'.join(sentence['aliases'])
            qids = '~*~'.join(sentence['qids'])
            spans = '~*~'.join(sentence['spans'])
            out_str = f"{i}|{aliases_to_predict}|{aliases}|{qids}|{spans}|" + sentence['sentence'] + '\n'
            out_file.write(out_str)

        for qid in sentence['qids']:
            if qid == 'None':
                none_count += 1 
            else:
                all_qids.add(qid)
    
    return all_qids, none_count

def load_and_dump_sentences(args, qm):
    ''' Load sentences from fp '''
    langid = args.langid
    sentences = []
    current_lines = []
    with open(args.dataset, 'r') as in_file:
        for i, line in enumerate(in_file):
            line = line.strip()         
            if i == 0:
                continue 
            elif NEW_DOC_DELIM in line:
                sentences.append(convert_lines_to_json(current_lines, qm, langid))
                current_lines = []
            else:
                current_lines.append(line.strip())
        sentences.append(convert_lines_to_json(current_lines, qm, langid))
    
    # The following assertions are taken from Ganea + Hoffman, Table 2 
    print("# of sentences: {}".format(len(sentences)))

    all_qids, none_count = write_output_file(sentences, args)
    print(f"{len(all_qids)} distinct QIDS. QID not found for {none_count} mentions.")

    # dump all QIDS
    with open(os.path.join(args.out_dir, 'kore50_qids.json'), 'w') as out_file: 
        json.dump(list(all_qids), out_file)
        



def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    title2qid, _, _, _ = prep_utils.load_qid_title_map(args.title_to_qid)

    qm = QIDMapper(title2qid)
    load_and_dump_sentences(args, qm)


if __name__ == '__main__':
    main()
