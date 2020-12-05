'''
This file generates an AIDA dataset where we replace aliases with their doc ids and labels
since we use unique aliases for the Pershina candidate list.

Sample command:
python3.6 -m bootleg_data_prep.benchmarks.pershina.convert_dataset_pershina --data_dir data/aida_title_include_aliases_in_prefix/unfiltered/ --out_dir data/aida_docwiki_pershina/unfiltered
'''

import glob
import sys
import argparse
import jsonlines
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="AIDA data directory")
    parser.add_argument('--out_dir', type=str, required=True, help="AIDA data directory to write with converted aliases")
    return parser.parse_args()

def normalize_doc_id(doc_id):
    if 'testa' in doc_id:
        norm_doc_id = doc_id.split('testa')[0]
    elif 'testb' in doc_id:
        norm_doc_id = doc_id.split('testb')[0]
    else:
        norm_doc_id = doc_id.split(' ')[0]
    assert norm_doc_id.isdigit()
    return norm_doc_id

def main():
    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    for file in glob.glob(os.path.join(data_dir, '*')):
        with jsonlines.open(file) as in_f, jsonlines.open(os.path.join(out_dir, os.path.basename(file)), 'w') as out_f:
            for line in in_f:
                # rename aliases
                aliases = line['aliases']
                doc_al_id_all = line['doc_al_id']
                doc_id = normalize_doc_id(line['doc_id'])
                line['aliases'] = [f'alias_{doc_id}_{doc_al_id}' for doc_al_id in (doc_al_id_all)]
                out_f.write(line)

if __name__ == '__main__':
    main()