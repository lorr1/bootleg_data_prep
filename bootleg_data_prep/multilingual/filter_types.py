import argparse
import os
import time

import ujson as json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lan_type_file', type=str, default='/lfs/raiders8/0/lorr1/de_bootleg/embs_de/wikidata_types.txt', help='File where Language filtered type assignment is.')
    parser.add_argument('--lan_type_id_map', type=str, default='/lfs/raiders8/0/lorr1/de_bootleg/embs_de/wikidata_to_typeid.json', help='File where Language filtered type ids are.')
    parser.add_argument('--en_type_file', type=str, default='/lfs/raiders8/0/lorr1/embs/wikidata_types_wiki_filt_occmonth2.txt', help='File where English filtered type assignment is.')
    parser.add_argument('--en_type_id_map', type=str, default='/lfs/raiders8/0/lorr1/embs/wikidata_to_typeid_occmonth2.json', help='File where English filtered type ids are.')

    args = parser.parse_args()
    return args

def load_id_to_typeqid(id_to_typeqid_f):
    typeqid_to_id = json.load(open(id_to_typeqid_f, "r"))
    id_to_typeqid = {v:k for k,v in typeqid_to_id.items()}
    return typeqid_to_id, id_to_typeqid

def load_types(type_file, id_to_typeqid):
    qid_to_typeqid = {}
    with open(type_file, "r") as in_f:
        for line in tqdm(in_f):
            qid, type_str = line.strip().split(",")
            type_ids = []
            if len(type_str.strip()) > 0:
                types = map(int, type_str.split("|"))
                type_ids = [id_to_typeqid[t] for t in types]
            qid_to_typeqid[qid] = type_ids
    return qid_to_typeqid

def get_all_used_typeqids(qid_to_typeqid):
    return set.union(*[set(v) for v in qid_to_typeqid.values()])

def filter_lan_types(qid_to_typeqid, used_typeqids, used_typeqid_to_id):
    filtered_qid_to_typeqid = {}
    for qid in tqdm(qid_to_typeqid):
        old_typeqids = qid_to_typeqid[qid]
        new_typeqids = [t for t in old_typeqids if t in used_typeqids]
        filtered_qid_to_typeqid[qid] = sorted([used_typeqid_to_id[t] for t in new_typeqids])
    return filtered_qid_to_typeqid

def write_types(out_file, type_list):
    with open(out_file, "w") as out_file:
        for qid, types in tqdm(type_list.items()):
            out_file.write(qid + "," + "|".join(map(str, types)) + "\n")
    print(f"Writtten to {out_file}")
    return


def main():
    gl_start = time.time()
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    en_typeqid_to_id, en_id_to_typeqid = load_id_to_typeqid(args.en_type_id_map)
    print(f"Loading english types")
    en_qid_to_typeqid = load_types(args.en_type_file, en_id_to_typeqid)
    lan_typeqid_to_id, lan_id_to_typeqid = load_id_to_typeqid(args.lan_type_id_map)
    print(f"Loading language types")
    lan_qid_to_typeqid = load_types(args.lan_type_file, lan_id_to_typeqid)
    en_used_typeqids = get_all_used_typeqids(en_qid_to_typeqid)
    print(f"Filtering types")
    lan_filtered_qid_to_typeid = filter_lan_types(lan_qid_to_typeqid, en_used_typeqids, en_typeqid_to_id)
    print(f"Writing out types")
    out_f = os.path.join(os.path.dirname(args.lan_type_file), "wikidata_types_wiki_filt.txt")
    write_types(out_f, lan_filtered_qid_to_typeid)

    out_f = os.path.join(os.path.dirname(args.lan_type_file), "wikidata_to_typeid_filt.json")
    with open(out_f, "w") as f:
        json.dump(en_typeqid_to_id, f)

    print(f"Finished data_filter in {time.time() - gl_start} seconds.")

if __name__ == '__main__':
    main()