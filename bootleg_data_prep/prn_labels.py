"""
Read json files where each line represents a wp article with tokenized sentences.
detect primary pronouns for person entities and link the pronouns to the primary entity

To get wikidata mappings

python3 -m bootleg_data_prep.wikidata.get_person_gender_qids \
        --data /lfs/raiders8/0/lorr1/wikidata_09_21 \
        --out_dir wikidata_output \
        --processes 10

To run
python prn.py data/wiki_dump/orig_wl/ data/wiki_dump/orig_wl_prn data/wiki_dump/orig_wl/entity_db/entity_mappings --num_workers 72 --swap_titles
"""
import random
import shutil
from collections import Counter
from glob import glob
import json
from multiprocessing import Pool
import os
from collections import defaultdict
import argh
import numpy as np
from tqdm import tqdm

# data prep
from bootleg_data_prep.language import ENSURE_ASCII, gender_qid_map, pronoun_map, pronoun_possessive_map, UNKNOWN
from bootleg.symbols.entity_symbols import EntitySymbols

person_set, gender_map = np.load('/dfs/scratch0/lorr1/projects/bootleg-data/data/wikidata_mappings/person.npy', allow_pickle=True)
print('person data loaded')

def prepare_person_numpy():
    """utility function for preparing person.npy"""
    person_set = set()
    with open('wikidata_09_21/wikidata_output/person_qids.json') as f:
        person_set = set(json.load(f))
    print('person list loaded')
    gender_map = {}
    with open('wikidata_09_21/wikidata_output/person_gender.json') as f:
        gender_map_old = json.load(f)
        gender_map = {}
        for qid, gender in gender_map_old.items():
            gender = gender_qid_map.get(gender, 5)
            gender_map[qid] = gender
    print('gender map loaded')
    return person_set, gender_map

# person_set, gender_map = prepare_person_numpy()
# print(list(person_set)[:10])
# print(list(gender_map.items())[:10])
    
def process_file(args):
    filename, output_path, swap_titles, only_first_prn = args
    with open(filename) as f:
        # get filename
        just_file = os.path.basename(filename)
        output_file = os.path.join(output_path, just_file)
        print(f"Writing {filename} to {output_file}")
        with open(output_file, 'w') as fout:
            res = []
            for line in f:
                j = json.loads(line)
                row = identify_primary_pronouns(j)
                res.append(row)
                # gender id, person pronoun id in sentence, title of doc
                g, p, title = row
                # if gender is female or male
                if 0 < g <= 2:
                    print(json.dumps(add_pronoun(j, g, swap_titles, only_first_prn), ensure_ascii=ENSURE_ASCII), file=fout)
                else:
                    print(line.strip(), file=fout)
            return res


def add_pronoun(doc, gender_id, swap_titles, only_first_prn):
    """Example input:
    {
      "doc_sent_idx": 0,
      "sentence": "Harry Shamoon is Professor Emeritus of endocrinology , diabetes and metabolism at the Albert Einstein College of Medicine and Montefiore Medical Center .",
      "aliases": [
        "diabetes",
        "albert einstein college of medicine",
        "montefiore medical center"
      ],
      "qids": [
        "Q12206",
        "Q2030894",
        "Q6905066"
      ],
      "spans": [
        "8:9",
        "13:18",
        "19:22"
      ],
      # optional
      "gold": [
        True,
        True,
        True
      ]
    },
    """
    sentences = doc.get('sentences', [])
    qid = doc.get('qid')
    doc_title = doc.get('title')
    doc_alias = qid2alias_global.get(qid, "")
    doc_title_spl = doc_title.split()
    doc_title_offset = len(doc_title.split())-1
    seed = str(qid[1:]) + str(doc_title)
    random.seed(seed)
    for sent in sentences:
        # Randomly choose if we are going to swap pronoun with full title, first name, or last name
        if swap_titles:
            # Choose between full name, first, or last
            choice_name = random.randint(0, 2)
            if choice_name == 0:
                title = doc_title_spl[0]
                title_offset = 0
            elif choice_name == 1:
                title = doc_title_spl[-1]
                title_offset = 0
            else:
                title = doc_title
                title_offset = doc_title_offset
        else:
            title = doc_title
            title_offset = doc_title_offset
        # possessive title
        title_possessive = f"{title} 's"
        title_possessive_offset = len(title_possessive.split())-1

        if len(sent["aliases"]) > 0:
            if type(sent["spans"][0]) is str:
                sent["spans"] = [list(map(int, s.split(":"))) for s in sent["spans"]]
            assert len(sent["spans"][0]) == 2
        if "gold" not in sent:
            assert "anchor" in sent
            sent["gold"] = sent["anchor"]
            del sent["anchor"]
        if "sources" not in sent:
            sent["sources"] = ["gold" if a else "prn" for a in sent["gold"]]
        sentence = sent['sentence']
        spans = sent['spans']
        tokens = sentence.split()
        new_labels = []
        new_tokens = []
        found_prn = False
        for i, token in enumerate(tokens):
            # If the pronoun matches the gender of the doc
            pronoun_id = pronoun_map.get(token.lower(), -1)
            pronoun_possessive_id = pronoun_possessive_map.get(token.lower(), -1)
            if pronoun_id == gender_id and len(doc_alias) > 0 and (not only_first_prn or not found_prn):
                # t is (new alias, qid, span, gold, source, offset) (to update spans according to making pronoun tokens become page titles)
                if swap_titles:
                    if pronoun_possessive_id == 0:
                        new_tokens.append(title)
                        t = (doc_alias, doc_alias, qid, [i, i+title_offset+1], False, 'prn', title_offset)
                    else:
                        new_tokens.append(title_possessive)
                        t = (doc_alias, doc_alias, qid, [i, i+title_possessive_offset+1], False, 'prn', title_possessive_offset)
                else:
                    new_tokens.append(token)
                    t = (doc_alias, doc_alias, qid, [i, i+1], False, 'prn', 0)
                new_labels.append(t)
                found_prn = True
            else:
                new_tokens.append(token)
        if new_labels:
            # print("OLD", json.dumps(sent, indent=4))
            # need to re-sort
            offsets = [0 for _ in sent['qids']]
            new_labels.extend([(a, us_a, q, sp, h, sr, o) for a, us_a, q, sp, h, sr, o in zip(sent['aliases'],
                                                                                  sent['unswap_aliases'],
                                                                                  sent['qids'],
                                                                                  spans,
                                                                                  sent['gold'],
                                                                                  sent['sources'],
                                                                                  offsets)])
            new_labels = sorted(new_labels, key=lambda x: x[3][0])
            # need to update spans as we replaced the pronouns with titles
            to_add = 0
            for j in range(len(new_labels)):
                new_labels[j][3][0] += to_add
                new_labels[j][3][1] += to_add
                to_add += new_labels[j][6]

            sent['aliases'] = [t[0] for t in new_labels]
            sent['unswap_aliases'] = [t[1] for t in new_labels]
            sent['qids'] = [t[2] for t in new_labels]
            sent['spans'] = [t[3] for t in new_labels]
            sent['gold'] = [t[4] for t in new_labels]
            sent['sources'] = [t[5] for t in new_labels]
            sent['sentence'] = ' '.join(new_tokens)
    #         print("NEW", json.dumps(sent, indent=4))
    return doc


def identify_primary_pronouns(doc):
    sentences = doc.get('sentences', [])
    prn_stats = Counter()
    for sent in sentences:
        sentence = sent['sentence']
        tokens = sentence.lower().split(' ')
        for token in tokens:
            if token in pronoun_map:
                prn_stats[pronoun_map[token]] += 1
    top_prn = prn_stats.most_common(2)
    if top_prn:
        # If there are more than one common pronoun and they appear equally, set to UNK
        if len(top_prn) > 1 and top_prn[1][1] * 1.0 / top_prn[0][1] > 0.9:
            person = UNKNOWN
        # If the most frequent appears infrequently, set to UNK
        elif top_prn[0][1] < 5:
            person = UNKNOWN
        else:
            person = top_prn[0][0]
    else:
        person = UNKNOWN
    qid = doc.get('qid')
    title = doc.get('title').replace(' ', '_')
    if qid in person_set:
        # Get gender of QID
        if qid in gender_map:
            return (gender_map[qid], person, title)
        else:
            return (-1, person, title)
    else:
        return (-2, person, title)


def get_qid2alias(entity_dump):
    # EntitySymbol stores a map of alias-QIDs. We need to build a map of QID->alias
    qid2alias_conflict = defaultdict(list)
    for alias in tqdm(entity_dump.get_all_aliases()):
        cands = entity_dump.get_qid_cands(alias)
        for idx, qid in enumerate(cands):
            if idx >= 30:
                break
            qid2alias_conflict[qid].append([alias, len(cands)])
    qid2alias = {}
    for qid in tqdm(qid2alias_conflict):
        # Add second sort key for determinism
        aliases = sorted(qid2alias_conflict[qid], key = lambda x: (x[1], x[0]), reverse=True)
        qid2alias[qid] = aliases[0][0]
    return qid2alias


def init_pool(args):
    qid2alias = args
    global qid2alias_global
    qid2alias_global = qid2alias


@argh.arg('input_path', help='where the annotation jsons are')
@argh.arg('output_path', help='where the annotation jsons are saved')
@argh.arg('entity_dir', help='where the entities are saved')
@argh.arg('--num_workers', help='parallelism')
@argh.arg('--swap_titles', action='store_true', help='swap pronouns for titles in sentence')
@argh.arg('--only_first_prn', action='store_true', help='label only first prounoun in sentence')
def main(input_path, output_path, entity_dir, num_workers=40, swap_titles=False, only_first_prn=False):
    print(f"input_path: {input_path}, output_path: {output_path}, entity_dir: {entity_dir}")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # load entity symbols
    print(f"Swap titles is {swap_titles} and Only first is {only_first_prn}")
    entity_dump = EntitySymbols.load_from_cache(load_dir=entity_dir)
    print(f"Loaded entity dump with {entity_dump.num_entities} entities.")
    qid2alias = get_qid2alias(entity_dump)
    # output is hardcoded. see process_file
    all_files = glob(os.path.join(input_path, "*.jsonl"), recursive=True)
    all_inputs = [tuple([all_files[i], output_path, swap_titles, only_first_prn]) for i in range(len(all_files))]
    stats = []
    c = 0
    t = 0
    with Pool(processes=num_workers, initializer=init_pool, initargs=tuple([qid2alias])) as pool:
        for res_list in tqdm(pool.imap_unordered(
                process_file,
                all_inputs,
                chunksize=1
            ), total=len(all_inputs)):
            for g, p, title in res_list:
                t += 1
                if 0 < g <= 2:
                    c += 1
                if g == -1 and p <= 2:
                    c += 1
                    # print out when wd gender and predicted gender disagree
                    stats.append({'wd_gender': g, 'pred_gender': p, 'title':title})
    print(f'final stats: {c} have genders in {t}')
    with open('pronoun_run_res.jsonl', 'w') as fout:
        for res in stats:
            print(json.dumps(res, ensure_ascii=ENSURE_ASCII), file=fout)
    entity_output_path = os.path.join(output_path, "entity_db/entity_mappings")
    print(f"Dumping entities to {entity_output_path}...")
    entity_dump.save(entity_output_path)

if __name__ == '__main__':
    argh.dispatch_command(main)
