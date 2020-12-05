"""
Code needed to prepare text_data, emb_data, and entity_data. 
The main entrypoints are prep_text_data and prep_entity_data (for both emb and entity).
There are also a couple helper functions.
"""

from collections import Counter, defaultdict
import logging
import gzip
import os

import argh
import scispacy
import spacy
from tqdm.auto import tqdm
import ujson as json


logger = logging.getLogger("bootleg")
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger.setLevel(logging.INFO)


def get_spacy_pipe(model_name="en_core_sci_lg"):
    """default to scispacy model"""
    logger.info('loading spacy model')
    nlp = spacy.load(model_name)
    def set_sent_starts(doc):
        for i, token in enumerate(doc):
            if i == len(doc) - 1:
                continue
            if token.text == '\n':
                # respect newlines
                doc[i + 1].sent_start = True
            elif token.text.endswith('.'):
                if len(token.text) > 1 or (len(token.text) == 1 and doc[i-1].idx + len(doc[i-1]) == token.idx):
                    # dot right before the next token
                    if doc[i + 1].idx == token.idx + len(token.text):
                        doc[i + 1].sent_start = False
        return doc

    # nlp.remove_pipe('sentence_segmenter')
    nlp.add_pipe(set_sent_starts, name='sentence_segmenter', before='parser')
    logger.info('loaded spacy model')
    return nlp


def annotate_doc(doc, nlp_pipe, stats):
    # concat the text so that the character offsets are consistent with the medmentions annotations
    ann = nlp_pipe(f"{doc['title']}\n{doc['abstract']}")
    doc['ann'] = ann.to_json()
    for s, e, span, _ in doc['labels']:
        stats['total num of mentions in medmentions'] += 1
        if ann.char_span(s, e) is None:
            stats['missing span'] += 1
            logger.error(f"{doc['id']}, span ({s}, {e}) = {span} not fetched")
        elif ann.char_span(s, e).text != span:
            stats['misaligned text'] += 1
            logger.error(f"{doc['id']}, span ({s}, {e}) = {span} but found {ann.char_span(s,e).text}")


def annotate(input_file, output_file, nlp_pipe):
    """
    Parse the medmentions file and annotate via spacy. Save results in output_file
    input_file: the medmention file in .gz
    output_file: a jsonl file with annotations
    nlp_pipe: a spacy pipe object
    """
    with gzip.open(input_file, 'rt') as f, open(output_file, 'w') as fout:
        doc = {"labels":[]}
        stats = Counter()
        for line in tqdm(f, desc=f"annotating {input_file}"):
            if not line.strip():
                annotate_doc(doc, nlp_pipe, stats)
                print(json.dumps(doc), file=fout)
                doc = {"labels":[]}
                continue
            fields = line.strip().split('\t')
            if len(fields) > 1:
                # annotations
                char_start = int(fields[1])
                char_end = int(fields[2])
                span_text = fields[3]
                # XXX this works for both st21pv and full
                ent = fields[5].replace('UMLS:', '')
                doc['labels'].append((char_start, char_end, span_text, ent))
            else:
                fields = line.strip().split('|', 2)
                if '|t|' in line:
                    doc['title'] = fields[2]
                    doc['id'] = fields[0]
                else:
                    doc['abstract'] = fields[2]
        # last doc
        if 'id' in doc:
            annotate_doc(doc, nlp_pipe, stats)
            print(json.dumps(doc), file=fout)
        logger.info(f'annotate stats: {stats}')


def convert_annotation_to_bootleg(input_annotation_file, output_bootleg_text_file, qid2title=None):
    """
    input_annotation_file: the same output file from "annotate"
    qid2title: the mapping from entity id to titles. used to filter out the annotations in entity ids not considered.
        if None, no filter.
    """
    stats = Counter()
    with open(input_annotation_file) as f, open(output_bootleg_text_file, 'w') as fout:
        sent_idx_counter = 0
        for line in tqdm(f, desc=f'converting {input_annotation_file}'):
            doc = json.loads(line)
            ann = doc['ann']
            doc_text = ann['text']
            for sent in ann['sents']:
                tokens = []
                sent_start = sent['start']
                sent_end = sent['end']
                for token in ann['tokens']:
                    if token['start'] >= sent_start:
                        if token['end'] > sent_end:
                            break
                        tokens.append(token)

                sent_text = ' '.join([doc_text[token['start']: token['end']] for token in tokens])
                spans = []
                d = {
                    'doc_id': doc['id'],
                    # bootleg
                    "sentence": sent_text,
                    "sent_idx_unq": sent_idx_counter,
                    "aliases": [],
                    "spans": [],
                    "qids": [],
                    "gold": [],
                }

                for start, end, span, ent in doc['labels']:
                    if not(sent_start <= start < sent_end):
                        continue
                    if doc_text[start:end] != span:
                        logger.error(f"{doc['id']}, span ({start}, {end}) = {span} but found {doc_text[start:end]}")
                        stats['token span text match fails'] += 1
                        pass
                    else:
                        token_start = None
                        token_end = None
                        for i, token in enumerate(tokens):
                            if token['start'] == start:
                                token_start = i
                            if token['end'] == end:
                                token_end = i + 1
                        if token_start is not None and token_end is not None:
                            if qid2title is None or ent.strip() in qid2title:
                                d["aliases"].append(span.lower())
                                d['spans'].append([token_start, token_end])
                                d['qids'].append(ent.strip())
                                d['gold'].append(True)
                            else:
                                stats['missing qid'] += 1
                        else:
                            logger.error(f"{doc['id']}, span {span} ({start}, {end}) -> {ent} missed in the sentence ({sent_start}, {sent_end}): {sent_text}")
                            stats['token offsets fails to find in the sentence'] += 1
                if d['aliases']:
                    print(json.dumps(d), file=fout)
                    sent_idx_counter += 1

    logger.info(f'convert stats: {stats}')


def split(input_bootleg_file, input_medmention_dir, output_dir):
    """Split the medmentions dataset prepared in the bootleg format into train/dev/test
    """
    all_docs = defaultdict(list)
    with open(input_bootleg_file) as f:
        for line in tqdm(f, desc=f'loading {input_bootleg_file}'):
            j = json.loads(line)
            all_docs[j['doc_id']].append((line.strip(), j))

    def subset(output_file, doc_ids):
        stats = Counter()
        stats['docs'] = len(doc_ids)
        with open(output_file, 'w') as fout:
            for doc_id in doc_ids:
                lst = all_docs[doc_id]
                for l, j in lst:
                    stats['mentions'] += len(j['aliases'])
                    stats['sentences'] += 1
                    print(l, file=fout)
        logger.info(f'{output_file} stats: {stats}')

    # split into train, dev, test
    train_docs = set(open(os.path.join(input_medmention_dir, 'corpus_pubtator_pmids_trng.txt')).read().strip().split('\n'))
    dev_docs = set(open(os.path.join(input_medmention_dir, 'corpus_pubtator_pmids_dev.txt')).read().strip().split('\n'))
    test_docs = set(open(os.path.join(input_medmention_dir, 'corpus_pubtator_pmids_test.txt')).read().strip().split('\n'))
    subset(os.path.join(output_dir, 'train.jsonl'), train_docs)
    subset(os.path.join(output_dir, 'dev.jsonl'), dev_docs)
    subset(os.path.join(output_dir, 'test.jsonl'), test_docs)
    logger.info('dataset done split into train/dev/test')


def prep_text_data(
    mm_dir,
    output_dir,
    dataset_name='full', # full or st21pv
):
    assert dataset_name in {'full', 'st21pv'}, 'dataset_name must be either full or st21pv'
    input_text_file = os.path.join(mm_dir, dataset_name, 'data/corpus_pubtator.txt.gz')
    text_dir = os.path.join(output_dir, dataset_name, 'text_data')
    os.makedirs(text_dir, exist_ok=True)

    # annotate the corpus
    output_ann_file = os.path.join(text_dir, 'corpus.jsonl')
    annotate(input_text_file, output_ann_file, get_spacy_pipe())

    # convert to bootleg
    output_bootleg_file = os.path.join(text_dir, 'all.jsonl')
    convert_annotation_to_bootleg(output_ann_file, output_bootleg_file, None)

    # split
    split(output_bootleg_file, os.path.join(mm_dir, dataset_name, 'data'), text_dir)


def get_alias_map_from_umls(input_entity_file, output_alias_to_qid_file):
    """use title and alias of each concept in umls to generate the mapping"""
    alias_map_umls = defaultdict(lambda: defaultdict(int))
    # output_alias_to_qid_umls_file = f'{mm_dir}/entity_data/alias2qids_umls.json'
    with open(input_entity_file) as f, open(output_alias_to_qid_file, 'w') as fout:
        for line in tqdm(f, desc=f'scanning {input_entity_file}'):
            j = json.loads(line)
            for x in j:
                eid = x['concept_id']
                for alias in x.get('aliases', []):
                    alias_map[alias.lower()][eid] += 1
                alias_map[x['canonical_name'].lower()][eid] += 5
        alias_map = {x: sorted(list(y.items()), key=lambda x: -x[1]) for x, y in alias_map.items()}
        print(json.dumps(alias_map), file=fout)


def load_scispacy_candgen():
    import scispacy.version
    ver = scispacy.version.VERSION

    from scispacy.candidate_generation import CandidateGenerator
    if ver in ['0.2.2', '0.2.3', '0.2.4']:
        filter_cands = False
        candgen = CandidateGenerator()
    else:
        filter_cands = True
        candgen = CandidateGenerator('umls')
    return filter_cands, candgen


def get_alias_map_from_scispacy(input_file, output_alias_to_qid_scispacy_file, k=30, qid_dict=None):
    """use the scispacy package (char tf-idf based) to generate a dataset-specific mapping
    Works for scispacy >= 0.2.2
    if version is 0.2.2, 0.2.3, 0.2.4, no need to filter candidates b/c they are using 2017aa cat1,2,9
    else umls is 2020aa, cat1,2,9. we need to filter the entities

    input_file: all examples in bootleg format
    k: top k candidates are kept
    """
    filter_cands, candgen = load_scispacy_candgen()
    assert not filter_cands or qid_dict is not None, f'for version ({ver}) > 0.2.4, a qid dictionary must be provided for filtering'

    alias2qids_scispacy = defaultdict(lambda: defaultdict(int))
    with open(input_file) as f:
        for line in tqdm(f, desc=f'generating cands for {input_file}'):
            j = json.loads(line)
            probe_aliases = [alias.lower() for alias in j['aliases'] if alias.lower() not in alias2qids_scispacy]
            cands = candgen(probe_aliases, k=k)

            for alias, cand_list in zip(probe_aliases, cands):
                if alias not in alias2qids_scispacy:
                    alias2qids_scispacy[alias] = sorted(
                        [(mc.concept_id, max(mc.similarities)) for mc in cand_list], key=lambda x: -x[1])
    logger.info(f'{len(alias2qids_scispacy)} aliases are produced')
    with open(output_alias_to_qid_scispacy_file, 'w') as fout:
        print(json.dumps(alias2qids_scispacy), file=fout)
    logger.info(f'{output_alias_to_qid_scispacy_file} is written')


def assess_scispacy_candidate_size_recall(input_file):
    """Show the candidate recall from the scispacy candgen given different top K values
    """
    _, candgen = load_scispacy_candgen()
    max_k = 200
    k_step = 10
    stats = Counter()
    with open(input_file) as f:
        for line in tqdm(f):
            j = json.loads(line)
            cands = candgen(j['aliases'], k=max_k)
            for k in range(k_step, max_k, k_step):
                for gold_entity, cand_list in zip(j['qids'], cands):
                    # if concept id is not in umls 2017aa, we ignore. it can slightly underestimate
                    # the cand recall @ k but shouldn't be significant
                    if any(gold_entity == mc.concept_id for mc in cand_list[:k]):
                        stats[f'hit@{k}'] += 1
                    else:
                        stats[f'miss@{k}'] += 1
    for k in range(k_step, max_k, k_step):
        print(f'recall@{k}:', stats[f'hit@{k}'] / (stats[f'miss@{k}'] + stats[f'hit@{k}']))


def evaluate_candidate_recall(input_file, alias_map_file, output_filter_file=None):
    """
    output_filter_file: is useful when we want to keep the training examples free from NC labels.
    """
    alias_map = json.loads(open(alias_map_file).read())
    stats = Counter()
    save_filter_file = output_filter_file is not None
    if save_filter_file:
        logger.info(f'saving a file filtering out spans without the gold entity in the candidates to {output_filter_file}')
        filter_file = open(output_filter_file, 'w')
    with open(input_file) as f:
        for line in tqdm(f, desc=f'generating cands for {input_file}'):
            j = json.loads(line)
            probe_aliases = [alias.lower() for alias in j['aliases']]
            hit_idxs = []
            for idx, (alias, qid) in enumerate(zip(probe_aliases, j['qids'])):
                stats['total'] += 1
                if qid in set(x[0] for x in alias_map[alias]):
                    stats['hits'] += 1
                    hit_idxs.append(idx)
            if save_filter_file:
                if hit_idxs:
                    for k in ['aliases', 'spans', 'qids', 'gold']:
                        j[k] = [j[k][idx] for idx in hit_idxs]
                    print(json.dumps(j), file=filter_file)
    if save_filter_file:
        filter_file.close()
        logger.info(f'The same set with spans w/o gold in candidates filtered is saved to {output_filter_file}')
    logger.info(f'{input_file}: candidate recall = {stats["hits"] / stats["total"]} ({stats})')


def prep_umls_rel_data(input_relationship_file, output_conn_file, output_qid_to_relations_file, output_relation_vocab_file):
    """
    `input_relationship_file` needs to be generated offline as follows:
    1. get the UMLS dump
    2. unzip 2017aa-1-meta.nlm
    3. Run something like the following python code
    concept_ids = {...} # the whole set of entity ids from UMLS
    stats = Counter()
    fout = open(..., 'w') # this will be `input_relationship_file`
    for part in ['aa', 'ab', 'ac', 'ad']:
        with gzip.open(f'2017AA/META/MRREL.RRF.{part}.gz') as f:
            for line in tqdm(f):
                fields = line.decode().strip().split('|')
                if fields[0] in concept_ids and fields[4] in concept_ids:
                    stats['rel'] += 1
                    stats[f'3:{fields[3]}'] += 1
                    stats[f'7:{fields[7]}'] += 1
                    print(json.dumps({'CUI1': fields[0], 'CUI2': fields[4], 'REL': fields[3], 'RELA': fields[7], 'RUI': fields[8]}), file=fout)
    """
    relations = set()
    rel2id = {}
    qid2rels = defaultdict(list)

    with open(input_relationship_file) as f:
        for line in tqdm(f, desc=f'scanning the relation data in {input_relationship_file}'):
            j = json.loads(line)
            rel = j['RELA']
            if not rel:
                # skip relationships where RELA is empty.
                continue
            if j["CUI1"] == j["CUI2"]:
                # skip self relationships
                continue
            relations.add((j["CUI1"], j["CUI2"]))
            if rel not in rel2id:
                rel2id[rel] = len(rel2id)
            rel_id = rel2id[rel]
            qid2rels[j["CUI1"]].append(rel_id)

    with open(output_conn_file, 'w') as fout:
        for a, b in relations:
            print(f'{a}\t{b}', file=fout)
    logger.info(f'{output_conn_file} generated: {len(relations)} relations')
    with open(output_qid_to_relations_file, 'w') as fout:
        print(json.dumps(qid2rels), file=fout)
    logger.info(f'{output_qid_to_relations_file} generated: {len(qid2rels)} entities have relations')
    rel_signs = Counter()
    rel_freq = Counter()
    rel_length = Counter()
    id2rel = {y:x for x, y in rel2id.items()}
    for qid, rels in tqdm(qid2rels.items()):
        rels = list(set(rels))
        rel_length[len(rels)] += 1
        rel_freq.update([id2rel[x] for x in rels])
        sorted_rels = ','.join(str(x) for x in sorted(rels))
        rel_signs[sorted_rels] += 1
    sum_freq = 0
    for k in range(20):
        sum_freq += rel_length[k]
        logger.info(f'relation coverage at {k} = {sum_freq / len(qid2rels)}')
    logger.info(f'most common relation types: {rel_freq.most_common(20)}')
    with open(output_relation_vocab_file, 'w') as fout:
        print(json.dumps(rel2id), file=fout)
    logger.info(f'{output_relation_vocab_file} generated: {len(rel2id)} unique relations')


def prep_entity_data(output_dir, dataset_name, input_entity_file, input_relationship_file):
    """
    (optional) download https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/umls_semantic_type_tree.tsv

    input_entity_file: download https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/umls_2017_aa_cat0129.json

    """
    # entity data is not dataset specific but for convenience we put them together
    entity_dir = os.path.join(output_dir, dataset_name, 'entity_data')
    os.makedirs(entity_dir, exist_ok=True)
    emb_dir = os.path.join(output_dir, dataset_name, 'emb_data')
    os.makedirs(emb_dir, exist_ok=True)
    text_dir = os.path.join(output_dir, dataset_name, 'text_data')

    # generate entity title and type files
    output_qid_to_entity_file = os.path.join(entity_dir, 'qid2title.json')
    output_qid_to_type_file = os.path.join(emb_dir, 'qid2types.json')
    output_type_vocab_file = os.path.join(emb_dir, 'type_vocab.json')
    # Example line: {"concept_id": "C0000039", "aliases": ["1,2-Dipalmitoylphosphatidylcholine", "1,2-Dipalmitoylphosphatidylcholine", "1,2 Dipalmitoylphosphatidylcholine", "1,2-Dihexadecyl-sn-Glycerophosphocholine", "1,2-Dihexadecyl-sn-Glycerophosphocholine", "1,2 Dihexadecyl sn Glycerophosphocholine", "1,2-Dipalmitoyl-Glycerophosphocholine", "1,2-Dipalmitoyl-Glycerophosphocholine", "1,2 Dipalmitoyl Glycerophosphocholine", "Dipalmitoylphosphatidylcholine", "Dipalmitoylphosphatidylcholine", "Dipalmitoylphosphatidylcholine", "Dipalmitoylphosphatidylcholine", "Dipalmitoylphosphatidylcholine", "Dipalmitoylglycerophosphocholine", "Dipalmitoylglycerophosphocholine", "Dipalmitoyllecithin", "Dipalmitoyllecithin", "3,5,9-Trioxa-4-phosphapentacosan-1-aminium, 4-hydroxy-N,N,N-trimethyl-10-oxo-7-((1-oxohexadecyl)oxy)-, inner salt, 4-oxide", "3,5,9-Trioxa-4-phosphapentacosan-1-aminium, 4-hydroxy-N,N,N-trimethyl-10-oxo-7-((1-oxohexadecyl)oxy)-, inner salt, 4-oxide", "Dipalmitoyl Phosphatidylcholine", "Dipalmitoyl Phosphatidylcholine", "Phosphatidylcholine, Dipalmitoyl", "1,2-Dipalmitoylphosphatidylcholine [Chemical/Ingredient]"], "types": ["T109", "T121"], "canonical_name": "1,2-Dipalmitoylphosphatidylcholine", "definition": "Synthetic phospholipid used in liposomes and lipid bilayers to study biological membranes. It is also a major constituent of PULMONARY SURFACTANTS."}
    title_len_limit = 510 # because of a limit in the bootleg code for a title to be shorter than 512 chars
    # entity type
    type_to_id = {}
    eid_to_type = defaultdict(set)
    def get_tid(t):
        if t not in type_to_id:
            type_to_id[t] = len(type_to_id)
        return type_to_id[t]

    stats = Counter()
    with open(input_entity_file) as f, open(output_qid_to_entity_file, 'w') as fout:
        for line in tqdm(f, desc=f'reading {input_entity_file}'):
            j = json.loads(line)
            for x in j:
                eid = x['concept_id']
                # types
                for sem_type in x.get('types', ()):
                    sem_type_id = get_tid(sem_type)
                    eid_to_type[eid].add(sem_type_id)
            # title
            print(json.dumps({x['concept_id']: x['canonical_name'][:title_len_limit] for x in j}), file=fout)
    logger.info(f"qid_to_entity generated to {output_qid_to_entity_file}")
    with open(output_qid_to_type_file, 'w') as fout:
        print(json.dumps({x: list(y) for x, y in eid_to_type.items()}), file=fout)
    logger.info(f'{len(eid_to_type)} entities have types, saved to {output_qid_to_type_file}. max len of types: {max(len(v) for v in eid_to_type.values())}')
    with open(output_type_vocab_file, 'w') as fout:
        print(json.dumps(type_to_id), file=fout)
    logger.info(f'{len(type_to_id)} types are saved to {output_type_vocab_file}')

    # alias map
    alias_map_file = os.path.join(entity_dir, 'alias2qids_scispacy.json')
    get_alias_map_from_scispacy(os.path.join(text_dir, 'all.jsonl'), alias_map_file)
    evaluate_candidate_recall(os.path.join(text_dir, 'test.jsonl'), alias_map_file)

    # relationship data
    output_conn_file = os.path.join(emb_dir, 'kg_conn.txt')
    output_qid_to_relations_file = os.path.join(emb_dir, 'qid2relations.json')
    output_relation_vocab_file = os.path.join(emb_dir, 'relation_vocab.json')
    # input_relationship_file = '/home/xling/data/health/2017AA-active/2017aa_cat0129_rels.jsonl'
    prep_umls_rel_data(
        input_relationship_file,
        output_conn_file,
        output_qid_to_relations_file,
        output_relation_vocab_file,
        )


if __name__ == '__main__':
    argh.dispatch_commands([
        prep_text_data,
        prep_entity_data,
        get_alias_map_from_scispacy,
        evaluate_candidate_recall,
        assess_scispacy_candidate_size_recall,
    ])
