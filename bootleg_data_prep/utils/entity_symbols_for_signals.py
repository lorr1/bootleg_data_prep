import os
import shutil
from argparse import Namespace
from rich.progress import track
from collections import defaultdict
import ujson
import marisa_trie
import numpy as np

from bootleg_data_prep.utils import data_prep_utils as prep_utils
from bootleg_data_prep.utils.constants import QIDCOUNT, TYPEWORDS, VOCAB, VOCABFILE, ALIAS2QID, QID2TYPEID_HY, \
    QID2TYPEID_WD, RELMAPPING, CTXRELS, QID2TYPEID_REL, RELATIONWORDS
from bootleg_data_prep.utils.classes.record_trie_collection import RecordTrieCollection
from bootleg_data_prep.utils.classes.type_symbols import TypeSymbols
from bootleg_data_prep.utils.classes.entity_symbols import EntitySymbols
from bootleg_data_prep.utils import utils as utils

"""
This is basically an entity symbols class except it only carries what it needs to generate slices. It's adds type information as well for the slice
generation. We don't just have entity_symbols in here to avoide duplicating unnecessary dictionaries in the multi-processing stages. We also
preprocess things (e.g., alias2qidonly) to hopefully speed up slice generation a bit.
"""

def init_type_words():
    return {}


def get_constants_file(dir):
    utils.ensure_dir(os.path.join(dir, f"entity_constants"))
    return os.path.join(dir, f"entity_constants/entity_constants.json")


def get_contextual_rel_dir(dir):
    res_dir = os.path.join(dir, f"entity_{CTXRELS}")
    utils.ensure_dir(res_dir)
    return res_dir


def get_qid_count_dir(dir):
    res_dir = os.path.join(dir, f"entity_{QIDCOUNT}")
    utils.ensure_dir(res_dir)
    return res_dir


def get_type_words_dir(dir):
    res_dir = os.path.join(dir, f"entity_{TYPEWORDS}")
    utils.ensure_dir(res_dir)
    return res_dir


def get_rel_type_words_dir(dir):
    res_dir = os.path.join(dir, f"entity_{RELATIONWORDS}")
    utils.ensure_dir(res_dir)
    return res_dir


def get_words_dir(dir):
    res_dir = os.path.join(dir, f"entity_{VOCAB}")
    utils.ensure_dir(res_dir)
    return res_dir


def get_tri_dir(dir):
    res_dir = os.path.join(dir, "record_trie")
    utils.ensure_dir(res_dir)
    return res_dir


def load_contextual_rels(dir):
    config_file = os.path.join(dir, "config.json")
    if utils.exists_dir(config_file):
        contextual_rels_config = utils.load_json_file(config_file)
        contextual_rels = marisa_trie.RecordTrie(contextual_rels_config["fmt"]).mmap(os.path.join(dir, f'{CTXRELS}.marisa'))
        return contextual_rels
    else:
        return None


def load_qid_count(dir):
    file = os.path.join(dir, f'{QIDCOUNT}.marisa')
    if utils.exists_dir(file):
        return marisa_trie.RecordTrie('<l').mmap(file)
    return None


def load_type_words(dir):
    config_file = os.path.join(dir, "config.json")
    if utils.exists_dir(config_file):
        type_words_config = utils.load_json_file(os.path.join(dir, "config.json"))
        type_words = marisa_trie.RecordTrie(type_words_config["fmt"]).mmap(os.path.join(dir, f'{TYPEWORDS}.marisa'))
        return type_words
    else:
        return None


def load_rel_type_words(dir):
    config_file = os.path.join(dir, "config.json")
    if utils.exists_dir(config_file):
        type_words_config = utils.load_json_file(os.path.join(dir, "config.json"))
        type_words = marisa_trie.RecordTrie(type_words_config["fmt"]).mmap(os.path.join(dir, f'{RELATIONWORDS}.marisa'))
        return type_words
    else:
        return None


def load_vocab(dir):
    vocab_file = os.path.join(dir, f'{VOCABFILE}.marisa')
    print(f"Reading in vocab from {vocab_file}")
    try:
        vocab = marisa_trie.Trie().mmap(vocab_file)
        return vocab
    except Exception as e:
        print(f"Error reading vocab {e}")
        return None

# To create this, call create_entity_symbols_for_slices() below
class EntitySymbolsForSlice:
    def __init__(self, entity_dump_dir, entity_symbols, qid_count,
                 tri_collection, max_types, max_types_rel, max_relations, max_candidates, qid2typeid_hy,
                 qid2typeid_wd, qid2typeid_rel, rel_mapping, contextual_rel,
                 contextual_rel_vocab_inv, affordance_types, affordance_rel_types,
                 vocab):

        self.entity_dump_dir = entity_dump_dir
        self.tri_collection = tri_collection
        if self.tri_collection is None:
            assert len(rel_mapping) > 0
            # This maps our keys that we use in the helper functions below to the right tri in tri collection.
            # The values are specific strings as outlines in the record trie collection class
            fmt_types = {ALIAS2QID: "qid_cand_with_score", RELMAPPING: "kg_relations"}
            max_values = {ALIAS2QID: entity_symbols.max_candidates, RELMAPPING: max_relations}
            input_dicts = {ALIAS2QID: entity_symbols.get_alias2qids(), RELMAPPING: rel_mapping}

            if len(qid2typeid_hy) > 0:
                qid2typeid_hy_filtered = {k: v for k, v in qid2typeid_hy.items() if len(v) > 0}
                max_l = 0
                for q in qid2typeid_hy_filtered:
                    max_l = max(max_l, len(qid2typeid_hy_filtered[q]))
                fmt_types[QID2TYPEID_HY] = "type_ids"
                max_values[QID2TYPEID_HY] = max_types
                input_dicts[QID2TYPEID_HY] = qid2typeid_hy_filtered

            if len(qid2typeid_wd) > 0:
                qid2typeid_wd_filtered = {k: v for k, v in qid2typeid_wd.items() if len(v) > 0}
                fmt_types[QID2TYPEID_WD] = "type_ids"
                max_values[QID2TYPEID_WD] = max_types
                input_dicts[QID2TYPEID_WD] = qid2typeid_wd_filtered

            if len(qid2typeid_rel) > 0:
                qid2typeid_rel_filtered = {k: v for k, v in qid2typeid_rel.items() if len(v) > 0}
                fmt_types[QID2TYPEID_REL] = "type_ids"
                max_values[QID2TYPEID_REL] = max_types_rel
                input_dicts[QID2TYPEID_REL] = qid2typeid_rel_filtered
            print(f"Max Values {max_values}")
            self.tri_collection = RecordTrieCollection(load_dir=None, input_dicts=input_dicts, vocabulary=entity_symbols.get_qid2eid(),
                                                       fmt_types=fmt_types, max_values=max_values)

        self.qid_count = qid_count
        self.max_types = max_types
        self.max_types_rel = max_types_rel
        self.max_relations = max_relations
        self.max_candidates = max_candidates
        self.affordance_types = affordance_types
        self.affordance_rel_types = affordance_rel_types
        self.contextual_rel_vocab_inv = contextual_rel_vocab_inv
        self.contextual_rel = contextual_rel
        self.vocab = vocab

    def get_qid_count(self, qid):
        # Trie stores lists of items and our item is a len 1 tuple
        if qid not in self.qid_count:
            return 0
        return self.qid_count[qid][0][0]

    def get_qid_cands(self, alias):
        return self.tri_collection.get_value(ALIAS2QID, alias, getter=lambda x: x[0])

    def get_qid_count_cands(self, alias):
        return self.tri_collection.get_value(ALIAS2QID, alias)

    def get_types(self, qid):
        return self.tri_collection.get_value(QID2TYPEID_HY, qid)

    def get_types_wd(self, qid):
        return self.tri_collection.get_value(QID2TYPEID_WD, qid)

    def get_types_rel(self, qid):
        return self.tri_collection.get_value(QID2TYPEID_REL, qid)

    def get_relations(self, qid):
        return self.tri_collection.get_value(RELMAPPING, qid)

    def qid_in_qid2typeid(self, qid):
        return self.tri_collection.is_tri_in_collection(QID2TYPEID_HY) and self.tri_collection.is_key_in_trie(QID2TYPEID_HY, qid)

    def qid_in_qid2typeid_wd(self, qid):
        return self.tri_collection.is_tri_in_collection(QID2TYPEID_WD) and self.tri_collection.is_key_in_trie(QID2TYPEID_WD, qid)

    def qid_in_qid2typeid_rel(self, qid):
        return self.tri_collection.is_tri_in_collection(QID2TYPEID_REL) and self.tri_collection.is_key_in_trie(QID2TYPEID_REL, qid)

    def qid_in_rel_mapping(self, qid):
        return self.tri_collection.is_tri_in_collection(RELMAPPING) and self.tri_collection.is_key_in_trie(RELMAPPING, qid)

    def get_all_aliases(self):
        return self.tri_collection.get_keys(ALIAS2QID)

    def get_relation(self, qid1, qid2):
        pair1 = self.format_relation_pair(qid1, qid2)
        pair2 = self.format_relation_pair(qid2, qid1)
        if pair1 in self.contextual_rel:
            return self.contextual_rel[pair1][0][0]
        elif pair2 in self.contextual_rel:
            return self.contextual_rel[pair2][0][0]
        return None

    def get_all_relations(self, qid1, qid2):
        pair1 = self.format_relation_pair(qid1, qid2)
        pair2 = self.format_relation_pair(qid2, qid1)
        rels = set()
        if pair1 in self.contextual_rel:
            for item in self.contextual_rel[pair1][0]:
                rels.add(item)
        if pair2 in self.contextual_rel:
            for item in self.contextual_rel[pair2][0]:
                rels.add(item)
        return rels

    def get_relation_name(self, relid):
        if relid is None or str(relid) not in self.contextual_rel_vocab_inv:
            return None
        return self.contextual_rel_vocab_inv[str(relid)]

    def format_relation_pair(self, head, tail):
        return f"{head}_{tail}"

    @classmethod
    def load(cls, entity_dump_dir):
        """ A method for loading the data for a separate entity_dump_directory. Good if the data is being moved around and the entity_dump dir is different."""
        max_types, max_types_rel, max_relations, max_candidates, contextual_rel_vocab_inv = cls.load_constants(entity_dump_dir)
        qid_count = load_qid_count(get_qid_count_dir(entity_dump_dir))
        tri_collection = RecordTrieCollection(load_dir=get_tri_dir(entity_dump_dir))
        contextual_rel = load_contextual_rels(get_contextual_rel_dir(entity_dump_dir))
        vocab = load_vocab(get_words_dir(entity_dump_dir))
        affordance_types = load_type_words(get_type_words_dir(entity_dump_dir))
        affordance_rel_types = load_rel_type_words(get_rel_type_words_dir(entity_dump_dir))
        print(f"TYPE AFF IS NONE {affordance_types is None}")
        print(f"REL TYPE AFF IS NONE {affordance_rel_types is None}")
        assert tri_collection is not None, f"Can't load this way with a None tri_collections as the type mappings are empty"
        entity_symbols = None
        qid2typeid_hy = {}
        qid2typeid_wd = {}
        qid2typeid_rel = {}
        rel_mapping = {}
        esp = cls(entity_dump_dir, entity_symbols, qid_count,
                 tri_collection, max_types, max_types_rel, max_relations, max_candidates, qid2typeid_hy,
                 qid2typeid_wd, qid2typeid_rel, rel_mapping, contextual_rel,
                 contextual_rel_vocab_inv, affordance_types, affordance_rel_types,
                 vocab)
        return esp

    def dump(self, entity_dump_dir):
        self.save_constants(entity_dump_dir)
        self.tri_collection.dump(save_dir=get_tri_dir(entity_dump_dir))

    @classmethod
    def load_constants(cls, entity_dump_dir):
        constants_file = get_constants_file(entity_dump_dir)
        utils.ensure_dir(os.path.basename(constants_file))
        constants_dict = utils.load_json_file(constants_file)
        max_types = constants_dict["max_types"]
        max_types_rel = constants_dict["max_types_rel"]
        max_relations = constants_dict["max_relations"]
        max_candidates = constants_dict["max_candidates"]
        contextual_rel_vocab_inv = constants_dict["contextual_rel_vocab_inv"]
        return max_types, max_types_rel, max_relations, max_candidates, contextual_rel_vocab_inv

    def save_constants(self, entity_dump_dir):
        save_file = get_constants_file(entity_dump_dir)
        save_dict = {
            "max_types": self.max_types,
            "max_types_rel": self.max_types_rel,
            "max_relations": self.max_relations,
            "max_candidates": self.max_candidates,
            "contextual_rel_vocab_inv": self.contextual_rel_vocab_inv
        }
        utils.dump_json_file(save_file, save_dict)
        return

    def __getstate__(self):
        state = self.__dict__.copy()
        self.save_constants(self.entity_dump_dir)
        self.tri_collection.dump(save_dir=get_tri_dir(self.entity_dump_dir))
        del state['qid_count']
        del state['tri_collection']
        del state['contextual_rel']
        del state['affordance_types']
        del state['affordance_rel_types']
        del state['vocab']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.qid_count = load_qid_count(get_qid_count_dir(self.entity_dump_dir))
        self.tri_collection = RecordTrieCollection(load_dir=get_tri_dir(self.entity_dump_dir))
        self.contextual_rel = load_contextual_rels(get_contextual_rel_dir(self.entity_dump_dir))
        self.vocab = load_vocab(get_words_dir(self.entity_dump_dir))
        self.affordance_types = load_type_words(get_type_words_dir(self.entity_dump_dir))
        self.affordance_rel_types = load_rel_type_words(get_rel_type_words_dir(self.entity_dump_dir))
        print(f"TYPE AFF IS NONE {self.affordance_types is None}")
        print(f"REL TYPE AFF IS NONE {self.affordance_rel_types is None}")


def load_relations(args, rel_file, all_qids):
    # load relations and build quick hash table look ups
    rel_mapping = {}
    num_lines = sum(1 for line in open(rel_file))
    with open(rel_file, 'r') as f:
        for line in track(f, total=num_lines):
            head, tail = line.strip().split()
            if head not in all_qids or tail not in all_qids:
                continue
            # add heads and tails
            if head in rel_mapping:
                rel_mapping[head].append(tail)
            else:
                rel_mapping[head] = [tail]
            if tail in rel_mapping:
                rel_mapping[tail].append(head)
            else:
                rel_mapping[tail] = [head]
    lens = np.array([len(str(rel_mapping[q])) for q in rel_mapping])
    print(f"Average number of connections {np.average(lens)}, 90th percentile {np.percentile(lens, 90)}, trimming to {args.max_relations}")
    for head in rel_mapping:
        rel_mapping[head] = set(rel_mapping[head][:args.max_relations])
    return rel_mapping

def load_contextual_relations(args, rel_file, entity_dump_dir, all_qids):
    all_relations = ujson.load(open(rel_file, "r"))
    keys = []
    values = []
    all_rels = sorted(list(set([k for rel_dict in track(all_relations.values(), total=len(all_relations)) for k in rel_dict.keys()])))
    rels_vocab = {r:i for i, r in enumerate(all_rels)}
    for head_qid in all_relations:
        if head_qid not in all_qids:
            continue
        for rel in all_relations[head_qid]:
            for tail_qid in all_relations[head_qid][rel]:
                if tail_qid not in all_qids:
                    continue
                key = f"{head_qid}_{tail_qid}"
                value = rels_vocab[rel]
                keys.append(key)
                values.append(tuple([value]))
    if len(values) == 0:
        return
    fmt = "<" + "l"
    trie = marisa_trie.RecordTrie(fmt, zip(keys, values))
    config = {"fmt": fmt}
    out_dir = get_contextual_rel_dir(entity_dump_dir)
    utils.ensure_dir(out_dir)
    utils.dump_json_file(filename=os.path.join(out_dir, "config.json"), contents=config)
    trie.save(os.path.join(out_dir, f'{CTXRELS}.marisa'))
    vocab_inv = {rels_vocab[k]:k for k in rels_vocab}
    return trie, vocab_inv

def load_tri_collection(args, entity_symbols):
    entity_dump_dir = os.path.join(args.data_dir, f"{args.subfolder_name}", "entity_db/entity_mappings")
    load_dir = get_tri_dir(entity_dump_dir)
    tri_collection = None
    # Load saved tri collection if it exists (it wll get reconstructed in EntitySymbolsForSlice if it doesn't exist)
    if utils.exists_dir(load_dir) and not args.overwrite_saved_entities:
        try:
            print(f"Loading trie from {load_dir}")
            tri_collection = RecordTrieCollection(load_dir=load_dir)
        except Exception as e:
            print(f"Tried to load record try in {load_dir} as that directory exists, but got error {e}. Will rebuild")

    # Load type mapping; this isn't slow so we always do it
    print(f"Reading HYEHA types from {args.emb_dir} with type file {args.hy_types}...")
    try:
        type_symbols_hy = TypeSymbols(entity_symbols=entity_symbols, emb_dir=args.emb_dir, max_types=args.max_types,
                                   emb_file="", type_vocab_file=args.hy_vocab, type_file=args.hy_types)
        qid2typeid_hy = type_symbols_hy.qid2typeid
    except Exception as e:
        print(f"Failed to read in hyena types: {e}")
        qid2typeid_hy = {}

    print(f"Reading WIKIDATA types from {args.emb_dir} with type file {args.wd_types}...")
    try:
        type_symbols_wd = TypeSymbols(entity_symbols=entity_symbols, emb_dir=args.emb_dir, max_types=args.max_types,
                                   emb_file="", type_vocab_file=args.wd_vocab, type_file=args.wd_types)
        qid2typeid_wd = type_symbols_wd.qid2typeid
    except Exception as e:
        print(f"Failed to read in wikidata types: {e}")
        qid2typeid_wd = {}

    print(f"Reading RELATION types from {args.emb_dir} with type file {args.wd_types}...")
    try:
        type_symbols_rel = TypeSymbols(entity_symbols=entity_symbols, emb_dir=args.emb_dir, max_types=args.max_types_rel,
                                   emb_file="", type_vocab_file=args.rel_vocab, type_file=args.rel_types)
        qid2typeid_rel = type_symbols_rel.qid2typeid
    except Exception as e:
        print(f"Failed to read in relation types: {e}")
        qid2typeid_rel = {}
    print(f"{len(qid2typeid_hy)} hyena qids have a type {len(qid2typeid_wd)} wd qids have a type {len(qid2typeid_rel)} rel qids have a type")

    # Load relation mapping; this is slower so we check if tri_collection exists
    print(f"Reading kg adj from {args.emb_dir} with kg file {args.kg_adj}...")
    if args.kg_adj == '' or tri_collection is not None:
        print("Skipping reading in kg...")
        rel_mapping = {}
    else:
        print(f"Reading kg_adj from {os.path.join(args.emb_dir, args.kg_adj)}...")
        rel_mapping = load_relations(args, os.path.join(args.emb_dir, args.kg_adj), entity_symbols.get_all_qids())
    print(f"Len {len(rel_mapping)} for rel_mapping")

    # Load contextual relation mappings
    print(f"Reading kg triples from {args.emb_dir} with kg file {args.kg_triples}...")
    if args.kg_triples == '':
        print("Skipping kg triples for contextual relations")
        rel_tri = {}
        rel_vocab_inv = {}
    else:
        print(f"Reading contextual kg triples from {args.kg_triples}")
        rel_tri, rel_vocab_inv = load_contextual_relations(args, os.path.join(args.emb_dir, args.kg_triples), entity_dump_dir, entity_symbols.get_all_qids())
    return tri_collection, qid2typeid_hy, qid2typeid_wd, qid2typeid_rel, rel_mapping, rel_tri, rel_vocab_inv


def load_word_bags(args):
    entity_dump_dir = os.path.join(args.data_dir, f"{args.subfolder_name}", "entity_db/entity_mappings")
    # Load affordances
    type_words_path = get_type_words_dir(entity_dump_dir)
    print(f"Reading in existing affordance bags from {type_words_path}")
    try:
        affordance_types = load_type_words(type_words_path)
    except Exception as e:
        print(f"Failed to read in affordance_types: {e}\n...you need to call the generate slices prep methods to build thiese")
        affordance_types = {}
    if affordance_types is None:
        affordance_types = {}
    print(f"{len(affordance_types)} for affordance_types")
    # Load affordances
    rel_type_words_path = get_rel_type_words_dir(entity_dump_dir)
    print(f"Reading in existing relation affordance bags from {rel_type_words_path}")
    try:
        affordance_rel_types = load_rel_type_words(rel_type_words_path)
    except Exception as e:
        print(f"Failed to read in affordance_types: {e}\n...you need to call the generate slices prep methods to build thiese")
        affordance_rel_types = {}
    if affordance_rel_types is None:
        affordance_rel_types = {}
    print(f"{len(affordance_rel_types)} for relation affordance_types")
    return affordance_types, affordance_rel_types

def load_qid_counts(args, entity_dump_dir):
    utils.ensure_dir(get_qid_count_dir(entity_dump_dir))
    qid_count = load_qid_count(get_qid_count_dir(entity_dump_dir))
    if qid_count is not None and not args.overwrite_saved_entities:
        print(f"Found existing qid_count at {get_qid_count_dir(entity_dump_dir)}")
        return qid_count
    print(f"Rebuilding qid_count")
    alias_qid_cnt = load_alias_qid(args)
    qid_count_dict = defaultdict(int)
    for al in alias_qid_cnt:
        for q in alias_qid_cnt[al]:
            qid_count_dict[q] += alias_qid_cnt[al][q]
    qid_count_dict = dict(qid_count_dict)
    out_file = os.path.join(get_qid_count_dir(entity_dump_dir), f'{QIDCOUNT}.marisa')
    print(f"Saving qid count in {out_file}")
    qid_count = prep_utils.create_single_item_trie(qid_count_dict, out_file=out_file)
    return qid_count


def load_alias_qid(args):
    if not args.not_use_aug_stats:
        ending_str = "withaugment"
    else:
        ending_str = "withoutaugment"
    load_dir_stats = os.path.join(args.data_dir, args.subfolder_name, "stats")
    load_file = os.path.join(load_dir_stats, f"alias_qid_traindata_{ending_str}.json")
    print(f"Loading alias qid counts from {load_file}")
    alias_qid = utils.load_json_file(load_file)
    return alias_qid


# This function creates the entity symbols used for slicing. It does some checkpointing to avoid re-reading in the kg relation data, but
# it could be improved. Any relation that seemed small or already in mmap format is simply reloaded.
def create_entity_symbols_for_slices(args, entity_symbols=None):
    entity_dump_dir = os.path.join(args.data_dir, f"{args.subfolder_name}", "entity_db/entity_mappings")
    print(f"Entity dump directory where all information is saved/loaded from is {entity_dump_dir}")
    if entity_symbols is None:
        entity_symbols = EntitySymbols.load_from_cache(load_dir=entity_dump_dir)
    qid_count = load_qid_counts(args, entity_dump_dir)
    # Tri colleciton stores all mappings over QIDs (alias to candidates, entity to types, entity to relations)
    # Note that tri_collection may be None upon return. If so, it gets reconstructed in EntitySymbolsForSlice
    tri_collection, qid2typeid_hy, qid2typeid_wd, qid2typeid_rel, rel_mapping, rel_tri, rel_vocab_inv = load_tri_collection(args, entity_symbols)
    # Stores information about what words are associated with each QID and with each single type and type combination
    affordance_types, affordance_rel_types = load_word_bags(args)
    # Has a mapping of all indexes used in affordance_types to their word counterparts
    vocab = load_vocab(get_words_dir(entity_dump_dir))
    print("TYPE QID LENGTHS WIKIDATA", len(qid2typeid_wd), "HYENA", len(qid2typeid_hy), "RELATIONS", len(qid2typeid_rel))
    entity_dump_for_slices = EntitySymbolsForSlice(entity_dump_dir, entity_symbols, qid_count, tri_collection,
                                                   args.max_types, args.max_types_rel, args.max_relations, entity_symbols.max_candidates,
                                                   qid2typeid_hy, qid2typeid_wd, qid2typeid_rel,
                                                   rel_mapping, rel_tri, rel_vocab_inv, affordance_types, affordance_rel_types, vocab)
    print(f"Finished building entity symbols")
    return entity_dump_for_slices

# Helper function to create the args for calling create_entity_symbols_for_slices()
def create_args(data_dir='/lfs/raiders8/0/lorr1/data/wiki_dump',
                subfolder_name='korealiases',
                emb_dir='/lfs/raiders8/0/lorr1/embs',
                not_use_aug_stats=False,
                max_types=3,
                max_types_rel=50,
                max_relations=150,
                kg_adj='kg_adj_0905.txt',
                kg_triples='kg_triples_0905.txt',
                hy_vocab='hyena_vocab.json',
                hy_types='hyena_types_0905.json',
                wd_vocab='wikidata_to_typeid_0905.json',
                wd_types='wikidata_types_0905.json',
                rel_vocab='relation_to_typeid_0905.json',
                rel_types='kg_relation_types_0905.json',
                overwrite_saved_entities=False):
    args = Namespace(
        data_dir=data_dir,
        subfolder_name=subfolder_name,
        emb_dir=emb_dir,
        not_use_aug_stats=not_use_aug_stats,
        max_types=max_types,
        max_types_rel=max_types_rel,
        max_relations=max_relations,
        kg_adj=kg_adj,
        kg_triples=kg_triples,
        hy_vocab=hy_vocab,
        hy_types=hy_types,
        wd_vocab=wd_vocab,
        wd_types=wd_types,
        rel_vocab=rel_vocab,
        rel_types=rel_types,
        overwrite_saved_entities=overwrite_saved_entities
    )
    return args

# Loads all entity symobl objects and saves in cache_dir for fast loading again
def load_entity_symbol_objs(args, cache_dir, overwrite_cache=False):
    if overwrite_cache and utils.exists_dir(cache_dir):
        print(f"Removing cache {cache_dir}...")
        shutil.rmtree(cache_dir)
    utils.ensure_dir(cache_dir)
    load_dir = os.path.join(args.data_dir, args.subfolder_name)
    entity_load_dir = os.path.join(load_dir, "entity_db/entity_mappings")
    print(f"Loading entity_symbols from {entity_load_dir}")
    entity_symbols = EntitySymbols.load_from_cache(load_dir=entity_load_dir)
    type_f_hy = os.path.join(cache_dir, "type_symbols_hy.pkl")
    type_f_wd = os.path.join(cache_dir, "type_symbols_wd.pkl")
    type_f_rel = os.path.join(cache_dir, "type_symbols_rel.pkl")
    try:
        entity_symbols_plus = EntitySymbolsForSlice.load(entity_dump_dir=entity_load_dir)
    except Exception as e:
        print(f"Error trying to load entity symbols plus from {entity_load_dir}. Error {e}. The file may not exist. Regenerating...")
        entity_symbols_plus = create_entity_symbols_for_slices(args, entity_symbols)
        # Saving constants. The other objects are precomputed files that are not dumped with the object.
        entity_symbols_plus.dump(entity_load_dir)
    print("Loaded entity symbols.")
    try:
        type_symbols_hy = utils.load_pickle_file(type_f_hy)
        type_symbols_wd = utils.load_pickle_file(type_f_wd)
        type_symbols_rel = utils.load_pickle_file(type_f_rel)
    except Exception as e:
        print(f"Error trying to load {type_f_hy} and {type_f_wd}. The file may not exist. Regenerating...")
        type_symbols_hy = TypeSymbols(entity_symbols=entity_symbols, emb_dir=args.emb_dir, max_types=args.max_types,
                                    emb_file="", type_vocab_file=args.hy_vocab, type_file=args.hy_types)
        type_symbols_wd = TypeSymbols(entity_symbols=entity_symbols, emb_dir=args.emb_dir, max_types=args.max_types,
                                   emb_file="", type_vocab_file=args.wd_vocab, type_file=args.wd_types)
        type_symbols_rel = TypeSymbols(entity_symbols=entity_symbols, emb_dir=args.emb_dir, max_types=args.max_types_rel,
                                   emb_file="", type_vocab_file=args.rel_vocab, type_file=args.rel_types)
        utils.dump_pickle_file(type_f_hy, type_symbols_hy)
        utils.dump_pickle_file(type_f_wd, type_symbols_wd)
        utils.dump_pickle_file(type_f_rel, type_symbols_rel)
    alias_qid_count = load_alias_qid(args)
    typeid2typename_hy = type_symbols_hy.typeid2typename
    type_vocab_hy = {i:v for v,i in typeid2typename_hy.items()}
    typeid2typename_wd = type_symbols_wd.typeid2typename
    type_vocab_wd = {i:v for v,i in typeid2typename_wd.items()}
    typeid2typename_rel = type_symbols_rel.typeid2typename
    type_vocab_rel = {i:v for v,i in typeid2typename_rel.items()}
    return entity_symbols, entity_symbols_plus, type_vocab_hy, typeid2typename_hy, type_vocab_wd, typeid2typename_wd, type_vocab_rel, typeid2typename_rel, alias_qid_count