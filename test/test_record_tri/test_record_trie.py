import shutil
import unittest
import os, sys

import marisa_trie
import numpy as np

from bootleg_data_prep.utils.classes.entity_symbols import EntitySymbols
from bootleg_data_prep.utils.classes.record_trie_collection import RecordTrieCollection

class EntitySymbolsSubclass(EntitySymbols):
    def __init__(self):
        self.max_candidates = 4
        # Used if we need to do any string searching for aliases. This keep track of the largest n-gram needed.
        self.max_alias_len = 1
        self._qid2title = {"Q1": "title 1", "Q2": "title 2", "Q3": "title 3", "Q4": "title 4", "Q5": "title 5", "Q6": "title 6", "Q7": "title 7"}
        self._qid2eid = {"Q1" : 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5, "Q6": 6, "Q7": 7}
        self._alias2qids = {
            "alias1": [["Q1", 10], ["Q2", 3]],
            "alias2": [["Q3", 100]],
            "alias3": [["Q1", 15], ["Q4", 5]],
            "alias4": [["Q4", 10], ["Q6", 7], ["Q7", 6]],
            "alias5": [["Q3", 100], ["Q6", 100]],
            "alias6": [["Q4", 15]],
            "alias7": [["Q7", 10], ["Q3", 3]],
            "alias8": [["Q5", 24]],
            "alias9": [["Q3", 15], ["Q6", 5]]
        }
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2

class TestRecordTrieCollection(unittest.TestCase):

    def test_construction_single(self):
        entity_dump_dir = "test/data/entity_loader/entity_data/entity_mappings"
        entity_symbols = EntitySymbolsSubclass()
        fmt_types = {"trie1": "qid_cand_with_score"}
        max_values = {"trie1": 3}
        input_dicts = {"trie1": entity_symbols.get_alias2qids()}
        record_trie = RecordTrieCollection(load_dir=None, input_dicts=input_dicts, vocabulary=entity_symbols.get_qid2eid(),
                                 fmt_types=fmt_types, max_values=max_values)
        truealias2qids = {
            "alias1": [["Q1", 10], ["Q2", 3]],
            "alias2": [["Q3", 100]],
            "alias3": [["Q1", 15], ["Q4", 5]],
            "alias4": [["Q4", 10], ["Q6", 7], ["Q7", 6]],
            "alias5": [["Q3", 100], ["Q6", 100]],
            "alias6": [["Q4", 15]],
            "alias7": [["Q7", 10], ["Q3", 3]],
            "alias8": [["Q5", 24]],
            "alias9": [["Q3", 15], ["Q6", 5]]
        }

        for al in truealias2qids:
            self.assertEqual(truealias2qids[al], record_trie.get_value("trie1", al))

        for al in truealias2qids:
            self.assertEqual([x[0] for x in truealias2qids[al]], record_trie.get_value("trie1", al, getter=lambda x: x[0]))

        for al in truealias2qids:
            self.assertEqual([x[1] for x in truealias2qids[al]], record_trie.get_value("trie1", al, getter=lambda x: x[1]))

        self.assertEqual(set(truealias2qids.keys()), set(record_trie.get_keys("trie1")))

        if os.path.exists(os.path.join(entity_dump_dir, "record_trie")):
            shutil.rmtree(os.path.join(entity_dump_dir, "record_trie"))

    def test_construction_double(self):
        truealias2qids = {
            "alias1": [["Q1", 10], ["Q2", 3]],
            "alias2": [["Q3", 100]],
            "alias3": [["Q1", 15], ["Q4", 5]],
            "alias4": [["Q4", 10], ["Q6", 7], ["Q7", 6]],
            "alias5": [["Q3", 100], ["Q6", 100]],
            "alias6": [["Q4", 15]],
            "alias7": [["Q7", 10], ["Q3", 3]],
            "alias8": [["Q5", 24]],
            "alias9": [["Q3", 15], ["Q6", 5]]
        }
        truealias2typeids = {
                        'Q1': [1, 2, 3],
                        'Q2': [8],
                        'Q3': [4],
                        'Q4': [2,4],
                        'Q5': [8],
                        'Q6': [4],
                        'Q7': [2,4]
                        }
        truerelations = {
                        'Q1': {"Q2", "Q3"},
                        'Q2': {"Q1"},
                        'Q3': {"Q2"},
                        'Q4': {"Q1", "Q2", "Q3"}
                        }
        entity_dump_dir = "test/data/entity_loader/entity_data/entity_mappings"
        entity_symbols = EntitySymbolsSubclass()
        fmt_types = {"trie1": "qid_cand_with_score", "trie2": "type_ids", "trie3": "kg_relations"}
        max_values = {"trie1": 3, "trie2": 3, "trie3": 3}
        input_dicts = {"trie1": entity_symbols.get_alias2qids(), "trie2": truealias2typeids, "trie3": truerelations}
        record_trie = RecordTrieCollection(load_dir=None, input_dicts=input_dicts, vocabulary=entity_symbols.get_qid2eid(),
                                 fmt_types=fmt_types, max_values=max_values)

        for al in truealias2qids:
            self.assertEqual(truealias2qids[al], record_trie.get_value("trie1", al))

        for qid in truealias2typeids:
            self.assertEqual(truealias2typeids[qid], record_trie.get_value("trie2", qid))

        for qid in truerelations:
            self.assertEqual(truerelations[qid], record_trie.get_value("trie3", qid))

        if os.path.exists(os.path.join(entity_dump_dir, "record_trie")):
            shutil.rmtree(os.path.join(entity_dump_dir, "record_trie"))

    def test_load_and_save(self):
        entity_dump_dir = "test/data/entity_loader/entity_data/entity_mappings"
        entity_symbols = EntitySymbolsSubclass()
        fmt_types = {"trie1": "qid_cand_with_score"}
        max_values = {"trie1": 3}
        input_dicts = {"trie1": entity_symbols.get_alias2qids()}
        record_trie = RecordTrieCollection(load_dir=None, input_dicts=input_dicts, vocabulary=entity_symbols.get_qid2eid(),
                                 fmt_types=fmt_types, max_values=max_values)

        record_trie.dump(save_dir=os.path.join(entity_dump_dir, "record_trie"))
        record_trie_loaded = RecordTrieCollection(load_dir=os.path.join(entity_dump_dir, "record_trie"))

        self.assertEqual(record_trie._fmt_types, record_trie_loaded._fmt_types)
        self.assertEqual(record_trie._max_values, record_trie_loaded._max_values)
        self.assertEqual(record_trie._stoi, record_trie_loaded._stoi)
        self.assertEqual(record_trie._record_tris, record_trie_loaded._record_tris)

        if os.path.exists(os.path.join(entity_dump_dir, "record_trie")):
            shutil.rmtree(os.path.join(entity_dump_dir, "record_trie"))

if __name__ == '__main__':
    unittest.main()
