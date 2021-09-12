import unittest
from argparse import Namespace

import marisa_trie
import ujson, os

from bootleg_data_prep.data_filter import filter_entity_symbols
from bootleg_data_prep.language import ensure_ascii
from bootleg_data_prep.utils.classes.entity_symbols import EntitySymbols


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
            "alias4": [["Q4", 10], ["Q6", 7], ["Q7", 6], ["Q5", 5]],
            "alias5": [["Q3", 100], ["Q6", 100]],
            "alias6": [["Q4", 15]],
            "alias7": [["Q7", 10], ["Q3", 3]],
            "alias8": [["Q5", 24]],
            "alias9": [["Q3", 15], ["Q6", 5]]
        }
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2

def write_data_to_file(filename, data):
    with open(filename, "w") as out_f:
        for json_data in data:
            out_f.write(ujson.dump(data, ensure_ascii=ensure_ascii) + "\n")
    return


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.data_dir = "test/test_data_filter"
        self.filename = os.path.join(self.data_dir, "input_data.jsonl")
        self.entity_symbols = EntitySymbolsSubclass()
        self.args = Namespace(
            no_filter_entities_data = False,
            max_candidates = 3
        )
        return

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        return

    def test_entity_filter(self):
        list_of_all_qids = [["Q5"]]
        qid2title, alias2qids, max_candidates, max_alias_len = filter_entity_symbols(self.args, list_of_all_qids, [], self.entity_symbols)
        resulting_qids = ["Q4", "Q5", "Q6", "Q7"]
        resulting_aliases = ["alias4", "alias8"]
        max_candidates_gold = 3
        max_alias_len_gold = 1
        qid2title_gold = {q: self.entity_symbols.get_title(q) for q in resulting_qids}
        alias2qids_gold = {al: self.entity_symbols.get_qid_count_cands(al)[:max_candidates_gold] for al in resulting_aliases}

        self.assertEqual(max_candidates, max_candidates_gold)
        self.assertEqual(max_alias_len, max_alias_len_gold)
        self.assertDictEqual(qid2title, qid2title_gold, f"{qid2title} versus gold {qid2title_gold}")
        self.assertDictEqual(alias2qids, alias2qids_gold, f"{alias2qids} versus gold {alias2qids_gold}")

if __name__ == '__main__':
    unittest.main()
