import unicodedata
from collections import defaultdict

import json
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from tqdm import tqdm

ps = PorterStemmer()
nlp = spacy.load("en_core_web_sm")
stopwords = set(stopwords.words('english'))

import bootleg_data_prep.utils.data_prep_utils as prep_utils

class Standard:

    def __init__(self, args, entity_dump):

        self.args = args
        self.entity_dump = entity_dump
        self.new_alias2qid = self.entity_dump._alias2qids
        self.load_candidates()

    def load_candidates(self):
        self.qid2alias = defaultdict(set)
        self.alias2qid = defaultdict(set)
        max_count = 0
        for alias in tqdm(self.entity_dump.get_all_aliases()):
            for qid in self.entity_dump.get_qid_cands(alias):
                self.qid2alias[qid].add(alias)
                self.alias2qid[alias].add(qid)

    def filter_sentence(self, sentence):
        success = 0
        no_alias_exists = 0
        qid_not_in_cands = 0
        errors = []
        new_sentence = {
            'sentence': sentence['sentence'],
            'sent_idx_unq': sentence['sent_idx_unq'],
            'aliases': [],
            'spans': [],
            'qids': [],
            'gold': []
        }
        phrase = sentence['sentence']
        aliases = sentence['aliases']
        spans = sentence['spans']
        if self.args.gold_given:
            qids = sentence['qids']
        else:
            qids = ["Q-1" for _ in range(len(sentence['aliases']))]
        gold = sentence.get('gold', [True for _ in range(len(sentence['aliases']))])

        for i in range(len(aliases)):
            alias = prep_utils.get_lnrm(aliases[i])
            if not alias in self.alias2qid:
                if self.args.verbose:
                    print(f"{alias} not in alias map. True QID: {qids[i]}")
                errors.append({
                    'alias': alias,
                    'qid': qids[i],
                    'sentence': sentence['sentence'],
                    'label': 'alias not found',
                })
                no_alias_exists += 1
                continue

            candidates = self.alias2qid[alias]
            qid_to_add = qids[i]
            # Take any qids to represent the gold QID for this data
            if not self.args.gold_given:
                qid_to_add = list(candidates)[0]
            if qid_to_add in candidates:
                assert int(spans[i][1]) <= len(sentence['sentence'].split()), sentence
                new_sentence['aliases'].append(alias)
                new_sentence['qids'].append(qid_to_add)
                new_sentence['spans'].append(spans[i])
                new_sentence['gold'].append(gold[i])
                success += 1
            else:
                if self.args.verbose:
                    print(f"Alias: {alias}, True QID: {qid_to_add}, Candidates in map: {candidates}")
                errors.append({
                    'alias': alias,
                    'candidates': list(candidates),
                    'qid': qid_to_add,
                    'sentence': sentence['sentence'],
                    'label': 'candidate not found',
                })
                qid_not_in_cands += 1

        # return number of aliases kept, and dropped
        return new_sentence, success, no_alias_exists, qid_not_in_cands, errors


class Contextual:

    def __init__(self, args, entity_dump, qid2page, alias_map, alias_tri, title_map):

        self.args = args
        self.entity_dump = entity_dump
        self.filtered_alias_map = alias_map
        self.title_map = title_map
        self.alias_tri = alias_tri
        self.qid2page = qid2page
        self.new_alias2qid = {}
        self.saved_metrics = defaultdict(int)
        self.detailed_metrics = defaultdict(list)

    def convert_char_to_word_boundaries(self, phrase, start_char, end_char):
        start_word = max(0, len(phrase[:start_char].split()))
        end_word = max(0, len(phrase[:end_char].split()))
        return start_word, end_word

    def get_lnrm(self, s):
        """Convert a string to its lnrm form
        We form the lower-cased normalized version l(s) of a string s by canonicalizing
        its UTF-8 characters, eliminating diacritics, lower-casing the UTF-8 and
        throwing out all ASCII-range characters that are not alpha-numeric.
        from http://nlp.stanford.edu/pubs/subctackbp.pdf Section 2.3
        Args:
            input string
        Returns:
            the lnrm form of the string
        """
        lnrm = unicodedata.normalize('NFD', str(s))
        lnrm = lnrm.lower()
        lnrm = ''.join([x for x in lnrm if (not unicodedata.combining(x)
                                            and x.isalnum() or x == ' ')]).strip()
        # will remove if there are any duplicate white spaces e.g. "the  alias    is here"
        lnrm = " ".join(lnrm.split())
        return lnrm

    def get_proper_nouns(self, text):
        words = text.split()
        tagged_text = pos_tag(words)
        # all_nouns = [word for word,pos in tagged_text if pos[:2] in ['NN'] and (word.lower() not in stopwords) and (word.lower().rstrip("s") not in stopwords)]
        all_nouns = []
        #
        # doc = nlp(text)
        # ent_idx = 0
        # while ent_idx < len(doc.ents):
        #     ent = doc.ents[ent_idx]
        #     ent_text = ent.text
        #     ent_idx += 1
        #     print("ENT TX", ent_text)
        i = 0
        nouns = []
        while i < len(words):
            if words[i][0].isupper() and words[i].lower() not in stopwords:
                start_i = i
                i += 1
                while True and i < len(words):
                    if words[i][0].isupper():
                        i += 1
                    else:
                        break
                nouns.append(' '.join(words[start_i: i]))
                continue
            i += 1
        full_nouns = set(all_nouns)
        for noun in nouns:
            full_nouns.add(noun)
            for part in noun.split():
                full_nouns.add(part)
        return list(full_nouns)

    def get_proper_nouns_old(self, text):
        nouns = []
        words = text.split()
        start = 0
        i = 0
        while i < len(words):
            if words[i][0].isupper():
                start_i = i
                i += 1
                while True and i < len(words):
                    if words[i][0].isupper():
                        i += 1
                    else:
                        break
                nouns.append(' '.join(words[start_i: i]))
                continue
            i += 1
        full_nouns = []
        for noun in nouns:
            full_nouns.append(noun)
            for part in noun.split():
                full_nouns.append(part)
        return list(set(full_nouns))

    def get_sub_alias_match(self, alias, alias2qids):
        assert len(alias.strip()) > 0, f"The alias is empty. This shouldn't happen."
        words = alias.split()
        subseqs = set()
        for i in range(len(words)):
            for j in range(i, len(words)+1):
                phrase = ' '.join(words[i:j])
                if len(phrase) > 0:
                    subseqs.add(phrase)
        subseqs = list(subseqs)
        subseqs.sort(key=lambda s: len(s), reverse=True)
        qid2count = defaultdict(int)
        curr_len = len(subseqs[0].split())
        for alias in subseqs:
            if len(alias.split()) != curr_len:
                if len(qid2count) > 0:
                    break
                curr_len = len(alias.split())
            if alias in alias2qids:
                for pair in alias2qids[alias]:
                    qid2count[pair[0]] = pair[1] + qid2count[pair[0]]
        candidates = [[qid, value] for qid, value in qid2count.items() if value > 0]
        return candidates

    def get_sub_alias_match2(self, alias, alias2qids):
        assert len(alias.strip()) > 0, f"The alias is empty. This shouldn't happen."
        words = alias.split()
        subseqs = set()
        for i in range(len(words)):
            for j in range(i, len(words)+1):
                phrase = ' '.join(words[i:j])
                if len(phrase) > 0 and len(phrase.split()) > 0:
                    subseqs.add(phrase)
        if len(subseqs) <= 0:
            return []
        subseqs = list(subseqs)
        subseqs.sort(key=lambda s: len(s), reverse=True)
        qid2count = defaultdict(int)
        curr_len = len(subseqs[0].split())
        for alias in subseqs:
            if len(alias.split()) != curr_len:
                if len(qid2count) > 0:
                    break
                curr_len = len(alias.split())
            if alias in alias2qids:
                for pair in alias2qids[alias]:
                    qid2count[pair[0]] = pair[1] + qid2count[pair[0]]
        candidates = [[qid, value] for qid, value in qid2count.items() if value > 0]
        return candidates

    def get_qids(self, target_alias, alias2qids):
        if target_alias in alias2qids:
            return alias2qids[target_alias]

        if target_alias.replace(" ", "") in alias2qids:
            return alias2qids[target_alias.replace(" ", "")]

        return self.get_sub_alias_match2(target_alias, alias2qids)


    def recompute_qid_rank(self, sample, alias, true_qid, qid2page, alias2qid, entity_dump):
        qid_pairs = self.get_qids(alias, alias2qid)
        if len(qid_pairs) == 0:
            self.saved_metrics["no_alias"] += 1
            self.detailed_metrics["no_alias"].append(
                {
                    "true_qid": true_qid,
                    "alias": alias
                }
            )
        if len(qid_pairs) <= 1:
            self.saved_metrics["len_1_alias"] += 1
            return qid_pairs
        # Each element in qid_pairs is [qid, qid_score]
        qid_pairs.sort(key=lambda x:x[1], reverse=True)
        context = sample['sentence']
        # Extract proper (capitalized) nouns from sentence
        prop_nouns = [word.lower() for word in self.get_proper_nouns(context) if word.lower() != alias]
        qids = [x[0] for x in qid_pairs]
        saved_counts = {}
        # Count number of times proper noun appears on Wikipedia page
        for i, qid in enumerate(qids):
            count = {p: 0 for p in prop_nouns}
            if qid in qid2page:
                for sentence in qid2page[qid]['sentences']:
                    text = " " + sentence['sentence'].lower() + " "
                    for term in prop_nouns:
                        if f" {term} " in text:
                            count[term] += 1
            saved_counts[qid] = count
        counts = [x[1] for x in qid_pairs]
        # Compute score for qid based on counts of proper nouns
        for i, qid in enumerate(qids):
            score = 0
            num_hits = 0
            for w in list(saved_counts[qid].keys()):
                x = saved_counts[qid][w]
                if x > 0:
                    num_hits += 1
                    # Heuristic threshold function
                    score += min(x, 5)*len(w.split())
            counts[i] += 10000*score
        # TODO: fix if acronym
        # SCORE BASED ON ALIAS TO ENTITY TITLE (partial)
        for i, qid in enumerate(qids):
            title = self.get_lnrm(entity_dump.get_title(qid).lower())
            title_score = fuzz.partial_ratio(title, alias)/100
            counts[i] += 100000*title_score
        # ADD TYPE SCORE

        scored_qids = [list(x) for x in zip(qids, counts)]
        scored_qids = sorted(scored_qids, key= lambda x: x[1], reverse = True)
        # Take top 5 most popular qids and remaining 25 from scored qids
        final_scored_qids = qid_pairs[:5]
        only_qids = [x[0] for x in final_scored_qids]
        for qid_pair in scored_qids:
            if len(final_scored_qids) == self.args.max_candidates:
                break
            if qid_pair[0] not in only_qids:
                only_qids.append(qid_pair[0])
                final_scored_qids.append(qid_pair)

        if self.args.gold_given:
            if true_qid not in only_qids[:self.args.max_candidates]:
                if true_qid in [x[0] for x in scored_qids]:
                    self.saved_metrics["in_long_cands_not_short"] += 1
                    self.detailed_metrics["in_long_cands_not_sort"].append(
                        {
                            "true_qid": true_qid,
                            "alias": alias,
                            "context": context,
                            "nouns": prop_nouns,
                            "saved_counts": saved_counts,
                            "scored_qids": scored_qids,
                            "titles": [entity_dump.get_title(q) for q in only_qids]
                        }
                    )
#                     print("TRUE QID", true_qid, entity_dump.get_title(true_qid))
#                     print("CONTEXT", context)
#                     print("NOUNS", prop_nouns)
#                     print("COUNTS", saved_counts[true_qid])
#                     for tqid in only_qids[:self.args.max_candidates]:
#                         print("OTHER COUNTS", tqid, entity_dump.get_title(tqid),  saved_counts[tqid])
#                     print("SCORES", scored_qids[:50])
#                     print("TITLES", [entity_dump.get_title(q) for q in only_qids[:50]])
                    # print("ALL SENTENCES", qid2page[true_qid]['sentences'])
                else:
                    self.saved_metrics["not_long_cands_not_short"] += 1
                    self.detailed_metrics["not_long_cands_not_sort"].append(
                        {
                            "true_qid": true_qid,
                            "alias": alias,
                            "context": context,
                            "nouns": prop_nouns,
                            "saved_counts": saved_counts,
                            "scored_qids": scored_qids,
                            "titles": [entity_dump.get_title(q) for q in only_qids]
                        }
                    )
            else:
                self.saved_metrics["success"] += 1
        return final_scored_qids[:self.args.max_candidates]

    def filter_sentence(self, sentence):
        success = 0
        no_alias_exists = 0
        qid_not_in_cands = 0
        no_qid_in_dump = 0
        errors = []
        new_sentence = {
            'sentence': sentence['sentence'],
            'sent_idx_unq': sentence['sent_idx_unq'],
            'aliases': [],
            'spans': [],
            'qids': [],
            'gold': [],
            'cands': []
        }
        phrase = sentence['sentence']
        aliases = sentence['aliases']
        spans = sentence['spans']
        # Uses NER tagger to expand aliases
        if self.args.expand_aliases:
            doc = nlp(phrase)
            al_idx = 0
            ent_idx = 0
            while ent_idx < len(doc.ents) and al_idx < len(aliases):
                ent = doc.ents[ent_idx]
                ent_text = ent.text
                ent_start_char = ent.start_char
                ent_end_char = ent.end_char
                # remove starting "the" and trailing "'s"
                if ent_text.startswith("the "):
                    ent_text = ent_text[4:]
                    ent_start_char += 4
                if ent_text.endswith(" 's"):
                    ent_text = ent_text[:-3]
                    ent_end_char -= 3
                start_span, end_span = self.convert_char_to_word_boundaries(phrase, ent_start_char, ent_end_char)
                cur_alias = aliases[al_idx]
                cur_spans = spans[al_idx]
                is_bleeding_into_next = False
                if al_idx+1 < len(spans):
                    next_cur_start_span = spans[al_idx+1][0]
                    is_bleeding_into_next = end_span >= next_cur_start_span
                if start_span >= cur_spans[1]:
                    al_idx += 1
                elif start_span <= cur_spans[0] and end_span >= cur_spans[1] and not is_bleeding_into_next and aliases[al_idx] != ent_text.lower():
#                     print("Swapping", start_span, end_span, cur_spans, cur_alias, "**", ent_text, "**", phrase)
                    spans[al_idx] = [start_span, end_span]
                    aliases[al_idx] = ent_text
                    al_idx += 1
                    ent_idx += 1
                else:
                    ent_idx += 1
        if self.args.gold_given:
            qids = sentence['qids']
        else:
            qids = ["Q-1" for _ in range(len(sentence['aliases']))]
        gold = sentence.get('gold', [True for _ in range(len(sentence['aliases']))])
        sent_idx_unq = sentence['sent_idx_unq']
        for i in range(len(aliases)):
            alias = prep_utils.get_lnrm(aliases[i])
            candidates = self.recompute_qid_rank(sentence, alias, qids[i], self.qid2page, self.filtered_alias_map, self.entity_dump)
            #######
            new_sentence["cands"].append(candidates)
            # With no golds and not nil model, need to set QIDs to be something in the dump; nil models can have Q-1
            qid_to_add = qids[i]
#             if not self.args.gold_given and len(candidates) > 0:
#                 qid_to_add = candidates[0][0]
            #print(f"Alias: {alias}, Candidates {candidates}")
            if len(candidates) == 0:
                # print(f"{alias} not in alias map. True QID: {qids[i]}")
                errors.append({
                    'alias': alias,
                    'qid': qid_to_add,
                    'sentence': sentence['sentence'],
                    'label': 'alias not found',
                })
                no_alias_exists += 1
                continue
            if self.args.gold_given and not self.entity_dump.qid_exists(qid_to_add):
                print(f"QID {qid_to_add} doesn't exist in entity dump for {alias}.")
                no_qid_in_dump += 1
                continue
            if qid_to_add in [x[0] for x in candidates]:
                assert int(spans[i][1]) <= len(sentence['sentence'].split()), sentence
                alias = f"{alias}_{sent_idx_unq}"
                self.new_alias2qid[alias] = candidates
                new_sentence['aliases'].append(alias)
                new_sentence['qids'].append(qid_to_add)
                new_sentence['spans'].append(spans[i])
                new_sentence['gold'].append(gold[i])
                success += 1
            else:
                # print(f"Alias: {alias}, True QID: {qids[i]}, Candidates in map: {candidates}")
                errors.append({
                    'alias': alias,
                    'candidates': list(candidates),
                    'qid': qid_to_add,
                    'sentence': sentence['sentence'],
                    'label': 'candidate not found',
                })
                assert int(spans[i][1]) <= len(sentence['sentence'].split()), sentence
                alias = f"{alias}_{sent_idx_unq}"
                self.new_alias2qid[alias] = candidates
                new_sentence['aliases'].append(alias)
                new_sentence['qids'].append(qid_to_add)
                new_sentence['spans'].append(spans[i])
                new_sentence['gold'].append(gold[i])
                qid_not_in_cands += 1

        # return number of aliases kept, and dropped
        return new_sentence, success, no_alias_exists, qid_not_in_cands, no_qid_in_dump, errors

class AIDACand:

    def __init__(self, args, entity_dump):

        self.args = args
        self.entity_dump = entity_dump
        self.load_aida_candidates()
        self.new_alias2qid = self.entity_dump._alias2qids

    def load_aida_candidates(self):
        print("Resetting alias2qid map in entity dump to be aida candidate map.")
        qids_dropped = set()
        qids_kept = set()
        alias2qid = json.load(open(self.args.aida_candidates))
        filtered_alias2qid = {}
        for alias, qid_pairs in alias2qid.items():
            filtered = []
            for qid, count in qid_pairs:
                if self.entity_dump.qid_exists(qid):
                    qids_kept.add(qid)
                    filtered.append([qid, count])
                else:
                    qids_dropped.add(qid)
            filtered_alias2qid[alias] = filtered
        self.entity_dump._alias2qids = filtered_alias2qid
        print(f"Finished! Dropped {len(qids_dropped)} qids. Kept {len(qids_kept)} qids.")

        # now set alias2qid and qid2alias
        self.qid2alias = defaultdict(set)
        self.alias2qid = defaultdict(set)
        for alias in filtered_alias2qid:
            for qid, _ in filtered_alias2qid[alias]:
                self.qid2alias[qid].add(alias)
                self.alias2qid[alias].add(qid)

    def filter_sentence(self, sentence):
        success = 0
        no_alias_exists = 0
        no_qid_in_dump = 0
        qid_not_in_cands = 0
        errors = []
        new_sentence = {
            'sentence': sentence['sentence'],
            'sent_idx_unq': sentence['sent_idx_unq'],
            'aliases': [],
            'spans': [],
            'qids': [],
            'gold': [],
            'doc_id': sentence['doc_id']
        }
        phrase = sentence['sentence']
        aliases = sentence['aliases']
        spans = sentence['spans']
        qids = sentence['qids']
        gold = sentence['gold']

        for i in range(len(aliases)):
            # The aida aliases are not-normalized
            alias = aliases[i]
            if not alias in self.alias2qid:
                if self.args.verbose:
                    print(f"{alias} not in aliase map. True QID: {qids[i]}")
                errors.append({
                    'alias': alias,
                    'qid': qids[i],
                    'sentence': sentence['sentence'],
                    'label': 'alias not found',
                    'other_aliases': ','.join(self.qid2alias[qids[i]])
                })
                no_alias_exists += 1
                continue
            if not self.entity_dump.qid_exists(qids[i]):
                no_qid_in_dump += 1
                print(f"QID {qids[i]} doesn't exist in entity dump.")
            candidates = self.alias2qid[alias]
            if qids[i] in candidates:
                success += 1
                assert int(spans[i][1]) <= len(sentence['sentence'].split()), sentence
                new_sentence['aliases'].append(alias)
                new_sentence['qids'].append(qids[i])
                new_sentence['spans'].append(spans[i])
                new_sentence['gold'].append(gold[i])
            else:
                print(f"Alias: {alias}, True QID: {qids[i]}, Candidates in map: {candidates}")
                errors.append({
                    'alias': alias,
                    'candidates': list(candidates),
                    'qid': qids[i],
                    'sentence': sentence['sentence'],
                    'label': 'candidate not found',
                    'other_aliases': ','.join(self.qid2alias[qids[i]])
                })
                # assert int(spans[i][1]) <= len(sentence['sentence'].split()), sentence
                # new_sentence['aliases'].append(alias)
                # new_sentence['qids'].append(qids[i])
                # new_sentence['spans'].append(spans[i])
                # new_sentence['gold'].append(gold[i])
                qid_not_in_cands += 1
        # return number of aliases kept, and dropped
        return new_sentence, success, no_alias_exists, qid_not_in_cands, no_qid_in_dump, errors