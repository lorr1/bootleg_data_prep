import unicodedata
from collections import defaultdict

import json

import nltk
import spacy
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from tqdm import tqdm

ps = PorterStemmer()
nlp = spacy.load("en_core_web_sm")
stopwords = set(stopwords.words('english'))

import bootleg_data_prep.utils.data_prep_utils as prep_utils

def spacy_to_split_remapping(tokens, spacy_tokens):
    norm_cur = 0
    spacy_cur = 0
    spacy_remap = [0] * len(spacy_tokens)
    spacy_tokens = [str(token) for token in spacy_tokens]
    cur_str = ""

    while norm_cur < len(tokens) and spacy_cur < len(spacy_tokens):
        if tokens[norm_cur] == spacy_tokens[spacy_cur]:
            spacy_remap[spacy_cur] = norm_cur
            spacy_cur+=1
            norm_cur+=1
        else:
            if tokens[norm_cur].startswith(cur_str + spacy_tokens[spacy_cur]):
                if tokens[norm_cur] == cur_str + spacy_tokens[spacy_cur]:
                    spacy_remap[spacy_cur] = norm_cur
                    spacy_cur+=1
                    norm_cur+=1
                    cur_str = ""
                else:
                    spacy_remap[spacy_cur] = norm_cur
                    cur_str += spacy_tokens[spacy_cur]
                    spacy_cur+=1
    return spacy_remap

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

    def get_lnrm(self, s, stripandlower):
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
        if not stripandlower:
            return s
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
            for j in range(i, len(words) + 1):
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
        wnl = nltk.WordNetLemmatizer()
        assert len(alias.strip()) > 0, f"The alias is empty. This shouldn't happen."
        words = alias.split()
        subseqs = set()
        if wnl.lemmatize(words[-1]) != words[-1]:
            subseqs.add(' '.join(words[:-1] + [wnl.lemmatize(words[-1])]))
        for i in range(len(words)):
            for j in range(i, len(words) + 1):
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
        qid_pairs.sort(key=lambda x: x[1], reverse=True)
        context = sample['sentence']
        prop_nouns = [word.lower() for word in self.get_proper_nouns(context) if word.lower() != alias]
        qids = [x[0] for x in qid_pairs]
        saved_counts = {}
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
                    score += min(x, 5) * len(w.split())
            counts[i] += 10000 * score
        # TODO: fix if acronym
        #         print(counts)
        #         ddd =[]
        #         eee = []
        #         for i, qid in enumerate(qids):
        #             title = self.get_lnrm(entity_dump.get_title(qid).lower())
        #             title_score = fuzz.partial_ratio(title, alias)/100
        #             title_score2 = nltk.jaccard_distance(set(alias.split()), set(title.split()))
        #             ddd.append(100000*title_score)
        #             eee.append(100*title_score2)
        #         print(ddd)
        #         print(eee)

        for i, qid in enumerate(qids):
            title = self.get_lnrm(entity_dump.get_title(qid).lower())
            title_score = fuzz.partial_ratio(title, alias) / 100
            counts[i] += 100000 * title_score
            title_score2 = 1 - nltk.jaccard_distance(set(alias.split()), set(title.split()))
            #             title_score2 = fuzz.ratio(title, alias)/100
            counts[i] += 10000 * title_score2
        #         print(counts)
        #         print([self.get_lnrm(entity_dump.get_title(qid).lower()) for i, qid in enumerate(qids)])

        scored_qids = [list(x) for x in zip(qids, counts)]
        scored_qids = sorted(scored_qids, key=lambda x: x[1], reverse=True)
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

    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def filter_sentence(self, sentence):
        print()
        print()
        print(sentence['sent_idx_unq'])
        #         print(sentence)
        total_cnt = 0
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
            'gold': []
        }

        phrase = sentence['sentence']
        aliases = sentence['aliases']
        spans = sentence['spans']
        if self.args.expand_aliases:
            #             print("PRINTING ALL QIDS:\n")
            #             for i in range(len(aliases)):
            #                 if sentence['qids'][i] in self.title_map:
            #                     print(sentence['qids'][i], self.title_map[sentence['qids'][i]], sentence['spans'][i], sentence['sentence'].split()[sentence['spans'][i][0]: sentence['spans'][i][1]])
            #                 else:
            #                     print(sentence['qids'][i], "None", sentence['spans'][i], sentence['sentence'].split()[sentence['spans'][i][0]: sentence['spans'][i][1]])
            #             print()

            doc = nlp(phrase)
            split_tokens = self.whitespace_tokenize(phrase)

            print("SANITY CHECK ALL ALIASES:\n")
            for i in range(len(aliases)):
                print("CHECKING>>>", aliases[i].replace(' ', '') in phrase.replace(' ', '').lower(),
                      phrase.replace(' ', '').lower().count(aliases[i].replace(' ', '')))

            for i in range(len(aliases)):
                if aliases[i].lower() != " ".join(split_tokens[spans[i][0]:spans[i][1]]).lower():
                    print("WARNING PREV>>>", aliases[i] in phrase.lower(), aliases[i].replace(' ', '') in phrase.replace(' ', '').lower(),
                          aliases[i], split_tokens[spans[i][0]:spans[i][1]])

            for i in range(len(aliases)):
                if aliases[i].lower() != " ".join(split_tokens[spans[i][0]:spans[i][1]]).lower():
                    st = spans[i][0]
                    ed = spans[i][1]
                    while st < len(split_tokens) and not split_tokens[st].lower().replace(' ', '').startswith(
                            aliases[i].lower().replace(' ', '')) and not aliases[i].lower().replace(' ', '').startswith(
                            split_tokens[st].lower().replace(' ', '')):
                        st += 1
                    while st > 0 and not split_tokens[st].lower().replace(' ', '').startswith(aliases[i].lower().replace(' ', '')) and not \
                    aliases[i].lower().replace(' ', '').startswith(split_tokens[st].lower().replace(' ', '')):
                        st -= 1
                    if ed < st + 1: ed = st + 1
                    while " ".join(split_tokens[st:ed]).lower().replace(' ', '').startswith(aliases[i].lower().replace(' ', '')) and " ".join(
                            split_tokens[st:ed]).lower().replace(' ', '') != aliases[i].lower().replace(' ', ''):
                        ed -= 1
                    while aliases[i].lower().replace(' ', '').startswith(" ".join(split_tokens[st:ed]).lower().replace(' ', '')) and " ".join(
                            split_tokens[st:ed]).lower().replace(' ', '') != aliases[i].lower().replace(' ', ''):
                        ed += 1
                    spans[i] = [st, ed]

            print("SPAN CORRECTION:\n")
            for i in range(len(aliases)):
                print("CHECKING>>>", aliases[i].replace(' ', '') in phrase.replace(' ', '').lower(),
                      phrase.replace(' ', '').lower().count(aliases[i].replace(' ', '')))

            for i in range(len(aliases)):
                if aliases[i].lower() != " ".join(split_tokens[spans[i][0]:spans[i][1]]).lower():
                    print("WARNING MID>>>", aliases[i] in phrase.lower(), aliases[i].replace(' ', '') in phrase.replace(' ', '').lower(),
                          aliases[i], split_tokens[spans[i][0]:spans[i][1]])

            spacy_tokens = [x for x in doc]
            spacy_to_split_remap = spacy_to_split_remapping(split_tokens, spacy_tokens)
            #             print(spacy_to_split_remap)

            #             print([(X.text, X.label_, X.start, X.end, split_tokens[spacy_to_split_remap[X.start]:spacy_to_split_remap[X.end]]) for X in doc.ents])
            ner_phrases = []
            for X in doc.ents:
                if spacy_to_split_remap[X.start] == spacy_to_split_remap[X.end]:
                    ner_phrases.append((spacy_to_split_remap[X.start], spacy_to_split_remap[X.end] + 1, X.label_))
                else:
                    ner_phrases.append((spacy_to_split_remap[X.start], spacy_to_split_remap[X.end], X.label_))

            flag = False
            for i in range(len(aliases)):
                if (sentence['spans'][i][0], sentence['spans'][i][1]) not in ner_phrases:
                    if sentence['qids'][i] in self.title_map:
                        if self.title_map[sentence['qids'][i]].lower() == ' '.join(
                                sentence['sentence'].split()[sentence['spans'][i][0]: sentence['spans'][i][1]]).lower():
                            continue
                        flag = True
                    else:
                        flag = True
            if flag: print(sentence)

            for i in range(len(aliases)):
                if (sentence['spans'][i][0], sentence['spans'][i][1]) not in ner_phrases:
                    if sentence['qids'][i] in self.title_map:
                        if self.title_map[sentence['qids'][i]].lower() == ' '.join(
                                sentence['sentence'].split()[sentence['spans'][i][0]: sentence['spans'][i][1]]).lower():
                            continue
                        #                         print(sentence)
                        #                         print(sentence['qids'][i], self.title_map[sentence['qids'][i]], sentence['spans'][i], sentence['sentence'].split()[sentence['spans'][i][0]: sentence['spans'][i][1]])
                        #                         print([(X.text, X.label_, X.start, X.end, split_tokens[spacy_to_split_remap[X.start]:spacy_to_split_remap[X.end]]) for X in doc.ents])
                        total_cnt += 1
                    else:
                        #                         print(sentence)
                        #                         print(sentence['qids'][i], "None", sentence['spans'][i], sentence['sentence'].split()[sentence['spans'][i][0]: sentence['spans'][i][1]])
                        #                         print([(X.text, X.label_, X.start, X.end, split_tokens[spacy_to_split_remap[X.start]:spacy_to_split_remap[X.end]]) for X in doc.ents])
                        total_cnt += 1
            #             print(spans, aliases)
            import copy
            ori_aliases = copy.deepcopy(aliases)

            #             prefix = ['the', "``", "-lrb-", "-rrb-"]
            #             suffix = ["'", "'s", "-lrb-", "-rrb-"]
            for i in range(len(aliases)):
                if (sentence['spans'][i][0], sentence['spans'][i][1]) in ner_phrases: continue
                for j in ner_phrases:
                    if j[0] <= sentence['spans'][i][0] and sentence['spans'][i][1] <= j[1]:
                        st = j[0]
                        ed = j[1]
                        check = False
                        for k in range(len(aliases)):
                            if i == k: continue
                            if len(set(list(range(st, ed))).intersection(set(list(range(sentence['spans'][k][0], sentence['spans'][k][1]))))) > 0:
                                check = True
                                break
                        if check: break
                        #                         if split_tokens[st].lower() in prefix: st += 1
                        #                         if ed-1>=0 and split_tokens[ed-1].lower() in suffix: ed -= 1
                        aliases[i] = " ".join(split_tokens[st:ed])
                        spans[i] = [st, ed]

            prefix = ['the', "``", "-lrb-", "-rrb-"]
            suffix = ["'", "'s", "-lrb-", "-rrb-"]
            for i in range(len(aliases)):
                #                 if (sentence['spans'][i][0], sentence['spans'][i][1]) in ner_phrases: continue
                #                 for j in ner_phrases:
                #                     if j[0] <= sentence['spans'][i][0] and sentence['spans'][i][1] <= j[1]:
                st = spans[i][0]
                ed = spans[i][1]
                if split_tokens[st].lower() in prefix: st += 1
                if ed - 1 >= 0 and split_tokens[ed - 1].lower() in suffix: ed -= 1
                aliases[i] = " ".join(split_tokens[st:ed])
                spans[i] = [st, ed]

            #             print(">>>>>>>>>>>>>", [x.lower() for x in ori_aliases] == [x.lower() for x in aliases], ori_aliases, aliases)
            #             print(split_tokens)
            #             print(ner_phrases)
            suf_comp = ["of", "'s"]
            pre_comp = ["of", "'s"]
            suf_comp = ["'s"]
            #             suf_comp = []
            for i in range(len(spans)):
                if split_tokens[spans[i][1]].lower() in suf_comp and spans[i][1] + 1 in [x[0] for x in ner_phrases]:
                    #                     print("@@@@@@@")
                    #                     print("ORIGINAL", aliases[i])
                    st = spans[i][0]
                    ed = spans[i][1]
                    for ner_phrase in ner_phrases:
                        if ner_phrase[0] == ed + 1 and ner_phrase[2] != 'GPE':
                            ed = ner_phrase[1]
                            break
                    aliases[i] = " ".join(split_tokens[st:ed])
                    spans[i] = [st, ed]
                #                     print("UPDATED", aliases[i])
                if spans[i][1] in [x[0] for x in ner_phrases]:
                    #                     print("@@@@@@@")
                    #                     print("ORIGINAL", aliases[i])
                    st = spans[i][0]
                    ed = spans[i][1]
                    for ner_phrase in ner_phrases:
                        if ner_phrase[0] == ed:
                            ed = ner_phrase[1]
                            break
                    aliases[i] = " ".join(split_tokens[st:ed])
                    spans[i] = [st, ed]
            #                     print("UPDATED", aliases[i])
            for i in range(len(spans)):
                if spans[i][0] > 0 and split_tokens[spans[i][0] - 1].lower() in pre_comp and spans[i][0] - 1 in [x[1] for x in ner_phrases]:
                    #                     print("@@@@@@@")
                    #                     print("ORIGINAL", aliases[i])
                    st = spans[i][0]
                    ed = spans[i][1]
                    for ner_phrase in ner_phrases:
                        if ner_phrase[1] == st - 1 and ner_phrase[2] == 'ORG':
                            st = ner_phrase[0]
                            break
                    aliases[i] = " ".join(split_tokens[st:ed])
                    spans[i] = [st, ed]
                #                     print("UPDATED", aliases[i])
                if spans[i][0] in [x[1] for x in ner_phrases]:
                    #                     print("@@@@@@@")
                    #                     print("ORIGINAL", aliases[i])
                    st = spans[i][0]
                    ed = spans[i][1]
                    for ner_phrase in ner_phrases:
                        if ner_phrase[0] == st:
                            st = ner_phrase[0]
                            break
                    aliases[i] = " ".join(split_tokens[st:ed])
                    spans[i] = [st, ed]
            #                     print("UPDATED", aliases[i])

            #             print("UPDATING..........")
            for i in range(len(aliases)):
                if (sentence['spans'][i][0], sentence['spans'][i][1]) not in ner_phrases:
                    if sentence['qids'][i] in self.title_map:
                        if self.title_map[sentence['qids'][i]].lower() == ' '.join(
                                sentence['sentence'].split()[sentence['spans'][i][0]: sentence['spans'][i][1]]).lower():
                            continue
                        #                         print(sentence)
                        #                         print(sentence['qids'][i], self.title_map[sentence['qids'][i]], sentence['spans'][i], sentence['sentence'].split()[sentence['spans'][i][0]: sentence['spans'][i][1]])
                        #                         print([(X.text, X.label_, X.start, X.end, split_tokens[spacy_to_split_remap[X.start]:spacy_to_split_remap[X.end]]) for X in doc.ents])
                        total_cnt += 1
                    else:
                        #                         print(sentence)
                        #                         print(sentence['qids'][i], "None", sentence['spans'][i], sentence['sentence'].split()[sentence['spans'][i][0]: sentence['spans'][i][1]])
                        #                         print([(X.text, X.label_, X.start, X.end, split_tokens[spacy_to_split_remap[X.start]:spacy_to_split_remap[X.end]]) for X in doc.ents])
                        total_cnt += 1

        #             print(spans, aliases)

        #             print()
        #             print()
        #             print(sentence['sentence'])
        #             for i in range(len(spans)):
        #                 if sentence['qids'][i] in self.title_map and aliases[i].lower() == self.title_map[sentence['qids'][i]].lower(): continue
        #                 if sentence['qids'][i] in self.title_map:
        #                     print("!!!!!!!!!!!!", sentence['qids'][i], self.title_map[sentence['qids'][i]], "_____", aliases[i])

        #             for i in range(len(aliases)):
        #                 if aliases[i].lower() != " ".join(split_tokens[spans[i][0]:spans[i][1]]).lower() or (sentence['qids'][i] in self.title_map and aliases[i].lower() != self.title_map[sentence['qids'][i]].lower()):
        #                     print("WARNING AFTER>>>", aliases[i] in phrase.lower(), aliases[i].replace(' ', '') in phrase.replace(' ', '').lower(), aliases[i], split_tokens[spans[i][0]:spans[i][1]], self.title_map[sentence['qids'][i]])

        #             al_idx = 0
        #             ent_idx = 0
        #             while ent_idx < len(doc.ents) and al_idx < len(aliases):
        #                 ent = doc.ents[ent_idx]
        #                 ent_text = ent.text
        #                 ent_start_char = ent.start_char
        #                 ent_end_char = ent.end_char
        #                 # remove starting "the" and trailing "'s"
        #                 if ent_text.startswith("the "):
        #                     ent_text = ent_text[4:]
        #                     ent_start_char += 4
        #                 if ent_text.endswith(" 's"):
        #                     ent_text = ent_text[:-3]
        #                     ent_end_char -= 3
        #                 start_span, end_span = self.convert_char_to_word_boundaries(phrase, ent_start_char, ent_end_char)
        #                 cur_alias = aliases[al_idx]
        #                 cur_spans = spans[al_idx]
        #                 is_bleeding_into_next = False
        #                 if al_idx+1 < len(spans):
        #                     next_cur_start_span = spans[al_idx+1][0]
        #                     is_bleeding_into_next = end_span >= next_cur_start_span
        #                 if start_span >= cur_spans[1]:
        #                     al_idx += 1
        #                 elif start_span <= cur_spans[0] and end_span >= cur_spans[1] and not is_bleeding_into_next and aliases[al_idx] != ent_text.lower():
        # #                     print("Swapping", start_span, end_span, cur_spans, cur_alias, "**", ent_text, "**", phrase)
        #                     spans[al_idx] = [start_span, end_span]
        #                     aliases[al_idx] = ent_text
        #                     al_idx += 1
        #                     ent_idx += 1
        #                 else:
        #                     ent_idx += 1
        if self.args.gold_given:
            qids = sentence['qids']
        else:
            qids = ["Q-1" for _ in range(len(sentence['aliases']))]
        gold = sentence.get('gold', [True for _ in range(len(sentence['aliases']))])
        sent_idx_unq = sentence['sent_idx_unq']
        new_sentence['cands'] = []
        for i in range(len(aliases)):
            self.temp += 1
            alias = self.get_lnrm(aliases[i])
            #             alias = prep_utils.get_lnrm(aliases[i])
            candidates = self.recompute_qid_rank(sentence, alias, qids[i], self.qid2page, self.filtered_alias_map, self.entity_dump)
            new_sentence['cands'].append(candidates)
            # With no golds and not nil model, need to set QIDs to be something in the dump; nil models can have Q-1
            qid_to_add = qids[i]
            #             if not self.args.gold_given and len(candidates) > 0:
            #                 qid_to_add = candidates[0][0]
            #             print(f"Alias: {alias}, Candidates {candidates}")
            if len(candidates) == 0:
                print(f"ERROR {alias} not in alias map. True QID: {qids[i]}")
                errors.append({
                    'alias': alias,
                    'qid': qid_to_add,
                    'sentence': sentence['sentence'],
                    'label': 'alias not found',
                })
                no_alias_exists += 1
                continue
            if self.args.gold_given and not self.entity_dump.qid_exists(qid_to_add):
                print(f"ERROR QID {qid_to_add} doesn't exist in entity dump for {alias}.")
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

        if len(new_sentence['aliases']) != len(ori_aliases):
            print("FINAL", ori_aliases, new_sentence['aliases'])
        #         print(total_cnt)
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