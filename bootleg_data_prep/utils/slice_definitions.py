import collections
import itertools

import marisa_trie
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from itertools import combinations
from collections import defaultdict
import string

import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.utils.constants import TAILONLY, TORSOONLY, \
    HEADONLY, TOESONLY

PUNC = string.punctuation
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add("also")
# Often left dangling in the sentence due to word splitting
STOPWORDS.add("s")

ps = PorterStemmer()

def format_relation_pair(head, tail):
    return f"{head}_{tail}"

def inv_relation_pair(pair_str):
    return pair_str.split("_")

def get_pair_key(pair):
    return f"{pair[0]}_{pair[1]}"

def get_shared_types(qidl, qidr, esp):
    if (qidl == qidr) or (not (esp.qid_in_qid2typeid_wd(qidl))) or (not (esp.qid_in_qid2typeid_wd(qidr))):
        return set()
    shared_true_types = set(esp.get_types_wd(qidl))
    shared_true_types = shared_true_types.intersection(set(esp.get_types_wd(qidr)))
    return shared_true_types

def get_test_stats_dict(args, alias_qid, entity_symbols_plus):
    # We first need to gather alias-entity pairs that meet specified criteria below for slices
    test_stats_dict = {}
    tail_qids_pairs = {alias: [qid for qid in entity_symbols_plus.get_qid_cands(alias) if entity_symbols_plus.get_qid_count(qid) < 11 and len(entity_symbols_plus.get_qid_cands(alias)) > 1] for alias in entity_symbols_plus.get_all_aliases()}
    tail_qids = defaultdict(set)
    for al in tail_qids_pairs:
        for qid in tail_qids_pairs[al]:
            tail_qids[al].add(qid)

    alias_qid_counter = collections.Counter({tuple([k, v]): alias_qid[k][v] for k in alias_qid for v in alias_qid[k] if len(alias_qid[k]) > 1})
    # hard to disambiguate if top two are close in count
    hard_to_disambig_pairs = {alias: entity_symbols_plus.get_qid_cands(alias)[:2]
                              for alias in entity_symbols_plus.get_all_aliases()
                              if (len(entity_symbols_plus.get_qid_cands(alias)) > 1
                                  and entity_symbols_plus.get_qid_count_cands(alias)[0][1] * 0.8 <= entity_symbols_plus.get_qid_count_cands(alias)[1][1])}
    hard_to_disambig = defaultdict(set)
    for al in hard_to_disambig_pairs:
        for qid in hard_to_disambig_pairs[al]:
            hard_to_disambig[al].add(qid)
    # skewed to disambiguate if has very tail entities
    skewed_disambig_pairs = {alias: entity_symbols_plus.get_qid_cands(alias)[1:]
                             for alias in entity_symbols_plus.get_all_aliases()
                             if (len(entity_symbols_plus.get_qid_cands(alias)) > 1 and entity_symbols_plus.get_qid_count_cands(alias)[1][1] >= 10
                                 and entity_symbols_plus.get_qid_count_cands(alias)[0][1] * 0.2 > entity_symbols_plus.get_qid_count_cands(alias)[1][1])}
    skewed_disambig = defaultdict(set)
    for al in skewed_disambig_pairs:
        for qid in skewed_disambig_pairs[al]:
            skewed_disambig[al].add(qid)

    # bottom half
    bottom_half_pairs = {alias: entity_symbols_plus.get_qid_cands(alias)[14:]
                             for alias in entity_symbols_plus.get_all_aliases()
                             if (len(entity_symbols_plus.get_qid_cands(alias))) > 15}
    bottom_half = defaultdict(set)
    for al in bottom_half_pairs:
        for qid in bottom_half_pairs[al]:
            bottom_half[al].add(qid)

    # popular
    popular_pairs = defaultdict(set)
    # Pair is ((al, qid), count)
    for pair in alias_qid_counter.most_common(args.topK):
        popular_pairs[pair[0][0]].add(pair[0][1])

    # least popular with at least 10
    least_popular_pairs = defaultdict(set)
    for pair in alias_qid_counter.most_common()[:-args.topK - 1:-1]:
        least_popular_pairs[pair[0][0]].add(pair[0][1])
    # long sentence
    long_sentences = 70
    # short sentences
    short_sentences = 10
    # unknown words
    test_stats_dict["K_most_popular"] = popular_pairs
    test_stats_dict["K_least_popular"] = least_popular_pairs
    test_stats_dict["tail_qids"] = tail_qids
    test_stats_dict["hard_to_disambig"] = hard_to_disambig
    test_stats_dict["skewed_disambig"] = skewed_disambig
    test_stats_dict["bottom_half"] = bottom_half
    test_stats_dict["long_sentences"] = long_sentences
    test_stats_dict["short_sentences"] = short_sentences
    return test_stats_dict

def get_kg_signal(qids, aliases, entity_symbols_plus):
    aliases_to_predict = []
    # Don't need to check full cross produce because relations are symmetric for the get_relations (not for the contextual_rel_pairs)
    for i, qid in enumerate(qids):
        # qid doesn't occur in a relation
        if not entity_symbols_plus.qid_in_rel_mapping(qid):
            continue
        heads_or_tails = entity_symbols_plus.get_relations(qid)
        # check for remaining qids in sentence
        for j in range(i+1, len(qids)):
            qid_other = qids[j]
            # Filter away matching qids; this shouldn't happen in a KG because an entity should not be connected to itself, but Wikidata is messy
            if qid_other == qid:
                continue
            if qid_other in heads_or_tails:
                aliases_to_predict.append(i)
                aliases_to_predict.append(j)
    aliases_to_predict = sorted(list(set(aliases_to_predict)))
    return aliases_to_predict

# For each integer in min_common_types, check if sequential pairs of QIDs share that integer number of types
# E.g. 1 type means they must share at least one type
# Returns a dict with min_common_type as key and aliases to predict as values
def exists_shared_true_type(qids, aliases, min_common_types, filter_types, entity_symbols_plus):
    alias_idx_per_type_cnt = {k:set() for k in min_common_types}
    # need at least three qids for this criteria
    # there may be zero qids due to dropping qids with golds of False
    if len(qids) <= 2:
        return alias_idx_per_type_cnt
    # take a three-way comparison of aliases to see if types match
    for subset_idx in range(0, len(qids)-2):
        left_qid = qids[subset_idx]
        left_i = subset_idx
        if not (entity_symbols_plus.qid_in_qid2typeid_wd(left_qid)):
            continue

        middle_qid = qids[subset_idx+1]
        middle_i = subset_idx+1
        # Filter away matching QIDs. This happens in sentences with pronoun coref such as "He said that his..."
        if middle_qid == left_qid:
            continue
        if not (entity_symbols_plus.qid_in_qid2typeid_wd(middle_qid)):
            continue

        right_qid = qids[subset_idx+2]
        right_i = subset_idx+2
        # Filter away matching QIDs. This happens in sentences with pronoun coref such as "He said that his..."
        if right_qid == left_qid or right_qid == middle_qid:
            continue
        if not (entity_symbols_plus.qid_in_qid2typeid_wd(right_qid)):
            continue

        shared_true_types = set(entity_symbols_plus.get_types_wd(left_qid))
        # If there is a filter and there are no types in the filter set, continue
        if len(filter_types) > 0 and len(filter_types.intersection(shared_true_types)) <= 0:
            continue
        shared_true_types = shared_true_types.intersection(set(entity_symbols_plus.get_types_wd(middle_qid)))
        shared_true_types = shared_true_types.intersection(set(entity_symbols_plus.get_types_wd(right_qid)))
        for k in min_common_types:
            if len(shared_true_types) >= k:
                alias_idx_per_type_cnt[k].add(left_i)
                alias_idx_per_type_cnt[k].add(middle_i)
                alias_idx_per_type_cnt[k].add(right_i)
    aliases_to_predict_final = {}
    for k in alias_idx_per_type_cnt:
        aliases_to_predict_final[k] = sorted(list(set(alias_idx_per_type_cnt[k])))
    return aliases_to_predict_final

def get_affordance_signal(qids, min_common_types, tokens, type_filter, type_affordance, check_type, get_types, entity_symbols_plus):
    # Stores weak vs strong consistency based on the number of typer per entity we consider to get affordances
    alias_idx_per_type_cnt = {k:set() for k in min_common_types}
    for i, qid in enumerate(qids):
        if not check_type(entity_symbols_plus, qid):
            continue
        types = sorted(list(get_types(entity_symbols_plus, qid)))
        # Vocab is a trie with values being a list of length 1 tuples
        sentence_ids = tokens[:]

        # If the type filter is not empty and there are no types in this intersection, continue
        if len(type_filter) > 0 and len(type_filter.intersection(types)) <= 0:
            continue
        # Get STRONG affordance with all three types, represented as the tuple
        if str(tuple(types)) in type_affordance:
            strong_affordance_ids = type_affordance[str(tuple(types))]
            # Record tie allows lists of objects, but as we know the max, we control the list outselves and have a single value map to a tuple
            assert len(strong_affordance_ids) == 1
            strong_affordance_ids = strong_affordance_ids[0]
            if len(set(strong_affordance_ids).intersection(sentence_ids)) > 0:
                alias_idx_per_type_cnt[3].add(i)
        # Get WEAK affordance with single types
        for type in types:
            if str(type) in type_affordance:
                weak_affordance_ids = type_affordance[str(type)]
                # Record tie allows lists of objects, but as we know the max, we control the list outselves and have a single value map to a tuple
                assert len(weak_affordance_ids) == 1
                weak_affordance_ids = weak_affordance_ids[0]
                if len(set(weak_affordance_ids).intersection(sentence_ids)) > 0:
                    alias_idx_per_type_cnt[1].add(i)
    aliases_to_predict_final = {}
    for k in alias_idx_per_type_cnt:
        aliases_to_predict_final[k] = sorted(list(set(alias_idx_per_type_cnt[k])))
    return aliases_to_predict_final

def get_same_types_among_cands(qids, aliases, entity_symbols_plus):
    aliases_to_predict_final = set()
    for i, (gold_qid, alias) in enumerate(zip(qids, aliases)):
        if not entity_symbols_plus.qid_in_qid2typeid_wd(gold_qid):
            continue
        cands = entity_symbols_plus.get_qid_cands(alias)
        if len(cands) <= 1:
            continue
        gold_types = set(entity_symbols_plus.get_types_wd(gold_qid))
        in_common = 0
        for cand in cands:
            if not entity_symbols_plus.qid_in_qid2typeid_wd(cand):
                in_common += 1
                continue
            types = set(entity_symbols_plus.get_types_wd(cand))
            if len(types.intersection(gold_types)) > 0:
                in_common += 1
        if in_common >= 0.6*len(cands):
            aliases_to_predict_final.add(i)
    return sorted(list(aliases_to_predict_final))


def get_same_relations_among_candidates(qids, aliases, entity_symbols_plus):
    aliases_to_predict_final = set()
    for i, (gold_qid, alias) in enumerate(zip(qids, aliases)):
        if not entity_symbols_plus.qid_in_rel_mapping(gold_qid):
            continue
        cands = entity_symbols_plus.get_qid_cands(alias)
        if len(cands) <= 1:
            continue
        for cand in cands:
            if cand == gold_qid:
                continue
            if not entity_symbols_plus.qid_in_rel_mapping(cand):
                continue
            if gold_qid in entity_symbols_plus.get_relations(cand) or cand in entity_symbols_plus.get_relations(gold_qid):
                aliases_to_predict_final.add(i)
    return sorted(list(aliases_to_predict_final))

def get_type_in_season_set(qids, entity_symbols_plus):
    types_of_intersect = {4, 12, 31, 41, 110, 277, 296, 361, 367, 431, 551, 611, 852, 874, 913, 936, 1035, 1465, 1507, 1601, 2128, 2372, 2569,
                          2737, 2808, 3061, 3094, 3514, 3662, 3788, 3984, 4068, 4550, 4662, 4817, 5017, 5029, 5492, 5508, 6074, 6162, 6547,
                          6626, 7176, 7177, 7180, 8611, 8992, 9459, 9817, 10398, 10460, 10629, 11687, 11958, 12475, 13111, 13223, 13373, 13617,
                          14191, 15039, 15218, 15367, 15625, 16294, 16605, 16733, 16789, 17398, 18284, 18364, 18427, 19014, 19332, 21686, 23319,
                          24464, 24688, 24794, 26161, 26834}
    aliases_to_predict_final = set()
    for i, qid in enumerate(qids):
        if not entity_symbols_plus.qid_in_qid2typeid_wd(qid):
            continue
        gold_types = set(entity_symbols_plus.get_types_wd(qid))
        if len(gold_types.intersection(types_of_intersect)) > 1:
            aliases_to_predict_final.add(i)
    return sorted(list(aliases_to_predict_final))


# For example: qids [Q1, Q2, Q3] and only Q2 had no type or kg, then output would be [1] as index into QIDs
def get_no_type_no_kg(qids, entity_symbols_plus):
    aliases_to_predict_final = set()
    for i, qid in enumerate(qids):
        no_wd_types = not entity_symbols_plus.qid_in_qid2typeid_wd(qid)
        no_rel_types = not entity_symbols_plus.qid_in_qid2typeid_rel(qid)
        no_kg_connections = not entity_symbols_plus.qid_in_rel_mapping(qid)
        if no_wd_types and no_rel_types and no_kg_connections:
            aliases_to_predict_final.add(i)
    return sorted(list(aliases_to_predict_final))


def get_no_type(qids, entity_symbols_plus):
    aliases_to_predict_final = set()
    for i, qid in enumerate(qids):
        if not (entity_symbols_plus.qid_in_qid2typeid_wd(qid)):
            aliases_to_predict_final.add(i)
    return sorted(list(aliases_to_predict_final))


def get_no_kg(qids, entity_symbols_plus):
    aliases_to_predict_final = set()
    for i, qid in enumerate(qids):
        if not entity_symbols_plus.qid_in_rel_mapping(qid):
            aliases_to_predict_final.add(i)
    return sorted(list(aliases_to_predict_final))


# Returns a true/false list on if QID has no textual cues or not. The TRUE QIDs are the one we keep.
def do_qids_have_no_textual_cues(qids, sentence_ids, entity_symbols_plus):
    entity_words = entity_symbols_plus.entity_words
    # Vocab is a trie with values being a list of length 1 tuples
    qids_to_ret = [False]*len(qids)
    if entity_words is None:
        entity_words = {}
    for i, qid in enumerate(qids):
        # May happen if QID is part of test/dev data; bag of words is over train only
        if qid not in entity_words:
            qids_to_ret[i] = True
            continue
        word_ids = entity_words[qid]
        # Record tie allows lists of objects, but as we know the max, we control the list ourselves and have a single value map to a tuple
        assert len(word_ids) == 1
        word_ids = word_ids[0]
        # Will return a tuple of word_ids; -1 means NULL word
        # ids = [w_id for w_id in word_ids if w_id != -1]
        # sentence_ids will never have a -1
        assert -1 not in sentence_ids
        if len(set(word_ids).intersection(sentence_ids)) <= 2:
            qids_to_ret[i] = True
    return qids_to_ret

def do_qids_meet_count(qids, entity_symbols_plus, tail_threshold, torso_threshold):
    qids_to_ret = {k: [False for _ in range(len(qids))] for k in [TAILONLY, TORSOONLY, HEADONLY, TOESONLY]}
    for i, qid in enumerate(qids):
        cnt = entity_symbols_plus.get_qid_count(qid)
        if cnt > torso_threshold:
            qids_to_ret[HEADONLY][i] = True
        elif cnt > tail_threshold:
            qids_to_ret[TORSOONLY][i] = True
        else:
            qids_to_ret[TAILONLY][i] = True
            if cnt <= 0:
                qids_to_ret[TOESONLY][i] = True
    return qids_to_ret
