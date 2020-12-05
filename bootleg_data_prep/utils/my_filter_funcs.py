# File for filter functions that either take the entire aliases2qids and qids2aliases dictionaries or a singe alias->qids or pageid->aliases item.
# True means filter
# False means keep
import re
import ujson as json
import itertools
import os

def prep_kg(args):
    all_triples_head = {}
    for trip in open(os.path.join(args.prep_file), "r").readlines():
        head, pred, tail = trip.split()
        if head not in all_triples_head:
            all_triples_head[head] = []
        all_triples_head[head].append({"head": head, "pred": pred, "tail": tail})
    to_keep = set()
    if args.filter_file != "":
        to_keep = set(json.load(open(args.filter_file, "r")))
    return {"all_triples_head": all_triples_head, "to_keep": to_keep}

def prep_standard(args):
    to_keep = set()
    if args.filter_file != "":
        res = json.load(open(args.filter_file, "r"))
        if type(res) is dict:
            res = res.keys()
        to_keep = set(res)
    return {"to_keep": to_keep}

# True means filter/remove
# False means keep

def true_filter(args, aliases, qids, sentence, extras):
    return True

def false_filter(args, aliases, qids, sentence, extras):
    return False

def long_sentence(phrase):
    if len(phrase.split(" ")) > 100:
        return True
    return False

def sentence_filter_short(args, aliases, qids, sentence, extras):
    return long_sentence(sentence)

def sentence_filterQID(args, aliases, qids, sentence, extras):
    qids_to_keep = extras['to_keep']
    discard = long_sentence(sentence)
    discard = discard | (len(set(qids).intersection(qids_to_keep)) == 0)
    return discard

def sentence_filterAliases(args, aliases, qids, sentence, extras):
    aliases_to_keep = extras['to_keep']
    discard = long_sentence(sentence)
    discard = discard | (len(set(aliases).intersection(aliases_to_keep)) == 0)
    return discard

def sentence_filterQIDMarriage(args, aliases, qids, sentence, extras):
    qids_to_keep = extras['to_keep']
    # Discard long sentences early to avoid large qid cross product
    discard = long_sentence(sentence)
    discard = discard | (len(set(qids).intersection(qids_to_keep)) == 0)
    if discard:
        return True

    all_triples_head = extras['all_triples_head']
    assert len(all_triples_head) > 0
    keep_by_marriage = False
    # Triples are not symmetric, so we need to do a full cross product here
    for (qid1, qid2) in itertools.product(*[qids, qids]):
        if qid1 == qid2:
            continue
        triples = all_triples_head.get(qid1, [])
        triples = list(filter(lambda trip: trip['tail'] == qid2 and trip['pred'] == 'P26', triples))
        if len(triples) > 0:
            keep_by_marriage = True
            break
    discard = not keep_by_marriage
    return discard