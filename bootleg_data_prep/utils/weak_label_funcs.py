import string

import marisa_trie
import nltk
from nltk.corpus import stopwords

from bootleg_data_prep.utils.data_prep_utils import get_lnrm

PUNC = string.punctuation
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add("also")
# Often left dangling in the sentence due to word splitting
STOPWORDS.add("s")
VERBS = set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
NOUNS = set(["NN", "NNS", "NNP", "NNPS", "PRP"])
TABLE = str.maketrans(dict.fromkeys(PUNC))  # OR {key: None for qkey in string.punctuation}


def regiater_funcs():
    all_funcs = {}

    def registrar(func):
        all_funcs[func.__name__] = func
        # normally a decorator returns a wrapped function, but here we return func unmodified, after registering it
        return func

    registrar.all = all_funcs
    return registrar


wl_func = regiater_funcs()

# ===================================================================
# UTILS
# ===================================================================
words_to_avoid = ["the", "a", "in", "of", "for", "at", "to", "with", "on", "from"]


def span_overlap(a, b):
    """
    Compute overlap of span where left is inclusive, right is exclusive.

    span_overlap([0, 3], [3, 5]) -> 0
    span_overlap([0, 3], [2, 5]) -> 1
    span_overlap([0, 5], [0, 2]) -> 2
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def find_aliases_in_sentence(sentence, all_aliases, max_alias_len, used_aliases=None):
    """
    Input: sentence, set or Tri of aliases, max alias length, list of [alias, span_l, span_r] for existing aliases, if any

    Output: list of [alias, span_l, span_r] for existing aliases and any n-grams from largest to smallest (starting at max alias length) that are
    in the list of all_aliases.
    """
    if used_aliases is None:
        used_aliases = []
    if len(all_aliases) == 0:
        return used_aliases
    table = str.maketrans(dict.fromkeys(PUNC))  # OR {key: None for key in string.punctuation}

    sentence_split_raw = sentence.split()
    tags = nltk.pos_tag(sentence_split_raw)
    # find largest aliases first
    for n in range(max_alias_len + 1, 0, -1):
        grams = nltk.ngrams(tags, n)
        w_st = -1
        w_end = n - 1
        for gram in grams:
            w_st += 1
            w_end += 1
            gram_words, gram_tags = zip(*gram)
            # If single word, must be noun (this will get rid of words like "the" or "as")
            if n == 1 and gram_tags[0] not in NOUNS:
                continue
            # For multi word aliases, make sure there is a noun in the phrase somewhere
            if n > 1 and not any(n in gram_tags for n in NOUNS):
                continue
            # If gram starts with stop word, move on because we'd rather try the one without
            # We also don't want punctuation words to be used at the beginning/end
            if gram_words[0] in words_to_avoid or gram_words[-1] in words_to_avoid or len(gram_words[0].translate(table).strip()) == 0 \
                    or len(gram_words[-1].translate(table).strip()) == 0:
                continue
            gram_attempt = get_lnrm(" ".join(gram_words), strip=True, lower=True)
            # print("NOLRNSM", " ".join(gram_words), "-- GA", gram_attempt, w_st, "to", w_end, "-- in aliases --", gram_attempt in all_aliases)
            if gram_attempt in all_aliases:
                keep = True
                # We start from the largest n-grams and go down in size. This prevents us from adding an alias that is a subset of another.
                # For example: "Tell me about the mother on how I met you mother" will find "the mother" as alias and "mother". We want to
                # only take "the mother" and not "mother" as it's likely more descriptive of the real entity.
                for u_al in used_aliases:
                    u_w_st = u_al[2]
                    u_w_end = u_al[3]
                    if span_overlap([w_st, w_end], [u_w_st, u_w_end]) > 0:
                        keep = False
                        break
                if not keep:
                    continue
                # print("Adding", gram_attempt, w_st, w_end)
                used_aliases.append(tuple([gram_attempt, "Q-1", w_st, w_end]))
    # sort based on closeness to alias
    sorted_aliases = sorted(used_aliases, key=lambda elem: [elem[2], elem[3]])
    for i in range(len(sorted_aliases) - 1):
        left = sorted_aliases[i]
        right = sorted_aliases[i + 1]
        assert span_overlap([left[2], left[3]], [right[2], right[3]]) == 0
    return sorted_aliases


def golds(doc_qid, sentence, spans, qids, aliases, document_alias2qids, wl_metadata):
    final_spans, final_qids, final_aliases = [], [], []
    return final_spans, final_qids, final_aliases


@wl_func
def aka(doc_qid, sentence, spans, qids, aliases, document_alias2qids, wl_metadata):
    if len(spans) > 0:
        spans_l, spans_r = list(zip(*spans))
        used_aliases = list(zip(aliases, qids, spans_l, spans_r))
    else:
        used_aliases = []
    doc_aliases = wl_metadata.get_all_aliases(doc_qid, set())
    res_item = find_aliases_in_sentence(sentence, marisa_trie.Trie(document_alias2qids.keys()), max_alias_len=8, used_aliases=used_aliases)

    final_spans, final_qids, final_aliases = [], [], []
    for al, q, sp_l, sp_r in res_item:
        if q == "Q-1":
            if al in doc_aliases and document_alias2qids[al] == doc_qid:
                final_spans.append([sp_l, sp_r])
                final_qids.append(doc_qid)
                final_aliases.append(al)
    return final_spans, final_qids, final_aliases
