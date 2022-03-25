import string
import unicodedata
import nltk
from nltk import tokenize
from nltk import corpus

PUNC = string.punctuation
PUNC_TRANSLATION_TABLE = str.maketrans(dict.fromkeys(PUNC))  # OR {key: None for key in string.punctuation}
BASE_STOPWORDS = {"的","一","不","在","人","有","是","为","以","于","上","他","而","后","之","来","及","了","因",
                  "下","可","到","由","这","与","也","此","但","并","个","其","已","无","小","我","们","起","最",
                  "再","今","去","好","只","又","或","很","亦","某","把","那","你","乃","它","吧","被","比","别",
                  "趁","当","从","得","打","凡","儿","尔","该","各","给","跟","和","何","还","即","几","既","看",
                  "据","距","靠","啦","另","么","每","嘛","拿","哪","您","凭","且","却","让","仍","啥","如","若",
                  "使","谁","虽","随","同","所","她","哇","嗡","往","些","向","沿","哟","用","咱","则","怎","曾",
                  "至","致","着","诸","自"}
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']
ENSURE_ASCII = False
CATEGORY_LINE_START = '[[Category:'
CATEGORY_LINE_CAPTURE = r'\[\[Category:([^\|]+).*\]\].*'
# https://en.wikipedia.org/wiki/Wikipedia:Namespace
STANDARD_NAMESPACE = {"category", "user", "help", "portal", "draft", "module", "file", "wikipedia", "wiktionary",
                      "wikt", "wp", "wt", "w", "cat", "image", "special", "template", "talk", "centralwikia",
                      "s", "creativecommons", "wikisource"}

##
# Recognize only these namespaces in links
# w: Internal links to the Wikipedia
# wiktionary: Wiki dictionary
# wikt: shortcut for Wiktionary
#
ACCEPTED_NAMESPACE = {'w', 'wiktionary', 'wikt'}

def sent_tokenize(sent):
    return tokenize.sent_tokenize(sent)

def sent_offset_tokenize(sent):
    return tokenize.punkt.PunktSentenceTokenizer().span_tokenize(sent)

def word_offset_tokenize(sent):
    return tokenize.WhitespaceTokenizer().span_tokenize(sent)

def word_tokenize(sent):
    return tokenize.word_tokenize(sent)

def pos_tag(tokens):
    return nltk.pos_tag(tokens)

def ngrams(tags, n):
    return nltk.ngrams(tags, n)

def get_lnrm(s, strip, lower):
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
    if not strip and not lower:
        return s
    lnrm = str(s)
    if lower:
        lnrm = lnrm.lower()
    if strip:
        lnrm = unicodedata.normalize('NFD', lnrm)
        lnrm = ''.join([x for x in lnrm if (not unicodedata.combining(x)
                                            and x.isalnum() or x == ' ')]).strip()
    # will remove if there are any duplicate white spaces e.g. "the  alias    is here"
    lnrm = " ".join(lnrm.split())
    return lnrm

class HumanNameParser:
    def __init__(self, name):
        self.last = ""
        self.first = name