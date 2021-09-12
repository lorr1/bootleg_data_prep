import string
import unicodedata
import nltk
from nltk import tokenize
from nltk import corpus
from nltk import PorterStemmer
from nameparser import HumanName

stopwords = corpus.stopwords
PUNC = string.punctuation
PUNC_TRANSLATION_TABLE = str.maketrans(dict.fromkeys(PUNC))  # OR {key: None for key in string.punctuation}
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add('also')
# Often left dangling in the sentence due to word splitting
STOPWORDS.add('s')
WORDS_TO_AVOID = ['the', 'a', 'in', 'of', 'for', 'at', 'to', 'with', 'on', 'from']
ignore_words = set(['The', 'the', 'Of', 'of', 'And', 'and', 'For', 'for', 'In', 'in', 'To', 'to']) # maybe also try adding 'with', 'a', 'on', etc. ??
VERBS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']
ensure_ascii = True

def sent_tokenize(sents):
    return tokenize.sent_tokenize(sents)

def word_tokenize(sent):
    return tokenize.word_tokenize(sent)

ps = PorterStemmer()

def stem(text):
    return ps.stem(text)

def pos_tag(tokens):
    return nltk.pos_tag(tokens)

def bigrams(tokens):
    return nltk.bigrams(tokens)

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

class HumanNameParser(HumanName):
    pass

pronoun_map = {
    'him': 1,
    "he": 1,
    "his": 1,
    "himself": 1,
    'her': 2,
    "she": 2,
    "herself": 2,
    'it': 3,
    "its": 3,
    "itself": 3,
    'they': 4,
    "them": 4,
    "themselves": 4,
}
pronoun_possessive_map = {
    'him': 0,
    "he": 0,
    "his": 1,
    "himself": 0,
    'her': 1,
    "she": 0,
    "herself": 0,
    'it': 0,
    "its": 0,
    "itself": 0,
    'they': 0,
    "them": 0,
    "themselves": 0
}
SINGULAR_MASCULINE = 1  # him, he, his, himself
SINGULAR_FEMININE = 2 # her, she, her, herself
SINGULAR_INPERSONAL = 3  # it, its, itself
PLURAL_PERSONAL = 4  # they, them, their, themselves
UNKNOWN = 5
gender_qid_map = {
    "novalue": 5,
    "Q1052281" : 5, # trans m->f
    "Q1097630" : 5, # intersex, both
    "Q1289754" : 5, # neutrois non-binary
    "Q12964198" : 5, # genderqueer
    "Q15145778" : 1, # cisgender male, assigned and identify male
    "Q15145779" : 2, # cisgender female, assigned and identify female
    "Q179294" : 1, # eunuch
    "Q18116794" : 5, # genderfluid
    "Q189125" : 5, # trans
    "Q207959" : 5, # androgyny, combine both traits
    "Q2333502" : 5, # skeleton
    "Q2449503" : 5, # trans f -> m
    "Q27679684" : 5, # transfeminine
    "Q27679766" : 5, # transmasculine
    "Q301702" : 5, # two spirit
    "Q3277905" : 5, # third
    "Q4233718" : 5, #annoymous
    "Q43445" : 2, # female organism
    "Q44148" : 1, # male organism
    "Q48270" : 5, # non-binary
    "Q499327" : 1, #masculine
    "Q505371" : 5, # agender
    "Q52261234" : 5, # neutral
    "Q6581072" : 2, # female
    "Q6581097" : 1, # male
    "Q660882" : 5, # hijra third
    "Q7130936" : 5, # pangender
    "Q859614" : 5, # bigender
    "somevalue": 5,
}
