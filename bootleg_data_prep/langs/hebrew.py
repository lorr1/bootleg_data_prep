import re
import string
import unicodedata

import nltk
import stanza
from hebrew_tokenizer import tokenize
from hebrew_tokenizer.groups import Groups

PUNC = string.punctuation
PUNC_TRANSLATION_TABLE = str.maketrans(dict.fromkeys(PUNC))  # OR {key: None for key in string.punctuation}
STOPWORDS = {
    'את', 'לא', 'של', 'אני', 'על', 'זה', 'עם', 'כל', 'הוא', 'אם', 'או', 'גם', 'יותר', 'יש', 'לי', 'מה', 'אבל', 'אז', 'טוב', 'רק', 'כי', 'שלי', 'היה',
    'אין', 'עוד', 'היא', 'אחד', 'עד', 'לך', 'כמו', 'להיות', 'אתה', 'כמה', 'אנחנו', 'הם', 'כבר', 'אפשר', 'תודה', 'שלא', 'אותו', 'מאוד', 'הרבה', 'ולא', 'ממש', 'לו',
    'מי', 'בית', 'שאני', 'יכול', 'שהוא', 'כך', 'הזה', 'איך', 'היום', 'קצת', 'עכשיו', 'שם', 'בכל', 'יהיה', 'תמיד', 'י', 'שלך', 'הכי', 'ש', 'בו', 'לעשות', 'צריך',
    'כן', 'פעם', 'לכם', 'ואני', 'משהו', 'אל', 'שלו', 'שיש', 'ו', 'וגם', 'אתכם', 'אחרי', 'בנושא', 'כדי', 'פשוט', 'לפני', 'שזה', 'אותי', 'אנו', 'למה', 'דבר', 'כאן', 'אולי'
}
EXTENDED_STOPWORDS = BASE_STOPWORDS = STOPWORDS
WORDS_TO_AVOID = ['של']
IGNORE_WORDS = {'של'}
VERBS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP']
ENSURE_ASCII = False

re_space_match = re.compile(r'\s+')

def tokens_to_words(tokens):
    return [token[1] for token in tokens]

def split_by_punct(tokens):
    sents = []
    start = 0
    last_token = len(tokens) - 1
    for i, token in enumerate(tokens):
        if token[0] == Groups.PUNCTUATION or i == last_token:
            new_sent = tokens_to_words(tokens[start: i])
            if len(new_sent) > 0:
                sents.append(new_sent)
            start = i + 1
    return sents

# This function does not pass the test! As it purposely does both sent_tokenize + word_tokenize an
def sent_tokenize(text):
    doc_text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    doc_text = re_space_match.sub(' ', doc_text)
    split_text = split_by_punct(list(tokenize(doc_text)))
    return split_text

def word_tokenize(sent):
    tokens = sent
    return tokens

def stem(text):
    return text  # There are hebrew stemmers, but to date they are... less accurate.

stanza.download('he')
snlp = stanza.Pipeline(lang='he', processors='tokenize,pos')

def pos_tag(tokens):
    res = []
    tokens = ' '.join(tokens)  # this is a very very shallow implementation...
    for tag in snlp(tokens).iter_tokens():
        res.append((tag.words[0].text, tag.words[0].upos)) # this is shallow too, as mwt is not handled at all...
    return res

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
        lnrm = unicodedata.normalize('NFKD', lnrm)
        lnrm = ''.join([x for x in lnrm if (not unicodedata.combining(x)
                                            and x.isalnum() or x == ' ')]).strip()
    # will remove if there are any duplicate white spaces e.g. "the  alias    is here"
    lnrm = " ".join(lnrm.split())
    return lnrm

class HumanNameParser:
    def __init__(self, name):
        parts = name.split(' ')
        self.first = parts[0]
        if len(parts) > 1:
            self.last = ' '.join(parts[1:]).replace('-', ' ')

# Hebrew needs POS tagging to work correctly. The following map is quite shallow and WILL hit a lot of false positive/negatives
pronoun_map = {
    'הוא': 1,
    "עצמו": 1,
    "שלו": 1,
    "בעצמו": 1,
    'היא': 2,
    "עצמה": 2,
    "שלה": 2,
    "שלה": 2,
    'זה': 3,
    "בזה": 3,
    'הם': 4,
    'הן': 4,
    "שלהם": 4,
    "שלהן": 4,
    "בשלהם": 4,
    "בשלהן": 4,
}
pronoun_possessive_map = {
    'הוא': 0,
    "עצמו": 0,
    "שלו": 1,
    "בעצמו": 0,
    'שלה': 1,
    "היא": 0,
    "עצמה": 0,
    "בעצמה": 0,
    'זה': 0,
    "בזה": 0,
    'הם': 0,
    'הן': 0,
    "שלהם": 1,
    "שלהן": 1,
    "עצמם": 0,
    "עצמן": 0
}

# In hebrew there are two plural sexes not one as in english - guess this needs to be addressed after integrating the person.npy model in the language model
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
