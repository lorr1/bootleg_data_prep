"""

This is a SHALLOW test to aid developing new language module,
  just to be able to assert types and expected data as the rest of the code will.

"""

import hebrew as language

assert isinstance(language.PUNC, str)
assert isinstance(language.PUNC_TRANSLATION_TABLE, dict)
assert isinstance(language.BASE_STOPWORDS, set)
assert isinstance(language.WORDS_TO_AVOID, list)
assert isinstance(language.IGNORE_WORDS, set)
assert isinstance(language.NOUNS, list)
sent_tokenized = language.sent_tokenize('משפט תקין, אומר הכל. אוכל עוגה - ויושן בעריסה')
assert isinstance(sent_tokenized, list)
assert isinstance(sent_tokenized[0], str)
word_tokenized = language.word_tokenize('one world')
assert isinstance(word_tokenized, list)
assert isinstance(word_tokenized[0], str)
pos_tagged = language.pos_tag(['בא', 'לי', 'פיצה'])
assert isinstance(pos_tagged, list)
assert isinstance(pos_tagged[0], tuple)
ngrams = language.ngrams(['שדגכ', 'דגכע', 'גגכע', 'כעיח', 'עיחל'], 3)
ngrams_list = list(ngrams)
assert isinstance(ngrams_list, list)
assert isinstance(ngrams_list[0], tuple)
normalized_text = language.get_lnrm('משרד החוץ\t פרסם', True, True)
assert normalized_text == 'משרד החוץ פרסם'
hn = language.HumanNameParser('הומר סימפסון')
assert hn.first == 'הומר'
assert hn.last == 'סימפסון'
assert isinstance(language.pronoun_map, dict)
assert isinstance(language.pronoun_possessive_map, dict)
assert isinstance(language.gender_qid_map, dict)
