"""

This is a SHALLOW test to aid developing new language module,
  just to be able to assert types and expected data as the rest of the code will.

"""

import english as language

assert isinstance(language.PUNC, str)
assert isinstance(language.PUNC_TRANSLATION_TABLE, dict)
assert isinstance(language.BASE_STOPWORDS, set)
assert isinstance(language.WORDS_TO_AVOID, list)
assert isinstance(language.IGNORE_WORDS, set)
assert isinstance(language.NOUNS, list)
sent_tokenized = language.sent_tokenize('one sentence. two sentence')
assert isinstance(sent_tokenized, list)
assert isinstance(sent_tokenized[0], str)
word_tokenized = language.word_tokenize('one world')
assert isinstance(word_tokenized, list)
assert isinstance(word_tokenized[0], str)
pos_tagged = language.pos_tag(['highway', 'to', 'hell'])
assert isinstance(pos_tagged, list)
assert isinstance(pos_tagged[0], tuple)
ngrams = language.ngrams(['qwer', 'erty', 'rtyu', 'asdf', 'sdfg'], 3)
ngrams_list = list(ngrams)
assert isinstance(ngrams_list, list)
assert isinstance(ngrams_list[0], tuple)
normalized_text = language.get_lnrm('Big Small\t business', True, True)
assert normalized_text == 'big small business'
hn = language.HumanNameParser('Homer Simpson')
assert hn.first == 'Homer'
assert hn.last == 'Simpson'
assert isinstance(language.pronoun_map, dict)
assert isinstance(language.pronoun_possessive_map, dict)
assert isinstance(language.gender_qid_map, dict)
