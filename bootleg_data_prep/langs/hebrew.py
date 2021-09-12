from hebrew_tokenizer import tokenize
from hebrew_tokenizer.groups import Groups

nltk.download('punkt')
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


def sent_tokenize(text):
    doc_text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    doc_text = re_space_match.sub(' ', doc_text)
    split_text = split_by_punct(list(tokenize(doc_text)))
    # split_text = sent_tokenize(text)
    # print(split_text)
    return split_text


def word_tokenize(sent):
    tokens = sent
    # tokens = word_tokenize(sent)
    # print(tokens)
    return tokens
