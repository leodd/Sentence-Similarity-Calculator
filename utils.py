import nltk


def tokenized_sentence(s):
    return nltk.word_tokenize(s)


def pos_tagged_sentence(l):
    return nltk.pos_tag(l)
