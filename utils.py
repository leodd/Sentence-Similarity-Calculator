from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.parse.corenlp import CoreNLPServer


def stringized_data(data):
    str_list = list()
    for key, item in data.items():
        str_list.append(str(key))
        for content_key, content in item.items():
            str_list.append('\n\t{}: {}'.format(content_key, content))
        str_list.append('\n')
    return ''.join(str_list)


def tokenized_sentence(s):
    return word_tokenize(s)


def pos_tagged_sentence(l):
    return
