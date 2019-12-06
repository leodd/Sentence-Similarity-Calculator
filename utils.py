from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from word_similarity import *
import torch
import numpy as np
import re
import json


def save_data(f, data):
    with open(f, 'w+') as file:
        file.write(json.dumps(data))


def load_data(f):
    with open(f, 'r') as file:
        s = file.read()
        return json.loads(s)


def stringized_data(data):
    str_list = list()

    for key, item in data.items():
        str_list.append(str(key))

        for content_key, content in item.items():
            str_list.append('\n\t{}: {}'.format(content_key, content))
        str_list.append('\n')

    return ''.join(str_list)


def list_to_string(l):
    return ' '.join(l)


def tokenized_sentence(s):
    res = word_tokenize(s)

    res = [word for word in res if not re.search('^[,.?!:;$%\-\"`\'/()\[\]{}]+$', word)]

    return res


def simplified_dependency_tree(dependency_tree):
    res = list()

    for i in range(1, len(dependency_tree.nodes)):
        term = dependency_tree.nodes[i]
        res.append((
            term['head'] - 1,
            term['rel']
        ))

    return res


def pos_tagged_lemmatized_sentence(dependency_tree):
    res = list()

    for i in range(1, len(dependency_tree.nodes)):
        term = dependency_tree.nodes[i]
        res.append((
            term['lemma'],
            term['tag']
        ))

    return res


def pos_tagged_sentence(l):
    return pos_tag(l)


def lemmatized_sentence(pos_l):
    res = list()

    lemmatizer = WordNetLemmatizer()

    for token, pos in pos_l:
        wordnet_pos = pos_to_wordnet_pos(pos)
        res.append(
            lemmatizer.lemmatize(token, pos='n' if wordnet_pos is None else wordnet_pos).lower()
        )

    return res


def pos_to_wordnet_pos(pos):
    wordnet_tagset = {
        'J': 'a',
        'N': 'n',
        'V': 'v',
        'R': 'r'
    }

    return wordnet_tagset.get(pos[0], None)


def wordnet_hypernyms(word, pos=None):
    res = list()

    ss = wn.synsets(word, pos=pos)

    for s in ss:
        res += s.hypernyms()

    return res


def wordnet_hyponyms(word, pos=None):
    res = list()

    ss = wn.synsets(word, pos=pos)

    for s in ss:
        res += s.hyponyms()

    return res


def wordnet_meronyms(word, pos=None):
    res = list()

    ss = wn.synsets(word, pos=pos)

    for s in ss:
        res += s.part_meronyms()

    return res


def wordnet_holonyms(word, pos=None):
    res = list()

    ss = wn.synsets(word, pos=pos)

    for s in ss:
        res += s.part_holonyms()

    return res


def get_wordset_by_pos(lemma_pos, pos_constraint):
    res = set()

    for lemma, pos in lemma_pos:
        if pos_constraint(pos):
            res.add((lemma, pos))

    return res


def list_to_hashable(l):
    return [tuple(item) for item in l]


def separate_data_by_class(data):
    res = [dict() for _ in range(5)]

    for k, item in data.items():
        res[item['Gold Tag'] - 1][k] = item

    return res


def data_to_XY(data, device=None, no_gold_tag=False):
    id_dict = dict()

    m = len(data)

    X = np.zeros((m, 6))
    Y = np.zeros(m, dtype=int)

    for i, (k, item) in enumerate(data.items()):
        X[i, 0] = item['d-sim']
        X[i, 1] = item['nn-cd-sim']
        X[i, 2] = item['nn-cd-num-diff']
        X[i, 3] = item['tf-idf']
        X[i, 4] = item['cosine-sim']
        X[i, 5] = item['jaccard-sim']
        # X[i, 6] =
        if not no_gold_tag:
            Y[i] = item['Gold Tag'] - 1
        id_dict[k] = i

    if device is None:
        return id_dict, X, Y

    X = torch.from_numpy(X).float().to(device)
    Y = torch.from_numpy(Y).long().to(device)

    return id_dict, X, Y
