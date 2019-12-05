from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import numpy as np
from itertools import product
from utils import *


semcor_ic = wordnet_ic.ic('ic-semcor.dat')


def lin_similarity(w1, w2, pos):
    res = 0

    ss1 = wn.synsets(w1, pos=pos)
    ss2 = wn.synsets(w2, pos=pos)

    for s1, s2 in product(ss1, ss2):
        d = s1.lin_similarity(s2, semcor_ic)
        res = max(res, 0 if d is None else d)

    return res


def wup_similarity(w1, w2, pos=None):
    res = 0

    ss1 = wn.synsets(w1, pos=pos)
    ss2 = wn.synsets(w2, pos=pos)

    for s1, s2 in product(ss1, ss2):
        d = s1.wup_similarity(s2)
        res = max(res, 0 if d is None else d)

    return res


def wordset_similarity(ws1, ws2):
    # word set is a set with word-pos tuple
    # e.g. {('apple', 'NN'), ('car', 'NN')}
    ws_common = ws1 | ws2
    ws_vector = list(ws_common)

    v1 = list()
    v2 = list()

    for word in ws_vector:
        _, max_similarity = most_similar_word(word, ws1)
        v1.append(max_similarity)

        _, max_similarity = most_similar_word(word, ws2)
        v2.append(max_similarity)

    v1 = np.array(v1)
    v2 = np.array(v2)

    print(v1)
    print(v2)

    return np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def most_similar_word(target, ws):
    max_matching = None
    max_similarity = 0

    for w in ws:
        similarity = wup_similarity(target[0], w[0])
        if similarity >= max_similarity:
            max_matching = w
            max_similarity = similarity

    return max_matching, max_similarity


if __name__ == '__main__':
    print(wordset_similarity({('a', 'DT'), ('25', 'CD'), ('percent', 'NN'), ('increase', 'NN'), ('would', 'MD')},
                             {('the', 'DT'), ('25', 'CD'), ('percent', 'NN'), ('hike', 'NN'), ('take', 'VBZ')}))
