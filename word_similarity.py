from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from itertools import product


semcor_ic = wordnet_ic.ic('ic-semcor.dat')


def lin_similarity(w1, w2, pos=None):
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
