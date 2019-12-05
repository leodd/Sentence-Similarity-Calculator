from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import re


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
