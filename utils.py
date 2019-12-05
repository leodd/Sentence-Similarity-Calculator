from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
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

    wordnet_tagset = {
        'J': 'a',
        'N': 'n',
        'V': 'v',
        'R': 'r'
    }

    lemmatizer = WordNetLemmatizer()

    for token, pos in pos_l:
        res.append(
            lemmatizer.lemmatize(token, pos=wordnet_tagset.get(pos[0], 'n')).lower()
        )

    return res
