from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser
import os
import re


# java_path = "C:/Program Files/Java/jdk-12.0.2/bin/java.exe"
# os.environ['JAVAHOME'] = java_path


lemmatizer = WordNetLemmatizer()


def stringized_data(data):
    str_list = list()

    for key, item in data.items():
        str_list.append(str(key))

        for content_key, content in item.items():
            str_list.append('\n\t{}: {}'.format(content_key, content))
        str_list.append('\n')

    return ''.join(str_list)


def tokenized_sentence(s):
    res = word_tokenize(s)

    res = [word for word in res if not re.search('^[,.?!\-\"\'()\[\]{}]+$', word)]

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

    for token, pos in pos_l:
        res.append(
            lemmatizer.lemmatize(token, pos=wordnet_tagset.get(pos[0], 'n')).lower()
        )

    return res


def parse_tree():
    STANFORD = os.path.join("D:/semester3/6320/project/CoreNLP", 'stanford-corenlp-full-2018-10-05')

    with CoreNLPServer(
        os.path.join(STANFORD, 'stanford-corenlp-3.9.2.jar'),
        os.path.join(STANFORD, 'stanford-corenlp-3.9.2-models.jar'),
    ):
        parser = CoreNLPParser()

        text = "The runner scored from second on a base hit"
        parse = next(parser.parse_text(text))
        # parse.draw()
    return



if __name__ == '__main__':
    parse_tree()
