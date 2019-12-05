from CorpusReader import CorpusReader
from utils import *
from stanford import CoreNLP
from feature import *
import sys


# data pre-processing
if False:
    reader = CorpusReader('data/test-set.txt')

    data = reader.data()

    for _, item in data.items():
        item['token1'] = tokenized_sentence(item['Sentence1'])
        item['token2'] = tokenized_sentence(item['Sentence2'])

    corenlp = CoreNLP(sys.argv)
    corenlp.start_server()
    for k, item in data.items():
        print(k)
        item['d-tree1'] = corenlp.dependency_parse_tree(list_to_string(item['token1']))
        item['d-tree2'] = corenlp.dependency_parse_tree(list_to_string(item['token2']))
    corenlp.stop_server()

    for _, item in data.items():
        item['lemma-pos1'] = pos_tagged_lemmatized_sentence(item['d-tree1'])
        item['lemma-pos2'] = pos_tagged_lemmatized_sentence(item['d-tree2'])

    for _, item in data.items():
        item['d-tree1'] = simplified_dependency_tree(item['d-tree1'])
        item['d-tree2'] = simplified_dependency_tree(item['d-tree2'])

    save_data('processed-data/test-set.json', data)

data = load_data('processed-data/train-set.json')

print(stringized_data(data))


