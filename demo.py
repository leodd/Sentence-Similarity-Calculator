from CorpusReader import CorpusReader
from utils import *


reader = CorpusReader('data/train-set.txt')

data = reader.data()

for _, item in data.items():
    item['Tokenized Sentence1'] = tokenized_sentence(item['Sentence1'])
    item['Tokenized Sentence2'] = tokenized_sentence(item['Sentence2'])

for _, item in data.items():
    item['Pos-tagged Sentence1'] = pos_tagged_sentence(item['Tokenized Sentence1'])
    item['Pos-tagged Sentence2'] = pos_tagged_sentence(item['Tokenized Sentence2'])

for _, item in data.items():
    item['Lemmatized Sentence1'] = lemmatized_sentence(item['Pos-tagged Sentence1'])
    item['Lemmatized Sentence2'] = lemmatized_sentence(item['Pos-tagged Sentence2'])

print(stringized_data(data))
