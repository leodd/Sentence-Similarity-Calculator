from utils import *
from nltk.corpus import wordnet


synset = wordnet.synsets("Travel")
print('Word and Type : ' + synset[0].name())
print('Synonym of Travel is: ' + synset[0].lemmas()[0].name())
print('The meaning of the word : ' + synset[0].definition())
print('Example of Travel : ' + str(synset[0].examples()))

# print(load_data('data/test-set.txt', encoding='utf8'))
