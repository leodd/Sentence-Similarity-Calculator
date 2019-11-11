from utils import *


data = load_data('data/test-set.txt', encoding='utf8')

for key, item in data.items():
    print(tokenized_words(item['Sentence1']))
