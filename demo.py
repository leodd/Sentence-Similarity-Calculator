from CorpusReader import CorpusReader
from utils import *
from stanford import CoreNLP
from dependency_tree import build_tree
from dependency_similarity import *
from feature import *
from NeuralLearner import *
import torch
import torch.nn as nn
import numpy as np
import sys


# data pre-processing
if False:
    reader = CorpusReader('data/test-set.txt')

    train_data = reader.data()

    for _, item in train_data.items():
        item['token1'] = tokenized_sentence(item['Sentence1'])
        item['token2'] = tokenized_sentence(item['Sentence2'])

    corenlp = CoreNLP(sys.argv)
    corenlp.start_server()
    for k, item in train_data.items():
        print(k)
        item['d-tree1'] = corenlp.dependency_parse_tree(list_to_string(item['token1']))
        item['d-tree2'] = corenlp.dependency_parse_tree(list_to_string(item['token2']))
    corenlp.stop_server()

    for _, item in train_data.items():
        item['lemma-pos1'] = pos_tagged_lemmatized_sentence(item['d-tree1'])
        item['lemma-pos2'] = pos_tagged_lemmatized_sentence(item['d-tree2'])

    for _, item in train_data.items():
        item['d-tree1'] = simplified_dependency_tree(item['d-tree1'])
        item['d-tree2'] = simplified_dependency_tree(item['d-tree2'])

    save_data('processed-data/test-set.json', train_data)

# features computation
if False:
    train_data = load_data('processed-data/dev-set.json')

    for k, item in train_data.items():
        print(k)

        # item['Gold Tag'] = int(item['Gold Tag'])
        #
        # root1 = build_tree(item['d-tree1'], item['lemma-pos1'])
        # root2 = build_tree(item['d-tree2'], item['lemma-pos2'])
        #
        # item['d-sim'] = dependency_tree_similarity(root1, root2)
        #
        # selected_wordset1 = get_wordset_by_pos(
        #     item['lemma-pos1'],
        #     lambda pos: pos[0] == 'N' or pos == 'CD'
        # )
        #
        # selected_wordset2 = get_wordset_by_pos(
        #     item['lemma-pos2'],
        #     lambda pos: pos[0] == 'N' or pos == 'CD'
        # )
        #
        # item['nn-cd-sim'] = jaccard_wordset_similarity(selected_wordset1, selected_wordset2)
        # item['nn-cd-num-diff'] = wordset_number_difference(selected_wordset1, selected_wordset2)

        item['tf-idf'] = tfidf_similarity(item['Sentence1'], item['Sentence2'])
        item['cosine-sim'] = cosine_similarity(
            list_to_hashable(item['lemma-pos1']),
            list_to_hashable(item['lemma-pos2'])
        )

        item['jaccard-sim'] = jaccard_similarity(item['Sentence1'], item['Sentence2'])

    save_data('processed-data/dev-set.json', train_data)

# learning
if True:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_data = load_data('processed-data/train-set.json')
    dev_data = load_data('processed-data/dev-set.json')

    id_dict, X, Y, weight = data_to_XY(train_data, device)
    dev_id_dict, dev_X, dev_Y, dev_weight = data_to_XY(dev_data, device)

    model = NeuralLearner([6, 30, 30, 5]).to(device)

    criterion = nn.CrossEntropyLoss(weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=1, weight_decay=0.01)

    for epoch in range(10000):
        # forward pass
        out = model(X)
        loss = criterion(out, Y)
        dev_out = model(dev_X)
        dev_loss = criterion(dev_out, dev_Y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item(), dev_loss.item())

    for k, i in dev_id_dict.items():
        print(k, dev_Y[i].item(), torch.max(dev_out[i], 0)[1])

    torch.save(model.state_dict(), 'result/model.ckpt')

# testing
if False:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_data = load_data('processed-data/dev-set.json')

    id_dict, X, Y, weight = data_to_XY(train_data, device)

    model = NeuralLearner([6, 30, 30, 5]).to(device)
    model.load_state_dict(
        torch.load('result/model.ckpt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    )
    model.eval()

    criterion = nn.CrossEntropyLoss(weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=1, weight_decay=0.0001)

    # forward pass
    out = model(X)
    loss = criterion(out, Y)

    print(loss.item())

    for k, i in id_dict.items():
        print(k, Y[i].item(), torch.max(out[i], 0)[1])

    torch.save(model.state_dict(), 'result/model.ckpt')

# print(stringized_data(data))
