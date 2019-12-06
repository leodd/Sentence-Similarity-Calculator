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
import xgboost as xgb


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
    train_data = load_data('processed-data/test-set.json')

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

        selected_wordset1 = get_wordset_by_pos(
            item['lemma-pos1'],
            lambda pos: pos[0] == 'V'
        )

        selected_wordset2 = get_wordset_by_pos(
            item['lemma-pos2'],
            lambda pos: pos[0] == 'V'
        )

        item['verb-sim'] = jaccard_wordset_similarity(selected_wordset1, selected_wordset2)

        # item['tf-idf'] = tfidf_similarity(item['Sentence1'], item['Sentence2'])
        # item['cosine-sim'] = cosine_similarity(
        #     list_to_hashable(item['lemma-pos1']),
        #     list_to_hashable(item['lemma-pos2'])
        # )
        #
        # item['jaccard-sim'] = jaccard_similarity(item['Sentence1'], item['Sentence2'])

        # item['wordset-sim'] = mutual_wordset_similarity(
        #     list_to_hashable(item['lemma-pos1']),
        #     list_to_hashable(item['lemma-pos2'])
        # )

        # item['rel-sim'] = dependency_similarity(item['d-tree1'], item['d-tree2'])

    save_data('processed-data/test-set.json', train_data)

# learning
if False:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_data = load_data('processed-data/train-set.json')
    dev_data = load_data('processed-data/dev-set.json')

    id_dict, X, Y = data_to_XY(train_data, device)
    dev_id_dict, dev_X, dev_Y = data_to_XY(dev_data, device)

    separated_data = separate_data_by_class(train_data)
    separated_XY = list()
    for c_data in separated_data:
        _, c_X, c_Y = data_to_XY(c_data, device)
        separated_XY.append((c_X, c_Y))

    model = NeuralLearner([8, 30, 30, 5]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0)

    for epoch in range(10000):
        idx = [torch.randint(len(separated_data[i]), size=(100,)) for i in range(5)]
        mini_X = torch.cat(
            [separated_XY[i][0][idx[i]] for i in range(5)],
            dim=0
        )
        mini_Y = torch.cat(
            [separated_XY[i][1][idx[i]] for i in range(5)],
            dim=0
        )

        # forward pass
        out = model(mini_X)
        loss = criterion(out, mini_Y)
        dev_out = model(dev_X)
        dev_loss = criterion(dev_out, dev_Y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item(), dev_loss.item())

    for k, i in dev_id_dict.items():
        print(k, dev_Y[i].item(), int(torch.max(dev_out[i], 0)[1]))

    torch.save(model.state_dict(), 'result/model.ckpt')

# testing
if False:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_data = load_data('processed-data/dev-set.json')

    id_dict, X, Y = data_to_XY(test_data, device, no_gold_tag=True)

    model = NeuralLearner([8, 30, 30, 5]).to(device)
    model.load_state_dict(
        torch.load('result/model.ckpt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    )
    model.eval()

    # forward pass
    out = model(X)

    res = 'id\tGold Tag'

    for k, i in id_dict.items():
        res += '\n{}\t{}'.format(k, torch.max(out[i], 0)[1])

    with open('result/dev_prediction.txt', 'w+') as file:
        file.write(res)

# Gradient Boosting pipeline
if True:
    train_data = load_data('processed-data/train-set.json')
    dev_data = load_data('processed-data/dev-set.json')

    id_dict, X, Y = data_to_XY(train_data)
    dev_id_dict, dev_X, dev_Y = data_to_XY(dev_data)

    dtrain = xgb.DMatrix(X, label=Y)
    dtest = xgb.DMatrix(dev_X, label=dev_Y)

    param = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'multi:softprob',
        'num_class': 5
    }

    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    num_round = 20
    bst = xgb.train(param, dtrain, num_round, evallist)

    out = bst.predict(dtest)

    accuracy = 0
    for k, i in dev_id_dict.items():
        if dev_Y[i] == np.argmax(out[i]):
            accuracy += 1
        print(k, dev_Y[i], np.argmax(out[i]))

    print(accuracy / len(dev_id_dict))

    res = 'id\tGold Tag'

    for k, i in dev_id_dict.items():
        res += '\n{}\t{}'.format(k, np.argmax(out[i]))

    with open('result/test_prediction.txt', 'w+') as file:
        file.write(res)

    # predict test data
    test_data = load_data('processed-data/dev-set.json')

    id_dict, X, Y = data_to_XY(test_data, no_gold_tag=True)
    dtest = xgb.DMatrix(X)

    out = bst.predict(dtest)

    res = 'id\tGold Tag'

    for k, i in id_dict.items():
        res += '\n{}\t{}'.format(k, np.argmax(out[i]))

    with open('result/test_prediction.txt', 'w+') as file:
        file.write(res)

# print(stringized_data(data))
