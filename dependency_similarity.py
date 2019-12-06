from word_similarity import *
import numpy as np


def dependency_tree_similarity(root1, root2):
    dependencies1 = get_dependency_list(root1)
    dependencies2 = get_dependency_list(root2)

    n1 = len(dependencies1)
    n2 = len(dependencies2)

    if min(n1, n2) == 0:
        return 0

    similarity_matrix = np.zeros((n1, n2))
    depth_weight_matrix = np.zeros((n1, n2))

    for i, d1 in enumerate(dependencies1):
        for j, d2 in enumerate(dependencies2):
            similarity_matrix[i, j] = \
                wup_similarity(d1[0], d2[0]) * wup_similarity(d1[1], d2[1]) * (1 if d1[2] == d2[2] else 0.3) * \
                np.e ** (- abs(d1[3] - d2[3]))
            depth_weight_matrix[i, j] = np.e ** (- max(d1[3], d2[3]) * 0.5)

    similarity_matrix *= depth_weight_matrix

    idx1 = np.argmax(similarity_matrix, axis=0)
    idx2 = np.argmax(similarity_matrix, axis=1)

    res = (np.sum(similarity_matrix[idx1, np.arange(n2)]) + np.sum(similarity_matrix[np.arange(n1), idx2])) / \
          (np.sum(depth_weight_matrix[idx1, np.arange(n2)]) + np.sum(depth_weight_matrix[np.arange(n1), idx2]))

    return res


def get_dependency_list(root):
    res = list()
    stack = [root]

    root.depth = 0

    allow_pos = {'N', 'V', 'J', 'R', 'C'}

    while len(stack) > 0:
        current = stack.pop()

        for child, rel in current.children:
            if child.pos[0] in allow_pos:
                res.append(
                    (current.lemma, child.lemma, rel, current.depth)
                )
                child.depth = current.depth + 1
                stack.append(child)

    return res
