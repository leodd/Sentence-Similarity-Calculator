class Node:
    def __init__(self, lemma, pos, children=None):
        self.lemma = lemma
        self.pos = pos
        if children is None:
            self.children = list()
        else:
            self.children = children


def build_tree(d_tree, lemma_pos):
    nodes = list()
    sentence_length = len(lemma_pos)

    for i in range(sentence_length):
        lemma, pos = lemma_pos[i]
        nodes.append(
            Node(lemma, pos)
        )

    root = None

    for i in range(sentence_length):
        parent, rel = d_tree[i]

        if parent == -1:
            root = nodes[i]
        else:
            nodes[parent].children.append(
                (nodes[i], rel)
            )

    return root
