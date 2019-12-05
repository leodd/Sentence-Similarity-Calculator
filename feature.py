from word_similarity import *
import numpy as np
def sentence_union(l1,l2):  #l1 and l2 should be lemma-post list
    s1 = set(l1)
    s2 = set(l2)
    union_set = s1 | s2
    # v1, v2 = [], []
    return  list(union_set) #,list(s1) ,list(s2)

def tf_idf_feature(l1, l2): #l1 and l2 should be lemma-post list
    s1 = set(l1)
    s2 = set(l2)
    # union_set = s1 | s2
    v1,v2 =[],[]
    union_set  = sentence_union(l1,l2)

    # form a set containing keywords of both strings
    for w in union_set:
        if w in s1:
            v1.append(1)
        else:
            v1.append(0)

        if w in s2:
            v2.append(1)
        else:
            v2.append(0)
    c =0

    # cosine formula
    for i in range(len(union_set)):
        c+=v1[i] *v2[i]

    cosine = c/float((sum(v1)*sum(v2))**0.5)

    return cosine

def semantic_similary(l1,l2):
    # input l1 ,l2 is lemma-pos list ,like [('a', 'DT'), ('25', 'CD'), ('percent', 'NN'), ('increase', 'NN')]

    union_set =sentence_union(l1,l2)
    v1,v2 =[],[]


    for token in union_set:
        if token in l1:
            v1.append(1)
        else:
            tep =0
            for char in l1:
                word = char[0]
                tep = max(wup_similarity(token[0],word),tep)
                v1.append(tep)

    for token in union_set:
        if token in l2:
            v2.append(1)
        else:
            tep =0
            for char in l2:
                word =char[0]
                tep = max(wup_similarity(token[0],word),tep)
                v2.append(tep)
    v1 = np.array(v1)
    v2 = np.array(v2)
    similarity = np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return similarity









if __name__ == '__main__':
    result= semantic_similary(
        [('a', 'DT'), ('25', 'CD'), ('percent', 'NN'), ('increase', 'NN'), ('would', 'MD')],
        [('the', 'DT'), ('25', 'CD'), ('percent', 'NN'), ('hike', 'NN'), ('take', 'VBZ')]
    )
    print(result)
