from word_similarity import *
import numpy as np
from nltk.corpus import stopwords
def sentence_union(l1,l2):  #l1 and l2 should be lemma-post list
    s1 = set(l1)
    s2 = set(l2)
    union_set = s1 | s2
    # v1, v2 = [], []
    return  list(union_set) #,list(s1) ,list(s2)

def unionWithStopWord(l1,l2):
    # l1 and l2 should be lemma-post list
    # return the two removed stopword set without tag and their union set
    s1 = set(l1)
    s2 = set(l2)
    # sw contains the list of stopwords
    sw = stopwords.words('english')

    # remove stop words from string
    X_set = {w[0] for w in s1 if w[0] not in sw}
    Y_set = {w[0] for w in s2 if w[0] not in sw}

    # form a set containing keywords of both strings
    union_set = X_set.union(Y_set)
    return union_set,X_set,Y_set


def tf_idf_feature(l1, l2): #l1 and l2 should be lemma-post list
    # [('a', 'DT'), ('25', 'CD'), ('percent', 'NN')

    v1,v2 =[],[]
    union_set ,s1,s2 = unionWithStopWord(l1,l2)

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
    u,s1,s2= unionWithStopWord(
[('a', 'DT'), ('25', 'CD'), ('percent', 'NN'), ('increase', 'NN'), ('would', 'MD'), ('raise', 'VB'), ('undergraduate', 'JJ'), ('tuition', 'NN'), ('to', 'TO'), ('about', 'IN'), ('5,247', 'CD'), ('annually', 'RB'), ('include', 'VBG'), ('miscellaneous', 'JJ'), ('campus-based', 'JJ'), ('fee', 'NNS')]
,
        [('annual', 'JJ'), ('uc', 'NN'), ('undergraduate', 'JJ'), ('tuition', 'NN'), ('to', 'TO'), ('4,794', 'CD'),
         ('and', 'CC'), ('graduate', 'JJ'), ('fee', 'NNS'), ('to', 'TO'), ('5,019', 'CD')]
    )
print(u)
print(s1)
print(s2)