def tf_idf_feature(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    union_set = s1 | s2
    v1,v2 =[],[]

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


if __name__ == '__main__':
    res = tf_idf_feature(
        ['mr', 'president', 'we', 'take', 'a', 'great', 'deal', 'of', 'care', 'over', 'prepare', 'the', 'karamanou', 'report', 'in', 'the', 'committee', 'on', 'citizens', 'freedoms', 'and', 'rights', 'justice', 'and', 'home', 'affairs'],
        ['mr', 'president', 'the', 'committee', 'on', 'citizens', 'freedoms', 'and', 'rights', 'justice', 'and', 'internal', 'affair', 'have', 'prepare', 'with', 'great', 'care', 'the', 'karamanou', 'report']
    )
    print(res)
