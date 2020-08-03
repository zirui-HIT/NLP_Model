def load_data(path, data_size=None):
    file = open(path, 'r')
    data = []
    cnt = 0

    for line in file:
        if cnt == data_size:
            break

        data.append(line.split())
        cnt = cnt + 1
    vocabulary = set([x for line in data for x in line])

    return data, list(vocabulary)


def load_embedding(embedding_path):
    file = open(embedding_path, 'r')

    matrix = []
    match = {}
    cnt = 0
    for line in file:
        if cnt == 0:
            cnt = cnt + 1
            continue

        vec = line.split()

        match[vec[0]] = cnt
        matrix.append(vec[1:])
        cnt = cnt + 1

    return match, matrix


def padding(sentence, max_length):
    ret = []

    for s in sentence:
        tmp = s[:max_length]
        for i in range(max_length - len(tmp)):
            tmp.append(0)
        ret.append(tmp)

    return ret


def word2vec(sentence, match):
    ret = []

    for s in sentence:
        vec = []
        for w in s:
            vec.append(match[w])
        ret.append(vec)
    max_length = len(max(ret, key=len))

    return ret, max_length


def list2torch(list_name, TYPE):
    import numpy as np
    import torch

    return torch.from_numpy(np.array(list_name)).type(TYPE)


def label2num(label):
    ret = []
    match = {
        'I-ORG': 0,
        'I-LOC': 1,
        'B-MISC': 2,
        'I-MISC': 3,
        'B-PER': 4,
        'B-ORG': 5,
        'O': 6,
        'I-PER': 7,
        'B-LOC': 8
    }

    for s in label:
        vec = []
        for w in s:
            vec.append(match[w])
        ret.append(vec)

    return ret
