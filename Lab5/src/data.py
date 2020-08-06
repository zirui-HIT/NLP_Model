def load_data(path, data_size=None):
    file = open(path, 'r', encoding='utf-8')
    data = []
    cnt = 0

    for line in file:
        if cnt == data_size:
            break

        line = line.strip()
        data.append(line.split())
        cnt = cnt + 1
    vocabulary = set([x for line in data for x in line])

    return data, list(vocabulary)


def load_embedding(embedding_path):
    file = open(embedding_path, 'r', encoding='utf-8')

    matrix = []
    match = {}
    cnt = 0
    for line in file:
        if cnt == 0:
            cnt = cnt + 1
            continue

        vec = line.split()

        match[vec[0]] = cnt
        matrix.append([float(x) for x in vec[1:]])
        cnt = cnt + 1
    matrix.insert(0, [0] * 32)

    return match, matrix


def padding(sentence, max_length=None, pad=0):
    ret = []

    for s in sentence:
        tmp = s[:max_length]
        for i in range(max_length - len(tmp)):
            tmp.append(pad)
        ret.append(tmp)

    return ret


def word2vec(sentence, match):
    ret = []

    for s in sentence:
        ret.append([match[w] for w in s])
    max_length = len(max(ret, key=len))

    return ret, max_length


def list2torch(list_name, TYPE):
    import numpy as np
    import torch

    return torch.from_numpy(np.array(list_name)).type(TYPE)


def tensor2vec(data, matrix):
    ret = []

    for line in data:
        ret.append([matrix[x] for x in line])

    import torch
    return list2torch(ret, torch.FloatTensor)
