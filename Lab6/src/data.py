def load_data(data_path, data_size=None, load_sentiment=False):
    import pandas as pd
    data = pd.read_csv(data_path, sep='\t')

    phrase = list(data['Phrase'][:data_size])
    phrase = [s.lower() for s in phrase]

    if load_sentiment is True:
        sentiment = data['Sentiment'][:data_size]
        sentiment = [int(s) for s in sentiment]
        return phrase, sentiment
    return phrase


def load_embedding(embedding_path):
    cnt = 0
    data = open(embedding_path, 'r')

    matrix = []
    vocabulary = {}
    cnt = 0
    for line in data:
        vec = line.split()

        if cnt == 0:
            embedding_dim = int(vec[1])
            cnt = cnt + 1
            continue

        vocabulary[vec[0]] = cnt
        matrix.append([float(x) for x in vec[1:]])
        cnt = cnt + 1
    matrix.insert(0, [0] * embedding_dim)

    return vocabulary, matrix, embedding_dim


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
        words = s.split()
        ret.append([match[w] for w in words])
    max_length = len(max(ret, key=len))

    return ret, max_length


def list2torch(list_name, TYPE):
    import numpy as np
    import torch

    return torch.from_numpy(np.array(list_name)).type(TYPE)


def rand_matrix(length, width):
    import random

    matrix = []
    for i in range(length):
        vec = []
        for j in range(width):
            vec.append(random.random())
        matrix.append(vec)
    return matrix
