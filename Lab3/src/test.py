from __future__ import division
import data
import torch
import numpy as np


def test(
    esim,
    test_data_path,
    vocabulary_path,
    embedding_matrix_path,
):
    sentence1, sentence2, label = data.load_data(test_data_path)

    vocabulary = data.load_vocabulary(vocabulary_path)

    embedding_matrix = data.load_embedding(embedding_matrix_path)
    embedding_matrix = data.list2torch(embedding_matrix)

    sentence1 = data.tokenizer(sentence1, vocabulary)
    sentence2 = data.tokenizer(sentence2, vocabulary)
    sentence1 = data.padding(sentence1)
    sentence2 = data.padding(sentence2)
    sentence1 = data.list2torch(sentence1)
    sentence2 = data.list2torch(sentence2)

    label = data.label2num(label)
    label = np.numpy(label)

    predicate_label = torch.max(esim(sentence1, sentence2), 1)[1]
    predicate_label = predicate_label.numpy()

    return len(np.argwhere(label != predicate_label)) / len(label)
