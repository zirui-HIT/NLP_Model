from __future__ import division
import data
import torch
import numpy as np


def test(
    esim,
    test_data_path,
    vocabulary_path,
    embedding_matrix_path,
    data_size=None
):
    sentence1, sentence2, label = data.load_data(test_data_path, data_size)

    vocabulary = data.load_vocabulary(vocabulary_path)

    sentence1, max_length1 = data.tokenizer(sentence1, vocabulary)
    sentence2, max_length2 = data.tokenizer(sentence2, vocabulary)
    max_length = max(max_length1, max_length2)

    sentence1 = data.padding(sentence1, max_length)
    sentence2 = data.padding(sentence2, max_length)
    sentence1 = data.list2torch(sentence1)
    sentence2 = data.list2torch(sentence2)

    label = data.label2num(label)
    label = np.array(label)

    _, out = esim(sentence1, sentence2)
    predicate_label = torch.max(out, 1)[1]
    predicate_label = predicate_label.numpy()

    return len(np.argwhere(label != predicate_label)) / len(label)
