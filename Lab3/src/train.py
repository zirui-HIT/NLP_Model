import torch
import data
from module import ESIM

if __name__ == '__main__':
    train_sentence1, train_sentence2, train_label = data.load_data(
        '../test/snli_1.0_train.txt', 5)
    test_sentence1, test_sentence2, ttest_label = data.load_data(
        '../test/snli_1.0_test.txt', 1)

    vocabulary = data.load_vocabulary('../data/vocabulary.txt')

    embedding_matrix = data.load_embedding('../data/word2vec.txt')
    embedding_matrix = data.list2torch(embedding_matrix)

    train_sentence1 = data.tokenizer(train_sentence1, vocabulary)
    train_sentence2 = data.tokenizer(train_sentence2, vocabulary)
    train_sentence1 = data.padding(train_sentence1)
    train_sentence2 = data.padding(train_sentence2)

    train_sentence1 = data.list2torch(train_sentence1)
    train_sentence2 = data.list2torch(train_sentence2)
