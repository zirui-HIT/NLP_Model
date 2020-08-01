from train import train
from test import test

if __name__ == '__main__':
    nn = train('../test/snli_1.0_train.txt', '../data/vocabulary.txt',
               '../data/word2vec.txt', '../model/esim.pkl')
    result = test(nn, '../test/snli_1.0_test.txt', '../data/vocabulary.txt',
                  '../data/word2vec.txt')

    print(result)
