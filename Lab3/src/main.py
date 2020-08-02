from train import train
from test import test

if __name__ == '__main__':
    nn = train('../test/snli_1.0_train.txt',
               '../data/vocabulary.txt',
               '../data/word2vec.txt',
               '../model/esim.pkl',
               lr=0.02,
               hidden_dim=300,
               batch_size=2048)
    result = test(nn,
                  '../test/snli_1.0_test.txt',
                  '../data/vocabulary.txt',
                  '../data/word2vec.txt')

    print(result)
