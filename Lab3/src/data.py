def load_data(path, size=None):
    import pandas as pd
    data = pd.read_table(path)

    return data['sentence1'][:size], data['sentence2'][:size], data[
        'gold_label'][:size]


def load_vocabulary(path):
    vocabulary = {}

    cnt = 0
    file = open(path, 'r')
    for line in file:
        vocabulary[line.strip()] = cnt
        cnt = cnt + 1

    return vocabulary


def load_embedding(path):
    ret = []

    file = open(path, 'r')
    for line in file:
        ret.append([float(s) for s in line.split()])

    return ret


def tokenizer(sentence, vocabulary):
    import re
    ret = []
    max_length = 0

    for s in sentence:
        vector = []
        words = re.split(r'[-\";,.!?\.\s]', str(s))
        for word in words:
            w = word.rstrip('\'').rstrip('\'s').lower()
            if w in vocabulary.keys():
                vector.append(vocabulary[w])

        max_length = max(max_length, len(vector))
        ret.append(vector)

    return ret, max_length


def padding(sentence, max_length):
    ret = []

    for s in sentence:
        tmp = s[:max_length]
        for i in range(max_length - len(tmp)):
            tmp.append(0)
        ret.append(tmp)

    return ret


'''
if __name__ == '__main__':
    train_sentence1, train_sentence2, train_label = load_data(
        '../test/snli_1.0_train.txt', 5)
    test_sentence1, test_sentence2, ttest_label = load_data(
        '../test/snli_1.0_test.txt', 1)

    vocabulary = load_vocabulary('../data/vocabulary.txt')

    tmp, max_length = tokenizer(train_sentence1, vocabulary)

    tmp = padding(tmp, max_length)
'''
