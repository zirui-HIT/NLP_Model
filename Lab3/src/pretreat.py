from data import load_data

DIM = 50
DATA_SIZE = 50000


def make_vocabulary(sentence, vec_path):
    import re

    vocabulary = []
    for s in sentence:
        words = str(s)
        words = re.split(r'[-\";,.!?\.\s]', words)
        vocabulary = vocabulary + words
    for i in range(len(vocabulary)):
        t = vocabulary[i]
        t = t.rstrip('\'').rstrip('\'s').strip().lower()
        vocabulary[i] = t
    vocabulary = set(vocabulary)

    final_vocabulary = [0]

    vec_file = open(vec_path, 'r', encoding='utf-8')
    for line in vec_file:
        word = line.split()
        if word[0] in vocabulary:
            final_vocabulary.append(word[0])
    vec_file.close()

    file = open('../data/vocabulary.txt', 'w')
    for word in final_vocabulary:
        file.write(str(word) + '\n')
    file.close()


def make_vector(vocabulary, vec_file, out_file):
    match = {}

    num = 0
    keys = vocabulary.keys()
    length = len(vocabulary) + 1

    file = open(vec_file, 'r', encoding='utf-8')
    for line in file:
        word = line.split()

        if word[0] in keys:
            vec = word[1:].copy()
            match[vocabulary[word[0]]] = vec
            num = num + 1

        if num == length - 1:
            break
    file.close()

    match[0] = [0.0] * DIM

    file = open(out_file, 'w')
    for i in range(length):
        for j in range(DIM):
            file.write(str(match[i][j]) + ' ')
        file.write('\n')
    file.close()


if __name__ == '__main__':
    train_sentence1, train_sentence2, train_label = load_data(
        '../test/snli_1.0_train.txt', DATA_SIZE)
    test_sentence1, test_sentence2, ttest_label = load_data(
        '../test/snli_1.0_test.txt', DATA_SIZE)

    make_vocabulary(
        train_sentence1 + train_sentence2 + test_sentence1 + test_sentence2,
        '../data/glove.6B.50d.txt')

    file = open('../data/vocabulary.txt', 'r')
    vocabulary = {}
    cnt = 0
    for line in file:
        if cnt == 0:
            cnt = cnt + 1
            continue
        vocabulary[line.strip()] = cnt
        cnt = cnt + 1
    file.close()

    make_vector(vocabulary, '../data/glove.6B.50d.txt',
                '../data/word2vec.txt')
