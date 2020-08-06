def tokenizer(data_path, save_path):
    import jieba

    read_file = open(data_path, 'r', encoding='utf-8')
    write_file = open(save_path, 'w', encoding='utf-8')

    for line in read_file:
        line = line.strip()
        if len(line) == 0:
            continue

        words = jieba.lcut(line)
        write_file.write('<BOS> ')
        for w in words:
            write_file.write(w + ' ')
        write_file.write('<EOS>')
        write_file.write('\n')

    read_file.close()
    write_file.close()


def word2vec(data_path, save_path, embedding_dim=32):
    from gensim.models import word2vec
    from gensim.models.word2vec import LineSentence

    model = word2vec.Word2Vec(LineSentence(data_path),
                              size=embedding_dim,
                              window=5,
                              min_count=1)
    model.wv.save_word2vec_format(save_path, binary=False)


if __name__ == '__main__':
    tokenizer('../test/poetryFromTang.txt', '../data/train.txt')
    word2vec(data_path='../data/train.txt', save_path='../data/embedding.txt')
