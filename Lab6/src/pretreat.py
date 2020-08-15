def output_phrase(data_path, phrase):
    file = open(data_path, 'w')

    for p in phrase:
        file.write(p + '\n')


def get_embedding(data_path, save_path, dim=64):
    from gensim.models import word2vec
    from gensim.models.word2vec import LineSentence

    model = word2vec.Word2Vec(LineSentence(data_path),
                              size=dim,
                              window=5,
                              min_count=1)
    model.wv.save_word2vec_format(save_path, binary=False)


if __name__ == '__main__':
    from data import load_data
    phrase = load_data('../test/train.tsv') + load_data('../test/test.tsv')

    output_phrase('../data/phrase.txt', phrase)

    get_embedding('../data/phrase.txt', '../data/embedding.txt')
