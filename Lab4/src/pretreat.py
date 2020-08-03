def make_embedding_matrix(input_path, embedding_dim, output_path):
    from gensim.models import word2vec

    sentences = word2vec.Text8Corpus(input_path)
    model = word2vec.Word2Vec(sentences, size=embedding_dim, window=5)
    model.wv.save_word2vec_format(output_path, binary=False)


if __name__ == '__main__':
    make_embedding_matrix('../test/all_text.txt', 50, '../data/word2vec.txt')
