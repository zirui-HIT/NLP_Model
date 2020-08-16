from __future__ import division
import data
import math
import torch
from transformer import Transformer
import torch.utils.data as Data


def train(input_path='../test/train.tsv',
          model_path='../model/transformer.pkl',
          embedding_path='../data/embedding.txt',
          data_size=None,
          batch_size=1000,
          head_num=4,
          hidden_size=64,
          hidden_dim=32,
          hidden_layer=2,
          dropout=0.1,
          epoch=20,
          lr=0.2):
    vocabulary, matrix, embedding_dim = data.load_embedding(embedding_path)
    matrix = data.list2torch(matrix, torch.FloatTensor)

    phrase, sentiment = data.load_data(input_path, data_size, True)

    phrase, max_length = data.word2vec(phrase, vocabulary)
    phrase = data.padding(phrase, max_length)

    phrase = data.list2torch(phrase, torch.LongTensor)
    sentiment = data.list2torch(sentiment, torch.LongTensor)

    position_matrix = []
    for i in range(max_length):
        vec = []
        for j in range(embedding_dim):
            if j % 2 == 0:
                vec.append(math.sin(i / math.pow(10000, j / embedding_dim)))
            else:
                vec.append(math.sin(i / math.pow(10000, j / embedding_dim)))
        position_matrix.append(vec)
    position_matrix = data.list2torch(position_matrix, torch.FloatTensor)

    torch_dataset = Data.TensorDataset(phrase, sentiment)
    loader = Data.DataLoader(dataset=torch_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    transformer = Transformer(vocabulary_size=len(vocabulary) + 1,
                              position_matrix=position_matrix,
                              max_length=max_length,
                              embedding_dim=embedding_dim,
                              embedding_matrix=matrix,
                              head_num=head_num,
                              hidden_size=hidden_size,
                              hidden_dim=hidden_dim,
                              hidden_layer=hidden_layer,
                              dropout=dropout)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    for e in range(epoch):
        for step, (current_phrase, current_sentiment) in enumerate(loader):
            out = transformer(current_phrase)
            loss = loss_func(out, current_sentiment)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(out.data)

    torch.save(transformer, model_path)


train(data_size=None,
      epoch=20,
      lr=0.02,
      head_num=4,
      hidden_size=256,
      hidden_dim=128,
      hidden_layer=4,
      dropout=0.5)
