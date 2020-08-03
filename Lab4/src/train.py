import data
from LSTM_CRF import LSTM_CRF
import torch
import torch.utils.data as Data


def train(data_path='../test/train.in',
          label_path='../test/train.out',
          embedding_path='../data/word2vec.txt',
          save_path='../model/lstm_crf.pkl',
          lr=0.2,
          dropout=0.1,
          data_size=None,
          hidden_dim=64,
          batch_size=1024,
          epoch=20):
    match, embedding_matrix = data.load_embedding(embedding_path)
    embedding_matrix = data.list2torch(embedding_matrix, torch.FloatTensor)

    sentence, _ = data.load_data(data_path, data_size)
    sentence, max_length = data.word2vec(sentence, match)
    sentence = data.padding(sentence, max_length)
    sentence = data.list2torch(sentence, torch.LongTensor)

    label, _ = data.load_data(label_path, data_size)
    label, max_length = data.label2num(label)
    label = data.padding(label, max_length, -1)
    label = data.list2torch(label, torch.LongTensor)

    torch_dataset = Data.TensorDataset(sentence, label)
    loader = Data.DataLoader(dataset=torch_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)

    lstm_crf = LSTM_CRF(vocabulary_size=embedding_matrix.size()[0],
                        embedding_dim=embedding_matrix.size()[1],
                        embedding_matrix=embedding_matrix,
                        hidden_size=hidden_dim,
                        dropout=dropout)
    optimizer = torch.optim.Adam(lstm_crf.parameters(), lr=lr)

    for e in range(epoch):
        for step, (current_sentence, current_label) in enumerate(loader):
            loss = lstm_crf(current_sentence, current_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(lstm_crf, save_path)
