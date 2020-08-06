import data
import torch
import torch.utils.data as Data
from module import RNN


def train(data_path='../data/train.txt',
          embedding_path='../data/embedding.txt',
          save_path='../model/RNN.pkl',
          data_size=None,
          lr=0.2,
          dropout=0.1,
          hidden_size=1024,
          batch_size=1024,
          epoch=20):
    match, matrix = data.load_embedding(embedding_path)
    embedding_matrix = data.list2torch(matrix, torch.FloatTensor)

    train_data, _ = data.load_data(data_path, data_size)
    train_data, max_length = data.word2vec(train_data, match)
    train_data = data.padding(train_data, len(max(train_data, key=len)))
    train_data = data.list2torch(train_data, torch.LongTensor)

    torch_dataset = Data.TensorDataset(train_data, train_data)
    loader = Data.DataLoader(dataset=torch_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    rnn = RNN(embedding_matrix.size()[0],
              embedding_matrix.size()[1], embedding_matrix, hidden_size,
              dropout)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    for epoch in range(epoch):
        for step, (current_data) in enumerate(loader):
            current_data = current_data[0]
            current_result = data.tensor2vec(current_data, matrix)

            out = rnn(current_data)
            loss = loss_func(out, current_result)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(rnn, save_path)
