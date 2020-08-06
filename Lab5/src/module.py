import torch


class RNN(torch.nn.Module):
    def __init__(self,
                 vocabulary_size,
                 embedding_dim,
                 embedding_matrix,
                 hidden_size=128,
                 dropout=0.1):
        super(RNN, self).__init__()

        self.embedding = torch.nn.Embedding(vocabulary_size,
                                            embedding_dim,
                                            padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)

        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_size,
                                 batch_first=True)

        self.dropout = torch.nn.Dropout(dropout)

        self.output = torch.nn.Linear(hidden_size, embedding_dim)

    def forward(self, x):
        x_mask = (x != 0).long()

        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = x * x_mask.unsqueeze(-1).expand(x.shape)
        x = self.output(x)

        return x
