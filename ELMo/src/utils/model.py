from typing import List
import torch
from typing import List


class ELMo(torch.nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, lstm_layer_num: int, hidden_size: int, padding_idx: int = None):
        """initialize ELMo

        Args:
            vocabulary_size: size of vocabulary
            embedding_dim: dim of embedding layer
            lstm_layer_num: num of lstm layer which is greater that 1
            hidden_size: size of every lstm layer
            padding_idx: idx of padding sign
        """
        super(ELMo, self).__init__()

        self._embedding = torch.nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embedding_dim, padding_idx=padding_idx)

        self._lstm = []
        if lstm_layer_num > 0:
            self._lstm.append(torch.nn.LSTM(input_size=embedding_dim,
                              hidden_size=hidden_size, bidirectional=True, batch_first=True))
        for i in range(1, lstm_layer_num):
            self._lstm.append(torch.nn.LSTM(input_size=hidden_size,
                              hidden_size=hidden_size, bidirectional=True, batch_first=True))

        # for fit
        self._linear = torch.nn.Linear(hidden_size, vocabulary_size)

        # for predict
        self._gamma = torch.nn.Parameter(torch.FloatTensor())
        self._s = torch.nn.Parameter(torch.FloatTensor(1 + lstm_layer_num))

    def fit(self, pieces: torch.LongTensor):
        x = self._embedding(pieces)
        for lstm in self._lstm:
            current_x, _ = lstm(x)
            x = x + current_x
        x = self._linear(x)
        return x

    def forward(self, pieces: torch.LongTensor):
        embedded = self._embedding(pieces)
        ret = self._s[0] * embedded

        x = embedded
        for i in range(len(self._lstm)):
            current_x, _ = self._lstm[i](x)
            ret = ret + self._s[i + 1] * current_x
            # ResNet
            x = x + current_x

        # embedding_dim + 2 * hidden_size
        return torch.cat((embedded, self._gamma * ret), 1)
