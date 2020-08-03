import torch
import numpy as np


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, embedding_matrix):
        super(EmbeddingLayer, self).__init__()

        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)

    def forward(self, x):
        x = self.embedding(x)

        return x


class RNNLayer(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, label_num, dropout):
        super(RNNLayer, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=embedding_dim,
                                  hidden_size=hidden_size,
                                  bidirectional=True,
                                  batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(2 * hidden_size, label_num)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


class CRF():
    def __init__(self, label_num):
        self.transition_matrix = torch.nn.Parameter(
            torch.rand(label_num, label_num))
        self.label_num = label_num

        torch.nn.init.normal_(self.transition_matrix)

    def forward(self, emission, mask):
        batch_size, length = mask.size()
        log_sum = torch.from_numpy(
            np.full((batch_size, self.label_num), -10000))
        log_sum[:, 0] = 0

        for t in range(length):
            mask_t = mask[:, t].unsqueeze(-1)
            emission_t = emission[:, t, :]
            log_sum_matrix = log_sum.unsqueeze(2).expand(
                -1, -1, self.label_num)
            emission_matrix_t = emission_t.unsqueeze(1).expand(
                -1, self.label_num, -1)

            log_sum = torch.logsumexp(
                log_sum_matrix + emission_matrix_t + self.transition_matrix,
                dim=1) * mask_t + log_sum * (1 - mask_t)

        return torch.logsumexp(log_sum, dim=1)

    def get_sentence_score(self, emission, label, mask):
        batch_size, length, label_num = emission.size()
        score = emission.new_zeros(batch_size)

        for i in range(length):
            mask_i = mask[:, i]
            emission_i = emission[:, i, :]

            emit_score = torch.cat([
                each_score[each_label].unsqueeze(-1)
                for each_score, each_label in zip(emission_i, label[:, i])
            ],
                                   dim=0)

            if i == length - 1:
                continue

            transition_score = torch.stack([
                self.transition_matrix[label[b, i], label[b, i + 1]]
                for b in range(batch_size)
            ])
            score += (emit_score + transition_score) * mask_i

        return score


class OutputLayer(torch.nn.Module):
    def __init__(self):
        super(OutputLayer, self).__init__()

    def forward(self, x, y):
        return (x - y).sum()
