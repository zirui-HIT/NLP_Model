import torch
from typing import List


class biLstmCrf(torch.nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int,
                 hidden_dim: int, label_dim: int, dropout: float,
                 padding_idx: int, begin_idx: int, end_idx: int):
        super(biLstmCrf, self).__init__()

        self._embedding = torch.nn.Embedding(num_embeddings=vocabulary_size,
                                             embedding_dim=embedding_dim,
                                             padding_idx=padding_idx)
        self._lstm = torch.nn.LSTM(input_size=embedding_dim,
                                   hidden_size=hidden_dim,
                                   bidirectional=True,
                                   batch_first=True)
        self._emission = torch.nn.Linear(in_features=2 * hidden_dim,
                                         out_features=label_dim)
        self._transition = torch.nn.Parameter(
            torch.rand([label_dim, label_dim]))

        self._end_idx = end_idx
        self._begin_idx = begin_idx

    def forward(self,
                tokens: torch.Tensor,
                length: List[int],
                labels: List[List[int]] = None):
        x = self._embedding(tokens)
        x, _ = self._lstm(x)
        x = self._emission(x)
        emission = torch.exp(x)
        transition = torch.exp(self._transition)

        size = list(x.size())
        batch_size = size[0]
        label_dim = size[2]
        if labels is None:
            best_point = torch.zeros_like(emission)
            path = torch.zeros_like(tokens)
            for i in range(batch_size):
                alpha = emission[i][0]
                for j in range(1, length[i] - 1):
                    for k in range(label_dim):
                        best_point[i][j][k] = torch.argmax(alpha +
                                                           (transition.T)[k])
                    for k in range(label_dim):
                        alpha[k] = (alpha + (transition.T)[k])[int(
                            best_point[i][j][k])]

                path[i][length[i] - 1] = self._end_idx
                for j in range(length[i] - 2, 0, -1):
                    path[i][j] = int(best_point[i][j + 1][path[i][j + 1]])
                path[i][0] = self._begin_idx

            return path
        else:
            neg_log_probability = torch.FloatTensor(batch_size)
            for i in range(batch_size):
                real_score = torch.clone(emission[i][0][labels[i][0]])
                for j in range(1, length[i]):
                    real_score += emission[i][j][labels[i][j]] + \
                        transition[labels[i][j - 1]][labels[i][j]]

                alpha = [emission[i][0]]
                for j in range(1, length[i]):
                    alpha.append(_log_sum_exp(
                        alpha[j-1].unsqueeze(1).expand(label_dim, label_dim) +
                        transition + emission[i][j].unsqueeze(0).expand(
                            label_dim, label_dim)))
                log_sum_score = _log_sum_exp(alpha[-1])

                neg_log_probability[i] = log_sum_score - real_score
                if neg_log_probability[i] < 0:
                    raise Exception('illegal')

            return torch.sum(neg_log_probability)


def _log_sum_exp(score):
    max_value = torch.max(score)
    return max_value + torch.log(
        torch.sum(torch.exp(score - max_value), dim=0))
