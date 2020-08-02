import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, embedding_matrix=None):
        super(EmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)

    def forward(self, x):
        x = self.embedding(x)

        return x


class EncodingLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(EncodingLayer, self).__init__()

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        return x


class LocalInferenceLayer(nn.Module):
    def __init__(self):
        super(LocalInferenceLayer, self).__init__()

        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

    def forward(self, p, h, p_mask, h_mask):
        e = torch.matmul(p, h.transpose(1, 2))

        inference_mask = torch.matmul(
            p_mask.unsqueeze(2).float(),
            h_mask.unsqueeze(1).float())
        e.masked_fill_(inference_mask < 1e-7, -1e7)

        h_score, p_score = self.softmax1(e), self.softmax2(e)
        h_ = h_score.transpose(1, 2).bmm(p)
        p_ = p_score.bmm(h)

        m_p = torch.cat((p, p_, p - p_, p * p_), dim=-1)
        m_h = torch.cat((h, h_, h - h_, h * h_), dim=-1)

        return m_p, m_h


class CompositionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.0):
        super(CompositionLayer, self).__init__()

        self.hidden = nn.Linear(input_dim, output_dim)
        self.lstm = nn.LSTM(output_dim,
                            hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)

        return x


class PoolingLayer(nn.Module):
    def __init__(self):
        super(PoolingLayer, self).__init__()

    def forward(self, x, x_mask):
        mask_expand = x_mask.unsqueeze(-1).expand(x.shape)

        x_ = x * mask_expand.float()
        v_avg = x_.sum(1) / x_mask.sum(-1).unsqueeze(-1).float()

        x_ = x.masked_fill(mask_expand == 0, -1e7)
        v_max = x_.max(1)[0]

        return torch.cat((v_avg, v_max), dim=-1)


class InferenceCompositionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.0):
        super(InferenceCompositionLayer, self).__init__()

        self.composition = CompositionLayer(input_dim, output_dim, hidden_dim,
                                            dropout)
        self.pooling = PoolingLayer()

    def forward(self, m_p, m_h, p_mask, h_mask):
        v_p, v_h = self.composition(m_p), self.composition(m_h)
        v_p_, v_h_ = self.pooling(v_p, p_mask), self.pooling(v_h, h_mask)

        return torch.cat((v_p_, v_h_), dim=-1)


class OutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim, result_num, dropout=0.0):
        super(OutputLayer, self).__init__()

        self.mlp = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(input_dim, output_dim), nn.ReLU(),
                                 nn.Linear(output_dim, result_num))

    def forward(self, x):
        return self.mlp(x)
