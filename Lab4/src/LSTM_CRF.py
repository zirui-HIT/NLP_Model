import torch
import layer


class LSTM_CRF(torch.nn.Module):
    def __init__(self,
                 vocabulary_size,
                 embedding_dim,
                 embedding_matrix=None,
                 hidden_size=128,
                 label_num=9,
                 dropout=0.1):
        super(LSTM_CRF, self).__init__()

        self.embedding = layer.EmbeddingLayer(vocabulary_size, embedding_dim,
                                              embedding_matrix)
        self.rnn = layer.RNNLayer(embedding_dim, hidden_size, label_num,
                                  dropout)
        self.crf = layer.CRF(label_num)
        self.out = layer.OutputLayer()

    def forward(self, x, y):
        mask_x = (x != 0).long()

        x = self.embedding(x)
        x = self.rnn(x)

        p_x = self.crf.forward(x, mask_x)
        r_x = self.crf.get_sentence_score(x, y, mask_x)

        return self.out(p_x, r_x)
