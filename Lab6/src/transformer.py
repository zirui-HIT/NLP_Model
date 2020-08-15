import layer
import torch


class Transformer(torch.nn.Module):
    def __init__(self,
                 vocabulary_size,
                 max_length,
                 position_matrix=None,
                 embedding_dim=64,
                 embedding_matrix=None,
                 head_num=4,
                 hidden_size=64,
                 hidden_dim=32,
                 hidden_layer=2,
                 output_dim=5,
                 dropout=0.1):
        super(Transformer, self).__init__()

        self.position_matrix = position_matrix
        self.hidden_layer = hidden_layer

        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)

        self.encoder = []
        self.encoder.append(
            layer.Encoder(embedding_dim, head_num, hidden_dim, hidden_dim))
        for i in range(1, hidden_layer):
            self.encoder.append(
                layer.Encoder(hidden_dim, head_num, hidden_dim, hidden_dim))

        self.decoder = []
        for i in range(0, hidden_layer):
            self.decoder.append(
                layer.Encoder(hidden_dim, head_num, hidden_dim, hidden_dim))

        self.dropout = torch.nn.Dropout(dropout)

        self.output = layer.OutputLayer(hidden_dim, max_length, output_dim)

    def forward(self, x):
        mask_x = (x != 0).long()

        x = self.embedding(x)
        position_matrix = self.position_matrix.unsqueeze(0).expand(x.shape)
        x = x + position_matrix

        mask_x = mask_x.unsqueeze(-1).expand(x.shape)
        x = x * mask_x.float()

        for i in range(self.hidden_layer):
            x = (self.encoder[i])(x)
            x = self.dropout(x)

        for i in range(self.hidden_layer):
            x = (self.decoder[i])(x)
            x = self.dropout(x)

        x = self.output(x)

        return x
