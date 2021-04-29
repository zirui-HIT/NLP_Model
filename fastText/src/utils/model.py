import torch


class FastText(torch.nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int,
                 dropout_rate: float, tree_size: int, padding_idx: int):
        """initialize FastText model

        Args:
            vocabulary_size: size of vocabulary
            embedding_dim: dim of out of embedding layer
            dropout_rate: rate of dropout which in [0, 1]
            tree_size: number of nodes of huffman tree
            padding_idx: sign of padding that will be ignored after embedding
        """
        super(FastText, self).__init__()

        self._embedding = torch.nn.Embedding(num_embeddings=vocabulary_size,
                                             embedding_dim=embedding_dim,
                                             padding_idx=padding_idx)

        self._tree_param = torch.nn.Linear(in_features=embedding_dim,
                                           out_features=tree_size)

    def forward(self,
                pieces: torch.Tensor):
        """calc log likelihood of path

        Args:
            pieces: sentences which have been indexed

        Returns:
            probability of taking left child on every node
        """
        embed = self._embedding(pieces)

        feature = torch.sum(embed, dim=1)

        trans = torch.sigmoid(self._tree_param(feature))

        return trans
