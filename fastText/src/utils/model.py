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
                pieces: torch.Tensor,
                tree_pos_path: torch.Tensor = None,
                tree_neg_path: torch.Tensor = None):
        """calc log likelihood of path

        Args:
            pieces: sentences which have been indexed
            tree_pos_path: 0/1 vector which means label passes left child of node
            tree_neg_path: 0/1 vector which means label passes right child of node

        Returns:
            log likelihood of sentences follow given path
        """
        # [batch_size, sentence_length, embedding_dim]
        embed = self._embedding(pieces)

        # [batch_size, embedding_dim]
        feature = torch.sum(embed, dim=1)

        if not (tree_pos_path is None or tree_neg_path is None):
            # [batch_size, tree_size]
            trans = torch.sigmoid(self._tree_param(feature))

            # [batch_size, 1]
            ones = torch.ones(trans.size(), requires_grad=True)
            if torch.cuda.is_available():
                ones = ones.cuda()
            log_likehood = torch.mul(tree_pos_path, trans) + torch.mul(
                tree_neg_path, torch.sub(ones, trans))

            ret = torch.sum(log_likehood, dim=1)
            return ret
        else:
            # TODO
            pass
