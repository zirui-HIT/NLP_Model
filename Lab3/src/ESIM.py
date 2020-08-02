import torch
import layer


class ESIM(torch.nn.Module):
    def __init__(self,
                 hidden_dim,
                 vocabulary_size,
                 embedding_matrix=None,
                 embedding_dim=50,
                 result_num=3,
                 dropout=0.0):
        super(ESIM, self).__init__()

        self.embedder = layer.EmbeddingLayer(vocabulary_size, embedding_dim,
                                             embedding_matrix)
        self.encoder = layer.EncodingLayer(embedding_dim, hidden_dim)
        self.inference = layer.LocalInferenceLayer()
        self.inference_composition = layer.InferenceCompositionLayer(
            hidden_dim * 8, hidden_dim, hidden_dim, dropout)
        self.out = layer.OutputLayer(hidden_dim * 8, hidden_dim, result_num,
                                     dropout)

    def forward(self, p, h):
        p_embeded = self.embedder(p)
        h_embeded = self.embedder(h)

        p_ = self.encoder(p_embeded)
        h_ = self.encoder(h_embeded)

        p_mask, h_mask = (p != 0).long(), (h != 0).long()
        m_p, m_h = self.inference(p_, h_, p_mask, h_mask)

        v = self.inference_composition(m_p, m_h, p_mask, h_mask)

        logits = self.out(v)
        return torch.nn.functional.softmax(logits, dim=1)
