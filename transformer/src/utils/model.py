import torch
import random


class Transformer(torch.nn.Module):
    def __init__(self, zh_input_size: int, en_input_size: int, embedding_dim: int, layer_num: int, head_num: int, padding_idx: int, bos_idx: int, dropout: float, max_length: int, teacher_forcing_ratio: float = None):
        super(Transformer, self).__init__()

        self._embedding_dim = embedding_dim
        self._max_length = max_length
        self._layer_num = layer_num
        self._bos_idx = bos_idx
        self._teacher_forcing_ratio = teacher_forcing_ratio

        self._zh_embedding = torch.nn.Embedding(
            zh_input_size, embedding_dim, padding_idx)
        self._en_embedding = torch.nn.Embedding(
            en_input_size, embedding_dim, padding_idx)
        self._encoders = [TransformerEncoder(
            head_num, embedding_dim) for i in range(layer_num)]
        self._decoders = [TransformerDecoder(
            head_num, embedding_dim) for i in range(layer_num)]
        self._linear = torch.nn.Linear(embedding_dim, zh_input_size)
        self._dropout = torch.nn.Dropout(dropout)

    def forward(self, pieces: torch.LongTensor, targets: torch.LongTensor = None):
        batch_size, seq_length = pieces.size()
        en_embedded = self._en_embedding(pieces)
        en_embedded = self._position_embedding(
            batch_size, seq_length) + en_embedded

        encoder_attention = [(self._encoders[0])(en_embedded)]
        for i in range(1, self._layer_num):
            encoder_attention.append(
                (self._encoders[i])(self._dropout(encoder_attention[-1])))

        output = []
        current_idx = self._bos_idx * torch.ones([batch_size, 1])
        for t in range(1, self._max_length):
            current_state = self._zh_embedding(
                current_idx) + self._position_embedding(batch_size, t, self._embedding_dim)
            current_state = torch.cat([current_state, torch.zeros(
                [batch_size, 1, self._embedding_dim])], dim=1)
            for i in range(self._layer_num):
                current_state = (self._decoders[i])(
                    self._dropout(current_state), encoder_attention[i])

            current_score = self._linear(current_state)
            current_score = current_score[:, t, :]
            output.append(current_score)

            # teacher forcing
            if not(self._teacher_forcing_ratio is None) and not(targets is None) and random.random() < self._teacher_forcing_ratio:
                current_idx = torch.cat([current_idx, targets[:, t, :]], dim=1)
            else:
                current_idx = torch.cat(
                    [current_idx, torch.argmax(current_score)], dim=1)

        output = torch.FloatTensor(output)
        return output.permute(1, 0, 2)

    def _position_embedding(self, batch_size: int, seq_length: int):
        from math import floor

        divisor = [pow(10000, -2 * floor(i / 2) / self._embedding_dim)
                   for i in range(self._embedding_dim)]
        dividend = range(seq_length)

        divisor = torch.FloatTensor([divisor])
        dividend = torch.FloatTensor([dividend])

        dividend = dividend.permute(1, 0)
        x = torch.matmul(dividend, divisor)

        coef_odd = torch.FloatTensor(
            [[i % 2 for i in range(self._embedding_dim)]])
        coef_even = torch.ones_like(coef_odd) - coef_odd
        embedded = coef_even * torch.sin(x) + coef_odd * torch.cos(x)

        return embedded.unsqueeze(0).repeat(batch_size, 1, 1)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, head_num: int, hidden_size: int):
        super(TransformerEncoder, self).__init__()

        self._multi_head_attention = MultiHeadAttention(head_num, hidden_size)
        self._layer_norm1 = torch.nn.LayerNorm(hidden_size)
        self._feed_forward_net = FeedForwardNet(hidden_size)
        self._layer_norm2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, state: torch.FloatTensor):
        attention = self._multi_head_attention(state, state, state)
        attention = self._layer_norm1(state + attention)

        hidden = self._feed_forward_net(attention)
        return self._layer_norm2(hidden)


class TransformerDecoder(torch.nn.Module):
    def __init__(self, head_num: int, hidden_size: int):
        super(TransformerDecoder, self).__init__()

        self._multi_head_attention = MultiHeadAttention(head_num, hidden_size)
        self._layer_norm1 = torch.nn.LayerNorm(hidden_size)
        self._encoder_multi_head_attention = MultiHeadAttention(
            head_num, hidden_size)
        self._layer_norm2 = torch.nn.LayerNorm(hidden_size)
        self._feed_forward_net = FeedForwardNet(hidden_size)
        self._layer_norm3 = torch.nn.LayerNorm(hidden_size)

    def forward(self, prev_state: torch.FloatTensor, encoder_state: torch.FloatTensor):
        attention = self._multi_head_attention(
            prev_state, prev_state, prev_state)
        attention = self._layer_norm1(prev_state + attention)

        encoder_attention = self._encoder_multi_head_attention(
            encoder_state, attention, encoder_state)
        encoder_attention = self._layer_norm2(attention + encoder_attention)

        hidden = self._feed_forward_net(encoder_attention)
        return self._layer_norm3(hidden)


class FeedForwardNet(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super(FeedForwardNet, self).__init__()

        self._sequential = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, state: torch.FloatTensor):
        return self._sequential(state)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, head_num: int, hidden_size: int):
        super(MultiHeadAttention, self).__init__()

        self._head_num = head_num
        self._hidden_size = hidden_size
        self._K_linear = []
        self._Q_linear = []
        self._V_linear = []
        for i in range(head_num):
            self._K_linear.append(torch.nn.Linear(hidden_size, hidden_size))
            self._Q_linear.append(torch.nn.Linear(hidden_size, hidden_size))
            self._V_linear.append(torch.nn.Linear(hidden_size, hidden_size))
        self._O_linear = torch.nn.Linear(head_num * hidden_size, hidden_size)

    def forward(self, Q: torch.FloatTensor, K: torch.FloatTensor, V: torch.FloatTensor):
        heads = []
        for i in range(self._head_num):
            current_Q = self._Q_linear[i](Q)
            current_K = self._K_linear[i](K)
            current_V = self._V_linear[i](V)
            heads.append(self._attention(current_Q, current_K, current_V))

        head = torch.cat(heads, dim=2)
        return self._O_linear(head)

    def _attention(self, Q: torch.FloatTensor, K: torch.FloatTensor, V: torch.FloatTensor):
        K = K.permute(0, 2, 1)
        d = pow(self._hidden_size, 0.5)
        return torch.matmul(torch.softmax(torch.matmul(Q, K) / d, dim=2), V)
