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

        self._encoders = torch.nn.Sequential()
        self._decoders = torch.nn.Sequential()
        for i in range(self._layer_num):
            self._encoders.add_module('encoder%d' % (i),
                                      TransformerEncoder(head_num, embedding_dim))
            self._decoders.add_module('decoder%d' % (i),
                                      TransformerDecoder(head_num, embedding_dim))

        self._linear = torch.nn.Linear(embedding_dim, zh_input_size)
        self._dropout = torch.nn.Dropout(dropout)

    def forward(self, pieces: torch.LongTensor, targets: torch.LongTensor = None):
        batch_size, seq_length = pieces.size()
        en_embedded = self._en_embedding(pieces)
        en_embedded = self._position_embedding(
            batch_size, seq_length) + en_embedded

        encoder_attention = [(self._encoders[0])(en_embedded)]
        for i in range(1, self._layer_num):
            current_dropout = self._dropout(encoder_attention[-1])
            encoder_attention.append(
                (self._encoders[i])(current_dropout))

        # begin idx
        current_idx = self._bos_idx * torch.ones([batch_size, 1])
        current_idx = current_idx.long()
        if torch.cuda.is_available():
            current_idx = current_idx.cuda()

        zeros = torch.zeros([batch_size, 1, self._embedding_dim])
        if torch.cuda.is_available():
            zeros = zeros.cuda()

        output = []
        for t in range(1, self._max_length):
            current_state = self._zh_embedding(
                current_idx) + self._position_embedding(batch_size, t)

            # predict in the last position
            current_state = torch.cat([current_state, zeros], dim=1)
            for i in range(self._layer_num):
                current_state = (self._decoders[i])(
                    self._dropout(current_state), encoder_attention[i])

            current_score = self._linear(current_state)
            current_score = current_score[:, t, :]
            output.append(current_score)

            # teacher forcing
            if not(self._teacher_forcing_ratio is None) and not(targets is None) and random.random() < self._teacher_forcing_ratio and t < len(targets[0]):
                current_idx = torch.cat(
                    [current_idx, targets[:, t].unsqueeze(1)], dim=1)
            else:
                predict_idx = torch.argmax(current_score, dim=1).unsqueeze(1)
                current_idx = torch.cat([current_idx, predict_idx], dim=1)

        output = torch.stack(output, dim=0)
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

        result = embedded.unsqueeze(0).repeat(batch_size, 1, 1)
        if torch.cuda.is_available():
            result = result.cuda()
        return result


class TransformerEncoder(torch.nn.Module):
    def __init__(self, head_num: int, hidden_size: int):
        super(TransformerEncoder, self).__init__()

        self._multi_head_attention = MultiHeadAttention(head_num, hidden_size)
        self._feed_forward_net = FeedForwardNet(hidden_size)

    def forward(self, state: torch.FloatTensor):
        attention = self._multi_head_attention(state, state, state)
        norm_attention = torch.layer_norm(
            state + attention, attention.size()[1:])

        hidden = self._feed_forward_net(norm_attention)
        return torch.layer_norm(norm_attention + hidden, hidden.size()[1:])


class TransformerDecoder(torch.nn.Module):
    def __init__(self, head_num: int, hidden_size: int):
        super(TransformerDecoder, self).__init__()

        self._multi_head_attention = MultiHeadAttention(head_num, hidden_size)
        self._encoder_multi_head_attention = MultiHeadAttention(
            head_num, hidden_size)
        self._feed_forward_net = FeedForwardNet(hidden_size)

    def forward(self, prev_state: torch.FloatTensor, encoder_state: torch.FloatTensor):
        attention = self._multi_head_attention(
            prev_state, prev_state, prev_state)
        norm_attention = torch.layer_norm(
            prev_state + attention, attention.size()[1:])

        encoder_attention = self._encoder_multi_head_attention(
            norm_attention, encoder_state, encoder_state)
        norm_encoder_attention = torch.layer_norm(
            attention + encoder_attention, encoder_attention.size()[1:])

        hidden = self._feed_forward_net(norm_encoder_attention)
        return torch.layer_norm(hidden, hidden.size()[1:])


class FeedForwardNet(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super(FeedForwardNet, self).__init__()

        self._linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self._linear2 = torch.nn.Linear(hidden_size, hidden_size)
        if torch.cuda.is_available():
            self._linear1 = self._linear1.cuda()
            self._linear2 = self._linear2.cuda()

    def forward(self, state: torch.FloatTensor):
        linear1_state = self._linear1(state)
        relu_state = torch.relu(linear1_state)
        linear2_state = self._linear2(relu_state)
        return linear2_state


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, head_num: int, hidden_size: int):
        super(MultiHeadAttention, self).__init__()

        self._head_num = head_num
        self._hidden_size = hidden_size
        self._K_linear = torch.nn.Sequential()
        self._Q_linear = torch.nn.Sequential()
        self._V_linear = torch.nn.Sequential()
        for i in range(head_num):
            self._K_linear.add_module('K%d' % (i),
                                      torch.nn.Linear(hidden_size, hidden_size))
            self._Q_linear.add_module('Q%d' % (i),
                                      torch.nn.Linear(hidden_size, hidden_size))
            self._V_linear.add_module('V%d' % (i),
                                      torch.nn.Linear(hidden_size, hidden_size))

            if torch.cuda.is_available():
                self._K_linear[i] = self._K_linear[i].cuda()
                self._Q_linear[i] = self._Q_linear[i].cuda()
                self._V_linear[i] = self._V_linear[i].cuda()

        self._O_linear = torch.nn.Linear(head_num * hidden_size, hidden_size)
        if torch.cuda.is_available():
            self._O_linear = self._O_linear.cuda()

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
        d = pow(self._hidden_size, 0.5)
        return torch.matmul(torch.softmax(torch.matmul(Q, K.permute(0, 2, 1)) / d, dim=2), V)
