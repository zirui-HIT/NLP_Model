import numpy as np
import pandas as pd
import torch
import torch.nn.functional as function
import torch.utils.data as Data

BATCH_SIZE = 1000
EPOCH = 20
LR = 0.2


def build_vocabulary(phrase):
    ret = dict()
    cnt = 1

    for s in phrase:
        for x in s.split():
            if (x in ret.keys()):
                continue

            ret[x] = cnt
            cnt = cnt + 1

    return ret


def tokenizer(phrase, vocabulary):
    ret = []

    for s in phrase:
        tmp = []
        for x in s.split():
            tmp.append(vocabulary[x])
        ret.append(tmp)

    return ret


def expand_with_zero(phrase, length):
    ret = []
    for s in phrase:
        tmp = s.copy()
        for i in range(length - len(s)):
            tmp.append(0)
        ret.append(tmp)
    return ret


class RNN(torch.nn.Module):
    def __init__(self, n_feature, embedding_dim, n_hidden, n_output):
        super(RNN, self).__init__()

        self.embedding = torch.nn.Embedding(n_feature, embedding_dim)

        self.hidden = torch.nn.LSTM(embedding_dim, n_hidden, batch_first=True)

        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        length = []
        for s in x:
            for i in range(len(s)):
                if (s[i] == 0):
                    length.append(i - 1)
                    break

        x = self.embedding(x)

        x, _ = self.hidden(x)

        for i in range(len(x)):
            current = x[i][length[i]].unsqueeze(0)
            if (i == 0):
                t = current
            else:
                t = torch.cat((t, current), dim=0)

        x = self.out(t)

        return function.softmax(x, dim=1)


if __name__ == '__main__':
    train_data = pd.read_csv('../train/train.tsv', sep='\t')
    test_data = pd.read_csv('../test/test.tsv', sep='\t')

    train_phrase = list(train_data['Phrase'][:1000])
    train_sentiment = list(train_data['Sentiment'][:1000])
    test_phrase = list(test_data['Phrase'])

    vocabulary = build_vocabulary(train_phrase + test_phrase)

    train_phrase = tokenizer(train_phrase, vocabulary)
    test_phrase = tokenizer(test_phrase, vocabulary)

    max_length = len(max(train_phrase + test_phrase, key=len, default=''))
    train_phrase = expand_with_zero(train_phrase, max_length + 1)
    test_phrase = expand_with_zero(test_phrase, max_length + 1)

    train_x = torch.from_numpy(np.array(train_phrase)).type(torch.LongTensor)
    test_x = torch.from_numpy(np.array(test_phrase)).type(torch.LongTensor)
    train_y = torch.from_numpy(np.array(train_sentiment)).type(
        torch.LongTensor)

    rnn = RNN(len(vocabulary) + 1, 16, 40, 5)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    torch_dataset = Data.TensorDataset(train_x, train_y)
    loader = Data.DataLoader(dataset=torch_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=2)

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            out = rnn(batch_x)
            loss = loss_func(out, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_y = torch.max(rnn(test_x), 1)[1]

    outData = pd.DataFrame({
        'PhraseId': test_data.PhraseId,
        'Sentiment': test_y
    })
    outData.to_csv('../test/RNN_result.csv', index=False)
