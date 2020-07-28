import math
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as function
import torch.utils.data as Data

BATCH_SIZE = 1000
EPOCH = 20

def build_vocabulary(phrase):
    ret = dict()
    cnt = 1
    
    for s in phrase:
        for x in s.split():
            if(x in ret.keys()):
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

class CNN(torch.nn.Module):
    def __init__(self, n_feature, embedding_dim, n_output, max_length):
        super(CNN, self).__init__()

        self.embedding = torch.nn.Embedding(n_feature, embedding_dim)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = embedding_dim,
                out_channels = 16,
                kernel_size = 5,
                stride = 1,
                padding = 2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(
                kernel_size = 2,
                padding = max_length % 2
            )
        )
        
        self.out = torch.nn.Linear(math.ceil(max_length / 2) * 16, n_output)

    def forward(self, x):
        x = self.embedding(x)

        x = x.permute(0, 2, 1)
        x = self.conv(x)
        
        x = x.view(x.size(0), -1)
        x = self.out(x)
        
        return x
    
if __name__ == '__main__': 
    train_data = pd.read_csv('../train/train.tsv', sep = '\t')
    test_data = pd.read_csv('../test/test.tsv', sep = '\t')

    train_phrase = list(train_data['Phrase'][:10000])
    train_sentiment = list(train_data['Sentiment'][:10000])
    test_phrase = list(test_data['Phrase'])

    vocabulary = build_vocabulary(train_phrase + test_phrase)
    
    train_phrase = tokenizer(train_phrase, vocabulary)
    test_phrase = tokenizer(test_phrase, vocabulary)

    max_length = len(max(train_phrase + test_phrase, key = len, default = ''))
    train_phrase = expand_with_zero(train_phrase, max_length)
    test_phrase = expand_with_zero(test_phrase, max_length)

    train_x = torch.from_numpy(np.array(train_phrase)).type(torch.LongTensor)
    test_x = torch.from_numpy(np.array(test_phrase)).type(torch.LongTensor)
    train_y = torch.from_numpy(np.array(train_sentiment)).type(torch.LongTensor)

    cnn = CNN(len(vocabulary) + 1, 32, 5, max_length)
    optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.2)
    loss_func = torch.nn.CrossEntropyLoss()

    torch_dataset = Data.TensorDataset(train_x, train_y)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2
    )
    
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            out = cnn(batch_x)
            loss = loss_func(out, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    test_y = torch.max(function.softmax(cnn(test_x), dim = 1), 1)[1]

    outData = pd.DataFrame({'PhraseId':test_data.PhraseId, 'Sentiment':test_y})
    outData.to_csv('../test/CNN_result.csv', index = False)
    
