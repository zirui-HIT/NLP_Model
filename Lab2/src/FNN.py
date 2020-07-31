import numpy
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as function
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer


class FNN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(FNN, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = function.relu(self.hidden(x))
        x = self.predict(x)
        return x


if __name__ == '__main__':
    trainData = pd.read_csv("../train/train.tsv", sep='\t')

    trainPhrase = trainData['Phrase'][:1000]
    trainSentiment = trainData['Sentiment'][:1000]

    testData = pd.read_csv("../test/test.tsv", sep='\t')
    testPhrase = testData['Phrase']

    vectorizer = CountVectorizer(ngram_range=(1, 1))
    vectorizer.fit(pd.concat([trainPhrase, testPhrase]))

    trainX = vectorizer.transform(trainPhrase).todense()
    testX = vectorizer.transform(testPhrase).todense()
    trainY = numpy.array(list(trainSentiment))

    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)

    pca = PCA(n_components=0.9).fit(trainX)
    trainX = pca.transform(trainX)
    testX = pca.transform(testX)
    (ndim, dim) = trainX.shape

    trainX = torch.from_numpy(trainX).type(torch.FloatTensor)
    testX = torch.from_numpy(testX).type(torch.FloatTensor)
    trainY = torch.from_numpy(trainY).type(torch.LongTensor)

    trainX, testX, trainY = Variable(trainX), Variable(testX), Variable(trainY)

    fnn = FNN(dim, dim, 5)
    optimizer = torch.optim.SGD(fnn.parameters(), lr=0.2)
    loss_func = torch.nn.CrossEntropyLoss()

    for i in range(50):
        out = fnn(trainX)
        loss = loss_func(out, trainY)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    testY = torch.max(function.softmax(fnn(testX), dim=1), 1)[1]

    outData = pd.DataFrame({'PhraseId': testData.PhraseId, 'Sentiment': testY})
    outData.to_csv('../test/FNN_result.csv', index=False)
