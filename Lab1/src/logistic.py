import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    trainData = pd.read_csv("../test/train.tsv", sep='\t')
    trainPhrase = trainData['Phrase']
    trainSentiment = trainData['Sentiment']

    testData = pd.read_csv("../test/test.tsv", sep='\t')
    testPhrase = testData['Phrase']

    vectorizer = CountVectorizer(ngram_range=(1, 3))
    vectorizer.fit(pd.concat([trainPhrase, testPhrase]))

    trainX = vectorizer.transform(trainPhrase)
    testX = vectorizer.transform(testPhrase)
    trainY = list(trainSentiment)

    model = LogisticRegression(max_iter=10000)
    model.fit(trainX, trainY)

    testY = model.predict(testX)

    outData = pd.DataFrame({'PhraseId': testData.PhraseId, 'Sentiment': testY})
    outData.to_csv('../test/result.csv', index=False)
