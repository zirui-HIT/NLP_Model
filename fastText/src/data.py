from typing import List, Dict
from copy import deepcopy
from torch.utils.data import DataLoader


class Vocabulary(object):
    """Vocabulary to record words.

    Map word to id or id to word.

    Attributes:
    """

    def __init__(self):
        """Instantiate vocabulary
        """
        self._word_count = 0
        self._index2word: Dict[int, str] = {}
        self._word2index: Dict[str, int] = {}

        self.append(['[BOS]', '[EOS]', '[SEP]', '[PAD]'])

    def append(self, words: List[str]):
        """Append words to vocabulary

        New words id leads from before.

        Args:
            words: words to be appended
        """
        for w in words:
            if w in self._word2index:
                continue
            self._index2word[self._word_count] = w
            self._word2index[w] = self._word_count
            self._word_count += 1

    def get(self, key):
        """Select wanted value

        Select id with word or word with id

        Args:
            key: key of the wanted item

        Returns:
            Corresponding value to get
        """
        if isinstance(key, str):
            if not(key in self._word2index):
                return self._word2index['SEP']
            return self._word2index[key]
        else:
            return self._index2word[key]

    def load(self, path: str):
        """load vocabulary from path

        Args:
            path: path of vocabulary
        """
        self._word2index.clear()
        self._index2word.clear()
        with open(path, 'r') as f:
            for line in f:
                items = line.split()
                word = items[0]
                index = int(items[1])

                self._word2index[word] = index
                self._index2word[index] = word
        self._word_count = len(self._word2index)

    def dump(self, path: str):
        """dump vocabulary to given path

        Args:
            path: path to be dumped
        """
        with open(path, 'w') as f:
            for w in self._word2index:
                f.write('%s %s\n' % (w, self._word2index[w]))

    def __hash__(self):
        return hash(self._word2index)

    def __len__(self):
        return self._word_count

    def __add__(self, o):
        words = []
        vocabulary = Vocabulary()

        for w in o._word2index:
            words.append(w)
        for w in self._word2index:
            words.append(w)

        vocabulary.append(words)
        return vocabulary


class Sentence(object):
    """Record sentence info
    """

    def __init__(self, words, label: int):
        """Instantiate sentence

        Args:
            words: words of sentence
            label: label of sentence
        """
        self._words = deepcopy(words)
        self._label = label

    def words(self):
        """words of sentence

        Returns:
            words of sentence
        """
        return deepcopy(self._words)

    def label(self) -> int:
        """label of sentence

        Returns:
            label of sentence
        """
        return self._label

    def __len__(self):
        return len(self._words)

    def __hash__(self):
        return hash(self._words) + hash(self._label)


class DataManager(object):
    """Record all items of data
    """

    def __init__(self, mode: str):
        """Instantiate dataset

        Args:
            mode: train, valid or test
        """
        self._sentences: List[Sentence] = []
        self._mode = mode

    def load(self, path: str) -> Vocabulary:
        """load data from path

        data must be saved in .csv or .tsv file

        Args:
            path: path of data

        Returns:
            vocabulary of loading data
        """
        import pandas as pd
        data = pd.read_csv(path, sep='\t')
        vocabulary = Vocabulary()

        for i in range(len(data)):
            words = data['Phrase'][i].split()
            words = ['<BOS>'] + words + ['<EOS>']
            self._sentences.append(Sentence(
                words, data['Sentiment'][i]))

            vocabulary.append(words)

        return vocabulary

    def package(self, batch_size: int, shuffle: bool = False):
        """package data

        package all data with DataLoader

        Args:
            batch_size: size of every batch
            shuffle: if reshuffle data when loading data

        Returns:
            DataLoader containing all data
        """
        words = []
        labels = []
        for s in self._sentences:
            words.append(s.word())
            labels.append(s.label())

        return DataLoader(dataset=_DataSet(words, labels),
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=_collate_fn)

    def __len__(self):
        return len(self._sentences)

    def __hash__(self):
        return hash(self._sentences)


class _DataSet(object):
    def __init__(self, words: List[List[str]], labels: List[int]):
        self._words = deepcopy(words)
        self._labels = deepcopy(labels)

    def __len__(self):
        return len(self._words)

    def __getitem__(self, i: int):
        return self._words[i], self._labels[i]


def _collate_fn(batch):
    attr_count = len(batch[0])
    ret = [[] for i in range(attr_count)]

    for i in range(len(batch)):
        for j in range(attr_count):
            ret[j].append(batch[i][j])

    return ret


if __name__ == '__main__':
    train_dm = DataManager('train')
    train_vocabulary = train_dm.load(
        'Data/Sentiment_Analysis_on_Movie_Reviews/train.tsv')
    valid_dm = DataManager('valid')
    valid_vocabulary = valid_dm.load(
        'Data/Sentiment_Analysis_on_Movie_Reviews/valid.tsv')

    vocabulary = train_vocabulary + valid_vocabulary
    pass
