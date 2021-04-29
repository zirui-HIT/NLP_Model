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

        self.append(['[BOS]', '[EOS]', '[PAD]'])

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
            if not (key in self._word2index):
                return self._word2index['[PAD]']
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

    def __init__(self, words: List[str], label: int, pid: int):
        """Instantiate sentence

        Args:
            words: words of sentence
            label: label of sentence
            pid: id of phrase
        """
        self._words = deepcopy(words)
        self._label = label
        self._pid = pid

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

    def pid(self) -> int:
        """pid of sentence

        Returns:
            pid of sentence
        """
        return self._pid

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

    def load(self, path: str, max_length: int = None) -> Vocabulary:
        """load data from path

        data must be saved in .csv or .tsv file

        Args:
            path: path of data
            max_length: max lines to load

        Returns:
            vocabulary of loading data
            occurrence of every label 
        """
        import pandas as pd
        data = pd.read_csv(path, sep='\t')
        vocabulary = Vocabulary()
        count: Dict[str, int] = {}
        if max_length is None:
            max_length = len(data)

        if self._mode == 'train' or self._mode == 'valid':
            for i in range(len(data)):
                if i >= max_length:
                    break

                words = data['Phrase'][i].split()
                words = ['<BOS>'] + words + ['<EOS>']
                label = data['Sentiment'][i]

                self._sentences.append(
                    Sentence(words, label, data['PhraseId'][i]))
                if not (label in count):
                    count[label] = 0
                count[label] += 1
                vocabulary.append(words)

            return vocabulary, count
        else:
            for i in range(len(data)):
                if i >= max_length:
                    break

                words = data['Phrase'][i].split()
                words = ['<BOS>'] + words + ['<EOS>']
                label = ''
                self._sentences.append(
                    Sentence(words, label, data['PhraseId'][i]))

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
            words.append(s.words())
            labels.append(s.label())

        return DataLoader(dataset=_DataSet(words, labels),
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=_collate_fn)

    def labels(self) -> List[int]:
        """labels of data

        Returns:
            labels of data
        """
        ret = [s.label() for s in self._sentences]
        return ret

    def pids(self) -> List[int]:
        """pids of data

        Returns:
            pids of data
        """
        ret = [s.pid() for s in self._sentences]
        return ret

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

    package = train_dm.package(32)
    for current_sentences, current_labels in package:
        pass
