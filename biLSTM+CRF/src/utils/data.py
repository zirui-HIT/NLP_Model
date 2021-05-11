from typing import List, Dict
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

        self.append(['[PAD]', '[UNK]'])

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

    def __getitem__(self, key):
        """Select wanted value

        Select id with word or word with id

        Args:
            key: key of the wanted item

        Returns:
            Corresponding value to get
        """
        if isinstance(key, str):
            if not (key in self._word2index):
                return self._word2index['[UNK]']
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
    def __init__(self, words: List[str], labels: List[str]):
        if len(words) != len(labels):
            raise Exception('length error')
        self._words = words
        self._labels = labels

    def words(self):
        return self._words

    def labels(self):
        return self._labels

    def __hash__(self):
        return hash(self._words)


class DataManager(object):
    def __init__(self, mode: str):
        self._sentences: List[Sentence] = []
        self._mode = mode

    def load(self, path: str, max_length: int = None):
        if self._mode == 'train':
            word_vocabulary = Vocabulary()
            label_vocabulary = Vocabulary()

        with open(path, 'r') as f:
            current_words: List[str] = []
            current_labels: List[str] = []
            for line in f:
                words = line.strip().split()
                if words[0] == '-DOCSTART-':
                    continue

                if len(words) == 0:
                    self._sentences.append(
                        Sentence(current_words, current_labels))
                    if self._mode == 'train':
                        word_vocabulary.append(current_words)
                        label_vocabulary.append(current_labels)

                    current_words = []
                    current_labels = []

                current_words.append(words[0])
                if self._mode != 'test':
                    current_labels.append(words[3])

        if self._mode == 'train':
            return word_vocabulary, label_vocabulary

    def label(self) -> List[List[int]]:
        return [s.labels() for s in self._sentences]

    def word(self) -> List[List[int]]:
        return [s.words() for s in self._sentences]

    def package(self, batch_size: int, shuffle: bool = True):
        words = []
        labels = []
        for s in self._sentences:
            words.append(s.words())
            labels.append(s.labels())

        return DataLoader(dataset=_DataSet(words, labels),
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=_collate_fn)


class _DataSet(object):
    def __init__(self, words: List[List[str]], labels: List[List[str]]):
        self._words = words
        self._labels = labels

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
