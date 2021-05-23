from os import curdir
from typing import Dict, List, Tuple
from nltk import word_tokenize
from jieba import cut
from torch.utils.data import DataLoader


class Vocabulary(object):
    def __init__(self):
        self._word2idx: Dict[str, int] = {}
        self._idx2word: Dict[int, str] = {}
        self._word_num: int = 0

        self.append('<UNK>')
        self.append('<PAD>')
        self.append('<BOS>')
        self.append('<EOS>')

    def append(self, word: str):
        """append new word to vocabulary

        Args:
            word: word to be appended
        """
        if not (word in self._word2idx):
            self._word2idx[word] = self._word_num
            self._idx2word[self._word_num] = word
            self._word_num += 1

    def dump(self, path: str):
        """dump info of vocabulary

        format of line is 'word idx'

        Args:
            path: path to dump
        """
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(self._word_num):
                f.write('%s %d\n' % (self._idx2word[i], i))

    def load(self, path: str):
        """load info of vocabulary

        Args:
            path: path to load
        """
        self._word_num = 0
        self._idx2word.clear()
        self._word2idx.clear()

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                current_word = words[0]
                current_idx = int(words[1])

                self._word2idx[current_word] = current_idx
                self._idx2word[current_idx] = current_word
                self._word_num += 1

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._word2idx:
                return self._word2idx[key]
            else:
                return self._word2idx['<UNK>']

        if key in self._idx2word:
            return self._idx2word[key]
        else:
            return '<UNK>'

    def __len__(self):
        return self._word_num


class Sentence(object):
    def __init__(self, en_sentence: List[str], zh_sentence: List[str]):
        self._en_sentence: List[str] = en_sentence
        self._zh_sentence: List[str] = zh_sentence

    def zh_sentence(self) -> List[str]:
        return self._zh_sentence

    def en_sentence(self) -> List[str]:
        return self._en_sentence

    def __hash__(self) -> int:
        return hash(self._en_sentence + self._zh_sentence)


class DataManager(object):
    def __init__(self, mode: str):
        self._sentences: List[Sentence] = []
        self._mode = mode

    def load(self, path: str, max_line: int, zh_stopwords_path: str = 'data/chinese_stopwords.txt'):
        """load data from given path

        Args:
            path: path of data
            max_line: max line to read

        Returns:
            Vocabulary: english vocabulary if mode is train
            Vocabulary: chinese vocabulary if mode is train
        """
        import json

        if self._mode == 'train':
            en_vocabulary = Vocabulary()
            zh_vocabulary = Vocabulary()

        cnt = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                current_data = json.loads(line)

                if self._mode != 'test':
                    current_zh_sentence = self._zh_tokenize(
                        current_data['chinese'])
                else:
                    current_zh_sentence = []
                current_en_sentence = self._en_tokenize(
                    current_data['english'])

                self._sentences.append(
                    Sentence(current_en_sentence, current_zh_sentence))

                if self._mode == 'train':
                    for w in current_zh_sentence:
                        zh_vocabulary.append(w)
                    for w in current_en_sentence:
                        en_vocabulary.append(w)

                cnt += 1
                if cnt >= max_line:
                    break

        if self._mode == 'train':
            return en_vocabulary, zh_vocabulary

    def zh_sentences(self):
        return [s.zh_sentence() for s in self._sentences]

    def en_sentences(self):
        return [s.en_sentence() for s in self._sentences]

    def package(self, batch_size: bool, shuffle: bool) -> DataLoader:
        """pack the data

        Args:
            batch_size: size of every batch
            shuffle: if reshuffle data when getting data

        Returns:
            DataLoader: english sentence, chinese sentence
        """
        en_sentences = []
        zh_sentences = []
        for s in self._sentences:
            en_sentences.append(s.en_sentence())
            zh_sentences.append(s.zh_sentence())

        return DataLoader(dataset=_DataSet(en_sentences, zh_sentences),
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=_collate_fn)

    def _en_tokenize(self, sentence: str) -> List[str]:
        words = word_tokenize(sentence)
        words = [w.lower() for w in words]
        return words

    def _zh_tokenize(self, sentence: str) -> List[str]:
        words = cut(sentence)
        return list(words)


class _DataSet(object):
    def __init__(self, en_sentences: List[str], zh_sentences: List[str]):
        self._en_sentences = en_sentences
        self._zh_sentences = zh_sentences

    def __len__(self):
        return len(self._en_sentences)

    def __getitem__(self, idx: int):
        return self._en_sentences[idx], self._zh_sentences[idx]


def _collate_fn(batch: _DataSet):
    attr_count = len(batch[0])
    ret = [[] for i in range(attr_count)]

    for i in range(len(batch)):
        for j in range(attr_count):
            ret[j].append(batch[i][j])

    return ret
