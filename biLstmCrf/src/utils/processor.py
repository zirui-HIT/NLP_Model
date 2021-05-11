import torch
import numpy as np
from tqdm import tqdm
from typing import List
from data import Vocabulary, DataManager


class Processor(object):
    def __init__(self,
                 batch_size: int,
                 word_vocabulary: Vocabulary = None,
                 slot_vocabulary: Vocabulary = None,
                 model: torch.nn.Module = None):
        self._word_vocabulary = word_vocabulary
        self._slot_vocabulary = slot_vocabulary
        self._batch_size = batch_size
        self._model = model

    def fit(self,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            path: str,
            train_data: DataManager,
            valid_data: DataManager = None):
        self._model.train()
        if valid_data is None:
            valid_data = train_data

        best_acc = 0
        package = train_data.package(self._batch_size, True)
        for e in range(epoch):
            loss_sum = 0
            for current_sentences, current_labels in tqdm(package):
                packed_sentences, packed_labels, length = self._wrap_sentence(
                    current_sentences, current_labels)

                loss = self._model(packed_sentences, length, packed_labels)
                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("epoch %d: loss is %f, best_acc is %f" %
                  (e, loss_sum, best_acc))

            valid_acc = self.validate(valid_data)
            if valid_acc > best_acc:
                best_acc = valid_data
                self.dump(path)

    def validate(self, data: DataManager):
        predicte_labels: List[List[int]] = self.predicte(data)
        labels = data.label()
        label_sum = 0
        label_cnt = 0

        for i in range(len(labels)):
            if len(predicte_labels[i]) != len(labels[i]):
                raise Exception('not same length')

            label_sum += len(labels[i])
            for j in range(len(labels[i])):
                if predicte_labels[i][j] == labels[i][j]:
                    label_cnt += 1

        return label_cnt / label_sum

    def predicte(self, data: DataManager):
        ret: List[List[int]] = []
        package = data.package(self._batch_size, False)
        for current_sentences, current_labels in tqdm(package):
            packed_sentences, length = self._wrap_sentence(
                current_sentences)

            predict_labels = list(self._model(current_sentences, length))
            ret = ret + [predict_labels[i][:length[i]]
                         for i in range(len(predict_labels))]

        return ret

    def dump(self, path: str):
        torch.save(self._model, path + '.pkl')
        self._slot_vocabulary.dump(path + '_slot_vocabulary.txt')
        self._word_vocabulary.dump(path + '_word_vocabulary.txt')

    def load(self, path: str):
        self._model = torch.load(path + '.pkl')
        self._slot_vocabulary = Vocabulary()
        self._word_vocabulary = Vocabulary()
        self._slot_vocabulary.load(path + '_slot_vocabulary.txt')
        self._word_vocabulary.load(path + '_word_vocabulary.txt')

    def _warp_sentence(self, sentences: List[List[str]],
                       labels: List[List[str]] = None):
        length: List[int] = []
        max_length = len(max(sentences, key=len))
        packed_sentences = []

        for i in range(len(sentences)):
            current_sentence = []
            length.append(len(sentences[i]))
            for j in range(length[i]):
                current_sentence.append(self._word_vocabulary(sentences[i][j]))
            packed_sentences.append(
                current_sentence + (max_length - length[i]) * self._word_vocabulary['[PAD]'])

        length = torch.LongTensor(length)
        packed_sentences = torch.LongTensor(length)

        if not(labels is None):
            packed_labels = []
            for i in range(len(labels)):
                current_label = []
                for j in range(length[i]):
                    current_label.append(self._slot_vocabulary(labels[i][j]))
                packed_labels.append(
                    current_label + (max_length - length[i]) * self._word_vocabulary['[PAD]'])

            packed_labels = torch.LongTensor(length)
            return packed_sentences, packed_labels, length

        return packed_sentences, length