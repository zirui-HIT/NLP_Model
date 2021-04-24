import torch
import numpy as np
from data import Vocabulary, DataManager
from copy import deepcopy
from typing import List
from tqdm import tqdm


class Processor(object):
    def __init__(self, vocabulary: Vocabulary, batch_size: int, shuffle: bool,
                 model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self._vocabulary = deepcopy(vocabulary)
        self._batch_size = batch_size
        self._shuffle = shuffle

        self._loss = torch.nn.L1Loss()
        self._model = deepcopy(model)
        self._optimizer = deepcopy(optimizer)

    def fit(self,
            path: str,
            train_data: DataManager,
            valid_data: DataManager = None,
            epoch: int = 1) -> float:
        """train model with data

        train model with train_data and update model by the accuracy of it

        Args:
            train_data: train data, which contains sentence and label
            valid_data: data used to select best model
            path: path to save model

        Returns:
            accuracy on train data
        """
        self._model.train()
        if valid_data is None:
            valid_data = deepcopy(train_data)

        best_accuracy = 0
        package = train_data.package(self._batch_size, self._shuffle)
        for e in len(epoch):
            for current_sentences, current_labels in tqdm(package,
                                                          ncols=len(package)):
                sentences = self._wrap_sentence(current_sentences)
                labels = torch.Tensor(current_labels, dtype=torch.LongTensor)

                predict_labels = self._model(sentences)
                loss = self._loss(predict_labels, labels)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            if self.validate(valid_data) > best_accuracy:
                self.dump(path)

            print("epoch %d best accuracy: %f" % (e, best_accuracy))

    def validate(self, data: DataManager) -> float:
        """validate model

        calculate accuracy on validation data with processor model

        Args:
            data: validate data that contains sentences and labels

        Returns:
            accuracy on validation data
        """
        actual_labels = data.labels()
        predict_labels = self.predict(data)

        if len(actual_labels) != len(predict_labels):
            raise Exception("inequality length of actual and predict labels")

        count = 0
        length = len(actual_labels)
        for i in range(length):
            if actual_labels[i] == predict_labels[i]:
                count += 1

        return count / length

    def predict(self, data: DataManager) -> List[int]:
        """predict label

        Args:
            data: test data that only contains sentences

        Returns:
            label of every data
        """
        self._model.eval()

        package = data.package(self._batch_size, False)
        result_labels = []
        for current_sentences in tqdm(package, ncols=len(package)):
            sentences, mask = self._wrap_sentence(current_sentences)

            predict_labels = self._model(sentences, mask)
            _, predict_labels = predict_labels.topk(1, dim=1)
            result_labels.extend(predict_labels)

        return result_labels

    def load(self, path: str):
        """load model and vocabulary from given path

        Args:
            path: path of model
        """
        self._model.load_state_dict(torch.load(path + '.pkl'))
        self._vocabulary.load(path + '.txt')

    def dump(self, path: str):
        """dump model and vocabulary to given path

        Args:
            path: path to be dumped
        """
        torch.save(self._model.state_dict(), path + '.pkl')
        self._vocabulary.dump(path + '.txt')

    def _wrap_sentence(self, sentences: List[List[str]]) -> List[List[int]]:
        indexes = [[self._vocabulary.get(w) for w in s] for s in sentences]
        length = len(max(indexes, key=len))
        pad_index = self._vocabulary.get('[PAD]')

        for i in range(indexes):
            indexes[i] = indexes[i] + \
                [pad_index for i in range((length - len(indexes)))]

        return torch.Tensor(indexes, dtype=torch.LongTensor)
