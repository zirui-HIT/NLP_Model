import torch
import numpy as np
from copy import deepcopy
from typing import List
from tqdm import tqdm
from utils.huffman import HuffmanTree
from utils.data import Vocabulary, DataManager
from utils.model import FastText


class Processor(object):
    def __init__(self,
                 batch_size: int,
                 shuffle: bool,
                 vocabulary: Vocabulary = None,
                 huffman_tree: HuffmanTree = None,
                 model: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None):
        self._vocabulary = deepcopy(vocabulary)
        self._huffman_tree = deepcopy(huffman_tree)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._loss = torch.nn.L1Loss()
        self._optimizer = optimizer
        self._model = model

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
        for e in range(epoch):
            for current_sentences, current_labels in tqdm(package,
                                                          ncols=len(package)):
                sentences = self._wrap_sentence(current_sentences)
                pos_path, neg_path = self._wrap_tree_path(current_labels)

                probability = self._model(sentences, pos_path, neg_path)
                ones = torch.ones(probability.size(), requires_grad=True)
                loss = self._loss(probability, ones)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            """
            if self.validate(valid_data) > best_accuracy:
                self.dump(path)
            """

            print("epoch %d's loss is %f" % (e, loss.item()))

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
        self._model = torch.load(path + '.pkl')
        self._vocabulary = Vocabulary()
        self._vocabulary.load(path + '_vocabulary.txt')
        self._huffman_tree = HuffmanTree()
        self._huffman_tree.load(path + 'huffman_tree.txt')

    def dump(self, path: str):
        """dump model and vocabulary to given path

        Args:
            path: path to be dumped
        """
        torch.save(self._model, path + '.pkl')
        self._vocabulary.dump(path + '_vocabulary.txt')
        self._huffman_tree.dump(path + '_huffman_tree.txt')

    def _wrap_sentence(self, sentences: List[List[str]]) -> torch.Tensor:
        indexes = [[self._vocabulary.get(w) for w in s] for s in sentences]
        length = len(max(sentences, key=len))
        pad_index = self._vocabulary.get('[PAD]')

        for i in range(len(indexes)):
            indexes[i] = indexes[i] + \
                [pad_index for i in range((length - len(indexes[i])))]

        return torch.LongTensor(indexes)

    def _wrap_tree_path(self, path: List[int]) -> torch.Tensor:
        pos_ret = []
        neg_ret = []
        for x in path:
            pos_nodes, neg_nodes = self._huffman_tree.get(x)

            pos_current = [1 if i in pos_nodes else 0 for i in range(
                len(self._huffman_tree))]
            neg_current = [1 if i in neg_nodes else 0 for i in range(
                len(self._huffman_tree))]

            pos_ret.append(pos_current)
            neg_ret.append(neg_current)

        return torch.LongTensor(pos_ret), torch.LongTensor(neg_ret)
