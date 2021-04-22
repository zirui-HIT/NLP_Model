import torch
import numpy as np
from data import Vocabulary, DataManager
from copy import deepcopy
from typing import List


class Processor(object):
    def __init__(self, vocabulary: Vocabulary, model):
        self._vocabulary = deepcopy(vocabulary)
        self._model = deepcopy(model)

    def fit(self, data: DataManager) -> float:
        """train model with data

        train model with train_data and update model by the accuracy of it

        Args:
            data: train data, which contains sentence and label

        Returns:
            accuracy on train data
        """
        pass

    def validate(self, data: DataManager) -> float:
        """validate model

        Args:
            data: validate data

        Returns:
            accuracy on validation data
        """
        pass

    def predict(self, data: DataManager) -> List[int]:
        """predict label

        Args:
            data: test data

        Returns:
            label of every data
        """
        pass

    def load(self, path: str):
        """load model and vocabulary from given path

        Args:
            path: path of model
        """
        self._model.load_state_dict(torch.load(path))
        self._vocabulary.load(path)

    def dump(self, path: str):
        """dump model and vocabulary to given path

        Args:
            path: path to be dumped
        """
        torch.save(self._model.state_dict(), path)
        self._vocabulary.dump(path)
    
    def _wrap_sentence(self, sentences: List[List[str]]) -> List[List[int]]:
        
