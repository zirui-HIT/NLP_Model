import torch
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
            shuffle: bool,
            dropout: float,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            path: str,
            train_data: DataManager,
            valid_data: DataManager = None):
        self._model.train()
        if valid_data is None:
            valid_data = train_data

        best_acc = 0
        package = train_data.package(self._batch_size, shuffle)
        for e in range(epoch):
            loss_sum = 0
            for current_sentence, current_label in tqdm(package):
                packed_sentences, packed_labels, length = self._wrap_sentence(
                    current_sentence, current_label)

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
                       labels: List[List[str]]):
        length: List[int] = []
        packed_sentences = []
        packed_labels = []
        for i in range(len(sentences)):
            current_sentence = []
            current_label = []
            length.append(len(sentences[i]))
            for j in range(length[i]):
                current_sentence.append(self._word_vocabulary(sentences[i][j]))
                current_label.append(self._slot_vocabulary(labels[i][j]))
            packed_sentences.append(current_sentence)
            packed_labels.append(current_label)
