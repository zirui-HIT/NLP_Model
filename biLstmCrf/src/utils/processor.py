import torch
from tqdm import tqdm
from typing import List
from utils.data import Vocabulary, DataManager


class Processor(object):
    def __init__(self,
                 batch_size: int,
                 word_vocabulary: Vocabulary = None,
                 label_vocabulary: Vocabulary = None,
                 model: torch.nn.Module = None):
        self._word_vocabulary = word_vocabulary
        self._label_vocabulary = label_vocabulary
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

            valid_acc = self.validate(valid_data)
            if valid_acc > best_acc:
                best_acc = valid_acc
                self.dump(path)

            print("epoch %d: loss is %f, best_acc is %.4f%%" %
                  (e, loss_sum, best_acc * 100))

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
        self._model.eval()

        ret: List[List[int]] = []
        package = data.package(self._batch_size, False)
        for current_sentences, current_labels in tqdm(package):
            packed_sentences, length = self._wrap_sentence(current_sentences)

            predict_labels = list(self._model(packed_sentences, length))
            ret = ret + [
                predict_labels[i][1:length[i] - 1]
                for i in range(len(predict_labels))
            ]
        self._model.train()

        ret = [[self._label_vocabulary[int(x)] for x in s] for s in ret]
        return ret

    def dump(self, path: str):
        torch.save(self._model, path + '.pkl')
        self._label_vocabulary.dump(path + '_slot_vocabulary.txt')
        self._word_vocabulary.dump(path + '_word_vocabulary.txt')

    def load(self, path: str):
        self._model = torch.load(path + '.pkl')
        self._label_vocabulary = Vocabulary()
        self._word_vocabulary = Vocabulary()
        self._label_vocabulary.load(path + '_slot_vocabulary.txt')
        self._word_vocabulary.load(path + '_word_vocabulary.txt')

    def _wrap_sentence(self,
                       sentences: List[List[str]],
                       labels: List[List[str]] = None):
        length = [2 + len(s) for s in sentences]
        max_length = max(length)

        packed_sentences = []
        for i in range(len(sentences)):
            sentence = ['[BOS]'] + sentences[i] + ['[EOS]']
            current_sentence = [self._word_vocabulary[w] for w in sentence]
            current_sentence += [
                self._word_vocabulary['[PAD]']
                for i in range(max_length - length[i])
            ]
            packed_sentences.append(current_sentence)

        packed_sentences = torch.LongTensor(packed_sentences)
        if torch.cuda.is_available():
            packed_sentences = packed_sentences.cuda()
        if labels is None:
            return packed_sentences, length

        packed_labels = []
        for i in range(len(labels)):
            label = ['[BOL]'] + labels[i] + ['[EOL]']
            current_label = [self._label_vocabulary[l] for l in label]
            current_label += [
                self._label_vocabulary['[PAD]']
                for i in range(max_length - length[i])
            ]
            packed_labels.append(current_label)

        return packed_sentences, packed_labels, length
