import torch
from tqdm import tqdm
from typing import List
from utils.data import Vocabulary, DataManager
from nltk.translate.bleu_score import sentence_bleu


class Processor(object):
    def __init__(self,
                 batch_size: int,
                 en_vocabulary: Vocabulary = None,
                 zh_vocabulary: Vocabulary = None,
                 model: torch.nn.Module = None):
        self._batch_size = batch_size
        self._en_vocabulary = en_vocabulary
        self._zh_vocabulary = zh_vocabulary
        self._model = model

    def fit(self,
            path: str,
            epoch: int,
            lr: float,
            train_data: DataManager,
            valid_data: DataManager = None):
        self._model.train()
        if valid_data is None:
            valid_data = train_data

        loss_function = torch.nn.CrossEntropy()
        optimizer = torch.optim.Adam(params=self._model.parameters(), lr=lr)
        self.dump(path, True)

        best_bleu = 0
        package = train_data.package(self._batch_size, True)
        for e in range(epoch):
            for current_en_sentences, current_zh_sentences in tqdm(package):
                current_en_sentences = self.wrap_sentence(
                    current_en_sentences, 'en', True)
                current_zh_sentences = self.wrap_sentence(
                    current_zh_sentences, 'zh', False)

                current_score = self._model(current_en_sentences,
                                            current_zh_sentences)
                loss = loss_function(current_score, current_zh_sentences)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            current_bleu = self.validate(valid_data)
            if current_bleu > best_bleu:
                best_bleu = current_bleu
                self.dump(path, False)

            print('epoch %d: loss is %f, best_bleu is %f' %
                  (e, loss.item(), best_bleu))

    def validate(self, data: DataManager):
        predict_zh_sentences = self.predict(data)
        real_zh_sentences = data.zh_sentences()

        # TODO calc BLEU
        bleu = 0
        for i in range(len(predict_zh_sentences)):
            bleu += sentence_bleu([predict_zh_sentences[i]],
                                  real_zh_sentences[i])

        self._model.train()
        return bleu / len(predict_zh_sentences)

    def predict(self, data: DataManager) -> List[List[str]]:
        self._model.eval()
        package = data.package(self._batch_size, False)
        ret = []
        for current_en_sentences, current_zh_sentences in tqdm(package):
            current_en_sentences = self._wrap_sentence(current_en_sentences,
                                                       'en', True)
            predict_score = self._model(current_en_sentences)
            predict_idx = torch.argmax(predict_score, dim=2)

            for b in predict_idx:
                current_sentence: List[str] = []
                for i in b:
                    current_word = self._zh_vocabulary(i)
                    if current_word == '<EOS>':
                        break
                    current_sentence.append(current_word)
                ret.append(current_sentence)

        return ret

    def dump(self, path: str, with_vocabulary: bool = False):
        torch.save(self._model, path + '.pkl')
        if with_vocabulary:
            self._en_vocabulary.dump(path + '_en.txt')
            self._zh_vocabulary.dump(path + '_zh.txt')

    def load(self, path: str):
        self._model = torch.load(path + '.pkl')

        self._en_vocabulary = Vocabulary()
        self._zh_vocabulary = Vocabulary()
        self._en_vocabulary.load(path + '_en.txt')
        self._zh_vocabulary.load(path + '_zh.txt')

    def _wrap_sentence(self,
                       sentences: List[List[str]],
                       language: str,
                       with_bos: bool = True):
        max_length: int = 0
        for i in range(len((sentences))):
            sentences[i] = sentences[i] + ['<EOS>']
            if with_bos:
                sentences[i] = ['<BOS>'] + sentences[i]
            max_length = max(max_length, len(sentences[i]))
        sentences = [
            s + ['<PAD>' for i in range(max_length - len(s))]
            for s in sentences
        ]

        if language == 'en':
            sentences = [[self._en_vocabulary[w] for w in s]
                         for s in sentences]
        else:
            sentences = [[self._zh_vocabulary[w] for w in s]
                         for s in sentences]

        return torch.LongTensor(sentences)
