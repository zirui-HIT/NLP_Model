import torch
import numpy
import random
from typing import List
from config import args
from utils.model import Transformer
from utils.data import DataManager
from utils.processor import Processor


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    set_random_seed(args.random_seed)

    if args.mode == 'train':
        train_data = DataManager('train')
        valid_data = DataManager('valid')
        max_length, en_vocabulary, zh_vocabulary = train_data.load(
            args.train_path)
        valid_data.load(args.valid_path)

        model = Transformer(zh_input_size=len(zh_vocabulary),
                            en_input_size=len(en_vocabulary),
                            embedding_dim=args.embedding_dim,
                            layer_num=args.layer_num,
                            head_num=args.head_num,
                            padding_idx=en_vocabulary['<PAD>'],
                            bos_idx=zh_vocabulary['<BOS>'],
                            dropout=args.dropout,
                            max_length=max_length,
                            teacher_forcing_ratio=args.teacher_forcing_ratio)
        if torch.cuda.is_availabel():
            model = model.cuda()

        processor = Processor(batch_size=args.batch_size,
                              en_vocabulary=en_vocabulary,
                              zh_vocabulary=zh_vocabulary,
                              model=model)
        processor.fit(path=args.model_path,
                      epoch=args.epoch,
                      lr=args.learning_rate,
                      train_data=train_data,
                      valid_data=valid_data)
    elif args.mode == 'predict':
        test_data = DataManager('test')
        test_data.load(args.test_path)

        processor = Processor(batch_size=args.batch_size)
        processor.load(args.model_path)

        predict_sentences = processor.predict(test_data)
        predict_concat_sentences: List[str] = []
        for s in predict_sentences:
            current_sentence = ""
            for w in s:
                current_sentence = current_sentence + w
            predict_concat_sentences.append(current_sentence)

        en_sentences = test_data.en_sentences()
        en_concat_sentences: List[str] = []
        for s in en_sentences:
            current_sentence = ""
            for w in s:
                current_sentence = current_sentence + w
            en_concat_sentences.append(current_sentence)

        import json
        result = [{'english': en_concat_sentences[i], 'chinese': predict_concat_sentences[i]}
                  for i in range(len(predict_concat_sentences))]
        with open(args.save_path, 'w', encoding='utf-8') as f:
            for r in result:
                json.dump(r, f, ensure_ascii=False)
                f.write('\n')
