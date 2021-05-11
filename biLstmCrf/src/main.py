import torch
import numpy
import random
import os
from biLstmCrf.src.config import args
from biLstmCrf.src.utils.data import DataManager
from biLstmCrf.src.utils.model import biLstmCrf
from biLstmCrf.src.utils.processor import Processor

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    numpy.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if args.mode == 'train':
        train_data = DataManager('train')
        valid_data = DataManager('valid')
        word_vocabulary, label_vocabulary = train_data.load(
            args.train_data_path)
        valid_data.load(args.valid_data_path)

        word_vocabulary.append(['[BOS]', '[EOS]'])
        label_vocabulary.append(['[BOL]', '[EOL]'])

        if os.path.isfile(args.model_path + '.pkl'):
            model = torch.load(args.model_path + '.pkl')
        else:
            model = biLstmCrf(vocabulary_size=len(word_vocabulary),
                              embedding_dim=args.embedding_dim,
                              hidden_dim=args.hidden_dim,
                              label_dim=len(label_vocabulary),
                              dropout=args.dropout,
                              padding_idx=word_vocabulary['[PAD]'],
                              begin_idx=label_vocabulary['[BOL]'],
                              end_idx=label_vocabulary['[EOL]'])
            if torch.cuda.is_available():
                model = model.cuda()

        processor = Processor(batch_size=args.batch_size,
                              word_vocabulary=word_vocabulary,
                              label_vocabulary=label_vocabulary,
                              model=model)

        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=args.learning_rate)
        processor(optimizer=optimizer,
                  epoch=args.epoch,
                  path=args.model_path,
                  train_data=train_data,
                  valid_data=valid_data)
    elif args.mode == 'predict':
        test_data = DataManager('test')

        processor = Processor(batch_size=args.batch_size)
        processor.load(args.model_path)

        predict_labels = processor.predict(test_data)
        words = test_data.word()

        with open(args.predict_data_path, 'w') as f:
            for i in range(len(words)):
                for j in range(len(words[i])):
                    f.write('%s %s\n' % (words[i][j], predict_labels[i][j]))
                f.write('\n')
