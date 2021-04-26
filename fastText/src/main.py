import torch
from config import args
from utils.data import DataManager
from utils.processor import Processor
from utils.huffman import HuffmanTree

if __name__ == '__main__':
    if args.mode == 'train':
        train_data = DataManager('train')
        valid_data = DataManager('valid')

        vocabulary, count = train_data.load(args.train_data_path,
                                            args.max_length)
        valid_data.load(args.valid_data_path, args.max_length)

        tree = HuffmanTree(count)

        processor = Processor(vocabulary, args.batch_size, True, args.lr)
        processor.fit(args.model_path, train_data, valid_data, args.epoch)
    elif args.mode == 'predict':
        test_data = DataManager('test')
        test_data.load(args.predict_data_path)

        processor = Processor(None, args.batch_size, False, args.lr)
        processor.load(args.model_path)

        pid = test_data.pids()
        predict_labels = processor.predict(test_data)

        import pandas as pd
        out = []
        for i in range(len(pid)):
            out.append({'PhraseId: %d, Sentiment: %d'} %
                       (pid[i], predict_labels[i]))
        out = pd.DataFrame(out, columns=['PhraseId', 'Sentiment'])
        out.to_csv(args.save_path, index=False)
