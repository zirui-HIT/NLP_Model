import torch
from config import args
from utils.model import FastText
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

        model = FastText(vocabulary_size=len(vocabulary),
                         embedding_dim=args.embedding_dim,
                         dropout_rate=args.dropout_rate,
                         tree_size=len(tree),
                         padding_idx=vocabulary.get('[PAD]'))
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=args.learning_rate)

        processor = Processor(batch_size=args.batch_size,
                              shuffle=False,
                              vocabulary=vocabulary,
                              huffman_tree=tree,
                              model=model,
                              optimizer=optimizer)
        processor.fit(args.model_path, train_data, valid_data, args.epoch)
    elif args.mode == 'predict':
        test_data = DataManager('test')
        test_data.load(args.predict_data_path)

        processor = Processor(batch_size=args.batch_size,
                              shuffle=False)
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
