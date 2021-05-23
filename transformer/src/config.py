import argparse

parser = argparse.ArgumentParser()

# general argument
parser.add_argument('--random_seed',
                    '-rs',
                    type=int,
                    default=0,
                    help="random seed")
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='train',
                    help="train model or predict result")
parser.add_argument('--model_path',
                    '-mp',
                    type=str,
                    default='transformer/model/transformer',
                    help="path to save model")
parser.add_argument('--max_length',
                    '-ml',
                    type=int,
                    default=256,
                    help="max length to be processed")
parser.add_argument('--batch_size',
                    '-bs',
                    type=int,
                    default=2,
                    help="batch size")
parser.add_argument('--embedding_dim',
                    '-ed',
                    type=int,
                    default=128,
                    help='dimension of embedding layer')
parser.add_argument('--head_num',
                    '-hn',
                    type=int,
                    default=4,
                    help='head number of multi-head attention')
parser.add_argument('--layer_num',
                    '-ln',
                    type=int,
                    default=4,
                    help='layer number')
parser.add_argument('--max_zh_length',
                    '-mzl',
                    type=int,
                    default=100,
                    help='max length of chinese sentence')

# train mode argument
parser.add_argument('--train_data_path',
                    '-tdp',
                    type=str,
                    default='Data/Machine_Translation/train.json',
                    help="path of train data")
parser.add_argument('--valid_data_path',
                    '-vdq',
                    type=str,
                    default='Data/Machine_Translation/valid.json',
                    help="path of valid data")
parser.add_argument('--epoch', '-e', type=int, default=10000, help="epoch")
parser.add_argument('--learning_rate',
                    '-lr',
                    type=float,
                    default=1e-3,
                    help="learning rate")
parser.add_argument('--dropout_ratio',
                    '-dr',
                    type=float,
                    default=0.3,
                    help="dropout")
parser.add_argument('--teacher_forcing_ratio',
                    '-tfr',
                    type=float,
                    default=0.2,
                    help='teacher forcing ratio')

# test mode argument
parser.add_argument('--predict_data_path',
                    '-pdp',
                    type=str,
                    default='Data/Machine_Translation/test.json',
                    help="path of test data")
parser.add_argument('--save_path',
                    '-sp',
                    type=str,
                    default='Data/Machine_Translation/result_transformer.json',
                    help="output path")

args = parser.parse_args()
