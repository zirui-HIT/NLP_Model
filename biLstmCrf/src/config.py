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
                    default='biLstmCrf/model/biLstmCrf',
                    help="path to save model")
parser.add_argument('--max_length',
                    '-ml',
                    type=int,
                    default=128,
                    help="max length to be processed")
parser.add_argument('--batch_size',
                    '-bs',
                    type=int,
                    default=16,
                    help="batch size")
parser.add_argument('--embedding_dim',
                    '-ed',
                    type=int,
                    default=16,
                    help='dimension of embedding layer')
parser.add_argument('--hidden_dim',
                    '-hd',
                    type=int,
                    default=16,
                    help='dimension of hidden layer')

# train mode argument
parser.add_argument('--train_data_path',
                    '-tdp',
                    type=str,
                    default='Data/Named_Entity_Recognition/train.txt',
                    help="path of train data")
parser.add_argument('--valid_data_path',
                    '-vdq',
                    type=str,
                    default='Data/Named_Entity_Recognition/valid.txt',
                    help="path of valid data")
parser.add_argument('--epoch', '-e', type=int, default=20, help="epoch")
parser.add_argument('--learning_rate',
                    '-lr',
                    type=float,
                    default=1e-2,
                    help="learning rate")
parser.add_argument('--dropout_rate',
                    '-dr',
                    type=float,
                    default=0.3,
                    help="dropout")

# test mode argument
parser.add_argument('--predict_data_path',
                    '-pdp',
                    type=str,
                    default='Data/Named_Entity_Recognition/test.txt',
                    help="path of test data")
parser.add_argument(
    '--save_path',
    '-sp',
    type=str,
    default='Data/Named_Entity_Recognition/result_biLstmCrf.tsv',
    help="output path")

args = parser.parse_args()
