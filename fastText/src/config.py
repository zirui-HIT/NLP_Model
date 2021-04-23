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
                    default='../model/fastText.pkl',
                    help="path to save model")
parser.add_argument('--max_length',
                    '-ml',
                    type=int,
                    default=None,
                    help="max length to be processed")
parser.add_argument('--epoch',
                    '-e',
                    type=int,
                    default=20,
                    help="epoch")
parser.add_argument('--batch_size',
                    '-bs',
                    type=int,
                    default=4,
                    help="batch size")
parser.add_argument('--learning_rate',
                    '-lr',
                    type=float,
                    default=1e-5,
                    help="learing rate")
parser.add_argument('--dropout',
                    '-d',
                    type=float,
                    default=0.3,
                    help="dropout")

# train mode argument
parser.add_argument('--train_data_path',
                    '-tdp',
                    type=str,
                    default='../../Data/Sentiment_Analysis_on_Movie_Reviews/train.tsv',
                    help="path of train data")
parser.add_argument('--valid_data_path',
                    'vdq',
                    type=str,
                    default='../../Data/Sentiment_Analysis_on_Movie_Reviews/valid.tsv',
                    help="path of valid data")

# test mode argument
parser.add_argument('--predict_data_path',
                    'pdp',
                    type=str,
                    default='../../Data/Sentiment_Analysis_on_Movie_Reviews/test.tsv',
                    help="path of test data")
parser.add_argument('--save_path',
                    '-sp',
                    type=str,
                    default='../../Data/Sentiment_Analysis_on_Movie_Reviews/result_fastText.txt',
                    help="output path")

args = parser.parse_args()
