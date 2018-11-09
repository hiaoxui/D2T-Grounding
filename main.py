import argparse
import nltk

from utils import measure
from config import cfg
from trainer import RWTrainer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', type=str, default='train', choices=['train', 'test'],
                        help='action, train or test. The default is train.')
    parser.add_argument('-k', type=int, default=2, help='Maximum length of word spans')
    parser.add_argument('-e', type=int, default=5, help='# of emission epochs')
    parser.add_argument('-t', type=int, default=5, help='# of transition epochs')
    parser.add_argument('-m', type=str, default='skip', choices=['without', 'with', 'skip'],
                        help='Null mode. See HMM.py for details.')
    parser.add_argument('-n', type=int, default=0, help='Prevent numerics-to-string alignment')
    parser.add_argument('-r', type=float, default=0.4, help='Null ratio for posterior regularization')
    parser.add_argument('-j', type=int, default=4, help='Number of workers (multiprocess)')
    parser.add_argument('-f', type=str, default=None,
                        help='Filter sentences. "L" for line-scores and "B" for box-scores')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_parser()
    cfg.k = args.k
    cfg.emit_epoch = args.e
    cfg.trans_epoch = args.t
    cfg.null_mode = args.m
    cfg.no_num = (args.n > 0)
    cfg.null_ratio = args.r
    cfg.jobs = args.j
    cfg.filter = args.f

    nltk.download('punkt')
    nltk.download('wordnet')
    # cfg.print_info()

    trainer = RWTrainer(cfg)
    todo = args.z

    if todo == 'train':
        trainer.train()
    elif todo == 'test':
        trainer.random_test()

    measure('END')
