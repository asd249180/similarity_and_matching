import sys
import argparse
import torch

import pickle
import os

if './' not in sys.path:
    sys.path.append('./')

from src.models import get_model
from src.dataset import get_n_classes_and_channels, get_datasets

from src.utils.eval_values import eval_net
from src.models.frankenstein_deprecated import FrankeinsteinNet, CelebaFrankenstein

from src.bin.eval_frank import get_frankenstein
from src.bin.compare import load_model
from src.comparators import BnActivation


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', help='Path to pickle')
    parser.add_argument('mode', help='Which matrix to use',
                        choices=['before', 'after', 'ps_inv'])
    parser.add_argument('-p', '--percentile', help='Percentile to use for runing batch norm', type=int, default=90)
    return parser.parse_args(args)


def load_pickle(pickle_path):
    return eval_frank(data_dict, mode, verbose)


def compare_activation(data_dict, mode, percentile=90, verbose=False):

    # Variables
    m2 = data_dict['params']['m2_path']
    layer_i = data_dict['params']['layer_i']
    data_name = data_dict['params']['dataset']

    # Dataset, train or test. Default is test
    dataset = get_datasets(data_name)['val']

    # Comparator class
    comparator = BnActivation(dataset, percentile=percentile)

    # Two models
    model1 = get_frankenstein(data_dict, mode)
    model2 = load_model(m2)

    # Comparison of two models on provided dataset
    similarities = comparator(model1, model2, layer_i=layer_i)

    # Eval results
    if verbose:
        print('MSE --> Below: {:3.4f} | Above: {:3.4f}'.format(*similarities))
    return similarities

def compare_from_pickle(pickle_path, mode, percentile=90, verbose=False):
    data_dict = pickle.load(open(pickle_path, "br"))
    return compare_activation(data_dict, mode,
                              percentile=percentile, verbose=verbose)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    compare_from_pickle(args.pickle, args.mode, args.percentile, verbose=True)


if __name__ == '__main__':
    main()
