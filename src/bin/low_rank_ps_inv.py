import sys
import argparse
import torch

import pickle
import os

import ntpath
import pandas as pd

import numpy as np

if './' not in sys.path:
    sys.path.append('./')

import matplotlib
import matplotlib.pyplot as plt

#import seaborn as sns

from glob import glob

from src.train import FrankModelTrainer
from src.models.frank.frankenstein import FrankeinsteinNet as FrankensteinNet
from src.utils.eval_values import frank_m2_similarity
from src.comparators.activation_comparator import ActivationComparator
from src.utils import config


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pkl', help='Path to Frank')
    # parser.add_argument('--ranks', nargs='+', default=[4,8,16])
    return parser.parse_args(args)


def run(conf):
    data_dict = pickle.load(open(conf.pkl, "br"))
    frank_model = FrankensteinNet.from_data_dict(data_dict, 'ps_inv')
    frank_model.to(config.device)

    dataset = data_dict['params']['dataset']
    comparator = ActivationComparator.from_frank_model(frank_model, 'front',
                                                       'end')
    trans_results = comparator(dataset, ['ls_sum', 'ls_orth'],
                               dataset_type='train')

    results = {}
    for rank, trans_m in trans_results['ls_sum'].items():
        frank_model.transform.load_trans_matrix(trans_m['w'], trans_m['b'])
        rel_acc = frank_m2_similarity(frank_model, dataset)['rel_acc']
        results[rank] = rel_acc

    trans_m = trans_results['ls_orth']
    frank_model.transform.load_trans_matrix(trans_m['w'], trans_m['b'])
    orth_rel_acc = frank_m2_similarity(frank_model, dataset)['rel_acc']

    ranks = [x for x in results]
    ranks.sort()

    df = pd.DataFrame({
        'm': 'ls_sum',
        'rel_acc': [results[x] for x in ranks],
        'low_rank': ranks,
        'front_layer': data_dict['params']['front_layer'],
    })
    df = df.append(
        {
            'm': 'ls_orth',
            'rel_acc': orth_rel_acc,
            'low_rank': 1000,
            'front_layer': data_dict['params']['front_layer']
        },
        ignore_index=True).reset_index(drop=True)

    df.loc[~df.low_rank.isin([4, 8, 16]), 'low_rank'] = 1000

    out_path = conf.pkl.replace('.pkl', '.csv')
    df.to_csv(out_path, index=False)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    conf = parse_args(args)
    run(conf)


if __name__ == '__main__':
    main()
