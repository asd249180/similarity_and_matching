import sys
import os
import argparse
import torch
import pickle
import pandas as pd

if './' not in sys.path:
    sys.path.append('./')

from src.comparators.psinv_attach import PsInvAttach
from src.dataset import get_n_classes_and_channels, get_datasets
from src.models import get_model

from src.bin.eval_net import get_modelname_and_dataset

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', help='Path to pickle')
    parser.add_argument('mode', help='Which matrix to use', choices=['before', 'after', 'ps_inv'])
    return parser.parse_args(args)

def load_model(path):
    if '.pt' in path:
        model_name, data_name = get_modelname_and_dataset(path)
        n_classes, n_channels = get_n_classes_and_channels(data_name)
        model = get_model(model_name, n_classes, n_channels)
        model.load_state_dict(torch.load(path), strict=True)
    else:
        model = get_model('inceptionv1', 40, 3, celeba_name=path)
    model.eval()

    return model

def get_dist(data_dict, mode, verbose=False):
    
    # Dataset, train or test. Default is test
    data_name = data_dict['params']['dataset']
    dataset = get_datasets(data_name)['val']

    # Comparator class
    comparator = PsInvAttach(dataset)

    # Two models
    model1 = load_model(data_dict['params']['m1_path'])
    model2 = load_model(data_dict['params']['m2_path'])

    # Trans matrix
    trans_matrix= data_dict['trans_m'][mode]['w']
    trans_matrix = trans_matrix.reshape(trans_matrix.shape[:2])

    # Get activation distance
    layer_i = data_dict['params']['layer_i']
    results = comparator.get_distance(model1, model2,
                                            layer_i, trans_matrix)

    # Show results
    if verbose:
        results = pd.DataFrame(results)
        print(results.describe())

    return results
    
def get_dist_from_pickle(pickle_path, mode, verbose=False):
    data_dict = pickle.load(open(pickle_path, "br"))
    return get_dist(data_dict, mode, verbose)
    

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    get_dist_from_pickle(args.pickle, args.mode, verbose=True)



if __name__ == '__main__':
    main()