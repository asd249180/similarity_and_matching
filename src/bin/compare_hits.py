import sys
import os
import torch
import logging
import argparse

if './' not in sys.path:
    sys.path.append('./')

from src.bin.eval_frank import eval_frank_from_pickle
from src.utils.eval_values import eval_net

from src.models import get_model
from src.dataset import get_n_classes_and_channels



def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('m', help='Path to model')
    parser.add_argument('f', help='Path to Frankenstein')
    parser.add_argument('fm', help='Frankenstein mode.', choices = {'after', 'ps_inv'})
    parser.add_argument('--debug', help='Debug True/False', action='store_true')
    return parser.parse_args(args)


def get_modelname_and_dataset(model_path):
    if not model_path.endswith('.pt'):
        return 'inceptionv1', 'celeba'
    else:
        path_split = model_path.split(os.sep)
        model_name = path_split[-4]
        data_name = path_split[-3]
        return model_name, data_name
    

def load_data(path):
    if '.pt' in path:
        model_name, data_name = get_modelname_and_dataset(path)
        n_classes, n_channels = get_n_classes_and_channels(data_name)
        model = get_model(model_name, n_classes, n_channels)
        model.load_state_dict(torch.load(path), strict=True)
        multilabel = False
    else:
        model = get_model('inceptionv1', 40, 3, celeba_name=path)
        multilabel = True
        data_name = 'celeba'
    model.eval()

    return model, data_name, multilabel


def run(model_path, frank_path, frank_mode, verbose=False):
    model, data_name, is_multilabel = load_data(model_path)
    mean_loss, mean_acc, model_hits = eval_net(model, data_name, is_multilabel, verbose=verbose)

    frank_loss, frank_acc, frank_hits = eval_frank_from_pickle(frank_path, frank_mode, verbose=verbose)

    total_overlap = (frank_hits == model_hits).sum()
    correct_overlap = (frank_hits[model_hits == frank_hits]).sum()
    frank_right_model_wrong = (frank_hits[model_hits != frank_hits]).sum()
    frank_wrong_model_right = (model_hits[model_hits != frank_hits]).sum()
    
    if verbose:
        print('Model loss: {:.3f} | Model accuracy: {:2.2f}%'.format(mean_loss, mean_acc*100.))
        print('Frank loss: {:.3f} | Frank accuracy: {:2.2f}%'.format(frank_loss, frank_acc*100.))
        print('Model right, Frank right: {}'.format(correct_overlap))
        print('Model right, Frank wrong: {}'.format(frank_wrong_model_right))
        print('Model wrong, Frank right: {}'.format(frank_right_model_wrong))
        print('Model wrong, Frank wrong: {}'.format(total_overlap - correct_overlap))
        #print('Number of same correct guesses: {} out of {} correct model guesses.'.format(correct_overlap, model_hits.sum()))
        #print('Number of same guesses: {} out of {} total model guesses.'.format(total_overlap, len(model_hits)))        
    return correct_overlap, total_overlap


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)

    print(torch.cuda.is_available())
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG if args.debug else logging.WARNING)

    run(args.m, args.f, args.fm, verbose=True)


if __name__ == '__main__':
    main()
