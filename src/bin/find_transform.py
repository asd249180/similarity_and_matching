import argparse
import logging
import os
import pickle
import sys
from datetime import datetime

import git

if './' not in sys.path:
    sys.path.append('./')

from src.dataset import get_datasets

from src.train import Trainer, ClassificationModelTrainer, FrankModelTrainer

from src.utils.eval_values import eval_net, frank_m2_similarity
from src.utils.to_pdf import to_pdf
from src.utils import config

from src.comparators import ActivationComparator

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('front_model', help='Path first model', type=str)
    parser.add_argument('end_model', help='Path to second model', type=str)
    parser.add_argument('front_layer',
                        help='Last layer of first model',
                        type=str)
    parser.add_argument('end_layer',
                        help='Last layer of second model',
                        type=str)
    parser.add_argument(
        'dataset',
        help='Name of dataset',
        choices=['fashion', 'mnist', 'cifar10', 'cifar100', 'celeba'])
    parser.add_argument('-o',
                        '--out-dir',
                        help='Dir to save matrix to',
                        default='results/')
    parser.add_argument('--run-name',
                        help='Name of the current run',
                        type=str,
                        default=None)
    parser.add_argument('--optimizer',
                        help='Optimizer',
                        choices=['adam', 'sgd'],
                        default='adam')
    parser.add_argument('--seed',
                        help='Seed of init and grad',
                        type=int,
                        default=0)
    parser.add_argument('-e',
                        '--epochs',
                        help='Number of epochs to train',
                        type=int,
                        default=30)
    parser.add_argument('-s',
                        '--save-frequency',
                        help='Save per iterations',
                        type=int,
                        default=10)
    parser.add_argument('-b',
                        '--batch-size',
                        help='Batch size',
                        type=int,
                        default=128)
    parser.add_argument('-lr',
                        '--lr',
                        help='Learning rate',
                        type=float,
                        default=1e-3)
    parser.add_argument('-wd',
                        '--weight-decay',
                        help='Weight decay',
                        type=float,
                        default=1e-4)
    parser.add_argument('--debug',
                        help='Debug True/False',
                        action='store_true')
    parser.add_argument('--flatten',
                        help='Flatten around transformation',
                        action='store_true')
    parser.add_argument('--l1', help='l1 value', type=float, default=0.)
    parser.add_argument('--cka-reg', help='Regularization on cka in stitching layer', type=float, default=0.)
    parser.add_argument('-i','--init',
                        help='Initial matrix',
                        choices=['random', 'perm', 'eye', 'ps_inv', 'ones-zeros'],
                        default='random')
    parser.add_argument('-m','--mask',
                        help='Mask applied on transformation',
                        choices=['identity', 'semi-match', 'abs-semi-match', 'random-permutation'],
                        default='identity')
    parser.add_argument('-r',
                        '--low-rank',
                        help='Rank',
                        type=int,
                        default=None)
    parser.add_argument(
        '--target-type',
        help='type of label to use for training',
        choices=['hard', 'soft_1', 'soft_2', 'soft_12', 'soft_1_plus_2'],
        default='hard')
    parser.add_argument('--temperature',
                        help='temperature of soft labels',
                        type=float,
                        default=1.)
    return parser.parse_args(args)


def get_hits_overlap(m1_hits, m2_hits):
    total_overlap = (m1_hits == m2_hits).sum()
    m1_right_m2_right = (m1_hits[m2_hits == m1_hits]).sum()
    m1_right_m2_wrong = (m1_hits[m2_hits != m1_hits]).sum()
    m1_wrong_m2_right = (m2_hits[m2_hits != m1_hits]).sum()
    m1_wrong_m2_wrong = total_overlap - m1_right_m2_right
    return {
        'rr': m1_right_m2_right,
        'rw': m1_right_m2_wrong,
        'wr': m1_wrong_m2_right,
        'ww': m1_wrong_m2_wrong
    }


def create_data(conf, trainer, trans_m_before_train):

    out_data = {}
    # Save trans learning curve
    out_data['trans_fit'] = trainer.model_trainer.stats

    # Save running parameters
    out_data['params'] = vars(conf)


    # Losses and accuracies
    frank_model = trainer.model_trainer.model
    multilabel = conf.dataset == 'celeba'
    m1_runner = ClassificationModelTrainer(frank_model.front_model,
                                           multilabel=multilabel)
    m2_runner = ClassificationModelTrainer(frank_model.end_model,
                                           multilabel=multilabel)
    m1_loss, m1_acc, m1_hits = eval_net(m1_runner, conf.dataset)
    m2_loss, m2_acc, m2_hits = eval_net(m2_runner, conf.dataset)
    trans_loss, trans_acc, frank_hits = eval_net(trainer.model_trainer,
                                                 conf.dataset)

    out_data['model_results'] = {
        'front': {
            'loss': m1_loss,
            'acc': m1_acc
        },
        'end': {
            'loss': m2_loss,
            'acc': m2_acc
        },
        'trans': {
            'loss': trans_loss,
            'acc': trans_acc
        }
    }

    # Extract weight and bias
    out_data['trans_m'] = {}
    out_data['trans_m']['before'] = trans_m_before_train
    out_data['trans_m']['after'] = _extract_trans_w_and_b(trainer.model)
    out_data['trans_m']['diff'] = _extract_trans_diff(out_data['trans_m'])

    # Save transformation matrices changing in time
    out_data['trans_history'] = trainer.model_trainer.transform_history

    if conf.low_rank is None:
        out_data['trans_m']['diff'] = _extract_trans_diff(out_data['trans_m'])

        if hasattr(frank_model.transform, "mask") and frank_model.transform.mask is not None:
            out_data['trans_m']['mask'] = {'w' : frank_model.transform.mask}

        # Comparisons
        frank_model = trainer.model_trainer.model
        comparator = ActivationComparator.from_frank_model(frank_model)
        comparator_frank = ActivationComparator.from_frank_model(frank_model, 'frank', 'end')
        group_at = 2500 if conf.dataset == 'celeba' else float('inf')
        batch_size = 50 if conf.dataset == 'celeba' else 2500
        logit_layer = 'logits' if conf.dataset == 'celeba' else 'fc'
        stop_at = 5 if conf.dataset == 'celeba' else float('inf')
        dataset_type = 'val'
        measures = comparator(conf.dataset, ['cka', 'ps_inv', 'l2'], batch_size, group_at, stop_at,
                              dataset_type)
        measures_frank = comparator_frank(conf.dataset, ['cka', 'l2'], batch_size, group_at, stop_at, dataset_type)

        # Frank logit measures
        logit_comparator_frank = ActivationComparator(frank_model,
                                                      frank_model.end_model,
                                                      logit_layer,logit_layer)
        logit_measures_frank = logit_comparator_frank(conf.dataset,
                                                     ['cka', 'l2'],
                                                     batch_size,
                                                     group_at,
                                                     stop_at,
                                                     dataset_type)

        # Cka
        out_data['cka'] = measures['cka']
        out_data['cka_frank'] = measures_frank['cka']

        # CCA
        # out_data['cca'] = measures['cca']
        # out_data['cca_frank'] = measures_frank['cca']

        # l2
        out_data['l2'] = measures['l2']
        out_data['l2_frank'] = measures_frank['l2']

        # Pseudo inverse
        w = measures['ps_inv']['w'][..., None, None]
        b = measures['ps_inv']['b']
        out_data['trans_m']['ps_inv'] = {'w' : w, 'b' : b}

        # # Difference to plot
        diff = out_data['trans_m']['ps_inv']['w'] - out_data['trans_m']['after']['w']
        out_data['trans_m']['ps_frank'] = {'w': diff}

        # psinv loss
        psinv_model_trainer = FrankModelTrainer.from_data_dict(out_data, 'ps_inv')
        ps_inv_loss, ps_inv_acc, ps_inv_hits = eval_net(psinv_model_trainer,
                                                        conf.dataset)

        ps_inv_model = psinv_model_trainer.model
        comparator_ps_inv = ActivationComparator.from_frank_model(ps_inv_model,
                                                                 'frank', 'end')
        measures_ps_inv = comparator_ps_inv(conf.dataset, ['cka', 'l2'],
                                            batch_size, group_at, stop_at, dataset_type)

        # PsInv logit mesasures
        logit_comparator_ps_inv = ActivationComparator(ps_inv_model,
                                                      ps_inv_model.end_model,
                                                      logit_layer,logit_layer)
        logit_measures_ps_inv = logit_comparator_ps_inv(conf.dataset,
                                                     ['cka', 'l2'],
                                                     batch_size,
                                                     group_at,
                                                     stop_at,
                                                     dataset_type)

        # Cka
        out_data['cka_ps_inv'] = measures_ps_inv['cka']


        # l2
        out_data['l2_ps_inv'] = measures_ps_inv['l2']

        out_data['model_results']['ps_inv'] = {
            'loss': ps_inv_loss,
            'acc': ps_inv_acc
        }

        # Logit cka & l2
        out_data['frank_m2_logit_cka'] = logit_measures_frank['cka']
        out_data['frank_m2_logit_l2'] = logit_measures_frank['l2']
        out_data['ps_inv_m2_logit_cka'] = logit_measures_ps_inv['cka']
        out_data['ps_inv_m2_logit_l2'] = logit_measures_ps_inv['l2']

        out_data['hits'] = {}
        out_data['hits']['m2_frank'] = get_hits_overlap(m2_hits, frank_hits)
        out_data['hits']['m2_ps_inv'] = get_hits_overlap(m2_hits, ps_inv_hits)
        out_data['hits']['frank_ps_inv'] = get_hits_overlap(
            frank_hits, ps_inv_hits)

        # Similarities
        frank_sim = frank_m2_similarity(frank_model, out_data['params']['dataset'], verbose=False)
        ps_inv_sim = frank_m2_similarity(ps_inv_model, out_data['params']['dataset'], verbose=False)
        out_data['m2_sim'] = {'ps_inv' : ps_inv_sim, 'after' : frank_sim}

    # Git info
    repo = git.Repo('./')
    out_data['git'] = {
        'branch': repo.active_branch.name,
        'commit': repo.head.commit.hexsha
    }

    out_data['runner_code'] = ' '.join(['python'] + sys.argv)

    return out_data


def _extract_trans_diff(data):
    before = data['before']
    after = data['after']
    w = after['w'] - before['w']
    b = after['b'] - before['b']
    return {'w': w, 'b': b}


def _extract_trans_w_and_b(model):
    w_b_dict = model.transform.get_param_dict()
    return w_b_dict


def save(conf, trainer, trans_m_before_train):

    data = create_data(conf, trainer, trans_m_before_train)

    # Create outgoing directory if not exist
    os.makedirs(os.path.join(conf.out_dir, 'matrix'), exist_ok=True)
    os.makedirs(os.path.join(conf.out_dir, 'pdf'), exist_ok=True)

    # Save pdf
    #filename = str(now)#.strftime("%Y-%m-%d--%H-%M-%S")
    now = datetime.now()
    filename = '{}-{}'.format(conf.front_layer, conf.end_layer) + str(now)
    if conf.low_rank is None:
        pdf_file = os.path.join(conf.out_dir, 'pdf', filename + '.pdf')
        to_pdf(data, pdf_file, now)

    # Save pickle
    matrix_file = os.path.join(conf.out_dir, 'matrix', filename + '.pkl')
    with open(matrix_file, 'wb') as f:
        pickle.dump(data, f)

def set_random_seeds(seed=0):
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

def run(conf):

    # Retreve save folder
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{conf.run_name}" if conf.run_name is not None else f"{now}"
    conf.out_dir = os.path.join(conf.out_dir, run_name)

    # Frankenstein model setup
    model_trainer = FrankModelTrainer.from_arg_config(conf)
    trans_m_before_train = _extract_trans_w_and_b(model_trainer.model)

    datasets = get_datasets(conf.dataset)

    # Train
    trainer = Trainer(
        datasets,
        model_trainer,
        batch_size=conf.batch_size,
        n_workers=4,
        drop_last=True,
        save_folder=conf.out_dir,
    )

    save_frequency = conf.save_frequency * conf.epochs
    trainer.train(conf.epochs, save_frequency, freeze_bn=True)

    # Save
    save(conf, trainer, trans_m_before_train)
    print('Done.')


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    set_random_seeds(args.seed)

    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
    logger = logging.getLogger("trans")
    logger.setLevel(logging.DEBUG if args.debug else logging.WARNING)

    run(args)


if __name__ == '__main__':
    main()
