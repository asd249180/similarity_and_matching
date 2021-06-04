import sys
import argparse
import torch

import pickle
import os

import numpy as np

if './' not in sys.path:
    sys.path.append('./')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import matplotlib.animation as anim

from src.models.frank.frankenstein import FrankeinsteinNet as FrankensteinNet
from src.comparators.activation_comparator import ActivationComparator
from src.comparators.labeled_activation_comparator import LabeledActivationComparator
from src.utils.eval_values import frank_m2_similarity

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('frank', help='Path to Frank')
    parser.add_argument('-n', '--n-iters', type=int, default=10)
    return parser.parse_args(args)


def get_subpath(frank_path, data_dict):
    checkpoint_folder = os.path.join(frank_path, 'Frank')
    #folder, _ = os.path.split(data_dict['params']['end_model'])
    #subfolder = os.path.join(*(folder.split('/')[2:]))
    #os.path.join(checkpoint_folder, subfolder)
    return checkpoint_folder

def animate(i, ws):
    plt.imshow(ws[i], cmap='gray'),

def save_video(statistics, filename):
    key_value = sorted(zip(statistics['ids'], statistics['w']))
    _, ws = zip(*key_value)
    
    fig = plt.figure()
    ani = anim.FuncAnimation(fig, animate, fargs = (ws,), frames=len(ws), interval=500)
    ani.save(filename, writer='imagemagick', fps=30)


def save_l2(statistics, filename):
    titles = {'l2' : 'act_l2_distance', 'labeled_l2' : 'labeled_act_l2_distance',
              'frank_acc' : 'frank_acc', 'cka' : 'cka', 'cca' : 'cca (Yanai)', 'lr' : 'linear regression'}
    ylabels = {'l2' : 'avg_l2_dist', 'labeled_l2' : 'avg_l2_dist', 'frank_acc' : 'accuracy', 'cka' : 'cka_val', 'cca' : 'cca_val',
               'lr' : 'reg value'}

    #If labeled comparator ran, then frank_acc was computed.
    if len(statistics['names']['labeled']) > 0:
        statistics['names']['unlabeled'].append('frank_acc')

    for label_type in statistics['names']:
        for name in statistics['names'][label_type]:
            if len(statistics[label_type][name]) != 0:
                key_value = sorted(zip(statistics['ids'], statistics[label_type][name]))
                x, y = zip(*key_value)
                print(x, y)

                plt.figure()
                if label_type == 'labeled':
                    vals = {k : [yi[k] for yi in y] for k in y[0]}
                    for k in vals:
                        plt.plot(x, vals[k])
                        plt.legend(list(vals.keys()))
                else:
                    plt.plot(x,y)
                plt.plot(x, statistics['crossentropy'])
                plt.title(titles[name])
                plt.xlabel('iter')
                plt.ylabel(ylabels[name])
                plt.savefig(filename + name + '.png')
                plt.close()

                vals = {}
                import pandas as pd
                vals['iter'] = x
                vals['cka'] = y
                vals['crossentropy'] = statistics['crossentropy']
                vals['m2_acc'] = statistics['m2_acc']
                df = pd.DataFrame(vals)
                df.to_csv(filename + name + '.csv', index=False)
    #             import pickle
    #             with open(filename + name + '.pkl', 'wb') as output:
    #                 pickle.dump(vals, output, pickle.HIGHEST_PROTOCOL)
    # return

def update_statistics_at_checkpoint(state_dict, frank_model, dataset, statistics):
    frank_model.transform.load_state_dict(state_dict)

    comparator_type_dict = {'labeled' : LabeledActivationComparator, 'unlabeled' : ActivationComparator}
    dataset_type_dict = {'labeled' : 'train', 'unlabeled' : 'val'}
    
    comparator_dict = {}
    results_dict = {}
    entropy_dict = {}
    m2_acc_dict = {}
    for label_type in comparator_type_dict:
        stat_names = statistics['names'][label_type]
        if len(stat_names) > 0:
            comparator_dict[label_type] = comparator_type_dict[label_type].from_frank_model(frank_model, 'frank', 'end')
            dataset_type = dataset_type_dict[label_type]
            entropy_dict[label_type] = frank_m2_similarity(frank_model, dataset)
            statistics['crossentropy'].append(entropy_dict[label_type]['crossentropy'])
            statistics['m2_acc'].append(entropy_dict[label_type]['same_class_out'])
            results_dict[label_type] = (comparator_dict[label_type])(dataset, stat_names, dataset_type=dataset_type)

    #If labeled comparator ran, then frank_acc was computed.
    if len(statistics['names']['labeled']) > 0:
        statistics['unlabeled']['frank_acc'].append(results_dict['labeled']['frank_acc'])

    for label_type in statistics['names']:
        for stat_name in statistics['names'][label_type]:
            statistics[label_type][stat_name].append(results_dict[label_type][stat_name])
            

    new_w = np.squeeze(frank_model.transform.transform.weight.detach().cpu().numpy())
    #print(new_w)
    #print('before', statistics['w'])
    statistics['w'].append(np.copy(new_w))
    #print('after', statistics['w'])
    
    return

def eval_statistics(pkl_path, n_iters, statistics_names):
    #pkl_folder = os.path.join(frank_path, 'matrix')
    #pkl_path = os.path.join(pkl_folder, os.listdir(pkl_folder)[0])
    data_dict = pickle.load(open(pkl_path, "br"))
    # checkpoints_folder = get_subpath(frank_path, data_dict)
    checkpoints = data_dict['trans_history']
    chk_indices = list(data_dict['trans_history'].keys())[:n_iters]
    chk_indices.sort()
    n_iters = min(n_iters, len(chk_indices))
    chk_indices = chk_indices[:n_iters]
    
    frank_model = FrankensteinNet.from_data_dict(data_dict, 'after')

    statistics = { label_type : {k : [] for k in statistics_names[label_type]} for label_type in statistics_names}
    statistics['names'] = statistics_names
    statistics['ids'] = chk_indices
    statistics['crossentropy'] = []
    statistics['m2_acc'] = []
    statistics['w'] = []
    
    #Labeled also in secret computes frank_acc, but not as a separate parameter, so add that one separately.
    if len(statistics['names']['labeled']) > 0:
        statistics['unlabeled']['frank_acc'] = []

    #for checkpoint in sorted(os.listdir(checkpoints_folder)[:2]):#DEBUG PURPOSE
    for chk_index in chk_indices:
        update_statistics_at_checkpoint(checkpoints[chk_index], frank_model, data_dict['params']['dataset'], statistics)


    # for checkpoint in sorted(os.listdir(checkpoints_folder)):
    #     if checkpoint.endswith(".pt"):
    #         checkpoint_path = os.path.join(checkpoints_folder, checkpoint)
    #         statistics['ids'].append(int(os.path.splitext(checkpoint)[0]))
    #         update_statistics_at_checkpoint(checkpoint_path, frank_model, data_dict['params']['dataset'], statistics)

    #DEBUG PURPOSE
    print(statistics['ids'])
    for label_type in statistics['names']:
        for name in statistics['names'][label_type]:
            print(statistics[label_type][name])
    return statistics

def get_id(frank_path):
    return os.path.join('statistics', os.path.split(frank_path)[1].replace('.pkl', ''))


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    #statistics names: 'l2', 'cka', 'cca', 'lr'
    statistics_names = {'labeled' : [], 'unlabeled' : ['cka']}

    statistics = eval_statistics(args.frank, args.n_iters, statistics_names)

    #Debug info.
    print([(key, statistics[key]) for key in statistics if key != 'w'])
    filename = get_id(args.frank)

    os.makedirs('statistics', exist_ok=True)
    save_l2(statistics,filename)
    save_video(statistics, filename + '.gif')

if __name__ == '__main__':
    main()
