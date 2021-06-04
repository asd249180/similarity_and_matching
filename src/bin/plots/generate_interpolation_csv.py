import pandas as pd
from matplotlib import pyplot as plt
import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import pickle
from dotmap import DotMap
from IPython.display import display
import sys
import argparse

if "./" not in sys.path:
    sys.path.append('./')

from src.train import FrankModelTrainer
from src.utils.eval_values import eval_net, frank_m2_similarity

orig_df = None

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('from_state', type=str, help='From state', choices=['random_frank', 'ps_inv_init', 'ps_inv_frank'])
    parser.add_argument('to_state', type=str, help='To state', choices=['random_frank', 'ps_inv_init', 'ps_inv_frank'])
    parser.add_argument('layer', type=str, help='Layer to check interpolation')
    parser.add_argument('-c','--csv', help="Path to summary csv", default='results/official/tiny_bn/summary-extra-random.csv')
    parser.add_argument('-o','--out-dir', help='Folder to save csv', default='results/interpolation')
    parser.add_argument('-n','--n-variants', help='Number of variants to try out', type=int, default=10)
    parser.add_argument('-s','--n-split', help='Number of splits', type=int, default=10)
    return parser.parse_args(args)

def get_data_dict(layer, init, i=0, can_be_second_random=False):
    df = orig_df.copy()
    df['front_type'] = df.front_model.apply(lambda x:x.split('/')[-2])
    df['end_type'] = df.end_model.apply(lambda x:x.split('/')[-2])
    filters = [
        df.init==init,
        df.front_layer==layer,
        df.front_model.str.contains(f"in{i}")
    ]
    df = df[np.logical_and.reduce(filters)]
    
    if init=='random' and can_be_second_random:
        pickle_path = df.iloc[1]['pickle']
    else:
        pickle_path = df.iloc[0]['pickle']
        
    with open(pickle_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def get_interpolation(data_dicts, modes, ratio):
    
    w1 = data_dicts[0]['trans_m'][modes[0]]['w']
    b1 = data_dicts[0]['trans_m'][modes[0]]['b']
    w2 = data_dicts[1]['trans_m'][modes[1]]['w']
    b2 = data_dicts[1]['trans_m'][modes[1]]['b']
    
    w = torch.Tensor((1-ratio) * w1 + ratio * w2)
    b = torch.Tensor((1-ratio) * b1 + ratio * b2)
    
    return w, b

def interpolate(states, layer, variant=0, split=10):
    
    modes, inits = [], []
    for state in states:
        if state=='random_frank':
            modes.append('after')
            inits.append('random')
        elif state=='ps_inv_init':
            modes.append('ps_inv')
            inits.append('ps_inv')
        elif state=='ps_inv_frank':
            modes.append('after')
            inits.append('ps_inv')
        else:
            raise ValueError(f'Unknown state {state}')
    
    data_dict1 = get_data_dict(layer, inits[0], variant, can_be_second_random=False)
    data_dict2 = get_data_dict(layer, inits[1], variant, can_be_second_random=True)
    data_name = data_dict1['params']['dataset']
    
    frank_trainer = FrankModelTrainer.from_data_dict(data_dict1, 'after')
    results = []
    
    for i in range(split+1):
        ratio = i/split
        w,b = get_interpolation([data_dict1, data_dict2], modes, ratio)
        frank_trainer.model.transform.load_trans_matrix(w,b)
        loss, acc, _ = eval_net(frank_trainer, data_name)
        metrics = frank_m2_similarity(frank_trainer.model, data_name)
        metrics['hard_loss'] = loss
        metrics['hard_acc'] = acc
        results.append(metrics)
        
    return results

def run(conf):
    
    global orig_df
    orig_df = pd.read_csv(conf.csv)

    for i in range(conf.n_variants):
        results = interpolate([conf.from_state, conf.to_state], conf.layer,
                              variant=i, split=conf.n_split)
        results = pd.DataFrame(results)
        filename = f"{conf.from_state}-{conf.to_state}-{conf.layer}-{i}.csv"
        os.makedirs(conf.out_dir, exist_ok=True)
        filepath = os.path.join(conf.out_dir, filename)
        results.to_csv(filepath, index=False)
        print(f"{filepath} saved.")

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    run(args)

if __name__ == '__main__':
    main()
