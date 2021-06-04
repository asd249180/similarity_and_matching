import numpy as np
import pickle
import sys


def get_diag_info(w, name = '', verbose=False):
    w = w.reshape(w.shape[:2])
    mask = np.eye(w.shape[0], dtype=bool)
    diag = w[mask]
    off_diag = w[~mask]

    if name != '':
        name = name + '_'

    diag_info = {
        name + 'diag' : analyse(diag, verbose=verbose, title='Diag'),
        name + 'off_diag' : analyse(off_diag, verbose=verbose, title='Off-Diag'),
        name + 'matrix_norm_ratio' : np.linalg.norm(w) / np.linalg.norm(np.eye(w.shape[0]))
    }

    return diag_info

def analyse(m, verbose=False, title=None):

    diag_info = {
        'mean' : m.mean().round(4),
        'max' : m.max().round(4),
        'min' : m.min().round(4),
        'median' : np.median(m).round(4),
        'std' : np.std(m).round(4),
        'fr_norm' : np.linalg.norm(m).round(4)
    }

    if verbose:
        title = '' if title is None else title
        print(title)
        print('Mean    : {:2.3f}'.format(diag_info['mean']))
        print('Max     : {:2.3f}'.format(diag_info['max']))
        print('Min     : {:2.3f}'.format(diag_info['min']))
        print('Median  : {:2.3f}'.format(diag_info['median']))
        print('Std     : {:2.3f}'.format(diag_info['std']))
        print('Fr.norm : {:2.3f}'.format(diag_info['fr_norm']))

    return diag_info


if __name__ == '__main__':
    # Read in pickle matrix
    filename = sys.argv[1]
    p = pickle.load(open(filename, "br"))

    # weight
    w = p['trans_m']['w']
    get_diag_info(w, verbose=True)
