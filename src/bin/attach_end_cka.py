import pickle
import os
import sys
from tqdm import tqdm
import threading
import glob
from dotmap import DotMap

if './' not in sys.path:
    sys.path.append('./')

from src.utils import config
from src.bin.compare_frank import run_comparison_by_conf
from src.models import FrankeinsteinNet
from src.utils.eval_values import frank_m2_similarity
from src.utils.to_pdf import to_pdf

src_root = 'results/30epoch_all/matrix'
dst_root = 'results/30epoch_attach_similarity/matrix'
os.makedirs(dst_root, exist_ok=True)
os.makedirs(dst_root.replace('matrix', 'pdf'), exist_ok=True)
pickles = set(glob.glob(os.path.join(src_root, '*.pkl')))
n_jobs = 1

def process_pickle(p):

    filename = os.path.split(p)[-1]
    dst_file = os.path.join(dst_root, filename)
    if os.path.isfile(dst_file):
        return

    with open(p, 'rb') as infile:
        data_dict = pickle.load(infile)

    attachement = {'m2_sim' : {}}
    for mode in ['ps_inv', 'after']:
        frank_net = FrankeinsteinNet.from_data_dict(data_dict, mode=mode).to(config.device)
        results = frank_m2_similarity(frank_net, data_dict['params']['dataset'], verbose=False)
        attachement['m2_sim'][mode] = results
    data_dict.update(attachement)

    # Save pdf
    pdf_file = dst_file.replace('.pkl', '.pdf').replace('matrix', 'pdf')
    to_pdf(data_dict, pdf_file, data_dict['time'])

    with open(dst_file, 'wb') as f:
        pickle.dump(data_dict, f)

def process_pickles(pbar):
    while len(pickles):
        p = pickles.pop()
        process_pickle(p)

def main():
    pbar = tqdm(total=len(pickles))
    threads = []
    for _ in range(n_jobs):
        t = threading.Thread(target=process_pickles,
                             args=(pbar, ))
        t.start()
        threads.append(t)

if __name__ == '__main__':
    main()