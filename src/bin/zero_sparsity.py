import copy
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from src.models.frank.frankenstein import FrankeinsteinNet
from src.train import FrankModelTrainer
from src.utils.eval_values import eval_net


def evaluate(frank, str_dataset):
    frank_trainer = FrankModelTrainer(frank, target_type='soft_2')
    loss, acc, hits = eval_net(frank_trainer, str_dataset)
    return acc, loss


def cli_main(args=None):
    import argparse
    from src.utils import config

    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('--pickle_path', help='Path to pickle', default="results/find_trans_tiny_abn_l1/merged/matrix")
    parser.add_argument('--out_dir', type=str,
                        default="results/find_trans_tiny_abn_l1/merged/matrix_zero_sparsity/",
                        help='Output folder for result pickle')
    parser.add_argument('--threshold', type=float,
                        default=1e-4,
                        help='threshold of sparsity')

    args = parser.parse_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    done_files = os.listdir(args.out_dir)
    device = config.device
    for file_name in tqdm(os.listdir(args.pickle_path)):
        if file_name in done_files:
            continue
        if file_name.endswith(".pkl"):
            file_path = os.path.join(args.pickle_path, file_name)
            with open(file_path, 'rb') as f:
                x = pickle.load(f)
            if "original" in x.keys():
                data_dict = copy.deepcopy(x["original"])
                data_dict["trans_m"] = {"after": {"w": x["trans_m"]["w"].detach().cpu().clone().numpy(),
                                                  "b": x["trans_m"]["b"].detach().cpu().clone().numpy()
                                                  }
                                        }
            else:
                data_dict = copy.deepcopy(x)
            if data_dict["params"]["front_model"] != data_dict["params"]["end_model"] and data_dict["params"]["init"] == "ps_inv":
                w = data_dict['trans_m']['after']['w'].copy()
                b = data_dict['trans_m']['after']['b'].copy()
                w[np.abs(w) < args.threshold] = 0.0

                data_dict['trans_m']["sparse"] = {"w": w, "b": b, "threshold": args.threshold}
                frank_model = FrankeinsteinNet.from_data_dict(data_dict, "sparse")
                frank_model.to(device)
                acc, ce = evaluate(frank_model, data_dict['params']['dataset'])
                if "sparse" not in x.keys():
                    x["sparse"] = {}
                x["sparse"][args.threshold] = {"w": w,
                                               "b": b,
                                               "threshold": args.threshold,
                                               "acc": acc,
                                               "cross_entropy": ce}
                out_file_path = os.path.join(args.out_dir, file_name)
                with open(out_file_path, 'wb') as handle:
                    pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
                del x, data_dict
                torch.cuda.empty_cache()


if __name__ == '__main__':
    cli_main()
