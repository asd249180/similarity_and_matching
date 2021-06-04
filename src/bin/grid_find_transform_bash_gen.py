import itertools
import glob
import os
from dotmap import DotMap
import subprocess
import sys
from tqdm import tqdm
import threading
import argparse
import typing
import time

def get_all_tasks():
    def str_form(experiment):
        e = DotMap(experiment)
        return f"python src/bin/find_transform.py {e.front_model} {e.end_model} " + \
            f"{e.layers} {e.layers} {e.dataset} --run-name {e.run_name} " + \
            f"-wd {e.weight_decay} --l1 {e.l1} -e {e.epochs} --init {e.init} " + \
            f"--seed {e.seed} --target_type {e.target_type} " + \
            f"--mask {e.mask}"  

    all_models = glob.glob('archive/Tiny10/CIFAR10/*/110000.pt')
    options = {
        'front_model' : all_models,
        'end_model' : all_models,
        'layers': [f"conv{i+1}" for i in range(8)],
        'dataset' : ['cifar10'],
        'run_name' : ['corr_collection'],
        'weight_decay' : [0.],
        'l1' : [0.],
        'mask' : ['semi-match'],
        'epochs' : [30],
        'target_type' : ['soft_2'],
        'init' : ['random', 'random', 'random', 'ps_inv'], 
    }
    keys, values = zip(*options.items())
    tasks = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for i, task in enumerate(tasks):
        task['seed'] = i
    tasks = [str_form(t) for t  in tasks]
    return set(tasks)

tasks = get_all_tasks()

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('gpu_id_per_task', nargs="+", help='E.g. 0 0 1 1')
    parser.add_argument('-o', '--out-file', default='results/corr_collection/tasks_done.txt')
    return parser.parse_args(args)

def remove_tasks_done(tasks, out_file):
    if not os.path.isfile(out_file):
        return tasks

    with open(out_file, 'r') as f:
        tasks_done = set(f.read().splitlines())


    return tasks.difference(tasks_done)

def process_commands(gpu, out_file, pbar):

    while len(tasks):
        task = tasks.pop()
        orig_task = task
        task = f"CUDA_VISIBLE_DEVICES={gpu} {task}"
        process = subprocess.Popen(task, shell=True, stdout=subprocess.DEVNULL)
        process.wait()
        if process.returncode == 0:
            with open(out_file, 'a') as f:
                f.write(f"{orig_task}\n")
        else:
            print(f"Warning: Task ran into error: {task}")
        pbar.update(1)

def main(args=None):
    global tasks

    if args is None:
        args = sys.argv[1:]
    conf = parse_args(args)
    tasks = remove_tasks_done(tasks, conf.out_file)

    if not len(tasks):
        print('Every task is done already.')
        return
    
    pbar = tqdm(total=len(tasks))
    threads = []
    for gpu_id in conf.gpu_id_per_task:
        t = threading.Thread(target=process_commands,
                             args=(gpu_id, conf.out_file, pbar))
        t.start()
        threads.append(t)

    # for t in threads:
    #     t.join()

    # pbar.close()

    

if __name__ == '__main__':
    main()
