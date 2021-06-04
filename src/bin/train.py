import argparse
import sys

if './' not in sys.path:
    sys.path.append('./')

from src.dataset import get_datasets
from src.train.trainer import Trainer
from src.train.classification_model_trainer import ClassificationModelTrainer


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('-m',
                        '--model',
                        help='Name of model',
                        type=str,
                        default='lenet')
    # Choices were: ['lenet', 'tiny10', 'inceptionv1', 'resnet20']

    parser.add_argument(
        '-d',
        '--dataset',
        help='Name of dataset',
        choices=['fashion', 'mnist', 'cifar10', 'cifar100', 'celeba'],
        default='mnist')
    parser.add_argument('-i',
                        '--init-noise',
                        help='Starting position of nn (seed)',
                        type=int,
                        default=0)
    parser.add_argument('-g',
                        '--gradient-noise',
                        help='Order of batches (seed)',
                        type=int,
                        default=0)
    parser.add_argument('-s',
                        '--save-frequency',
                        help='How often to save in number of iterations',
                        type=int,
                        default=10000)
    parser.add_argument('-o',
                        '--out-dir',
                        help='Folder to save networks to',
                        default='snapshots')
    parser.add_argument('-e',
                        '--epochs',
                        help='Number of epochs to train',
                        type=int,
                        default=300)
    parser.add_argument('-b',
                        '--batch-size',
                        help='Batch size',
                        type=int,
                        default=128)
    parser.add_argument('-lr',
                        '--lr',
                        help='Learning rate',
                        type=float,
                        default=1e-1)
    parser.add_argument('-wd',
                        '--weight-decay',
                        help='Weight decay',
                        type=float,
                        default=1e-4)
    parser.add_argument('-opt',
                        '--optimizer',
                        help='Optimizer',
                        choices=['adam', 'sgd'],
                        default='sgd')
    return parser.parse_args(args)


def run(conf):

    # Setup
    model_trainer = ClassificationModelTrainer.from_arg_config(conf)
    datasets = get_datasets(conf.dataset)

    # Print model on screen
    model_trainer.summarize(datasets['train'][0][0].shape)

    # Train
    trainer = Trainer(
        datasets,
        model_trainer,
        gradient_noise=conf.gradient_noise,
        batch_size=conf.batch_size,
        n_workers=4,
        drop_last=True,
        save_folder=conf.out_dir,
    )

    trainer.train(conf.epochs, conf.save_frequency)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    run(args)


if __name__ == '__main__':
    main()