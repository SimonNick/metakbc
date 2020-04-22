# -*- coding: utf-8 -*-

import argparse

from metakbc.learn import learn
from metakbc.datasets import Dataset

import wandb

datasets = ['Toy', 'Toy2', 'Toy2_2', 'Toy2_3', 'nations', 'umls']
models = ['DistMult', 'ComplEx']
optimizers = ['SGD', 'Adagrad']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Relational learning contraction")

    parser.add_argument('--dataset',        default="Toy",      choices=datasets,       help="Dataset")
    parser.add_argument('--model',          default="ComplEx",  choices=models,         help="Model")
    parser.add_argument('--optimizer',      default="Adagrad",  choices=optimizers,     help="Optimizer")
    parser.add_argument('--meta_optimizer', default="Adagrad",  choices=optimizers,     help="Meta optimizer")
    parser.add_argument('--lr',             default=0.1,        type=float,             help="Learning rate of the optimizer")
    parser.add_argument('--meta_lr',        default=0.1,        type=float,             help="Learning rate of the meta optimizer")
    parser.add_argument('--epochs_outer',   default=100,        type=int,               help="Number of outer epochs")
    parser.add_argument('--batches_train',  default=5,          type=int,               help="How many batches of the training dataset should be used for training")
    parser.add_argument('--batches_valid',  default=5,          type=int,               help="How many batches of the validation dataset should be used for evaluation")
    parser.add_argument('--valid',          default=10,         type=int,               help="Number of skipped epochs until evaluation")
    parser.add_argument('--epochs_adv',     default=10,         type=int,               help="Number of epochs for the adversary")
    # parser.add_argument('--epochs_dis',     default=10,         type=int,               help="Number of epochs for the discriminator")
    parser.add_argument('--constraints',    default=10,         type=int,               help="Number of constraints")
    parser.add_argument('--rank',           default=100,        type=int,               help="Rank of the tensor decomposition")
    parser.add_argument('--batch_size',     default=32,         type=int,               help="Batch size for training and evaluation")
    parser.add_argument('--lam',            default=1.0,        type=float,             help="Weight of the violation loss")
    parser.add_argument('--logging',        default=False,      action='store_true',    help="Whether to use wandb.com for logging")

    args = parser.parse_args()

    if args.logging:
        wandb.init(project="test1")
        wandb.config.dataset = args.dataset
        wandb.config.model = args.model
        wandb.config.optimizer = args.optimizer
        wandb.config.meta_optimizer = args.meta_optimizer
        wandb.config.lr = args.lr
        wandb.config.meta_lr = args.meta_lr
        wandb.config.epochs_outer = args.epochs_outer
        wandb.config.batches_train = args.batches_train
        wandb.config.batches_valid = args.batches_valid
        wandb.config.valid = args.valid
        wandb.config.epochs_adv = args.epochs_adv
        # wandb.config.epochs_dis = args.epochs_dis
        wandb.config.constraints = args.constraints
        wandb.config.rank = args.rank
        wandb.config.batch_size = args.batch_size
        wandb.config.lam = args.lam

    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)), end=", ")
    print("")

    learn(Dataset(args.dataset),
          args.model,
          args.optimizer,
          args.meta_optimizer,
          args.lr,
          args.meta_lr,
          args.epochs_outer,
          args.batches_train,
          args.batches_valid,
          args.valid,
          args.epochs_adv,
        #   args.epochs_dis,
          args.constraints,
          args.rank,
          args.batch_size,
          args.lam,
          args.logging)