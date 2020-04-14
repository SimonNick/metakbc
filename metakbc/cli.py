# -*- coding: utf-8 -*-

import argparse

from metakbc.learn import learn
from metakbc.datasets import Dataset

import wandb

datasets = ['Toy']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Relational learning contraction")

    parser.add_argument('--dataset', choices=datasets, help="Dataset")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size")
    parser.add_argument('--epochs_inner', default=10, type=int, help="Number of inner epochs")
    parser.add_argument('--epochs_outer', default=10, type=int, help="Number of outer epochs")
    parser.add_argument('--epochs_adv', default=10, type=int, help="Number of epochs for the adversary")
    parser.add_argument('--epochs_dis', default=10, type=int, help="Number of epochs for the discriminator")
    parser.add_argument('--valid', default=3, type=int, help="Number of skipped epochs until evaluation")
    parser.add_argument('--rank', default=4, type=int, help="Rank of the tensor decomposition")
    parser.add_argument('--logging', default=False, action='store_true', help="Whether to use wandb.com for logging")

    args = parser.parse_args()

    if args.logging:
        wandb.init(project="test1")
        wandb.config.epochs_inner = args.epochs_inner
        wandb.config.epochs_outer = args.epochs_outer
        wandb.config.epochs_adv = args.epochs_adv
        wandb.config.epochs_dis = args.epochs_dis
        wandb.config.valid = args.valid
        wandb.config.rank = args.rank
        wandb.config.batch_size = args.batch_size

    learn(Dataset(args.dataset),
          args.epochs_inner,
          args.epochs_outer,
          args.epochs_adv,
          args.epochs_dis,
          args.valid,
          args.rank,
          args.batch_size,
          args.logging)