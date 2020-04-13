# -*- coding: utf-8 -*-

import argparse

from metakbc.learn import learn
from metakbc.datasets import Dataset

datasets = ['Toy']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Relational learning contraction")

    parser.add_argument('--dataset', choices=datasets, help="Dataset")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs")
    parser.add_argument('--valid', default=3, type=int, help="Number of skipped epochs until evaluation")
    parser.add_argument('--rank', default=4, type=int, help="Rank of the tensor decomposition")

    args = parser.parse_args()

    learn(Dataset(args.dataset),
          args.epochs,
          args.valid,
          args.rank,
          args.batch_size)