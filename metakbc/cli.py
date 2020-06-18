# -*- coding: utf-8 -*-

import argparse

from metakbc.learn import learn

import wandb

datasets = ['Toy', 'Toy2', 'Toy2_2', 'Toy2_3', 'nations', 'umls', 'Toy_A,B=>C']
models = ['DistMult', 'ComplEx']
optimizers = ['SGD', 'Adagrad']
methods = ['offline', 'online']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Relational learning contraction")

    parser.add_argument('--dataset',        default="Toy",      choices=datasets,       help="Dataset")
    parser.add_argument('--model',          default="ComplEx",  choices=models,         help="Model")

    parser.add_argument('--method',         default='offline',  choices=methods,        help="Whether to do offline or online metalearning")
    parser.add_argument('--lam',            default=1.0,        type=float,             help="Weight of the violation loss")

    parser.add_argument('--optimizer',      default="Adagrad",  choices=optimizers,     help="Optimizer")
    parser.add_argument('--meta_optimizer', default="Adagrad",  choices=optimizers,     help="Meta optimizer")
    parser.add_argument('--adv_optimizer',  default="Adagrad",  choices=optimizers,     help="Adversary optimizer")

    parser.add_argument('--lr',             default=0.1,        type=float,             help="Learning rate of the optimizer")
    parser.add_argument('--meta_lr',        default=0.1,        type=float,             help="Learning rate of the meta optimizer")
    parser.add_argument('--adv_lr',         default=0.5,        type=float,             help="Learning rate of the adversary optimizer")

    parser.add_argument('--epochs_outer',   default=100,        type=int,               help="Number of outer epochs")
    parser.add_argument('--epochs_inner',   default=100,        type=int,               help="Number of inner epochs for offline metalearning")
    parser.add_argument('--batches_train',  default=5,          type=int,               help="How many batches of the training dataset should be used for training for online metalearning")
    parser.add_argument('--batches_valid',  default=5,          type=int,               help="How many batches of the validation dataset should be used for evaluation for online metalearning")
    parser.add_argument('--epochs_adv',     default=100,        type=int,               help="Number of epochs for the adversary")
    parser.add_argument('--valid',          default=10,         type=int,               help="Number of skipped epochs until evaluation")

    parser.add_argument('--rank',           default=100,        type=int,               help="Rank of the tensor decomposition")
    parser.add_argument('--batch_size',     default=32,         type=int,               help="Batch size for training and evaluation")

    parser.add_argument('--print_clauses',  default=False,      action='store_true',    help="Whether to print the weights of the clauses during training")
    parser.add_argument('--logging',        default=False,      action='store_true',    help="Whether to use wandb.com for logging")

    args = parser.parse_args()

    if args.logging:
        wandb.init(project="test1")

        wandb.config.dataset = args.dataset
        wandb.config.model = args.model

        wandb.config.method = args.method
        wandb.config.lam = args.lam

        wandb.config.optimizer = args.optimizer
        wandb.config.meta_optimizer = args.meta_optimizer
        wandb.config.adv_optimizer = args.meta_optimizer

        wandb.config.lr = args.lr
        wandb.config.meta_lr = args.meta_lr
        wandb.config.adv_lr = args.meta_lr

        wandb.config.n_epochs_outer = args.epochs_outer
        if method == 'offline':
            wandb.config.n_epochs_inner = args.epochs_inner
        else:
            wandb.config.batches_train = args.batches_train
            wandb.config.batches_valid = args.batches_valid
        wandb.config.epochs_adv = args.epochs_adv
        wandb.config.valid = args.valid

        wandb.config.rank = args.rank
        wandb.config.batch_size = args.batch_size
        
        wandb.print_clauses = args.print_clauses

    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)), end=", ")
    print("")

    learn(args.dataset,
          args.model,
          #
          args.method,
          args.lam,
          #
          args.optimizer,
          args.meta_optimizer,
          args.adv_optimizer,
          #
          args.lr,
          args.meta_lr,
          args.adv_lr,
          #
          args.epochs_outer,
          args.epochs_inner,
          args.batches_train,
          args.batches_valid,
          args.epochs_adv,
          args.valid,
          #
          args.rank,
          args.batch_size,
          #
          args.print_clauses,
          args.logging)