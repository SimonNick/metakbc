# -*- coding: utf-8 -*-

import argparse

from metakbc.learn import learn

import wandb
import datetime
import json

datasets = ['Toy_A=>B_16', 'Toy_A=>B_1024', 'Toy_A=>B,C=>D_1024', 'Toy_A,B=>C_16', 'Toy_A,B=>C_1024',  'Toy_A,B=>C,D,E=>F_1024', 'Toy_mixed', 'nations', 'umls', 'countries']
models = ['DistMult', 'ComplEx']
optimizers = ['SGD', 'Adagrad']
methods = ['offline', 'online']
rule_methods = ['attention', 'combinatorial']
adv_methods = ['embedding', 'entity']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Relational learning contraction")

    parser.add_argument('--dataset',        default="Toy",      choices=datasets,       help="Dataset")
    parser.add_argument('--model',          default="ComplEx",  choices=models,         help="Model")

    parser.add_argument('--method',         default='offline',  choices=methods,        help="Whether to do offline or online metalearning")
    parser.add_argument('--rule_method',    default='attention',choices=rule_methods,   help="The type of rule learning method")
    parser.add_argument('--adv_method',     default='embedding',choices=adv_methods,    help="The method for adversarial training")
    parser.add_argument('--lam',            default=0.5,        type=float,             help="Weight of the violation loss")
    parser.add_argument('--no_learn_lam',   default=False,      action='store_true',    help="Don't learn the lambda")

    parser.add_argument('--optimizer',      default="Adagrad",  choices=optimizers,     help="Optimizer")
    parser.add_argument('--meta_optimizer', default="Adagrad",  choices=optimizers,     help="Meta optimizer")
    parser.add_argument('--adv_optimizer',  default="Adagrad",  choices=optimizers,     help="Adversary optimizer")

    parser.add_argument('--lr',             default=0.1,        type=float,             help="Learning rate of the optimizer")
    parser.add_argument('--meta_lr',        default=0.1,        type=float,             help="Learning rate of the meta optimizer")
    parser.add_argument('--adv_lr',         default=0.05,       type=float,             help="Learning rate of the adversary optimizer")

    parser.add_argument('--epochs_outer',   default=100,        type=int,               help="Number of outer epochs")
    parser.add_argument('--epochs_inner',   default=5,          type=int,               help="Number of inner epochs for offline metalearning")
    parser.add_argument('--batches_train',  default=5,          type=int,               help="How many batches of the training dataset should be used for training for online metalearning")
    parser.add_argument('--epochs_adv',     default=100,        type=int,               help="Number of epochs for the adversary")
    parser.add_argument('--valid',          default=1,          type=int,               help="Number of skipped epochs until evaluation")

    parser.add_argument('--rank',           default=100,        type=int,               help="Rank of the tensor decomposition")
    parser.add_argument('--batch_size',     default=32,         type=int,               help="Batch size for training and evaluation")
    parser.add_argument('--reg_weight',     default=1e-3,       type=float,             help="Weight of the N3 regularizer")

    parser.add_argument('--seed',           default=42,         type=int,               help="Used seed")

    parser.add_argument('--print_clauses',  default=False,      action='store_true',    help="Whether to print the weights of the clauses during training")
    parser.add_argument('--logging',        default=False,      action='store_true',    help="Whether to use wandb.com for logging")
    parser.add_argument('--output_file',    default=None,       type=str,               help="Name of the JSON output file where all results and used parameters should be stored")

    args = parser.parse_args()

    if args.logging:
        name = datetime.datetime.now().strftime("%A %d.%m. %H:%M")
        wandb.init(project="meta-kbc", name=name)

        wandb.config.dataset = args.dataset
        wandb.config.model = args.model

        wandb.config.method = args.method
        wandb.config.rule_method = args.rule_method
        wandb.config.adv_method = args.adv_method
        wandb.config.lam = args.lam
        wandb.config.learn_lam = not args.no_learn_lam

        wandb.config.optimizer = args.optimizer
        wandb.config.meta_optimizer = args.meta_optimizer
        wandb.config.adv_optimizer = args.adv_optimizer

        wandb.config.lr = args.lr
        wandb.config.meta_lr = args.meta_lr
        wandb.config.adv_lr = args.adv_lr

        wandb.config.n_epochs_outer = args.epochs_outer
        if args.method == 'offline':
            wandb.config.n_epochs_inner = args.epochs_inner
        else:
            wandb.config.batches_train = args.batches_train
        wandb.config.epochs_adv = args.epochs_adv
        wandb.config.valid = args.valid

        wandb.config.rank = args.rank
        wandb.config.batch_size = args.batch_size
        wandb.config.reg_weight = args.reg_weight

        wandb.config.seed = args.seed
        
        wandb.print_clauses = args.print_clauses

    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)), end=", ")
    print("")

    metrics, losses = learn(args.dataset,
          args.model,
          #
          args.method,
          args.rule_method,
          args.adv_method,
          args.lam,
          not args.no_learn_lam,
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
          args.epochs_adv,
          args.valid,
          #
          args.rank,
          args.batch_size,
          args.reg_weight,
          #
          args.seed,
          #
          args.print_clauses,
          args.logging)

    if args.output_file is not None:
        print("\nStoring results in '{}'".format(args.output_file))
        args_dict = {arg: getattr(args, arg) for arg in vars(args)}
        total_dict = {'args': args_dict, 'metrics': metrics, 'losses': losses}

        with open(args.output_file, 'w') as json_file:
            json.dump(total_dict, json_file, indent='\t')