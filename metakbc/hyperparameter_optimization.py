from ax import optimize
import json
import argparse
from metakbc.learn import learn

datasets = ['Toy_A=>A_10', 'Toy_A=>A_1024', 'Toy_A=>B_10', 'Toy_A=>B_1024', 'Toy_A=>B,C=>D_1024', 'Toy_A,B=>C_16', 'Toy_A,B=>C_1024',  'Toy_A,B=>C,D,E=>F_1024', 'Toy_mixed', 'nations', 'umls', 'countries']
methods = ['offline', 'online']
rule_methods = ['attention', 'combinatorial']
adv_methods = ['embedding', 'entity']

parser = argparse.ArgumentParser(description="Hyperparameter optimization using ax.dev")
parser.add_argument('--dataset',        default="Toy",      choices=datasets,       help="Dataset")
parser.add_argument('--method',         default='offline',  choices=methods,        help="Learning method")
parser.add_argument('--trials',         default=15,         type=int,               help="Total number of trials for HPO")
parser.add_argument('--rule_method',    default='attention',choices=rule_methods,   help="The type of rule learning method")
parser.add_argument('--adv_method',     default='embedding',choices=adv_methods,    help="The method for adversarial training")
parser.add_argument('--output_dir',     default=None,       type=str,               help="Name of the JSON output dir where the final results should be stored")
args = parser.parse_args()

epochs_outer = {
    'Toy_A=>B_10': [4, 6, 8, 10],
    'Toy_A=>B_1024': [10, 15, 20, 25],
    'Toy_A=>B,C=>D_1024': [10, 15, 20, 25],
    'Toy_A,B=>C_16': [10, 15, 20, 25],
    'Toy_A,B=>C_1024': [10, 15, 20, 25],
    'Toy_A,B=>C,D,E=>F_1024': [25, 50, 75, 100],
    'Toy_mixed': [25, 50, 75, 100]
    }

grid = [
    {
        "name": "dataset_str",
        "type": "fixed",
        "value": args.dataset,
    },
    {
        "name": "model_str",
        "type": "fixed",
        "value": "ComplEx",
    },
    {
        "name": "method",
        "type": "fixed",
        "value": args.method,
    },
    {
        "name": "rule_method",
        "type": "fixed",
        "value": args.rule_method,
    },
    {
        "name": "adv_method",
        "type": "fixed",
        "value": args.adv_method,
    },
    {
        "name": "lam",
        "type": "fixed",
        "value": 0.5,
    },
    {
        "name": "learn_lam",
        "type": "fixed",
        "value": True,
    },
    {
        "name": "optimizer_str",
        "type": "fixed",
        "value": "Adagrad",
    },
    {
        "name": "meta_optimizer_str",
        "type": "fixed",
        "value": "Adagrad",
    },
    {
        "name": "adv_optimizer_str",
        "type": "fixed",
        "value": "Adagrad",
    },
    {
        "name": "lr",
        "type": "choice",
        "values": [1e-1, 1e-2],
    },
    {
        "name": "meta_lr",
        "type": "choice",
        "values": [1e-0, 5e-1, 1e-1, 5e-2, 1e-2],
    },
    {
        "name": "adv_lr",
        "type": "fixed",
        "value": 0.05,
    },
    {
        "name": "n_epochs_adv",
        "type": "choice",
        "values": [10, 20, 50, 100],
    },
    {
        "name": "rank",
        "type": "choice",
        "values": [50, 100, 200],
    },
    {
        "name": "batch_size",
        "type": "choice",
        "values": [100, 200, 1000],
    },
    {
        "name": "reg_weight",
        "type": "choice",
        "values": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
    }
]

grid_offline = [
    {
        "name": "n_epochs_outer",
        "type": "choice",
        "values": epochs_outer[args.dataset],
    },
    {
        "name": "n_epochs_inner",
        "type": "choice",
        "values": [5, 10, 15, 20],
    },
]

grid_online = [
    {
        "name": "n_epochs_outer",
        "type": "choice",
        "values": [e * 10 for e in epochs_outer[args.dataset]],
    },
    {
        "name": "n_batches_train",
        "type": "choice",
        "values": [1, 2, 5, 10],
    },
    {
        "name": "n_batches_valid",
        "type": "choice",
        "values": [1, 2, 5, 10],
    },
]

def _evaluation_func(p, **kwargs):
    _, loss_total = learn(**p, **kwargs, n_valid=0, print_clauses=False, logging=False)
    return loss_total['valid']

if args.method == "offline":
    evaluation_func = lambda p: _evaluation_func(p, n_batches_train=0, n_batches_valid=0)
    params = grid + grid_offline
else:
    evaluation_func = lambda p: _evaluation_func(p, n_epochs_inner=0)
    params = grid + grid_online


best_parameters, best_values, experiment, model = optimize(
    parameters=params,
    evaluation_function=evaluation_func,
    minimize=True,
    total_trials=args.trials
)

with open('{}/best_params_{}_{}_{}_{}.json'.format(args.output_dir, args.dataset, args.method, args.rule_method, args.adv_method), 'w') as json_file:
    json.dump(dict(sorted(best_parameters.items())), json_file, indent="\t")