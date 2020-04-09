#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim, Tensor

import torch.nn.functional as F

# import tensorflow as tf
# import tensorboard as tb

from torch.utils.tensorboard import SummaryWriter

import higher

from metakbc.training.data import Data, X_to_dicts
from metakbc.training.batcher import Batcher

from metakbc.models import BaseModel, DistMult, ComplEx

from metakbc.graph.base import get_graph_features

from metakbc.regularizers import F2, L1, N3

from sklearn.preprocessing import normalize

from metakbc.regularizers import ConstantAdaptiveRegularizer
from metakbc.regularizers import LinearAdaptiveRegularizer
from metakbc.regularizers import GatedLinearAdaptiveRegularizer

from metakbc.evaluation import evaluate

from typing import List, Tuple, Optional, Union, Dict

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

torch.autograd.set_detect_anomaly(True)

torch.set_num_threads(multiprocessing.cpu_count())
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tH@1 {metrics["hits@1"]:.6f}\tH@3 {metrics["hits@3"]:.6f}\t' \
        f'H@5 {metrics["hits@5"]:.6f}\tH@10 {metrics["hits@10"]:.6f}'


def get_loss(X: Tensor,
             entity_embeddings: Union[Tensor, nn.Embedding],
             predicate_embeddings: Union[Tensor, nn.Embedding],
             model: BaseModel,
             loss_function: nn.CrossEntropyLoss,
             sp_to_o: Optional[Dict[Tuple[int, int], List[int]]] = None,
             po_to_s: Optional[Dict[Tuple[int, int], List[int]]] = None) -> Tuple[Tensor, List[Tensor]]:

    def lookup(_X, _emb):
        return _emb(_X) if isinstance(_emb, nn.Embedding) else F.embedding(_X, _emb)

    xs_batch_emb = lookup(X[:, 0], entity_embeddings)
    xp_batch_emb = lookup(X[:, 1], predicate_embeddings)
    xo_batch_emb = lookup(X[:, 2], entity_embeddings)
    emb = entity_embeddings.weight if isinstance(entity_embeddings, nn.Embedding) else entity_embeddings

    sp_scores, po_scores = model.forward(xp_batch_emb, xs_batch_emb, xo_batch_emb, entity_embeddings=emb)
    factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

    if sp_to_o is not None:
        sp_mask = torch.zeros_like(sp_scores)
        for i in range(X.shape[0]):
            key = (X[i, 0].item(), X[i, 1].item())
            if key in sp_to_o:
                indices = [j for j in sp_to_o[key] if j != X[i, 2].item()]
                sp_mask[i, indices] = -np.inf
        sp_scores = sp_scores + sp_mask

    if po_to_s is not None:
        po_mask = torch.zeros_like(po_scores)
        for i in range(X.shape[0]):
            key = (X[i, 1].item(), X[i, 2].item())
            if key in po_to_s:
                indices = [j for j in po_to_s[key] if j != X[i, 0].item()]
                po_mask[i, indices] = -np.inf
        po_scores = po_scores + po_mask

    s_loss = loss_function(sp_scores, X[:, 2])
    o_loss = loss_function(po_scores, X[:, 0])

    loss = s_loss + o_loss

    return loss, factors


def main(argv):
    parser = argparse.ArgumentParser('Meta-KBC', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', action='store', required=True, type=str)

    parser.add_argument('--dev', action='store', type=str, default=None)
    parser.add_argument('--test', action='store', type=str, default=None)

    parser.add_argument('--test-i', action='store', type=str, default=None)
    parser.add_argument('--test-ii', action='store', type=str, default=None)

    # model params
    parser.add_argument('--model', '-m', action='store', type=str, default='distmult',
                        choices=['distmult', 'complex'])

    parser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    parser.add_argument('--batch-size', '-b', action='store', type=int, default=100)
    parser.add_argument('--eval-batch-size', '-B', action='store', type=int, default=None)

    # training params
    parser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    parser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.1)

    parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad',
                        choices=['adagrad', 'adam', 'sgd'])

    parser.add_argument('--regularizer-type', '--regularizer', '-R', action='store', type=str, default='N3',
                        choices=['L1', 'F2', 'N3'])
    parser.add_argument('--regularizer-weight-type', '-W', action='store', type=str, default='none',
                        choices=['none', 'graph', 'latent'])

    parser.add_argument('--lookahead-steps', '--LS', '-S', action='store', type=int, default=1)
    parser.add_argument('--lookahead-learning-rate', '--LL', action='store', type=float, default=0.01)
    parser.add_argument('--lookahead-sample-size', '--LSS', action='store', type=float, default=None)
    parser.add_argument('--lookahead-masked', '--LM', action='store_true', default=False)

    parser.add_argument('--seed', action='store', type=int, default=0)

    parser.add_argument('--validate-every', '-V', action='store', type=int, default=None)

    parser.add_argument('--input-type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])

    parser.add_argument('--load', action='store', type=str, default=None)
    parser.add_argument('--save', action='store', type=str, default=None)

    parser.add_argument('--quiet', '-q', action='store_true', default=False)
    parser.add_argument('--tensorboard', action='store', type=str, default=None)

    args = parser.parse_args(argv)

    import pprint
    logger.info(pprint.pformat(vars(args)))

    train_path = args.train

    dev_path = args.dev
    test_path = args.test

    test_i_path = args.test_i
    test_ii_path = args.test_ii

    model_name = args.model
    optimizer_name = args.optimizer

    embedding_size = args.embedding_size

    batch_size = args.batch_size
    eval_batch_size = batch_size if args.eval_batch_size is None else args.eval_batch_size

    nb_epochs = args.epochs
    seed = args.seed

    learning_rate = args.learning_rate

    regularizer_type = args.regularizer_type
    regularizer_weight_type = args.regularizer_weight_type

    nb_lookahead_steps = args.lookahead_steps
    lookahead_lr = args.lookahead_learning_rate if args.lookahead_learning_rate is not None else learning_rate
    lookahead_sample_size = args.lookahead_sample_size
    is_lookahead_masked = args.lookahead_masked

    validate_every = args.validate_every
    input_type = args.input_type

    load_path = args.load
    save_path = args.save

    is_quiet = args.quiet
    tensorboard_path = args.tensorboard

    # set the seeds
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    writer = SummaryWriter(tensorboard_path) if tensorboard_path is not None else None

    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    graph_features = get_graph_features(data.train_triples, data.entity_to_idx, data.predicate_to_idx)
    graph_features = normalize(graph_features, axis=0, norm='max')
    graph_features_t = torch.tensor(graph_features, dtype=torch.float32, requires_grad=False)

    nb_graph_features = graph_features.shape[1]

    triples_name_pairs = [
        (data.dev_triples, 'dev'),
        (data.test_triples, 'test'),
        (data.test_i_triples, 'test-I'),
        (data.test_ii_triples, 'test-II'),
    ]

    rank = embedding_size * 2 if model_name in {'complex'} else embedding_size
    init_size = 1e-3

    entity_embeddings = nn.Embedding(data.nb_entities, rank, sparse=False)
    predicate_embeddings = nn.Embedding(data.nb_predicates, rank, sparse=False)

    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    param_module = nn.ModuleDict({'entities': entity_embeddings, 'predicates': predicate_embeddings}).to(device)
    if load_path is not None:
        param_module.load_state_dict(torch.load(load_path))

    regularizer_factory = {
        'L1': lambda *args, **kwargs: L1(*args, **kwargs),
        'F2': lambda *args, **kwargs: F2(*args, **kwargs),
        'N3': lambda *args, **kwargs: N3(*args, **kwargs),
    }

    regularizer = regularizer_factory[regularizer_type]()

    regularizer_rel_weight_model = ConstantAdaptiveRegularizer(regularizer=regularizer).to(device)

    if regularizer_weight_type in {'none'}:
        regularizer_ent_weight_model = ConstantAdaptiveRegularizer(regularizer=regularizer)
    else:
        nb_features = nb_graph_features if regularizer_weight_type in {'graph'} else embedding_size
        regularizer_ent_weight_model = GatedLinearAdaptiveRegularizer(regularizer=regularizer, nb_features=nb_features)

    regularizer_ent_weight_model = regularizer_ent_weight_model.to(device)

    parameter_lst = nn.ParameterList([entity_embeddings.weight, predicate_embeddings.weight]).to(device)

    hyperparameter_lst = [p for p in regularizer_rel_weight_model.parameters()] + \
                         [p for p in regularizer_ent_weight_model.parameters()]

    hyperparameter_lst = nn.ParameterList(hyperparameter_lst).to(device)

    model_factory = {
        'distmult': lambda: DistMult(),
        'complex': lambda: ComplEx()
    }

    model = model_factory[model_name]().to(device)

    logger.info('Model state:')
    for param_tensor in param_module.state_dict():
        tensor = param_module.state_dict()[param_tensor]
        logger.info(f'\t{param_tensor}\t{tensor.size()}\t{tensor.device}')

    logger.info('Hyperparams')
    for hyperparam_tensor in hyperparameter_lst:
        logger.info(f'\t{hyperparam_tensor.shape}\t{hyperparam_tensor.device}')

    optimizer_factory = {
        'adagrad': lambda *args, **kwargs: optim.Adagrad(*args, **kwargs),
        'adam': lambda *args, **kwargs: optim.Adam(*args, **kwargs),
        'sgd': lambda *args, **kwargs: optim.SGD(*args, **kwargs)
    }

    optimizer = optimizer_factory[optimizer_name](parameter_lst, lr=learning_rate, initial_accumulator_value=1e-45)

    hyper_optimizer = optimizer_factory[optimizer_name](hyperparameter_lst, lr=lookahead_lr)

    logger.info(optimizer)
    logger.info(hyper_optimizer)

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                           test_triples=triples, all_triples=data.all_triples,
                           entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                           model=model, batch_size=eval_batch_size, device=device)
        if writer is not None:
            writer.add_scalars(f'Ranking/{name}', {n.upper().replace("@", "_"): v for n, v in metrics.items()}, 0)

    for epoch_no in range(1, nb_epochs + 1):
        batcher = Batcher(data.nb_examples, batch_size, 1, random_state)
        nb_batches = len(batcher.batches)

        epoch_loss_values = []
        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
            step_no = ((epoch_no - 1) * nb_batches) + batch_no

            diff_opt = higher.get_diff_optim(optimizer, parameter_lst, device=device, track_higher_grads=True)

            e_tensor_lh = entity_embeddings.weight
            p_tensor_lh = predicate_embeddings.weight

            for i in range(nb_lookahead_steps):
                batch_start_lh, batch_end_lh = batcher.batches[(batch_no + i) % nb_batches]

                indices_lh = batcher.get_batch(batch_start_lh, batch_end_lh)
                x_batch_lh = torch.from_numpy(data.X[indices_lh, :].astype('int64')).to(device)

                loss_lh, factors_lh = get_loss(x_batch_lh, e_tensor_lh, p_tensor_lh, model, loss_function)

                features_p_lh, features_s_lh, features_o_lh = factors_lh
                if regularizer_weight_type in {'graph'}:
                    features_s_lh = F.embedding(x_batch_lh[:, 0], graph_features_t)
                    features_o_lh = F.embedding(x_batch_lh[:, 2], graph_features_t)

                reg_rel_lh = regularizer_rel_weight_model(factors_lh[0], features_p_lh)

                reg_s_lh = regularizer_ent_weight_model(factors_lh[1], features_s_lh)
                reg_o_lh = regularizer_ent_weight_model(factors_lh[2], features_o_lh)

                loss_lh += (reg_rel_lh + reg_s_lh + reg_o_lh) / 3.0

                e_tensor_lh, p_tensor_lh = diff_opt.step(loss_lh, params=parameter_lst)

            dev_indices = None
            if lookahead_sample_size is not None:
                dev_indices = random_state.permutation(data.dev_X.shape[0])[:lookahead_sample_size]

            x_val_batch = torch.from_numpy(data.dev_X[:dev_indices, :].astype('int64')).to(device)

            sp_to_o = po_to_s = None
            if is_lookahead_masked is True:
                sp_to_o, po_to_s = X_to_dicts(np.concatenate((data.X, data.dev_X), axis=0))

            loss_val, _ = get_loss(x_val_batch, e_tensor_lh, p_tensor_lh, model, loss_function, sp_to_o, po_to_s)

            loss_val.backward()

            if writer is not None:
                writer.add_scalar('Loss/Lookahead', loss_val.item(), step_no)

            hyper_optimizer.step()

            optimizer.zero_grad()
            hyper_optimizer.zero_grad()

            regularizer_rel_weight_model.project_()
            regularizer_ent_weight_model.project_()

            indices = batcher.get_batch(batch_start, batch_end)
            x_batch = torch.from_numpy(data.X[indices, :].astype('int64')).to(device)

            loss, factors = get_loss(x_batch, entity_embeddings, predicate_embeddings, model, loss_function)

            features_p, features_s, features_o = factors
            if regularizer_weight_type in {'graph'}:
                features_s = F.embedding(x_batch[:, 0], graph_features_t)
                features_o = F.embedding(x_batch[:, 2], graph_features_t)

            reg_rel = regularizer_rel_weight_model(factors[0], features_p)

            reg_s = regularizer_ent_weight_model(factors[1], features_s)
            reg_o = regularizer_ent_weight_model(factors[2], features_o)

            loss += (reg_rel + reg_s + reg_o) / 3.0

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
            hyper_optimizer.zero_grad()

            loss_value = loss.item()
            epoch_loss_values += [loss_value]

            if writer is not None:
                writer.add_scalar('Loss/Train', loss_value, step_no)

                rel_weights = regularizer_rel_weight_model.values_()
                ent_weights = regularizer_ent_weight_model.values_()

                writer.add_scalars('Rel/Weights', {k: v for k, v in rel_weights.items()}, step_no)
                writer.add_scalars('Ent/Weights', {k: v for k, v in ent_weights.items()}, step_no)

                dev_x = torch.from_numpy(data.dev_X[:dev_indices, :].astype('int64')).to(device)
                dev_loss, _ = get_loss(dev_x, entity_embeddings, predicate_embeddings, model, loss_function)
                writer.add_scalar('Loss/Dev', dev_loss, step_no)

                test_x = torch.from_numpy(data.test_X[:dev_indices, :].astype('int64')).to(device)
                test_loss, _ = get_loss(test_x, entity_embeddings, predicate_embeddings, model, loss_function)
                writer.add_scalar('Loss/Test', test_loss, step_no)

            if not is_quiet:
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f}')

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

        if validate_every is not None and epoch_no % validate_every == 0:
            for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
                metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                                   test_triples=triples, all_triples=data.all_triples,
                                   entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                                   model=model, batch_size=eval_batch_size, device=device)
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\t{name} results\t{metrics_to_str(metrics)}')
                if writer is not None:
                    ranking_values = {n.upper().replace("@", "_"): v for n, v in metrics.items()}
                    writer.add_scalars(f'Ranking/{name}', ranking_values, (epoch_no - 1) * nb_batches)

    for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                           test_triples=triples, all_triples=data.all_triples,
                           entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                           model=model, batch_size=eval_batch_size, device=device)
        logger.info(f'Final \t{name} results\t{metrics_to_str(metrics)}')

    if save_path is not None:
        torch.save(param_module.state_dict(), save_path)

    logger.info("Training finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
