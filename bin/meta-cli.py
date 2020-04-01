#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim, Tensor
from torch.nn import Parameter

from torch.utils.tensorboard import SummaryWriter

import higher

from metakbc.training.data import Data
from metakbc.training.batcher import Batcher

from metakbc.models import BaseModel, DistMult, ComplEx

from metakbc.regularizers import F2, L1, N3
from metakbc.evaluation import evaluate

from typing import List, Tuple

import logging


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

torch.set_num_threads(multiprocessing.cpu_count())


def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tH@1 {metrics["hits@1"]:.6f}\tH@3 {metrics["hits@3"]:.6f}\t' \
        f'H@5 {metrics["hits@5"]:.6f}\tH@10 {metrics["hits@10"]:.6f}'


def get_loss(X: Tensor,
             entity_embeddings: nn.Embedding,
             predicate_embeddings: nn.Embedding,
             model: BaseModel,
             loss_function: nn.CrossEntropyLoss) -> Tuple[Tensor, List[Tensor]]:
    xs_batch_emb = entity_embeddings(X[:, 0])
    xp_batch_emb = predicate_embeddings(X[:, 1])
    xo_batch_emb = entity_embeddings(X[:, 2])

    sp_scores, po_scores = model.forward(xp_batch_emb, xs_batch_emb, xo_batch_emb,
                                         entity_embeddings=entity_embeddings.weight)
    factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

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

    parser.add_argument('--L1', action='store', type=float, default=None)
    parser.add_argument('--F2', action='store', type=float, default=None)
    parser.add_argument('--N3', action='store', type=float, default=None)

    parser.add_argument('--lookahead-steps', '--LS', '-S', action='store', type=int, default=0)
    parser.add_argument('--lookahead-learning-rate', '--LL', action='store', type=float, default=None)
    parser.add_argument('--lookahead-sample-size', '--LSS', action='store', type=float, default=None)

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
    pprint.pprint(vars(args))

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

    L1_weight = Parameter(torch.tensor(args.L1, dtype=torch.float32), requires_grad=True) if args.L1 is not None else None
    F2_weight = Parameter(torch.tensor(args.F2, dtype=torch.float32), requires_grad=True) if args.F2 is not None else None
    N3_weight = Parameter(torch.tensor(args.N3, dtype=torch.float32), requires_grad=True) if args.N3 is not None else None

    nb_lookahead_steps = args.lookahead_steps

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

    parameter_lst = nn.ParameterList([entity_embeddings.weight, predicate_embeddings.weight]).to(device)

    hyperparameter_lst = ([] if L1_weight is None else [L1_weight]) + \
                         ([] if F2_weight is None else [F2_weight]) + \
                         ([] if N3_weight is None else [N3_weight])
    hyperparameter_lst = nn.ParameterList(hyperparameter_lst).to(device)

    model_factory = {
        'distmult': lambda: DistMult(),
        'complex': lambda: ComplEx()
    }

    assert model_name in model_factory
    model = model_factory[model_name]().to(device)

    logger.info('Model state:')
    for param_tensor in param_module.state_dict():
        logger.info(f'\t{param_tensor}\t{param_module.state_dict()[param_tensor].size()}')

    optimizer_factory = {
        'adagrad': lambda *args, **kwargs: optim.Adagrad(*args, **kwargs),
        'adam': lambda *args, **kwargs: optim.Adam(*args, **kwargs),
        'sgd': lambda *args, **kwargs: optim.SGD(*args, **kwargs)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name](parameter_lst, lr=learning_rate)

    print(optimizer)

    hyper_optimizer = optimizer_factory[optimizer_name](hyperparameter_lst, lr=0.01)

    print(hyper_optimizer)

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    L1_reg, F2_reg, N3_reg = L1(), F2(), N3()

    for triples, name in [(t, n) for t, n in triples_name_pairs if len(t) > 0]:
        metrics = evaluate(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                           test_triples=triples, all_triples=data.all_triples,
                           entity_to_index=data.entity_to_idx, predicate_to_index=data.predicate_to_idx,
                           model=model, batch_size=eval_batch_size, device=device)
        if writer is not None:
            writer.add_scalars(f'Ranking/{name}', {n.upper().replace("@", "_"): v for n, v in metrics.items()}, 0)
            # writer.add_embedding(entity_embeddings.weight, sorted(data.entity_to_idx.items(), key=lambda item: item[1]), global_step=0, tag='Entities')
            # writer.add_embedding(predicate_embeddings.weight, sorted(data.predicate_to_idx.items(), key=lambda item: item[1]), global_step=0, tag='Predicates')

    for epoch_no in range(1, nb_epochs + 1):
        batcher = Batcher(data.nb_examples, batch_size, 1, random_state)
        nb_batches = len(batcher.batches)

        epoch_loss_values = []
        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
            diff_opt = higher.get_diff_optim(optimizer, parameter_lst)

            entity_embeddings_lh = entity_embeddings
            predicate_embeddings_lh = predicate_embeddings

            for i in range(nb_lookahead_steps):
                batch_start_lh, batch_end_lh = batcher.batches[(batch_no + i) % nb_batches]

                indices_lh = batcher.get_batch(batch_start_lh, batch_end_lh)
                x_batch_lh = torch.from_numpy(data.X[indices_lh, :].astype('int64')).to(device)

                loss_lh, factors_lh = get_loss(x_batch_lh, entity_embeddings_lh, predicate_embeddings_lh, model, loss_function)
                if L1_weight is not None:
                    loss_lh += L1_weight * L1_reg(factors_lh)
                if F2_weight is not None:
                    loss_lh += F2_weight * F2_reg(factors_lh)
                if N3_weight is not None:
                    loss_lh += N3_weight * N3_reg(factors_lh)

                e_tensor_lh, p_tensor_lh, = diff_opt.step(loss_lh, params=parameter_lst)

                entity_embeddings_lh = nn.Embedding.from_pretrained(e_tensor_lh, freeze=False, sparse=False)
                predicate_embeddings_lh = nn.Embedding.from_pretrained(p_tensor_lh, freeze=False, sparse=False)

            x_val_batch = torch.from_numpy(data.dev_X[:500, :].astype('int64')).to(device)
            loss_val, _ = get_loss(x_val_batch, entity_embeddings_lh, predicate_embeddings_lh, model, loss_function)

            loss_val.backward()

            if writer is not None:
                writer.add_scalar('Loss/Lookahead', loss_val.item(), ((epoch_no - 1) * nb_batches) + batch_no)

            hyper_optimizer.step()
            hyper_optimizer.zero_grad()

            if L1_weight is not None:
                L1_weight.data.clamp_(0)

            if F2_weight is not None:
                F2_weight.data.clamp_(0)

            if N3_weight is not None:
                N3_weight.data.clamp_(0)

            indices = batcher.get_batch(batch_start, batch_end)
            x_batch = torch.from_numpy(data.X[indices, :].astype('int64')).to(device)

            loss, factors = get_loss(x_batch, entity_embeddings, predicate_embeddings, model, loss_function)
            loss += L1_weight * L1_reg(factors) + F2_weight * F2_reg(factors) + N3_weight * N3_reg(factors)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_value = loss.item()
            epoch_loss_values += [loss_value]

            if writer is not None:
                writer.add_scalar('Loss/Train', loss_value, ((epoch_no - 1) * nb_batches) + batch_no)

                weights = {'L1': L1_weight.data if L1_weight is not None else None,
                           'F2': F2_weight.data if F2_weight is not None else None,
                           'N3': N3_weight.data if N3_weight is not None else None}
                print(weights)
                writer.add_scalars('Weights', {k: v for k, v in weights.items() if v is not None}, ((epoch_no - 1) * nb_batches) + batch_no)

                dev_x = torch.from_numpy(data.dev_X[:100, :].astype('int64')).to(device)
                dev_loss, _ = get_loss(dev_x, entity_embeddings, predicate_embeddings, model, loss_function)
                writer.add_scalar('Loss/Dev', dev_loss, ((epoch_no - 1) * nb_batches) + batch_no)

                test_x = torch.from_numpy(data.test_X[:100, :].astype('int64')).to(device)
                test_loss, _ = get_loss(test_x, entity_embeddings, predicate_embeddings, model, loss_function)
                writer.add_scalar('Loss/Test', test_loss, ((epoch_no - 1) * nb_batches) + batch_no)

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
                    writer.add_scalars(f'Ranking/{name}', {n.upper().replace("@", "_"): v for n, v in metrics.items()}, (epoch_no - 1) * nb_batches)
                    # writer.add_embedding(entity_embeddings.weight, sorted(data.entity_to_idx.items(), key=lambda item: item[1]), global_step=(epoch_no - 1) * nb_batches, tag='Entities')
                    # writer.add_embedding(predicate_embeddings.weight, sorted(data.predicate_to_idx.items(), key=lambda item: item[1]), global_step=(epoch_no - 1) * nb_batches, tag='Predicates')

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
