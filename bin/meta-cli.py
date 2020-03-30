#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import multiprocessing
import numpy as np

import torch
from torch import nn, optim
from torch.nn import Parameter

from torch.utils.tensorboard import SummaryWriter

import higher

from metakbc.training.data import Data
from metakbc.training.batcher import Batcher

from metakbc.models import DistMult, ComplEx

from metakbc.regularizers import F2, L1, N3
from metakbc.evaluation import evaluate

import logging


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

torch.set_num_threads(multiprocessing.cpu_count())


def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tH@1 {metrics["hits@1"]:.6f}\tH@3 {metrics["hits@3"]:.6f}\t' \
        f'H@5 {metrics["hits@5"]:.6f}\tH@10 {metrics["hits@10"]:.6f}'


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

    parser.add_argument('--lookahead-steps', '-S', action='store', type=int, default=0)

    parser.add_argument('--seed', action='store', type=int, default=0)

    parser.add_argument('--validate-every', '-V', action='store', type=int, default=None)

    parser.add_argument('--input-type', '-I', action='store', type=str, default='standard',
                        choices=['standard', 'reciprocal'])

    parser.add_argument('--load', action='store', type=str, default=None)
    parser.add_argument('--save', action='store', type=str, default=None)

    parser.add_argument('--quiet', '-q', action='store_true', default=False)

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

    L1_weight = Parameter(torch.tensor(args.L1, dtype=torch.float64), requires_grad=True)
    F2_weight = Parameter(torch.tensor(args.F2, dtype=torch.float64), requires_grad=True)
    N3_weight = Parameter(torch.tensor(args.N3, dtype=torch.float64), requires_grad=True)

    nb_lookahead_steps = args.lookahead_steps

    validate_every = args.validate_every
    input_type = args.input_type

    load_path = args.load
    save_path = args.save

    is_quiet = args.quiet

    # set the seeds
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    writer = SummaryWriter('runs/meta_1')

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

    parameter_lst = nn.ParameterList([entity_embeddings.weight, predicate_embeddings.weight])
    hyperparameter_lst = nn.ParameterList([L1_weight, F2_weight, N3_weight])

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
    # optimizer = optimizer_factory['sgd'](parameter_lst, lr=0.1)

    print(optimizer)

    hyper_optimizer = optimizer_factory[optimizer_name](hyperparameter_lst, lr=learning_rate)

    print(hyper_optimizer)

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    L1_reg, F2_reg, N3_reg = L1(), F2(), N3()

    for epoch_no in range(1, nb_epochs + 1):
        batcher = Batcher(data.nb_examples, batch_size, 1, random_state)
        nb_batches = len(batcher.batches)

        epoch_loss_values = []
        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
            diff_opt = higher.get_diff_optim(optimizer, parameter_lst)

            indices = batcher.get_batch(batch_start, batch_end)
            x_batch = torch.from_numpy(data.X[indices, :].astype('int64')).to(device)

            xs_batch_emb = entity_embeddings(x_batch[:, 0])
            xp_batch_emb = predicate_embeddings(x_batch[:, 1])
            xo_batch_emb = entity_embeddings(x_batch[:, 2])

            sp_scores, po_scores = model.forward(xp_batch_emb, xs_batch_emb, xo_batch_emb,
                                                 entity_embeddings=entity_embeddings.weight)
            factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

            s_loss = loss_function(sp_scores, x_batch[:, 2])
            o_loss = loss_function(po_scores, x_batch[:, 0])

            loss = s_loss + o_loss
            loss += L1_weight * L1_reg(factors) + \
                    F2_weight * F2_reg(factors) + \
                    N3_weight * N3_reg(factors)

            loss.backward(retain_graph=True)

            optimizer.step()

            loss_value = loss.item()
            epoch_loss_values += [loss_value]

            entity_embeddings_lh = entity_embeddings
            predicate_embeddings_lh = predicate_embeddings

            loss_lh = loss

            for i in range(nb_lookahead_steps):

                if i > 0:
                    batch_start_lh, batch_end_lh = batcher.batches[(batch_no + i) % nb_batches]

                    indices_lh = batcher.get_batch(batch_start_lh, batch_end_lh)
                    x_batch_lh = torch.from_numpy(data.X[indices_lh, :].astype('int64')).to(device)

                    xs_batch_emb_lh = entity_embeddings(x_batch_lh[:, 0])
                    xp_batch_emb_lh = predicate_embeddings(x_batch_lh[:, 1])
                    xo_batch_emb_lh = entity_embeddings(x_batch_lh[:, 2])

                    sp_scores_lh, po_scores_lh = model.forward(xp_batch_emb_lh, xs_batch_emb_lh, xo_batch_emb_lh,
                                                               entity_embeddings=entity_embeddings_lh.weight)
                    factors_lh = [model.factor(e) for e in [xp_batch_emb_lh, xs_batch_emb_lh, xo_batch_emb_lh]]

                    s_loss_lh = loss_function(sp_scores_lh, x_batch_lh[:, 2])
                    o_loss_lh = loss_function(po_scores_lh, x_batch_lh[:, 0])

                    loss_lh = s_loss_lh + o_loss_lh
                    loss_lh += L1_weight * L1_reg(factors_lh) + \
                               F2_weight * F2_reg(factors_lh) + \
                               N3_weight * N3_reg(factors_lh)

                e_tensor_lh, p_tensor_lh, = diff_opt.step(loss_lh, params=parameter_lst)

                entity_embeddings_lh = nn.Embedding.from_pretrained(e_tensor_lh, freeze=False, sparse=False)
                predicate_embeddings_lh = nn.Embedding.from_pretrained(p_tensor_lh, freeze=False, sparse=False)

            x_val_batch = torch.from_numpy(data.dev_X.astype('int64')).to(device)

            xs_batch_emb_val = entity_embeddings_lh(x_val_batch[:, 0])
            xp_batch_emb_val = predicate_embeddings_lh(x_val_batch[:, 1])
            xo_batch_emb_val = entity_embeddings_lh(x_val_batch[:, 2])

            sp_scores_val, po_scores_val = model.forward(xp_batch_emb_val, xs_batch_emb_val, xo_batch_emb_val,
                                                         entity_embeddings=entity_embeddings_lh.weight)

            s_loss_val = loss_function(sp_scores_val, x_val_batch[:, 2])
            o_loss_val = loss_function(po_scores_val, x_val_batch[:, 0])

            loss_val = s_loss_val + o_loss_val




            loss_val.backward()
            hyper_optimizer.step()




            optimizer.zero_grad()
            hyper_optimizer.zero_grad()

            L1_weight.data.clamp_(0)
            F2_weight.data.clamp_(0)
            N3_weight.data.clamp_(0)

            print(L1_weight.data, F2_weight.data, N3_weight.data)

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
