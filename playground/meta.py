#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import multiprocessing
import numpy as np

import torch
from torch import nn, optim

from metakbc.models import DistMult, ComplEx
from metakbc.regularizers import F2, N3

import higher

import logging


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

torch.set_num_threads(multiprocessing.cpu_count())


def main(argv):
    init_size = 1e-3

    entity_embeddings = nn.Embedding(1024, 10, sparse=True)
    predicate_embeddings = nn.Embedding(1024, 10, sparse=True)

    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    parameters_lst = nn.ModuleDict({
        'entities': entity_embeddings,
        'predicates': predicate_embeddings
    })

    opt = optim.Adagrad(parameters_lst.parameters(), lr=0.1)
    model = ComplEx(entity_embeddings=entity_embeddings)
    loss_function = nn.CrossEntropyLoss(reduction='mean')

    N3_reg = N3()

    xp = torch.from_numpy(np.array([0]))
    xs = torch.from_numpy(np.array([1]))
    xo = torch.from_numpy(np.array([2]))

    xp_emb = predicate_embeddings(xp)
    xs_emb = entity_embeddings(xs)
    xo_emb = entity_embeddings(xo)

    print(model.score(xp_emb, xs_emb, xo_emb))

    sp_scores, po_scores = model.forward(xp_emb, xs_emb, xo_emb)

    s_loss = loss_function(sp_scores, xo)
    o_loss = loss_function(po_scores, xs)
    loss = s_loss + o_loss

    loss.backward()

    opt.step()
    opt.zero_grad()

    xp_emb = predicate_embeddings(xp)
    xs_emb = entity_embeddings(xs)
    xo_emb = entity_embeddings(xo)

    print(model.score(xp_emb, xs_emb, xo_emb))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
