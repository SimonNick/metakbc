#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import torch
from torch import nn, optim, Tensor

from metakbc.models import BaseModel, DistMult, ComplEx
from metakbc.regularizers import F2, N3

import higher

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    init_size = 1e-3

    entity_embeddings = nn.Embedding(32, 10, sparse=False)
    predicate_embeddings = nn.Embedding(64, 10, sparse=False)

    entity_embeddings.weight.data *= init_size
    predicate_embeddings.weight.data *= init_size

    N3_weight = nn.Parameter(torch.tensor(1e-3), requires_grad=True)

    parameters_lst = nn.ModuleDict({
        'entities': entity_embeddings,
        'predicates': predicate_embeddings
    })

    zzz = [x for x in parameters_lst.parameters()] + [N3_weight]
    meta_zzz = [N3_weight]

    opt = optim.Adagrad(zzz, lr=0.1)
    meta_opt = optim.Adagrad(meta_zzz, lr=0.1)

    model = ComplEx(entity_embeddings=entity_embeddings)

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    N3_reg = N3()

    xp = torch.from_numpy(np.array([0]))
    xs = torch.from_numpy(np.array([0]))
    xo = torch.from_numpy(np.array([0]))

    xp_emb = predicate_embeddings(xp)
    xs_emb = entity_embeddings(xs)
    xo_emb = entity_embeddings(xo)

    tmp = model.score(xp_emb, xs_emb, xo_emb)
    print(tmp)

    for i in range(64):
        with higher.innerloop_ctx(model, opt, copy_initial_weights=False) as (fmodel, diffopt):
            for _ in range(4):
                xp_emb = predicate_embeddings(xp)
                xs_emb = entity_embeddings(xs)
                xo_emb = entity_embeddings(xo)

                sp_scores, po_scores = fmodel.forward(xp_emb, xs_emb, xo_emb)
                loss = (1 - N3_weight) * loss_function(sp_scores, xo) + N3_weight * loss_function(po_scores, xs)
                tmp = diffopt.step(loss)

                print([x.shape for x in tmp])

                sys.exit(0)

            xp_emb = predicate_embeddings(xp)
            xs_emb = entity_embeddings(xs)
            xo_emb = entity_embeddings(xo)

            sp_scores, po_scores = fmodel.forward(xp_emb, xs_emb, xo_emb)
            meta_loss = loss_function(sp_scores, xo) + loss_function(po_scores, xs)

            meta_loss.backward()

            meta_opt.step()
            meta_opt.zero_grad()

            print(meta_loss)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
