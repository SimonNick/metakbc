#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import multiprocessing
import numpy as np

import torch
from torch import nn, optim, Tensor

from metakbc.models import BaseModel, DistMult, ComplEx
from metakbc.regularizers import F2, N3

import higher

from typing import Tuple, Optional, List, Dict

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


class FinalModel(nn.Module):
    def __init__(self,
                 entity_embeddings: nn.Embedding,
                 predicate_embeddings: nn.Embedding,
                 model: BaseModel,
                 extra_params: nn.ParameterList = None):
        super().__init__()
        self.embeddings = nn.ModuleList([entity_embeddings, predicate_embeddings])
        self.model = model
        self.extra_params = extra_params

    def __call__(self,
                 rel: Tensor,
                 arg1: Tensor,
                 arg2: Tensor,
                 entity_embeddings: Optional[nn.Embedding] = None,
                 predicate_embeddings: Optional[nn.Embedding] = None) -> Tensor:
        entity_emb = self.embeddings[0]
        predicate_emb = self.embeddings[1]

        if entity_embeddings is not None and predicate_embeddings is not None:
            entity_emb = entity_embeddings
            predicate_emb = predicate_embeddings

        rel_emb = predicate_emb(rel)
        arg1_emb = entity_emb(arg1)
        arg2_emb = entity_emb(arg2)

        return self.model.score(rel_emb, arg1_emb, arg2_emb)

    def forward(self,
                rel: Tensor,
                arg1: Optional[Tensor],
                arg2: Optional[Tensor],
                entity_embeddings: Optional[nn.Embedding] = None,
                predicate_embeddings: Optional[nn.Embedding] = None) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        entity_emb = self.embeddings[0]
        predicate_emb = self.embeddings[1]

        if entity_embeddings is not None and predicate_embeddings is not None:
            entity_emb = entity_embeddings
            predicate_emb = predicate_embeddings

        rel_emb = predicate_emb(rel)
        arg1_emb = arg1 if arg1 is None else entity_emb(arg1)
        arg2_emb = arg2 if arg2 is None else entity_emb(arg2)

        return self.model.forward(rel_emb, arg1_emb, arg2_emb)

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.model.factor(embedding_vector)


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

    zzz = [x for x in parameters_lst.parameters()] # + [N3_weight]
    meta_zzz = [N3_weight]

    opt = optim.Adagrad(zzz, lr=0.1)
    meta_opt = optim.Adagrad(meta_zzz, lr=0.1)

    scoring_model = ComplEx(entity_embeddings=entity_embeddings)
    model = FinalModel(entity_embeddings=entity_embeddings,
                       predicate_embeddings=predicate_embeddings,
                       model=scoring_model)
                       # extra_params=nn.ParameterList([N3_weight]))

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    N3_reg = N3()

    xp = torch.from_numpy(np.array([0]))
    xs = torch.from_numpy(np.array([0]))
    xo = torch.from_numpy(np.array([0]))

    tmp = model(xp, xs, xo)
    print(tmp)

    for i in range(64):
        with higher.innerloop_ctx(model, opt, copy_initial_weights=False) as (fmodel, diffopt):
            for _ in range(4):
                sp_scores, po_scores = fmodel.forward(xp, xs, xo)
                loss = (1 - N3_weight) * loss_function(sp_scores, xo) + N3_weight * loss_function(po_scores, xs)
                a = diffopt.step(loss)

                # for x in a:
                #     print('XXX', x.shape)

                # print(len(list(fmodel.parameters())))

                # entity_emb = nn.Embedding.from_pretrained(a[0], freeze=False, sparse=True)
                # predicate_emb = nn.Embedding.from_pretrained(a[1], freeze=False, sparse=True)
                # tmp = fmodel(xp, xs, xo, entity_embeddings=entity_emb, predicate_embeddings=predicate_emb)
                # print(tmp)

            sp_scores, po_scores = fmodel.forward(xp, xs, xo)
            meta_loss = loss_function(sp_scores, xo) + loss_function(po_scores, xs)

            # G = torch.autograd.grad(meta_loss, N3_weight, allow_unused=True)
            # print(G)

            meta_loss.backward()

            meta_opt.step()
            meta_opt.zero_grad()

        # print(i, N3_weight.data)
        print(meta_loss)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
