#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import torch
from torch import nn, optim

from metakbc.models import ComplEx
from metakbc.regularizers import N3

import higher

import logging

torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main(argv):
    torch.random.manual_seed(0)

    e_tensor = torch.rand(8, 10, requires_grad=True)
    p_tensor = torch.rand(16, 10, requires_grad=True)

    entity_embeddings = nn.Embedding.from_pretrained(e_tensor, freeze=False, sparse=False)
    predicate_embeddings = nn.Embedding.from_pretrained(p_tensor, freeze=False, sparse=False)

    model = ComplEx(entity_embeddings=entity_embeddings)
    regularizer = N3()

    param_lst = [e_tensor, p_tensor]
    opt = optim.SGD(param_lst, lr=0.1)

    loss_function = nn.CrossEntropyLoss(reduction='mean')

    xp = torch.from_numpy(np.array([0]))
    xs = torch.from_numpy(np.array([0]))
    xo = torch.from_numpy(np.array([0]))

    diff_opt = higher.get_diff_optim(opt, param_lst)

    for i in range(6):
        entity_emb = nn.Embedding.from_pretrained(e_tensor, freeze=False, sparse=False)
        predicate_emb = nn.Embedding.from_pretrained(p_tensor, freeze=False, sparse=False)

        xp_emb = predicate_emb(xp)
        xs_emb = entity_emb(xs)
        xo_emb = entity_emb(xo)

        sp_scores, po_scores = model.forward(xp_emb, xs_emb, xo_emb, entity_embeddings=e_tensor)

        loss = loss_function(sp_scores, xo) + loss_function(po_scores, xs)
        loss += 0.01 * regularizer([xp_emb, xs_emb, xo_emb])

        print(loss)

        e_tensor, p_tensor = diff_opt.step(loss, params=param_lst)
        # print('XXX', e_tensor[0, 0])



if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
