# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import normalize, softmax

import numpy as np

from metakbc.models import DistMult, ComplEx
from metakbc.datasets import Dataset
from metakbc.evaluation import evaluate
from metakbc.regularizers import InconsistencyLoss

from higher.optim import DifferentiableSGD
import higher

from typing import List

import wandb
import sys
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def learn(dataset: Dataset,
          model_str: str,
          optimizer_str: str,
          meta_optimizer: str,
          lr: float,
          meta_lr: float,
          n_epochs_outer: int,
          n_batches_train: int,
          n_batches_valid: int,
          n_valid: int,
        #   n_epochs_adv: int,
        #   n_epochs_dis: int,
          rank: int,
          batch_size: int,
          lam: float,
          logging: bool) -> None:

    # ==========================================
    n_constraints = 2
    n_predicates = dataset.n_predicates

    # Hyperparameters that we want to learn
    w_head = torch.empty((n_constraints, n_predicates)).normal_().to(device)
    w_head.requires_grad = True
    w_body = torch.empty((n_constraints, n_predicates)).normal_().to(device)
    w_body.requires_grad = True
    # ==========================================

    model = {
        "DistMult": lambda: DistMult(size=dataset.get_shape(), rank=rank).to(device),
        "ComplEx": lambda: ComplEx(size=dataset.get_shape(), rank=rank).to(device),
    }[model_str]()

    optim = {
        "SGD": lambda: torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "Adagrad": lambda: torch.optim.Adagrad(model.parameters(), lr=lr),
    }[optimizer_str]()

    meta_optim = {
        "SGD": lambda: torch.optim.SGD([w_head, w_body], lr=meta_lr, momentum=0.9),
        "Adagrad": lambda: torch.optim.Adagrad([w_head, w_body], lr=meta_lr),
    }[meta_optimizer]()

    for e_outer in range(n_epochs_outer):

        all_batches = [b for b in dataset.get_batches('train', batch_size, shuffle=True)]
        for k in range(math.ceil(len(all_batches) / n_batches_train)):

            print("\rOuter epoch: {:4}, k={}".format(e_outer+1,k+1), end="")

            with higher.innerloop_ctx(model, optim, copy_initial_weights=True) as (model_monkeypatch, diff_optim):

                for batch in all_batches[k*n_batches_train:(k+1)*n_batches_train]:

                    # ==========================================
                    # INNER LOOP
                    # ==========================================
                    loss = torch.nn.CrossEntropyLoss()

                    batch = batch.to(device)
                    s_idx = batch[:, 0]
                    p_idx = batch[:, 1]
                    o_idx = batch[:, 2]

                    score_s = model_monkeypatch.score_subjects(p_idx, o_idx)
                    score_o = model_monkeypatch.score_objects(s_idx, p_idx)
                    loss_fact = loss(score_s, s_idx) + loss(score_o, o_idx)

                    emb_body = softmax(w_body, dim=1) @ model_monkeypatch.emb_p
                    emb_head = softmax(w_head, dim=1) @ model_monkeypatch.emb_p

                    loss_inc = torch.sum(torch.max(torch.abs(emb_body - emb_head), dim=1)[0])
                    loss_total = loss_fact + lam * loss_inc

                    diff_optim.step(loss_total)

                    with torch.no_grad():
                        model_monkeypatch.emb_p /= model_monkeypatch.emb_p.norm(dim=1, p=2).view(-1, 1).detach()
                        model_monkeypatch.emb_so /= model_monkeypatch.emb_so.norm(dim=1, p=2).view(-1, 1).detach()

                # ==========================================
                # META-OPTIMIZATION
                # ==========================================
                loss_valid = 0
                loss = torch.nn.CrossEntropyLoss(reduction='sum')
                for k, batch in enumerate(dataset.get_batches('valid', batch_size, shuffle=True)):
                    if k == n_batches_valid: break

                    batch = batch.to(device)
                    s_idx = batch[:, 0]
                    p_idx = batch[:, 1]
                    o_idx = batch[:, 2]

                    score_s = model_monkeypatch.score_subjects(p_idx, o_idx)
                    score_o = model_monkeypatch.score_objects(s_idx, p_idx)
                    loss_valid += loss(score_s, s_idx) + loss(score_o, o_idx)

                meta_optim.zero_grad() 
                loss_valid.backward() 
                meta_optim.step()

            # ==========================================
            # copy learned weights to original model
            with torch.no_grad():
                model.emb_so.copy_(model_monkeypatch.emb_so)
                model.emb_p.copy_(model_monkeypatch.emb_p)

        if e_outer % n_valid == 0:
    
            # ==========================================
            # FULL EVALUATION
            # ==========================================
            splits = ['train', 'valid']
            metrics_dict = evaluate(dataset, splits, model, batch_size)
            loss_total = {s: 0 for s in splits}
            loss = torch.nn.CrossEntropyLoss(reduction='sum')
            for s in splits:
                for batch in dataset.get_batches(s, batch_size, shuffle=False):

                    batch = batch.to(device)
                    s_idx = batch[:, 0]
                    p_idx = batch[:, 1]
                    o_idx = batch[:, 2]

                    score_s = model.score_subjects(p_idx, o_idx)
                    score_o = model.score_objects(s_idx, p_idx)
                    loss_total[s] += loss(score_s, s_idx) + loss(score_o, o_idx)

            # ==========================================
            # LOGGING
            # ==========================================
            loss_total = {s: l.item() for s,l in loss_total.items()}

            print("\r" + 100*" ", end="")
            print("\rLoss (train): {:.2f} \t Loss (valid): {:.2f} \t MRR (train): {:.2f} \t MRR (valid): {:.2f}".format(loss_total['train'], loss_total['valid'], metrics_dict['train']['MRR'], metrics_dict['valid']['MRR']))
            print(softmax(w_body, dim=1).cpu().detach().numpy())
            print(softmax(w_head, dim=1).cpu().detach().numpy())

            if logging:
                wandb.log({**metrics_dict, 'epoch_outer': e_outer})
                wandb.log({**loss_total, 'epoch_outer': e_outer})