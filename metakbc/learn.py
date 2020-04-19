# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import normalize, softmax

from metakbc.models import DistMult, ComplEx
from metakbc.datasets import Dataset
from metakbc.evaluation import evaluate
from metakbc.regularizers import InconsistencyLoss

from higher.optim import DifferentiableSGD
import higher

from typing import List

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def learn(dataset: Dataset,
          model: str,
          optimizer: str,
          meta_optimizer: str,
          lr: float,
          meta_lr: float,
          n_epochs_inner: int,
          n_epochs_outer: int,
          n_epochs_adv: int,
          n_epochs_dis: int,
          n_valid: int,
          rank: int,
          lam: float,
          batch_size: int,
          chunk_size: int,
          logging: bool) -> None:

    # ==========================================
    n_constraints = 2
    n_predicates = dataset.n_predicates

    w_head = torch.empty((n_constraints, n_predicates)).normal_().to(device)
    w_head.requires_grad = True
    w_body = torch.empty((n_constraints, n_predicates)).normal_().to(device)
    w_body.requires_grad = True
    # ==========================================

    model_ = {
        "DistMult": lambda: DistMult(size=dataset.get_shape(), rank=rank).to(device),
        "ComplEx": lambda: ComplEx(size=dataset.get_shape(), rank=rank).to(device),
    }[model]()

    optim_ = {
        "SGD": lambda: torch.optim.SGD(model_.parameters(), lr=lr, momentum=0.9),
        "Adagrad": lambda: torch.optim.Adagrad(model_.parameters(), lr=lr),
    }[optimizer]()

    meta_optim = {
        "SGD": lambda: torch.optim.SGD([w_head, w_body], lr=meta_lr, momentum=0.9),
        "Adagrad": lambda: torch.optim.Adagrad([w_head, w_body], lr=meta_lr),
    }[meta_optimizer]()

    loss = torch.nn.CrossEntropyLoss()

    for e_outer in range(n_epochs_outer):
        
        for i, chunk in enumerate(dataset.get_batches("train", chunk_size, shuffle=True)):

            with higher.innerloop_ctx(model_, optim_, copy_initial_weights=False) as (model, optim):

                for e_inner in range(n_epochs_inner):

                    n_examples = chunk.shape[0]
                    examples = chunk[torch.randperm(n_examples)]
                    batch_begin = 0
                    while batch_begin < n_examples:

                        batch = examples[batch_begin:batch_begin + batch_size]
                        batch = batch.to(device)
                        s_idx = batch[:, 0]
                        p_idx = batch[:, 1]
                        o_idx = batch[:, 2]

                        score_s = model.score_subjects(p_idx, o_idx)
                        score_o = model.score_objects(s_idx, p_idx)
                        loss_fact = loss(score_s, s_idx) + loss(score_o, o_idx)


                        emb_body = softmax(w_body, dim=1) @ model.emb_p
                        emb_head = softmax(w_head, dim=1) @ model.emb_p

                        loss_inc = torch.sum(torch.max(torch.abs(emb_body - emb_head), dim=1)[0])
                        loss_total = loss_fact + lam * loss_inc

                        optim.step(loss_total)

                        model.emb_p /= model.emb_p.norm(dim=1, p=2).view(-1, 1).detach()
                        model.emb_so /= model.emb_so.norm(dim=1, p=2).view(-1, 1).detach()

                        batch_begin += batch_size

                    print("\rOuter epoch: {:4} \t chunk: {} \t inner epoch: {:4} \t loss: {:.3f}".format(e_outer+1,i+1,e_inner+1,loss_total.item()), end="")


                # ==========================================
                # END INNER LOOP / META-OPTIMIZATION
                # ==========================================
                loss_valid = 0
                for batch_valid in dataset.get_batches('valid', batch_size, shuffle=True):

                    batch_valid = batch_valid.to(device)
                    s_idx_valid = batch_valid[:, 0]
                    p_idx_valid = batch_valid[:, 1]
                    o_idx_valid = batch_valid[:, 2]

                    score_s = model.score_subjects(p_idx_valid, o_idx_valid)
                    score_o = model.score_objects(s_idx_valid, p_idx_valid)
                    loss_valid += loss(score_s, s_idx_valid) + loss(score_o, o_idx_valid)

                meta_optim.zero_grad() 
                loss_valid.backward() 
                meta_optim.step()

        # ==========================================
        # FULL EVALUATION
        # ==========================================
        splits = ['train', 'valid']
        metrics_dict = evaluate(dataset, splits, model, batch_size)
        loss_total = {s: 0 for s in splits}
        for s in splits:
            for batch in dataset.get_batches(s, batch_size, shuffle=True):

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

        print("\rLoss (train): {:.2f} \t Loss (valid): {:.2f} \t MRR (train): {:.2f} \t MRR (valid): {:.2f}".format(loss_total['train'], loss_total['valid'], metrics_dict['train']['MRR'], metrics_dict['valid']['MRR']))
        print(softmax(w_body, dim=1).cpu().detach().numpy())
        print(softmax(w_head, dim=1).cpu().detach().numpy())

        if logging:
            wandb.log({**metrics_dict, 'epoch_outer': e_outer})
            wandb.log({**loss_total, 'epoch_outer': e_outer})