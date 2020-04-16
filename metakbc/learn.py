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
          logging: bool) -> None:

    n_constraints = 2
    n_predicates = 4

    w_head = torch.empty((n_constraints, n_predicates)).normal_()
    w_head.requires_grad = True
    w_body = torch.empty((n_constraints, n_predicates)).normal_()
    w_body.requires_grad = True

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

    # optim = DifferentiableSGD(optim_, model_.parameters()) 
    # model = model_

    loss = torch.nn.CrossEntropyLoss()
    reg = InconsistencyLoss()

    # outer loop
    for e_outer in range(n_epochs_outer):

        with higher.innerloop_ctx(model_, optim_, copy_initial_weights=False) as (model, optim):

            loss_total = torch.Tensor([0])

            for e_inner in range(n_epochs_inner):

                print("\rOuter epoch: {:4} \t inner epoch: {:4} \t loss: {:.3f}".format(e_outer+1,e_inner+1,loss_total.item()), end="")

                # ==========================================
                # ADVERSARY
                # =======k===================================
                # # x1 = torch.empty(rank).normal_()
                # # x2 = torch.empty(rank).normal_()
                # x1 = torch.empty(2*rank).normal_()
                # x2 = torch.empty(2*rank).normal_()
                # x1.requires_grad = True
                # x2.requires_grad = True

                # optim_adv_ = torch.optim.SGD([x1, x2], lr=0.01, momentum=0.9)

                # # model2 = DistMult(size=dataset.get_shape(), rank=rank).to(device)
                # model2 = ComplEx(size=dataset.get_shape(), rank=rank).to(device)
                # with torch.no_grad():
                #     for p1, p2 in zip(model.parameters(), model2.parameters()):
                #         p2.copy_(p1)
                # optim_adv = optim_adv_

                # # model2 = model
                # # optim_adv = DifferentiableSGD(optim_adv_, [x1, x2])

                # theta_father = model2.emb_p[0]
                # theta_parent = model2.emb_p[1]

                # for e_adv in range(n_epochs_adv):

                #     score_father = model2._scoring_func(x1, theta_father, x2)
                #     score_parent = model2._scoring_func(x1, theta_parent, x2)

                #     loss_inc = torch.nn.functional.relu(score_father - score_parent)
                #     loss_inc *= -1 # perform gradient ascent

                #     # optim_adv.step(loss_inc, [x1, x2])
                #     optim_adv.zero_grad()
                #     loss_inc.backward()
                #     optim_adv.step()

                #     with torch.no_grad():
                #         x1 /= x1.norm(p=2)
                #         x2 /= x2.norm(p=2)


                # ==========================================
                # DISCRIMINATOR
                # ==========================================
                for e_dis in range(n_epochs_dis):

                    for batch in dataset.get_batches("train", batch_size, shuffle=True):

                        batch = batch.to(device)
                        s_idx = batch[:, 0]
                        p_idx = batch[:, 1]
                        o_idx = batch[:, 2]

                        score_s = model.score_subjects(p_idx, o_idx)
                        score_o = model.score_objects(s_idx, p_idx)
                        loss_fact = loss(score_s, s_idx) + loss(score_o, o_idx)

                        # theta_father = model.emb_p[0]
                        # theta_parent = model.emb_p[1]
                        # score_father = model._scoring_func(x1, theta_father, x2)
                        # score_parent = model._scoring_func(x1, theta_parent, x2)
                        # loss_inc = torch.nn.functional.relu(score_father - score_parent)

                        emb_body = softmax(w_body, dim=1) @ model.emb_p
                        emb_head = softmax(w_head, dim=1) @ model.emb_p

                        loss_inc = torch.sum(torch.max(torch.abs(emb_body - emb_head), dim=1)[0])

                        # loss_inc = reg([model.emb_p])
                        loss_total = loss_fact + lam * loss_inc

                        optim.step(loss_total)
                        # optim.zero_grad() 
                        # loss_total.backward() 
                        # optim.step()

                        model.emb_p /= model.emb_p.norm(dim=1, p=2).view(-1, 1).detach()
                        model.emb_so /= model.emb_so.norm(dim=1, p=2).view(-1, 1).detach()

                # ==========================================
                # EVALUATION METRICS
                # ==========================================
                # if e_inner % n_valid == 0:

                    # metrics_dict = evaluate(dataset, ["train", "valid"], model, batch_size)

                    # if logging:
                    #     metrics_dict['epoch_inner'] = e_inner
                    #     wandb.log(metrics_dict)
                    # else:
                    #     print("train", metrics_dict['train']['MRR'])
                        # print("valid", metrics_dict['valid']['MRR'])

            # ==========================================
            # END INNER LOOP / OUTER LOOP
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

            meta_optim.zero_grad() 
            loss_total['valid'].backward() 
            meta_optim.step()

            optim_.zero_grad()

            # ==========================================
            # LOGGING
            # ==========================================
            loss_total = {s: l.item() for s,l in loss_total.items()}

            print("\rLoss (train): {:.2f} \t Loss (valid): {:.2f} \t MRR (train): {:.2f} \t MRR (valid): {:.2f}".format(loss_total['train'], loss_total['valid'], metrics_dict['train']['MRR'], metrics_dict['valid']['MRR']))
            print(softmax(w_body, dim=1).detach().numpy())
            print(softmax(w_head, dim=1).detach().numpy())

            if logging:
                wandb.log({**metrics_dict, 'epoch_outer': e_outer})
                wandb.log({**loss_total, 'epoch_outer': e_outer})