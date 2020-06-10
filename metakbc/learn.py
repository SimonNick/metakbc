# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import normalize, softmax, cross_entropy

from metakbc.models import DistMult, ComplEx
from metakbc.datasets import Dataset
from metakbc.evaluation import evaluate, build_filters
from metakbc.regularizers import N3
from metakbc.adversary import Adversary
from metakbc.clause import LearnedClause

import higher

from typing import List

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def learn(dataset: Dataset,
          model_str: str,
          optimizer_str: str,
          meta_optimizer: str,
          adv_optimizer: str,
          lr: float,
          meta_lr: float,
          adv_lr: float,
          n_epochs_outer: int,
          n_batches_train: int,
          n_batches_valid: int,
          n_valid: int,
          n_epochs_adv: int,
        #   n_epochs_dis: int,
          rank: int,
          batch_size: int,
          lam: float,
          logging: bool) -> None:

    # ==========================================
    # A(X1, X2) => B(X1, X2)
    clause1_func = lambda x1, x2, phi1, phi2: phi1(x1, x2) - phi2(x1, x2)
    clause1 = LearnedClause(n_variables=2, n_relations=2, clause_loss_func=clause1_func, n_constraints=4, n_predicates=dataset.n_predicates)

    # A(X1, X2) => A(X2, X1)
    clause2_func = lambda x1, x2, phi1: phi1(x1, x2) - phi1(x2, x1)
    clause2 = LearnedClause(n_variables=2, n_relations=1, clause_loss_func=clause2_func, n_constraints=10, n_predicates=dataset.n_predicates)

    clauses = [clause1, clause2]
    # ==========================================

    regularizer = N3()
    filters = build_filters(dataset)
    adversary = Adversary(clauses).to(device)
    adversarial_examples = None

    model = {
        "DistMult": lambda: DistMult(size=dataset.get_shape(), rank=rank).to(device),
        "ComplEx": lambda: ComplEx(size=dataset.get_shape(), rank=rank).to(device),
    }[model_str]()

    optim = {
        "SGD": lambda: torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "Adagrad": lambda: torch.optim.Adagrad(model.parameters(), lr=lr),
    }[optimizer_str]()

    meta_optim = {
        "SGD": lambda: torch.optim.SGD(adversary.parameters(), lr=meta_lr, momentum=0.9),
        "Adagrad": lambda: torch.optim.Adagrad(adversary.parameters(), lr=meta_lr),
    }[meta_optimizer]()

    for e_outer in range(n_epochs_outer):

        print("\rOuter epoch: {:4}".format(e_outer+1), end="")

        batches = dataset.get_batches('train', batch_size, shuffle=True)
        epoch_complete = False
        while not epoch_complete:

            # ==========================================
            # ADVERSERIAL TRAINING
            # ==========================================
            adversarial_examples = adversary.generate_adversarial_examples(model, n_epochs_adv, adv_optimizer, adv_lr, adversarial_examples=adversarial_examples)

            with higher.innerloop_ctx(model, optim, copy_initial_weights=True) as (fmodel, diff_optim):

                # ==========================================
                # TRAINING
                # ==========================================
                for k, batch in enumerate(batches):
                    if k == n_batches_train: break

                    batch = batch.to(device)
                    s_idx, p_idx, o_idx = batch[:, 0], batch[:, 1], batch[:, 2]

                    score_s = fmodel.score_subjects(p_idx, o_idx)
                    score_o = fmodel.score_objects(s_idx, p_idx)

                    loss_fact = cross_entropy(score_s, s_idx) + cross_entropy(score_o, o_idx)
                    loss_inc = adversary(fmodel, adversarial_examples)
                    loss_reg = regularizer(fmodel.factors(s_idx, p_idx, o_idx))
                    loss_total = loss_fact + lam * loss_inc + 1e-3 * loss_reg
                    # print("loss fact: {} | loss inc: {} | loss reg: {}".format(loss_fact.item(), loss_inc.item(), loss_reg.item()))

                    diff_optim.step(loss_total)
            
                else:
                    epoch_complete = True

                # ==========================================
                # META-OPTIMIZATION
                # ==========================================
                loss_valid = 0
                for k, batch in enumerate(dataset.get_batches('valid', batch_size, shuffle=True)):
                    if k == n_batches_valid: break

                    batch = batch.to(device)
                    s_idx, p_idx, o_idx = batch[:, 0], batch[:, 1], batch[:, 2]

                    score_s = fmodel.score_subjects(p_idx, o_idx)
                    score_o = fmodel.score_objects(s_idx, p_idx)
                    loss_valid += cross_entropy(score_s, s_idx) + cross_entropy(score_o, o_idx)

                meta_optim.zero_grad() 
                loss_valid.backward() 
                meta_optim.step()

                # ==========================================
                # copy learned weights to original model
                # ==========================================
                with torch.no_grad():
                    model.emb_so.copy_(fmodel.emb_so)
                    model.emb_p.copy_(fmodel.emb_p)

        if e_outer % n_valid == 0:
    
            # ==========================================
            # EVALUATION
            # ==========================================
            splits = ['train', 'valid']
            metrics_dict = evaluate(dataset, splits, model, batch_size, filters)
            loss_total = {s: 0 for s in splits}
            for s in splits:
                for batch in dataset.get_batches(s, batch_size):
                    batch = batch.to(device)
                    s_idx, p_idx, o_idx = batch[:, 0], batch[:, 1], batch[:, 2]
                    score_s = model.score_subjects(p_idx, o_idx)
                    score_o = model.score_objects(s_idx, p_idx)
                    loss_total[s] += (cross_entropy(score_s, s_idx, reduction='sum') + cross_entropy(score_o, o_idx, reduction='sum')).item()

            # ==========================================
            # LOGGING
            # ==========================================
            print("\r" + 100*" ", end="")
            print("\rLoss: {:.0f} | {:.0f}   MRR: {:.3f} | {:.3f}   HITS@1: {:.3f} | {:.3f}   HITS@3: {:.3f} | {:.3f}   HITS@5: {:.3f} | {:.3f}   HITS@10: {:.3f} | {:.3f}".format(loss_total['train'], loss_total['valid'], metrics_dict['train']['MRR'], metrics_dict['valid']['MRR'], metrics_dict['train']['HITS@1'], metrics_dict['valid']['HITS@1'], metrics_dict['train']['HITS@3'], metrics_dict['valid']['HITS@3'], metrics_dict['train']['HITS@5'], metrics_dict['valid']['HITS@5'], metrics_dict['train']['HITS@10'], metrics_dict['valid']['HITS@10']))

            # for i, clause in enumerate(clauses):
            #     for j, w in enumerate(clause.weights):
            #         print("clause {}, matrix {}:".format(i,j))
            #         print(softmax(w, dim=1).cpu().detach().numpy())

            if logging:
                wandb.log({**metrics_dict, 'epoch_outer': e_outer})
                wandb.log({**loss_total, 'epoch_outer': e_outer})
                for i, clause in enumerate(clauses):
                    wandb.log({'clause_{}'.format(i): clause.visualize_weights(), 'epoch_outer': e_outer})