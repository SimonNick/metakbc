# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import normalize, softmax, cross_entropy
from torch.nn.utils import clip_grad_value_

from metakbc.models import DistMult, ComplEx
from metakbc.datasets import Dataset
from metakbc.evaluation import evaluate, build_filters
from metakbc.regularizers import N3
from metakbc.adversary import Adversary
from metakbc.clause import LearnedClause
from metakbc.clauses import load_clauses
from metakbc.visualization import visualize_embeddings, visualize_clause, PCA_entity_embeddings

import numpy as np
import random

import higher

from typing import List

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42


def learn(dataset_str: str,
          model_str: str,
          #
          method: str,
          rule_method: str,
          adv_method: str,
          lam: float,
          learn_lam: bool,
          #
          optimizer_str: str,
          meta_optimizer_str: str,
          adv_optimizer_str: str,
          #
          lr: float,
          meta_lr: float,
          adv_lr: float,
          #
          n_epochs_outer: int,
          n_epochs_inner: int, # only for offline metalearning
          n_batches_train: int, # only for online metalearning
          n_epochs_adv: int,
          n_valid: int,
          #
          rank: int,
          batch_size: int,
          reg_weight: float,
          #
          seed: int,
          #
          print_clauses: bool,
          logging: bool) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    lam = torch.Tensor([lam]).to(device)
    if learn_lam:
        lam.requires_grad = True
    regularizer = N3()
    minimum_lambda = 1e-6
    dataset = Dataset(dataset_str)
    filters = build_filters(dataset)
    clauses = load_clauses(dataset, rule_method)
    adversary = Adversary(clauses).to(device)
    adversarial_embeddings = None
    adversarial_examples = None

    meta_optim = {
        "SGD": lambda: torch.optim.SGD([*adversary.parameters(), lam], lr=meta_lr, momentum=0.9),
        "Adagrad": lambda: torch.optim.Adagrad([*adversary.parameters(), lam], lr=meta_lr),
    }[meta_optimizer_str]()


    def train_model(fmodel, diff_optim, batch, adversarial_embeddings):

        batch = batch.to(device)
        s_idx, p_idx, o_idx = batch[:, 0], batch[:, 1], batch[:, 2]

        score_s = fmodel.score_subjects(p_idx, o_idx)
        score_o = fmodel.score_objects(s_idx, p_idx)

        loss_fact = cross_entropy(score_s, s_idx) + cross_entropy(score_o, o_idx)
        loss_inc = adversary(fmodel, adversarial_embeddings)
        loss_reg = regularizer(fmodel.factors(s_idx, p_idx, o_idx))
        loss_total = loss_fact + lam * loss_inc + reg_weight * loss_reg

        diff_optim.step(loss_total)


    def meta_optimize(fmodel):

        loss_valid = 0
        for batch in dataset.get_batches('valid', batch_size, shuffle=True):

            batch = batch.to(device)
            s_idx, p_idx, o_idx = batch[:, 0], batch[:, 1], batch[:, 2]

            score_s = fmodel.score_subjects(p_idx, o_idx)
            score_o = fmodel.score_objects(s_idx, p_idx)
            loss_valid += cross_entropy(score_s, s_idx) + cross_entropy(score_o, o_idx)

        meta_optim.zero_grad() 
        loss_valid.backward() 
        # clip_grad_value_(adversary.parameters(), 1e-4)
        meta_optim.step()

        if learn_lam:
            # prevent lambda from becoming negative
            with torch.no_grad():
                lam[0] = torch.clamp(lam, minimum_lambda) 


    def create_model():

        model = {
            "DistMult": lambda: DistMult(size=dataset.get_shape(), rank=rank).to(device),
            "ComplEx": lambda: ComplEx(size=dataset.get_shape(), rank=rank).to(device),
        }[model_str]()

        optim = {
            "SGD": lambda: torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
            "Adagrad": lambda: torch.optim.Adagrad(model.parameters(), lr=lr),
        }[optimizer_str]()

        return model, optim


    if method == "online":
        model, optim = create_model()
    else:
        model = None

    # ==========================================
    # TRAINING LOOPS
    # ==========================================
    for e_outer in range(n_epochs_outer):

        if method == "offline": # offline metalearning

            # create new model every time
            model, optim = create_model()

            with higher.innerloop_ctx(model, optim, copy_initial_weights=True) as (fmodel, diff_optim):

                for e_inner in range(n_epochs_inner):

                    # ADVERSARIAL TRAINING
                    if adv_method == 'embedding':
                        adversarial_embeddings = adversary.generate_adversarial_embeddings(model, n_epochs_adv, adv_optimizer_str, adv_lr, init_random=True, adversarial_embeddings=adversarial_embeddings)
                        
                        # create a copy that is kept in the memory (necessary for meta-optimization)
                        adversarial_embeddings_copy = [variables.clone().detach() for variables in adversarial_embeddings]
                    else:
                        adversarial_examples = adversary.generate_adversarial_examples(dataset, model, n_epochs_adv, adversarial_examples=adversarial_examples)

                    # TRAINING
                    for k, batch in enumerate(dataset.get_batches('train', batch_size, shuffle=True)):
                        print("\rOuter epoch: {:4}, inner epoch: {:4}, batch: {:4}".format(e_outer+1, e_inner+1, k+1), end="")

                        if adv_method == 'entity':
                            # create a copy that is kept in the memory (necessary for meta-optimization)
                            adversarial_embeddings_copy = [model.emb_so[examples].clone().detach() for examples in adversarial_examples]

                        train_model(fmodel, diff_optim, batch, adversarial_embeddings_copy)

                    # copy learned weights to original model
                    with torch.no_grad():
                        model.emb_so.copy_(fmodel.emb_so)
                        model.emb_p.copy_(fmodel.emb_p)
                    
                # META-OPTIMIZATION
                meta_optimize(fmodel)
        
        else: # online metalearning

            print("\rOuter epoch: {:4}".format(e_outer+1), end="")

            batches = dataset.get_batches('train', batch_size, shuffle=True)
            epoch_complete = False
            while not epoch_complete:

                # ADVERSARIAL TRAINING
                if adv_method == 'embedding':
                    adversarial_embeddings = adversary.generate_adversarial_embeddings(model, n_epochs_adv, adv_optimizer_str, adv_lr, init_random=True, adversarial_embeddings=adversarial_embeddings)
                else:
                    adversarial_examples = adversary.generate_adversarial_examples(dataset, model, n_epochs_adv, adversarial_examples=adversarial_examples)

                with higher.innerloop_ctx(model, optim, copy_initial_weights=True) as (fmodel, diff_optim):

                    # TRAINING
                    for k, batch in enumerate(batches):
                        if k == n_batches_train: break

                        if adv_method == 'entity':
                            # create a copy that is kept in the memory (necessary for meta-optimization)
                            adversarial_embeddings = [model.emb_so[examples].clone().detach() for examples in adversarial_examples]

                        train_model(fmodel, diff_optim, batch, adversarial_embeddings)
                    else:
                        epoch_complete = True

                    # META LEARNING
                    meta_optimize(fmodel)

                    # copy learned weights to original model
                    with torch.no_grad():
                        model.emb_so.copy_(fmodel.emb_so)
                        model.emb_p.copy_(fmodel.emb_p)


        if (n_valid == 0 and e_outer == n_epochs_outer -1 ) or (n_valid is not 0 and e_outer % n_valid == 0):

            # ==========================================
            # EVALUATION
            # ==========================================
            metrics_dict = evaluate(dataset, dataset.splits, model, batch_size, filters)
            loss_total = {s: 0 for s in dataset.splits}
            for s in dataset.splits:
                for batch in dataset.get_batches(s, batch_size):
                    batch = batch.to(device)
                    s_idx, p_idx, o_idx = batch[:, 0], batch[:, 1], batch[:, 2]
                    score_s = model.score_subjects(p_idx, o_idx)
                    score_o = model.score_objects(s_idx, p_idx)
                    loss_total[s] += (cross_entropy(score_s, s_idx, reduction='sum') + cross_entropy(score_o, o_idx, reduction='sum')).item()
                loss_total[s] /= len(dataset.get_examples(s))

            # ==========================================
            # LOGGING
            # ==========================================
            print("\r" + 100*" ", end="")
            print("\rLoss: {:.2f} | {:.2f} | {:.2f}   MRR: {:.3f} | {:.3f} | {:.3f}   Lambda: {:.3f}".format(loss_total['train'], loss_total['valid'], loss_total['test'], metrics_dict['train']['MRR'], metrics_dict['valid']['MRR'], metrics_dict['test']['MRR'], lam.item()))

            # print the weights of all clauses
            if print_clauses:
                for i, clause in enumerate(clauses):
                    if rule_method == 'attention':
                        print(softmax(clause.weights, dim=2).cpu().detach().numpy())
                    else:
                        print(torch.sigmoid(clause.weights).cpu().detach().numpy())

            if logging:
                wandb.log({**metrics_dict, 'epoch_outer': e_outer})
                wandb.log({**loss_total, 'epoch_outer': e_outer})
                wandb.log({"lambda": lam.item(), 'epoch_outer': e_outer})
    
    if logging:
        for i, clause in enumerate(clauses):
            wandb.log({'clause_{}'.format(i): visualize_clause(clause), 'epoch_outer': e_outer})
        wandb.log({'embeddings': wandb.Image(visualize_embeddings(model)), 'epoch_outer': e_outer})
        wandb.log({'PCA embeddings': wandb.Image(PCA_entity_embeddings(model, dataset)), 'epoch_outer': e_outer})

    return metrics_dict, loss_total