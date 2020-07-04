# -*- coding: utf-8 -*-

import torch
from torch import Tensor, LongTensor
from torch.nn import Parameter
from torch.nn.functional import softmax

from metakbc.models import BaseModel
from metakbc.datasets import Dataset
from metakbc.clause import LearnedClause

from typing import Tuple, Optional, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Adversary(torch.nn.Module):

    def __init__(self, clauses: List[LearnedClause]) -> None:
        super().__init__()

        self.clauses = clauses
        for i, clause in enumerate(self.clauses):
            for j, p in enumerate(clause.parameters()):
                self.register_parameter("clause {}, parameter {}".format(i,j), p)


    def generate_adversarial_examples(self, dataset: Dataset, model: BaseModel, n_epochs_adv: int, adversarial_examples: List[LongTensor] = None) -> List[LongTensor]:

        if adversarial_examples == None:
            adversarial_examples = []
            for clause in self.clauses:
                adversarial_examples += [torch.randint(0, dataset.n_entities, (clause.n_variables, clause.n_constraints))]

        adversarial_embeddings = []
        for examples in adversarial_examples:
            adversarial_embeddings += [model.emb_so[examples].detach()]

        for clause, examples, embeddings in zip(self.clauses, adversarial_examples, adversarial_embeddings):

            loss_inc = clause.inconsistency_loss(model, embeddings, relu=False, sum_loss=False)

            for _ in range(n_epochs_adv):

                new_examples = torch.randint(0, dataset.n_entities, (clause.n_variables, clause.n_constraints))
                new_variables = model.emb_so[new_examples]
                new_loss_inc = clause.inconsistency_loss(model, new_variables, relu=False, sum_loss=False)

                mask = new_loss_inc > loss_inc
                loss_inc[mask] = new_loss_inc[mask]
                examples[:, mask] = new_examples[:, mask]
                embeddings[:, mask] = new_variables[:, mask]
            
        return adversarial_examples


    def generate_adversarial_embeddings(self, model: BaseModel, n_epochs_adv: int, optimizer_str: str, lr: float, init_random: bool = True, adversarial_embeddings: List[Tensor] = None) -> List[Tensor]:

        # initialise list that holds adversarial examples for all clauses
        if adversarial_embeddings == None:
            adversarial_embeddings = []
            for clause in self.clauses:

                # either initialise variables at random or use existing entity embeddings
                if init_random:
                    variables = torch.empty((clause.n_variables, clause.n_constraints, model.rank)).normal_().to(device)
                else:
                    rand_indices = torch.randint(0, model.emb_so.shape[0], (clause.n_variables, clause.n_constraints))
                    variables = model.emb_so[rand_indices].clone().detach()

                variables.requires_grad = True
                adversarial_embeddings += [variables]


        for clause, variables in zip(self.clauses, adversarial_embeddings):

            optim = {
                "SGD": lambda: torch.optim.SGD([variables], lr=lr, momentum=0.9),
                "Adagrad": lambda: torch.optim.Adagrad([variables], lr=lr),
            }[optimizer_str]()

            for _ in range(n_epochs_adv):

                loss_inc = clause.inconsistency_loss(model, variables, relu=False)
                loss_inc *= -1 # perform gradient ascent

                optim.zero_grad()
                loss_inc.backward()
                optim.step()

                # project variables onto unit sphere
                with torch.no_grad():
                    variables /= variables.norm(p=2, dim=2).view(clause.n_variables, clause.n_constraints, 1).detach()

        return adversarial_embeddings


    def inconsistency_loss(self, model: BaseModel, adversarial_embeddings: List[Tensor]) -> Tensor:
        loss_inc = 0
        for clause, variables in zip(self.clauses, adversarial_embeddings):
            loss_inc += clause.inconsistency_loss(model, variables)
        return loss_inc


    def forward(self, *args):
        return self.inconsistency_loss(*args)