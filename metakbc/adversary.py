
# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import softmax

from metakbc.models import BaseModel
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


    def generate_adversarial_examples(self, model: BaseModel, n_epochs_adv: int, optimizer_str: str, lr: float, adversarial_examples: List[List[Tensor]] = None) -> List[List[Tensor]]:
        
        # initialise list that holds adversarial examples for all clauses
        if adversarial_examples == None:
            adversarial_examples = []
            for clause in self.clauses:
                variables = []
                for _ in range(clause.n_variables):
                    variable = torch.empty((clause.n_constraints, model.rank)).normal_().to(device)
                    variable.requires_grad = True
                    variables += [variable]
                adversarial_examples += [variables]

        for clause, variables in zip(self.clauses, adversarial_examples):

            optim = {
                "SGD": lambda: torch.optim.SGD(variables, lr=lr, momentum=0.9),
                "Adagrad": lambda: torch.optim.Adagrad(variables, lr=lr),
            }[optimizer_str]()

            for _ in range(n_epochs_adv):

                loss_inc = clause.inconsistency_loss(model, *variables, relu=False)
                loss_inc *= -1 # perform gradient ascent

                optim.zero_grad()
                loss_inc.backward()
                optim.step()

                # project variables onto unit sphere
                for variable in variables:
                    with torch.no_grad():
                        variable /= variable.norm(p=2, dim=1).view(-1, 1).detach()

        return adversarial_examples


    def inconsistency_loss(self, model: BaseModel, adversarial_examples: List[List[Tensor]]) -> Tensor:
        loss_inc = 0
        for clause, variables in zip(self.clauses, adversarial_examples):
            loss_inc += clause.inconsistency_loss(model, *variables)
        return loss_inc


    def forward(self, *args):
        return self.inconsistency_loss(*args)