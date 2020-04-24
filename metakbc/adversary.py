
# -*- coding: utf-8 -*-

import torch
from torch import LongTensor, Tensor
from torch.nn import Parameter
from torch.nn.functional import softmax

from metakbc.models import BaseModel

from typing import Tuple, Optional, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LearnedClause(torch.nn.Module):

    def __init__(self, n_variables: int, n_relations: int, n_predicates: int, n_constraints: int, loss_func) -> None:
        super().__init__()

        self.n_variables = n_variables # unused
        self.n_relations = n_relations
        self.n_predicates = n_predicates
        self.n_constraints = n_constraints
        self.loss_func = loss_func

        self.weights = torch.nn.ParameterList([Parameter(torch.empty((n_constraints, n_predicates)).normal_(0, 1e-3)) for _ in range(n_relations)]).to(device)

    def inconsistency_loss(self, model, temperature, *variables, relu=True) -> Tensor:
        predicate_emb = [softmax(self.weights[i] / temperature, dim=1) @ model.emb_p for i in range(self.n_relations)]
        phis = [lambda x, y, i=i: model._scoring_func(x, predicate_emb[i], y) for i in range(self.n_relations)]
        if relu:
            return torch.sum(torch.nn.functional.relu(self.loss_func(*variables, *phis)))
        return torch.sum(self.loss_func(*variables, *phis))


class Adversary(torch.nn.Module):

    def __init__(self, n_constraints: int, n_predicates: int, rank: int, n_epochs_adv: int) -> None:
        super().__init__()

        self.temperature = 1.

        clause1_func = lambda x1, x2, phi1, phi2: phi1(x1, x2) - phi2(x1, x2)
        clause2_func = lambda x1, x2, phi1: phi1(x1, x2) - phi1(x2, x1)
        clause1 = LearnedClause(2, 2, n_predicates, 4, clause1_func)
        clause2 = LearnedClause(2, 1, n_predicates, 10, clause2_func)

        # self.clauses = [clause1, clause2]
        self.clauses = [clause1]
        for i, clause in enumerate(self.clauses):
            for j, p in enumerate(clause.parameters()):
                self.register_parameter("clause {}, parameter {}".format(i,j), p)

        self.n_constraints = n_constraints
        self.n_predicates = n_predicates
        self.n_epochs_adv = n_epochs_adv
        self.rank = rank

    def generate_adversarial_examples(self, model: BaseModel, adversarial_examples=None) -> Tuple:
        
        if adversarial_examples == None:
            adversarial_examples = []
            for clause in self.clauses:
                variables = []
                for _ in range(clause.n_variables):
                    variable = torch.empty((clause.n_constraints, 2 * self.rank)).normal_().to(device)
                    variable.requires_grad = True
                    variables += [variable]
                adversarial_examples += [variables]

        new_adversarial_examples = []
        for clause, variables in zip(self.clauses, adversarial_examples):

            optim_adv = torch.optim.Adagrad(variables, lr=0.5)

            for e_adv in range(self.n_epochs_adv):

                loss_inc = clause.inconsistency_loss(model, self.temperature, *variables, relu=False)
                loss_inc *= -1 # perform gradient ascent

                optim_adv.zero_grad()
                loss_inc.backward()
                optim_adv.step()

                for variable in variables:
                    with torch.no_grad():
                        variable /= variable.norm(p=2, dim=1).view(-1, 1).detach()

            new_adversarial_examples += [variables]

        return new_adversarial_examples


    def eval(self, model: BaseModel, adversarial_examples) -> Tensor:
        loss_inc = 0
        for clause, variables in zip(self.clauses, adversarial_examples):
            loss_inc += clause.inconsistency_loss(model, self.temperature, *variables)
        return loss_inc

    def forward(self, *args):
        return self.eval(*args)