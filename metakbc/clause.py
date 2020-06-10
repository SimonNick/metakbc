# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import softmax

from metakbc.models import BaseModel

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np

from typing import Tuple, Optional, List, Callable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearnedClause(torch.nn.Module):

    def __init__(self, n_variables: int, n_relations: int, clause_loss_func: Callable, n_constraints: int, n_predicates: int) -> None:
        '''
        Represents a learned clause of a given shape.

        Args:
            n_variables: The number of variables that are involved in the clause
            n_relations: The number of relations / predicates that are involved in the clause
            clause_loss_func:
                The function used to calculate the score of the clause which is the difference between the score of the body and head of the clause.
                It has signature: loss(x1, x2, ..., xN, phi1, phi2, ..., phiM) where x1, ..., xN are the N variables involved in the clause and phi1, ..., phiM are the
                M scoring functions of signature phi_k(X,Y) used to calculate the score of the k-th relation evaluated on X and Y.
            n_constraints: The number of clauses of the given shape that should be learned
            n_predicates: The number of total predicates in the dataset

        Example:
            The clause (Head :- Body) of the shape
                B(X1, X2) :- A(X1, X2)
            has 2 variables (X1 and X2), two relations (A and B) and can be represented by the loss function
                Loss(X1, X2, Phi_A, Phi_B) = Phi_A(X1, X2) - Phi_B(X1, X2)
            where Phi_A and Phi_B are the scores of relation A and B, respectively.
            
            Thus, the above example means that n_variables=2, n_relations=2 and clause_loss_func=lambda x1, x2, phi_a, phi_b: phi_a(x1, x2) - phi_b(x1, x2).
            The user can specify how many clauses of this shape should be learned through n_constraints. The number of total predicates available in the dataset
            has to be specified through n_predicates.
        '''
        super().__init__()

        self.n_variables = n_variables
        self.n_relations = n_relations
        self.clause_loss_func = clause_loss_func
        self.n_constraints = n_constraints
        self.n_predicates = n_predicates

        self.weights = torch.nn.ParameterList([Parameter(torch.empty((n_constraints, n_predicates)).normal_(0, 1e-3)) for _ in range(n_relations)]).to(device)


    def inconsistency_loss(self, model: BaseModel, *variables, relu: bool = True) -> Tensor:
        '''
        Calculates the inconsistency loss of the clause. If several constraints are learned it returns the sum of inconsistency losses.

        Args:
            model: The model used for the calculation of the scores
            *variables: The variables x1, ..., xN used for evaluations of the clause
            relu: Whether or not to pass the inconsistency losses through a ReLU function before summing them
        Returns:
            Sum of the inconsistency losses
        '''

        # construct the predicate embeddings using a weighted sum over all predicates
        predicate_embeddings = [softmax(self.weights[i], dim=1) @ model.emb_p for i in range(self.n_relations)]

        # create the phi_k functions used to calculate the loss of the k-th relation
        phis = [lambda x, y, i=i: model._scoring_func(x, predicate_embeddings[i], y) for i in range(self.n_relations)]

        if relu:
            return torch.sum(torch.nn.functional.relu(self.clause_loss_func(*variables, *phis)))
        return torch.sum(self.clause_loss_func(*variables, *phis))


    def visualize_weights(self):
        fig = plt.figure()
        for i in range(self.n_relations):
            softmax_values = softmax(self.weights[i], dim=1).cpu().detach().numpy()
            ax = fig.add_subplot(1, self.n_relations, i+1)
            im = ax.matshow(softmax_values, cmap=matplotlib.cm.Greys_r, vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(softmax_values):
                ax.text(j, i, '{:0.2f}'.format(v), ha='center', va='center')
        return fig