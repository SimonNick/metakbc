import torch

from metakbc.clause import LearnedClause
from metakbc.datasets import Dataset

from typing import List

def load_clauses(dataset: Dataset) -> List[LearnedClause]:

    if dataset.name == "Toy_A,B=>C":

        # A(X1, X2), B(X2, X3) => C(X1, X3)
        clause_func = lambda x1, x2, x3, phi1, phi2, phi3: torch.min(phi1(x1, x2), phi2(x2, x3)) - phi3(x1, x3)
        clause = LearnedClause(n_variables=3, n_relations=3, clause_loss_func=clause_func, n_constraints=1, n_predicates=dataset.n_predicates)

        return [clause]

    else:
        error("Not implemented")