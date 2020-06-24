import torch

from metakbc.clause import LearnedClause
from metakbc.datasets import Dataset

from typing import List

def load_clauses(dataset: Dataset) -> List[LearnedClause]:

    if dataset.name in ["Toy_A,B=>C_16", "Toy_A,B=>C_1024"]:

        # A(X1, X2), B(X2, X3) => C(X1, X3)
        clause_func = lambda x1, x2, x3, phi1, phi2, phi3: torch.min(phi1(x1, x2), phi2(x2, x3)) - phi3(x1, x3)
        clause = LearnedClause(n_variables=3, n_relations=3, clause_loss_func=clause_func, n_constraints=1, n_predicates=dataset.n_predicates)

        return [clause]

    if dataset.name == "Toy_A,B=>C,D,E=>F_1024":

        # A(X1, X2), B(X2, X3) => C(X1, X3)
        clause_func = lambda x1, x2, x3, phi1, phi2, phi3: torch.min(phi1(x1, x2), phi2(x2, x3)) - phi3(x1, x3)
        clause = LearnedClause(n_variables=3, n_relations=3, clause_loss_func=clause_func, n_constraints=2, n_predicates=dataset.n_predicates)

        return [clause]

    if dataset.name in ["Toy_A=>B_10", "Toy_A=>B_1024"]:

        # A(X1, X2) => B(X1, X2)
        clause_func = lambda x1, x2, phi1, phi2: phi1(x1, x2) - phi2(x1, x2)
        clause = LearnedClause(n_variables=2, n_relations=2, clause_loss_func=clause_func, n_constraints=1, n_predicates=dataset.n_predicates)

        return [clause]

    if dataset.name in ["Toy_A=>B,C=>D_5000", "Toy_A=>B,C=>D_1024"]:

        # A(X1, X2) => B(X1, X2)
        clause_func = lambda x1, x2, phi1, phi2: phi1(x1, x2) - phi2(x1, x2)
        clause = LearnedClause(n_variables=2, n_relations=2, clause_loss_func=clause_func, n_constraints=2, n_predicates=dataset.n_predicates)

        return [clause]

    else:
        error("Not implemented")

        # A(X1, X2) => A(X2, X1)
        # clause2_func = lambda x1, x2, phi1: phi1(x1, x2) - phi1(x2, x1)
        # clause2 = LearnedClause(n_variables=2, n_relations=1, clause_loss_func=clause2_func, n_constraints=10, n_predicates=dataset.n_predicates)