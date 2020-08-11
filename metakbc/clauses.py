import torch

from metakbc.clause import LearnedClause
from metakbc.datasets import Dataset

from typing import List

def load_clauses(dataset: Dataset, method: str) -> List[LearnedClause]:

    if dataset.name in ["Toy_A=>B_16", "Toy_A=>B_1024"]:

        # A(X1, X2) => B(X1, X2)
        clause_func = lambda x1, x2, phi1, phi2: phi1(x1, x2) - phi2(x1, x2)
        clause = LearnedClause(n_variables=2, n_relations=2, clause_loss_func=clause_func, n_constraints=1, n_predicates=dataset.n_predicates, method=method)

        return [clause]

    if dataset.name in ["Toy_A=>B,C=>D_1024"]:

        # A(X1, X2) => B(X1, X2)
        clause_func = lambda x1, x2, phi1, phi2: phi1(x1, x2) - phi2(x1, x2)
        clause = LearnedClause(n_variables=2, n_relations=2, clause_loss_func=clause_func, n_constraints=2, n_predicates=dataset.n_predicates, method=method)

        return [clause]

    if dataset.name in ["Toy_A,B=>C_16", "Toy_A,B=>C_1024"]:

        # A(X1, X2), B(X2, X3) => C(X1, X3)
        clause_func = lambda x1, x2, x3, phi1, phi2, phi3: torch.min(phi1(x1, x2), phi2(x2, x3)) - phi3(x1, x3)
        clause = LearnedClause(n_variables=3, n_relations=3, clause_loss_func=clause_func, n_constraints=1, n_predicates=dataset.n_predicates, method=method)

        return [clause]

    if dataset.name in ["Toy_A,B=>C,D,E=>F_1024"]:

        # A(X1, X2), B(X2, X3) => C(X1, X3)
        clause_func = lambda x1, x2, x3, phi1, phi2, phi3: torch.min(phi1(x1, x2), phi2(x2, x3)) - phi3(x1, x3)
        clause = LearnedClause(n_variables=3, n_relations=3, clause_loss_func=clause_func, n_constraints=10, n_predicates=dataset.n_predicates, method=method)

        return [clause]

    if dataset.name in ["Toy_mixed"]:

        # A(X1, X2) => A(X2, X1)
        clause_func1 = lambda x1, x2, phi1: 100 * (phi1(x1, x2) - phi1(x2, x1))
        clause1 = LearnedClause(n_variables=2, n_relations=1, clause_loss_func=clause_func1, n_constraints=100, n_predicates=dataset.n_predicates, method=method)

        # A(X1, X2) => B(X1, X2)
        clause_func2 = lambda x1, x2, phi1, phi2: phi1(x1, x2) - phi2(x1, x2)
        clause2 = LearnedClause(n_variables=2, n_relations=2, clause_loss_func=clause_func2, n_constraints=200, n_predicates=dataset.n_predicates, method=method)

        # A(X1, X2), B(X2, X3) => C(X1, X3)
        clause_func3 = lambda x1, x2, x3, phi1, phi2, phi3: torch.min(phi1(x1, x2), phi2(x2, x3)) - phi3(x1, x3)
        clause3 = LearnedClause(n_variables=3, n_relations=3, clause_loss_func=clause_func3, n_constraints=200, n_predicates=dataset.n_predicates, method=method)

        return [clause1, clause2, clause3]

    if dataset.name == "countries":

        # A(X1, X2) => A(X2, X1)
        clause_func1 = lambda x1, x2, phi1: 100 * (phi1(x1, x2) - phi1(x2, x1))
        clause1 = LearnedClause(n_variables=2, n_relations=1, clause_loss_func=clause_func1, n_constraints=100, n_predicates=dataset.n_predicates, method=method)

        # A(X1, X2), A(X2, X3) => A(X1, X3)
        clause_func2 = lambda x1, x2, x3, phi1: torch.min(phi1(x1, x2), phi1(x2, x3)) - phi1(x1, x3)
        clause2 = LearnedClause(n_variables=3, n_relations=1, clause_loss_func=clause_func2, n_constraints=200, n_predicates=dataset.n_predicates, method=method)

        return [clause1, clause2]

    if dataset.name == "nations":

        # A(X1, X2) => B(X1, X2)
        clause_func1 = lambda x1, x2, phi1, phi2: phi1(x1, x2) - phi2(x2, x1)
        clause1 = LearnedClause(n_variables=2, n_relations=2, clause_loss_func=clause_func1, n_constraints=100, n_predicates=dataset.n_predicates, method=method)

        # A(X1, X2), A(X2, X3) => A(X1, X3)
        clause_func2 = lambda x1, x2, x3, phi1: torch.min(phi1(x1, x2), phi1(x2, x3)) - phi1(x1, x3)
        clause2 = LearnedClause(n_variables=3, n_relations=1, clause_loss_func=clause_func2, n_constraints=200, n_predicates=dataset.n_predicates, method=method)

        return [clause1, clause2]

    if dataset.name == "umls":

        # A(X1, X2) => B(X1, X2)
        clause_func1 = lambda x1, x2, phi1, phi2: phi1(x1, x2) - phi2(x2, x1)
        clause1 = LearnedClause(n_variables=2, n_relations=2, clause_loss_func=clause_func1, n_constraints=100, n_predicates=dataset.n_predicates, method=method)

        # A(X1, X2), A(X2, X3) => A(X1, X3)
        clause_func2 = lambda x1, x2, x3, phi1: torch.min(phi1(x1, x2), phi1(x2, x3)) - phi1(x1, x3)
        clause2 = LearnedClause(n_variables=3, n_relations=1, clause_loss_func=clause_func2, n_constraints=200, n_predicates=dataset.n_predicates, method=method)

        return [clause1, clause2]

    else:
        error("Not implemented")