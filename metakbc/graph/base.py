# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

from typing import Tuple, List, Dict, Set, Optional, Union

import logging

logger = logging.getLogger(__name__)


def to_networkx(triples: List[Tuple[str, str, str]],
                entity_to_idx: Dict[str, int],
                predicate_to_idx: Dict[str, int],
                predicates: Optional[Set[str]] = None,
                is_multidigraph: bool = False) -> nx.DiGraph:

    _triples = triples if predicates is None else [(s, p, o) for s, p, o in triples if p in predicates]

    G = nx.MultiDiGraph() if is_multidigraph else nx.DiGraph()

    entities = sorted({s for (s, _, _) in triples} | {o for (_, _, o) in triples})
    G.add_nodes_from([entity_to_idx[e] for e in entities])

    if is_multidigraph:
        G.add_edges_from([(entity_to_idx[s], entity_to_idx[o], {'p': predicate_to_idx[p]}) for s, p, o in _triples])
    else:
        edge_lst = sorted({(entity_to_idx[s], entity_to_idx[o]) for s, p, o in _triples})
        G.add_edges_from(edge_lst)

    return G


def get_features(triples: List[Tuple[str, str, str]],
                 entity_to_idx: Dict[str, int],
                 predicate_to_idx: Dict[str, int],
                 predicates: Optional[Set[str]] = None):
    G = to_networkx(triples, entity_to_idx, predicate_to_idx, predicates, is_multidigraph=False)
    mG = to_networkx(triples, entity_to_idx, predicate_to_idx, predicates, is_multidigraph=True)

    pagerank = nx.pagerank(G)
    hubs, authorities = nx.hits(G)

    degree = mG.degree()
    in_degree = mG.in_degree()
    out_degree = mG.out_degree()

    nb_entities = max(v for _, v in entity_to_idx.items()) + 1
    nb_features = 6

    res = np.zeros(shape=(nb_entities, nb_features))

    for k, v in degree:
        res[k, 0] = v

    for k, v in in_degree:
        res[k, 1] = v

    for k, v in out_degree:
        res[k, 1] = v

    for k, v in pagerank.items():
        res[k, 3] = v

    for k, v in hubs.items():
        res[k, 4] = v

    for k, v in authorities.items():
        res[k, 5] = v

    return res


if __name__ == '__main__':
    triples = [
        ('a', 'p', 'b'),
        ('a', 'p', 'c'),
        ('b', 'q', 'd')
    ]

    entity_to_idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    predicate_to_idx = {'p': 0, 'q': 1}

    features = get_features(triples, entity_to_idx, predicate_to_idx)

    print(features)
