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
    uG = G.to_undirected()

    mG = to_networkx(triples, entity_to_idx, predicate_to_idx, predicates, is_multidigraph=True)
    umG = mG.to_undirected()

    hubs, authorities = nx.hits(G)

    feature_lst = [
        mG.degree(),
        mG.in_degree(),
        mG.out_degree(),
        nx.pagerank(G),
        hubs,
        authorities,
        nx.degree_centrality(mG),
        nx.in_degree_centrality(mG),
        nx.out_degree_centrality(mG),
        nx.katz_centrality(G),
        nx.closeness_centrality(mG),
        nx.current_flow_closeness_centrality(umG),
        nx.betweenness_centrality(G),
        nx.current_flow_closeness_centrality(umG),
        nx.communicability_betweenness_centrality(uG)
    ]

    nb_entities = max(v for _, v in entity_to_idx.items()) + 1
    nb_features = len(feature_lst)

    res = np.zeros(shape=(nb_entities, nb_features))

    for i, f in enumerate(feature_lst):
        for k, v in (f.items() if isinstance(f, dict) else f):
            res[k, i] = v

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
