# -*- coding: utf-8 -*-

import networkx as nx

from typing import Tuple, List, Dict, Set, Optional

import logging

logger = logging.getLogger(__name__)


def to_networkx(triples: List[Tuple[str, str, str]],
                entity_to_idx: Dict[str, int],
                predicate_to_idx: Dict[str, int],
                is_multidigraph: bool = False,
                predicates: Optional[Set[str]] = None) -> nx.Graph:
    _triples = triples if predicates is None else [(s, p, o) for s, p, o in triples if p in predicates]
    entities = sorted({s for (s, _, _) in _triples} | {o for (_, _, o) in _triples})
    G = nx.MultiDiGraph() if is_multidigraph else nx.DiGraph()
    G.add_nodes_from([entity_to_idx[e] for e in entities])
    if is_multidigraph:
        G.add_edges_from([(entity_to_idx[s], entity_to_idx[o], {'p': predicate_to_idx[p]}) for s, p, o in _triples])
    else:
        edge_lst = sorted({(entity_to_idx[s], entity_to_idx[o]) for s, p, o in _triples})
        G.add_edges_from(edge_lst)
    return G


if __name__ == '__main__':
    triples = [
        ('a', 'p', 'b'),
        ('a', 'p', 'c'),
        ('b', 'q', 'd')
    ]

    G = to_networkx(triples, {'a': 0, 'b': 1, 'c': 2, 'd': 3}, {'p': 0, 'q': 1}, is_multidigraph=False)

    print(G.edges(data=True))
    print(G.nodes)

    print(nx.pagerank(G))
    print(nx.hits(G))
    print(nx.degree(G))
    print(G.in_degree())
    print(G.out_degree())
