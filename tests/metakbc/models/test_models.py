# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn

from metakbc.models import DistMult, ComplEx

import pytest


@pytest.mark.light
def test_distmult_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(128):
        with torch.no_grad():
            entity_embeddings = nn.Embedding(nb_entities, embedding_size, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size, sparse=True)

            entity_embeddings.weight.data *= init_size
            predicate_embeddings.weight.data *= init_size

            model = DistMult(entity_embeddings)

            xs = torch.from_numpy(rs.randint(nb_entities, size=32))
            xp = torch.from_numpy(rs.randint(nb_predicates, size=32))
            xo = torch.from_numpy(rs.randint(nb_entities, size=32))

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores = model.forward(xp_emb, xs_emb, xo_emb)
            inf = model.score(xp_emb, xs_emb, xo_emb)

            scores_sp, scores_po = scores

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            for i in range(xs.shape[0]):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-5, atol=1e-5)


@pytest.mark.light
def test_complex_v1():
    nb_entities = 10
    nb_predicates = 5
    embedding_size = 10

    init_size = 1.0

    rs = np.random.RandomState(0)

    for _ in range(128):
        with torch.no_grad():
            entity_embeddings = nn.Embedding(nb_entities, embedding_size * 2, sparse=True)
            predicate_embeddings = nn.Embedding(nb_predicates, embedding_size * 2, sparse=True)

            entity_embeddings.weight.data *= init_size
            predicate_embeddings.weight.data *= init_size

            model = ComplEx(entity_embeddings)

            xs = torch.from_numpy(rs.randint(nb_entities, size=32))
            xp = torch.from_numpy(rs.randint(nb_predicates, size=32))
            xo = torch.from_numpy(rs.randint(nb_entities, size=32))

            xs_emb = entity_embeddings(xs)
            xp_emb = predicate_embeddings(xp)
            xo_emb = entity_embeddings(xo)

            scores = model.forward(xp_emb, xs_emb, xo_emb)
            inf = model.score(xp_emb, xs_emb, xo_emb)

            scores_sp, scores_po = scores

            inf = inf.cpu().numpy()
            scores_sp = scores_sp.cpu().numpy()
            scores_po = scores_po.cpu().numpy()

            for i in range(xs.shape[0]):
                np.testing.assert_allclose(inf[i], scores_sp[i, xo[i]], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(inf[i], scores_po[i, xs[i]], rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
