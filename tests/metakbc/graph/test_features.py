# -*- coding: utf-8 -*-

import os
import sys

from metakbc.training.data import Data
from metakbc.graph.base import get_features

import pytest

import logging

logger = logging.getLogger(__name__)


@pytest.mark.light
def test_features_v1():
    nb_features = None
    for root, dirs, files in os.walk('data/'):
        for file in files:
            if file.endswith('.tsv'):
                data = Data(train_path='data/wn18rr/train.tsv')
                features = get_features(data.train_triples, data.entity_to_idx, data.predicate_to_idx)
                if nb_features is None:
                    nb_features = features.shape[1]
                assert features.shape[1] == nb_features


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    pytest.main([__file__])
    # test_features_v1()
