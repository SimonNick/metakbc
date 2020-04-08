# -*- coding: utf-8 -*-

import torch

from metakbc.regularizers import L1, F2, N3

import pytest


@pytest.mark.light
def test_regularizers_v1():
    regularizer_lst = [L1(), F2(), N3()]
    for regularizer in regularizer_lst:
        T = torch.rand(32, 64)
        assert [x for x in regularizer([T]).shape] == []
        assert [x for x in regularizer([T], dim=1).shape] == [32]


if __name__ == '__main__':
    pytest.main([__file__])

