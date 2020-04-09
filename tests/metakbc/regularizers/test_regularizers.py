# -*- coding: utf-8 -*-

import torch

from metakbc.regularizers import L1, F2, N3
from metakbc.regularizers.adaptive import LinearAdaptiveRegularizer
from metakbc.regularizers.adaptive import GatedLinearAdaptiveRegularizer
from metakbc.regularizers.adaptive import ConstantAdaptiveRegularizer

import pytest


@pytest.mark.light
def test_regularizers_v1():
    T = torch.rand(32, 64)
    regularizer_lst = [L1(), F2(), N3()]

    for regularizer in regularizer_lst:
        assert [x for x in regularizer([T]).shape] == []
        assert [x for x in regularizer([T], dim=1).shape] == [32]


@pytest.mark.light
def test_adaptive_regularizers_v1():
    T = torch.rand(32, 64)
    reg = N3()

    car = ConstantAdaptiveRegularizer(reg)
    lar = LinearAdaptiveRegularizer(reg, 64)
    glar = GatedLinearAdaptiveRegularizer(reg, 64)

    assert [x for x in car(T, T).shape] == []
    assert [x for x in lar(T, T).shape] == []
    assert [x for x in glar(T, T).shape] == []


if __name__ == '__main__':
    pytest.main([__file__])
    # test_adaptive_regularizers_v1()
