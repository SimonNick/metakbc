# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch import nn

from metakbc.regularizers import N3

import pytest


@pytest.mark.light
def test_regularizers_v1():
    regularizer = N3()

    T = torch.rand(32, 64)

    print(T.shape)

    R = regularizer([T])

    print(R.shape)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_regularizers_v1()
