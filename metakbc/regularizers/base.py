# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor

from typing import List

import logging

logger = logging.getLogger(__name__)


class Regularizer(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self,
                 factors: List[Tensor]):
        raise NotImplementedError


class F2(Regularizer):
    def __init__(self):
        super().__init__()

    def __call__(self,
                 factors: List[Tensor]):
        norm = 0
        for f in factors:
            norm += torch.sum(f ** 2)

        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self):
        super().__init__()

    def __call__(self,
                 factors: List[Tensor]):

        norm = 0
        for f in factors:
            norm += torch.sum(torch.abs(f) ** 3)

        return norm / factors[0].shape[0]
