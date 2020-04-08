# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor

from typing import List

import logging

logger = logging.getLogger(__name__)


class AdaptiveRegularizer(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self,
                 features: Tensor) -> Tensor:
        raise NotImplementedError


class LinearRegularizer(AdaptiveRegularizer):
    def __init__(self, nb_features: int):
        super().__init__()
        self.projection = nn.Linear(nb_features, 1)

    def __call__(self,
                 features: Tensor) -> Tensor:
        res = torch.sigmoid(self.projection(features))
        return res
