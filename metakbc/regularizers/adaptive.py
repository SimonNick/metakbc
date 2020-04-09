# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor

from metakbc.regularizers import Regularizer

import logging

logger = logging.getLogger(__name__)


class AdaptiveRegularizer(nn.Module, ABC):
    def __init__(self,
                 regularizer: Regularizer) -> None:
        super().__init__()
        self.regularizer = regularizer

    @abstractmethod
    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def projection_(self):
        raise NotImplementedError


class ConstantAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self,
                 regularizer: Regularizer):
        super().__init__(regularizer)
        self.weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        norm_values = self.regularizer([factor], dim=1)
        return self.weight * torch.sum(norm_values)

    def projection_(self):
        self.weight.data.clamp_(0.0)


class LinearAdaptiveRegularizer(AdaptiveRegularizer):
    def __init__(self,
                 regularizer: Regularizer,
                 nb_features: int):
        super().__init__(regularizer)
        self.projection = nn.Linear(nb_features, 1)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        weight_values = torch.relu(self.projection(features)).view(-1)
        norm_values = self.regularizer([factor], dim=1)
        return torch.sum(weight_values * norm_values)

    def projection_(self):
        pass


class GatedLinearAdaptiveRegularizer(LinearAdaptiveRegularizer):
    def __init__(self,
                 regularizer: Regularizer,
                 nb_features: int):
        super().__init__(regularizer, nb_features)
        self.gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

    def __call__(self,
                 factor: Tensor,
                 features: Tensor) -> Tensor:
        res = super().__call__(factor, features)
        return self.gate * res

    def projection_(self):
        super().projection_()
        self.gate.data.clamp_(0)
