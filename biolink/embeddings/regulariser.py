#extension from KBC Facebook

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class F2(Regularizer):
    def __init__(self, weight: float):
        super(F2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(f ** 2)
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self, weight: float, tucker_weight=None):
        super(N3, self).__init__()
        self.weight = weight
        self.tucker_weight = tucker_weight

    def forward(self, factors):
        norm = 0
        for i,f in enumerate(factors):
            if not (self.tucker_weight is None) and (i == 3):
                norm += self.tucker_weight * torch.sum(
                    torch.abs(f) ** 3)
            else:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 3
                )
            # norm += self.weight * torch.sum(
            #         torch.abs(f) ** 3
            #     )

        return norm / factors[0].shape[0]
