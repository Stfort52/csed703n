from abc import ABC, abstractmethod

from torch import LongTensor, Tensor, nn


class BasePE(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: LongTensor) -> Tensor: ...

    @property
    @abstractmethod
    def max_len(self) -> int | None: ...

    @property
    @abstractmethod
    def coupled(self) -> bool: ...
