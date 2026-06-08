"""Logical memory objects tracked by LeanTrain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TensorRole(str, Enum):
    PARAMETER = "parameter"
    GRADIENT = "gradient"
    OPTIMIZER_STATE = "optimizer_state"
    ACTIVATION = "activation"
    TEMPORARY = "temporary"


@dataclass(slots=True, frozen=True)
class TensorObject:
    """A logical tensor independent of its physical residency."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    role: TensorRole

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total
