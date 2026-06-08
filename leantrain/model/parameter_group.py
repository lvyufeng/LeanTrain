"""Parameter grouping for contiguous host/device layouts."""

from __future__ import annotations

from dataclasses import dataclass, field

from leantrain.memory.object import TensorObject


@dataclass(slots=True)
class ParameterGroup:
    """A group of parameters that should move and execute together."""

    name: str
    tensors: list[TensorObject] = field(default_factory=list)
    structure_id: str | None = None

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)
