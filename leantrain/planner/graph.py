"""Task graph definitions for planned training execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskKind(str, Enum):
    H2D = "h2d"
    D2H = "d2h"
    COMPUTE_FORWARD = "compute_forward"
    COMPUTE_BACKWARD = "compute_backward"
    RECOMPUTE = "recompute"
    ACCUMULATE_GRADIENT = "accumulate_gradient"
    OPTIMIZER_STEP = "optimizer_step"
    EVICT = "evict"


@dataclass(slots=True)
class Task:
    """A scheduled unit of copy, compute, memory, or optimizer work."""

    id: str
    kind: TaskKind
    depends_on: list[str] = field(default_factory=list)
    estimated_bytes: int | None = None
    estimated_seconds: float | None = None


@dataclass(slots=True)
class TaskGraph:
    """A dependency graph emitted by a planner and consumed by a runtime."""

    tasks: list[Task] = field(default_factory=list)
