"""Initial planning policy placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PlannerPolicy:
    """High-level knobs selected from hardware and model constraints."""

    dtype: str
    checkpoint_interval: int
    prefetch_depth: int
    gradient_slab_count: int
