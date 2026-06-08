"""Execution template descriptors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ExecutionTemplate:
    """Reusable compute structure for compatible parameter groups."""

    id: str
    description: str
    max_workspace_bytes: int | None = None
