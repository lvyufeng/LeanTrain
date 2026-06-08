"""Memory tier and residency state definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryTier(str, Enum):
    GPU_VRAM = "gpu_vram"
    PINNED_HOST_RAM = "pinned_host_ram"
    PAGEABLE_HOST_RAM = "pageable_host_ram"
    NVME = "nvme"


@dataclass(slots=True, frozen=True)
class Residency:
    """Where a tensor currently has a valid physical copy."""

    tier: MemoryTier
    device_id: int | None = None
    valid: bool = True
    authoritative: bool = False
