"""Local hardware probing.

The first implementation is intentionally conservative: it records host memory and
CUDA devices when PyTorch is installed, without requiring CUDA to be available.
Bandwidth microbenchmarks will be added after the basic project skeleton is in
place.
"""

from __future__ import annotations

import os

from leantrain.hardware.profile import GPUProfile, HardwareProfile


def _host_ram_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size * page_count)


def _probe_torch_gpus() -> list[GPUProfile]:
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    gpus: list[GPUProfile] = []
    for device_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_id)
        major, _minor = props.major, props.minor
        # Ampere and newer NVIDIA GPUs expose BF16 tensor cores. Keep this as a
        # capability hint; kernel availability still needs runtime validation.
        supports_bf16 = major >= 8
        gpus.append(
            GPUProfile(
                id=device_id,
                name=props.name,
                vram_bytes=int(props.total_memory),
                supports_bf16=supports_bf16,
            )
        )
    return gpus


def probe_hardware() -> HardwareProfile:
    """Return a best-effort hardware profile for the local machine."""

    return HardwareProfile(
        host_ram_bytes=_host_ram_bytes(),
        gpus=_probe_torch_gpus(),
    )
