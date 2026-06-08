"""Local hardware probing."""

from __future__ import annotations

import os

from leantrain.hardware.numa import read_numa_nodes
from leantrain.hardware.profile import GPUProfile, HardwareProfile
from leantrain.hardware.topology import (
    pci_bus_id_to_numa_node,
    query_gpu_numa_affinity,
    query_gpu_pcie_groups,
    query_nvidia_smi_gpus,
)


def _host_ram_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size * page_count)


def _probe_gpus() -> list[GPUProfile]:
    smi_by_index = {gpu.index: gpu for gpu in query_nvidia_smi_gpus()}
    numa_by_index = query_gpu_numa_affinity()
    pcie_group_by_index = query_gpu_pcie_groups()
    torch_gpus = _probe_torch_gpus(smi_by_index, numa_by_index, pcie_group_by_index)
    if torch_gpus:
        return torch_gpus

    return [
        GPUProfile(
            id=gpu.index,
            name=gpu.name,
            vram_bytes=gpu.vram_bytes,
            pci_bus_id=gpu.pci_bus_id,
            uuid=gpu.uuid,
            numa_node=numa_by_index.get(gpu.index, pci_bus_id_to_numa_node(gpu.pci_bus_id)),
            pcie_group=pcie_group_by_index.get(gpu.index),
        )
        for gpu in sorted(smi_by_index.values(), key=lambda item: item.index)
    ]


def _probe_torch_gpus(smi_by_index, numa_by_index, pcie_group_by_index) -> list[GPUProfile]:
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
        smi_gpu = smi_by_index.get(device_id)
        gpus.append(
            GPUProfile(
                id=device_id,
                name=props.name,
                vram_bytes=int(props.total_memory),
                pci_bus_id=None if smi_gpu is None else smi_gpu.pci_bus_id,
                uuid=None if smi_gpu is None else smi_gpu.uuid,
                numa_node=numa_by_index.get(
                    device_id,
                    None if smi_gpu is None else pci_bus_id_to_numa_node(smi_gpu.pci_bus_id),
                ),
                pcie_group=pcie_group_by_index.get(device_id),
                supports_bf16=supports_bf16,
            )
        )
    return gpus


def probe_hardware() -> HardwareProfile:
    """Return a best-effort hardware profile for the local machine."""

    return HardwareProfile(
        host_ram_bytes=_host_ram_bytes(),
        numa_nodes=read_numa_nodes(),
        gpus=_probe_gpus(),
    )
