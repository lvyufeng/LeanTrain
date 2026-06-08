"""Optional NVIDIA topology discovery.

This module uses ``nvidia-smi`` when it is available. Importing it does not
require NVIDIA drivers, CUDA, or PyTorch.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class NvidiaSMIGPU:
    """One GPU row returned by ``nvidia-smi --query-gpu``."""

    index: int
    name: str
    pci_bus_id: str | None = None
    uuid: str | None = None
    vram_bytes: int | None = None


def query_nvidia_smi_gpus() -> list[NvidiaSMIGPU]:
    """Return GPU identity/topology hints from nvidia-smi if available."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,pci.bus_id,uuid,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []

    if completed.returncode != 0:
        return []

    return parse_nvidia_smi_gpu_query(completed.stdout)


def parse_nvidia_smi_gpu_query(output: str) -> list[NvidiaSMIGPU]:
    """Parse ``nvidia-smi --query-gpu`` CSV output."""

    gpus: list[NvidiaSMIGPU] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        gpus.append(
            NvidiaSMIGPU(
                index=index,
                name=fields[1],
                pci_bus_id=_none_if_empty(fields[2]),
                uuid=_none_if_empty(fields[3]),
                vram_bytes=_parse_memory_total_mib(fields[4]) if len(fields) >= 5 else None,
            )
        )
    return gpus


def query_gpu_numa_affinity() -> dict[int, int]:
    """Return GPU index to NUMA affinity from ``nvidia-smi topo -m``."""

    return gpu_numa_affinity_from_topology_rows(query_nvidia_smi_topology_matrix())


def query_gpu_pcie_groups() -> dict[int, str]:
    """Return coarse PCIe groups from ``nvidia-smi topo -m`` PIX islands."""

    return gpu_pcie_groups_from_topology_rows(query_nvidia_smi_topology_matrix())


def gpu_numa_affinity_from_topology_rows(rows: list[list[str]]) -> dict[int, int]:
    """Extract GPU NUMA affinity from parsed topology rows.

    NVIDIA's table layout changes when NICs are present. Rather than depending on
    exact column offsets, use the last numeric field in each GPU row. This maps
    current ``nvidia-smi topo -m`` output where the tail is either
    ``CPU Affinity, NUMA Affinity`` or ``CPU Affinity, NUMA Affinity, GPU NUMA ID``.
    """

    affinity: dict[int, int] = {}
    for row in rows:
        if not row or not row[0].startswith("GPU"):
            continue
        index_text = row[0].removeprefix("GPU")
        if not index_text.isdigit():
            continue
        for field in reversed(row[1:]):
            if field.isdigit():
                affinity[int(index_text)] = int(field)
                break
    return affinity


def gpu_pcie_groups_from_topology_rows(rows: list[list[str]]) -> dict[int, str]:
    """Group GPUs that have PIX connectivity in the topology matrix.

    This is a coarse scheduling hint for shared PCIe-switch islands. It is not a
    full graph model, but it is useful for early copy-staggering policies.
    """

    gpu_rows = [row for row in rows if row and row[0].startswith("GPU")]
    gpu_count = len(gpu_rows)
    parent = list(range(gpu_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row_index, row in enumerate(gpu_rows):
        for col_index, value in enumerate(row[1 : gpu_count + 1]):
            if value == "PIX":
                union(row_index, col_index)

    root_to_group: dict[int, str] = {}
    groups: dict[int, str] = {}
    for row_index, row in enumerate(gpu_rows):
        root = find(row_index)
        if root not in root_to_group:
            root_to_group[root] = f"pcie{len(root_to_group)}"
        gpu_index_text = row[0].removeprefix("GPU")
        if gpu_index_text.isdigit():
            groups[int(gpu_index_text)] = root_to_group[root]
    return groups


def query_nvidia_smi_topology_matrix() -> list[list[str]]:
    """Return the raw ``nvidia-smi topo -m`` table as rows of columns.

    The matrix is intentionally kept textual because NVIDIA topology labels vary
    by driver and system generation. The scheduler can later interpret labels
    such as PIX, PXB, PHB, NODE, SYS, and NVLink variants.
    """

    try:
        completed = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if completed.returncode != 0:
        return []

    return parse_nvidia_smi_topology_matrix(completed.stdout)


def parse_nvidia_smi_topology_matrix(output: str) -> list[list[str]]:
    """Parse the whitespace table emitted by ``nvidia-smi topo -m``."""

    rows: list[list[str]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Legend"):
            continue
        fields = line.split()
        if not fields:
            continue
        if not fields[0].startswith("GPU"):
            continue
        if len(fields) > 1 and fields[1].startswith("GPU"):
            continue
        rows.append(fields)
    return rows


def pci_bus_id_to_numa_node(
    pci_bus_id: str | None,
    pci_root: Path = Path("/sys/bus/pci/devices"),
) -> int | None:
    """Return the NUMA node associated with a PCI bus id from sysfs."""

    if pci_bus_id is None:
        return None

    candidates = [pci_bus_id]
    if pci_bus_id.startswith("00000000:"):
        candidates.append("0000:" + pci_bus_id.removeprefix("00000000:"))

    for candidate in candidates:
        numa_file = pci_root / candidate / "numa_node"
        try:
            value = int(numa_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        return None if value < 0 else value
    return None


def _parse_memory_total_mib(value: str) -> int | None:
    try:
        return int(value.split()[0]) * 1024 * 1024
    except (IndexError, ValueError):
        return None


def _none_if_empty(value: str) -> str | None:
    return value if value and value != "[N/A]" else None
