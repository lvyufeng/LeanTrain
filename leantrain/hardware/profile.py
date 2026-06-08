"""Hardware profile data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _format_cpu_summary(cpus: list[int]) -> str:
    if not cpus:
        return "unknown"

    ranges: list[str] = []
    sorted_cpus = sorted(cpus)
    start = previous = sorted_cpus[0]
    for cpu in sorted_cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(_format_cpu_range(start, previous))
        start = previous = cpu
    ranges.append(_format_cpu_range(start, previous))
    return f"{','.join(ranges)} ({len(sorted_cpus)} CPUs)"


def _format_cpu_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


@dataclass(slots=True)
class GPUProfile:
    """Description of one accelerator visible to LeanTrain."""

    id: int
    name: str
    vram_bytes: int | None = None
    pci_bus_id: str | None = None
    uuid: str | None = None
    numa_node: int | None = None
    pcie_group: str | None = None
    supports_bf16: bool | None = None


@dataclass(slots=True)
class NUMANodeProfile:
    """Description of one host NUMA node."""

    id: int
    cpus: list[int] = field(default_factory=list)
    memory_bytes: int | None = None


@dataclass(slots=True)
class HardwareProfile:
    """Machine description consumed by planners and schedulers."""

    host_ram_bytes: int | None = None
    numa_nodes: list[NUMANodeProfile] = field(default_factory=list)
    gpus: list[GPUProfile] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = ["LeanTrain hardware profile"]
        if self.host_ram_bytes is not None:
            lines.append(f"- Host RAM: {self.host_ram_bytes / 1024**3:.1f} GiB")
        else:
            lines.append("- Host RAM: unknown")

        if self.numa_nodes:
            lines.append(f"- NUMA nodes: {len(self.numa_nodes)}")
            for node in self.numa_nodes:
                memory = (
                    "unknown"
                    if node.memory_bytes is None
                    else f"{node.memory_bytes / 1024**3:.1f} GiB"
                )
                cpu_range = _format_cpu_summary(node.cpus)
                lines.append(f"  - node{node.id}: memory={memory}, cpus={cpu_range}")
        else:
            lines.append("- NUMA nodes: unknown")

        if not self.gpus:
            lines.append("- GPUs: none detected")
            return "\n".join(lines)

        lines.append(f"- GPUs: {len(self.gpus)}")
        for gpu in self.gpus:
            vram = "unknown" if gpu.vram_bytes is None else f"{gpu.vram_bytes / 1024**3:.1f} GiB"
            bf16 = "unknown" if gpu.supports_bf16 is None else str(gpu.supports_bf16)
            bus = "unknown" if gpu.pci_bus_id is None else gpu.pci_bus_id
            numa = "unknown" if gpu.numa_node is None else str(gpu.numa_node)
            pcie_group = "unknown" if gpu.pcie_group is None else gpu.pcie_group
            lines.append(
                f"  - cuda:{gpu.id}: {gpu.name}, VRAM={vram}, BF16={bf16}, "
                f"PCI={bus}, NUMA={numa}, group={pcie_group}"
            )
        return "\n".join(lines)
