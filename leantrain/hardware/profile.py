"""Hardware profile data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class GPUProfile:
    """Description of one accelerator visible to LeanTrain."""

    id: int
    name: str
    vram_bytes: int | None = None
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
        else:
            lines.append("- NUMA nodes: unknown")

        if not self.gpus:
            lines.append("- GPUs: none detected")
            return "\n".join(lines)

        lines.append(f"- GPUs: {len(self.gpus)}")
        for gpu in self.gpus:
            vram = "unknown" if gpu.vram_bytes is None else f"{gpu.vram_bytes / 1024**3:.1f} GiB"
            bf16 = "unknown" if gpu.supports_bf16 is None else str(gpu.supports_bf16)
            lines.append(f"  - cuda:{gpu.id}: {gpu.name}, VRAM={vram}, BF16={bf16}")
        return "\n".join(lines)
