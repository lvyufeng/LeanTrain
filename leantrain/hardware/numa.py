"""Best-effort NUMA discovery from Linux sysfs."""

from __future__ import annotations

from pathlib import Path

from leantrain.hardware.profile import NUMANodeProfile


def parse_cpu_list(text: str) -> list[int]:
    """Parse Linux CPU list syntax such as ``0-3,8,10-11``."""

    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return cpus


def read_numa_nodes(sysfs_root: Path = Path("/sys/devices/system/node")) -> list[NUMANodeProfile]:
    """Read NUMA node descriptions from Linux sysfs if available."""

    if not sysfs_root.exists():
        return []

    nodes: list[NUMANodeProfile] = []
    for node_dir in sorted(sysfs_root.glob("node*")):
        node_id_text = node_dir.name.removeprefix("node")
        if not node_id_text.isdigit():
            continue

        node_id = int(node_id_text)
        cpus = _read_node_cpus(node_dir)
        memory_bytes = _read_node_memory_bytes(node_dir)
        nodes.append(NUMANodeProfile(id=node_id, cpus=cpus, memory_bytes=memory_bytes))

    return nodes


def _read_node_cpus(node_dir: Path) -> list[int]:
    cpulist = node_dir / "cpulist"
    try:
        return parse_cpu_list(cpulist.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _read_node_memory_bytes(node_dir: Path) -> int | None:
    meminfo = node_dir / "meminfo"
    try:
        lines = meminfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    prefix = f"Node {node_dir.name.removeprefix('node')} MemTotal:"
    for line in lines:
        if not line.startswith(prefix):
            continue
        fields = line.split()
        if len(fields) < 4 or fields[-1] != "kB":
            return None
        try:
            return int(fields[-2]) * 1024
        except ValueError:
            return None
    return None
