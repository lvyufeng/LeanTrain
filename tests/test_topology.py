from pathlib import Path

from leantrain.hardware.topology import (
    gpu_numa_affinity_from_topology_rows,
    gpu_pcie_groups_from_topology_rows,
    parse_nvidia_smi_gpu_query,
    parse_nvidia_smi_topology_matrix,
    pci_bus_id_to_numa_node,
)


def test_parse_nvidia_smi_gpu_query():
    output = """0, NVIDIA GeForce RTX 4090, 00000000:3B:00.0, GPU-abc
1, NVIDIA GeForce RTX 4090, 00000000:5E:00.0, GPU-def
"""

    gpus = parse_nvidia_smi_gpu_query(output)

    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[0].pci_bus_id == "00000000:3B:00.0"
    assert gpus[1].uuid == "GPU-def"


def test_parse_nvidia_smi_gpu_query_with_memory():
    output = "0, NVIDIA GeForce RTX 4090, 00000000:3B:00.0, GPU-abc, 24564 MiB\n"

    gpus = parse_nvidia_smi_gpu_query(output)

    assert gpus[0].vram_bytes == 24564 * 1024 * 1024


def test_parse_nvidia_smi_topology_matrix():
    output = """
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-23            0
GPU1    PHB      X      24-47           1

Legend:
  X    = Self
"""

    rows = parse_nvidia_smi_topology_matrix(output)

    assert rows[0][:3] == ["GPU0", "X", "PHB"]
    assert rows[1][:3] == ["GPU1", "PHB", "X"]


def test_gpu_numa_affinity_from_topology_rows():
    rows = [
        ["GPU0", "X", "PHB", "0-23", "0", "N/A"],
        ["GPU1", "PHB", "X", "24-47", "1", "N/A"],
    ]

    assert gpu_numa_affinity_from_topology_rows(rows) == {0: 0, 1: 1}


def test_gpu_pcie_groups_from_topology_rows():
    rows = [
        ["GPU0", "X", "PIX", "NODE", "NODE", "0-23", "0"],
        ["GPU1", "PIX", "X", "NODE", "NODE", "0-23", "0"],
        ["GPU2", "NODE", "NODE", "X", "PIX", "48-71", "2"],
        ["GPU3", "NODE", "NODE", "PIX", "X", "48-71", "2"],
    ]

    assert gpu_pcie_groups_from_topology_rows(rows) == {
        0: "pcie0",
        1: "pcie0",
        2: "pcie1",
        3: "pcie1",
    }


def test_pci_bus_id_to_numa_node(tmp_path: Path):
    device_dir = tmp_path / "0000:3B:00.0"
    device_dir.mkdir()
    (device_dir / "numa_node").write_text("2\n", encoding="utf-8")

    assert pci_bus_id_to_numa_node("00000000:3B:00.0", tmp_path) == 2
