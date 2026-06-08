from leantrain.hardware.numa import parse_cpu_list
from leantrain.hardware.profile import GPUProfile, HardwareProfile, NUMANodeProfile


def test_parse_cpu_list():
    assert parse_cpu_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]


def test_hardware_profile_to_dict():
    profile = HardwareProfile(
        host_ram_bytes=1024,
        numa_nodes=[NUMANodeProfile(id=0, cpus=[0, 1], memory_bytes=1024)],
        gpus=[
            GPUProfile(
                id=0,
                name="RTX 4090",
                vram_bytes=24 * 1024**3,
                pci_bus_id="00000000:3B:00.0",
                supports_bf16=True,
            )
        ],
    )

    data = profile.to_dict()

    assert data["host_ram_bytes"] == 1024
    assert data["numa_nodes"][0]["cpus"] == [0, 1]
    assert data["gpus"][0]["name"] == "RTX 4090"
    assert data["gpus"][0]["pci_bus_id"] == "00000000:3B:00.0"


def test_hardware_profile_summary_mentions_gpu():
    profile = HardwareProfile(gpus=[GPUProfile(id=0, name="RTX 2080 Ti", supports_bf16=False)])

    summary = profile.summary()

    assert "RTX 2080 Ti" in summary
    assert "BF16=False" in summary
