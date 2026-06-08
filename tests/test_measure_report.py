from leantrain.hardware.report import render_measurement_report


def test_render_measurement_report():
    measurement = {
        "schema_version": 1,
        "profile": {
            "host_ram_bytes": 2 * 1024**3,
            "numa_nodes": [{"id": 0, "cpus": [0, 1], "memory_bytes": 2 * 1024**3}],
            "gpus": [
                {
                    "id": 0,
                    "name": "RTX 4090",
                    "vram_bytes": 24 * 1024**3,
                    "pci_bus_id": "00000000:3B:00.0",
                    "uuid": "GPU-test",
                    "numa_node": 0,
                    "pcie_group": "pcie0",
                    "supports_bf16": True,
                }
            ],
        },
        "bandwidth": {
            "single_gpu": [
                {
                    "device_id": 0,
                    "direction": "h2d",
                    "pinned": True,
                    "bytes_per_copy": 256 * 1024**2,
                    "repeats": 3,
                    "mean_seconds": 0.01,
                    "gb_per_second": 26.8,
                }
            ],
            "multi_gpu": [],
            "errors": [],
        },
    }

    report = render_measurement_report(measurement)

    assert "LeanTrain Hardware Measurement Report" in report
    assert "RTX 4090" in report
    assert "26.80 GB/s" in report
    assert "pcie0" in report
