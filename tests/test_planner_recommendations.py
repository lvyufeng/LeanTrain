import json
import subprocess
import sys

from leantrain.planner.policies import recommend_from_measurement, render_recommendations

GIB = 1024**3


def _gpu(gpu_id, name="NVIDIA GeForce RTX 4090", vram_gib=24, bf16=True, group="pcie0"):
    return {
        "id": gpu_id,
        "name": name,
        "vram_bytes": vram_gib * GIB,
        "pci_bus_id": f"00000000:{gpu_id:02X}:00.0",
        "uuid": f"GPU-{gpu_id}",
        "numa_node": 0 if gpu_id < 4 else 2,
        "pcie_group": group,
        "supports_bf16": bf16,
    }


def test_recommend_8x4090_with_staggered_bandwidth():
    measurement = {
        "schema_version": 1,
        "settings": {"stagger_seconds": 0.002},
        "profile": {
            "host_ram_bytes": 2 * 1024**4,
            "numa_nodes": [],
            "gpus": [_gpu(i, group=f"pcie{i // 2}") for i in range(8)],
        },
        "bandwidth": {
            "single_gpu": [],
            "multi_gpu": [
                {
                    "scope": "all",
                    "device_ids": list(range(8)),
                    "mode": "simultaneous_h2d",
                    "pinned": True,
                    "bytes_per_device": 256 * 1024**2,
                    "repeats": 3,
                    "mean_seconds": 0.03,
                    "aggregate_gb_per_second": 80.0,
                    "per_device_gb_per_second": 10.0,
                    "stagger_seconds": 0.0,
                },
                {
                    "scope": "all_staggered",
                    "device_ids": list(range(8)),
                    "mode": "staggered_h2d",
                    "pinned": True,
                    "bytes_per_device": 256 * 1024**2,
                    "repeats": 3,
                    "mean_seconds": 0.025,
                    "aggregate_gb_per_second": 92.0,
                    "per_device_gb_per_second": 11.5,
                    "stagger_seconds": 0.002,
                },
            ],
            "errors": [],
        },
    }

    recommendation = recommend_from_measurement(measurement)

    assert recommendation.target == "8x4090"
    assert recommendation.policy.dtype == "bf16"
    assert recommendation.policy.prefetch_depth == 2
    assert recommendation.policy.copy_stagger_ms == 2.0
    assert recommendation.policy.parallelism == "host_backed_data_parallel_first"
    assert recommendation.scheduler[0].key == "pcie_copy_islands"
    assert "pcie0" in recommendation.scheduler[0].value


def test_recommend_4x2080ti_conservative_policy():
    measurement = {
        "schema_version": 1,
        "settings": {},
        "profile": {
            "host_ram_bytes": 1024**4,
            "numa_nodes": [],
            "gpus": [
                _gpu(i, name="NVIDIA GeForce RTX 2080 Ti", vram_gib=11, bf16=False, group=f"pcie{i // 2}")
                for i in range(4)
            ],
        },
        "bandwidth": {"single_gpu": [], "multi_gpu": [], "errors": []},
    }

    recommendation = recommend_from_measurement(measurement)

    assert recommendation.target == "4x2080ti"
    assert recommendation.policy.dtype == "fp16"
    assert recommendation.policy.loss_scaling == "dynamic"
    assert recommendation.policy.prefetch_depth == 1
    assert recommendation.policy.gradient_slab_count == 1
    assert recommendation.policy.attention_backend == "conservative_sdpa_or_xformers"
    assert any(item.key == "bf16_default" and item.value is False for item in recommendation.scheduler)


def test_recommend_unknown_profile_warns_without_measurements():
    measurement = {
        "schema_version": 1,
        "settings": {},
        "profile": {
            "host_ram_bytes": None,
            "numa_nodes": [],
            "gpus": [{"id": 0, "name": "Unknown GPU", "vram_bytes": 8 * GIB, "supports_bf16": None}],
        },
        "bandwidth": {"single_gpu": [], "multi_gpu": [], "errors": []},
    }

    recommendation = recommend_from_measurement(measurement)

    assert recommendation.target == "unknown"
    assert recommendation.policy.dtype == "fp16"
    assert recommendation.policy.prefetch_depth == 1
    assert any("BF16 support is unknown" in warning for warning in recommendation.warnings)


def test_render_recommendations_markdown():
    recommendation = recommend_from_measurement(
        {
            "profile": {"gpus": [_gpu(0)]},
            "bandwidth": {"single_gpu": [], "multi_gpu": [], "errors": []},
            "settings": {},
        }
    )

    rendered = render_recommendations(recommendation)

    assert "LeanTrain Planner Recommendations" in rendered
    assert "Policy dtype" in rendered
    assert "Scheduler Recommendations" in rendered


def test_recommend_cli_json(tmp_path):
    measurement = {
        "schema_version": 1,
        "settings": {},
        "profile": {"gpus": [_gpu(i, group=f"pcie{i // 2}") for i in range(8)]},
        "bandwidth": {"single_gpu": [], "multi_gpu": [], "errors": []},
    }
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(measurement), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "leantrain.cli", "recommend", str(path), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(completed.stdout)
    assert data["target"] == "8x4090"
    assert data["policy"]["dtype"] == "bf16"
