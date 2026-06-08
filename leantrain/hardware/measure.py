"""Hardware measurement suite orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from leantrain.hardware.bandwidth import benchmark_copy_bandwidth, benchmark_multi_gpu_h2d_bandwidth
from leantrain.hardware.profile import HardwareProfile
from leantrain.hardware.probe import probe_hardware

SCHEMA_VERSION = 1


def run_measurement_suite(
    *,
    devices: list[int] | None = None,
    include_bandwidth: bool = True,
    include_pageable: bool = False,
    include_multi: bool = True,
    include_grouped: bool = True,
    size_mb: int = 256,
    repeats: int = 10,
    warmup: int = 3,
    stagger_seconds: float = 0.0,
) -> dict[str, Any]:
    """Collect a hardware profile and optional bandwidth measurements.

    CUDA/PyTorch failures are recorded as measurement errors instead of aborting
    the whole suite, so ``measure`` still produces a useful profile on machines
    where GPUs are visible to ``nvidia-smi`` but not usable by PyTorch.
    """

    profile = probe_hardware()
    selected_devices = _select_devices(profile, devices)
    measurement: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "settings": {
            "devices": selected_devices,
            "include_bandwidth": include_bandwidth,
            "include_pageable": include_pageable,
            "include_multi": include_multi,
            "include_grouped": include_grouped,
            "size_mb": size_mb,
            "repeats": repeats,
            "warmup": warmup,
            "stagger_seconds": stagger_seconds,
        },
        "profile": profile.to_dict(),
        "bandwidth": {
            "single_gpu": [],
            "multi_gpu": [],
            "errors": [],
        },
    }

    if not include_bandwidth or not selected_devices:
        return measurement

    for device_id in selected_devices:
        _measure_single_device(
            measurement,
            device_id=device_id,
            size_mb=size_mb,
            repeats=repeats,
            warmup=warmup,
            pinned=True,
        )
        if include_pageable:
            _measure_single_device(
                measurement,
                device_id=device_id,
                size_mb=size_mb,
                repeats=repeats,
                warmup=warmup,
                pinned=False,
            )

    if include_multi and len(selected_devices) > 1:
        _measure_multi_device(
            measurement,
            scope="all",
            device_ids=selected_devices,
            size_mb=size_mb,
            repeats=repeats,
            warmup=warmup,
            pinned=True,
            stagger_seconds=0.0,
        )
        if stagger_seconds > 0:
            _measure_multi_device(
                measurement,
                scope="all_staggered",
                device_ids=selected_devices,
                size_mb=size_mb,
                repeats=repeats,
                warmup=warmup,
                pinned=True,
                stagger_seconds=stagger_seconds,
            )
        if include_pageable:
            _measure_multi_device(
                measurement,
                scope="all_pageable",
                device_ids=selected_devices,
                size_mb=size_mb,
                repeats=repeats,
                warmup=warmup,
                pinned=False,
                stagger_seconds=0.0,
            )

    if include_grouped:
        for group_name, group_devices in _pcie_groups(profile, selected_devices).items():
            if len(group_devices) < 2:
                continue
            _measure_multi_device(
                measurement,
                scope=f"group:{group_name}",
                device_ids=group_devices,
                size_mb=size_mb,
                repeats=repeats,
                warmup=warmup,
                pinned=True,
                stagger_seconds=0.0,
            )
            if stagger_seconds > 0:
                _measure_multi_device(
                    measurement,
                    scope=f"group:{group_name}:staggered",
                    device_ids=group_devices,
                    size_mb=size_mb,
                    repeats=repeats,
                    warmup=warmup,
                    pinned=True,
                    stagger_seconds=stagger_seconds,
                )

    return measurement


def save_measurement(measurement: dict[str, Any], output_path: Path) -> None:
    """Write a measurement JSON file, creating parent directories."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(measurement, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_measurement(input_path: Path) -> dict[str, Any]:
    """Load a measurement JSON file."""

    return json.loads(input_path.read_text(encoding="utf-8"))


def _measure_single_device(
    measurement: dict[str, Any],
    *,
    device_id: int,
    size_mb: int,
    repeats: int,
    warmup: int,
    pinned: bool,
) -> None:
    try:
        results = benchmark_copy_bandwidth(
            device_id=device_id,
            size_mb=size_mb,
            repeats=repeats,
            warmup=warmup,
            pinned=pinned,
        )
    except Exception as exc:  # noqa: BLE001 - record and continue measurement suite.
        _record_error(measurement, scope=f"single:{device_id}", error=exc)
        return

    measurement["bandwidth"]["single_gpu"].extend(result.to_dict() for result in results)


def _measure_multi_device(
    measurement: dict[str, Any],
    *,
    scope: str,
    device_ids: list[int],
    size_mb: int,
    repeats: int,
    warmup: int,
    pinned: bool,
    stagger_seconds: float,
) -> None:
    try:
        result = benchmark_multi_gpu_h2d_bandwidth(
            device_ids=device_ids,
            size_mb=size_mb,
            repeats=repeats,
            warmup=warmup,
            pinned=pinned,
            stagger_seconds=stagger_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - record and continue measurement suite.
        _record_error(measurement, scope=f"multi:{scope}", error=exc)
        return

    data = result.to_dict()
    data["scope"] = scope
    measurement["bandwidth"]["multi_gpu"].append(data)


def _record_error(measurement: dict[str, Any], *, scope: str, error: Exception) -> None:
    measurement["bandwidth"]["errors"].append(
        {
            "scope": scope,
            "type": type(error).__name__,
            "message": str(error),
        }
    )


def _select_devices(profile: HardwareProfile, devices: list[int] | None) -> list[int]:
    available = [gpu.id for gpu in profile.gpus]
    if devices is None:
        return available
    available_set = set(available)
    return [device_id for device_id in devices if device_id in available_set]


def _pcie_groups(profile: HardwareProfile, selected_devices: list[int]) -> dict[str, list[int]]:
    selected = set(selected_devices)
    groups: dict[str, list[int]] = {}
    for gpu in profile.gpus:
        if gpu.id not in selected or gpu.pcie_group is None:
            continue
        groups.setdefault(gpu.pcie_group, []).append(gpu.id)
    return groups
