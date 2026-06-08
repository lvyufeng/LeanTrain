"""Markdown reports for hardware measurement JSON files."""

from __future__ import annotations

from typing import Any

from leantrain.planner.policies import recommend_from_measurement


def render_measurement_report(measurement: dict[str, Any]) -> str:
    """Render a saved measurement dictionary as a Markdown report."""

    profile = measurement.get("profile", {})
    bandwidth = measurement.get("bandwidth", {})
    lines: list[str] = ["# LeanTrain Hardware Measurement Report", ""]

    lines.extend(_render_profile(profile))
    lines.extend(_render_single_gpu_bandwidth(bandwidth.get("single_gpu", [])))
    lines.extend(_render_multi_gpu_bandwidth(bandwidth.get("multi_gpu", [])))
    lines.extend(_render_errors(bandwidth.get("errors", [])))
    lines.extend(_render_scheduler_hints(profile, bandwidth))

    return "\n".join(lines).rstrip() + "\n"


def _render_profile(profile: dict[str, Any]) -> list[str]:
    lines = ["## Profile", ""]
    host_ram = profile.get("host_ram_bytes")
    lines.append(f"- Host RAM: {_format_gib(host_ram)}")
    lines.append(f"- NUMA nodes: {len(profile.get('numa_nodes', []))}")
    lines.append(f"- GPUs: {len(profile.get('gpus', []))}")
    lines.append("")

    if profile.get("gpus"):
        lines.extend([
            "| GPU | Name | VRAM | NUMA | PCIe Group | PCI Bus | BF16 |",
            "|---:|---|---:|---:|---|---|---|",
        ])
        for gpu in profile["gpus"]:
            lines.append(
                "| "
                f"{gpu.get('id')} | "
                f"{gpu.get('name', 'unknown')} | "
                f"{_format_gib(gpu.get('vram_bytes'))} | "
                f"{_format_unknown(gpu.get('numa_node'))} | "
                f"{_format_unknown(gpu.get('pcie_group'))} | "
                f"{_format_unknown(gpu.get('pci_bus_id'))} | "
                f"{_format_unknown(gpu.get('supports_bf16'))} |"
            )
        lines.append("")

    return lines


def _render_single_gpu_bandwidth(results: list[dict[str, Any]]) -> list[str]:
    lines = ["## Single-GPU Copy Bandwidth", ""]
    if not results:
        lines.extend(["No single-GPU bandwidth results recorded.", ""])
        return lines

    lines.extend([
        "| GPU | Direction | Host Memory | Size | Mean | Bandwidth |",
        "|---:|---|---|---:|---:|---:|",
    ])
    for result in results:
        lines.append(
            "| "
            f"{result['device_id']} | "
            f"{result['direction'].upper()} | "
            f"{'pinned' if result['pinned'] else 'pageable'} | "
            f"{_format_mib(result['bytes_per_copy'])} | "
            f"{result['mean_seconds'] * 1000:.3f} ms | "
            f"{result['gb_per_second']:.2f} GB/s |"
        )
    lines.append("")
    return lines


def _render_multi_gpu_bandwidth(results: list[dict[str, Any]]) -> list[str]:
    lines = ["## Multi-GPU H2D Bandwidth", ""]
    if not results:
        lines.extend(["No multi-GPU bandwidth results recorded.", ""])
        return lines

    lines.extend([
        "| Scope | GPUs | Mode | Host Memory | Size/GPU | Mean | Aggregate | Per GPU |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    for result in results:
        lines.append(
            "| "
            f"{result.get('scope', 'unknown')} | "
            f"{','.join(str(device) for device in result['device_ids'])} | "
            f"{result['mode']} | "
            f"{'pinned' if result['pinned'] else 'pageable'} | "
            f"{_format_mib(result['bytes_per_device'])} | "
            f"{result['mean_seconds'] * 1000:.3f} ms | "
            f"{result['aggregate_gb_per_second']:.2f} GB/s | "
            f"{result['per_device_gb_per_second']:.2f} GB/s |"
        )
    lines.append("")
    return lines


def _render_errors(errors: list[dict[str, Any]]) -> list[str]:
    if not errors:
        return []

    lines = ["## Measurement Errors", ""]
    for error in errors:
        lines.append(
            f"- `{error.get('scope', 'unknown')}`: "
            f"{error.get('type', 'Error')}: {error.get('message', '')}"
        )
    lines.append("")
    return lines


def _render_scheduler_hints(profile: dict[str, Any], bandwidth: dict[str, Any]) -> list[str]:
    recommendation = recommend_from_measurement({"profile": profile, "bandwidth": bandwidth, "settings": {}})
    lines = ["## Scheduler Hints", ""]
    for item in recommendation.scheduler:
        lines.append(f"- **{item.key}**: {_format_unknown(item.value)}")
        lines.append(f"  - {item.rationale}")
    for warning in recommendation.warnings:
        lines.append(f"- Warning: {warning}")
    lines.append("")
    return lines


def _format_gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 1024**3:.1f} GiB"


def _format_mib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 1024**2:.1f} MiB"


def _format_unknown(value) -> str:
    return "unknown" if value is None else str(value)
