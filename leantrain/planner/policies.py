"""Measurement-driven planner policy recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GIB = 1024**3


@dataclass(slots=True, frozen=True)
class PlannerPolicy:
    """High-level knobs selected from hardware and measurement constraints."""

    dtype: str
    checkpoint_interval: int
    prefetch_depth: int
    gradient_slab_count: int
    loss_scaling: str | None = None
    residency: str = "layer_streaming"
    parallelism: str = "single_gpu_streaming"
    copy_stagger_ms: float = 0.0
    pinned_pool_policy: str = "bounded"
    attention_backend: str = "auto"
    lm_head_chunking: str = "auto"


@dataclass(slots=True, frozen=True)
class SchedulerRecommendation:
    """One scheduler-facing recommendation and its rationale."""

    key: str
    value: str | int | float | bool | list[int] | dict[str, list[int]] | None
    rationale: str
    confidence: str = "medium"


@dataclass(slots=True, frozen=True)
class PlannerRecommendation:
    """Complete planner recommendation derived from a measurement file."""

    target: str
    policy: PlannerPolicy
    scheduler: list[SchedulerRecommendation]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recommend_from_measurement(measurement: dict[str, Any]) -> PlannerRecommendation:
    """Recommend initial planner and scheduler policy from measurement JSON."""

    profile = measurement.get("profile", {})
    bandwidth = measurement.get("bandwidth", {})
    settings = measurement.get("settings", {})

    target = detect_hardware_target(profile)
    groups = group_gpus_by_pcie(profile)
    warnings = _initial_warnings(profile, bandwidth, groups)
    policy = _base_policy_for_target(target, profile)
    copy_stagger_ms, copy_rationale, copy_confidence = _recommend_copy_stagger_ms(
        bandwidth,
        settings,
        has_multi_gpu=len(profile.get("gpus", [])) > 1,
    )
    if copy_stagger_ms:
        policy = _replace_policy(policy, copy_stagger_ms=copy_stagger_ms)

    scheduler = _scheduler_recommendations(
        target=target,
        profile=profile,
        groups=groups,
        copy_stagger_ms=copy_stagger_ms,
        copy_rationale=copy_rationale,
        copy_confidence=copy_confidence,
    )
    return PlannerRecommendation(target=target, policy=policy, scheduler=scheduler, warnings=warnings)


def render_recommendations(recommendation: PlannerRecommendation) -> str:
    """Render planner recommendations as Markdown."""

    policy = recommendation.policy
    lines = [
        "# LeanTrain Planner Recommendations",
        "",
        f"- Detected target: {recommendation.target}",
        f"- Policy dtype: {policy.dtype}",
        f"- Loss scaling: {_format_optional(policy.loss_scaling)}",
        f"- Residency: {policy.residency}",
        f"- Parallelism: {policy.parallelism}",
        f"- Checkpoint interval: {policy.checkpoint_interval}",
        f"- Prefetch depth: {policy.prefetch_depth}",
        f"- Gradient slab count: {policy.gradient_slab_count}",
        f"- Copy staggering: {policy.copy_stagger_ms:.3f} ms",
        f"- Pinned pool policy: {policy.pinned_pool_policy}",
        f"- Attention backend: {policy.attention_backend}",
        f"- LM-head chunking: {policy.lm_head_chunking}",
        "",
        "## Scheduler Recommendations",
        "",
    ]

    if recommendation.scheduler:
        for item in recommendation.scheduler:
            lines.append(f"- **{item.key}**: {_format_value(item.value)}")
            lines.append(f"  - Why: {item.rationale}")
            lines.append(f"  - Confidence: {item.confidence}")
    else:
        lines.append("No scheduler recommendations were generated.")

    if recommendation.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in recommendation.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines).rstrip() + "\n"


def detect_hardware_target(profile: dict[str, Any]) -> str:
    """Detect a coarse hardware target from profile data."""

    gpus = profile.get("gpus", [])
    if _mostly_named(gpus, "4090") and len(gpus) >= 8:
        return "8x4090"
    if _mostly_named(gpus, "2080 ti") and len(gpus) >= 4:
        return "4x2080ti"
    if _mostly_named(gpus, "2080ti") and len(gpus) >= 4:
        return "4x2080ti"
    return "unknown"


def group_gpus_by_pcie(profile: dict[str, Any]) -> dict[str, list[int]]:
    """Group GPUs by measured PCIe group labels."""

    groups: dict[str, list[int]] = {}
    for gpu in profile.get("gpus", []):
        group = gpu.get("pcie_group")
        gpu_id = gpu.get("id")
        if group is None or gpu_id is None:
            continue
        groups.setdefault(group, []).append(gpu_id)
    return {group: sorted(devices) for group, devices in sorted(groups.items())}


def _base_policy_for_target(target: str, profile: dict[str, Any]) -> PlannerPolicy:
    if target == "8x4090":
        return PlannerPolicy(
            dtype="bf16" if _profile_supports_bf16(profile) is not False else "fp16",
            checkpoint_interval=2,
            prefetch_depth=2,
            gradient_slab_count=2,
            residency="block_streaming_double_buffered",
            parallelism="host_backed_data_parallel_first",
            pinned_pool_policy="bounded_per_pcie_group",
        )

    if target == "4x2080ti":
        return PlannerPolicy(
            dtype="fp16",
            checkpoint_interval=1,
            prefetch_depth=1,
            gradient_slab_count=1,
            loss_scaling="dynamic",
            residency="one_layer_or_minimal_block",
            parallelism="single_gpu_streaming_first",
            pinned_pool_policy="small_bounded",
            attention_backend="conservative_sdpa_or_xformers",
            lm_head_chunking="aggressive",
        )

    supports_bf16 = _profile_supports_bf16(profile)
    min_vram = _min_gpu_vram(profile)
    return PlannerPolicy(
        dtype="bf16" if supports_bf16 is True else "fp16",
        checkpoint_interval=1,
        prefetch_depth=1 if min_vram is not None and min_vram <= 12 * GIB else 2,
        gradient_slab_count=1 if min_vram is not None and min_vram <= 12 * GIB else 2,
        loss_scaling=None if supports_bf16 is True else "dynamic",
        residency="one_layer_or_minimal_block" if min_vram is not None and min_vram <= 12 * GIB else "layer_streaming",
        parallelism="single_gpu_streaming_first",
        pinned_pool_policy="bounded",
        attention_backend="auto" if supports_bf16 is True else "conservative",
        lm_head_chunking="auto" if min_vram is None or min_vram > 12 * GIB else "aggressive",
    )


def _scheduler_recommendations(
    *,
    target: str,
    profile: dict[str, Any],
    groups: dict[str, list[int]],
    copy_stagger_ms: float,
    copy_rationale: str,
    copy_confidence: str,
) -> list[SchedulerRecommendation]:
    recommendations: list[SchedulerRecommendation] = []

    if groups:
        recommendations.append(
            SchedulerRecommendation(
                key="pcie_copy_islands",
                value=groups,
                rationale="GPUs in the same PCIe group should be treated as copy-concurrency islands.",
                confidence="high",
            )
        )

    recommendations.append(
        SchedulerRecommendation(
            key="copy_stagger_ms",
            value=copy_stagger_ms,
            rationale=copy_rationale,
            confidence=copy_confidence,
        )
    )

    if target == "8x4090":
        recommendations.extend([
            SchedulerRecommendation(
                key="numa_placement",
                value="place pinned buffers on the GPU-local NUMA node",
                rationale="4090 workstations are PCIe/NUMA bandwidth limited before they are FLOP limited.",
                confidence="high",
            ),
            SchedulerRecommendation(
                key="tensor_parallel_default",
                value=False,
                rationale="Avoid tensor parallelism as the default on PCIe-only 4090 machines unless layer size requires it.",
                confidence="medium",
            ),
        ])
    elif target == "4x2080ti":
        recommendations.extend([
            SchedulerRecommendation(
                key="activation_policy",
                value="aggressive_checkpoint_and_recompute",
                rationale="11 GiB VRAM leaves little room for activations and attention workspaces.",
                confidence="high",
            ),
            SchedulerRecommendation(
                key="bf16_default",
                value=False,
                rationale="RTX 2080 Ti does not provide native BF16 tensor cores; use FP16 with dynamic loss scaling.",
                confidence="high",
            ),
        ])
    elif len(profile.get("gpus", [])) > 1:
        recommendations.append(
            SchedulerRecommendation(
                key="parallelism_default",
                value="host_backed_data_parallel_with_measurement_gates",
                rationale="Multiple GPUs were detected, but the target is not recognized; start with conservative host-backed data parallelism.",
                confidence="low",
            )
        )

    return recommendations


def _recommend_copy_stagger_ms(
    bandwidth: dict[str, Any],
    settings: dict[str, Any],
    *,
    has_multi_gpu: bool,
) -> tuple[float, str, str]:
    multi = bandwidth.get("multi_gpu", [])
    simultaneous = [result for result in multi if result.get("mode") == "simultaneous_h2d"]
    staggered = [result for result in multi if result.get("mode") == "staggered_h2d"]

    if simultaneous and staggered:
        best_sim = max(result.get("aggregate_gb_per_second", 0.0) for result in simultaneous)
        best_stagger = max(result.get("aggregate_gb_per_second", 0.0) for result in staggered)
        if best_stagger > best_sim * 1.05:
            stagger_ms = float(settings.get("stagger_seconds", 0.002)) * 1000.0
            return (
                stagger_ms,
                "Staggered H2D aggregate bandwidth exceeded simultaneous H2D by more than 5%.",
                "high",
            )
        return (
            0.0,
            "Measured staggered H2D did not clearly improve aggregate bandwidth over simultaneous H2D.",
            "medium",
        )

    if has_multi_gpu:
        return (
            2.0,
            "No complete simultaneous-vs-staggered H2D comparison was found; use conservative PCIe copy staggering until measured.",
            "low",
        )

    return (0.0, "Single-GPU profile does not require copy staggering.", "medium")


def _initial_warnings(
    profile: dict[str, Any],
    bandwidth: dict[str, Any],
    groups: dict[str, list[int]],
) -> list[str]:
    warnings: list[str] = []
    gpus = profile.get("gpus", [])
    if not gpus:
        warnings.append("No GPUs were found in the measurement profile; recommendations are CPU/profile-only.")
    if len(gpus) > 1 and not groups:
        warnings.append("No PCIe groups were detected; avoid assuming uniform multi-GPU copy bandwidth.")
    if len(gpus) > 1 and not bandwidth.get("multi_gpu"):
        warnings.append("No multi-GPU H2D measurements were recorded; copy-concurrency guidance is conservative.")
    if bandwidth.get("errors"):
        warnings.append("Measurement errors were recorded; inspect the measurement report before trusting bandwidth-derived recommendations.")
    if _profile_supports_bf16(profile) is None:
        warnings.append("BF16 support is unknown; verify with PyTorch before using BF16 defaults.")
    return warnings


def _replace_policy(policy: PlannerPolicy, **changes) -> PlannerPolicy:
    data = asdict(policy)
    data.update(changes)
    return PlannerPolicy(**data)


def _profile_supports_bf16(profile: dict[str, Any]) -> bool | None:
    values = [gpu.get("supports_bf16") for gpu in profile.get("gpus", [])]
    known = [value for value in values if value is not None]
    if not known:
        return None
    return all(bool(value) for value in known)


def _min_gpu_vram(profile: dict[str, Any]) -> int | None:
    values = [gpu.get("vram_bytes") for gpu in profile.get("gpus", []) if gpu.get("vram_bytes")]
    return min(values) if values else None


def _mostly_named(gpus: list[dict[str, Any]], needle: str) -> bool:
    if not gpus:
        return False
    names = [str(gpu.get("name", "")).lower().replace("geforce", "") for gpu in gpus]
    matches = sum(1 for name in names if needle in name)
    return matches >= max(1, len(gpus) // 2)


def _format_optional(value: str | None) -> str:
    return "none" if value is None else value


def _format_value(value) -> str:
    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {','.join(str(item) for item in items)}" for key, items in value.items()
        )
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)
