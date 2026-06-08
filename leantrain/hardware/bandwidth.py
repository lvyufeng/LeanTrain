"""CPU↔GPU bandwidth microbenchmarks.

These helpers require PyTorch and CUDA at runtime, but importing the module does
not require either dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from time import perf_counter, sleep


@dataclass(slots=True, frozen=True)
class CopyBandwidthResult:
    """Measured bandwidth for one copy direction and host allocation mode."""

    device_id: int
    direction: str
    pinned: bool
    bytes_per_copy: int
    repeats: int
    mean_seconds: float
    gb_per_second: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class MultiGPUCopyBandwidthResult:
    """Measured aggregate H2D bandwidth across multiple GPUs."""

    device_ids: list[int]
    mode: str
    pinned: bool
    bytes_per_device: int
    repeats: int
    mean_seconds: float
    aggregate_gb_per_second: float
    per_device_gb_per_second: float
    stagger_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def benchmark_copy_bandwidth(
    *,
    device_id: int = 0,
    size_mb: int = 256,
    repeats: int = 20,
    warmup: int = 5,
    pinned: bool = True,
) -> list[CopyBandwidthResult]:
    """Measure H2D and D2H bandwidth for one CUDA device.

    The result is intended as a practical scheduler signal, not a perfect PCIe
    benchmark. It uses one contiguous float32 tensor and synchronizes around each
    copy to measure elapsed wall time.
    """

    if size_mb <= 0:
        raise ValueError("size_mb must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    torch = _import_torch_with_cuda()
    if device_id < 0 or device_id >= torch.cuda.device_count():
        raise ValueError(f"invalid CUDA device id: {device_id}")

    device = torch.device(f"cuda:{device_id}")
    element_size = torch.empty((), dtype=torch.float32).element_size()
    numel = size_mb * 1024 * 1024 // element_size
    bytes_per_copy = numel * element_size

    host = torch.empty(numel, dtype=torch.float32, pin_memory=pinned)
    device_tensor = torch.empty(numel, dtype=torch.float32, device=device)

    h2d_times = _measure_copies(
        torch=torch,
        repeats=repeats,
        warmup=warmup,
        copy=lambda: device_tensor.copy_(host, non_blocking=pinned),
    )
    d2h_times = _measure_copies(
        torch=torch,
        repeats=repeats,
        warmup=warmup,
        copy=lambda: host.copy_(device_tensor, non_blocking=pinned),
    )

    return [
        _result(device_id, "h2d", pinned, bytes_per_copy, repeats, h2d_times),
        _result(device_id, "d2h", pinned, bytes_per_copy, repeats, d2h_times),
    ]


def benchmark_multi_gpu_h2d_bandwidth(
    *,
    device_ids: list[int] | None = None,
    size_mb: int = 256,
    repeats: int = 10,
    warmup: int = 3,
    pinned: bool = True,
    stagger_seconds: float = 0.0,
) -> MultiGPUCopyBandwidthResult:
    """Measure aggregate H2D bandwidth when several GPUs copy from host memory.

    ``stagger_seconds=0`` launches copies as close together as Python can manage.
    A positive stagger inserts a delay between device launches to expose whether
    staggering reduces PCIe/NUMA contention.
    """

    if size_mb <= 0:
        raise ValueError("size_mb must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if stagger_seconds < 0:
        raise ValueError("stagger_seconds must be non-negative")

    torch = _import_torch_with_cuda()
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        raise ValueError("device_ids must not be empty")

    for device_id in device_ids:
        if device_id < 0 or device_id >= torch.cuda.device_count():
            raise ValueError(f"invalid CUDA device id: {device_id}")

    element_size = torch.empty((), dtype=torch.float32).element_size()
    numel = size_mb * 1024 * 1024 // element_size
    bytes_per_device = numel * element_size

    hosts = [torch.empty(numel, dtype=torch.float32, pin_memory=pinned) for _ in device_ids]
    devices = [torch.device(f"cuda:{device_id}") for device_id in device_ids]
    device_tensors = [torch.empty(numel, dtype=torch.float32, device=device) for device in devices]
    streams = [torch.cuda.Stream(device=device) for device in devices]

    for _ in range(warmup):
        _launch_h2d_copies(torch, hosts, device_tensors, streams, pinned, stagger_seconds)
        _sync_devices(torch, device_ids)

    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        _launch_h2d_copies(torch, hosts, device_tensors, streams, pinned, stagger_seconds)
        _sync_devices(torch, device_ids)
        timings.append(perf_counter() - start)

    mean_seconds = mean(timings)
    aggregate_bytes = bytes_per_device * len(device_ids)
    aggregate_gb_per_second = aggregate_bytes / mean_seconds / 1_000_000_000
    return MultiGPUCopyBandwidthResult(
        device_ids=device_ids,
        mode="staggered_h2d" if stagger_seconds else "simultaneous_h2d",
        pinned=pinned,
        bytes_per_device=bytes_per_device,
        repeats=repeats,
        mean_seconds=mean_seconds,
        aggregate_gb_per_second=aggregate_gb_per_second,
        per_device_gb_per_second=aggregate_gb_per_second / len(device_ids),
        stagger_seconds=stagger_seconds,
    )


def _launch_h2d_copies(torch, hosts, device_tensors, streams, pinned: bool, stagger_seconds: float) -> None:
    for host, device_tensor, stream in zip(hosts, device_tensors, streams, strict=True):
        with torch.cuda.stream(stream):
            device_tensor.copy_(host, non_blocking=pinned)
        if stagger_seconds:
            sleep(stagger_seconds)


def _sync_devices(torch, device_ids: list[int]) -> None:
    for device_id in device_ids:
        torch.cuda.synchronize(device_id)


def _import_torch_with_cuda():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("bandwidth benchmark requires PyTorch") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("bandwidth benchmark requires a CUDA device")
    return torch


def _measure_copies(*, torch, repeats: int, warmup: int, copy) -> list[float]:
    timings: list[float] = []

    for _ in range(warmup):
        copy()
    torch.cuda.synchronize()

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        copy()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end) / 1000.0)

    return timings


def _result(
    device_id: int,
    direction: str,
    pinned: bool,
    bytes_per_copy: int,
    repeats: int,
    timings: list[float],
) -> CopyBandwidthResult:
    mean_seconds = mean(timings)
    gb_per_second = bytes_per_copy / mean_seconds / 1_000_000_000
    return CopyBandwidthResult(
        device_id=device_id,
        direction=direction,
        pinned=pinned,
        bytes_per_copy=bytes_per_copy,
        repeats=repeats,
        mean_seconds=mean_seconds,
        gb_per_second=gb_per_second,
    )
