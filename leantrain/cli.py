"""Command line entry points for LeanTrain."""

from __future__ import annotations

import argparse
import json

from leantrain.hardware.bandwidth import benchmark_copy_bandwidth, benchmark_multi_gpu_h2d_bandwidth
from leantrain.hardware.probe import probe_hardware


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leantrain")
    subcommands = parser.add_subparsers(dest="command", required=True)

    probe = subcommands.add_parser("probe", help="Inspect the local hardware profile.")
    probe.add_argument(
        "--json",
        action="store_true",
        help="Print the hardware profile as JSON.",
    )

    bandwidth = subcommands.add_parser(
        "bandwidth",
        help="Measure CPU↔GPU copy bandwidth for one CUDA device.",
    )
    bandwidth.add_argument("--device", type=int, default=0, help="CUDA device id to benchmark.")
    bandwidth.add_argument("--size-mb", type=int, default=256, help="Copy size in MiB.")
    bandwidth.add_argument("--repeats", type=int, default=20, help="Measured copy repetitions.")
    bandwidth.add_argument("--warmup", type=int, default=5, help="Warmup copy repetitions.")
    bandwidth.add_argument(
        "--pageable",
        action="store_true",
        help="Use pageable host memory instead of pinned host memory.",
    )
    bandwidth.add_argument("--json", action="store_true", help="Print results as JSON.")

    multi_bandwidth = subcommands.add_parser(
        "multi-bandwidth",
        help="Measure aggregate multi-GPU host-to-device bandwidth.",
    )
    multi_bandwidth.add_argument(
        "--devices",
        type=str,
        default="all",
        help="Comma-separated CUDA device ids, or 'all'.",
    )
    multi_bandwidth.add_argument("--size-mb", type=int, default=256, help="Copy size per GPU in MiB.")
    multi_bandwidth.add_argument("--repeats", type=int, default=10, help="Measured repetitions.")
    multi_bandwidth.add_argument("--warmup", type=int, default=3, help="Warmup repetitions.")
    multi_bandwidth.add_argument(
        "--stagger-ms",
        type=float,
        default=0.0,
        help="Delay between launching each GPU copy, in milliseconds.",
    )
    multi_bandwidth.add_argument(
        "--pageable",
        action="store_true",
        help="Use pageable host memory instead of pinned host memory.",
    )
    multi_bandwidth.add_argument("--json", action="store_true", help="Print result as JSON.")

    return parser


def _parse_devices(value: str) -> list[int] | None:
    if value == "all":
        return None
    try:
        devices = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("devices must be 'all' or comma-separated integers") from exc
    if not devices:
        raise argparse.ArgumentTypeError("devices must not be empty")
    return devices


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "probe":
        profile = probe_hardware()
        if args.json:
            print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(profile.summary())
        return

    if args.command == "bandwidth":
        results = benchmark_copy_bandwidth(
            device_id=args.device,
            size_mb=args.size_mb,
            repeats=args.repeats,
            warmup=args.warmup,
            pinned=not args.pageable,
        )
        if args.json:
            print(json.dumps([result.to_dict() for result in results], indent=2, ensure_ascii=False))
        else:
            for result in results:
                host_memory = "pinned" if result.pinned else "pageable"
                print(
                    f"cuda:{result.device_id} {result.direction.upper()} "
                    f"{host_memory} {result.bytes_per_copy / 1024**2:.1f} MiB: "
                    f"{result.gb_per_second:.2f} GB/s "
                    f"({result.mean_seconds * 1000:.3f} ms avg, n={result.repeats})"
                )
        return

    if args.command == "multi-bandwidth":
        result = benchmark_multi_gpu_h2d_bandwidth(
            device_ids=_parse_devices(args.devices),
            size_mb=args.size_mb,
            repeats=args.repeats,
            warmup=args.warmup,
            pinned=not args.pageable,
            stagger_seconds=args.stagger_ms / 1000.0,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            host_memory = "pinned" if result.pinned else "pageable"
            print(
                f"cuda:{','.join(str(device) for device in result.device_ids)} "
                f"{result.mode} {host_memory} {result.bytes_per_device / 1024**2:.1f} MiB/GPU: "
                f"aggregate={result.aggregate_gb_per_second:.2f} GB/s, "
                f"per_gpu={result.per_device_gb_per_second:.2f} GB/s "
                f"({result.mean_seconds * 1000:.3f} ms avg, n={result.repeats})"
            )
        return

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
