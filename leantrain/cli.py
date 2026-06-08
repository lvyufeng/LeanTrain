"""Command line entry points for LeanTrain."""

from __future__ import annotations

import argparse
import json

from leantrain.hardware.bandwidth import benchmark_copy_bandwidth
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

    return parser


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

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
