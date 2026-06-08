"""Command line entry points for LeanTrain."""

from __future__ import annotations

import argparse
import json

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

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
