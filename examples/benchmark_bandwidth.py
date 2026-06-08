"""Run a CPU↔GPU copy bandwidth benchmark."""

from leantrain.hardware.bandwidth import benchmark_copy_bandwidth


if __name__ == "__main__":
    for result in benchmark_copy_bandwidth():
        print(result)
