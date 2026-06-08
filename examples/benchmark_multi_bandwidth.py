"""Run an aggregate multi-GPU H2D bandwidth benchmark."""

from leantrain.hardware.bandwidth import benchmark_multi_gpu_h2d_bandwidth


if __name__ == "__main__":
    print(benchmark_multi_gpu_h2d_bandwidth())
