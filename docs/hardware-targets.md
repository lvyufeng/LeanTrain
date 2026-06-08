# Hardware Targets

LeanTrain starts from a specific deployment reality: workstation-scale machines with many consumer NVIDIA GPUs and very large host memory. The framework should not assume H100-style 80GB VRAM, NVLink/NVSwitch, or datacenter interconnects. It should treat CPU RAM as the center of training state and use GPUs as transient compute devices behind PCIe bandwidth constraints.

## Target A: 8x RTX 4090 + 2TB RAM

### Useful capacity

- GPU count: 8
- Per-GPU VRAM: 24GB
- Aggregate VRAM: 192GB
- Host RAM: 2TB
- Likely interconnect: PCIe, no NVLink between GPUs
- Native low precision: FP16 and BF16 are available on Ada GPUs

This machine has strong aggregate compute, but the effective training scale is usually limited by per-GPU VRAM and host-to-device bandwidth rather than raw FLOPs.

### Main constraints

1. **No NVLink assumption**
   - Cross-GPU communication should be treated as expensive.
   - Tensor parallelism is possible, but not the first default.
   - Pipeline parallelism and host-backed data parallelism are more natural first targets.

2. **PCIe and NUMA bandwidth dominate scaling**
   - If all 8 GPUs simultaneously stream the same layer weights from CPU RAM, host memory bandwidth and PCIe root complexes can saturate.
   - The scheduler must know which GPU is attached to which NUMA node and should place pinned buffers accordingly.

3. **Pinned memory is precious**
   - Pinned host memory improves H2D/D2H transfer, but too much pinned memory can degrade system performance.
   - LeanTrain should use bounded pinned pools rather than pinning arbitrary tensors.

4. **Layer-streaming needs staggering**
   - Naive data parallel workers may synchronize their CPU→GPU transfers and create bandwidth spikes.
   - The runtime should stagger prefetch phases or group GPUs by PCIe/NUMA topology.

5. **CPU optimizer state fits, but not for free**
   - A first-order Adam-style CPU memory estimate is roughly 12 bytes per parameter for FP32 master weights plus first and second moments.
   - 2TB RAM gives a high theoretical ceiling, but dataloader memory, activations, pinned buffers, checkpointing, Python overhead, page cache, and rollout engines reduce usable capacity.

### Default policy for this target

LeanTrain should default to:

- BF16 training where numerically acceptable.
- CPU master parameters and optimizer states.
- Layer or block streaming with double-buffered H2D transfer.
- Bounded pinned CPU pools for parameter flats, activation checkpoints, and gradient slabs.
- Bandwidth-aware host-backed data parallelism as the first multi-GPU baseline.
- Pipeline parallelism across layer ranges as the first non-baseline multi-GPU mode.
- Tensor parallelism only when a layer cannot fit in 24GB or when measurements show the PCIe cost is acceptable.
- NUMA-aware allocation and worker binding.
- Continuous profiling of copy time, compute time, overlap ratio, CPU bandwidth, and per-GPU utilization.

### Key experiments

Before designing advanced schedulers, measure:

- Full measurement capture using `python -m leantrain.cli measure --devices all --output profiles/8x4090-measured.json`.
- Markdown report generation using `python -m leantrain.cli report profiles/8x4090-measured.json --output reports/8x4090.md`.
- Single-GPU H2D bandwidth from pageable and pinned memory using `python -m leantrain.cli bandwidth`.
- 8-GPU simultaneous H2D bandwidth using `python -m leantrain.cli multi-bandwidth --devices all`.
- Staggered H2D bandwidth using `python -m leantrain.cli multi-bandwidth --devices all --stagger-ms 2`.
- Per-GPU H2D bandwidth grouped by NUMA node or PCIe root complex.
- D2H gradient copy bandwidth under compute load.
- BF16 matmul/attention throughput for representative model blocks.
- Copy/compute overlap with one-buffer, two-buffer, and N-buffer prefetch.
- Data parallel scaling with synchronized versus staggered layer prefetch.

## Target B: 4x RTX 2080 Ti + 1TB RAM

### Useful capacity

- GPU count: 4
- Per-GPU VRAM: 11GB
- Aggregate VRAM: 44GB
- Host RAM: 1TB
- Likely interconnect: PCIe, no useful peer-to-peer bandwidth assumption
- Native low precision: FP16, no native BF16

This machine is a stricter target. It should be treated as the small-VRAM and legacy-precision profile.

### Main constraints

1. **Very small per-GPU VRAM**
   - 11GB leaves little room for layer weights, activations, attention workspaces, gradient slabs, temporary logits, and CUDA allocator fragmentation.
   - The default residency policy should be one layer or one small block at a time.

2. **No native BF16**
   - BF16 configurations from modern frameworks should not be copied directly.
   - FP16 requires loss scaling and more careful optimizer/update behavior.

3. **Attention backend limitations**
   - Newer Flash Attention kernels may be unavailable or slower on Turing.
   - LeanTrain should support conservative fallbacks such as PyTorch SDPA, xFormers where available, or explicit chunked attention paths.

4. **Throughput is secondary to feasibility**
   - The initial goal is stable training of smaller models or small fine-tuning workloads, not maximum FLOPs utilization.
   - Larger models require aggressive recompute, chunking, and possibly optimizer-state compression.

5. **Host memory still enables large state**
   - 1TB RAM is enough for CPU master training state for models far larger than 11GB VRAM, but runtime overhead and slow PCIe streaming make practical limits much lower than theoretical RAM capacity.

### Default policy for this target

LeanTrain should default to:

- FP16 training with dynamic loss scaling.
- FP32 CPU master weights and optimizer states initially.
- Optional 8-bit optimizer states later for larger models.
- One-layer or minimal-block GPU residency.
- Aggressive activation checkpointing and recompute.
- Small gradient slab counts.
- Aggressive chunked LM-head loss.
- Conservative attention backend selection.
- Smaller default sequence length and microbatch size.
- Clear OOM diagnostics and automatic batch-size/resource estimation.

### Key experiments

Measure separately from the 4090 target:

- FP16 stability with dynamic loss scaling on small transformer blocks.
- Maximum layer template size that fits reliably with attention workspace.
- Chunked LM-head memory savings versus overhead.
- Sequence length limits under one-layer residency.
- Single-GPU and four-GPU H2D bandwidth.
- Whether peer-to-peer transfer is usable; assume no until measured.

## Shared hardware model

LeanTrain should represent hardware explicitly rather than hiding it behind `cuda:N` strings.

A minimal hardware profile should include:

```yaml
hardware:
  host_ram_bytes: 2199023255552
  numa_nodes:
    - id: 0
      cpus: [0-31]
      memory_bytes: 1099511627776
    - id: 1
      cpus: [32-63]
      memory_bytes: 1099511627776
  gpus:
    - id: 0
      name: RTX 4090
      vram_bytes: 25769803776
      pci_bus_id: "00000000:3B:00.0"
      uuid: GPU-...
      numa_node: 0
      pcie_group: A
      supports_bf16: true
    - id: 1
      name: RTX 4090
      vram_bytes: 25769803776
      numa_node: 0
      pcie_group: A
      supports_bf16: true
```

The scheduler should consume this profile to decide:

- where CPU buffers live;
- which GPUs prefetch together;
- when to stagger copies;
- how many pinned slabs to allocate;
- whether BF16 is allowed;
- whether pipeline, data, or hybrid parallelism is the default.

## Practical design rule

For these machines, LeanTrain should optimize for **bounded, predictable memory residency** before peak throughput. A slow but stable training step is useful; an OOM-prone or bandwidth-saturating step is not.
