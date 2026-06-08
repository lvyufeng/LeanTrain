# LeanTrain Architecture

LeanTrain is a memory-first AI training framework for large-host-memory, consumer-GPU workstations. Its core assumption is that CPU RAM is the center of training state, while GPUs and future NPUs are coprocessors that temporarily execute selected work.

This document describes the initial architecture direction.

## Design principles

### 1. Memory is the source of truth

Training state should be owned by the memory runtime, not by a particular GPU module instance.

Parameters, optimizer states, gradients, activations, temporary workspaces, and checkpoint buffers should all be represented as memory objects with explicit location and lifetime.

### 2. Accelerators are transient executors

A GPU or NPU is a device that accepts tasks:

- copy this parameter group into device workspace;
- execute this operator/block;
- copy this gradient or activation back;
- release this workspace.

The framework should not assume the model permanently lives on the accelerator.

### 3. Data movement is part of the plan

CPU↔GPU copies, GPU↔GPU copies, pinned-memory staging, eviction, and recomputation are not hidden side effects. They are explicit tasks with cost and dependencies.

### 4. Predictability before peak throughput

The first goal is stable execution with bounded memory. Throughput optimization comes after the runtime can explain and control residency.

### 5. Hardware profiles matter

LeanTrain should behave differently on:

- 8x RTX 4090 + 2TB RAM;
- 4x RTX 2080 Ti + 1TB RAM;
- future GPU/NPU machines.

Dtype, attention backend, prefetch depth, parallelism mode, pinned buffer size, and checkpoint interval should all be profile-sensitive.

## System overview

```text
+-------------------------------------------------------------------+
|                              User API                              |
|   model loading · dataset · optimizer · training objective · hints  |
+--------------------------------+----------------------------------+
                                 |
+--------------------------------v----------------------------------+
|                           Model Adapter                            |
|   HF introspection · parameter groups · operator templates · IR     |
+--------------------------------+----------------------------------+
                                 |
+--------------------------------v----------------------------------+
|                              Planner                               |
|   lifetimes · memory budget · recompute policy · parallel strategy  |
+--------------------------------+----------------------------------+
                                 |
+--------------------------------v----------------------------------+
|                       Memory Runtime / Scheduler                   |
|   residency · prefetch · evict · copy queues · device work queues   |
+-----------+--------------------+--------------------+-------------+
            |                    |                    |
+-----------v--------+  +--------v---------+  +-------v-------------+
| Host Memory Manager|  | GPU Worker(s)    |  | NPU / Other Worker  |
| pageable+pinned    |  | transient compute|  | transient compute   |
+--------------------+  +------------------+  +---------------------+
```

## Core abstractions

### TensorObject

A logical tensor known to the runtime.

Fields should include:

- logical name;
- shape;
- dtype;
- size in bytes;
- owner state, such as parameter, gradient, activation, optimizer state, or temporary;
- lifetime interval;
- alias/view relationships;
- current residency;
- allowed memory tiers;
- recompute source if any.

### MemoryTier

A storage tier with different capacity and cost.

Initial tiers:

```text
GPU VRAM
Pinned Host RAM
Pageable Host RAM
NVMe / mmap checkpoint storage later
```

Future tiers can include remote memory, CXL memory, or NPU-local memory.

### Residency

The runtime's answer to: where is this tensor now, and is it valid?

A tensor may have multiple residences if copies are valid in multiple places. The runtime must know which one is authoritative.

Examples:

```text
param.master: valid in pageable host RAM
param.prefetch_flat: valid in pinned host RAM
param.device_view: valid in GPU 0 staging buffer
activation.checkpoint: valid in pinned host RAM
activation.recomputed: valid in GPU 1 workspace until block end
```

### TensorGroup / ParameterGroup

A group of tensors copied and executed together.

For transformer models, a group may correspond to one decoder layer or a fused block. It should support flat physical layout for efficient DMA:

```text
ParameterGroup(layer_12)
  logical tensors:
    self_attn.q_proj.weight
    self_attn.k_proj.weight
    mlp.up_proj.weight
    ...
  host flat buffer: pinned
  GPU views: offsets into staging buffer
```

### ExecutionTemplate

A reusable compute structure for compatible groups.

Examples:

- dense transformer decoder block;
- hybrid attention block;
- MoE block;
- embedding;
- LM head;
- optimizer update shard.

Execution templates separate model structure from memory placement.

### Task

A unit of scheduled work.

Initial task types:

- `H2D`: host to device copy;
- `D2H`: device to host copy;
- `ComputeForward`;
- `ComputeBackward`;
- `Recompute`;
- `AccumulateGradient`;
- `OptimizerStep`;
- `Evict`;
- `CheckpointWrite` later;
- `CheckpointRead` later.

Each task has dependencies, resource requirements, and estimated cost.

### HardwareProfile

A measured and/or configured description of the machine:

- host RAM capacity;
- NUMA nodes;
- CPU core groups;
- GPUs and VRAM;
- GPU-to-NUMA mapping;
- PCIe groups;
- supported dtypes;
- measured H2D/D2H bandwidth;
- copy concurrency limits;
- attention backend availability.

This profile should drive planner defaults.

## Execution model

### Single-GPU streaming baseline

The first execution mode should stream one layer or block at a time through a single GPU.

```text
1. Keep all master params and optimizer states in CPU RAM.
2. Flatten each layer's params into host layout.
3. Allocate two GPU staging buffers.
4. Prefetch layer i+1 while computing layer i.
5. Save selected activations to CPU according to checkpoint policy.
6. During backward, recompute blocks from checkpoints.
7. Copy gradients back through gradient slabs.
8. Accumulate/update on CPU.
```

This mode validates the memory runtime before multi-GPU complexity.

### Host-backed data parallel

The first multi-GPU mode should be data parallel with CPU master state.

```text
GPU 0: microbatch A, streams layer weights from CPU
GPU 1: microbatch B, streams layer weights from CPU
...
CPU: owns shared params and accumulates gradients
```

Unlike a naive implementation, LeanTrain should schedule H2D copies with awareness of PCIe/NUMA bandwidth. It should avoid all GPUs requesting the same large transfer simultaneously unless measurements show that it is safe.

### Pipeline parallel

Pipeline parallelism should be the first major differentiation from a pure CPU-offload trainer.

```text
GPU 0: layer range 0-7
GPU 1: layer range 8-15
GPU 2: layer range 16-23
GPU 3: layer range 24-31
```

Weights may still be CPU-owned, but layer ranges can have more persistent or semi-persistent staging policies. This reduces repeated streaming of the entire model to every GPU.

For PCIe workstations, pipeline parallelism may be more practical than tensor parallelism because it avoids frequent all-reduce/all-gather within every layer.

### Hybrid data + pipeline parallel

On 8x RTX 4090, a likely useful layout is:

```text
2 data-parallel replicas x 4 pipeline stages
```

or:

```text
4 data-parallel replicas x 2 pipeline stages
```

The planner should choose based on:

- model layer sizes;
- sequence length;
- microbatch count;
- H2D bandwidth;
- stage balance;
- gradient accumulation requirements.

### Tensor parallel later

Tensor parallelism should not be the default for the target machines because GPU-to-GPU communication is expensive without NVLink. It becomes useful when:

- a single layer cannot fit in one GPU's staging/workspace budget;
- long-context attention or very wide MLPs require splitting;
- measured peer-to-peer bandwidth is good enough;
- the user accepts communication overhead for feasibility.

## Planner responsibilities

The planner should produce a task graph from a model and hardware profile.

Initial decisions:

- dtype policy: BF16 on 4090, FP16 with loss scaling on 2080 Ti;
- parameter grouping;
- activation checkpoint interval;
- prefetch depth;
- pinned memory budget;
- gradient slab count;
- chunked LM-head size;
- attention backend;
- single-GPU, data-parallel, or pipeline mode;
- recompute versus store decisions.

Later decisions:

- optimizer state compression;
- NVMe offload;
- dynamic scheduling based on profiler feedback;
- automatic microbatch and sequence-length tuning.

## Runtime responsibilities

The runtime should execute the planner's task graph while enforcing memory limits.

It must provide:

- pinned host allocator;
- pageable host allocator wrappers;
- GPU staging buffer allocator;
- stream/event management;
- copy queues;
- compute queues;
- eviction policy;
- gradient slab pool;
- CPU accumulation workers;
- profiling hooks;
- error reporting that identifies which memory budget failed.

A useful error should say, for example:

```text
OOM while allocating GPU 2 attention workspace for layer 17.
Budget: 11.0GB VRAM
Resident: 7.2GB layer staging, 1.1GB activation, 1.4GB temporary
Suggested actions: reduce seq_len, increase checkpointing, reduce LM-head chunk, use smaller block group.
```

## Model adapter responsibilities

The first adapter can target HuggingFace causal language models.

It should discover or define:

- embedding;
- decoder layers;
- final norm;
- LM head;
- attention masks and RoPE/cache behavior;
- parameter groups;
- compatible execution templates;
- loss function requirements.

Initially, introspection is acceptable. Over time, LeanTrain should move toward an explicit IR so unusual architectures are not handled by fragile special cases.

## Hardware probing

LeanTrain should include a probe command early:

```bash
leantrain probe --output hardware.yaml
```

It should measure:

- CPU RAM size;
- NUMA topology;
- GPU list and VRAM;
- dtype support;
- H2D/D2H bandwidth per GPU;
- simultaneous H2D bandwidth;
- pinned allocation limits;
- copy/compute overlap;
- attention backend availability.

The output becomes a `HardwareProfile` used by the planner.

## Development phases

### Phase 0: documentation and probes

- Document hardware targets.
- Analyze MegaTrain.
- Implement hardware probing and microbenchmarks.
- Produce machine profiles for 8x4090 and 4x2080Ti systems.

### Phase 1: single-GPU CPU-master prototype

- Load a small HF causal LM on CPU.
- Build parameter groups.
- Allocate pinned flats and GPU staging buffers.
- Run forward with one-layer streaming.
- Add double buffering.
- Add chunked LM-head loss.
- Add checkpoint/recompute backward.
- Add CPU optimizer step.

### Phase 2: profiling-driven tuning

- Report copy/compute/overlap timing.
- Auto-select prefetch depth and checkpoint interval.
- Tune pinned buffer budgets.
- Add OOM diagnostics.

### Phase 3: host-backed data parallel

- Spawn one worker per GPU.
- Share CPU master state.
- Schedule H2D transfers with bandwidth limits.
- Accumulate gradients on CPU.
- Compare synchronized and staggered prefetch.

### Phase 4: pipeline parallel

- Partition layers across GPUs.
- Add microbatch pipeline schedule.
- Manage CPU-backed stage boundaries.
- Compare against host-backed data parallel on 8x4090.

### Phase 5: legacy/small-VRAM profile

- Add RTX 2080 Ti profile defaults.
- Add dynamic loss scaling.
- Add aggressive chunking defaults.
- Validate stable FP16 fine-tuning on small models.

### Phase 6: advanced memory tiers

- Add NVMe/mmap checkpoint/offload tier.
- Add optimizer-state compression.
- Add adaptive scheduler feedback.

## Near-term repository layout

A possible initial layout:

```text
leantrain/
  __init__.py
  hardware/
    profile.py
    probe.py
    bandwidth.py
  memory/
    object.py
    allocator.py
    residency.py
  model/
    hf_adapter.py
    parameter_group.py
    templates.py
  planner/
    graph.py
    policies.py
  runtime/
    scheduler.py
    worker.py
    streams.py
    slabs.py
  training/
    loss.py
    optimizer.py
    loop.py
examples/
  probe_hardware.py
  train_tiny.py
  train_hf_causal_lm.py
docs/
  hardware-targets.md
  megatrain-analysis.md
  architecture.md
```

## Initial north star

LeanTrain should first prove this statement:

> On a PCIe workstation with large host RAM, a training runtime can keep model state in CPU memory, stream bounded working sets through consumer GPUs, and make explicit scheduling decisions that trade recompute, transfer, and residency without surprising OOMs.

Once this is true for one GPU, the core research question becomes how to schedule multiple GPUs without saturating shared host-memory and PCIe bandwidth.
