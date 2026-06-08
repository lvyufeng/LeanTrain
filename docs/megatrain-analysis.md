# MegaTrain Analysis

This note summarizes what LeanTrain should learn from the sibling MegaTrain project at `/mnt/data/lvyufeng/MegaTrain`.

MegaTrain is directly relevant because it already explores a RAM-centric training architecture: parameters and optimizer states live in CPU memory, while GPUs temporarily execute layers. LeanTrain should reuse the strongest ideas, but its goal should be broader: a memory-first runtime for PCIe multi-GPU workstations rather than a model-specific training script.

## MegaTrain's core idea

MegaTrain describes itself as a RAM-centric architecture for full-precision training of very large LLMs. Its practical execution model is:

1. Load HuggingFace model weights on CPU.
2. Keep master parameters and optimizer states in host RAM.
3. Move only the currently needed layer or block to GPU.
4. Run forward/backward on GPU.
5. Copy gradients back to CPU.
6. Update weights with a CPU optimizer.
7. Reuse GPU buffers for the next layer.

The main implementation is in:

- `/mnt/data/lvyufeng/MegaTrain/infinity/model/cpu_master.py`
- `/mnt/data/lvyufeng/MegaTrain/infinity/model/mp_state.py`
- `/mnt/data/lvyufeng/MegaTrain/infinity/model/mp_worker.py`

The README and some older files are useful for context, but `cpu_master.py` is the production path worth studying.

## Features worth reusing

### 1. CPU master parameters

MegaTrain's strongest design choice is to make CPU RAM the owner of training state. GPU memory is a temporary workspace, not the source of truth.

LeanTrain should preserve this principle, but generalize it into explicit residency tracking:

```text
logical tensor -> physical storage -> residency state -> device view
```

The runtime should always know whether a tensor is in pageable host RAM, pinned host RAM, GPU VRAM, NVMe, or some future memory tier.

### 2. Flat pinned layer buffers

MegaTrain flattens layer parameters into pinned CPU memory and copies them into GPU staging buffers. This is better than copying many small tensors one by one.

LeanTrain should turn this into a first-class abstraction:

```text
ParameterGroup
  tensors: [q_proj.weight, k_proj.weight, v_proj.weight, ...]
  host_layout: flat pinned buffer
  device_layout: views into a GPU staging buffer
  structure_id: shared by compatible layers
```

This is important for PCIe systems because large contiguous DMA transfers are easier to schedule and measure.

### 3. Double-buffered H2D transfer

MegaTrain overlaps GPU compute for one layer with CPU→GPU prefetch for the next layer. LeanTrain should adopt this as the minimum viable streaming pattern.

The initial scheduler can be:

```text
prefetch layer i+1 while computing layer i
swap buffers
repeat
```

Later extensions should support:

- N-buffer prefetch;
- forward and backward prefetch policies;
- per-GPU prefetch queues;
- staggered multi-GPU copies;
- topology-aware copy grouping.

### 4. Recompute/checkpoint interval

MegaTrain saves only periodic activations and recomputes between checkpoints during backward. This is essential when GPU memory is small.

LeanTrain should not treat checkpoint interval as only a manual knob. It should become a planner decision based on:

- GPU memory budget;
- CPU RAM budget;
- expected H2D/D2H bandwidth;
- recompute cost;
- layer size;
- sequence length;
- attention workspace size.

### 5. Gradient slab pool

MegaTrain uses a fixed pool of gradient slabs to move GPU gradients back to CPU and accumulate them asynchronously. This avoids allocation churn and gives the runtime backpressure when slabs are exhausted.

LeanTrain should keep this pattern and improve it with:

- per-GPU slab pools;
- NUMA-local slab allocation;
- bounded pinned memory accounting;
- optional gradient compression;
- explicit D2H scheduling;
- better profiling of gradient transfer overlap.

### 6. Structure-based layer grouping

MegaTrain groups layers by structure, which helps handle hybrid attention and MoE models where not all decoder blocks have the same parameter layout.

LeanTrain should keep the idea, but name it more generally:

```text
ExecutionTemplate
  compatible parameter structure
  compatible forward signature
  required kernels
  maximum workspace size
```

A template can then be instantiated on GPU staging buffers without reallocating a new module for every layer.

### 7. Chunked LM-head loss

MegaTrain avoids materializing full `[batch, sequence, vocab]` logits at once by chunking the LM-head loss. This is important for both target machines, especially 2080 Ti.

LeanTrain should support chunked loss from the start, and later add:

- vocabulary chunking;
- sequence chunking;
- fused cross entropy where available;
- sampled or adaptive losses for extreme vocabularies.

### 8. GPU buffer release/rebuild

MegaTrain's VERL integration can release training GPU buffers so another engine such as SGLang can use the same GPU for rollout. This is a useful pattern for RL and mixed inference/training.

LeanTrain should model GPU buffers as leases:

```text
lease GPU workspace -> run training block -> release workspace
```

That allows future integration with rollout, evaluation, or serving engines.

### 9. Ref-in-actor pointer swap

For RLHF/GRPO/DPO, MegaTrain snapshots reference parameters and swaps CPU tensor pointers instead of maintaining a full second GPU model. This is elegant and memory-friendly.

LeanTrain should reuse this idea in any future RL adapter, while keeping it outside the core runtime.

## Features to reuse with caution

### Multi-GPU data parallel without NCCL

MegaTrain's multi-GPU mode uses separate worker processes. Workers read shared CPU weights and accumulate shared CPU gradients. This is simple and avoids distributed setup.

For LeanTrain, this is a good baseline but not enough.

On 8x RTX 4090, all GPUs may try to stream the same layer weights at the same time. Without NVLink, this can saturate host memory and PCIe bandwidth. A LeanTrain scheduler needs:

- copy staggering;
- NUMA-aware CPU memory placement;
- per-PCIe-group concurrency limits;
- pipeline parallel alternatives;
- profiling-driven scheduling decisions.

### HuggingFace introspection

MegaTrain introspects HuggingFace models to find embeddings, decoder layers, norms, and LM heads. This makes model support broad, but it can be brittle for exotic models, dynamic routing, VLMs, and custom `forward()` signatures.

LeanTrain should use introspection for early prototypes, then move toward a small intermediate representation that describes:

- parameters;
- operators;
- activation lifetimes;
- required kernels;
- memory locations;
- recompute boundaries.

### VERL integration

MegaTrain's VERL integration is valuable, especially for GRPO. But VERL is a large external framework. LeanTrain should keep RL integration behind an adapter and avoid letting RL-specific concerns shape the core memory runtime too early.

## Things not to base LeanTrain on

Some MegaTrain files appear experimental or stale relative to the main implementation:

- `/mnt/data/lvyufeng/MegaTrain/infinity/true_cpu_offloading.py`
  - Useful as a simple conceptual prototype, but not the production path.
- `/mnt/data/lvyufeng/MegaTrain/infinity/optimizer.py`
  - Standalone optimizer implementation; the main training path uses other optimizer handling.
- `/mnt/data/lvyufeng/MegaTrain/infinity/memory/manager.py`
  - Generic memory manager that does not appear to drive the current CPU-master training path.
- `/mnt/data/lvyufeng/MegaTrain/infinity/scheduler/executor.py`
  - Generic DAG executor idea, but not clearly wired into the production path.
- `/mnt/data/lvyufeng/MegaTrain/QUICKSTART.md`
  - Contains stale paths such as older `examples/train_cpu_master_v10.py` references.

These files can inspire names or concepts, but LeanTrain should study the current `cpu_master.py` path first.

## LeanTrain differentiation

LeanTrain should differ from MegaTrain in several deliberate ways.

### 1. Memory-first runtime, not only CPU-offloaded training

MegaTrain is RAM-centric, but much of the logic is embedded in one model/trainer path. LeanTrain should make memory management the core abstraction:

```text
TensorObject
MemoryTier
Residency
Transfer
Eviction
Recompute
ExecutionTask
```

The framework should be able to reason about memory before it reasons about model-specific training loops.

### 2. Bandwidth-aware multi-GPU scheduling

LeanTrain's initial hardware target includes 8x RTX 4090. That makes PCIe/NUMA bandwidth a first-class constraint.

LeanTrain should schedule not only compute tasks, but also copy tasks:

```text
H2D(parameter group L on GPU 3)
compute(forward L on GPU 3)
D2H(gradient group L from GPU 3)
evict(buffer L on GPU 3)
```

The scheduler should avoid launching all transfers at once just because all GPUs are idle.

### 3. Parallelism modes beyond data parallel

LeanTrain should support a progression of parallel modes:

1. Single-GPU layer streaming.
2. Host-backed data parallel.
3. Pipeline parallel over layer ranges.
4. Hybrid data + pipeline parallel.
5. Optional tensor parallel only where measurements justify it.

This is especially important for 4090 workstations where GPU-to-GPU communication is relatively expensive.

### 4. Legacy GPU profile

LeanTrain should explicitly support RTX 2080 Ti style constraints:

- FP16 instead of BF16;
- dynamic loss scaling;
- minimal residency;
- aggressive chunking;
- conservative attention kernels;
- small defaults;
- optimizer-state compression later.

MegaTrain's modern BF16-oriented defaults cannot simply be copied.

### 5. Measurement-first development

LeanTrain should start with hardware probes and microbenchmarks:

- H2D/D2H bandwidth;
- simultaneous multi-GPU transfers;
- pinned allocation behavior;
- NUMA effects;
- copy/compute overlap;
- attention workspace size;
- FP16/BF16 stability.

These measurements should feed scheduler decisions rather than remain external benchmark notes.

## Initial conclusion

MegaTrain proves that CPU-master, GPU-transient LLM training is practical. LeanTrain should use it as a reference implementation, especially for layer streaming, pinned flats, double buffering, recompute, gradient slabs, and HF model discovery.

LeanTrain's unique contribution should be a cleaner memory-first runtime and scheduler designed for large-RAM PCIe workstations: predictable residency, explicit transfer planning, NUMA-aware multi-GPU scheduling, and profiles for both modern 4090-class GPUs and older 2080 Ti-class GPUs.
