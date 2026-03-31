# Deep Dive: BLOOM-176B Training Logs (BigScience)

**Source**: https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml
**Type**: Production training scripts, SLURM configs, engineering chronicles, math docs
**Date Reviewed**: 2026-03-31

---

## 1. Model Size Formula

The BigScience repo provides an explicit parameter count formula used across their entire model family. For standard GPT-2 architecture:

```
total_params = NLAYERS * (12 * NHIDDEN^2 + 13 * NHIDDEN) + VOCAB_SIZE * NHIDDEN + SEQ_LEN * NHIDDEN + 2 * NHIDDEN
```

For BLOOM specifically (uses ALiBi instead of positional embeddings, plus an extra embedding LayerNorm):

```
total_params = NLAYERS * (12 * NHIDDEN^2 + 13 * NHIDDEN) + VOCAB_SIZE * NHIDDEN + 4 * NHIDDEN
```

The `4*h` replaces `s*h + 2*h` because ALiBi removes the positional embedding matrix (`s*h`) and the embedding LayerNorm adds `2*h` (weight + bias) on top of the existing `2*h` for the final LayerNorm.

**Breakdown of per-transformer-block parameters** (`12*h^2 + 13*h`):
- QKV projection weights: `3 * h * h = 3h^2`
- Multi-head output projection weight: `h * h = h^2`
- QKV projection biases: `3 * h = 3h`
- Multi-head output projection bias: `h`
- Feed-forward expansion weight: `h * 4h = 4h^2`
- Feed-forward contraction weight: `4h * h = 4h^2`
- Feed-forward expansion bias: `4h`
- Feed-forward contraction bias: `h`
- LayerNorm 1 (weight + bias): `2h`
- LayerNorm 2 (weight + bias): `2h`
- **Total per block: 12h^2 + 13h**

**Quick approximation for large models** (where `12*l*h^2` dominates):
```
params_approx = 12 * NLAYERS * NHIDDEN^2
```

### Per-layer and per-embedding sizes (BLOOM-176B)

- NHIDDEN=14336, NLAYERS=70, VOCAB_SIZE=250880 (padded from 250680)
- **Embedding size**: `v * h = 250880 * 14336 = 3,596,615,680` params (7.2 GB in bf16)
- **One transformer block**: `12 * 14336^2 + 13 * 14336 = 2,466,437,120` params (4.9 GB in bf16)
- **Total model**: ~176B params

Key insight: with a 250K vocabulary, the embedding layer is approximately the same size as a single transformer block. This necessitated the `--pp-partition-method 'type:transformer|embedding'` rebalancing.

---

## 2. Memory Formulas

### 2a. Bytes per parameter for checkpoint storage

```
bf16 weights only: 2 bytes per param
Full checkpoint: 14 bytes per param
  - 8 bytes for fp32 optimizer states (Adam: momentum + variance)
  - 4 bytes for fp32 master weights
  - 2 bytes for bf16 working weights
```

**Validated against real data**:
- bf16 weights: `176e9 * 2 / 2^30 = 327.82 GB` (measured: 329 GB)
- Full checkpoint: `176e9 * 14 / 2^30 = 2294.77 GB` (measured: 2.3 TB)

### 2b. BF16Optimizer additional memory

The BF16Optimizer accumulates gradients in fp32 and maintains an **unsharded** static fp32 buffer. This costs an additional `4 bytes * params` of memory that is NOT split across GPUs. This is a trade-off: it saves communication overhead by not sharding, at the cost of extra memory per GPU.

So the actual per-GPU memory formula for BLOOM's setup (BF16 + custom ZeRO-1-equivalent) was:
```
per_gpu_memory = (model_params_on_this_gpu * 2)           # bf16 weights
               + (model_params_on_this_gpu * 4)           # fp32 master weights (unsharded grad buf)
               + (total_params / DP * 8)                  # fp32 optimizer states (sharded by ZeRO-1)
               + activations + CUDA_kernels_overhead
```

### 2c. Measured memory per GPU from topology experiments

The prequel document contains extensive empirical memory measurements across many configurations.

**Memory vs. topology (200B model, 48 nodes, various TP/PP/DP)**:

| DP | TP | PP | MBS | Mem/GPU |
|---:|---:|---:|----:|--------:|
| 12 |  8 |  4 |   1 |   47 GB |
|  9 |  8 |  5 |   1 |   44 GB |
|  8 |  8 |  6 |   1 |   39 GB |
|  6 |  8 |  7 |   1 |   39 GB |
|  6 |  8 |  8 |   1 |   36 GB |
|  5 |  8 |  9 |   1 |   37 GB |
|  4 |  8 | 10 |   1 |   35 GB |
|  4 |  8 | 11 |   1 |   32 GB |
|  4 |  8 | 12 |   1 |   30 GB |

Key observation: **More PP stages per GPU means less memory per GPU** (model is spread thinner), but also means fewer layers per GPU so the computation is more spread out. However, the above shows memory goes DOWN with more PP because each GPU holds fewer layers.

**Memory vs. MBS (microbatch size) at fixed topology**:

| PP | DP | MBS | Mem/GPU |
|---:|---:|----:|--------:|
| 10 |  4 |   1 |   35 GB |
| 10 |  4 |   2 |   43 GB |
| 10 |  4 |   4 |   55 GB |
| 10 |  4 |   8 |   76 GB |

Memory scales roughly linearly with MBS due to activation memory.

**Memory vs. DP with ZeRO-1**:

| GPUs | DP | Mem/GPU | Notes |
|-----:|---:|--------:|-------|
|   80 |  1 |   75 GB | 200B model, bf16 |
|  160 |  2 |   53 GB | 200B model, bf16 |

ZeRO-1 shards optimizer states across DP ranks, so doubling DP roughly halves the optimizer portion of memory.

### 2d. Memory formula for per-GPU with TP+PP

From the prequel analysis of 104B model:
```
params_per_gpu_per_layer = (12 * h^2 + 13 * h) / TP
params_per_gpu_embedding = (v * h) / TP
layers_per_gpu = NLAYERS / PP

total_params_per_gpu_with_emb = layers_per_gpu * params_per_gpu_per_layer + params_per_gpu_embedding
total_params_per_gpu_without_emb = layers_per_gpu * params_per_gpu_per_layer

memory_per_gpu = total_params_per_gpu * 18  # bytes (2 bf16 + 4 fp32 + 4+4 fp32 optimizer + 4 fp32 grads)
                 + activations_memory
                 + CUDA_kernel_overhead (~1.3-2 GB)
```

For 104B (TP=4, PP=32, 32GB GPUs):
- GPU with embedding: `2*403M + 146M = 953M params * 18 = 17 GB` + 4 GB activations + 1.3 GB CUDA = ~22 GB
- GPU without embedding: `2*403M = 807M params * 18 = 15 GB` + 2 GB activations + 1.3 GB CUDA = ~18 GB

### 2e. BF16 weight breakdown by component

```python
# BLOOM-176B specific
NHIDDEN=14336; NLAYERS=70; SEQ_LEN=2048; VOCAB_SIZE=250680
# BF16 (2 bytes per param):
# Transformer block: 2*(12*h^2 + 13*h) = 4.59 GB
# Embedding + other: 2*(v*h + 4*h) = 6.75 GB
# Total: 70*4.59 + 6.75 = 328.34 GB
```

---

## 3. FLOPS / Throughput Formulas

### 3a. TFLOPs calculation

```
TFLOPs = model_size_in_B * C * 2 * seqlen * global_batch_size / (time_per_iteration_sec * total_gpus * 1e3)
```

Where `C`:
- `C = 4` when using activation checkpointing ("hardware TFLOPs") -- accounts for recomputation
- `C = 3` without activation checkpointing ("model TFLOPs")

The factor breakdown: `C * 2` where:
- `2` = multiply + add operations
- `C = 3` = forward (1) + backward (2)
- `C = 4` = forward (1) + recomputation in backward (1) + backward (2)

Reference: Equation 3 of Section 5.1 of "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (arXiv:2104.04473).

### 3b. Training time estimation

```
training_time_seconds = 8 * T * P / (n * X)
```

Where:
- T = number of tokens (in billions, multiply by 1e9)
- P = number of parameters
- n = number of GPUs
- X = achieved TFLOPs per GPU

The factor of 8 breaks down as: `2 * (1 + 2 + 1)` = `multiply_add * (forward + backward + recomputation)`

**Example (BLOOM-176B)**:
```python
# 450B tokens, 176B params (actually ~167B for compute), 384 GPUs, 150 TFLOPs
training_days = 450e9 * 8 * 167e9 / (384 * 150e12 * 86400)
# = ~120.8 days
```

### 3c. Inference FLOPs per layer

```
FLOPs_per_layer = 24 * B * s * h^2 + 4 * B * s^2 * h
```
Where B = batch size, s = sequence length, h = hidden size.

---

## 4. Pipeline Parallelism Efficiency

### 4a. Pipeline bubble efficiency formula

From DeepSpeed documentation within the repo:

```
pipeline_efficiency = GAS / (GAS + PP - 1)
```

Where GAS = gradient accumulation steps = GBS / (MBS * DP).

For 90% efficiency: `GAS >= 8 * PP`
For 94% efficiency: `GAS >= 16 * PP`

### 4b. Pipeline fill condition

The pipeline is first fully filled when:
```
GBS >= PP * MBS * DP
```

For BLOOM-176B (PP=12, MBS=2, DP=8): first full fill at `GBS = 12 * 2 * 8 = 192`. This is why they changed the batch rampup to start at 192 instead of 16.

### 4c. PP partition rebalancing for large vocabularies

When `VOCAB_SIZE * NHIDDEN` is on the same order as a transformer block size (`12 * NHIDDEN^2 + 13 * NHIDDEN`), the embedding layers need to be treated as equivalent to transformer blocks for PP partitioning:

```
--pp-partition-method 'type:transformer|embedding'
```

This gives partition layout (BLOOM-176B, PP=12, 70 layers):
```
PP rank 0:  [embedding | 5 transformer blocks]
PP rank 1:  [6 transformer blocks]
...
PP rank 10: [6 transformer blocks]
PP rank 11: [5 transformer blocks | embedding]
```

Without this, ranks 0 and 11 would use much more memory and would be slower.

**Divisibility constraint change**: Instead of `NLAYERS % PP == 0`, the constraint becomes `(NLAYERS + 2) % PP == 0` when treating each embedding layer as an equivalent block.

---

## 5. Hardware Configuration (BLOOM-176B Production)

### Cluster
- **GPUs**: 416 A100 80GB GPUs (52 nodes), using 384 (48 nodes) + 32 (4 nodes) reserve
- **Per node**: 8 GPUs with NVLink 4 inter-GPU connects, 4 OmniPath links
- **CPU**: AMD, 512 GB RAM per node
- **GPU memory per node**: 640 GB (8 x 80 GB)
- **Inter-node**: Omni-Path Architecture (OPA) -- NOT InfiniBand
- **NCCL network**: Fully dedicated subnet

### Production parallelism config
- TP=4, PP=12, DP=8
- One replica = 48 GPUs (TP * PP = 4 * 12)
- 8 replicas (384 / 48 = 8)
- MBS=2, GBS=2048
- ZeRO stage 0 (BF16Optimizer implements its own ZeRO-1 equivalent internally)

### Why TP=4 instead of TP=8

Despite NVIDIA's general recommendation of `TP = GPUs per node` for large models, the BigScience team found through extensive benchmarking that **TP=4 was faster than TP=8** for their setup:

| DP | TP | PP | MBS | Sec/it | TFLOPs |
|---:|---:|---:|----:|-------:|-------:|
|  4 |  8 | 12 |   2 | 121.63 | 128.92 |
|  8 |  4 | 12 |   2 | 102.03 | 153.68 |

TP=4/PP=12 was ~19% faster than TP=8/PP=6. This is because:
1. TP communication (all-reduce) happens within every transformer block and is latency-sensitive
2. With OPA interconnect (not NVLink between nodes), reducing TP meant less inter-GPU communication overhead
3. Higher PP with lower TP allowed more DP (8 vs 4), and ZeRO-1 benefits from higher DP

---

## 6. Tile and Wave Quantization for GPU Efficiency

From the prequel experiments, choosing NHIDDEN for optimal GPU utilization:

```
A100 has 108 SMs (Streaming Multiprocessors)

For tile quantization: NHIDDEN % 128 == 0
For wave quantization: NHIDDEN % 108 == 0  (number of SMs)
For TP splitting:      NHIDDEN % TP == 0

Combining all three with TP=8:
  NHIDDEN = LCM(128, 108, 8) * c = 864 * c  (e.g., 864*16 = 13824)
```

This optimization produced a **massive** throughput improvement:

| NHIDDEN | Size | Sec/it | TFLOPs | Notes |
|--------:|-----:|-------:|-------:|-------|
|   14400 | 200B | 221.21 |  94.39 | Unoptimized |
|   13824 | 187B | 147.43 | 130.53 | Wave+tile optimized, MBS=4 |

That is a **38% throughput improvement** just from choosing the right hidden dimension for GPU wave/tile quantization.

For NHEADS, an important constraint for fused softmax kernels:
```
(NHEADS / TP) * MBS % 4 == 0  # required for optimized fused softmax
```

Benchmarked NHEADS impact (NHIDDEN=14336, TP=8, MBS=4):

| NHEADS | Sec/it | TFLOPs |
|-------:|-------:|-------:|
|     16 | 121.03 | 133.20 |
|     64 | 120.18 | 134.15 |  <-- fastest
|    112 | 138.72 | 116.21 |
|    128 | 124.89 | 129.08 |
|    256 | 132.85 | 121.35 |

NHEADS=64 was optimal (14336/64 = 224 head dimension).

---

## 7. Throughput Data Points

### BLOOM-176B steady-state
- **At GBS=2048 (full speed)**: 149-150 TFLOPs per GPU, ~105 sec/iteration
- **At GBS=16 (ramp start)**: 8 TFLOPs per GPU, ~15 sec/iteration
- **At GBS=192 (revised ramp start)**: 73 TFLOPs per GPU
- **At GBS=512**: ~113.5 TFLOPs (from batch ramp data)
- **Samples per second at full speed**: 19.46
- **A100 BF16 theoretical peak**: 312 TFLOPs
- **Achieved utilization**: 150/312 = **48% of theoretical peak**

### Performance impact of CUDA_LAUNCH_BLOCKING=1

Surprisingly, on this system at scale, `CUDA_LAUNCH_BLOCKING=1` (making all CUDA ops synchronous) had minimal to no throughput impact. This was necessary to prevent system-wide hanging with 40+ nodes.

### ZeRO-Stage performance comparison

| ZeRO Stage | Mem/GPU | Sec/it | TFLOPs |
|-----------:|--------:|-------:|-------:|
|          1 |   37 GB | 120.29 | 134.02 |
|          0 |   72 GB | 137.34 | 113.02 |

ZeRO-1 was ~18% faster than ZeRO-0 for the same topology because it reduced memory pressure enough to allow better computation overlap.

---

## 8. Batch Size Ramp-Up Strategy

### Formula
```
--rampup-batch-size <start_GBS> <increment> <ramp_samples>
```

BLOOM-176B final config: `--rampup-batch-size 192 16 9_765_625`

- Start at GBS=192 (first GBS where pipeline is fully filled: PP*MBS*DP = 12*2*8 = 192, though revised from original plan)
- Increase by 16 every `ramp_samples / ((target_GBS - start_GBS) / increment)` samples
- Target GBS=2048
- Ramp over 9,765,625 samples (~20B tokens)
- Continue at GBS=2048 for remaining ~430B tokens

### Steps calculation during ramp

Number of intervals: `(target_GBS - start_GBS) / increment = (2048 - 192) / 16 = 116`
Samples per interval: `9,765,625 / 116 = 84,186`

To find steps to reach a specific GBS during ramp:
```perl
perl -le '$x+=76894/(16*$_) for 1..$ARGV[0]/16; print int $x' TARGET_GBS
```

### Average batch size during ramp (for time estimation)
```
rampup_batch_size_avg = 0.5 * (global_batch_size + start_batch_size)
```

---

## 9. Checkpoint Size and Storage

### Per-checkpoint sizes
- **bf16 weights only**: 329 GB (2 bytes/param)
- **Full checkpoint (weights + optimizer)**: 2.3 TB (14 bytes/param)
  - bf16 weights: 2 bytes
  - fp32 weights: 4 bytes
  - fp32 optimizer (Adam momentum): 4 bytes
  - fp32 optimizer (Adam variance): 4 bytes

### Checkpoint timing
- Save time: ~40 seconds for 2.3 TB checkpoint
- Frequency: every 100 iterations (~3 hours of training)

### Checkpoint storage policy
- Keep last 15-20 intermediary checkpoints for rollback
- Back up every 1000th checkpoint to long-term storage
- Back up to Google Cloud Storage every 1-2 weeks

---

## 10. Training Time and Cost Data

### Total training timeline
- **Start**: March 11, 2022, 11:42am PST
- **Epoch 1 complete**: June 28, 2022 (iteration 85,376; 325B tokens consumed)
- **Switched from 48 to 24 nodes**: July 4, 2022
- **Planned training**: 450B tokens / 220M samples / ~115K iterations

### At-scale performance
- Iteration time: ~105 seconds at GBS=2048 on 48 nodes
- After node reduction (24 nodes): ~216 seconds per iteration
- Eval: 12 minutes for 29 tasks (0.7% of training time)
- Estimated total: ~120 days best case (no downtime)
- GPU-hours: `384 GPUs * 120 days * 24 hours = ~1.1M GPU-hours`

### Allocated compute
- ~3M GPU-hours total allocation
- Estimate: ~2M GPU-hours needed (2x safety margin over theoretical minimum of ~889K)

Formula for allocated hours sufficiency:
```python
compute_hours = 8 * tokens * params / (achieved_tflops_per_gpu * 3600)
# = 8 * 300e9 * 200e9 / (150e12 * 3600) = 888,889 GPU-hours
```

---

## 11. Smaller Model Family Configurations

The BigScience project trained a family of models with consistent architecture choices:

| Model | NLAYERS | NHIDDEN | NHEADS | TP | PP | DP | MBS | GBS | GPU Type | Nodes |
|------:|--------:|--------:|-------:|---:|---:|---:|----:|----:|---------:|------:|
|  350M |      14 |    1024 |     16 |  2 |  2 | 16 |   1 | 512 |  V100-32 |     8 |
|  760M |      24 |    1536 |     16 |  2 |  2 | 16 |   1 | 512 |  V100-32 |     8 |
| 1.3B  |      24 |    2048 |     16 |  2 |  2 | 16 |   1 | 512 |  V100-32 |    16 |
| 2.5B  |      30 |    2560 |     32 |  4 |  2 | 16 |   1 | 512 |  V100-32 |    32 |
| 6.3B  |      30 |    4096 |     32 |  4 |  4 |  8 |   1 | 512 |  V100-32 |    32 |
|  176B |      70 |   14336 |    112 |  4 | 12 |  8 |   2 |2048 | A100-80  |    48 |

Note: smaller models used V100-32GB with fp16, not bf16. Only the 176B used A100-80GB with bf16.

---

## 12. Sanity Check Constraints

These are hard constraints that must be satisfied:

```
NHIDDEN % NHEADS == 0                           # head dimension must be integer
GBS % (MBS * DP) == 0                           # batch must divide evenly
NLAYERS % PP == 0                               # layers must divide evenly across PP stages
(NHEADS / TP) * MBS % 4 == 0                    # fused softmax kernel constraint
NHIDDEN % 128 == 0                              # tile quantization
NHIDDEN % TP == 0                               # TP splitting
```

Additional constraints discovered during BLOOM training:
- Cannot change TP size when restarting from checkpoint
- Can change PP (via universal checkpoint conversion)
- Cannot change max LR on restart without `--override-lr-scheduler`

---

## 13. Engineering Lessons and Gotchas

### 13a. BF16 vs FP16 trade-offs
- BF16 has better numerical range (no overflow risks) but lower precision
- BF16 training was significantly more stable for BLOOM-176B
- BF16Optimizer accumulates gradients in fp32 (critical for stability)
- FP16 was used for smaller models on V100s (which don't support bf16)

### 13b. Initialization std matters enormously

Standard initialization `--init-method-std 0.02` (Megatron default) was too large for 100B+ models. Training couldn't get past 24B tokens with the default.

Working formulas:
- "Transformers without Tears": `sqrt(2 / (NHIDDEN * 5))` = `sqrt(0.4 / NHIDDEN)`
- MT-NLG 530B paper: `sqrt(1 / (NHIDDEN * 3))` = `sqrt(0.333 / NHIDDEN)`

BLOOM used the smaller (530B) init: for NHIDDEN=14336, `init_std = 0.0048`.

Internally, Megatron also applies a second rescaling for specific layers:
```python
std = sigma / math.sqrt(2.0 * num_layers)
```
This rescales the 2nd MLP layer and attention output projection.

### 13c. Optimizer choice matters for memory
- `torch.optim.Adam` on 200B model: 1st node 61 GB, all other nodes 47 GB
- `apex.optimizers.FusedAdam` on 200B model: 1st node 51 GB, all other nodes 44 GB
- FusedAdam saved ~10 GB on the first node and ~3 GB on other nodes

### 13d. Hardware failure frequency
- GPU crashes occurred roughly weekly
- Each crash lost up to 3 hours of work (100 iterations * ~105 sec/iteration)
- One slow GPU caused a 5% throughput drop (149 -> 140 TFLOPs) affecting the entire cluster
- Node identification for faulty GPUs was a major operational challenge

### 13e. Training hanging root causes
1. **NCCL communications at scale (40+ nodes)**: Solved by `CUDA_LAUNCH_BLOCKING=1`
2. **Evaluation hanging**: DataLoader race condition in PyTorch -- solved by setting `num_workers=0` for validation
3. **SLURM not quitting on crashes**: Solved by `NCCL_ASYNC_ERROR_HANDLING=1` and SRUN `--wait=60 --kill-on-bad-exit=1`

### 13f. Layer norm TP sync bug
The BF16Optimizer had a bug where gradient clipping was not applied on TP ranks > 1, causing layer norm weights to drift out of sync. Band-aid fix: `all_reduce` with `ReduceOp.AVG` on layer norm weights before each forward pass.

### 13g. Loss spikes
First loss spike at iteration 31,216. Loss went from 2.2 to 5.1 (lm loss). Recovered in ~30 iterations. Grad norm went from 0.2 to 960. No intervention needed -- the training was stable enough to self-recover.

---

## 14. Topology Selection Heuristics

From the BigScience team's extensive benchmarking:

### Rule of thumb for optimization priority
1. Use as much DP as possible (fastest)
2. Use PP next (slower than DP but memory-efficient)  
3. Use TP only when needed (communication-heavy, never beyond single node)

### Decision process
1. Determine minimum `TP * PP` needed to fit model in memory
2. Empirical rule: `model_size_bytes * 18 < 75% of GPU_memory` means no model parallelism needed
3. Prefer PP over TP unless a single layer doesn't fit on one GPU
4. Then try different MBS values (MBS=2 was generally optimal, MBS=4 close behind)
5. Ensure `GBS % (MBS * DP)` constraint is met for batch ramp-up

### Key finding: More GPUs != always faster
The team found that **TFLOPs per GPU can be high with fewer GPUs** while total wall-clock time is slower:
> "We are bound by compute of each GPU and we barely use half the GPU memory. Trying to pack more on each GPU slows the ensemble down."

The goal is minimum time per iteration, not maximum TFLOPs per GPU.

---

## 15. What is Unique / Non-Obvious

1. **Embedding-as-transformer-block PP partitioning**: When vocab is large enough that embedding ≈ transformer block in size, treat it as an equivalent block for PP scheduling. This is not a common default in calculators.

2. **BF16 ZeRO-stage 0 that is actually ZeRO-1**: The BF16Optimizer implements its own ZeRO-1 equivalent internally, so the DeepSpeed config says `stage: 0` but optimizer states are actually sharded. Calculator must know about this.

3. **Unsharded fp32 grad accumulation buffer**: The BF16Optimizer keeps a full (unsharded) fp32 gradient buffer of `4 bytes * total_params_on_gpu`. This is a hidden memory cost not captured by standard ZeRO-1 formulas.

4. **Wave/tile quantization impact**: Choosing NHIDDEN to align with GPU SM count (108 for A100) and tile size (128) produced a 38% throughput improvement. This is not a memory formula but dramatically affects training time estimates.

5. **TP < GPUs_per_node can be faster**: Counter to conventional wisdom, TP=4 was faster than TP=8 on A100 nodes for BLOOM. The OPA interconnect (not InfiniBand) and the specific model architecture made lower TP + higher PP + higher DP the winning combination.

6. **CUDA_LAUNCH_BLOCKING=1 has minimal throughput impact at scale**: Making CUDA synchronous, which should theoretically slow everything down, had negligible impact on 384-GPU training. This is because at scale, communication latency dominates over CUDA kernel launch latency.

7. **14 bytes per param for full checkpoints (not 18)**: Because BLOOM used bf16 (not fp16), the checkpoint stores bf16+fp32 weights plus fp32 optimizer states = 2+4+4+4 = 14 bytes, not the typical 18 bytes assumed for mixed-precision fp16 training (where you have fp16+fp32 weights + fp32 optimizer = 2+4+4+4+4 = 18, though this varies by framework).

8. **Days left estimation formula**: A practical on-the-fly formula used during training:
```perl
perl -le 'print 105 * (341_000_000_000-shift) / (2048*2048*3600*24)' CURRENT_CONSUMED_TOKENS
```
This computes `(time_per_iteration * remaining_tokens / (seq_len * GBS)) / seconds_per_day`.

---

## 16. Relevant File Paths in Cloned Repo

- `/tmp/bigscience-repo/train/tr11-176B-ml/README.md` -- Main documentation with all formulas and configs
- `/tmp/bigscience-repo/train/tr11-176B-ml/tr11-176B-ml.slurm` -- Production training SLURM script
- `/tmp/bigscience-repo/train/tr11-176B-ml/chronicles-prequel.md` -- Topology benchmarking data (65KB)
- `/tmp/bigscience-repo/train/tr11-176B-ml/chronicles.md` -- Training log with all incidents
- `/tmp/bigscience-repo/math/README.md` -- FLOPS and training time formulas
- `/tmp/bigscience-repo/train/memory.md` -- Activation partitioning note
- `/tmp/bigscience-repo/train/tflops_optimization.md` -- Topology optimization heuristics
- `/tmp/bigscience-repo/train/sanity-checks.md` -- Configuration constraints
- `/tmp/bigscience-repo/train/lessons-learned.md` -- Training stability lessons
- `/tmp/bigscience-repo/experiments/gpt2-utils.md` -- Model size calculation formula
- `/tmp/bigscience-repo/jz/frameworks/deepspeed.md` -- Pipeline efficiency formulas, memory estimates
- `/tmp/bigscience-repo/train/tr11-176B-ml/smaller_models/` -- All smaller model SLURM scripts
