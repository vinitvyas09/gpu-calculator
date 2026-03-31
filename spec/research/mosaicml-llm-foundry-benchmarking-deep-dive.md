# MosaicML LLM Foundry Benchmarking -- Deep Dive

**Source**: https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/benchmarking/README.md  
**Repo**: https://github.com/mosaicml/llm-foundry  
**Companion**: https://github.com/mosaicml/composer (SpeedMonitor callback)  
**Reference paper**: Korthikanti et al., 2022 (https://arxiv.org/abs/2205.05198) -- "Reducing Activation Recomputation in Large Transformer Models"

---

## 1. MFU and HFU Formulas (Exact Implementation)

### 1.1 Core Definitions

MosaicML defines MFU and HFU as follows (from the README and `collect_results.py`):

**MFU (Model FLOPs Utilization)**: Uses the theoretical minimum FLOPs for one forward+backward pass (3x forward). Independent of implementation details like activation checkpointing.

**HFU (Hardware FLOPs Utilization)**: Attempts to account for actual executed FLOPs, including recomputation from activation checkpointing (4x forward instead of 3x).

### 1.2 MFU Formula

```
flops_per_token = 2 * n_params
flops_per_seq = flops_per_token * seq_len

attn_flops_per_seq = n_layers * 2 * 2 * (d_model * (seq_len**2))

MFU = (3 * flops_per_seq + 3 * attn_flops_per_seq) * throughput_seq_per_sec / (num_gpus * GPU_AVAILABLE_FLOPS)
```

Breakdown of `3 * flops_per_seq`:
- `2 * n_params` = FLOPs per token (each param does 1 MAC = 2 FLOPs)
- `* seq_len` = FLOPs per sequence
- `* 3` = 3 passes: forward, backward (activations), backward (parameter gradients)

Breakdown of `attn_flops_per_seq = n_layers * 2 * 2 * (d_model * seq_len^2)`:
- There are 2 attention operations per layer: Q*K^T and attn*V
- Each operation is 2 FLOPs per element (MAC)
- Matrix size: d_model * seq_len^2 per layer
- This is then multiplied by 3 (same 3-pass reasoning) in the MFU formula

### 1.3 HFU Formula

```python
if activation_checkpointing:
    HFU = (4 * flops_per_seq + 4 * attn_flops_per_seq) * throughput_seq_per_sec / (num_gpus * GPU_AVAILABLE_FLOPS)
else:
    HFU = MFU  # same as MFU when no activation checkpointing
```

The factor changes from 3 to 4 because activation checkpointing adds an extra forward pass recomputation during the backward pass, making it: 1 (fwd) + 1 (recompute fwd) + 2 (bwd) = 4 forward-equivalent passes.

**Important note from the README**: The HFU numbers are approximations. Actual HFU would be higher because:
1. Norm, activation function, and residual FLOPs are excluded
2. All recomputation is not fully accounted for
3. With Flash Attention, there is an extra recompute factor (the attention multiplier would be **5 instead of 4** for Flash Attention recomputation in the forward pass)

### 1.4 The `flops_per_batch` Method (Runtime)

From `llmfoundry/models/mpt/modeling_mpt.py`:

```python
def flops_per_batch(self, batch: Mapping):
    bs, msl = batch['input_ids'].shape[0:2]
    params = self.n_active_params
    params_flops_per_token = 2 * params
    params_flops_per_seq = params_flops_per_token * msl
    attn_flops_per_seq = self.get_attention_flops(msl)
    return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs

def get_attention_flops(self, msl: int) -> int:
    return (
        self.model.config.n_layers * 2 * 2 *
        (self.model.config.d_model * (msl**2))
    )
```

This method returns total FLOPs for a batch (forward + backward, no activation checkpointing overhead). The SpeedMonitor callback uses this to compute MFU at runtime.

### 1.5 Active Parameter Counting for FLOP Estimation

From `llmfoundry/models/utils/mpt_param_count.py`, `n_active_params` differs from `n_total_params`:

```python
def mpt_get_active_params(mpt_model) -> int:
    if mpt_model.config.ffn_config['ffn_type'] in ffns_with_megablocks:
        # MoE: active params = (params_per_expert / local_experts) * top_k
        params = megablocks_n_active_params(mpt_model)
    else:
        params = sum(p.numel() for p in mpt_model.parameters())
    if not mpt_model.model.transformer.config.tie_word_embeddings:
        # Embedding layers are lookup tables, NOT counted in FLOP computation
        params -= mpt_model.model.transformer.wte.weight.numel()
    return params
```

**Critical insight**: When `tie_word_embeddings=False`, the embedding weight is SUBTRACTED from active params for FLOPs computation because embedding lookups are not MACs -- they are table lookups. This means:

```
n_active_params (for FLOPs) = n_total_params - embedding_params (if not tied)
```

For MoE models, active params = `(expert_params / num_local_experts) * top_k + non_expert_params`.

---

## 2. GPU Peak FLOPS Dictionary

From `composer/callbacks/speed_monitor.py` -- the definitive table of peak theoretical FLOPS used for MFU/HFU computation:

```python
GPU_AVAILABLE_FLOPS = {
    'h200-sxm': {  # Same as H100 SXM
        'bf16': 1.979e15 / 2,    # = 989.5 TFLOPS
        'fp16': 1.979e15 / 2,    # = 989.5 TFLOPS
        'fp8':  3.958e15 / 2,    # = 1979 TFLOPS
        'tf32': 989e12 / 2,      # = 494.5 TFLOPS
        'fp32': 67e12,           # = 67 TFLOPS
    },
    'h100-sxm': {
        'bf16': 1.979e15 / 2,    # = 989.5 TFLOPS
        'fp16': 1.979e15 / 2,    # = 989.5 TFLOPS
        'fp8':  3.958e15 / 2,    # = 1979 TFLOPS
        'tf32': 989e12 / 2,      # = 494.5 TFLOPS
        'fp32': 67e12,           # = 67 TFLOPS
    },
    'h100-pcie': {
        'bf16': 1.513e15 / 2,    # = 756.5 TFLOPS
        'fp16': 1.513e15 / 2,    # = 756.5 TFLOPS
        'fp8':  3.026e15 / 2,    # = 1513 TFLOPS
        'tf32': 756e12 / 2,      # = 378 TFLOPS
        'fp32': 51e12,           # = 51 TFLOPS
    },
    'a100': {  # SXM and PCIe have same FLOP counts
        'bf16': 312e12,          # = 312 TFLOPS
        'fp16': 312e12,          # = 312 TFLOPS
        'tf32': 156e12,          # = 156 TFLOPS
        'fp32': 19.5e12,         # = 19.5 TFLOPS
    },
    'a10': {
        'bf16': 125e12,
        'fp16': 125e12,
        'tf32': 62.5e12,
        'fp32': 31.2e12,
    },
    'v100-sxm': {
        'fp16': 125e12,
        'fp32': 15.7e12,
    },
    'v100-pcie': {
        'fp16': 112e12,
        'fp32': 14e12,
    },
    't4': {
        'fp16': 65e12,
        'fp32': 8.1e12,
        'int8': 130e12,
    },
}
```

**Key detail**: NVIDIA publishes spec sheets with a 2x sparsity factor. MosaicML divides by 2 to get the dense (non-sparse) FLOPS, which is the relevant number for training. The `/ 2` in the H100 values reflects this.

---

## 3. Throughput Benchmark Data (Validation Data)

### 3.1 Model Architecture Dimensions

All MPT models use `expansion_ratio=4` and `vocab_size=50368`:

| Model | d_model | n_heads | n_layers | Actual n_params (from benchmarks) |
|-------|---------|---------|----------|-----------------------------------|
| 125m  | 768     | 12      | 12       | 125,311,488                       |
| 350m  | 1024    | 16      | 24       | 355,985,408                       |
| 760m  | 1536    | 12      | 24       | 760,470,528                       |
| 1b    | 2048    | 16      | 24       | 1,315,950,592                     |
| 3b    | 2560    | 20      | 32       | 2,651,837,440                     |
| 7b    | 4096    | 32      | 32       | 6,658,859,008                     |
| 13b   | 5120    | 40      | 40       | 12,853,954,560                    |
| 30b   | 7168    | 56      | 48       | 29,975,214,080                    |
| 70b   | 8192    | 64      | 80       | 64,862,437,376                    |

Note: n_heads modified for some models to satisfy `d_head=128` (FlashAttention requirement):
- 760m: 16 -> 12 heads
- 1b: 24 -> 16 heads
- 3b: 32 -> 20 heads

Also note: `n_params` varies with seq_len because position embeddings change size. The values above are for `seq_len=2048`.

### 3.2 H100 80GB BF16 -- Key Throughput Data Points (8 GPUs, seq_len=2048)

| Model | MFU% | HFU% | Act.Ckpt | Sharding | T/s/GPU | MicroBS |
|-------|------|------|----------|----------|---------|---------|
| 70b   | 42.57| 56.76| True     | FULL_SHARD| 1,039  | 8 (64 GPUs) |
| 30b   | 38.11| 50.82| True     | FULL_SHARD| 2,002  | 3       |
| 13b   | 39.79| 39.79| False    | FULL_SHARD| 4,792  | 2       |
| 7b    | 46.44| 46.44| False    | FULL_SHARD| 10,643 | 6       |
| 3b    | ~42  | ~42  | False    | FULL_SHARD| ~20,964| 5 (4096 sl) |
| 1b    | 41.82| 41.82| False    | FULL_SHARD| 45,455 | 14      |

### 3.3 A100 80GB BF16 -- Key Throughput Data Points (8 GPUs, seq_len=2048)

| Model | MFU% | HFU% | Act.Ckpt | Sharding | T/s/GPU | MicroBS |
|-------|------|------|----------|----------|---------|---------|
| 30b   | 55.30| 73.74| True     | FULL_SHARD| 916    | 3       |
| 13b   | 59.57| 59.57| False    | FULL_SHARD| 2,262  | 2       |
| 7b    | 64.23| 64.23| False    | FULL_SHARD| 4,641  | 6       |
| 3b    | 62.11| 62.11| False    | FULL_SHARD| 10,811 | 10      |
| 1b    | 59.86| 59.86| False    | FULL_SHARD| 20,513 | 14      |

### 3.4 H100 80GB BF16 Large-Scale (128-512 GPUs, seq_len=2048)

| Model | GPUs | MFU% | HFU% | Act.Ckpt | Sharding | T/s/GPU |
|-------|------|------|------|----------|----------|---------|
| 70b   | 512  | 41.25| 55.0 | True     | FULL_SHARD| 1,007  |
| 70b   | 256  | 42.42| 56.56| True     | FULL_SHARD| 1,035  |
| 70b   | 128  | 43.36| 57.81| True     | FULL_SHARD| 1,058  |
| 30b   | 512  | 40.27| 53.69| True     | FULL_SHARD| 2,115  |
| 13b   | 512  | 41.12| 54.83| True     | FULL_SHARD| 4,952  |
| 7b    | 512  | 42.2 | 42.2 | False    | FULL_SHARD| 9,670  |
| 3b    | 512  | 39.24| 39.24| False    | SHARD_GRAD_OP| 21,664|
| 1b    | 512  | 36.65| 36.65| False    | SHARD_GRAD_OP| 39,837|

**Key observations for MFU scaling**:
- MFU decreases slightly as GPU count increases (communication overhead)
- 70b on 128 GPUs: 43.36% vs 512 GPUs: 41.25% (about 5% relative loss)
- Smaller models lose more MFU at scale (1b drops from ~42% at 8 GPUs to ~37% at 512)
- A100 shows higher MFU% than H100 -- this is because MFU is relative to peak. A100 peak BF16 is 312 TFLOPS, H100 is 989.5 TFLOPS. The H100 has higher absolute throughput but lower utilization percentage.

### 3.5 Sequence Length Impact on MFU (H100 80GB, 8 GPUs)

For 7B model:

| SeqLen | MFU% | HFU% | Act.Ckpt | T/s/GPU |
|--------|------|------|----------|---------|
| 512    | 42.83| 57.11| True     | 10,203  |
| 1024   | 42.83| 57.11| True     | 10,203  |
| 2048   | 46.44| 46.44| False    | 10,643  |
| 4096   | 40.42| 53.90| True     | 8,611   |
| 8192   | 37.14| 49.52| True     | 6,935   |
| 16384  | ---  | ---  | True     | ---     |
| 32768  | 30.94| 41.25| True     | 3,318   |
| 65536  | 28.59| 38.13| True     | 1,956   |

**Key insight**: MFU drops significantly at long sequence lengths. This is because:
1. The quadratic attention cost becomes dominant
2. Activation checkpointing becomes necessary (reducing effective throughput)
3. Communication overhead increases with larger activations

### 3.6 H100 FP8 Results

| Model | SeqLen | GPUs | MFU% | HFU% | Model TFLOP | T/s/GPU |
|-------|--------|------|------|------|-------------|---------|
| 3b    | 2048   | 8    | 27.70| 27.70| 548         | 30,586  |
| 3b    | 8192   | 8    | 23.28| 23.28| 460         | 19,146  |
| 1b    | 8192   | 8    | 20.71| 20.71| 409         | 32,010  |

**Note on FP8 MFU**: The MFU percentages look low, but this is because MFU is computed against the FP8 peak (1979 TFLOPS). The Model TFLOP column shows the actual TFLOPS achieved (548 for 3B FP8 vs 418 for 3B BF16), which is a ~30% improvement in absolute throughput.

---

## 4. Parallelism Configurations

### 4.1 FSDP Sharding Strategy Choices

MosaicML uses FSDP (Fully Sharded Data Parallelism) exclusively -- no tensor parallelism or pipeline parallelism in these benchmarks.

| Model Size | Sharding Strategy | Activation Checkpointing | Notes |
|------------|-------------------|--------------------------|-------|
| <= 3B      | SHARD_GRAD_OP     | False                    | At large scale (128+ GPUs); equivalent to ZeRO Stage 2 |
| <= 3B      | FULL_SHARD        | False                    | At smaller scale (8-64 GPUs) |
| 7B         | FULL_SHARD        | False                    | At seq_len=2048; True at longer seq_len |
| 13B        | FULL_SHARD        | True at large scale      | False at small scale with short seq |
| 30B+       | FULL_SHARD        | True                     | Always |
| 70B        | FULL_SHARD        | True                     | Always |

FSDP sharding strategies map to ZeRO stages:
- `FULL_SHARD` = ZeRO Stage 3 (shards params + grads + optimizer states)
- `SHARD_GRAD_OP` = ZeRO Stage 2 (shards grads + optimizer states only)
- `NO_SHARD` = DDP (no sharding)

### 4.2 Mixed Precision Modes

| GPU/Scale | Precision | MP Mode |
|-----------|-----------|---------|
| H100 large-scale (128+ GPUs) | amp_bf16 | DEFAULT |
| H100 small-scale (8-64 GPUs) | amp_bf16 | DEFAULT (some runs) |
| A100 80GB | amp_bf16 | DEFAULT |
| A100 80GB | bf16 | PURE |
| A100 40GB | bf16 | PURE |
| 70B any | bf16 | PURE |

`PURE` mixed precision means all params/grads/optimizer states in bf16 (reduces memory). `DEFAULT` means optimizer states in fp32, parameters cast during forward/backward.

### 4.3 Rough Memory Capacity Heuristic

From `submit_benchmarks.py`, line 573:

```python
def run_check_capacity(model_yaml, gpu_num, gpu_type, p_multiplier=16):
    # Extract param count in billions
    # Extract GPU memory in GB
    if p_multiplier * b_params > gpu_num * gpu_mem:
        return False  # Won't fit
    return True
```

They use `p_multiplier=4` when actually launching runs (line 618), meaning the rule of thumb is:

```
Required total GPU memory >= 4 * model_size_in_billions (GB)
```

For example: 7B model needs 4 * 7 = 28 GB total across all GPUs. With 8x 80GB GPUs = 640 GB total, this is easily satisfied. The 16x multiplier in the function default is more conservative.

This is a very rough heuristic for "will this OOM?" -- it does NOT account for optimizer states, activations, etc. precisely. It's just a quick filter.

---

## 5. What This Source Does NOT Cover

1. **No tensor parallelism or pipeline parallelism** -- all benchmarks use FSDP only
2. **No LoRA/PEFT** benchmarks
3. **No memory estimation formulas** -- this is a throughput benchmarking suite, not a memory calculator
4. **No GQA adjustment in attention FLOPs** -- the `get_attention_flops` method uses `d_model * seq_len^2` which assumes full MHA. For GQA models (fewer KV heads), this overestimates attention FLOPs
5. **No accounting for norm/activation/residual FLOPs** -- explicitly excluded from all FLOP calculations
6. **No cost estimation** -- only throughput/utilization metrics

---

## 6. What Is Unique / Non-Obvious

### 6.1 Embedding Subtraction from Active Params

The FLOP formula uses `n_active_params` which **excludes embedding weights** when embeddings are not tied. This is because embeddings are lookup tables, not matrix multiplications. Most naive FLOP calculators include embedding params, which overestimates FLOPs by:

```
overestimate = 2 * vocab_size * d_model * seq_len * 3 (for fwd+bwd)
```

For MPT-7B: `2 * 50368 * 4096 * 2048 * 3 = ~2.53 TFLOP` per batch -- not negligible.

### 6.2 Attention FLOPs Are Additive to Parameter FLOPs

The total FLOPs formula is:

```
total = 3 * (2 * n_params * seq_len) + 3 * (n_layers * 2 * 2 * d_model * seq_len^2)
      = 3 * param_flops + 3 * attention_flops
```

The attention term is NOT included in the `2 * n_params` approximation. The `2 * n_params` covers all linear layers (QKV projections, output projections, FFN), but the QK^T and Attn*V matrix multiplications are **additional** quadratic-in-seq-len operations that must be added separately.

### 6.3 HFU == MFU When No Activation Checkpointing

This is explicitly coded: `if not activation_checkpointing: hfu = mfu`. The 4x multiplier only applies when activation checkpointing is enabled. This matches the theory: without recomputation, the hardware executes exactly the model FLOPs.

### 6.4 Flash Attention Changes HFU Further

The README notes that with Flash Attention, the attention multiplier should be **5 instead of 4** because Flash Attention does its own recomputation in the forward pass. This is NOT implemented in their formulas -- it's called out as a known approximation.

### 6.5 MFU Paradox: A100 Shows Higher MFU% Than H100

A100 at 8 GPUs shows ~60-64% MFU for 7B, while H100 shows ~46%. This does NOT mean A100 is better utilized. It means A100's peak (312 TFLOPS) is closer to achievable throughput, while H100's peak (989.5 TFLOPS) is harder to saturate. In absolute tokens/sec, H100 is ~2.3x faster.

### 6.6 Small Models at Scale Lose MFU

1B model at 8 GPUs: ~42% MFU. At 512 GPUs: ~37% MFU. The communication overhead dominates when each GPU has very little work per step. This has practical implications: the calculator should warn users when they over-provision GPUs for small models.

### 6.7 The Model TFLOP Column

The benchmark tables include a `Model TFLOP` column which is the achieved TFLOPS per GPU:

```python
model_tflop = int((3 * flops_per_seq + 3 * attn_flops_per_seq) * throughput / gpus / 1e12)
```

This is useful for validating that the calculator's throughput estimates are in the right ballpark.

### 6.8 Microbatch Size Selection Pattern

From the sweep configurations, clear patterns emerge for how large the microbatch can be on H100 80GB with BF16:

| Model | Max microbatch_size (seq_len=2048, no act.ckpt) | With act.ckpt |
|-------|--------------------------------------------------|---------------|
| 125m  | 40                                               | N/A           |
| 350m  | 32                                               | N/A           |
| 760m  | 24                                               | N/A           |
| 1b    | 14                                               | N/A           |
| 3b    | 10                                               | N/A           |
| 7b    | 6                                                | 16 (at 4096)  |
| 13b   | 2 (no act.ckpt)                                  | 10 (at 4096)  |
| 30b   | N/A (always act.ckpt)                            | 3             |

These microbatch sizes can serve as validation data for the memory estimator.

---

## 7. Formulas Summary for Calculator Spec

### Total FLOPs per training step (per sequence)

```
F_per_seq = 3 * (2 * P_active * S + L * 4 * D * S^2)
```

Where:
- `P_active` = number of active parameters (excluding embedding lookup if untied)
- `S` = sequence length
- `L` = number of layers
- `D` = d_model (hidden dimension)
- The `3` accounts for forward + backward (2x forward for bwd)
- `4 = 2 * 2` in the attention term: 2 attention ops (QK^T, Attn*V) x 2 FLOPs per MAC

### MFU

```
MFU = F_per_seq * throughput_seqs_per_sec / (N_gpu * F_peak)
```

### HFU (with activation checkpointing)

```
HFU = (4/3) * F_per_seq * throughput_seqs_per_sec / (N_gpu * F_peak)
```

Or equivalently, replace the `3` multiplier with `4`:
```
F_per_seq_with_recompute = 4 * (2 * P_active * S + L * 4 * D * S^2)
HFU = F_per_seq_with_recompute * throughput_seqs_per_sec / (N_gpu * F_peak)
```

### Training Time Estimation (inverse of MFU)

```
T_seconds = total_training_FLOPs / (N_gpu * F_peak * MFU)
```

Where `total_training_FLOPs = F_per_seq * total_sequences = F_per_seq * (total_tokens / seq_len)`.

If using the common `6PD` approximation: `total_training_FLOPs = 6 * P * D` where D = total training tokens. The MosaicML formula adds the attention quadratic term on top.

---

## 8. Key File Paths in LLM Foundry Repository

- `/tmp/llm-foundry/scripts/train/benchmarking/README.md` -- benchmark tables, MFU/HFU formulas (prose)
- `/tmp/llm-foundry/scripts/train/benchmarking/collect_results.py` -- MFU/HFU computation code
- `/tmp/llm-foundry/scripts/train/benchmarking/submit_benchmarks.py` -- benchmark config generation, memory capacity heuristic
- `/tmp/llm-foundry/scripts/train/benchmarking/sweep.py` -- large-scale sweep configs (FSDP strategy choices)
- `/tmp/llm-foundry/scripts/train/benchmarking/sweep.sh` -- comprehensive sweep across all seq_len / model sizes
- `/tmp/llm-foundry/llmfoundry/models/mpt/modeling_mpt.py` -- `flops_per_batch()` and `get_attention_flops()` methods
- `/tmp/llm-foundry/llmfoundry/models/utils/mpt_param_count.py` -- active param counting (MoE-aware, embedding subtraction)
- `/tmp/llm-foundry/scripts/train/yamls/pretrain/mpt-*.yaml` -- model architecture configs
- `/tmp/composer/composer/callbacks/speed_monitor.py` -- GPU_AVAILABLE_FLOPS dictionary, SpeedMonitor callback (runtime MFU logging)

