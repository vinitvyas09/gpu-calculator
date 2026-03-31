# Meta Llama 3 Training -- Deep Dive for LLM Training GPU Calculator

**Sources examined:**
- Blog post: https://ai.meta.com/blog/meta-llama-3/
- Technical paper: "The Llama 3 Herd of Models" (arXiv 2407.21783)
- ISCA 2025 paper: "Scaling Llama 3 Training with Efficient Parallelism Strategies"
- HuggingFace model card: https://huggingface.co/meta-llama/Llama-3.1-405B
- HuggingFace blog: https://huggingface.co/blog/llama31

**Date:** 2026-03-31

---

## 1. Model Architecture Specifications

### 1.1 Architecture Table (All Sizes)

| Parameter          | Llama 3 8B | Llama 3 70B | Llama 3 405B |
|--------------------|-----------|------------|-------------|
| Layers             | 32        | 80         | 126         |
| Model Dimension    | 4,096     | 8,192      | 16,384      |
| FFN Dimension      | 14,336    | 28,672     | 53,248      |
| Attention Heads    | 32        | 64         | 128         |
| KV Heads (GQA)     | 8         | 8          | 8           |
| Peak Learning Rate | 3e-4      | 1.5e-4     | 8e-5        |
| Vocabulary Size    | 128,000   | 128,000    | 128,000     |
| RoPE Base Freq     | 500,000   | 500,000    | 500,000     |
| Context Window     | 128K      | 128K       | 128K        |

**Key architectural choices:**
- Standard decoder-only transformer (NOT mixture-of-experts) for training stability
- Grouped Query Attention (GQA) with only 8 KV heads regardless of model size
- SwiGLU activation function
- RMSNorm (not LayerNorm)
- 128K token vocabulary (100K from tiktoken3 + 28K for non-English languages)
- Token compression ratio: 3.94 characters/token (English), up from Llama 2's 3.17
- RoPE base frequency: theta = 500,000 (supports contexts up to 32,768 natively; extended to 128K)

### 1.2 Why 126 Layers Instead of 128

The 405B model uses 126 layers (not 128) as a deliberate pipeline parallelism optimization:
- The embedding layer on the first PP stage and the output head on the last PP stage create memory/compute imbalance
- One layer was removed from the first PP rank (to reduce peak memory) and one from the last PP rank (to balance computational workload)
- This optimization reduces maximum allocated memory by **5 GB** and improves TFLOPs by **6.5%**
- With this layer reduction, **activation recomputation can be turned off entirely** for 8K sequences

**Calculator relevance:** When modeling pipeline parallelism, must account for the fact that first and last stages carry extra memory from embedding/output head layers. The layer count should be adjusted (or memory per stage should be non-uniform) to reflect this.

---

## 2. Training Compute and Cost Data

### 2.1 Total Training Compute

| Model       | GPU Hours   | Power (W) | CO2 (Location) | CO2 (Market) |
|-------------|-------------|-----------|-----------------|--------------|
| Llama 3.1 8B  | 1.46M       | 700       | 420 tons        | 0 tons       |
| Llama 3.1 70B | 7.0M        | 700       | 2,040 tons      | 0 tons       |
| Llama 3.1 405B| 30.84M      | 700       | 8,930 tons      | 0 tons       |
| **Total**   | **39.3M**   | 700       | **11,390 tons** | **0 tons**   |

### 2.2 FLOPs Verification

For 405B model:
- Total training FLOPs: **3.8 x 10^25**
- Using the 6ND approximation: 6 * 405e9 * 15.6e12 = 3.79e25 (matches closely)
- This is ~50x more compute than Llama 2's largest model

### 2.3 Training Time Verification

Using the formula: `T = Total_FLOPs / (num_gpus * achieved_tflops * 1e12)`
- 3.8e25 / (16,384 * 400e12) = 5.80e6 seconds = ~67 days
- With 16,384 GPUs at 400 TFLOP/s: 30.84M GPU-hours / 16,384 = 1,882 hours = ~78 days wall-clock
- The discrepancy accounts for the fact that part of training was on 8,192 GPUs (Stage 1) and >90% effective training time

### 2.4 Cost Estimate

At ~$2/GPU-hour (cloud H100 rate): 39.3M * $2 = **~$78.6M total** (all three models)
For 405B alone: 30.84M * $2 = **~$61.7M**

---

## 3. 4D Parallelism Configuration

### 3.1 Parallelism Dimension Ordering

The four parallelism dimensions, in order of innermost to outermost:
1. **Tensor Parallelism (TP)** -- splits weight tensors across devices within a node
2. **Context Parallelism (CP)** -- splits input sequence across devices
3. **Pipeline Parallelism (PP)** -- partitions model vertically by layers
4. **Data Parallelism (DP/FSDP)** -- replicates model, shards optimizer/gradients

### 3.2 Parallelism Configuration Table (405B)

| Stage | GPUs   | TP | CP | PP | DP  | Seq Len  | Batch/DP | Tokens/Batch | TFLOP/GPU | BF16 MFU |
|-------|--------|----|----|----|----|----------|----------|-------------|-----------|----------|
| 1     | 8,192  | 8  | 1  | 16 | 64 | 8,192    | 32       | 16M          | 430       | 43%      |
| 2     | 16,384 | 8  | 1  | 16 | 128| 8,192    | 16       | 16M          | 400       | 41%      |
| 3     | 16,384 | 8  | 16 | 16 | 4  | 131,072  | 16       | 16M          | 380       | 38%      |

**Key observations for calculator:**
- TP=8 always (matches intra-node NVLink connectivity on 8-GPU servers)
- PP=16 always for 405B (126 layers / 16 = ~8 layers per stage, minus adjustments)
- Total GPUs = TP * CP * PP * DP (8 * 1 * 16 * 64 = 8,192; 8 * 1 * 16 * 128 = 16,384; 8 * 16 * 16 * 4 = 8,192... wait, that's 8,192 not 16,384)

**Correction on Stage 3:** 8 * 16 * 16 * 4 = 8,192, but the table says 16,384 GPUs. This means the product TP*CP*PP*DP does not directly equal total GPUs -- there may be additional dimensions or the DP value accounts for FSDP sharding across a different grouping. The paper states 16,384 GPUs were used; it is possible CP and DP overlap differently. Alternatively, the stage 3 DP=4 may mean 4 data-parallel replicas of (TP=8 * CP=16 * PP=16 = 2,048)-GPU units, giving 4 * 2,048 = 8,192 -- indicating only 8,192 GPUs were active for the long-context stage, or more likely that the numbers work out differently. **The paper explicitly states 16,384 GPUs for stage 3.** This warrants verification: TP=8, CP=16, PP=16, DP=4 => 8*16*16*4 = 8,192 -- which is half of 16,384. The discrepancy may mean DP includes FSDP sharding on top, effectively DP_shard * DP_replicas.

### 3.3 MFU Analysis

- MFU range: **38-43%** for BF16 on H100
- H100 BF16 peak: ~989 TFLOP/s (but the paper uses 1,000 TFLOP/s as the reference)
  - 430 TFLOP/s / 989 TFLOP/s = 43.5% (matches 43%)
  - 400 TFLOP/s / 989 TFLOP/s = 40.4% (matches 41%)
  - 380 TFLOP/s / 989 TFLOP/s = 38.4% (matches 38%)
- MFU drop from 43% to 41% when doubling GPUs attributed to lower batch size per DP group (32 -> 16) needed to keep total batch at 16M tokens
- MFU drop from 41% to 38% for long-context stage due to CP communication overhead (7.64% of total elapsed time)

### 3.4 Heuristic: How to Choose Parallelism Dimensions

Based on Llama 3 empirical decisions:

1. **TP = number of GPUs per node** (typically 8). TP uses NVLink which has the highest bandwidth. Never cross node boundaries with TP.
2. **PP = ceil(model_layers / layers_per_stage)**. Chosen to fit model in memory. For 405B: 126 layers / 16 stages ~ 8 layers/stage.
3. **CP = sequence_length / target_per_gpu_seq_length**. For 128K context: CP=16 so each GPU sees 8K tokens.
4. **DP = total_GPUs / (TP * CP * PP)**. Whatever is left after the other dimensions are set.
5. **Batch/DP** must be >= 2*PP for efficient pipeline scheduling (1F1B). If not, use ZeRO-2 with all-forward-all-backward schedule.

---

## 4. Pipeline Parallelism Details

### 4.1 Schedule: Interleaved 1F1B

Llama 3 uses an **interleaved 1F1B schedule** with V virtual pipeline stages per rank (not simple 1F1B).

**Pipeline bubble ratio formula:**
```
bubble_ratio = (PP - 1) / (V * M)
```
Where:
- PP = number of pipeline stages
- V = number of virtual stages per rank
- M = number of micro-batches

Equivalently: `bubble_ratio = (pp - 1) / (nmb * v)`

To minimize bubble: prefer smaller PP, more micro-batches (nmb), and more virtual stages (v).

The paper reports achieving a **~5% bubble ratio** in practice.

### 4.2 Warm-up Micro-batches Formula

Each pipeline stage with v virtual chunks processes a specific number of micro-batches before the first backward computation:

```
mu(s) = p * (v - 1) + 2 * (p - s - 1) + 1
```

Or equivalently from the ISCA25 paper:
```
warm_up_microbatches = (v - 1) * nc + 2 * (pp - ppr * v - 1)
```

Where:
- p = number of pipeline stages
- v = number of virtual stages per rank
- s = stage index (0-indexed)
- nc = number of consecutive micro-batches per stage
- ppr = pipeline rank index

### 4.3 Activation Memory in Pipeline Parallelism

Activation memory during warm-up phase on device 1 reduces from M_a (full) to:
```
activation_memory_per_device = (1 + delta) * M_a / p
```
Where: `delta = 2 * (p - 1) / n` and n = number of slices per sequence.

The warm-up phase determines peak activation memory because it accumulates activations from multiple micro-batches before any backward pass frees them.

### 4.4 Pipeline Parallelism + FSDP Interaction

Two strategies depending on batch size:
- **If batch_size >= 2 * PP**: Use FSDP **ZeRO-1** with interleaved 1F1B schedule
  - Retains unsharded gradients across virtual stages
  - Trades increased memory for reduced communication overhead
- **If batch_size < 2 * PP**: Use FSDP **ZeRO-2** with all-forward-all-backward schedule
  - Reshards gradients to save memory
  - Additional gradient reduce-scatter communication cost

### 4.5 Layer Assignment Optimization

For the 405B with PP=16:
- First PP rank: fewer Transformer layers (to compensate for embedding layer memory)
- Last PP rank: fewer Transformer layers (to compensate for output head computation)
- Result: 126 layers instead of 128 = 16 stages * 8 layers

---

## 5. Context Parallelism Details

### 5.1 Implementation: All-Gather Based (NOT Ring Attention)

Llama 3 uses an **all-gather based context parallelism** approach:
1. All-gather the Key (K) and Value (V) tensors across CP ranks
2. Compute attention output locally for the local Query (Q) chunk
3. This is simpler than ring attention and avoids fragmented compute kernels

**Why not ring attention:**
- When CP is large and sequence length per GPU is small, ring-style attention suffers from fragmented compute kernels with lower compute efficiency
- Ring attention also has overhead from merging partial attention results
- All-gather approach achieves comparable performance with simpler implementation

### 5.2 Communication Overhead

- CP communication: **7.64% of total elapsed time** for 128K context (Stage 3)
- The communication cost of CP scales **linearly** with sequence length
- The computation scales **quadratically** with sequence length
- Therefore, CP communication overhead decreases as a fraction of total time for longer sequences

### 5.3 Scaling Properties

All-gather CP achieves **3.89x attention latency reduction** on 4 GPUs vs 1 GPU (near-linear scaling for the attention computation portion).

### 5.4 Interaction with Pipeline Parallelism

When CP is enabled, it affects peak memory:
- CP splits along sequence length, so each GPU holds fewer tokens
- PP makes peak memory **independent of batch size** (activations are freed between micro-batches)
- Combined: reducing DP and increasing CP (while increasing batch/DP to maintain global batch) can reduce peak memory because PP keeps memory independent of batch size while CP reduces per-token memory

---

## 6. Data Parallelism / FSDP Details

### 6.1 FSDP Implementation

Llama 3 uses FSDP (Fully Sharded Data Parallel) which:
- Shards model parameters, optimizer states, and gradients across DP ranks
- Equivalent to ZeRO Stage 3 (with some differences)

### 6.2 Key Optimization: No Reshard After Forward

Standard FSDP reshards parameters after the forward pass. Llama 3's implementation **does NOT reshard after forward computation** to avoid an extra all-gather communication during the backward pass. This trades memory for communication reduction.

### 6.3 Numerical Precision in FSDP

Three specific FP32 accumulation requirements:
1. **FP32 gradient accumulation** during backward computation over multiple micro-batches in PP
2. **FP32 reduce-scatter** of gradients across DP workers in FSDP
3. **FP32 accumulation** for intermediate tensors that are used multiple times

---

## 7. Training Recipe Details

### 7.1 Pre-training Schedule (405B)

| Phase | Batch Size (tokens) | Sequence Length | Token Count at Start | Duration |
|-------|---------------------|-----------------|---------------------|----------|
| Phase 1 | 4M | 4,096 | 0 | Until 252M tokens |
| Phase 2 | 8M | 8,192 | 252M tokens | Until 2.87T tokens |
| Phase 3 | 16M | 8,192 | 2.87T tokens | Until ~15.6T tokens |

**Optimizer:** AdamW
- Peak learning rate: 8e-5
- Linear warmup: 8,000 steps
- Cosine decay: to 8e-7 over 1,200,000 steps
- Weight decay: 0.1 * current_learning_rate at each step

### 7.2 Long Context Extension Recipe

After initial pre-training on 8K context, gradually extended to 128K over **six stages**:
- Starting from 8K context window
- Ending at 128K context window
- Total long-context pre-training: ~800B tokens
- Parallelism reconfigured: CP increased from 1 to 16, DP reduced from 128 to 4

### 7.3 Annealing Phase

- Final 40M tokens: learning rate linearly annealed to 0
- Context length maintained at 128K
- Polyak averaging (checkpoint averaging) applied during annealing
- Data mixture adjusted to upsample high-quality sources

### 7.4 Scaling Law Observation

Llama 3 8B continued improving performance after training on **two orders of magnitude more data** (15T+ tokens) beyond the Chinchilla-optimal compute budget (~200B tokens for an 8B model). This suggests the Chinchilla scaling laws significantly underestimate the benefit of additional data for smaller models.

---

## 8. Hardware and Infrastructure

### 8.1 GPU and Server Configuration

- **GPU**: NVIDIA H100-80GB with HBM3, running at 700W TDP
- **Server**: Meta Grand Teton AI server platform
  - 8 GPUs per server
  - 2 CPUs per server
  - Intra-server: NVLink connecting all 8 GPUs
- **Cluster**: Two custom-built 24K GPU clusters
  - Active training: up to 16,384 GPUs simultaneously

### 8.2 Network Topology

- **Architecture**: Three-layer Clos network using RoCE (RDMA over Converged Ethernet)
- **Switches**: Arista 7800 and Minipack2 OCP rack switches
- **Link speed**: 400 Gbps between GPUs (inter-node)
- **Pod structure**: 192 racks connected via Cluster Switches = 3,072 GPUs per pod
- **Full cluster**: 8 pods connected at aggregation layer with 1:7 oversubscription ratio
- **Load balancing**: Enhanced-ECMP creating **16 network flows** between any two GPUs
- **Notable**: Llama 3 405B used RoCE (not InfiniBand); smaller models used InfiniBand (Nvidia Quantum2)

### 8.3 Storage

- **File system**: Tectonic (Meta's distributed file system)
- **Capacity**: 240 PB across 7,500 SSD-equipped servers
- **Sustained throughput**: 2 TB/s
- **Peak throughput**: 7 TB/s
- **Checkpoint size**: 1 MB to 4 GB per GPU

### 8.4 H100 Peak Performance Reference

For MFU calculations, the reference peak for H100:
- BF16 Tensor Core: ~989 TFLOP/s (based on 430/0.43 ~= 1000, paper seems to use ~1000 TFLOP/s)
- The achieved 400-430 TFLOP/s represents real-world throughput per GPU

---

## 9. Reliability and Operational Data

### 9.1 Effective Training Time

- **>90% effective training time** achieved despite daily interruptions
- Blog post claimed "more than 95%" effective training time
- ~3x improvement over Llama 2 training efficiency

### 9.2 Failure Analysis (54-day snapshot)

| Category | Count | % of Unexpected |
|----------|-------|-----------------|
| Total interruptions | 466 | - |
| Planned maintenance | 47 | - |
| Unexpected interruptions | 419 | 100% |
| Hardware issues | ~327 | ~78% |
| GPU failures (faulty GPUs) | 148 | 35.3% |
| HBM3 memory failures | 72 | 17.2% |
| SRAM failures | 19 | 4.5% |
| System processor failures | 17 | 4.1% |
| Network issues | ~35 | ~8.4% |
| Manual interventions needed | 3 | 0.7% |

**Mean time between failures**: 466 interruptions / 54 days = ~8.6 interruptions/day = one failure every ~2.8 hours on a 16K GPU cluster.

### 9.3 Environmental Effects

- Diurnal throughput variation: **1-2%** based on time of day (temperature effects)
- Power consumption fluctuations: "tens of megawatts" across data center during synchronized GPU operations

---

## 10. Memory Estimation and Formulas

### 10.1 Memory from HuggingFace Blog (Inference)

| Model | FP16 | FP8 | INT4 |
|-------|------|-----|------|
| 8B | 16 GB | 8 GB | 4 GB |
| 70B | 140 GB | 70 GB | 35 GB |
| 405B | 810 GB | 405 GB | 203 GB |

### 10.2 KV Cache Memory (FP16, Inference)

| Model | 1K tokens | 16K tokens | 128K tokens |
|-------|-----------|-----------|------------|
| 8B | 0.125 GB | 1.95 GB | 15.62 GB |
| 70B | 0.313 GB | 4.88 GB | 39.06 GB |
| 405B | 0.984 GB | 15.38 GB | 123.05 GB |

**KV cache formula verification for 405B at 128K:**
```
KV_cache = 2 * n_layers * n_kv_heads * head_dim * seq_len * bytes_per_element
         = 2 * 126 * 8 * (16384/128) * 131072 * 2
         = 2 * 126 * 8 * 128 * 131072 * 2
         = 2 * 126 * 8 * 128 * 131072 * 2
         = 68,719,476,736 bytes = 64 GB
```
Hmm, that gives ~64 GB, not 123 GB. Let me recalculate:
```
head_dim = model_dim / n_heads = 16384 / 128 = 128
KV_cache = 2 * n_layers * n_kv_heads * head_dim * seq_len * bytes
         = 2 * 126 * 8 * 128 * 128000 * 2
         = 66,060,288,000 bytes ≈ 61.5 GB
```
The HF blog value of 123 GB at 128K suggests a different formula or includes overhead. Possibly they use `2 * n_layers * (n_kv_heads * head_dim) * seq_len * 2` but with n_kv_heads=8 this should still give ~62 GB for FP16. The discrepancy may be due to using full attention heads (128) instead of KV heads (8) for the cache, or there is a factor-of-2 difference in what "128K" means.

**At 16K tokens with 8 KV heads:**
```
KV = 2 * 126 * 8 * 128 * 16384 * 2 = 8,455,716,864 bytes ≈ 7.87 GB
```
But the blog says 15.38 GB, which is ~2x. This suggests the blog may compute KV cache with full heads (not GQA heads) or use a different formula. **This is a potential gotcha for our calculator -- clarify whether KV cache uses n_kv_heads or n_attention_heads.**

### 10.3 Training Memory Requirements (from HuggingFace Blog)

| Model | Full Fine-tuning | LoRA | QLoRA |
|-------|-----------------|------|-------|
| 8B | 60 GB | 16 GB | 6 GB |
| 70B | 500 GB | 160 GB | 48 GB |
| 405B | 3.25 TB | 950 GB | 250 GB |

**Full fine-tuning verification for 405B:**
- Weights in BF16: 405B * 2 = 810 GB
- Gradients in BF16: 810 GB
- Adam optimizer (FP32 momentum + variance): 405B * 4 * 2 = 3,240 GB
- Total model states: 810 + 810 + 3,240 = 4,860 GB
- But blog says 3.25 TB, which is less. This suggests mixed precision with BF16 master weights:
  - Weights BF16: 810 GB
  - Gradients BF16: 810 GB  
  - Optimizer: FP32 master weights (1,620 GB) + momentum (1,620 GB) + variance (1,620 GB) = 4,860 GB -- no, that's still more.
  
  Actually for standard mixed precision training:
  - FP32 master weights: 405B * 4 = 1,620 GB
  - BF16 working weights: 405B * 2 = 810 GB
  - BF16 gradients: 810 GB
  - FP32 optimizer momentum: 1,620 GB
  - FP32 optimizer variance: 1,620 GB
  - Total: ~6,480 GB

  The blog's 3.25 TB figure likely excludes FP32 master copy and optimizer, or assumes pure BF16 training:
  - BF16 weights: 810 GB
  - BF16 gradients: 810 GB
  - BF16 optimizer states: 810 GB * 2 = 1,620 GB
  - Total: 3,240 GB ≈ 3.25 TB
  
  This matches if they assume the optimizer states are also in BF16 (or FP16), which is non-standard but matches the number. **Calculator should clarify precision assumptions.**

### 10.4 Internal Memory Estimation Tools

The paper mentions: "we develop a memory consumption estimator and a performance-projection tool which helped us explore various parallelism configurations." These are internal Meta tools, not publicly available. The paper does not provide the formulas used in these tools.

---

## 11. FLOPs Calculation Formulas

### 11.1 Simple Approximation (6ND)

```
Total_Training_FLOPs = 6 * N * D
```
Where:
- N = number of parameters (405B)
- D = number of training tokens (15.6T)
- Factor 6 = 2 (forward) + 4 (backward, since backward is ~2x forward)

Verification: 6 * 405e9 * 15.6e12 = 3.79e25 (matches paper's 3.8e25)

### 11.2 Detailed FLOPs Breakdown (Per Token)

From NVIDIA NGC/community formulas for Llama-style models:

```
model_flops_per_token = seq_len * (attention_flops + mlp_flops + embedding_flops)
```

Where:
```
attention_flops = 12 * L * h^2 * (1 + n_kv_groups/n_heads + seq_len/h)
mlp_flops       = 18 * L * d_ffn * h
embedding_flops = 6 * V * h
```

And:
- L = number of layers
- h = hidden dimension (model dimension)
- d_ffn = FFN intermediate dimension
- V = vocabulary size
- n_heads = number of attention heads
- n_kv_groups = number of KV head groups
- seq_len = sequence length

**Note**: The factor 18 for MLP (instead of 16) accounts for SwiGLU which has 3 weight matrices in the FFN instead of 2.

For Llama 3 405B with GBS=1:
```
Per-token model FLOPs = 2.17e16
```

### 11.3 Training Time Formula

```
training_time_seconds = Total_FLOPs / (num_gpus * achieved_tflops_per_gpu * 1e12)
```

Or equivalently:
```
training_time_seconds = 6 * N * D / (num_gpus * peak_tflops * MFU * 1e12)
```

---

## 12. Numerical Stability Details

### 12.1 Mixed Precision Strategy

Llama 3 uses **BF16 mixed precision** with selective FP32 accumulation:

| Operation | Precision |
|-----------|-----------|
| Forward computation (GEMM) | BF16 inputs, FP32 internal accumulation |
| Backward computation (gradients) | BF16 with FP32 accumulation over micro-batches |
| Gradient reduce-scatter (FSDP) | FP32 |
| Intermediate tensors (used multiple times) | FP32 accumulation |
| Optimizer states (Adam momentum/variance) | FP32 |
| Communication (all-gather for CP) | BF16 |

### 12.2 Loss Spike Handling

The paper discusses debugging numerical issues at scale:
- Non-parallel implementation used as reference to isolate bugs from precision issues
- "Critical gradient buffers" identified that require high-precision FP32 accumulation
- Inherent accumulation error from floating-point arithmetic grows with number of gradient accumulations (L2 norm increases)
- Larger gradient accumulation steps => higher discrepancy

---

## 13. Key Findings for Calculator Implementation

### 13.1 Formulas to Implement

1. **Parameter count** for GQA models:
   ```
   P = L * (12*h^2 + 13*h) + V*h + h  [approximate, need to account for GQA]
   ```
   More precisely for GQA:
   ```
   P_per_layer = h * (h + 2*n_kv_heads*head_dim + d_ffn*3)  [for SwiGLU]
              plus biases and norms
   ```

2. **Training FLOPs**: `C = 6 * N * D`

3. **Training time**: `T = C / (G * F_peak * MFU)`

4. **Pipeline bubble ratio**: `bubble = (PP-1) / (V*M)` where V=virtual stages, M=micro-batches

5. **MFU**: `MFU = achieved_tflops / peak_tflops` (BF16 peak for H100: ~989 TFLOP/s)

6. **Memory per GPU with FSDP**:
   - Model states (weights + optimizer + gradients) sharded across DP ranks
   - Activations determined by PP warm-up micro-batches and sequence length/CP

### 13.2 Heuristics for Parallelism Configuration

1. TP = GPUs per node (8 for H100 DGX/Grand Teton)
2. PP = enough to fit model in memory; use interleaved schedule
3. CP = 1 for standard sequences; increase for long context (CP = seq_len / 8192)
4. DP = total_GPUs / (TP * CP * PP)
5. Batch/DP >= 2 * PP for efficient 1F1B scheduling
6. If batch/DP < 2 * PP: switch from ZeRO-1 to ZeRO-2 (gradient resharding)
7. Global batch size (tokens) should remain constant when scaling GPUs (adjust DP and batch/DP inversely)

### 13.3 Unique Insights (Non-obvious)

1. **Layer count is not a free parameter in PP**: Must account for embedding/output head imbalance. Llama 3 removed 2 layers to balance PP stages.
2. **No activation checkpointing needed at 8K**: With balanced PP, 126 layers, and TP=8, activation memory fits without recomputation.
3. **FSDP does NOT reshard after forward**: Trades memory for communication. Standard FSDP implementations may differ.
4. **ZeRO-1 vs ZeRO-2 depends on batch/PP ratio**: batch >= 2*PP => ZeRO-1; batch < 2*PP => ZeRO-2
5. **CP communication is only 7.64% overhead** at 128K context with CP=16, suggesting CP scales well.
6. **Diurnal throughput variation of 1-2%** from temperature -- relevant for cost estimation accuracy.
7. **MFU drops ~2% when doubling GPUs** (43% -> 41%) purely from batch size reduction per DP group.
8. **MFU drops additional ~3% for long context** (41% -> 38%) from CP communication overhead.
9. **One failure every ~2.8 hours** on 16K GPU cluster -- relevant for effective training time estimates (>90%).
10. **Batch size schedule matters**: Start small (4M tokens), double twice (to 16M) during training. This is a training stability technique, not just efficiency.

### 13.4 Empirical MFU Reference Points

| Configuration | MFU | Notes |
|--------------|-----|-------|
| 8K H100, TP=8, PP=16, DP=64, 8K seq | 43% | Best case, 32 batch/DP |
| 16K H100, TP=8, PP=16, DP=128, 8K seq | 41% | 2% loss from halved batch/DP |
| 16K H100, TP=8, CP=16, PP=16, DP=4, 128K seq | 38% | Additional 3% from CP overhead |

These are valuable calibration points: a well-optimized 405B training run on H100s achieves 38-43% MFU.

---

## 14. Comparison with Other Training Reports

| System | Model | GPUs | MFU | Notes |
|--------|-------|------|-----|-------|
| Llama 3 405B | 405B | 16K H100 | 38-43% | RoCE, 4D parallelism |
| PaLM | 540B | 6,144 TPUv4 | 46.2% (57.8% HFU) | Pathways framework |
| GPT-4 (est.) | ~1.8T MoE | ~25K A100 | ~30-35% (est.) | Estimated |

---

## 15. Relevant File Paths

The downloaded PDFs (if needed for further analysis):
- ISCA25 paper: `/Users/vinitvyas09/.claude/projects/-Users-vinitvyas09-code-personal-llm-training-gpu-calculator/407fc3a3-846e-4121-9ae4-48e28e68b26e/tool-results/webfetch-1774994726552-cbpot2.pdf`
- Full Llama 3 paper: `/Users/vinitvyas09/.claude/projects/-Users-vinitvyas09-code-personal-llm-training-gpu-calculator/407fc3a3-846e-4121-9ae4-48e28e68b26e/tool-results/webfetch-1774994844785-idqls5.pdf`

---

## Sources

- [Meta Llama 3 Blog Post](https://ai.meta.com/blog/meta-llama-3/)
- [The Llama 3 Herd of Models (arXiv 2407.21783)](https://arxiv.org/abs/2407.21783)
- [ar5iv HTML version](https://ar5iv.labs.arxiv.org/html/2407.21783)
- [Scaling Llama 3 Training with Efficient Parallelism Strategies (ISCA 2025)](https://dl.acm.org/doi/10.1145/3695053.3731410)
- [HuggingFace Llama 3.1 Blog](https://huggingface.co/blog/llama31)
- [HuggingFace Model Card](https://huggingface.co/meta-llama/Llama-3.1-405B)
- [Meta Llama 3.1 Blog Post](https://ai.meta.com/blog/meta-llama-3-1/)
- [GitHub Issue: Understanding GPU Hours](https://github.com/meta-llama/llama3/issues/91)
