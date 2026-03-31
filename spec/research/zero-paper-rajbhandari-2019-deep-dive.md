# Deep Dive: ZeRO -- Memory Optimizations Toward Training Trillion Parameter Models

**Paper**: Rajbhandari, Rasley, Ruwase, He (Microsoft Research, 2019/2020)
**URL**: https://arxiv.org/abs/1910.02054
**Version analyzed**: arXiv v3, 13 May 2020 (20 pages)

---

## 1. Core Taxonomy: Model States vs Residual States

The paper's foundational contribution is decomposing GPU training memory into two categories:

### Model States
- **Parameters** (fp16 copy for forward/backward)
- **Gradients** (fp16 copy accumulated during backward)
- **Optimizer states** (fp32 master weights, fp32 momentum, fp32 variance for Adam)

### Residual States
- **Activations** (stored from forward pass for backward pass)
- **Temporary buffers** (e.g., fused gradient all-reduce buffers)
- **Memory fragmentation** (unusable gaps from allocation/deallocation patterns)

This is the first paper to systematically separate these two categories and address them with distinct optimization strategies (ZeRO-DP for model states, ZeRO-R for residual states).

---

## 2. Mixed Precision Memory Model (Section 3.1)

The paper establishes the memory model for mixed-precision (fp16/fp32) training with Adam optimizer. This is the foundational derivation that all ZeRO formulas build upon.

### Setup
- Model has Psi parameters (Psi = number of parameters, a scalar count)
- Mixed precision: forward/backward in fp16, optimizer step in fp32
- Adam optimizer requires: momentum (fp32) and variance (fp32)

### Byte Accounting Per Parameter

| Component | Precision | Bytes per parameter | Total bytes |
|-----------|-----------|-------------------|-------------|
| Parameters (working copy) | fp16 | 2 | 2*Psi |
| Gradients | fp16 | 2 | 2*Psi |
| FP32 master copy of parameters | fp32 | 4 | 4*Psi |
| Adam momentum | fp32 | 4 | 4*Psi |
| Adam variance | fp32 | 4 | 4*Psi |

### The K Multiplier

The paper defines **K** as the memory multiplier of the optimizer states. Specifically:

```
K = (additional bytes per parameter consumed by optimizer states)
  = fp32 master weights + fp32 momentum + fp32 variance
  = 4 + 4 + 4
  = 12 bytes per parameter
```

IMPORTANT: K includes the fp32 master copy of parameters, NOT just momentum and variance. This is because during mixed precision training, the optimizer must maintain fp32 weights for numerical stability. The fp32 master weights are grouped WITH the optimizer states, not with the "parameters" category.

### Total Model State Memory

```
M_model_states = 2*Psi + 2*Psi + K*Psi = (2 + 2 + K)*Psi = (4 + K)*Psi
```

With Adam mixed precision (K=12):
```
M_model_states = (4 + 12)*Psi = 16*Psi bytes
```

### Concrete Example (from paper)
GPT-2 with 1.5B parameters:
- fp16 parameters alone: 2 * 1.5B = 3 GB
- Total model states: 16 * 1.5B = 24 GB
- This is why a 3 GB model cannot train on a 32 GB GPU with standard DP

### Generalized K Values for Other Optimizers

The paper notes K is optimizer-dependent (Section 3.1):
- **Adam (mixed precision)**: K = 12 (fp32 params + momentum + variance)
- **SGD with momentum (mixed precision)**: K = 8 (fp32 params + momentum)
- **SGD without momentum (mixed precision)**: K = 4 (fp32 params only)

The paper uses K=12 for all examples since Adam is the standard.

---

## 3. ZeRO-DP: Memory Formulas for Each Stage (Section 5)

ZeRO-DP has three cumulative optimization stages. Let:
- Psi = number of model parameters
- N_d = data parallel degree (number of GPUs in DP group)
- K = optimizer state memory multiplier (12 for Adam mixed precision)

### Baseline (no ZeRO): Standard Data Parallelism

Every GPU holds a full replica of all model states:
```
M_baseline = (2 + 2 + K) * Psi = (4 + K) * Psi
```
With K=12: M_baseline = 16*Psi

### Stage 1 -- P_os: Optimizer State Partitioning (Section 5.1)

Each GPU stores only 1/N_d of the optimizer states. Parameters and gradients remain fully replicated.

```
M_stage1 = 2*Psi + 2*Psi + (K*Psi)/N_d = 4*Psi + (K*Psi)/N_d
```

With K=12: M_stage1 = 4*Psi + 12*Psi/N_d

**Mechanism**: Optimizer states are partitioned into N_d equal partitions. The i-th GPU only stores and updates the optimizer states for the i-th partition. After the optimizer step, an all-gather collects the updated parameters across all GPUs.

**Memory savings**: When N_d is large, the K*Psi/N_d term vanishes:
```
M_stage1 -> 4*Psi  (as N_d -> infinity)
```
This is a 4x reduction from 16*Psi. The paper explicitly states "leading to a 4x reduction."

### Stage 2 -- P_os+g: Optimizer State + Gradient Partitioning (Section 5.2)

Each GPU stores only 1/N_d of the gradients AND 1/N_d of the optimizer states. Parameters remain fully replicated.

```
M_stage2 = 2*Psi + (2*Psi + K*Psi)/N_d = 2*Psi + ((2 + K)*Psi)/N_d
```

With K=12: M_stage2 = 2*Psi + 14*Psi/N_d

**Mechanism**: Gradients are reduce-scattered (instead of all-reduced) so each GPU ends up with only its partition's reduced gradients. After the optimizer step, an all-gather distributes updated parameters.

**Memory savings**: When N_d is large:
```
M_stage2 -> 2*Psi  (as N_d -> infinity)
```
This is an 8x reduction from 16*Psi. The paper explicitly states "leading to a 8x reduction."

### Stage 3 -- P_os+g+p: Optimizer State + Gradient + Parameter Partitioning (Section 5.3)

Everything is partitioned: parameters, gradients, and optimizer states.

```
M_stage3 = (2*Psi + 2*Psi + K*Psi) / N_d = ((4 + K)*Psi) / N_d
```

With K=12: M_stage3 = 16*Psi/N_d

**Mechanism**: Each GPU only stores 1/N_d of all model states. Parameters needed for forward/backward are received from the responsible GPU via broadcast (all-gather). After use, non-local parameters are discarded.

**Memory savings**: Linear reduction with N_d. No limit:
```
M_stage3 = 16*Psi/N_d
```
With N_d=64: M_stage3 = Psi/4 (64x reduction from baseline)

---

## 4. Figure 1: The Canonical Memory Diagram

The paper's Figure 1 (page 3) provides the definitive visual representation. Reproduced with all formulas and computed values:

**Parameters**: Psi = 7.5B, K = 12 (Adam mixed precision), N_d = 64

| Configuration | Formula | Computed Value |
|--------------|---------|---------------|
| Baseline (DP) | (2 + 2 + K) * Psi | (2+2+12) * 7.5B = 120 GB |
| P_os | 2*Psi + 2*Psi + K*Psi/N_d | 15 + 15 + 12*7.5/64 = 31.4 GB |
| P_os+g | 2*Psi + (2+K)*Psi/N_d | 15 + 14*7.5/64 = 16.6 GB |
| P_os+g+p | (2+2+K)*Psi/N_d | 16*7.5/64 = 1.875 GB ~= 1.9 GB |

**Verification of 31.4 GB**: 2*7.5B*1e-9 bytes = 15 GB (params) + 15 GB (grads) + 12*7.5B/64*1e-9 = 1.40625 GB (optimizer states) = 31.40625 GB. Correct.

**Verification of 16.6 GB**: 2*7.5B = 15 GB (params) + (2+12)*7.5B/64 = 14*7.5/64 = 1.640625 GB (grads + opt states) = 16.640625 GB. Correct.

**Verification of 1.9 GB**: (2+2+12)*7.5B/64 = 16*7.5/64 = 1.875 GB. Correct (paper rounds to 1.9).

---

## 5. Table 1: Per-Device Memory Consumption (Section 5.4)

This is the primary validation target. All values are in GB.

### 7.5B Model (K=12, 16*Psi = 120 GB total model states)

| DP (N_d) | P_os | P_os+g | P_os+g+p |
|----------|------|--------|----------|
| 1 | 120 | 120 | 120 |
| 4 | 52.5 | 41.3 | 30 |
| 16 | 35.6 | 21.6 | 7.5 |
| 64 | **31.4** | **16.6** | 1.88 |
| 256 | 30.4 | 15.4 | 0.47 |
| 1024 | 30.1 | 15.1 | 0.12 |

Bold = fits in 32 GB V100

Verification of selected values:
- P_os at N_d=4: 4*7.5 + 12*7.5/4 = 30 + 22.5 = 52.5 GB. Correct.
- P_os at N_d=16: 4*7.5 + 12*7.5/16 = 30 + 5.625 = 35.625 -> 35.6 GB. Correct.
- P_os+g at N_d=4: 2*7.5 + 14*7.5/4 = 15 + 26.25 = 41.25 -> 41.3 GB. Correct.
- P_os+g at N_d=16: 2*7.5 + 14*7.5/16 = 15 + 6.5625 = 21.5625 -> 21.6 GB. Correct.
- P_os+g+p at N_d=4: 16*7.5/4 = 30 GB. Correct.
- P_os+g+p at N_d=16: 16*7.5/16 = 7.5 GB. Correct.

### 128B Model (K=12, 16*Psi = 2048 GB total model states)

| DP (N_d) | P_os | P_os+g | P_os+g+p |
|----------|------|--------|----------|
| 1 | 2048 | 2048 | 2048 |
| 4 | 896 | 704 | 512 |
| 16 | 608 | 368 | 128 |
| 64 | 536 | 284 | **32** |
| 256 | 518 | 263 | 8 |
| 1024 | 513 | 257 | 2 |

Bold = fits in 32 GB V100

Verification:
- P_os at N_d=4: 4*128 + 12*128/4 = 512 + 384 = 896. Correct.
- P_os+g at N_d=4: 2*128 + 14*128/4 = 256 + 448 = 704. Correct.
- P_os+g+p at N_d=64: 16*128/64 = 32 GB. Correct.

### 1T Model (K=12, 16*Psi = 16000 GB total model states)

| DP (N_d) | P_os | P_os+g | P_os+g+p |
|----------|------|--------|----------|
| 1 | 16000 | 16000 | 16000 |
| 4 | 7000 | 5500 | 4000 |
| 16 | 4750 | 2875 | 1000 |
| 64 | 4187 | 2218 | 250 |
| 256 | 4046 | 2054 | 62.5 |
| 1024 | 4011 | 2013 | **15.6** |

Bold = fits in 32 GB V100

Verification:
- P_os at N_d=4: 4*1000 + 12*1000/4 = 4000 + 3000 = 7000 GB. Correct.
- P_os+g+p at N_d=1024: 16*1000/1024 = 15.625 -> 15.6 GB. Correct.

### Key Insight from Table 1
With N_d=1024 and P_os+g+p (ZeRO-3), a 1 TRILLION parameter model fits on 1024 GPUs with only 15.6 GB model state memory per GPU. This is the paper's headline claim.

---

## 6. Table 2: Maximum Model Size Analysis (Section 5.4)

This table shows the maximum model size that can fit in memory under different ZeRO stages, assuming 32 GB V100 GPUs.

| MP | GPUs | Baseline | P_os | P_os+g | P_os+g+p | Measured Baseline | Measured ZeRO-DP (P_os) |
|----|------|----------|------|--------|----------|-------------------|------------------------|
| 1 | 64 | 2B | **7.6B** | 14.4B | 128B | 1.3B | **6.2B** |
| 2 | 128 | 4B | **15.2B** | 28.8B | 256B | 2.5B | **12.5B** |
| 4 | 256 | 8B | **30.4B** | 57.6B | 0.5T | 5B | **25B** |
| 8 | 512 | 16B | **60.8B** | 115.2B | 1T | 10B | **50B** |
| 16 | 1024 | 32B | **121.6B** | 230.4B | 2T | 20B | **100B** |

The measured model size with P_os matches the theoretical maximum, validating the formulas. The measured baseline model size is lower than theoretical because residual memory (activations, buffers) consumes significant space.

### How the Theoretical Values are Derived

For the theoretical maximum model size, solve for Psi given 32 GB per GPU:

- **Baseline**: (4+K)*Psi = 32 GB * N_gpu/MP (aggregate memory). With K=12: Psi = 32*N_gpu / (16*MP). For MP=1, 64 GPUs: Psi = 32*64/16 = 128B... Wait, that gives the aggregate, but the paper seems to be computing per-GPU. Let me reconsider.

Actually, for baseline DP, each GPU holds ALL model states, so: (4+K)*Psi <= 32 GB. With K=12: Psi <= 32/16 = 2B. With MP=m, parameters per GPU = Psi/m, so memory per GPU = 16*Psi/m <= 32. Thus Psi <= 2B*m. For MP=16: Psi <= 32B. This matches the table.

For P_os with N_d GPUs (N_d = N_gpu / MP): 4*Psi + 12*Psi/N_d <= 32*m (per-GPU memory / model parallel share). Actually more precisely: 4*Psi/m + 12*Psi/(m*N_d) <= 32. So Psi <= 32*m / (4 + 12/N_d). For MP=1, N_d=64: Psi <= 32 / (4 + 12/64) = 32 / 4.1875 = 7.64B ~ 7.6B. Correct.

For P_os+g: 2*Psi/m + 14*Psi/(m*N_d) <= 32. Psi <= 32*m / (2 + 14/N_d). For MP=1, N_d=64: Psi <= 32/(2+14/64) = 32/2.21875 = 14.42B ~ 14.4B. Correct.

For P_os+g+p: 16*Psi/(m*N_d) <= 32. Psi <= 2*m*N_d. For MP=1, N_d=64: Psi <= 128B. Correct.

---

## 7. Communication Volume Analysis (Section 7)

### Baseline: Standard Data Parallel Communication

Standard DP uses all-reduce to synchronize gradients. All-reduce is implemented as reduce-scatter + all-gather. For data of size Psi elements:
- Reduce-scatter: Psi elements transmitted per GPU
- All-gather: Psi elements transmitted per GPU
- **Total: 2*Psi elements per GPU per training step**

This 2*Psi is the baseline for comparison.

### ZeRO Stage 1 (P_os) Communication Volume

P_os partitions optimizer states but still needs to synchronize gradients and distribute updated parameters. The communication is:
- All-reduce on gradients: 2*Psi (same as baseline, via reduce-scatter + all-gather)

**Total: 2*Psi (identical to baseline)**

No additional communication cost over standard DP.

### ZeRO Stage 2 (P_os+g) Communication Volume

P_os+g partitions both optimizer states and gradients. Communication:
- Reduce-scatter on gradients: Psi (each GPU gets its partition of reduced gradients)
- All-gather on updated parameters: Psi (distribute updated parameters after optimizer step)

**Total: Psi + Psi = 2*Psi (identical to baseline)**

No additional communication cost over standard DP. This is a key result: ZeRO gets up to 8x memory reduction with ZERO communication overhead.

### ZeRO Stage 3 (P_os+g+p) Communication Volume

P_os+g+p also partitions parameters. Requires additional communication to gather parameters for forward/backward:
- Reduce-scatter on gradients: Psi
- All-gather on parameters for forward pass: Psi (each layer's params gathered before forward, discarded after)
- All-gather on parameters for backward pass: Psi (each layer's params gathered again before backward)

**Total: Psi + Psi + Psi = 3*Psi (1.5x baseline)**

The paper explicitly states: "The total volume is therefore 3*Psi which is 1.5x compared to the baseline."

### Summary Table

| Stage | Communication Volume | Ratio to Baseline | Memory Reduction |
|-------|---------------------|--------------------|-----------------|
| Baseline DP | 2*Psi | 1.0x | 1x |
| ZeRO-1 (P_os) | 2*Psi | 1.0x | up to 4x |
| ZeRO-2 (P_os+g) | 2*Psi | 1.0x | up to 8x |
| ZeRO-3 (P_os+g+p) | 3*Psi | 1.5x | up to N_d x |

CRITICAL INSIGHT: ZeRO Stages 1 and 2 achieve massive memory savings with ZERO communication overhead. Stage 3 trades a modest 50% increase in communication for linear memory scaling.

---

## 8. Residual Memory (ZeRO-R) Details (Sections 3.2, 6)

### 8.1 Activation Memory (Sections 3.2, 6.1)

The paper provides an activation memory formula in footnote 3 (page 8):
```
Activation memory for GPT-2-like architecture:
  ~12 * hidden_dim * batch_size * seq_length * num_transformer_layers bytes
```

Concrete examples from the paper:
- GPT-2 1.5B, seq_length=1K, batch_size=32: ~60 GB activations
- With activation checkpointing (sqrt reduction, 33% recompute overhead): ~8 GB
- GPT-like 100B, batch_size=32, with activation checkpointing: ~60 GB

### 8.2 Partitioned Activation Checkpointing -- P_a (Section 6.1)

When combined with model parallelism (MP degree = N_m), ZeRO-R partitions activation checkpoints across MP GPUs instead of replicating them:
- Standard MP: Each GPU stores a full copy of activations for its partition -> redundant
- ZeRO P_a: Activation checkpoints are partitioned across N_m GPUs, only materialized (via all-gather) when needed for recomputation
- Reduces activation memory by factor of N_m

**Example from paper**: 100B model, batch_size=32, seq_length=1024, MP=16:
- Standard: ~33 GB per GPU for activation checkpoints
- With P_a: ~2 GB per GPU (16x reduction)
- Can further offload the 2 GB to CPU, reducing activation memory to ~0

### 8.3 P_a Communication Overhead (Section 8)

Communication analysis for partitioned activation checkpointing in the context of Megatron-LM with MP:

Without P_a (standard Megatron-LM with activation checkpointing):
- Each transformer block: 2 all-reduces of size (batch * seq_length * hidden_dim) in forward
- 2 all-reduces for forward recomputation
- 2 all-reduces in backward
- Total per block: 12 * seq_length * hidden_dim (since all-reduce comm = 2 * message_size)

With P_a:
- Additional 1 all-gather per transformer block (to reconstruct partitioned activation checkpoint)
- All-gather size: seq_length * hidden_dim
- Total overhead: seq_length * hidden_dim per block

Therefore:
```
P_a overhead / baseline MP communication = (seq_length * hidden_dim) / (12 * seq_length * hidden_dim) < 1/10
```

P_a adds less than 10% communication overhead relative to baseline model parallelism.

### 8.4 CPU Offloading of Activation Checkpoints -- P_a+cpu (Section 6.1)

When even partitioned activation checkpoints are too large:
- Offload partitioned checkpoints to CPU memory
- 2x data movement overhead compared to P_a (write to CPU + read back)
- Reduces GPU activation memory to ~0
- Beneficial when: CPU data transfer overhead < DP communication savings from larger batch size

### 8.5 Constant Size Buffers -- C_B (Section 6.2)

Standard approach: Fuse all gradients into one large buffer for all-reduce efficiency. Buffer size is proportional to model size.
- Problem: For a 3B parameter model, a 32-bit fused buffer = 12 GB

ZeRO-R approach: Use constant-size fused buffers regardless of model size.
- Buffer size does not scale with model parameters
- Large enough for efficient all-reduce (good bandwidth utilization)
- The paper does not specify an exact buffer size, but refers to "performance-efficient constant-size"

### 8.6 Memory Defragmentation -- M_D (Section 6.3)

Problem: Interleaving of short-lived and long-lived tensors causes fragmentation.
- Forward: Checkpointed activations (long-lived) interleaved with discarded activations (short-lived)
- Backward: Parameter gradients (long-lived) interleaved with activation gradients (short-lived)
- Can cause OOM with 30%+ memory still free (but not contiguous)

ZeRO-R solution: Pre-allocate contiguous memory chunks for activation checkpoints and gradients. Copy tensors into pre-allocated memory as they are produced.

---

## 9. ZeRO Configuration Table (Table 3)

The paper evaluates 5 configurations combining ZeRO-DP and ZeRO-R:

| Config | ZeRO-DP | ZeRO-R |
|--------|---------|--------|
| 1 | P_os | C_B + M_D |
| 2 | P_os | C_B + M_D + P_a |
| 3 | P_os+g | C_B + M_D |
| 4 | P_os+g | C_B + M_D + P_a |
| 5 | P_os+g | C_B + M_D + P_a+cpu |

NOTE: P_os+g+p (Stage 3) was not implemented in the ZeRO-100B implementation evaluated in this paper. The paper explicitly states (page 5): "We plan to release all implementations described in this paper by end of May 2020 and extend it further to support 1 trillion parameters by enabling ZeRO-DP stage 3 partitioning parameters (P_os+g+p)."

---

## 10. ZeRO + Model Parallelism Combined (Section 1)

ZeRO can be combined with MP for maximum memory reduction:
```
Max memory reduction = N_d * N_m
```
Where N_d = data parallel degree and N_m = model parallel degree.

Example from paper: 1024 GPUs with 16-way MP (within DGX-2 nodes) and 64-way DP (across nodes):
- N_m = 16, N_d = 64
- Theoretical memory reduction = 1024x
- Can fit 1T+ parameters

This is shown in Table 2 (rightmost column, bottom row): with MP=16, 1024 GPUs, P_os+g+p can fit a 2T parameter model.

---

## 11. Model Configurations Used in Experiments (Table 4)

| Model Size | Layers | Hidden Dim |
|-----------|--------|------------|
| 1.5B | 48 | 1600 |
| 8B | 72 | 3072 |
| 40B-60B | 88, 132 | 4096 |
| 80B-170B | 100, 125, 150 | 8192 |
| 140B-170B | 175, 212 | 8192 |

For scalability experiments (Figures 3, 4):

| Model Size | Layers | Hidden Dim |
|-----------|--------|------------|
| 1.16B-2.5B | 24, 34, 54 | 1920 |
| 4B | 64 | 2304 |
| 6B-8B | 52, 72 | 3072 |
| 10B-13B | 50, 54, 58, 62 | 4096 |
| 60B | 75 | 8192 |

---

## 12. Performance Results (Key Numbers)

- **ZeRO-100B**: Up to 170B parameters on 400 V100 GPUs
- **Throughput**: 38 TFlops per GPU on 100B model (30%+ of V100 peak)
- **Aggregate**: 15 PetaFlops on 400 GPUs for 100B model
- **Super-linear speedup**: Observed from 64 to 400 GPUs (Figure 3) because ZeRO-DP frees memory as N_d increases, allowing larger batch sizes and better arithmetic intensity
- **Without MP**: ZeRO-100B trains up to 13B parameters on 128 GPUs with 40+ TFlops/GPU
- **Baseline comparison**: Standard DP (PyTorch DDP) limited to 1.4B parameters on 32 GB GPUs
- **Turing-NLG**: 17B parameter model trained end-to-end with ZeRO, 41.4 TFlops/GPU sustained throughput

---

## 13. Communication Bandwidth Context

The paper provides important bandwidth numbers for understanding communication overhead:
- **Intra-node (NVSwitch)**: 300 GB/s per link
- **Inter-node (InfiniBand EDR)**: 12.5 GB/s per link
- Going beyond a single DGX-2 node with MP causes communication bandwidth to drop from 300 to 12.5 GB/s (24x drop), resulting in significant performance degradation
- This is why ZeRO (data parallelism based) is more scalable than MP for multi-node

---

## 14. Edge Cases and Constraints

### 14.1 Parameter Divisibility
ZeRO partitions model states into N_d equal parts. The paper does not explicitly discuss what happens when Psi is not evenly divisible by N_d, but the formulas assume clean divisibility. In practice, implementations pad or use unequal partition sizes.

### 14.2 Activation Checkpointing is Assumed
The memory savings from ZeRO-DP apply to model states only. For large models, activation memory can still be the bottleneck. The paper's measured results use activation checkpointing for all large models. The calculator must account for activation memory separately.

### 14.3 Super-Linear Speedup Mechanism
ZeRO-100B shows super-linear speedup (Figure 3) because:
1. Adding more GPUs increases N_d
2. Higher N_d reduces per-GPU memory via ZeRO
3. Freed memory allows larger per-GPU batch size
4. Larger batch size improves arithmetic intensity (more compute per communication event)
This means throughput-per-GPU increases with N_d, not just total throughput.

### 14.4 Residual Memory Lower Bound
Even with ZeRO Stage 3, residual memory (activations, buffers, fragmentation) sets a lower bound on per-GPU memory. The paper's Table 2 shows that measured model sizes are consistently ~50-60% of theoretical maximums, meaning residual memory consumes 40-50% of GPU memory in practice. This is a critical factor for any calculator.

### 14.5 ZeRO-3 Layer Gathering Constraint
During forward and backward, ZeRO-3 must all-gather the full parameters of at least one layer at a time. This means the per-GPU memory must be at least:
```
M_min_per_gpu >= size_of_largest_layer_parameters
```
This is not explicitly stated as a formula in the paper but is implied by the mechanism in Section 5.3.

### 14.6 Batch Size vs. N_d Coupling
In standard DP, the global batch size = micro_batch_size * N_d. For very large N_d, the global batch size can become too large for good convergence. The paper acknowledges this in footnotes but states that for models up to 1K GPUs, batch sizes are still in acceptable ranges.

---

## 15. What Is Unique About This Paper / Non-Obvious Insights

### 15.1 K Includes FP32 Master Weights
The paper groups fp32 master weights as part of the optimizer states (inside K), not as part of the "parameters" category. This means K=12 for Adam, not K=8 (just momentum+variance). This is because the fp32 master copy is only needed for the optimizer step, making it semantically an optimizer state. This grouping matters for the ZeRO-1 formula: ZeRO-1 shards K (including master weights), not just momentum/variance.

### 15.2 Communication Equivalence of Stages 1 and 2
The fact that ZeRO-1 and ZeRO-2 have IDENTICAL communication volume to standard DP (2*Psi per step) is non-obvious and arguably the paper's most important practical result. It means you can get 4-8x memory savings for free in terms of communication cost.

### 15.3 Stage 3 Communication is Only 1.5x, Not 3x
Despite needing to gather parameters for both forward and backward passes, Stage 3 only adds 50% communication (3*Psi vs 2*Psi). This is because the gradient reduce-scatter (Psi) was already in the baseline. Only the two parameter all-gathers (Psi + Psi) are new, but one of them replaces the all-gather that was already in the baseline all-reduce.

Wait, let me re-derive. Baseline = reduce-scatter(Psi) + all-gather(Psi) = 2*Psi. ZeRO-3 = reduce-scatter(Psi) + all-gather-fwd(Psi) + all-gather-bwd(Psi) = 3*Psi. The all-gather-fwd and all-gather-bwd are for parameters (not gradients). The reduce-scatter is for gradients. So compared to baseline which has reduce-scatter + all-gather (for gradients), ZeRO-3 replaces the all-gather-of-gradients with all-gather-of-params-fwd + all-gather-of-params-bwd. Net: 3*Psi vs 2*Psi = 1.5x.

### 15.4 Memory Formula is Independent of Model Architecture
The ZeRO formulas depend only on Psi (total parameter count), K (optimizer memory multiplier), and N_d (DP degree). They are agnostic to model architecture (number of layers, hidden dim, attention heads, etc.). Architecture only matters for activation memory (residual states), not model states.

### 15.5 Temporary Buffer Memory Can Be Huge
The paper notes that for a 1.5B parameter model, a flattened fp32 gradient buffer consumes 6 GB. This scales linearly: a 10B model would need 40 GB just for the temporary buffer. ZeRO-R addresses this with constant-size buffers, but the paper's observation implies that any calculator must account for temporary buffer memory, especially without ZeRO-R.

### 15.6 Memory Fragmentation Can Waste 30%+ of GPU Memory
The paper reports memory fragmentation can cause OOM with over 30% of memory still free. This is a purely practical concern that analytical memory calculators cannot predict, but the calculator should include a fragmentation safety margin (e.g., recommend using only 80-85% of theoretical GPU memory).

### 15.7 Table 2: Measured vs Theoretical
The ratio of measured model size to theoretical maximum is consistently ~50% for baseline (without ZeRO). For example, with MP=1 and 64 GPUs: theoretical=2B, measured=1.3B (65%). With MP=8 and 512 GPUs: theoretical=16B, measured=10B (62.5%). This ~60% ratio is a useful heuristic for accounting for residual memory overhead in practice.

---

## 16. Formulas Summary for Calculator Implementation

### Input Parameters
- Psi: number of model parameters (scalar)
- N_d: data parallel degree
- K: optimizer state memory multiplier (default 12 for Adam mixed precision)
- beta_param: bytes per parameter for working copy (2 for fp16)
- beta_grad: bytes per gradient (2 for fp16)

### Model State Memory Formulas

```
# Total bytes for model states (all per parameter, multiply by Psi for total)
Phi = beta_param + beta_grad + K  # total bytes per parameter for all model states
# Phi = 2 + 2 + 12 = 16 for Adam mixed-precision fp16

# Baseline (no ZeRO)
M_baseline = Phi * Psi

# ZeRO Stage 1 (optimizer state partitioning)
M_zero1 = (beta_param + beta_grad) * Psi + K * Psi / N_d
         = (2 + 2) * Psi + 12 * Psi / N_d   # for fp16 + Adam

# ZeRO Stage 2 (optimizer state + gradient partitioning)
M_zero2 = beta_param * Psi + (beta_grad + K) * Psi / N_d
         = 2 * Psi + (2 + 12) * Psi / N_d   # for fp16 + Adam

# ZeRO Stage 3 (optimizer state + gradient + parameter partitioning)
M_zero3 = Phi * Psi / N_d
         = 16 * Psi / N_d                     # for fp16 + Adam
```

### Communication Volume Formulas (elements per GPU per step)

```
# Baseline DP: reduce-scatter + all-gather
C_baseline = 2 * Psi

# ZeRO-1: same as baseline
C_zero1 = 2 * Psi

# ZeRO-2: same as baseline
C_zero2 = 2 * Psi

# ZeRO-3: reduce-scatter + all-gather(fwd) + all-gather(bwd)
C_zero3 = 3 * Psi
```

### Activation Memory (approximate, for GPT-2-like transformers)

```
M_activations = 12 * d * b * s * L  # bytes, where:
# d = hidden dimension
# b = micro batch size
# s = sequence length
# L = number of transformer layers

# With activation checkpointing (sqrt approximation):
M_activations_ckpt = 2 * d * b * s * sqrt(L)  # rough approximation from paper
# The paper says "approximately the square root of the total activations"
```

### Combined Total Memory Per GPU

```
M_total = M_model_states(zero_stage) + M_activations + M_buffers + M_fragmentation
```

Where M_buffers and M_fragmentation are empirical/heuristic terms.

---

## 17. Relationship to Our Existing Spec

The existing spec at `spec/llm-training-gpu-calculator-spec.md` Section 5.2 already contains the ZeRO formulas with generalized notation (Phi, K_opt, beta_grad). Comparing against the paper:

### What the Spec Already Has Correct
1. All four ZeRO stage formulas match the paper exactly
2. Communication volumes for all stages match (2*Psi, 2*Psi, 2*Psi, 3*Psi)
3. K_opt derivation is correct (K=12 for Adam mixed precision)
4. FSDP-to-ZeRO equivalence table is correct
5. Parameter divisibility warning is mentioned
6. ZeRO + PP compatibility constraints are correct
7. CPU/NVMe offloading memory model is present
8. Minimum GPU memory floor (largest layer) is documented
9. ZeRO stage selection heuristic (prefer lowest stage that fits) matches paper's implicit guidance

### What I Found That Could Augment the Spec

1. **Table 1 as a validation target**: The spec should include Table 1's exact numbers as validation test cases for the calculator. All numbers can be reproduced from the formulas.

2. **Measured vs. Theoretical ratio**: The ~60% ratio from Table 2 (measured/theoretical model size) is a useful practical heuristic. It suggests adding a "practical capacity" output that is ~60% of theoretical ZeRO capacity, or equivalently using only ~60% of GPU memory for model states (leaving 40% for activations + residual).

3. **The activation memory formula**: The paper gives `12 * d * b * s * L` bytes as the activation memory for GPT-2-like models (footnote 3, page 8). The existing spec's activation memory section (5.3) should be cross-referenced.

4. **Super-linear speedup**: The calculator could note that ZeRO enables larger batch sizes with more GPUs, which means throughput-per-GPU can increase (not just stay flat) as you add GPUs.

5. **P_a activation partitioning**: The paper's ZeRO-R partitioned activation checkpointing (reducing activation memory by the MP degree) could be added as an advanced feature. This is distinct from ZeRO-DP stages and applies when using TP/MP.

6. **Temporary buffer sizing**: The paper's insight that a flattened fp32 buffer costs 4*Psi bytes is a useful formula for the buffer memory estimate. For a 1.5B model: 4*1.5B = 6 GB. This is already partially captured in the spec's Section 5.4 but the relationship to model size (4*Psi bytes for fp32 fused buffer) could be made more explicit.

7. **30% fragmentation safety margin**: The paper's observation that OOM can occur with 30%+ free memory due to fragmentation suggests the calculator should reserve at least 15-20% of GPU memory as a safety margin beyond the analytical estimate.

---

## 18. Paper Citation

```
@article{rajbhandari2019zero,
  title={ZeRO: Memory Optimizations Toward Training Trillion Parameter Models},
  author={Rajbhandari, Samyam and Rasley, Jeff and Ruwase, Olatunji and He, Yuxiong},
  journal={arXiv preprint arXiv:1910.02054},
  year={2019}
}
```

Published at SC20 (International Conference for High Performance Computing).
