# Deep Dive: FlashAttention (Dao et al., 2022) and FlashAttention-2 (Dao, 2023)

**Source**: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (arXiv:2205.14135)
**Secondary Source**: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (arXiv:2307.08691)
**Authors**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re
**Relevance**: Directly impacts the activation memory formula in our GPU calculator. FlashAttention eliminates the O(s^2) attention matrix from GPU HBM, replacing it with O(s) per-head statistics.

---

## 1. Core Problem and Insight

### The Problem
Standard self-attention computes:
```
S = Q * K^T    in R^{N x N}
P = softmax(S) in R^{N x N}
O = P * V      in R^{N x d}
```
Where N = sequence length, d = head dimension. Standard implementations materialize S and P in HBM, consuming O(N^2) memory. For GPT-2 with N=1024, d=64: the attention matrix S alone is 1024x1024 = 1M elements per head, while Q,K,V are each only 1024x64 = 65K elements per head. The ratio is N/d, which grows with sequence length.

### The IO-Awareness Insight
The key insight is NOT about reducing FLOPs (FlashAttention actually does MORE FLOPs due to recomputation). It is about reducing HBM reads/writes by exploiting the GPU memory hierarchy:

- **HBM** (High Bandwidth Memory): 40-80 GB, bandwidth 1.5-2.0 TB/s (A100)
- **SRAM** (on-chip shared memory): 192 KB per SM, 108 SMs on A100, bandwidth ~19 TB/s
- SRAM is ~13x faster but ~200,000x smaller than HBM

Standard attention is **memory-bound** (low arithmetic intensity): most time is spent reading/writing S, P matrices from HBM, not computing them. FlashAttention eliminates these HBM reads/writes entirely for the intermediate matrices.

---

## 2. Memory Model: Standard Attention vs FlashAttention

### 2.1 Standard Attention Memory (What Gets Stored in HBM)

Per attention head, the forward pass materializes and stores:
```
S = Q * K^T     : N x N elements   (stored for backward)
P = softmax(S)  : N x N elements   (stored for backward)
P_dropped       : N x N elements   (dropout mask, stored for backward)
O = P * V       : N x d elements   (output, passed to next layer)
```

**Total HBM storage per head for backward pass**: 2 * N^2 + N * d (S and P matrices, plus optional dropout mask which is also N x N).

In the Korthikanti et al. formulation used in our spec, across all heads, this manifests as the `5 * a * s / d` term in:
```
M_act_layer = s * b * d * (34 + 5 * a * s / d) bytes
```

**Breaking down the `5as/d` term**:
The attention score matrices stored per layer are:
- Q*K^T result (pre-softmax scores): a * s * s elements = a * s^2 (in bf16: 2 bytes each)
- softmax output: a * s * s elements = a * s^2 (in bf16: 2 bytes each)
- attention dropout mask: a * s * s elements = a * s^2 (1 byte each, stored as uint8/bool)

Total bytes = a * s^2 * (2 + 2 + 1) = 5 * a * s^2 bytes per sample.

Expressed per the formula normalization (divide by s*b*d, multiply back in the formula):
```
5 * a * s^2 * b / (s * b * d) = 5 * a * s / d
```
Hence the `5as/d` coefficient.

### 2.2 FlashAttention Memory (What Gets Stored in HBM)

FlashAttention **never materializes** the N x N attention matrices S and P in HBM. Instead, it:
1. Computes S, P in tiles on SRAM (never written to HBM)
2. Stores only the **softmax normalization statistics** for the backward pass recomputation

**FlashAttention-1** stores TWO statistics per row:
- `m` (row-wise max): vector in R^N (one scalar per row per head)
- `l` (row-wise sum of exponentials): vector in R^N (one scalar per row per head)

**FlashAttention-2** stores ONE combined statistic per row (key optimization):
- `L = m + log(l)` (the logsumexp): vector in R^N (one scalar per row per head)

This is explicitly stated in FlashAttention-2 Section 3.1.1: "We do not have to save both the max m^(j) and the sum of exponentials l^(j) for the backward pass. We only need to store the logsumexp L^(j) = m^(j) + log(l^(j))."

**Memory per head stored in HBM for backward**:
- FlashAttention-1: 2 * N floats (m and l, each N elements in fp32) = 8N bytes per head
- FlashAttention-2: 1 * N floats (L only, N elements in fp32) = 4N bytes per head
- Plus O (the output): N * d elements in bf16 = 2Nd bytes per head

**Also stored**: The pseudo-random number generator state R for dropout (small, constant size per layer, not per-element). This replaces the N x N dropout mask.

### 2.3 Exact Replacement Formula for `5as/d`

With FlashAttention-2 (the practical version used today), the attention-related stored activations change from:
```
WITHOUT FlashAttention:
  5 * a * s^2 * b bytes per layer    (the 5as/d term, after multiplying out)
  
  Breakdown:
  - QK^T scores:    2 * a * s^2 * b bytes  (bf16)
  - softmax output: 2 * a * s^2 * b bytes  (bf16)
  - dropout mask:   1 * a * s^2 * b bytes  (uint8)

WITH FlashAttention-2:
  4 * a * s * b bytes per layer      (logsumexp L only, in fp32)
  
  Breakdown:
  - logsumexp L:    4 * a * s * b bytes    (fp32, one float per row per head)
  - (dropout mask replaced by RNG state, negligible)
```

**Reduction ratio** = (5 * a * s^2 * b) / (4 * a * s * b) = 5s/4

For s=2048: reduction = 2560x
For s=4096: reduction = 5120x
For s=16384: reduction = 20480x

This explains the "10-20x memory savings at long sequences" claim from the paper -- at seq lengths of 8K-16K, the reduction in the attention component alone is enormous.

### 2.4 Updated Activation Memory Formulas

**No checkpointing, no FlashAttention (baseline)**:
```
M_act_layer = s * b * d * (34 + 5 * a * s / d) bytes
```

**No checkpointing, WITH FlashAttention**:
```
M_act_layer = s * b * d * 34 + 4 * a * s * b bytes
            = s * b * (34 * d + 4 * a) bytes
```

The `4 * a * s * b` term (logsumexp statistics) is typically negligible. For a model with d=4096, a=32:
- Linear term: 34 * d = 139,264 per element
- FlashAttention statistics: 4 * a = 128 per element
- Ratio: 128 / 139,264 = 0.09%

So in practice, with FlashAttention the formula simplifies to:
```
M_act_layer ≈ s * b * d * 34 bytes  (within 0.1% for typical architectures)
```

Or more precisely, preserving the Korthikanti form factor:
```
M_act_layer = s * b * d * (34 + 4*a/(d)) bytes
```
where `4*a/d` replaces `5*a*s/d`. Note: this is NOT `4*a*s/d` -- the key difference is the removal of the factor of `s` in the numerator, which is what makes it O(s) instead of O(s^2).

**Selective checkpointing, WITH FlashAttention and Tensor Parallelism**:
```
M_act_layer = s * b * d * (10 + 24/N_tp) + 4 * a * s * b / N_tp bytes
```
The logsumexp statistics are also sharded by TP since attention heads are distributed.

**Selective checkpointing + Sequence Parallelism + FlashAttention**:
```
M_act_layer = s * b * d * (10/N_tp + 24/N_tp) + 4 * a * s * b / N_tp bytes
            = s * b * d * (34/N_tp) + 4 * a * s * b / N_tp bytes
```

### 2.5 What the Spec Currently Says vs What Should Change

The current spec (Section 5.3) states:
```
M_act_layer = s * b * d * (10 + 24/N_tp) bytes  (the 5as/d term disappears)
```
and then says the replacement is `4 * a * s * b / N_tp` bytes.

**This is correct.** The spec accurately captures the FlashAttention memory model. The key formula is:
- The `5*a*s/d` quadratic-in-s term is replaced by `4*a/d` (linear in s, constant in s after the s*b*d multiplier)
- The `4` comes from one fp32 float (4 bytes) for the logsumexp per row per head

**One refinement**: The spec says "one float for row-max and one for log-sum-exp" which was true for FlashAttention-1. FlashAttention-2 combines these into a single logsumexp L = m + log(l). The spec should be updated to reflect that modern FlashAttention (v2+) stores a single fp32 logsumexp per row per head, not two separate values. This halves the statistics from 8 bytes to 4 bytes per row per head, but the spec's formula of `4 * a * s * b / N_tp` is already correct for FlashAttention-2.

---

## 3. IO Complexity Analysis

### 3.1 Standard Attention IO (HBM accesses)

**Theorem 2** (from the paper): Standard attention (Algorithm 0) requires Theta(Nd + N^2) HBM accesses.

Breakdown:
- Computing S = QK^T: read Q, K (Nd each), write S (N^2) = Theta(Nd + N^2)
- Computing P = softmax(S): read S (N^2), write P (N^2) = Theta(N^2)
- Computing O = PV: read P (N^2), V (Nd), write O (Nd) = Theta(Nd + N^2)
- **Total: Theta(Nd + N^2)**

### 3.2 FlashAttention IO (HBM accesses)

**Theorem 2**: FlashAttention requires Theta(N^2 * d^2 * M^{-1}) HBM accesses, where M = SRAM size.

For typical values (d=64-128, M~100KB), d^2 is many times smaller than M, so FlashAttention requires many times fewer HBM accesses. Concrete comparison from Figure 2 data (GPT-2 medium, seq len 1024, head dim 64, 16 heads, batch 64, A100):

| Metric | Standard | FlashAttention |
|--------|----------|----------------|
| GFLOPs | 66.6     | 75.2           |
| HBM R/W (GB) | 4.4 | 7.3           |
| Runtime (ms) | 41.7  | 7.3           |

FlashAttention does **more** FLOPs (75.2 vs 66.6 GFLOPs, ~13% more due to recomputation) but **fewer** HBM accesses, resulting in 5.7x faster runtime.

### 3.3 Block Size Formulas

The block sizes are determined by SRAM capacity M and head dimension d:

```
B_c = ceil(M / (4d))
B_r = min(ceil(M / (4d)), d)
```

These ensure that the blocks of K, V (size B_c x d), Q, O (size B_r x d), and the attention tile S_ij (size B_r x B_c) all fit in SRAM simultaneously:
```
B_c * d = O(M)  =>  B_c = O(M/d)
B_r * d = O(M)  =>  B_r = O(M/d)
B_r * B_c = O(M)  (the attention block must also fit)
```

Practical block sizes (FlashAttention-2): typically {64, 128} x {64, 128}, tuned per head dimension d and device shared memory size.

### 3.4 Lower Bound (Proposition 3)

No exact attention algorithm can achieve o(N^2 * d^2 * M^{-1}) HBM accesses for all M in [d, Nd]. FlashAttention is thus **asymptotically optimal**.

### 3.5 Backward Pass IO

**Theorem 5**: Standard backward pass requires Theta(Nd + N^2) HBM accesses. FlashAttention backward pass requires Theta(N^2 * d^2 * M^{-1}) HBM accesses (same asymptotic as forward).

---

## 4. FLOPs Analysis

FlashAttention does NOT reduce FLOPs. It actually increases them slightly due to recomputation in the backward pass.

### 4.1 Forward Pass FLOPs
```
Forward FLOPs per head = 4 * N^2 * d   (2 matmuls: QK^T and PV, each 2*N^2*d)
Total forward FLOPs    = 4 * N^2 * d * a  (across all heads)
```

With causal mask: approximately half the entries are computed, so ~2 * N^2 * d * a.

### 4.2 Backward Pass FLOPs

Standard backward: 4 matmuls (dV, dP, dQ, dK), roughly 2x forward FLOPs.

FlashAttention backward: 5 matmuls in the backward pass (recomputes S_ij = Q_i * K_j^T in addition to the 4 gradient matmuls), so roughly 2.5x forward FLOPs.

**Total FLOPs ratio** (FlashAttention vs standard):
- Forward: 1x (identical)
- Backward: 5/4 = 1.25x (25% more due to recomputation of S)
- Combined (fwd + 2x bwd): (1 + 2*1.25) / (1 + 2*1) = 3.5/3 = 1.17x (~17% more total FLOPs)

From FlashAttention-2, the FLOPs formula for attention:
```
FLOPs_attention_forward = 4 * s^2 * d_head * n_heads
FLOPs_attention_backward = 2.5 * FLOPs_attention_forward
```

The standard full-model FLOPs formula (Megatron-LM):
```
FLOPs_total = 6 * s * P + 12 * L * d * s^2
```
where the second term is the attention FLOPs. FlashAttention does not change this formula because it computes exact attention -- the same FLOPs happen, just in a different order (tiled).

---

## 5. Wall-Clock Speedup Numbers

### 5.1 Attention-only benchmarks (A100 GPU)

From FlashAttention-1 paper Table 9 (forward pass, with dropout + masking, batch=16, 8 heads, d=64):

| Seq Length | PyTorch (ms) | FlashAttention (ms) | Speedup |
|-----------|-------------|-------------------|---------|
| 128       | 0.36        | 0.04              | 9.0x    |
| 256       | 0.34        | 0.06              | 5.7x    |
| 512       | 0.78        | 0.06              | 13.0x   |
| 1024      | 2.54        | 0.21              | 12.1x   |
| 2048      | 9.33        | 0.82              | 11.4x   |
| 4096      | 36.33       | 2.85              | 12.7x   |
| 8192      | -           | 10.41             | (OOM)   |
| 16384     | -           | 41.74             | (OOM)   |

### 5.2 End-to-end training speedups

**BERT-large (seq 512)**: 15% speedup (17.4 min vs 20.0 min, 8xA100)
**GPT-2 small (seq 1K)**: 3.5x speedup over HuggingFace, 2.0x over Megatron-LM
**GPT-2 medium (seq 1K)**: 3.0x over HuggingFace, 1.8x over Megatron-LM
**Long-range arena (seq 1K-4K)**: 2.4x speedup

### 5.3 FlashAttention-2 speedups (over FlashAttention-1)

**Attention kernel**: 1.7-3.0x faster than FlashAttention-1, 3-10x faster than standard PyTorch
**End-to-end training**:
| Model | No FlashAttn | FlashAttn-1 | FlashAttn-2 |
|-------|-------------|-------------|-------------|
| GPT3-1.3B 2k ctx | 142 TFLOPs/s | 189 TFLOPs/s | 196 TFLOPs/s |
| GPT3-1.3B 8k ctx | 72 TFLOPs/s | 170 TFLOPs/s | 220 TFLOPs/s |
| GPT3-2.7B 2k ctx | 149 TFLOPs/s | 189 TFLOPs/s | 205 TFLOPs/s |
| GPT3-2.7B 8k ctx | 80 TFLOPs/s | 175 TFLOPs/s | 225 TFLOPs/s |

FlashAttention-2 reaches 225 TFLOPs/s = 72% MFU on A100 (theoretical peak 312 TFLOPs/s for bf16).

### 5.4 Hardware-specific speedup (FlashAttention-1)

**A100 (d=64)**: 2-4x speedup across seq lengths 128-4096, highest with dropout+masking
**A100 (d=128)**: Up to 3x with causal mask, ~2x without masking
**RTX 3090**: 2.5-4.5x (higher speedup than A100 due to lower HBM bandwidth ratio: 900 GB/s vs 1.5 TB/s)
**T4**: 1.5-3.5x (smaller SRAM means smaller block sizes, matching IO complexity analysis)

---

## 6. Memory Footprint Benchmarks (Table 21)

Actual measured memory usage (MB) on A100, no dropout, no masking, batch=16, 8 heads, d=64:

| Seq Length | PyTorch | FlashAttention | Ratio |
|-----------|---------|----------------|-------|
| 128       | 36      | 22             | 1.6x  |
| 256       | 104     | 44             | 2.4x  |
| 512       | 336     | 104            | 3.2x  |
| 1024      | 1184    | 209            | 5.7x  |
| 2048      | 4416    | 418            | 10.6x |
| 4096      | 17024   | 836            | 20.4x |
| 8192      | (OOM)   | 1672           | -     |
| 16384     | (OOM)   | 3344           | -     |
| 32768     | (OOM)   | 6688           | -     |
| 65536     | (OOM)   | 13376          | -     |

Key observations:
- **FlashAttention memory scales linearly** with sequence length (doubling seq len doubles memory)
- **PyTorch memory scales quadratically** (doubling seq len quadruples memory)
- At seq len 4096: **20x memory reduction**
- At seq len 2048: **10x memory reduction**
- At seq len 1024: **~6x memory reduction**
- At seq len 512: **~3x memory reduction**

Block-sparse FlashAttention has identical memory to dense FlashAttention (same tiles, just skips zero blocks).

---

## 7. Backward Pass Details

### 7.1 What Is Stored from Forward Pass for Backward

**Standard attention** stores for backward:
- S matrix (N x N per head) -- needed for softmax gradient
- P matrix (N x N per head) -- needed for dV and dS computations
- Dropout mask (N x N per head) -- needed to apply same dropout pattern

**FlashAttention-1** stores for backward:
- O (output, N x d per head) -- needed for gradient computation
- m (row-max, N per head, fp32) -- needed for softmax recomputation
- l (row-sum-exp, N per head, fp32) -- needed for softmax recomputation
- RNG state R (small constant) -- for regenerating dropout mask

**FlashAttention-2** stores for backward:
- O (output, N x d per head) -- needed for gradient computation
- L = m + log(l) (logsumexp, N per head, fp32) -- combined statistic
- RNG state R (small constant) -- for regenerating dropout mask

### 7.2 Backward Pass Algorithm (FlashAttention-2, Algorithm 4 from FA1 / Algorithm 2 from FA2)

The backward pass recomputes the attention matrix from Q, K, V and the stored logsumexp L:

1. Compute D_i = rowsum(dO_i * O_i) for all rows (the "D" vector, size N)
2. For each tile (i, j):
   a. Recompute S_ij = tau * Q_i * K_j^T (on SRAM, never hits HBM)
   b. Recompute P_ij = diag(l_i)^{-1} * exp(S_ij^masked - m_i)  [or equivalently exp(S_ij - L_i) in FA2]
   c. Regenerate dropout mask from saved RNG state
   d. Compute dV_j += (P_ij^dropped)^T * dO_i
   e. Compute dP_ij = dO_i * V_j^T
   f. Compute dS_ij = P_ij * (dP_ij - D_i) (pointwise)
   g. Compute dQ_i += tau * dS_ij * K_j
   h. Compute dK_j += tau * dS_ij^T * Q_i

This requires O(N^2 * d) FLOPs (same asymptotic as standard, but with the extra S_ij recomputation adding ~25% more).

### 7.3 Key Observation for Calculator

The backward pass has **5 matmuls** per tile (vs 2 in forward), making the backward pass FLOPs ~2.5x the forward pass for FlashAttention. This contrasts with standard attention where backward is ~2x forward.

For the calculator's compute time estimation:
```
FLOPs_with_flash = FLOPs_without_flash * (1 + additional_recompute_fraction)
```
The additional recompute fraction is small in practice because:
1. The attention recomputation is only 1 extra matmul per tile in the backward pass
2. Attention FLOPs are a minority of total model FLOPs (the 12*L*d*s^2 term vs the 6*s*P term)
3. For large models (P >> 12*L*d*s), attention is <10% of total FLOPs

---

## 8. Interaction with Activation Checkpointing

### 8.1 FlashAttention IS a Form of Selective Gradient Checkpointing

The paper explicitly states: "This can be seen as a form of selective gradient checkpointing." (Section 3.1)

FlashAttention stores O and the softmax statistics (m, l or L) from the forward pass, then recomputes S and P in the backward pass from Q, K, V. This is equivalent to checkpointing just O and the normalization constants, then recomputing the attention matrix.

### 8.2 Does FlashAttention Make Selective Checkpointing Redundant?

**For the attention component: YES.** The `5as/d` term that selective checkpointing targets is already eliminated by FlashAttention. When both are enabled, there is no additional saving from selective checkpointing on the attention portion.

**For the MLP component: NO.** Selective checkpointing can still save MLP activations (the `24/N_tp` part of the formula). However, in practice:
- With FlashAttention: `M_act = s*b*d*(10 + 24/N_tp)` (selective checkpointing of attention is redundant)
- With FlashAttention + full checkpointing: `M_act = 2*s*b*d` (only layer input stored)

The spec correctly notes: "When both Flash Attention and selective checkpointing are enabled, use the Flash Attention formula rather than double-counting the savings."

### 8.3 Combined Strategies

| Configuration | Per-Layer Activation Memory |
|---|---|
| No FA, no checkpointing | `s*b*d*(34 + 5as/d)` |
| No FA, selective checkpointing | `s*b*d*(10 + 24/N_tp + 5as/(d*N_tp))` |
| No FA, full checkpointing | `2*s*b*d` (+ recompute working memory) |
| FA, no checkpointing | `s*b*d*34 + 4*a*s*b` ≈ `s*b*d*34` |
| FA, selective checkpointing | `s*b*d*(10 + 24/N_tp) + 4*a*s*b/N_tp` |
| FA, full checkpointing | `2*s*b*d` (+ recompute working memory without 5as/d) |

Note: With FA + full checkpointing, the recompute working memory during backward also drops the `5as/d` term:
```
M_recomp_working_with_FA = s * b * d * 34 + 4 * a * s * b bytes
```
(vs `s * b * d * (34 + 5as/d)` without FA). This is a significant practical difference for long sequences.

---

## 9. Block-Sparse FlashAttention

### IO Complexity
**Proposition 4**: Block-sparse FlashAttention requires Theta(Nd + N^2 * d^2 * M^{-1} * s_frac) HBM accesses, where s_frac is the fraction of nonzero blocks in the block-sparsity mask.

For large N with sparsity s_frac = N^{-1/2} or s_frac = N^{-1} * log(N):
```
IO = Theta(N * sqrt(N)) or Theta(N * log(N))
```

### Memory
Block-sparse FlashAttention has **identical** memory footprint to dense FlashAttention (same O(N) per head). Only the runtime changes.

### Practical Speedup
2-4x faster than dense FlashAttention, proportional to sparsity ratio.

---

## 10. What Is Unique / Non-Obvious

### 10.1 FlashAttention-2's Single Logsumexp Optimization
FlashAttention-1 stored both m (row-max) and l (row-sum-exp) = 8 bytes per row per head.
FlashAttention-2 stores only L = m + log(l) = 4 bytes per row per head.
This is a 2x reduction in the statistics overhead, though both are negligible compared to the O(s^2) savings.

**For the calculator**: The spec currently says "one float for row-max and one for log-sum-exp" which describes FlashAttention-1. Since all modern implementations use FlashAttention-2+, the formula should say "one fp32 float for the combined logsumexp" = 4 bytes per row per head. The total is `4 * a * s * b / N_tp` bytes, which the spec already has correctly.

### 10.2 Dropout Mask Handling
Standard attention stores the N x N dropout mask (1 byte per element = a * s^2 bytes per layer per sample). FlashAttention replaces this with the RNG state (a few bytes total). This is part of the `5as/d` -> negligible savings. The dropout mask is regenerated in the backward pass from the saved RNG state, producing exactly the same mask.

### 10.3 The FLOPs Increase Is Real but Small
FlashAttention increases total training FLOPs by ~17% (forward same, backward 25% more due to S recomputation). But since attention is typically <15% of total model FLOPs for large models, the net increase is <3% of total training FLOPs. The wall-clock speedup (2-4x) more than compensates.

### 10.4 Causal Mask Optimization
With causal masking, FlashAttention can skip ~50% of blocks (those entirely above the diagonal), giving ~1.7-1.8x additional speedup over non-causal FlashAttention. This is free -- no code change needed, just block-level skipping.

### 10.5 MQA/GQA Support (FlashAttention-2)
FlashAttention-2 supports Multi-Query Attention and Grouped-Query Attention by implicitly duplicating K, V heads. The memory for the logsumexp statistics is still proportional to the number of Q heads (not KV heads), so for GQA with a_kv < a:
```
Logsumexp memory = 4 * a * s * b / N_tp bytes  (based on Q heads, not KV heads)
```
This is correct because each Q head has its own softmax normalization.

### 10.6 Hardware Sensitivity
Speedup depends on HBM bandwidth / SRAM size ratio:
- Higher speedup on GPUs with lower HBM bandwidth relative to compute (RTX 3090 > A100)
- Lower speedup on GPUs with smaller SRAM (T4 < A100) because smaller blocks mean more passes
- The IO complexity Theta(N^2 * d^2 / M) directly shows: larger SRAM -> fewer HBM accesses -> faster

### 10.7 Head Dimension Impact on Block Size
Larger head dimensions (d=128 vs d=64) require more SRAM per block, leading to smaller block sizes and slightly less speedup. From FlashAttention-2 Section 3.3: "Typically we choose blocks of size {64, 128} x {64, 128}, depending on the head dimension d and the device shared memory size."

---

## 11. Corrections / Refinements for the Spec

### 11.1 CONFIRMED CORRECT in current spec:
- The `5as/d` term disappearing with FlashAttention
- The replacement memory being `4 * a * s * b / N_tp` bytes
- The Flash Attention formula: `s * b * d * (10 + 24/N_tp)` for selective checkpointing + FA
- The interaction between selective checkpointing and FlashAttention (no double-counting)
- The transient recomputation working memory also losing the `5as/d` term

### 11.2 SHOULD BE REFINED in current spec:
1. **Statistics description**: Change "one float for row-max and one for log-sum-exp" to "one fp32 float for the combined logsumexp (FlashAttention-2 combines row-max and log-sum-exp into a single value L = m + log(l))". The formula `4 * a * s * b / N_tp` is already correct for this.

2. **FLOPs impact**: The spec should note that FlashAttention increases backward pass FLOPs for the attention component by ~25% (5 matmuls vs 4 in the tiled backward pass). For total model FLOPs this is typically <3% increase. This matters for the compute time estimation module.

3. **Memory footprint table**: The spec could benefit from a concrete memory comparison table like the one in Section 6 above, showing the dramatic scaling difference at various sequence lengths.

4. **FlashAttention as default**: The paper notes FlashAttention has seen "wide adoption in large-scale training." The calculator should default to FlashAttention=ON and make the user explicitly disable it, rather than the reverse.

---

## 12. Summary of Formulas for Calculator

### Memory (per transformer layer, per micro-batch):

**Without FlashAttention:**
```
M_attention_activations = 5 * a * s^2 * b bytes
```

**With FlashAttention (v2+):**
```
M_attention_activations = 4 * a * s * b bytes  (negligible for typical architectures)
```

**Savings:**
```
Memory_saved = (5 * a * s^2 * b) - (4 * a * s * b)
             = a * s * b * (5s - 4) bytes
             ≈ 5 * a * s^2 * b bytes  (for large s)
```

### IO Complexity:
```
Standard attention HBM accesses:        Theta(Nd + N^2)
FlashAttention HBM accesses:            Theta(N^2 * d^2 / M)
Block-sparse FlashAttention:            Theta(Nd + N^2 * d^2 * s_frac / M)
```

### FLOPs:
```
Standard attention forward FLOPs:       4 * N^2 * d * a
Standard attention backward FLOPs:      8 * N^2 * d * a
FlashAttention forward FLOPs:           4 * N^2 * d * a  (same)
FlashAttention backward FLOPs:          10 * N^2 * d * a  (25% more due to S recomputation)
```

### Block sizes:
```
B_c = ceil(M / (4d))
B_r = min(ceil(M / (4d)), d)
```
Practical values: {64, 128} x {64, 128}

---

## 13. Relevant File Paths

- FlashAttention-1 paper PDF: `/Users/vinitvyas09/.claude/projects/-Users-vinitvyas09-code-personal-llm-training-gpu-calculator/d91b4879-75e5-4f21-8298-32855ec485d1/tool-results/webfetch-1774994804029-2d9own.pdf`
- FlashAttention-2 paper PDF: `/Users/vinitvyas09/.claude/projects/-Users-vinitvyas09-code-personal-llm-training-gpu-calculator/d91b4879-75e5-4f21-8298-32855ec485d1/tool-results/webfetch-1774994876139-q2c1ot.pdf`
- Current calculator spec: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/spec/llm-training-gpu-calculator-spec.md`
