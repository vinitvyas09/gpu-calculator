# FlashAttention-3: Deep Dive for GPU Calculator Spec

**Source**: https://tridao.me/blog/2024/flash3/
**Paper**: Dao et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (arXiv:2407.08608, NeurIPS 2024)
**Date reviewed**: 2026-03-31

---

## Executive Summary

FlashAttention-3 achieves 740 TFLOP/s on H100 SXM (75% utilization), up from ~35% utilization with FlashAttention-2 on the same hardware. It introduces three techniques: (1) asynchronous warp-specialized overlap of computation and data movement via TMA, (2) interleaved block-wise matmul and softmax pipelining, and (3) FP8 block quantization with incoherent processing (Hadamard transform). FP8 FA3 reaches ~1.2 PFLOP/s on H100.

For the GPU calculator, FlashAttention-3 is primarily relevant for:
- Eliminating the quadratic attention activation memory term (5*a*s/d in the Korthikanti formula)
- Providing a concrete utilization ceiling for attention computation on H100
- FP8 attention throughput doubling potential
- Enabling longer sequence lengths without memory explosion

---

## 1. Memory Formulas: Standard Attention vs FlashAttention

### 1.1 Standard Attention Memory (O(N^2))

Standard self-attention materializes the full N x N attention matrix in HBM:

```
Per head, per layer:
  S = Q * K^T          -> N x N matrix (stored in HBM)
  P = softmax(S)       -> N x N matrix (stored in HBM)
  dropout_mask(P)      -> N x N matrix (stored in HBM, 1 byte per element)
  P * V                -> N x d_head output

Memory for attention scores per layer (from Korthikanti et al., 2022):
  M_attn_scores = 5 * a * s^2 * b bytes
```

Breakdown of the 5*a*s^2*b term:
- QK^T scores (S): 2 * a * s^2 * b bytes (stored in bf16/fp16)
- Softmax output (P): 2 * a * s^2 * b bytes (stored in bf16/fp16)
- Attention dropout mask: 1 * a * s^2 * b bytes (1 bit per element, stored as bytes)

For the full per-layer activation memory (Korthikanti et al.):
```
M_act_layer = s * b * d * (34 + 5 * a * s / d) bytes
            = s * b * d * 34 + 5 * a * s^2 * b    bytes
              ^linear part    ^quadratic part
```

**Concrete example**: LLaMA 7B (a=32, d=4096, s=4096, b=1):
- Quadratic term: 5 * 32 * 4096^2 * 1 = 2.68 GB per layer
- Linear term: 4096 * 1 * 4096 * 34 = 0.57 GB per layer
- The quadratic term is 4.7x the linear term at s=4096

At s=32K, the quadratic term grows to 171 GB per layer -- completely infeasible.

### 1.2 FlashAttention Memory (O(N))

FlashAttention eliminates the N x N materialization entirely. It stores only:

```
Per layer:
  Q, K, V inputs:     already counted in linear activation terms
  Output O:           already counted in linear activation terms
  Softmax statistics: 4 * a * s * b / N_tp bytes
    (per head: one float for row-max m_i, one float for log-sum-exp l_i, times seq length)
```

The spec already captures this correctly:
```
With Flash Attention:
  M_act_layer = s * b * d * (10 + 24/N_tp) bytes
  (the 5*a*s/d term disappears)
```

The softmax statistics replacement is:
```
M_softmax_stats = 4 * a * s * b / N_tp bytes
```
This is negligible for typical configs. Example: LLaMA 7B at s=4096, b=1: 4 * 32 * 4096 * 1 = 0.5 MB (vs 2.68 GB without FA).

### 1.3 Memory Savings Summary

| Config (LLaMA 7B, b=1) | s=2048 | s=4096 | s=8192 | s=32768 | s=131072 |
|---|---|---|---|---|---|
| Quadratic term (5*a*s^2*b) per layer | 0.67 GB | 2.68 GB | 10.7 GB | 171 GB | 2.74 TB |
| FA softmax stats replacement | 0.25 MB | 0.5 MB | 1 MB | 4 MB | 16 MB |
| Savings per layer | 0.67 GB | 2.68 GB | 10.7 GB | 171 GB | 2.74 TB |
| Savings across 32 layers | 21.5 GB | 85.9 GB | 343 GB | 5.5 TB | 87.7 TB |

At s >= 4096, FlashAttention is not optional -- it is required for feasibility.

---

## 2. Performance Metrics and Hardware Utilization

### 2.1 H100 SXM Hardware Reference (as used by FA3)

From the FA3 paper, the exact H100 SXM specifications used:

```
H100 SXM Specifications (from FA3 paper):
  Peak FP16/BF16 dense matmul:     989 TFLOP/s
  Peak FP8 dense matmul:           1,979 TFLOP/s (2x FP16)
  Special functions (exp, etc):    3.9 TFLOP/s (256x less than matmul!)
  HBM (GMEM):                     80 GiB @ 3.35 TB/s
  L2 cache:                       50 MiB @ 12 TB/s
  Shared memory (SMEM):           228 KiB per SM @ 31 TB/s aggregate
  Register file:                  256 KiB per SM
  Streaming Multiprocessors:      132 SMs
  Boost clock:                    1,830 MHz
```

**Critical insight for the calculator**: The 3.9 TFLOP/s for special functions (exponential, used in softmax) is 256x slower than matmul throughput. This means softmax is a severe bottleneck even though it represents a tiny fraction of total FLOPs. For head dimension 128, the exponential operations in softmax consume ~50% of execution time despite representing only 1/512th of total matmul FLOPs. This explains why attention utilization is lower than dense matmul utilization and why FA3's asynchronous softmax overlap is so important.

### 2.2 FlashAttention-3 Performance

```
FA3 FP16 Forward:  ~640-740 TFLOP/s (65-75% of H100 peak 989 TFLOP/s)
FA3 FP16 Backward: ~570-640 TFLOP/s (lower than forward due to 5 matmuls + more softmax)
FA3 FP8 Forward:   ~1.2-1.3 PFLOP/s (61-66% of H100 FP8 peak 1,979 TFLOP/s)

Utilization = achieved_TFLOPS / peak_TFLOPS
FA3 FP16: 740 / 989 = 74.8% ~ 75%
FA3 FP8:  1200 / 1979 = 60.6% ~ 61%
```

### 2.3 FA3 vs FA2 Speedup

```
Forward pass:   1.5-2.0x faster than FA2
Backward pass:  1.5-1.75x faster than FA2
Overall:        ~1.6-1.8x speedup for training (fwd + bwd)

FA2 on H100:   ~35% utilization (~346 TFLOP/s FP16)
FA3 on H100:   ~75% utilization (~740 TFLOP/s FP16)
Ratio:          740/346 = 2.14x for forward peak
```

### 2.4 FA3 vs Standard Attention

```
FA3 vs standard PyTorch attention:  3-16x faster
  - Short sequences (512): ~3x faster
  - Medium sequences (2K-4K): ~5-8x faster
  - Long sequences (8K-16K): ~10-16x faster
```

---

## 3. Attention FLOPs Formulas

### 3.1 Forward Pass FLOPs

The FA3 paper provides the canonical formula:

```
FLOPs_attn_fwd = 4 * s^2 * d_head * n_heads
               = 4 * s^2 * d       (where d = n_heads * d_head)
```

Breakdown:
- Q * K^T:    2 * s^2 * d FLOPs (one matmul of [s, d] x [d, s])
- Scores * V: 2 * s^2 * d FLOPs (one matmul of [s, s] x [s, d_head] per head)

With causal masking, divide by 2 (approximately half the entries are computed):
```
FLOPs_attn_fwd_causal = 2 * s^2 * d
```

### 3.2 Backward Pass FLOPs

```
FLOPs_attn_bwd = FLOPs_attn_fwd * 2.5
```

Rationale: Forward has 2 matmuls, backward has 5 matmuls (due to recomputation of attention scores):
1. Recompute Q * K^T (for the backward pass)
2. dP = dO * V^T
3. dV = P^T * dO
4. dS = dP * diag(softmax) computations
5. dQ = dS * K, dK = dS^T * Q

Total attention FLOPs (fwd + bwd) per training step:
```
FLOPs_attn_total = 4 * s^2 * d * 3.5 = 14 * s^2 * d
```

Note: The existing spec uses 12*L*d*s for the attention quadratic term in the PaLM formula. This counts only the 2 forward matmuls (Q*K^T and scores*V) times 3 for fwd+bwd: 2 * 2*s*d * 3 = 12*s*d per layer. The FA3 paper's 2.5x backward multiplier (vs 2x in the standard formula) accounts for the recomputation cost specific to FlashAttention's backward pass. This is consistent -- both approaches count the same operations.

---

## 4. FP8 Attention Details

### 4.1 FP8 Format

```
FP8 e4m3 format:
  1 sign bit + 4 exponent bits + 3 mantissa bits = 8 bits total
  Effective precision: ~3.5 decimal digits
  Dynamic range: limited (4-bit exponent vs 5-bit in FP16)
```

### 4.2 Block Quantization

Instead of per-tensor quantization (one scale factor for the entire Q, K, or V matrix), FA3 uses block quantization:

```
Block quantization:
  Split Q, K, V into blocks of size B_r x d or B_c x d
  One scalar scale factor per block
  Quantize each block independently to FP8

Memory for scale factors:
  num_blocks = ceil(s / B_r)  (or B_c)
  scale_storage = num_blocks * 4 bytes (FP32 scale per block)
  Negligible relative to the attention matrices
```

### 4.3 Incoherent Processing (Hadamard Transform)

To reduce FP8 quantization error caused by outlier values in Q and K:

```
Pre-quantization transform:
  Q_hat = Q * M
  K_hat = K * M
  Where M is a random orthogonal matrix (M * M^T = I)

Since M * M^T = I:
  (Q * M) * (K * M)^T = Q * M * M^T * K^T = Q * K^T
  The transform is mathematically exact.

Implementation:
  M = D * H  (product of random diagonal +/-1 matrix D and Hadamard matrix H)
  Complexity: O(d * log(d)) per row instead of O(d^2) for general matrix multiply
  This "spreads out" outlier values across dimensions, making quantization more uniform.

Error reduction: 2.6x lower RMSE than baseline FP8 quantization
  Baseline FP8 RMSE: 2.4e-2
  FA3 FP8 RMSE:      9.1e-3
```

### 4.4 FP8 Memory Implications for Calculator

For the GPU calculator, FP8 attention has these implications:

1. **Compute throughput**: FP8 matmul is 2x faster than FP16 on H100 (1,979 vs 989 TFLOP/s peak). FA3 FP8 achieves ~1.2 PFLOP/s.

2. **Memory**: FP8 attention does NOT reduce activation memory because:
   - The intermediate attention scores are never materialized in HBM (FlashAttention tiles them in SRAM)
   - Q, K, V inputs are typically still stored in bf16 in HBM and converted to FP8 on-the-fly in SRAM
   - The FP32 accumulator for QK^T is converted to FP8 in registers for the P*V matmul, avoiding extra memory traffic

3. **Accuracy**: Block quantization + incoherent processing makes FP8 attention viable for training without quality loss. The 2.6x error reduction from incoherent processing is critical for maintaining training stability.

---

## 5. Implementation Details Relevant to Calculator

### 5.1 Tiling and Block Sizes

FlashAttention operates with block sizes determined by SRAM capacity:

```
Block size formula (from original FlashAttention):
  B_c = ceil(M / (4 * d))
  
Where:
  M = SRAM size per SM (192 KB on A100, 228 KB on H100)
  d = head dimension
  Factor of 4: accounts for Q, K, V block and output O block
  
Example (H100, d=128):
  B_c = ceil(228 * 1024 / (4 * 128 * 2)) = ceil(233472 / 1024) = 228 rows per block
  (assuming bf16, 2 bytes per element)
```

### 5.2 IO Complexity

From the original FlashAttention paper:

```
Standard attention HBM accesses: O(N * d + N^2)
  - Load Q, K, V: O(N * d)
  - Write/read S and P: O(N^2)

FlashAttention HBM accesses: O(N^2 * d^2 / M)
  Where M = SRAM size

For typical values (d=128, M=228KB on H100):
  d^2 = 16384
  M = 233472 bytes (228 KiB)
  d^2 / M = 0.07
  
  So FA accesses: O(0.07 * N^2) vs standard: O(N^2)
  This is ~14x fewer HBM accesses for d=128 on H100.
```

### 5.3 Warp Specialization and Asynchronous Pipelining

FA3 exploits H100-specific hardware features. This is relevant for understanding why FA3 achieves 75% utilization vs FA2's 35%:

**Progressive throughput gains (FA3 ablation study, head_dim=128, non-causal):**

```
Baseline (FA2-style on H100):                ~350 TFLOP/s  (~35%)
+ WGMMA + TMA (new H100 instructions):       ~500 TFLOP/s  (~51%)
+ Pingpong warp scheduling:                  ~570 TFLOP/s  (~58%)
+ Inter-warpgroup overlap (pingpong):         ~620 TFLOP/s  (~63%)
+ Intra-warpgroup 2-stage pipelining:         ~640-660 TFLOP/s (~65-67%)
+ Further optimizations:                      ~740 TFLOP/s  (~75%)
```

Key hardware features:
- **WGMMA** (Warpgroup Matrix Multiply-Accumulate): Higher throughput than older mma.sync
- **TMA** (Tensor Memory Accelerator): Hardware-accelerated global-to-shared memory transfers, reducing register pressure
- **Pingpong scheduling**: Two warpgroups alternate, overlapping softmax of one block with GEMM of the next

### 5.4 Backward Pass Structure

```
FA3 Backward Pass (Algorithm 3):
  3 warp roles:
    1. Producer warpgroup: TMA loads of K, V, Q, dO
    2. Consumer warpgroups: Compute dK, dV, dQ via GEMMs
    3. dQ-writer warp: Atomic accumulation to global dQ

  Key computation:
    dS = (diag(p) - p*p^T) * dP    for p = softmax(s)
    
  Backward stores only:
    - O (output, N x d): already in activation memory
    - L (logsumexp, N per head): 4 bytes per position per head
    
  Everything else (S, P, dS) is recomputed in SRAM tiles
```

---

## 6. Impact on Training Memory Budget

### 6.1 How FlashAttention Affects Total Training Memory

The total per-GPU memory formula is:
```
M_total = M_model_states + M_activations + M_temporary + M_communication
```

FlashAttention affects ONLY M_activations. Specifically, per layer:

```
Without FA (no checkpointing):
  M_act_layer = s * b * d * (34 + 5 * a * s / d)

With FA (no checkpointing):
  M_act_layer = s * b * d * (10 + 24/N_tp) + 4 * a * s * b / N_tp
                                               ^negligible softmax stats

With FA + full activation checkpointing:
  M_act_layer = 2 * s * b * d  (same as without FA -- only store layer input)
```

**When FA matters most**: FA provides the largest relative savings when:
1. Sequence length is long (s >= 4096)
2. Activation checkpointing is NOT used (the quadratic term is fully stored)
3. Activation memory dominates (small models on many GPUs with ZeRO-3)

**When FA matters least**: With full activation checkpointing, FA provides no additional memory savings (the layer input is stored either way). FA still provides a large speed benefit even with checkpointing because the recomputation in the backward pass uses the efficient tiled algorithm.

### 6.2 FlashAttention + Selective Checkpointing Interaction

The spec already notes this correctly: when both FA and selective checkpointing are enabled, the 5*a*s/d term is already eliminated by FA, so selective checkpointing provides no additional memory savings for the attention portion. The formula collapses to:

```
With FA + selective checkpointing:
  M_act_layer = s * b * d * (10 + 24/N_tp)   (same as FA alone)
```

### 6.3 Sequence Length Scaling Under FlashAttention

Memory scaling comparison per layer:

```
Without FA:  M_act ~ O(s^2 * a * b) + O(s * b * d)    [quadratic + linear]
With FA:     M_act ~ O(s * b * d)                       [linear only]

Compute scaling (unchanged by FA):
  FLOPs_attn ~ O(s^2 * d)    [quadratic in sequence length regardless]
```

FlashAttention makes memory linear in sequence length, but compute remains quadratic. For very long sequences (s >= 32K), compute (not memory) becomes the bottleneck, and techniques like Ring Attention or Context Parallelism are needed to distribute the quadratic compute.

---

## 7. Spec Change Assessment

### 7.1 Already Correctly Captured in Spec

The existing spec (Section 5.3) correctly handles:
- FlashAttention eliminating the 5*a*s/d term from activation memory
- The replacement softmax statistics term (4*a*s*b/N_tp bytes)
- The interaction with selective checkpointing (Section 5.3 "Flash Attention + selective checkpointing interaction")
- Flash Attention toggle as a calculator input (Section 11.2, input #9)
- The transient recomputation working memory formula adjustment when FA is enabled

### 7.2 Potential Additions to Consider

1. **H100 hardware utilization context**: The spec could note that FlashAttention-3 achieves 75% utilization on H100 as a reference point for attention-heavy workloads. This is NOT the same as overall training MFU (which includes FFN, communication, etc.) but sets an upper bound for the attention portion.

2. **FP8 attention throughput note**: When the user selects FP8 precision on H100, the calculator could note that FP8 FlashAttention achieves ~1.2 PFLOP/s (vs ~740 TFLOP/s for FP16), representing a ~1.6x speedup on the attention portion. However, since FA3 FP8 is specific to H100 Hopper architecture and not yet widely deployed in training frameworks, this should be advisory only.

3. **Attention bottleneck at extreme sequence lengths**: The spec mentions context parallelism in the Llama 3 MFU table but does not explicitly note that at very long sequences (s >= 32K), attention compute becomes the dominant cost even though its memory is linear under FA. The 12*L*d*s attention term in the PaLM formula already captures this, but the calculator could add advisory text about FlashAttention not solving the compute quadratic.

4. **Softmax throughput bottleneck**: The 256x gap between matmul and special function throughput (989 TFLOP/s vs 3.9 TFLOP/s for exp()) could be noted in the MFU breakdown discussion (Section 6.3) as a concrete example of why "non-matmul operations" reduce MFU by ~15-20%.

### 7.3 What NOT to Change

- The memory formulas are already correct and complete
- The FLOP formulas do not need FA3-specific changes (FA does not change total FLOPs, only memory access patterns)
- The H100 specs in the GPU table (Section 7) already list the correct 989 TFLOP/s dense BF16 peak
- MFU guidelines do not need FA3-specific adjustment (FA3's 75% is attention-only utilization, not overall training MFU)

---

## 8. Key Quantitative Reference Table

| Metric | Value | Source |
|---|---|---|
| FA3 FP16 peak throughput | 740 TFLOP/s | Blog/Paper |
| FA3 FP8 peak throughput | ~1.2-1.3 PFLOP/s | Blog/Paper |
| H100 FP16 peak (dense) | 989 TFLOP/s | NVIDIA spec |
| H100 FP8 peak (dense) | 1,979 TFLOP/s | NVIDIA spec |
| H100 special function (exp) throughput | 3.9 TFLOP/s | Paper |
| FA3 FP16 utilization | 75% | Blog/Paper |
| FA2 on H100 utilization | 35% | Blog/Paper |
| FA3 vs FA2 forward speedup | 1.5-2.0x | Blog/Paper |
| FA3 vs FA2 backward speedup | 1.5-1.75x | Blog/Paper |
| FA3 vs standard attention | 3-16x faster | Blog |
| FP8 quantization error (FA3) | RMSE 9.1e-3 | Paper |
| FP8 baseline quantization error | RMSE 2.4e-2 | Paper |
| Incoherent processing error reduction | 2.6x | Blog/Paper |
| H100 HBM bandwidth | 3.35 TB/s | Paper |
| H100 SMEM per SM | 228 KiB | Paper |
| H100 SMs | 132 | Paper |
| Attention FLOPs fwd | 4 * s^2 * d | Paper |
| Attention FLOPs bwd/fwd ratio | 2.5 | Paper |
| Max demonstrated context length | 1M tokens (Llama 3) | Blog |

---

## Sources

- [FlashAttention-3 Blog Post (Tri Dao)](https://tridao.me/blog/2024/flash3/)
- [FlashAttention-3 Paper (arXiv:2407.08608)](https://arxiv.org/abs/2407.08608)
- [FlashAttention-3 HTML Paper (ar5iv)](https://ar5iv.labs.arxiv.org/html/2407.08608)
- [FlashAttention-3 (Together AI Blog)](https://www.together.ai/blog/flashattention-3)
- [FlashAttention-3 Summary (Wentao's Blog)](https://wentao.site/flash_attention_v3_summary/)
- [Original FlashAttention Paper (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135)
- [ELI5: FlashAttention (Aleksa Gordic)](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
- [FlashAttention Primer (Aman.ai)](https://aman.ai/primers/ai/flashattention/)
- [Reducing Activation Recomputation (Korthikanti et al., arXiv:2205.05198)](https://arxiv.org/abs/2205.05198)
- [NVIDIA H100 Specifications](https://www.nvidia.com/en-us/data-center/h100/)
