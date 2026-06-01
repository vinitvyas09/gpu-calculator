# Deep Dive: Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

**Paper**: Narayanan et al., 2021 (arXiv:2104.04473v5)  
**Authors**: Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, Matei Zaharia (NVIDIA, Stanford, Microsoft Research)  
**Focus for calculator**: Pipeline parallelism formulas, TP communication volumes, combined PTD-P parallelism memory model, per-layer parameter counts, throughput/MFU measurements, interleaved scheduling, activation recomputation memory model, and the vocabulary padding constraint.

---

## 1. Notation (Section 3.1 of Paper)

The paper defines:

| Symbol | Meaning |
|--------|---------|
| p | Pipeline-model-parallel size (number of pipeline stages) |
| t | Tensor-model-parallel size |
| d | Data-parallel size |
| n | Total GPUs; `p * t * d = n` |
| B | Global batch size (provided as input) |
| b | Microbatch size |
| m | Number of microbatches per pipeline: `m = (1/b) * (B/d)` or equivalently `m = B/(b*d)` |
| b' | Ratio of batch size to microbatch size: `b' = B/b` |
| l | Number of transformer layers |
| h | Hidden size |
| a | Number of attention heads |
| s | Sequence length |
| V | Vocabulary size |
| v | Number of virtual pipeline stages (interleaved schedule) |
| l^stage | Number of layers in a pipeline stage |
| t_f | Time to execute a single microbatch's forward pass |
| t_b | Time to execute a single microbatch's backward pass |
| t_id | Ideal processing time per iteration |
| t_pb | Pipeline bubble time |

**Mapping to our spec notation**: `p = N_pp`, `t = N_tp`, `d = N_dp`, `n = N_gpu`, `l = L`, `h = d_model`, `a = n_heads`.

---

## 2. Pipeline Parallelism Formulas (Section 2.2)

### 2.1 Default (Non-Interleaved) 1F1B Schedule -- Bubble Fraction

The GPipe schedule executes all forward passes first, then all backward passes. The pipeline bubble consists of `(p-1)` forward passes at the start and `(p-1)` backward passes at the end:

```
t_pb = (p - 1) * (t_f + t_b)
```

The ideal processing time for the batch is:

```
t_id = m * (t_f + t_b)
```

Therefore the **bubble time fraction** is:

```
Bubble fraction (default) = t_pb / t_id = (p - 1) / m
```

For the bubble to be small, we need `m >> p`.

**Spec comparison**: Our spec (Section 5.7) uses a slightly different form: `Bubble fraction = (N_pp - 1) / (num_microbatches + N_pp - 1)`. These are NOT equivalent. The paper's formula `(p-1)/m` is the ratio of bubble time to ideal compute time. The spec's formula `(N_pp-1)/(m + N_pp - 1)` is the ratio of bubble time to TOTAL time (ideal + bubble). The spec formula equals `(p-1) / (m + p - 1)`, which is the fraction of wall-clock time wasted. These are related by:
- Paper: bubble_fraction_of_ideal = (p-1)/m
- Spec: bubble_fraction_of_total = (p-1)/(m + p-1)

The paper's formula is more optimistic (denominates by ideal time only). For m >> p they converge. **ACTION**: The spec should clarify which convention it uses. The formula `(N_pp-1)/(m + N_pp-1)` is the more standard wall-clock fraction and is correct for our purpose.

**Memory implication of GPipe**: GPipe requires stashing activations for ALL m microbatches simultaneously, giving very high activation memory.

### 2.2 PipeDream-Flush (1F1B) Schedule Memory Advantage

The 1F1B schedule (PipeDream-Flush) limits in-flight microbatches to the pipeline depth `p`, not `m`:

> "This schedule requires activations to be stashed for p or fewer microbatches (compared to m microbatches for the GPipe schedule). Consequently, when m >> p, PipeDream-Flush is much more memory-efficient than GPipe."

The bubble time is the same as GPipe: `(p-1)/m`. The memory advantage is the key benefit.

### 2.3 Interleaved 1F1B Schedule -- Bubble Fraction (Section 2.2.2)

Each device performs computation for `v` model chunks (subsets of layers) instead of 1. If each device has `v` virtual stages, the forward/backward time for each chunk is `t_f/v` and `t_b/v`.

```
t_pb_interleaved = (p - 1) * (t_f + t_b) / v
```

The **interleaved bubble time fraction** is:

```
Bubble fraction (interleaved) = t_pb_int / t_id = (1/v) * (p - 1) / m
```

This reduces the bubble by a factor of `v` compared to non-interleaved.

**Constraint**: The number of microbatches in a batch must be an integer multiple of the number of pipeline stages. With `v` virtual stages per device and `p` pipeline stages, microbatches must be a multiple of `p * v` (from paper: "with 4 devices, the number of microbatches in a batch must be a multiple of 4").

**Spec comparison**: Our spec (Section 5.7) has: `Bubble fraction (interleaved) = (N_pp - 1) / (VP * num_microbatches + N_pp - 1)`. This is the wall-clock fraction form. The paper's `(1/v) * (p-1)/m` is the ideal-time fraction form. Again, for m >> p they converge.

**Communication cost**: The interleaved schedule increases communication by a factor of `v` because each virtual stage boundary requires a send/receive. The paper notes: "Quantitatively, the amount of communication also increases by v."

### 2.4 Pipeline Bubble Size vs Data Parallelism (Section 3.2)

When `d = 1` (no data parallelism), `t * p = n`. The bubble size in terms of `t`:

```
(p - 1) / m = (n/t - 1) / m
```

As `t` increases, the bubble decreases for fixed `B`, `b`, and `d`.

### 2.5 Pipeline Bubble Size with Data Parallelism (Section 3.3.1)

With `t = 1`, `m = b'/d = B/(b*d)`, and `p = n/(t*d) = n/d`:

```
(p - 1) / m = (n/d - 1) / (b'/d) = (n - d) / b'
```

---

## 3. Tensor Model Parallelism (Section 2.3)

### 3.1 MLP Partitioning

The MLP consists of two GEMMs with a GeLU non-linearity:

```
Y = GeLU(XA)
Z = Dropout(YB)
```

The weight matrix `A` is split column-wise: `A = [A_1, A_2]`. This allows the GeLU to be applied independently:

```
[Y_1, Y_2] = [GeLU(XA_1), GeLU(XA_2)]
```

The second weight matrix `B` is split row-wise: `B = [B_1; B_2]`, and `Y = [Y_1, Y_2]`. The output is then reduced across GPUs before the dropout layer.

### 3.2 Self-Attention Partitioning

The Q, K, V matrices are partitioned in a column-parallel fashion. The output linear layer has its weight partitioned across rows. This requires:
- 2 all-reduce operations in the **forward pass** (one after attention output projection, one after FFN second linear)
- 2 all-reduce operations in the **backward pass** (corresponding gradients)

Total: **4 all-reduce operations per layer per training step**.

### 3.3 TP Communication Volume (Section 3.2)

For pipeline parallelism, the communication between consecutive stages for each microbatch is:

```
Comm_PP_per_microbatch = b * s * h  (elements; in bytes: b * s * h * beta)
```

This is point-to-point communication (cheap).

For tensor model parallelism, tensors of size `b * s * h` need to be all-reduced among `t` model replicas **twice each** in the forward and backward pass for each layer. Total communication per layer per microbatch:

```
Comm_TP_per_layer_per_microbatch = 8 * b * s * h * ((t-1)/t)  (elements)
```

The factor 8 comes from: 2 all-reduces in forward + 2 all-reduces in backward, each all-reduce communicating `2 * (t-1)/t * b * s * h` elements (ring all-reduce).

Per device, per microbatch, the total TP communication for all layers in its pipeline stage is:

```
Comm_TP_per_device_per_microbatch = l^stage * 8 * b * s * h * ((t-1)/t)
```

Where `l^stage` is the number of layers in the pipeline stage.

**Spec comparison**: Our spec (Section 5.6) has: `Comm_tp_per_layer = 4 * 2 * (N_tp - 1) / N_tp * b * s * d * beta`. This matches: 4 operations, each with ring all-reduce cost `2 * (t-1)/t * b * s * h * beta`. The factor structure is consistent.

### 3.4 Scatter/Gather Communication Optimization (Section 4.1)

Without scatter/gather, the same tensor is sent redundantly 8 times between corresponding GPUs on adjacent pipeline stages over InfiniBand. With the optimization, each GPU scatters its chunk to the correct rank on the next node via InfiniBand, then does an NVLink all-gather to reconstruct.

**Result**: Communication between consecutive pipeline stages reduces from `b*s*h` per link to `b*s*h/t` per link:

```
Comm_PP_with_scatter_gather = b * s * h / t  (per pair of consecutive stages)
```

**Spec impact**: Our spec does not currently model scatter/gather optimization. This is relevant for multi-node PP configurations where TP is also used. The optimization reduces inter-node PP communication by `t`x (where `t` is the TP degree). This is significant: for t=8, PP communication drops to 1/8th.

---

## 4. Parameter Count Formula (Equation 2, Section 5.1)

The paper gives the total parameter count for a GPT model:

```
P = 12 * l * h^2 * (1 + 13/(12*h) + (V+s)/(12*l*h))
```

**Breaking this down:**
- `12*l*h^2` = baseline transformer parameters (the "12Ld^2" quick estimate from our spec)
- `13/(12*h)` correction = layer norm parameters, bias terms, and other per-layer small terms
- `(V+s)/(12*l*h)` correction = vocabulary embedding (`V*h`) + positional embedding (`s*h`) divided across the denominator

**For their configurations**: V = 51,200 (padded from 51,200; they use "vocabulary size of 51,200, multiple of 1024"), s = 2048.

**Spec comparison**: Our spec (Section 3.2) uses `Psi approx 12 * L * d^2`. The paper's formula includes the correction terms `13/(12h)` and `(V+s)/(12lh)`. For large models (h > 4096), `13/(12h) < 0.03%` and is negligible. The `(V+s)/(12lh)` term matters more for shallow/wide models or large vocabularies. Our spec handles these corrections through the detailed parameter count (Section 3.1) rather than correction terms, which is the right approach.

---

## 5. FLOPs Formula (Equation 3, Appendix)

### 5.1 Per-Layer FLOP Breakdown (Appendix: Floating-Point Operations)

The paper provides a detailed per-layer FLOP count:

**Attention block:**
- Key, query, value transformation: `6 * B * s * h^2` (3 projections, each `2*B*s*h^2`)
- Attention matrix computation (Q*K^T): `2 * B * s^2 * h`
- Attention over values (scores * V): `2 * B * s^2 * h`
- Post-attention linear projection: `2 * B * s * h^2`

Total attention: `8*B*s*h^2 + 4*B*s^2*h`

Note: The KQV term is `6*B*s*h^2`, NOT `8*B*s*h^2` as some sources give, because the paper counts only the 3 weight matrices (Q, K, V) not the output projection separately.

**Feed-forward network:**
- First linear (h -> 4h): `2 * B * s * h * 4h = 8*B*s*h^2`
- Second linear (4h -> h): `2 * B * s * 4h * h = 8*B*s*h^2`

Total FFN: `16*B*s*h^2`

**Per transformer layer, forward pass:**
```
FLOPs_per_layer_fwd = (8*B*s*h^2 + 4*B*s^2*h) + 16*B*s*h^2
                    = 24*B*s*h^2 + 4*B*s^2*h
```

**Backward pass**: 2x forward (gradients w.r.t. both input and weights).

**With activation recomputation**: Extra forward pass before backward = 3x forward total.

**Total per layer (with activation recomputation):**
```
FLOPs_per_layer_total = 4 * (24*B*s*h^2 + 4*B*s^2*h) = 96*B*s*h^2 * (1 + s/(6*h))
```

The factor 4 comes from: 1 forward + 1 recomputed forward + 2 backward.

**Note**: Without activation recomputation, the factor is 3 (1 forward + 2 backward):
```
FLOPs_per_layer_no_recomp = 3 * (24*B*s*h^2 + 4*B*s^2*h) = 72*B*s*h^2 * (1 + s/(6*h))
```

### 5.2 Full Model FLOPs (Equation 3)

Including the logit layer (lm_head):

```
F = 96 * B * s * l * h^2 * (1 + s/(6*h) + V/(16*l*h))
```

**Breaking this down:**
- `96*B*s*l*h^2` = transformer layers (with activation recomputation)
- `s/(6*h)` = attention quadratic term (same as the PaLM `12Lds` term)
- `V/(16*l*h)` = logit layer contribution. Forward: `2*B*s*h*V`, backward: `4*B*s*h*V`, total: `6*B*s*h*V = 96*B*s*l*h^2 * V/(16*l*h)`.

**Spec comparison**: Our spec (Section 4.1) uses `C = 6*Psi*D` or `C = (6*Psi + 12*L*d*s)*D`. The paper's formula is per-iteration (multiplied by `B*s`), while ours is per-token (multiplied by `D`). They are equivalent when accounting for the token count `D = iterations * B * s`. The paper explicitly includes activation recomputation in its base formula (factor 96 = 4 * 24), while our spec separates it (6Psi base + 33% overhead). The paper's `V/(16lh)` term corresponds to our spec's note about the logit output projection.

**Important note from paper**: "This is a lower bound for the true FLOP count but should be close to the actual value." They count one FLOP as a floating-point operation (multiply or add each count as 1). They also note: "We count a FLOP as a floating-point operation regardless of precision."

### 5.3 Training Time Estimate (Equation 4)

The paper derives a simplified training time formula. For configurations where `6h >> s`, `16lh >> (V+s)`, and `12h >> 13`:

```
End-to-end training time ≈ 8*T*P / (n*X)
```

Where:
- T = total training tokens
- P = total model parameters (from Equation 2)
- n = total GPUs
- X = achieved per-GPU teraFLOP/s throughput (empirically measured)

The factor 8 (not 6) accounts for activation recomputation (4x forward instead of 3x).

**Spec comparison**: Our spec uses `T = C / (N_gpu * F_peak * MFU)` which is equivalent: `C = 8*P*T` (with recomputation) and `X = F_peak * MFU`.

**Worked example from paper**: GPT-3 175B, T = 300B tokens, n = 1024 A100 GPUs, X = 140 TFLOP/s. Time = 8 * 300e9 * 175e9 / (1024 * 140e12) = **34 days**.

For 1T parameter model: T = 450B tokens, n = 3072, X = 163 TFLOP/s. Time = 8 * 450e9 * 1e12 / (3072 * 163e12) = **84 days** (~3 months).

---

## 6. Model Configurations and Throughput (Table 1)

This is the key scaling table from the paper:

| Params (B) | Heads | Hidden | Layers | TP | PP | GPUs | Batch | TFLOP/s/GPU | % Peak | Agg PF/s |
|-----------|-------|--------|--------|----|----|------|-------|-------------|--------|----------|
| 1.7 | 24 | 2304 | 24 | 1 | 1 | 32 | 512 | 137 | 44% | 4.4 |
| 3.6 | 32 | 3072 | 30 | 2 | 1 | 64 | 512 | 138 | 44% | 8.8 |
| 7.5 | 32 | 4096 | 36 | 4 | 1 | 128 | 512 | 142 | 46% | 18.2 |
| 18.4 | 48 | 6144 | 40 | 8 | 1 | 256 | 1024 | 135 | 43% | 34.6 |
| 39.1 | 64 | 8192 | 48 | 8 | 2 | 512 | 1536 | 138 | 44% | 70.8 |
| 76.1 | 80 | 10240 | 60 | 8 | 4 | 1024 | 1792 | 140 | 45% | 143.8 |
| 145.6 | 96 | 12288 | 80 | 8 | 8 | 1536 | 2304 | 148 | 47% | 227.1 |
| 310.1 | 128 | 16384 | 96 | 8 | 16 | 1920 | 2160 | 155 | 50% | 297.4 |
| 529.6 | 128 | 20480 | 105 | 8 | 35 | 2520 | 2520 | 163 | 52% | 410.2 |
| 1008.0 | 160 | 25600 | 128 | 8 | 64 | 3072 | 3072 | 163 | 52% | 502.0 |

**Key observations for our calculator:**
1. **TP stays at 8 for all models >= 18.4B** -- confirming TP should max out at the number of GPUs per node (8 for DGX A100)
2. **PP starts at models > 18.4B** -- when the model no longer fits in a single node even with TP=8
3. **Peak efficiency improves with model size** (44% at 1.7B, 52% at 529B-1T) due to larger matmuls having better arithmetic intensity
4. **MFU range**: 43-52% on A100, validating our spec's "40-50%" range for large models
5. All models use V = 51,200 (padded to multiple of 1024) and s = 2048
6. Batch sizes scale with model size (512 at 1.7B to 3072 at 1T)

**Spec impact**: Our MFU guidelines (Section 6.3) show "Large model (10B-100B): 40-50%" and "Very large (100B+): 35-45%". The paper's data shows 43-52% for 1.7B-1T on A100. This is consistent -- A100-era efficiency was slightly better than H100-era at equivalent scale because the models at 100B+ scale on H100 tend to use more complex parallelism. No spec change needed.

---

## 7. Combined Parallelism Memory Model

### 7.1 Takeaway #1: TP First, PP Second

> "When considering different forms of model parallelism, tensor model parallelism should generally be used up to degree g when using g-GPU servers, and then pipeline model parallelism can be used to scale up to larger models across servers."

This is the same guidance as our spec's recommendation engine (Section 9): TP within node first (up to 8), then PP across nodes.

### 7.2 Takeaway #2: Data Parallelism for Scaling

> "A total model-parallel size of M = t * p should be used so that the model's parameters and intermediate metadata fit in GPU memory; data parallelism can be used to scale up training to more GPUs."

### 7.3 Takeaway #3: Microbatch Size Matters

> "The optimal microbatch size b depends on the throughput and memory footprint characteristics of the model, as well as the pipeline depth p, data-parallel size d, and batch size B."

Microbatch size affects both arithmetic intensity (larger b = better GPU utilization) and pipeline bubble (larger b = fewer microbatches m = larger bubble).

### 7.4 Memory Partitioning

With combined TP + PP:
- **Parameters per GPU**: `Psi / (t * p)` approximately (layers split by PP, each layer's weights split by TP)
- **Optimizer states per GPU**: Proportional to parameters per GPU, further sharded by ZeRO across DP dimension
- **Activations per GPU**: Per-layer activations reduced by TP (the `24/t` and `5as/(d*t)` terms from Korthikanti), number of layers reduced by PP (`l/p` layers per stage)

The paper does not give explicit activation memory formulas -- those come from the follow-up Korthikanti et al. (2022) paper. But it establishes the framework:

```
Total memory per GPU = (Psi_params/(t*p)) * Phi_per_param  [model states]
                     + activation_per_layer * (l/p)         [activations]
                     + buffers                               [communication]
```

---

## 8. Activation Recomputation Analysis (Section 3.5)

### 8.1 Optimal Checkpoint Interval

The paper presents the optimization for choosing the number of activation checkpoints `c`:

Given `l` layers per pipeline stage and:
- `A^input` = size of input activations of a layer
- `A^intermediate` = size of intermediate activations per layer

Total memory footprint with `c` checkpoints:

```
M_activations = c * A^input + (l/c) * A^intermediate
```

The minimum of this function occurs at:

```
c_optimal = sqrt(l * (A^intermediate / A^input))
```

**In practice**: "checkpointing every 1 or 2 transformer layers is optimal."

**Spec comparison**: Our spec discusses full checkpointing (`2*s*b*d` per layer) and selective checkpointing but does not include the optimal checkpoint interval formula. **ACTION**: Consider adding this formula as guidance for "partial checkpointing" -- it's the NeMo `recompute_num_layers` parameter. Our spec does mention this in Section 5.3 under "Block-level partial recomputation" but doesn't provide the optimization formula.

### 8.2 Memory Impact

With activation recomputation:
- Only **input activations** for each pipeline stage need to be stored (not intermediate activations)
- For 1F1B schedule: activations for `p` microbatches (pipeline depth) need to be kept
- For GPipe: activations for all `m` microbatches need to be kept

---

## 9. ZeRO-3 Comparison (Table 2, Section 5.2)

The paper compares PTD-P (Pipeline-Tensor-Data Parallelism) against ZeRO-3 without model parallelism:

| Scheme | Params (B) | Model-parallel | Batch | GPUs | Microbatch | TFLOP/s/GPU | Train time 300B (days) |
|--------|-----------|----------------|-------|------|------------|-------------|------------------------|
| ZeRO-3 | 174.6 | 1 | 1536 | 384 | 4 | 144 | 90 |
| ZeRO-3 | 174.6 | 1 | 1536 | 768 | 2 | 88 | 74 |
| ZeRO-3 | 174.6 | 1 | 1536 | 1536 | 1 | 44 | 74 |
| ZeRO-3 | 529.6 | 1 | — | 2560* | 640 | 4 | 138 | 169 |
| ZeRO-3 | 529.6 | 1 | 2240 | 1120 | 2 | 98 | 137 |
| ZeRO-3 | 529.6 | 1 | 2240 | 2240 | 1 | 48 | 140 |
| PTD-P | 174.6 | 96 | 1536 | 384 | 1 | 153 | 84 |
| PTD-P | 174.6 | 96 | 1536 | 768 | 1 | 149 | 43 |
| PTD-P | 174.6 | 96 | 1536 | 1536 | 1 | 141 | 23 |
| PTD-P | 529.6 | 280 | 2240 | 560 | 1 | 171 | 156 |
| PTD-P | 529.6 | 280 | 2240 | 1120 | 1 | 167 | 80 |
| PTD-P | 529.6 | 280 | 2240 | 2240 | 1 | 159 | 42 |

*Note: The 530B model did not fit on 560 GPUs with ZeRO-3 (microbatch size 4), requiring 2560 GPUs.

**Key finding**: PTD-P outperforms ZeRO-3 by **70%** throughput when doubling GPUs (keeping batch size the same). This is because PTD-P uses TP within a node (NVLink communication) rather than requiring all-gathers across all nodes.

**Spec impact**: This validates our spec's guidance (Section 9) that ZeRO-3 should be a fallback when TP+PP cannot be used, not a primary strategy for very large models. PTD-P achieves 141-171 TFLOP/s per GPU vs 44-153 for ZeRO-3 at the same model size.

---

## 10. TP Communication Analysis (Section 3.2)

### 10.1 TP vs PP Communication Trade-off

**PP communication** per microbatch (between consecutive stages):
```
Comm_PP = b * s * h  (point-to-point, cheap)
```

**TP communication** per layer per microbatch (all-reduce):
```
Comm_TP_per_layer = 8 * b * s * h * (t-1)/t
```

Per device per microbatch (all layers in stage):
```
Comm_TP_per_device = l^stage * 8 * b * s * h * (t-1)/t
```

### 10.2 Effective Communication with Scatter/Gather (Section 4.1)

With scatter/gather optimization:
```
Comm_PP_optimized = b * s * h / t  (per pair of consecutive stages)
```

This is key for multi-node training: the optimization reduces PP inter-node communication by `t`x at the cost of more NVLink intra-node communication (all-gather after scatter).

---

## 11. Vocabulary Padding (Section 5.1)

The paper states:

> "All models use a vocabulary size (denoted by V) of 51,200 (multiple of 1024)"

The actual GPT vocabulary is 50,257 tokens. The paper pads to 51,200 which is divisible by 1024. Megatron-LM pads the vocabulary to be divisible by `128 * t` for TP compatibility.

**Spec comparison**: Our spec (Section 9 Constraints) already includes this: "Megatron-LM pads the vocabulary size to be divisible by `128 * N_tp`". The paper confirms this with their V = 51,200 = 50 * 1024 for the 1T model with TP=8 (51,200 / (128 * 8) = 50, clean division).

---

## 12. TP Constraints and Divisibility

From the paper's tensor parallelism description:
- **Attention heads must be divisible by TP degree**: The Q, K, V matrices are partitioned column-wise in a column-parallel fashion. The output linear layer has its weight partitioned across rows. This requires each GPU to handle an equal number of attention heads.
- **FFN intermediate dimension must be divisible by TP degree**: The first GEMM (h -> 4h) is split column-wise (`A = [A_1, A_2]`), and the second GEMM (4h -> h) is split row-wise (`B = [B_1; B_2]`).
- **TP should not exceed GPUs per node**: The paper strongly recommends TP within a single multi-GPU server due to NVLink bandwidth requirements.

**Spec comparison**: Our spec already covers all these constraints (Section 9).

---

## 13. Empirical Scaling Observations

### 13.1 Super-linear Scaling (Table 1 discussion)

> "We see super-linear scaling to 3072 A100 GPUs (384 DGX A100 nodes), since GPU utilization improves as the models get larger (larger matrix multiplications) without significant increase in the communication time relative to computation time."

This is important: for our calculator, MFU should increase with model size (not be constant), validating the tiered MFU defaults in our spec.

### 13.2 TP vs PP (Figure 13)

For a 162.2B model on 64 A100s with batch size 128:
- (PP=2, TP=32): ~75 TFLOP/s
- (PP=4, TP=16): ~125 TFLOP/s
- (PP=8, TP=8): ~165 TFLOP/s (best)
- (PP=16, TP=4): ~145 TFLOP/s
- (PP=32, TP=2): ~100 TFLOP/s

**Peak performance is at TP=8** (single node), confirming the heuristic that TP should equal GPUs per node.

### 13.3 Microbatch Size Impact (Figure 16)

For a 91B model with (t,p) = (8,8):
- Microbatch 1: ~120 TFLOP/s (batch 128) / ~100 TFLOP/s (batch 512)
- Microbatch 2: ~160 TFLOP/s (batch 128) / ~170 TFLOP/s (batch 512) -- best
- Microbatch 4: ~140 TFLOP/s (both batch sizes)
- Microbatch 8: ~100 TFLOP/s (both batch sizes)

**Optimal microbatch is 2 for this configuration**, not 1 and not the largest possible. This is because larger microbatches reduce the number of microbatches `m`, increasing the bubble, while too-small microbatches underutilize GPU compute.

### 13.4 Activation Recomputation (Figure 17, Section 5.6)

For a 145B model with 128 A100 GPUs, (t,p) = (8,16):
- Without recomputation: Higher throughput at small batch sizes (up to ~33% faster at batch 2-4)
- With recomputation: Higher throughput at large batch sizes (up to 2x at batch 128-256)
- Crossover: Around batch size 8-16

> "Activation recomputation leads to up to 33% lower throughput... due to the extra forward pass. However, activation recomputation is needed to support larger batch sizes."

**Spec comparison**: Our spec notes ~33% compute overhead for full recomputation and ~39% wall-clock overhead. The paper's data is consistent.

### 13.5 Scatter/Gather Optimization (Figure 18)

For 175B model on 96 GPUs with interleaved schedule:
- Without optimization: ~120 TFLOP/s at large batch
- With optimization: ~135 TFLOP/s at large batch
- Improvement: **up to 11% throughput gain** for communication-intensive schedules

### 13.6 Fused Operators (Section 5.8)

- 175B model: throughput increased by **19%** with fusion (113 -> 135 TFLOP/s)
- 530B model: throughput increased by **11%** (133 -> 148 TFLOP/s)

This establishes the importance of kernel fusion for training throughput.

---

## 14. Inter-Node Communication Bandwidth (Section 5.9)

On the 1T parameter model with 3072 GPUs:
- **PP communication**: 892 GB/s effective bisection bandwidth (point-to-point between pipeline stages)
- **DP communication**: 12.9 TB/s effective bisection bandwidth (all-reduce across data-parallel replicas)
- **Interconnect**: 8 NVIDIA Mellanox 200Gbps HDR InfiniBand HCAs per node

---

## 15. Checkpoint Size (Section 5.10)

> "The trillion-parameter model has a checkpoint of size 13.8 terabytes."

For the 1T model: 13.8 TB / 1.008T params = 13.7 bytes/param. This is consistent with mixed-precision training checkpoints that persist the low-precision model weights plus fp32 master weights and Adam moments (2 + 4 + 4 + 4 = 14 bytes/param), with small differences from exact parameter count, formatting, and metadata.

> "The initial load of checkpoints for the trillion-parameter model by all 384 nodes (3072 GPUs) reaches a peak read bandwidth of 1TB/s."

**Spec comparison**: Our spec (Section 5.1) gives restart checkpoint size as model parameters plus optimizer states. For default mixed AdamW this is `14 * Psi bytes` before file-format overhead, close to the paper's ~13.7 bytes/param observation.

---

## 16. Hardware Details

- **Platform**: NVIDIA DGX A100 nodes (Selene supercomputer)
- **GPUs per node**: 8 NVIDIA 80-GB A100 GPUs
- **Intra-node**: NVLink and NVSwitch
- **Inter-node**: 8 Mellanox 200Gbps HDR InfiniBand HCAs (application) + 2 HCAs (storage)
- **Network topology**: Three-level (leaf, spine, core) fat-tree with 850 switches
- **Filesystem**: All-NVMe shared parallel filesystem
- **Peak BF16 throughput per GPU**: 312 TFLOP/s (A100)
- **Mixed precision**: All experiments used mixed precision (bf16/fp32)
- **Data layout**: Changed from `[b, s, a, h]` to `[s, b, a, h]` to enable strided batched GEMMs and avoid memory-intensive transpose operations

---

## 17. What's Unique / Non-Obvious

### 17.1 The Interleaved Schedule's Memory-Communication Trade-off

The interleaved schedule reduces bubble by `v`x but increases communication by `v`x. The paper shows this is worthwhile because the scatter/gather optimization keeps communication costs manageable. Without scatter/gather, the interleaved schedule actually performs WORSE than the default at large batch sizes.

### 17.2 Pipeline Bubble Convention

The paper uses `(p-1)/m` which denominates by ideal time, while most calculators use `(p-1)/(m+p-1)` which denominates by total time. The difference matters when `m` is small relative to `p`. For `p=8, m=16`: paper gives 43.75%, total-time gives 30.4%. For `p=8, m=64`: paper gives 10.9%, total-time gives 9.9%.

### 17.3 Data Layout Optimization

The paper changed the data layout from `[batch, seq, heads, hidden]` to `[seq, batch, heads, hidden]` to avoid memory-intensive transpose operations and enable strided batched GEMM kernels. This is a framework implementation detail but affects real throughput significantly (part of the 19% gain from fused operators).

### 17.4 TP Creates Smaller Matrices, Reducing GPU Utilization

> "A high degree of model parallelism can create small matrix multiplications (GEMMs), potentially decreasing GPU utilization."

This is a subtle point: while TP reduces per-GPU memory, it also reduces the size of each GEMM, which can push the computation from compute-bound to memory-bandwidth-bound. This is why TP > 8 (or even TP > 4 for smaller models) can decrease throughput.

### 17.5 The Training Time Formula Uses 8TP Not 6TP

The paper's time formula (Equation 4) uses `8*T*P/(n*X)` because it assumes activation recomputation (factor 4 instead of 3 for forward+backward multiplier). Our spec separates this: base formula uses `6*Psi*D` with a separate recomputation overhead multiplier. Both approaches are valid but this could cause confusion when comparing.

### 17.6 530B ZeRO-3 Does Not Fit on 560 GPUs

The 530B model with ZeRO-3 and microbatch size 4 could not fit on 560 GPUs, requiring 640 GPUs with reduced batch size. This demonstrates that ZeRO-3 alone is insufficient for very large models -- you still need significant per-GPU memory for the current layer's gathered parameters and activations.

---

## 18. Formulas Summary for Calculator Implementation

### Pipeline Bubble Formulas

```python
# Non-interleaved (1F1B or GPipe)
# Wall-clock fraction (spec convention):
bubble_fraction_default = (p - 1) / (m + p - 1)

# Interleaved with v virtual stages:
# Wall-clock fraction:
bubble_fraction_interleaved = (p - 1) / (v * m + p - 1)

# Paper convention (fraction of ideal time):
bubble_fraction_paper_default = (p - 1) / m
bubble_fraction_paper_interleaved = (1 / v) * (p - 1) / m
```

### Parameter Count

```python
# Full formula (Equation 2)
P = 12 * l * h**2 * (1 + 13/(12*h) + (V + s)/(12*l*h))

# Simplified
P_approx = 12 * l * h**2
```

### FLOPs (Equation 3, with activation recomputation)

```python
# With activation recomputation
F_recomp = 96 * B * s * l * h**2 * (1 + s/(6*h) + V/(16*l*h))

# Without activation recomputation (factor 72 instead of 96)
F_no_recomp = 72 * B * s * l * h**2 * (1 + s/(6*h) + V/(16*l*h))
```

### Training Time (Equation 4, simplified)

```python
# Simplified (assumes 6h >> s, 16lh >> V+s, 12h >> 13)
T_training = 8 * T_tokens * P / (n_gpus * X_tflops_per_gpu)
```

### TP Communication Volume

```python
# Per layer, per microbatch (total across forward + backward)
comm_tp_per_layer = 8 * b * s * h * (t - 1) / t  # in elements

# Per device, per microbatch
comm_tp_per_device = l_per_stage * comm_tp_per_layer

# With scatter/gather, PP communication
comm_pp_per_microbatch = b * s * h / t  # reduced by t
```

### Activation Recomputation Optimal Checkpoints

```python
import math
c_optimal = math.sqrt(l_per_stage * (A_intermediate / A_input))
M_activations = c_optimal * A_input + (l_per_stage / c_optimal) * A_intermediate
```

### Microbatch Count

```python
m = B / (b * d)  # microbatches per pipeline
# Constraint for interleaved: m must be divisible by p
```

---

## 19. Spec Change Recommendations

### 19.1 Already Covered in Spec (Confirmed by Paper)

1. **Pipeline bubble formula** -- spec has the wall-clock fraction form, which is correct for our purposes
2. **TP communication volume** (Section 5.6) -- matches paper's 4 all-reduce per layer
3. **TP <= GPUs per node heuristic** (Section 9) -- confirmed by paper's empirical data
4. **Vocabulary padding** (Section 9 Constraints) -- confirmed by paper's V=51,200
5. **MFU ranges** (Section 6.3) -- paper's 43-52% on A100 is consistent with spec's ranges
6. **Activation recomputation overhead** -- spec's 33% compute / 39% wall-clock matches
7. **Most-loaded stage concept** (Section 5.7) -- spec handles this
8. **FLOPs formula with attention quadratic term** -- spec's PaLM formula captures this
9. **Training time formula** -- spec's `T = C / (N_gpu * F_peak * MFU)` is equivalent

### 19.2 Missing from Spec / Should Be Added

1. **Scatter/gather communication optimization for PP**: When TP is combined with PP, inter-node PP communication drops by `t`x. Formula: `Comm_PP = b*s*h/t` instead of `b*s*h`. This matters for multi-node PP+TP configurations and should be noted in Section 5.6 or 6.4.

2. **Optimal activation checkpoint interval formula**: `c_optimal = sqrt(l * (A_intermediate / A_input))` for partial checkpointing. This is the mathematical basis for NeMo's `recompute_num_layers` and should be added to Section 5.3 alongside the existing "Block-level partial recomputation" discussion.

3. **Interleaved schedule communication overhead**: The interleaved schedule increases PP communication by `v`x (number of virtual stages). This should be added to the interleaved PP discussion in Section 5.7.

4. **Interleaved schedule microbatch constraint**: Microbatches must be a multiple of `p` (not just `>= p-1`). This is a stronger constraint than the existing 1F1B minimum. Should be noted in Section 9 Constraints.

5. **Paper's exact parameter formula** (Equation 2): `P = 12*l*h^2 * (1 + 13/(12h) + (V+s)/(12lh))` as a closed-form alternative to the detailed parameter count. Could be useful as a Quick Mode accuracy improvement.

### 19.3 Different from Spec / Clarification Needed

1. **Paper's FLOPs formula bakes in activation recomputation** (96 factor = 4*24) while spec separates base FLOPs (6PsiD) from recomputation overhead. Both are correct but users comparing against the paper need to know this.

2. **Pipeline bubble convention**: The paper uses `(p-1)/m` (fraction of ideal time) while the spec uses `(p-1)/(m+p-1)` (fraction of total time). These are not the same and diverge significantly when `m` is comparable to `p`. The spec should explicitly state it uses the total-time convention and note that some references (including this paper) use the ideal-time convention.

### 19.4 Does Not Affect Spec (Context/Validation Only)

1. The paper's comparison table (Table 2) showing PTD-P outperforming ZeRO-3 by 70% validates the spec's guidance that ZeRO-3 is a fallback.
2. The checkpoint size of 13.8 TB for 1T model validates the revised restart checkpoint formula of model parameters plus optimizer state, about 14 bytes/parameter for default mixed AdamW.
3. The data layout optimization `[s,b,a,h]` is an implementation detail, not a calculator concern.
4. The 534B model not fitting on 560 GPUs with ZeRO-3 confirms the minimum GPU memory floor concept.

---

## 20. Files and Paths Referenced

- **Downloaded PDF**: `/Users/vinitvyas09/.claude/projects/-Users-vinitvyas09-code-personal-llm-training-gpu-calculator/d91b4879-75e5-4f21-8298-32855ec485d1/tool-results/webfetch-1774994741679-42mnxp.pdf`
- **Current spec**: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/spec/llm-training-gpu-calculator-spec.md`
- **Research sources list**: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/spec/research-sources.md`
