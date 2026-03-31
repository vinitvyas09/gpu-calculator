# Reducing Activation Recomputation in Large Transformer Models -- Deep Dive

**Paper:** Korthikanti, Casper, Lym, McAfee, Andersch, Shoeybi, Catanzaro (NVIDIA), 2022
**ArXiv:** https://arxiv.org/abs/2205.05198
**Date of analysis:** 2026-03-31

---

## 1. Why This Paper Matters for Our Calculator

This is the primary source for activation memory formulas in transformer training. It introduces:
- The canonical `sbh(34 + 5as/h)` formula with full component-level derivation
- Selective activation recomputation (recompute only cheap-to-recompute, large-memory activations)
- Sequence parallelism (partition non-tensor-parallel regions along sequence dim)
- The precise relationship between Model FLOPs Utilization (MFU) and Hardware FLOPs Utilization (HFU)
- Exact formulas for how each parallelism strategy reduces activation memory

All formulas use **bytes** as the unit and assume **mixed precision (fp16/bf16)** training where activations are stored in 16-bit format (2 bytes per element) except dropout masks (1 byte per element).

---

## 2. Variable Definitions (Table 1)

| Variable | Definition |
|----------|-----------|
| `a` | number of attention heads |
| `b` | microbatch size |
| `h` | hidden dimension size |
| `L` | number of transformer layers |
| `p` | pipeline parallel size |
| `s` | sequence length |
| `t` | tensor parallel size |
| `v` | vocabulary size |

**IMPORTANT naming note:** The paper uses `h` for hidden dimension. Many other references (including the Megatron codebase) use `d` or `d_model`. Our spec should be aware that `sbh(34 + 5as/h)` and `sbd(34 + 5as/d)` are the same formula.

---

## 3. The Full Activation Memory Formula: Equation 1

### 3.1 The Formula

```
Activations memory per layer = sbh * (34 + 5*a*s/h)   [bytes]
```

This is Equation (1) in the paper. It applies to a single transformer layer with **no model parallelism**.

### 3.2 Complete Breakdown of the 34 Constant

The paper derives this by walking through every operation in a transformer layer. The architecture is: LayerNorm -> Self-Attention -> Dropout -> Add -> LayerNorm -> MLP -> Dropout -> Add.

**Assumption:** All activations stored in fp16 (2 bytes/element), except dropout masks (1 byte/element). Layer norm mean/variance (2*sb elements each) are negligible vs sbh since h >> 1.

#### 3.2.1 Attention Block: 11*sbh + 5*a*s^2*b

| Component | Storage | Size (bytes) | Notes |
|-----------|---------|-------------|-------|
| Linear projection input (QKV) | Input activations | 2*sbh | Shared input to Q, K, V projections |
| Attention dropout mask | Mask | sbh | 1 byte per element, sbh elements |
| Q matrix | For QK^T backprop | 2*sbh | Stored after QKV linear |
| K matrix | For QK^T backprop | 2*sbh | Stored after QKV linear |
| QK^T result | Before softmax | 2*a*s^2*b | Shape: [b, a, s, s], fp16 |
| Softmax output | For backprop | 2*a*s^2*b | Shape: [b, a, s, s], fp16 |
| Softmax dropout mask | Mask | a*s^2*b | 1 byte per element |
| Attention-over-values output | Dropout output (2as^2b) + V (2sbh) | 2*a*s^2*b + 2*sbh | Need dropout output and V for backprop |
| **Attention subtotal** | | **11*sbh + 5*a*s^2*b** | |

Detailed element count for attention:
- Input to QKV linear: 2*sbh (fp16 input)
- Dropout mask after attention: 1*sbh
- Q: 2*sbh
- K: 2*sbh  
- V: 2*sbh (stored as part of attention-over-values)
- QK^T: 2*a*s^2*b
- Softmax output: 2*a*s^2*b
- Softmax dropout mask: 1*a*s^2*b
- Dropout output (softmax dropout applied): 2*a*s^2*b (needed for attention-over-values backprop -- note: this is the same as storing dropout(softmax(QK^T)) which is used to compute dropout(softmax(QK^T)) @ V)
- Post-attention linear input: 2*sbh (this is the output of attention-over-values)

Wait -- let me re-derive this more carefully from the paper text:

**The paper's exact accounting (Section 4.1):**

1. "The linear projection stores its input activations with size **2sbh** and the attention dropout requires a mask with size **sbh**" -- This is the output linear projection (after self-attention) and the dropout after it.

2. Self-attention elements:
   - "Q, K, V matrix multiplies: We only need to store their shared input with size **2sbh**" -- input to QKV
   - "QK^T matrix multiply: It requires storage of both Q and K with total size **4sbh**" -- Q and K separately
   - "Softmax: Softmax output with size **2as^2b** is required for back-propagation"
   - "Softmax dropout: Only a mask with size **as^2b** is needed"
   - "Attention over Values (V): We need to store the dropout output (2as^2b) and the Values (2sbh) and therefore need **2as^2b + 2sbh** of storage"

Summing attention block:
- 2sbh (linear projection input) + sbh (attention dropout mask) + 2sbh (QKV shared input) + 4sbh (Q and K) + 2as^2b (softmax) + as^2b (softmax dropout mask) + 2as^2b (attention dropout output) + 2sbh (V)
- = (2 + 1 + 2 + 4 + 2)*sbh + (2 + 1 + 2)*as^2b
- = **11*sbh + 5*a*s^2*b**

#### 3.2.2 MLP Block: 19*sbh

| Component | Storage | Size (bytes) | Notes |
|-----------|---------|-------------|-------|
| First linear (h->4h) input | Input activations | 2*sbh | fp16 |
| Second linear (4h->h) input | Input activations | 8*sbh | 4h width, fp16: 2*sb*(4h) = 8sbh |
| GeLU input | For GeLU backprop | 8*sbh | 4h width, fp16: needs input to recompute derivative |
| Dropout mask | Mask | sbh | 1 byte per element |
| **MLP subtotal** | | **19*sbh** | |

Note on the GeLU: The paper says "The GeLU non-linearity also needs its input with size 8sbh for back-propagation." The GeLU derivative requires the input value (unlike ReLU which only needs the sign). This is why the GeLU input (which is 4h wide) must be stored: 2 * sb * 4h = 8sbh.

**Critical detail:** The paper assumes `d_ff = 4h` (the standard GPT-3 ratio). If `d_ff != 4h`, the 19 constant changes. Specifically:
- The 8sbh terms (two of them: GeLU input and second linear input) become `2*sb*d_ff` each = `2*sb*d_ff * 2 = 4*sb*d_ff`
- The formula becomes: `2sbh + 4*sb*d_ff + sbh = 3sbh + 4*sb*d_ff`
- With d_ff = 4h: 3sbh + 16sbh = 19sbh (checks out)

**Generalized MLP formula:**
```
MLP activations = sbh * (3 + 4*d_ff/h)   [bytes]
```

#### 3.2.3 Layer Norms: 4*sbh

| Component | Storage | Size (bytes) | Notes |
|-----------|---------|-------------|-------|
| LayerNorm 1 input | For backprop | 2*sbh | fp16, before attention block |
| LayerNorm 2 input | For backprop | 2*sbh | fp16, before MLP block |
| **LayerNorm subtotal** | | **4*sbh** | |

The paper notes: "Each layer norm stores its input with size 2sbh and therefore in total, we will need 4sbh of storage." The mean and variance (2*sb each) are negligible since h is large (order of thousands).

#### 3.2.4 Grand Total

```
Attention:  11*sbh + 5*a*s^2*b
MLP:        19*sbh
LayerNorm:   4*sbh
─────────────────────────
Total:      34*sbh + 5*a*s^2*b = sbh*(34 + 5*a*s/h)
```

The `5*a*s^2*b` term is factored as `sbh * 5*a*s/h` to show it as a correction to the base 34 constant.

### 3.3 When is the 5as/h Term Significant?

For large models where h >> s, the term is small. For models where s is large relative to h, it dominates.

| Model | a | s | h | 5as/h | Relative to 34 |
|-------|---|---|---|-------|---------------|
| GPT-3 (175B) | 96 | 2048 | 12288 | 80 | 2.35x (70% of total) |
| MT-NLG (530B) | 128 | 2048 | 20480 | 64 | 1.88x (65% of total) |
| GPT-2 (1.5B) | 25 | 1024 | 1600 | 80 | 2.35x |
| LLaMA-70B | 64 | 4096 | 8192 | 160 | 4.7x (82% of total) |

**Key insight for our calculator:** For modern long-context models (s=4096, 8192, or more), the 5as/h term DOMINATES the memory. This is precisely the term that selective recomputation eliminates and that Flash Attention also eliminates.

---

## 4. Tensor Parallelism Formula: Equation 2

With t-way tensor parallelism (Megatron-style, splitting attention heads and MLP columns/rows):

```
Activations memory per layer = sbh * (10 + 24/t + 5*a*s/(h*t))   [bytes]
```

### 4.1 Derivation of the 10 and 24 Split

Tensor parallelism parallelizes the attention and MLP blocks but leaves layer norms and dropouts **replicated** across all tensor-parallel ranks.

**Replicated (not divided by t) -- the "10" term:**
- 2 layer norm inputs: 2 * 2sbh = 4sbh
- 2 dropout masks: 2 * sbh = 2sbh  
- 2 dropout inputs (for the dropout after attention and after MLP): 2 * 2sbh = 4sbh

Total replicated: 4 + 2 + 4 = **10*sbh**

**Parallelized (divided by t) -- the "24" term:**
The remaining 34 - 10 = 24 comes from activations inside the attention and MLP blocks that are naturally split across tensor-parallel ranks:
- QKV shared input: 2sbh (actually this is replicated... but the Q,K,V outputs are split)
- Q: 2sbh/t, K: 2sbh/t
- GeLU input: 8sbh/t
- Second linear input: 8sbh/t
- Other attention internals

Total parallelized: **24*sbh/t**

And the attention scores term becomes: **5*a*s^2*b/t** = sbh * 5as/(ht)

Hence: `sbh * (10 + 24/t + 5as/(ht))`

### 4.2 Critical Observation

The 10sbh term is NOT divided by t. This means that as t grows, the replicated portion dominates. For t=8, the formula becomes:
```
sbh * (10 + 3 + 5as/(8h)) = sbh * (13 + 5as/(8h))
```
So roughly 10/13 = 77% of the non-attention-score memory is wasted replication. This is the motivation for sequence parallelism.

---

## 5. Sequence Parallelism Formula: Equation 4

Sequence parallelism partitions the replicated regions (layer norms, dropouts) along the sequence dimension. The key insight: these operations are independent along the sequence dimension, so each tensor-parallel rank only needs to store 1/t of the sequence for these operations.

### 5.1 Communication Pattern

Sequence parallelism introduces new collective operations:
- `g`: all-gather in forward, reduce-scatter in backward
- `g_bar`: reduce-scatter in forward, all-gather in backward

These replace the existing `f` (identity forward, all-reduce backward) and `f_bar` (all-reduce forward, identity backward) operators. Since a ring all-reduce = reduce-scatter + all-gather, the total communication volume is unchanged. **Sequence parallelism adds zero communication overhead.**

### 5.2 The Formula

```
Activations memory per layer = sbh * (10/t + 24/t + 5*a*s/(h*t))
                              = sbh/t * (34 + 5*a*s/h)   [bytes]
```

This is exactly Equation 1 divided by t. Both the previously-replicated 10sbh AND the previously-parallelized 24sbh are now divided by t.

**The 5x improvement claim:** Compared to tensor-parallel baseline (Eq 2), sequence parallelism provides roughly a factor of (10 + 24/t) / (34/t) improvement for the non-attention terms. For t=8: (10 + 3) / (34/8) = 13 / 4.25 = 3.06x improvement on the non-attention terms alone.

---

## 6. Pipeline Parallelism Formula: Equation 5

### 6.1 The Formula

```
Total activations memory = sbhL/t * (34 + 5*a*s/h)   [bytes]
```

**Key subtlety:** Pipeline parallelism does NOT divide by p. The first pipeline stage must store activations for L layers worth of microbatches (p microbatches in flight, each with L/p layers = L total layer-activations).

The paper explains: "To keep the pipeline pressurized and avoid extra idle time, the first stage must store activations for p microbatches. Each stage contains L/p layers so the first stage must store p * L/p = L layers worth of activations."

### 6.2 Interleaved Pipeline Schedule Correction

For interleaved pipeline schedules (e.g., Megatron-LM's interleaving with m virtual stages per rank), the formula needs a correction factor:

```
Total activations memory = sbhL/t * (34 + 5*a*s/h) * (1 + (p-1)/(p*m))
```

Where m is the number of interleaving stages. This comes from the interleaved schedule requiring activations for `L * (1 + (p-1)/(p*m))` layers.

For non-interleaved (m=1): the factor becomes `(1 + (p-1)/p)` which approaches 2 for large p.

### 6.3 Non-Uniform Memory Across Pipeline Ranks

The paper's Appendix B shows that activation memory is NOT uniform across pipeline ranks:
- Rank 0 (first stage) has the highest memory: it stores activations for ALL in-flight microbatches
- Memory decreases linearly along pipeline ranks
- The output-tensor-deallocation optimization saves `sbh*r` memory per rank, where r is the number of microbatches in flight at that rank, peaking at r=p on the first stage

For the 530B model config (s=2048, b=1, h=20480, p=35), the savings from deallocation on the first pipeline stage = sbhp = 2048*1*20480*35 * 2 bytes = 2.73 GB.

### 6.4 Extra Activation Memory (Section 4.3)

Beyond the per-layer formula, there is additional memory for:
- Input embeddings dropout: `sbhp/t` (parallelized along sequence dim)
- Final layer norm: `2sbh/t`
- Output layer projection: `2sbh/t`  
- Cross-entropy loss logits (fp32): `4sbv/t`

Total extra:
```
sbhL/t * (p/L + delta_{p=1} * 4/L * (1 + v/h))
```

Where delta_{p=1} = 1 if p=1, 0 otherwise (the output layer terms only apply to the first pipeline stage, which is the only stage when p=1).

The paper notes these are negligible: "for a model with 22B parameters, these extra terms account for less than 0.01% of the total activation memory requirements."

---

## 7. Full Activation Recomputation

When full activation checkpointing is used (store only layer inputs, recompute everything in backward):

```
Activations memory per layer = 2*sbh   [bytes]
```

This stores only the input to each transformer layer (fp16, 2 bytes per element, shape s*b*h).

Total memory: `2*sbhL` (without any model parallelism).

With tensor parallelism: `2*sbhL/t` (can further partition along sequence dim on each rank).

**The overhead:** Full recomputation adds an extra forward pass per layer. The paper measures 30-40% computational overhead in practice (Table 4 shows 39% for the 22B model).

---

## 8. Selective Activation Recomputation: Equation 6

### 8.1 What is Recomputed vs Stored

**Recomputed (not stored):** The 5as/h term -- specifically:
- QK^T matrix multiply result: 2*a*s^2*b bytes
- Softmax output: 2*a*s^2*b bytes
- Softmax dropout mask: a*s^2*b bytes
- Attention-over-values dropout output: 2*a*s^2*b bytes (the dropout(softmax(QK^T)) matrix)

These are the attention score matrices. They are:
1. **Large** -- they scale as O(s^2) rather than O(s*h)
2. **Cheap to recompute** -- they are element-wise or small matrix ops (softmax, dropout) plus one matmul (QK^T)

**Stored (not recomputed):** Everything else -- the 34*sbh term:
- All MLP activations (19sbh)
- All layer norm inputs (4sbh)  
- Q, K, V matrices (6sbh -- for backprop through QK^T and attention-over-values)
- Linear projection inputs and dropout masks (5sbh)

### 8.2 The Formula

```
Total required memory = 34 * sbhL/t   [bytes]
```

This is Equation (6). It assumes both tensor parallelism and sequence parallelism.

Without sequence parallelism (tensor parallel only):
```
Activations memory per layer = sbh * (10 + 24/t)   [bytes]
```

### 8.3 Memory Savings

For GPT-3 (a=96, s=2048, h=12288):
- 5as/h = 80 vs the constant 34
- Selective recomputation saves 80/(34+80) = **70%** of activation memory

For MT-NLG (a=128, s=2048, h=20480):
- 5as/h = 64 vs the constant 34
- Selective recomputation saves 64/(34+64) = **65%** of activation memory

### 8.4 Compute Overhead

The recomputation cost is small because the recomputed operations have low arithmetic intensity:
- QK^T matmul: 2Bs^2h operations (compared to 6Bsh^2 for QKV linear)
- Softmax: O(as^2b) element-wise ops
- Dropout: O(as^2b) element-wise ops

For GPT-3: overhead = 2.7% of total FLOPs
For MT-NLG: overhead = 1.6% of total FLOPs

These numbers are derived in Appendix A (see Section 10 below).

---

## 9. Summary Table: All Configurations (Table 2)

| Configuration | Activations Memory Per Layer (bytes) |
|---|---|
| No parallelism | `sbh * (34 + 5as/h)` |
| Tensor parallel (baseline) | `sbh * (10 + 24/t + 5as/(ht))` |
| Tensor + sequence parallel | `sbh * (34/t + 5as/(ht))` = `sbh/t * (34 + 5as/h)` |
| Tensor + selective recomputation | `sbh * (10 + 24/t)` |
| Tensor + sequence + selective | `sbh * 34/t` |
| Full activation recomputation | `sbh * 2` |

**For our calculator, the key formula is:**
```
activation_memory_per_layer = sbh/t * (34 + 5as/h)   [with tensor+sequence parallelism, no recomputation]
activation_memory_per_layer = 34 * sbh/t              [with tensor+sequence+selective recomputation]
activation_memory_per_layer = 2 * sbh                 [with full recomputation]
```

**The "5x reduction" claim:** Comparing tensor-parallel baseline (Eq 2) to tensor+sequence+selective (Eq 6):
- For t=8, GPT-3 params: Eq2 = sbh*(10 + 3 + 10) = 23*sbh vs Eq6 = sbh*34/8 = 4.25*sbh
- Ratio: 23/4.25 = 5.4x reduction

---

## 10. FLOPs Formulas (Appendix A)

### 10.1 Model FLOPs (Equation 7)

The theoretical minimum FLOPs for one forward + backward pass:

```
Model FLOPs per iteration = 72 * B * L * s * h^2 * (1 + s/(6h) + v/(12hL))
```

Where B = global batch size (total tokens = B*s).

**Derivation breakdown (forward pass only, backward = 2x forward):**

Attention block:
- QKV linear: 3 * 2Bsh^2 = 6Bsh^2 FLOPs
- QK^T: 2Bs^2h FLOPs  
- Attention over V: 2Bs^2h FLOPs
- Post-attention linear: 2Bsh^2 FLOPs
- Attention total: 8Bsh^2 + 4Bs^2h

MLP block:
- First linear (h->4h): 2Bs*h*4h = 8Bsh^2
- Second linear (4h->h): 2Bs*4h*h = 8Bsh^2
- MLP total: 16Bsh^2

Per-layer total (forward): 24Bsh^2 + 4Bs^2h
All layers (forward): L * (24Bsh^2 + 4Bs^2h) = 24BLsh^2 * (1 + s/(6h))

Output logits layer (forward): 2Bshv

Total forward: 24BLsh^2 + 4BLs^2h + 2Bshv
Forward + backward (3x forward): 72BLsh^2 + 12BLs^2h + 6Bshv
= 72BLsh^2 * (1 + s/(6h) + v/(12hL))

### 10.2 Hardware FLOPs with Selective Recomputation (Equation 8)

Selective recomputation adds an extra forward pass of only the attention score computation (QK^T and attention-over-V):

Extra FLOPs per layer = 2Bs^2h (QK^T) + 2Bs^2h (attention-over-V) = 4Bs^2h

For all layers: 4BLs^2h additional FLOPs.

```
Hardware FLOPs per iteration = 72 * B * L * s * h^2 * (1 + s/(3h) + v/(12hL))
```

Note the s/(3h) instead of s/(6h) -- the attention-score recomputation doubles the s^2 contribution.

### 10.3 HFU/MFU Ratio (Equation 9)

```
Hardware FLOPs / Model FLOPs ≈ 1 + s/(6h)
```

This approximation holds when 3h >> s and 12hL >> v.

For GPT-3 (s=2048, h=12288): ratio ≈ 1 + 2048/73728 ≈ 1.028 (2.8% overhead)
For a model with s=8192, h=4096: ratio ≈ 1 + 8192/24576 ≈ 1.33 (33% overhead)

**For our calculator:** This ratio tells us how to convert between MFU and HFU when selective recomputation is used. HFU = MFU * (hardware FLOPs / model FLOPs).

---

## 11. Per-Layer Timing Data (Table 4)

Measured on the 22B model (a=64, h=6144, L=48, t=8, s=2048), single transformer layer:

| Experiment | Forward (ms) | Backward (ms) | Combined (ms) | Overhead vs Baseline |
|-----------|------------|-------------|------------|----------|
| Baseline (no recompute) | 7.7 | 11.9 | 19.6 | -- |
| Sequence Parallelism only | 7.2 | 11.8 | 19.0 | -3% (speedup) |
| Full recompute (baseline) | 7.7 | 19.5 | 27.2 | +39% |
| Selective Recompute only | 7.7 | 13.2 | 20.9 | +7% |
| Selective + Sequence | 7.2 | 13.1 | 20.3 | +4% |

**Key observations:**
1. Sequence parallelism provides a 6% forward speedup (7.7 -> 7.2ms) because layer-norm and dropout operate on 1/t of the data
2. Full recomputation adds 7.6ms to backward (64% backward overhead), 39% total overhead
3. Selective recomputation adds only 1.3ms to backward (11% backward overhead), 7% total overhead
4. Combined selective + sequence: only 4% total overhead with nearly all the memory savings
5. The 39% overhead for full recompute (vs expected 50% for double forward) is because of an optimization where all-reduce communication overlaps with the linear weight gradient computation in backward

---

## 12. End-to-End Results (Table 5)

| Model Size | Full Recompute (sec) | Present Work (sec) | Throughput Increase | Model FLOPs Util | Hardware FLOPs Util |
|-----------|-------------------|-----------------|------------------|-----------------|-------------------|
| 22B | 1.42 | 1.10 | 29.0% | 41.5% | 43.7% |
| 175B (GPT-3) | 18.13 | 13.75 | 31.8% | 51.4% | 52.8% |
| 530B (MT-NLG) | 49.05 | 37.83 | 29.7% | 56.0% | 57.0% |
| 1T | 94.42 | 71.49 | 32.1% | 56.3% | 57.0% |

**Key observations:**
- "Present work" = sequence parallelism + selective activation recomputation
- Throughput increase is consistently 29-32% across all model sizes
- MFU increases from ~42% to ~56% at the 530B/1T scale
- HFU is slightly higher than MFU because selective recomputation does extra compute
- The HFU/MFU gap is small (1-2%) because the recomputed operations are cheap
- A100 peak = 312 TFLOPS/sec (footnote 5)
- For 530B with 8-way data parallelism (2240 GPUs total), MFU drops slightly from 56.0% to 54.2%

---

## 13. Model Configurations Used (Table 3)

| Model | Attn Heads | Hidden Size | Layers | TP | PP | GPUs | Global Batch | Micro Batch |
|-------|-----------|------------|--------|---|---|------|-------------|------------|
| 22B | 64 | 6144 | 48 | 8 | 1 | 8 | 4 | 4 |
| 175B (GPT-3) | 96 | 12288 | 96 | 8 | 8 | 64 | 64 | 1 |
| 530B (MT-NLG) | 128 | 20480 | 105 | 8 | 35 | 280 | 280 | 1 |
| 1T | 160 | 25600 | 128 | 8 | 64 | 512 | 512 | 1 |

All models use:
- s = 2048
- v = 51200
- Mixed precision on A100 80GB GPUs
- Interleaving stages m=3 for 175B and 530B
- No data parallelism in Table 5 results

---

## 14. Pipeline Parallelism Memory Optimization (Appendix B)

The paper shows (Figure 9) that memory is non-uniform across pipeline ranks for the 530B model (p=35 stages):
- Rank 0: ~13 GB activations (without dealloc), ~10.5 GB (with dealloc)
- Rank 34: ~1.5 GB activations
- The deallocation optimization saves sbhr bytes per rank (r = microbatches in flight)

---

## 15. Microbatch-Level Activation Recomputation (Appendix C)

A hybrid approach for pipeline-parallel training:
- Store ALL activations for some microbatches (no recomputation needed)
- Checkpoint (store only layer inputs) for remaining microbatches
- As backpropagation frees memory from completed microbatches, later microbatches can store full activations

The number of outstanding microbatch backpropagation steps at pipeline stage S is: `max(0, p - S)`.

Improvement is modest: +0.7% MFU for 175B, +0.4% MFU for 530B (because selective recomputation overhead is already only ~2%).

---

## 16. Flash Attention Interaction

**The paper does not discuss Flash Attention.** However, the interaction is critical for our calculator:

Flash Attention (Dao et al., 2022) fuses the QK^T, softmax, dropout, and attention-over-V operations into a single kernel that does NOT materialize the full attention matrix in HBM. This eliminates the need to store:
- QK^T result: 2*a*s^2*b bytes
- Softmax output: 2*a*s^2*b bytes  
- Softmax dropout mask: a*s^2*b bytes
- Attention dropout output: 2*a*s^2*b bytes (used in attention-over-V backprop -- Flash Attention recomputes this in backward)

This is exactly the `5*a*s^2*b` = `sbh * 5as/h` term.

**With Flash Attention, the formula simplifies to:**
```
Activations memory per layer = 34 * sbh   [no parallelism, with Flash Attention]
Activations memory per layer = 34 * sbh/t  [with tensor+sequence parallelism + Flash Attention]
```

This is identical to the selective recomputation formula (Equation 6), because Flash Attention achieves the same effect -- it avoids storing the attention score matrices. The difference is that Flash Attention does this via kernel fusion rather than explicit checkpointing + recomputation, and it is generally faster than even storing the activations (due to reduced HBM traffic).

**For our calculator:** If the user selects Flash Attention, use the `34*sbh` formula (drop the `5as/h` term). If they select selective recomputation without Flash Attention, also use `34*sbh`. These stack: if both are enabled, the memory savings is the same as either alone (they address the same term). The compute overhead of selective recomputation becomes zero with Flash Attention since Flash Attention's backward pass inherently recomputes these values.

---

## 17. The d_ff Correction

**The paper assumes d_ff = 4h throughout.** The generalized formulas when d_ff != 4h are:

### 17.1 MLP Component Generalized

From Section 3.2.2 above:
```
MLP activations = sb * (2h + 2*d_ff + 2*d_ff + h) = sb * (3h + 4*d_ff)
```

Breaking down:
- First linear input: 2*sbh (fp16, h-wide)
- GeLU input: 2*sb*d_ff (fp16, d_ff-wide)  
- Second linear input: 2*sb*d_ff (fp16, d_ff-wide)
- Dropout mask: sb*h (1 byte, h-wide, after the MLP output projection)

Wait -- let me reconsider. The dropout mask is after the full MLP, so it operates on the h-dimensional output. And the paper counts it as sbh (1 byte per element, sbh elements).

So:
```
MLP = 2*sbh + 2*sb*d_ff + 2*sb*d_ff + sbh = 3*sbh + 4*sb*d_ff
```

With d_ff = 4h: 3sbh + 16sbh = 19sbh. Correct.

### 17.2 Generalized Full Formula

```
Activations per layer = sb * (11h + 5*a*s + 3h + 4*d_ff + 4h)
                       = sb * (18h + 4*d_ff + 5*a*s)
```

Wait, let me recount carefully:

- Attention: 11*sbh + 5*a*s^2*b
- MLP: 3*sbh + 4*sb*d_ff
- LayerNorm: 4*sbh
- Total: (11 + 3 + 4)*sbh + 4*sb*d_ff + 5*a*s^2*b = 18*sbh + 4*sb*d_ff + 5*a*s^2*b

With d_ff = 4h: 18sbh + 16sbh + 5as^2b = 34sbh + 5as^2b. Correct.

**Generalized formula (for our calculator):**
```
Activations per layer = sb * (18*h + 4*d_ff + 5*a*s)   [bytes, no parallelism]
```

Or equivalently:
```
Activations per layer = sbh * (18 + 4*d_ff/h + 5*a*s/h)   [bytes]
```

**For the standard d_ff = 4h case:** `sbh * (18 + 16 + 5as/h) = sbh * (34 + 5as/h)`.

With selective recomputation (or Flash Attention), drop the 5as term:
```
Activations per layer = sb * (18*h + 4*d_ff)   [bytes, no parallelism, selective recompute or Flash Attention]
```

With d_ff = 4h: sb * (18h + 16h) = 34*sbh. Correct.

### 17.3 Impact of GLU Variants (SwiGLU, GeGLU)

The paper does not address GLU variants. However, for models using SwiGLU (like LLaMA), the MLP structure changes from:
```
Standard: Linear(h -> d_ff) -> GeLU -> Linear(d_ff -> h)
SwiGLU:   Linear(h -> d_ff) * SiLU(Linear(h -> d_ff)) -> Linear(d_ff -> h)
```

For SwiGLU with intermediate size d_ff (each of the two gate projections has output dim d_ff):
- Gate projection input: 2*sbh (shared input)
- Up projection input: 2*sbh (same shared input, but may or may not be stored separately)
- Gate projection output: 2*sb*d_ff (for SiLU backprop)
- Up projection output: 2*sb*d_ff (for element-wise multiply backprop)
- SiLU input: 2*sb*d_ff (for SiLU derivative -- like GeLU, needs input)
- Element-wise product result: 2*sb*d_ff (input to down projection)
- Down projection input: 2*sb*d_ff (same as above, already counted)
- Dropout mask: sbh

This is an area our calculator should handle but is NOT covered by this paper. The paper's formulas assume the standard FFN structure.

---

## 18. What's Unique / Non-Obvious

1. **The 10sbh replication problem:** Without sequence parallelism, 10 out of 34 units of activation memory are replicated across ALL tensor-parallel ranks. This means tensor parallelism alone has diminishing returns for activation memory -- increasing t from 8 to 16 barely helps because the 10sbh term dominates.

2. **Zero communication overhead for sequence parallelism:** Because ring all-reduce = reduce-scatter + all-gather, replacing all-reduces with the sequence-parallel pattern of reduce-scatters and all-gathers uses the same bandwidth. This is not obvious and is a key engineering insight.

3. **Selective recomputation is better than layer-level granularity:** The standard approach of checkpointing some layers fully and storing others fully is coarse-grained and doesn't work well when there are few layers per device (e.g., MT-NLG has only 3 layers per device with 35-way PP). Selective recomputation operates within each layer.

4. **The HFU/MFU distinction matters for benchmarking:** When reporting training throughput, one must distinguish between model FLOPs (theoretical minimum) and hardware FLOPs (what the GPU actually computes). The ratio is approximately 1 + s/(6h) with selective recomputation. For long-context training, this can be significant.

5. **Pipeline parallelism memory is worst on rank 0:** The first pipeline stage stores p * (L/p) = L layers of activations because it must keep all in-flight microbatches. This is NOT reduced by pipeline parallel size p. Our calculator should account for this when estimating peak memory.

6. **The interleaving correction factor (1 + (p-1)/(pm)):** This multiplicative factor on activation memory is often overlooked. For p=35, m=3: factor = 1 + 34/105 = 1.324, meaning 32.4% more activation memory than the naive formula suggests.

7. **GeLU requires storing its input, ReLU does not:** The 8sbh term for GeLU input storage is because GeLU's derivative depends on the input value. If ReLU is used instead, only a 1-bit mask per element is needed (sb*4h bits = sb*h/2 bytes), saving roughly 7.5*sbh per layer. Our calculator should adjust for activation function choice, though GeLU/SiLU are standard in modern LLMs.

8. **The formula ignores residual connection storage:** The residual (skip) connections add the output of attention/MLP to the input. The paper's formula accounts for the inputs to layer norms (which are the residual stream values) but does not separately account for the "add" operation since the inputs are already stored.

---

## 19. Formulas for Our Calculator -- Summary

### 19.1 Per-Layer Activation Memory (bytes)

**General form (with tensor+sequence parallelism):**
```python
def activation_memory_per_layer(s, b, h, a, t, d_ff, recomputation="none", flash_attention=False):
    """
    Returns activation memory in bytes for one transformer layer.
    
    Args:
        s: sequence length
        b: microbatch size
        h: hidden dimension
        a: number of attention heads
        t: tensor parallel size (1 if no TP)
        d_ff: feedforward intermediate dimension (typically 4*h)
        recomputation: "none", "selective", or "full"
        flash_attention: whether Flash Attention is used
    """
    if recomputation == "full":
        return 2 * s * b * h  # Only store layer input (fp16)
    
    # Base memory (everything except attention scores)
    # Attention: 11*sbh (with d_ff=4h standard attention components)
    # MLP: 3*sbh + 4*sb*d_ff  
    # LayerNorm: 4*sbh
    # With sequence parallelism, everything divides by t
    base = s * b * (18 * h + 4 * d_ff) / t
    
    # Attention score memory (the 5as^2b term)
    if recomputation == "selective" or flash_attention:
        attn_scores = 0  # These are recomputed or fused
    else:
        attn_scores = 5 * a * s * s * b / t
    
    return base + attn_scores
```

### 19.2 Total Activation Memory (bytes)

```python
def total_activation_memory(s, b, h, a, L, t, p, d_ff, 
                            recomputation="none", flash_attention=False,
                            interleaving_stages=None):
    """
    Total activation memory for the model (worst-case, first pipeline stage).
    """
    per_layer = activation_memory_per_layer(s, b, h, a, t, d_ff, 
                                            recomputation, flash_attention)
    
    # First pipeline stage stores L layers worth (not L/p)
    total = per_layer * L
    
    # Interleaving correction
    if interleaving_stages is not None and interleaving_stages > 1:
        m = interleaving_stages
        total *= (1 + (p - 1) / (p * m))
    
    return total
```

### 19.3 FLOPs Formulas

```python
def model_flops(B, L, s, h, v):
    """Model FLOPs for one iteration (forward + backward)."""
    return 72 * B * L * s * h**2 * (1 + s/(6*h) + v/(12*h*L))

def hardware_flops_selective(B, L, s, h, v):
    """Hardware FLOPs with selective activation recomputation."""
    return 72 * B * L * s * h**2 * (1 + s/(3*h) + v/(12*h*L))

def hfu_mfu_ratio(s, h):
    """Approximate ratio of hardware to model FLOPs with selective recompute."""
    return 1 + s / (6 * h)
```

---

## 20. Gaps and Limitations

1. **No GQA/MQA support:** The paper assumes standard multi-head attention (MHA). For Grouped Query Attention (GQA) or Multi-Query Attention (MQA), the K and V storage in the attention block changes. With GQA using g groups: K and V storage becomes 2*sb*h*(g/a) each instead of 2*sbh each. The QK^T and softmax terms remain unchanged.

2. **No SwiGLU/GLU variant support:** As discussed in Section 17.3.

3. **No expert parallelism / MoE:** The paper only covers dense transformers.

4. **Assumes d_ff = 4h:** The generalized formula is in Section 17 but the paper itself never presents it.

5. **No LoRA/adapter consideration:** All formulas assume full fine-tuning.

6. **No context parallelism / ring attention:** The paper predates these techniques.

7. **The formula is approximate:** It ignores small buffers (layer norm mean/variance at 2*sb each), residual addition intermediates, and embedding activations. The paper acknowledges this approximation is accurate to within 0.01%.

8. **A100-specific measurements:** All timing data is on A100 80GB. The relative overhead percentages should generalize, but absolute times will differ on other hardware.
