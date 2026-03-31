# Deep Dive: Megatron-LM -- Training Multi-Billion Parameter Language Models Using Model Parallelism

**Paper**: Shoeybi et al., 2019 (arXiv:1909.08053v4, March 2020 revision)
**Authors**: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (NVIDIA)
**Source URL**: https://arxiv.org/abs/1909.08053
**Date reviewed**: 2026-03-31

---

## 1. Paper Summary

This is the foundational paper for **intra-layer tensor parallelism** in transformers. It introduces a method to split individual weight matrices of transformer layers across multiple GPUs using only a few all-reduce communication primitives inserted into standard PyTorch code -- no custom compiler or library required. The approach is orthogonal to pipeline parallelism.

Key results: trained up to 8.3B parameter GPT-2 and 3.9B parameter BERT models on 512 V100 GPUs. Achieved 15.1 PetaFLOPs sustained with 76% weak scaling efficiency relative to a 39 TeraFLOPs single-GPU baseline (which itself was 30% of V100 peak).

---

## 2. Tensor Parallelism Mechanism -- Exact Splitting

### 2.1 MLP Block (Feed-Forward Network)

The MLP consists of two linear layers with a GeLU activation in between:

```
Y = GeLU(X A)       ... Equation (1) in paper
Z = Dropout(Y B)
```

Where `X` is the input (shape: `[b*s, d]`), `A` is the first weight matrix (shape: `[d, d_ff]`), `B` is the second weight matrix (shape: `[d_ff, d]`).

**Key insight -- why column-parallel first, row-parallel second:**

**Option A (split A by rows, X by columns):** Would produce `GeLU(X_1 A_1 + X_2 A_2)`, but since GeLU is nonlinear, `GeLU(a+b) != GeLU(a) + GeLU(b)`. This **requires a synchronization point** (all-reduce) before the GeLU. Bad.

**Option B (split A by columns):** Each GPU computes `GeLU(X A_i)` independently since `[Y_1, Y_2] = [GeLU(X A_1), GeLU(X A_2)]` (Equation 3 in paper). The nonlinearity can be applied independently to each partition. **No synchronization needed** before GeLU. Good.

So the final MLP parallelism scheme is:

1. **First GEMM (column-parallel)**: Split `A = [A_1, A_2, ..., A_{N_tp}]` along columns. Each GPU `i` holds `A_i` of shape `[d, d_ff/N_tp]`. Input `X` is replicated (via the `f` identity operator). Each GPU computes `Y_i = GeLU(X A_i)` locally, producing output of shape `[b*s, d_ff/N_tp]`.

2. **Second GEMM (row-parallel)**: Split `B` along rows: `B = [B_1; B_2; ...; B_{N_tp}]`. Each GPU `i` holds `B_i` of shape `[d_ff/N_tp, d]`. Each GPU computes `Z_i = Y_i B_i` of shape `[b*s, d]`. Then an **all-reduce** (the `g` operator) sums `Z = sum(Z_i)` across GPUs before dropout.

**Communication: exactly 1 all-reduce in forward pass for the entire MLP block.**

### 2.2 Self-Attention Block

The self-attention block has Q, K, V projection matrices and an output projection matrix.

**Key insight**: Multi-head attention is *inherently parallel* across heads. Each head's Q/K/V projections and attention computation are independent.

1. **Q, K, V projections (column-parallel)**: The weight matrices `W_Q, W_K, W_V` (each of shape `[d, d]`) are split column-wise. With `N_tp` GPUs, each GPU gets `a/N_tp` attention heads worth of Q, K, V. Each GPU computes attention locally for its subset of heads -- no communication needed for the Q/K/V GEMMs or the attention computation itself.

2. **Output projection (row-parallel)**: The output projection `W_O` (shape `[d, d]`) is split row-wise. Each GPU holds `W_O_i` of shape `[d/N_tp, d]`. Each GPU computes its partial output, then an **all-reduce** sums results before dropout.

**Communication: exactly 1 all-reduce in forward pass for the entire attention block.**

### 2.3 The f and g Operators

The paper defines two conjugate communication operators:

**`f` operator** (Code 1 in paper):
```python
class f(torch.autograd.Function):
    def forward(ctx, x):
        return x                    # identity in forward
    def backward(ctx, gradient):
        all_reduce(gradient)        # all-reduce in backward
        return gradient
```

**`g` operator** (conjugate of `f`):
```python
class g(torch.autograd.Function):
    def forward(ctx, x):
        all_reduce(x)              # all-reduce in forward
        return x
    def backward(ctx, gradient):
        return gradient             # identity in backward
```

These are placed at the boundaries of each parallel region:
- `f` at the **input** of each parallel block (identity forward, all-reduce backward)
- `g` at the **output** of each parallel block (all-reduce forward, identity backward)

### 2.4 Total Communication per Transformer Layer

From Figure 4 of the paper:

> "There are 4 total communication operations in the forward and backward pass of a single model parallel transformer layer."

Specifically:
- **Forward pass**: 2 all-reduces (1 from MLP `g` operator, 1 from attention `g` operator)
- **Backward pass**: 2 all-reduces (1 from MLP `f` operator backward, 1 from attention `f` operator backward)

**Total per layer per training step: 4 all-reduce operations.**

Each all-reduce transfers tensors of shape `[b, s, d]` (the hidden-state activations), so each all-reduce moves `b * s * d * beta` bytes per GPU in a ring all-reduce implementation (with the standard `2 * (N_tp - 1) / N_tp` bandwidth cost factor).

```
Comm_tp_per_layer = 4 x (2 x (N_tp - 1) / N_tp) x b x s x d x beta  bytes
```

For the full model with pipeline parallelism:
```
Comm_tp_total = (L / N_pp) x Comm_tp_per_layer
```

---

## 3. Embedding and Output Layer Parallelism

### 3.1 Input Embedding

The input embedding matrix `E` has shape `[V, d]` (vocabulary size by hidden dimension). It is parallelized along the vocabulary dimension (column-wise):

```
E = [E_1, E_2, ..., E_{N_tp}]   (each E_i has shape [V/N_tp, d])
```

Since each GPU's partition only contains a subset of the vocabulary, an **all-reduce (`g` operator)** is required after the input embedding lookup to combine partial results.

### 3.2 Output Embedding (Logit Projection)

The output embedding (language model head) shares weights with the input embedding in GPT-2 and BERT. Since the embedding is column-partitioned, each GPU computes partial logits for its vocabulary slice:

```
[Y_1, Y_2] = [X E_1, X E_2]   (parallel GEMM, Equation in Section 3)
```

**Naive approach**: All-gather to reconstruct full logits `Y = all-gather([Y_1, Y_2])`, then compute cross-entropy loss. This communicates `b x s x V` elements -- **huge** because V is large (50K-130K+).

**Megatron optimization**: Fuse the cross-entropy loss computation with the parallel logit computation. Each GPU computes a partial cross-entropy on its vocabulary slice, then only scalar losses are communicated. This reduces communication from `b x s x V` elements to `b x s` scalar values -- a massive reduction.

> "Communicating scalar losses instead of logits is a huge reduction in communication that improves the efficiency of our model parallel approach."

---

## 4. Vocabulary Padding for TP

From Section 5.1 of the paper:

> "The original vocabulary size was 50,257, however, to have efficient GEMMs for the logit layer, it is beneficial for the per-GPU vocabulary size to be a multiple of 128. Since we study up to 8-way model parallelism, we pad the vocabulary such that it is divisible by 128 x 8 = 1024, resulting in a padded vocabulary size of 51,200."

**Formula**:
```
V_padded = ceil(V / (128 x N_tp)) x (128 x N_tp)
```

This ensures that each GPU's vocabulary slice `V_padded / N_tp` is a multiple of 128, which is required for efficient Tensor Core utilization on NVIDIA GPUs (Tensor Cores operate on tiles of 8x8 or 16x16, and 128-alignment ensures good tile occupancy for the logit GEMM).

Example: V=50,257 with N_tp=8 produces V_padded = ceil(50257/1024) x 1024 = 50 x 1024 = 51,200.

---

## 5. Memory Reduction from TP

The paper does not provide explicit per-GPU memory formulas, but the mechanism implies:

### 5.1 Weight Memory

Each transformer layer's weight matrices are split across N_tp GPUs:
- **Attention Q/K/V projections**: Each of shape `[d, d]` split column-wise. Per-GPU: `3 x d x d/N_tp` parameters.
- **Attention output projection**: Shape `[d, d]` split row-wise. Per-GPU: `d x d/N_tp` parameters (equivalently `d/N_tp x d`).
- **MLP first linear**: Shape `[d, d_ff]` split column-wise. Per-GPU: `d x d_ff/N_tp` parameters.
- **MLP second linear**: Shape `[d_ff, d]` split row-wise. Per-GPU: `d_ff/N_tp x d` parameters.
- **LayerNorm parameters**: **Duplicated** on every GPU (not split). Each LayerNorm has `2d` parameters (scale + bias). There are 2 LayerNorms per layer.
- **Embedding**: Split along vocabulary dimension. Per-GPU: `V_padded/N_tp x d` parameters.

Per-layer parameter count per GPU:
```
Psi_layer_per_gpu = (4 x d^2 + 2 x d x d_ff) / N_tp + 4d   (LayerNorm duplicated)
```

For typical models where d_ff = 4d:
```
Psi_layer_per_gpu = (4d^2 + 8d^2) / N_tp + 4d = 12d^2 / N_tp + 4d
```

The `4d` LayerNorm overhead is negligible compared to `12d^2/N_tp` for large d, so:
```
Psi_layer_per_gpu ≈ Psi_layer / N_tp   (approximately)
```

### 5.2 Optimizer State Memory

Optimizer states (Adam momentum and variance) correspond 1:1 to parameters, so they are also reduced by N_tp:
```
M_optimizer_per_gpu ≈ M_optimizer / N_tp
```

### 5.3 Activation Memory

The paper does not detail activation memory formulas, but later work (Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models") provides the detailed breakdown showing that some activation tensors within TP-parallel regions are naturally sharded by 1/N_tp while others (LayerNorm outputs, dropout masks on residual connections) remain full-size on each GPU. Sequence parallelism (introduced in that follow-up work) addresses the unsharded activations.

### 5.4 What is Duplicated (Not Split)

The paper explicitly notes these are duplicated across all TP ranks:
- LayerNorm parameters (scale, bias)
- Dropout on residual connections (outside model-parallel regions)
- Residual connections themselves (computed identically by each GPU using the all-reduced result)

> "Rather than having one GPU compute part of the dropout, layer normalization, or residual connections and broadcast the results to other GPUs, we choose to duplicate the computation across GPUs."

This is a deliberate design choice: duplicating cheap element-wise operations avoids extra communication.

---

## 6. Attention Head Constraints

From the attention splitting mechanism:

**N_tp must evenly divide the number of attention heads `a`** because each GPU gets `a/N_tp` complete attention heads. The Q, K, V weight columns corresponding to each head must stay together on the same GPU (since per-head attention computation is done locally).

From Table 1, the paper keeps `hidden_size_per_attention_head = d/a = 96` constant and scales heads:

| Hidden Size | Attention Heads | Params | Model Parallel GPUs |
|-------------|----------------|--------|---------------------|
| 1536 | 16 | 1.2B | 1 |
| 1920 | 20 | 2.5B | 2 |
| 2304 | 24 | 4.2B | 4 |
| 3072 | 32 | 8.3B | 8 |

Note: 16 heads / 1 GPU = 16 heads per GPU; 20 heads / 2 GPUs = 10 heads per GPU; 24 heads / 4 GPUs = 6 heads per GPU; 32 heads / 8 GPUs = 4 heads per GPU.

**The paper does not discuss GQA/MQA** (grouped/multi-query attention), as these were not yet common. For GQA, the constraint is that N_tp must divide the number of KV heads `a_kv`, which is the tighter constraint.

---

## 7. FFN Splitting Details

For the FFN with inner dimension `d_ff`:

- First linear `A`: shape `[d, d_ff]`, split column-wise into N_tp chunks of `[d, d_ff/N_tp]` each.
- Second linear `B`: shape `[d_ff, d]`, split row-wise into N_tp chunks of `[d_ff/N_tp, d]` each.

**Constraint**: `N_tp must divide d_ff evenly`. For standard architectures where `d_ff = 4 x d`, this is automatically satisfied when N_tp divides d. For SwiGLU architectures (e.g., LLaMA) where `d_ff` can be a non-standard value like `11008`, this becomes an additional binding constraint.

The paper does not discuss SwiGLU (it predates LLaMA), but the same column-parallel / row-parallel pattern applies to the gate projection in gated FFN variants:
- Gate projection `W_gate`: split column-wise (same as first linear)
- Up projection `W_up`: split column-wise
- Down projection `W_down`: split row-wise

---

## 8. Scaling and Efficiency Results

### 8.1 Single GPU Baseline

- 1.2B parameter model on 1 NVIDIA V100 32GB GPU
- Sustains **39 TeraFLOPs** (30% of theoretical peak FLOPS for V100)
- This is the baseline for all scaling measurements

### 8.2 Weak Scaling Efficiency (Model Parallel Only)

From Figure 5:
| GPUs | Weak Scaling |
|------|-------------|
| 1 | 100% (baseline) |
| 2 | 95% |
| 4 | 82% (implied from figure) |
| 8 | 77% |

### 8.3 Weak Scaling Efficiency (Model + Data Parallel)

From Figure 5:
| GPUs | Weak Scaling |
|------|-------------|
| 8 | 90% (implied) |
| 64 | 83% |
| 128 | 81% (implied) |
| 256 | 79% |
| 512 | 74% |

### 8.4 Peak Throughput

- 8.3B parameters on 512 GPUs (8-way TP x 64-way DP)
- **15.1 PetaFLOPs sustained** across entire application
- 76% scaling efficiency vs single-GPU baseline

### 8.5 Effect of Attention Heads on Scaling

From Appendix D, Table 7 (8.3B model with 8-way TP):

| Attention Heads | Hidden Size per Head | Scaling Efficiency |
|-----------------|---------------------|-------------------|
| 16 | 192 | 82% |
| 24 | 128 | 80% |
| 32 | 96 | 77% |

More heads = more but smaller GEMMs = worse scaling efficiency. The attention softmax dimension also grows with more heads (more elements in the softmax), adding overhead.

### 8.6 Strong Scaling (1.2B Model)

From Appendix D, Table 8 (fixed model size, fixed batch size of 8):

| GPUs | Speedup |
|------|---------|
| 1 | 1.0x |
| 2 | 1.64x |
| 4 | 2.34x |
| 8 | 2.98x |

Strong scaling efficiency drops significantly because the per-GPU computation shrinks while communication volume stays roughly constant (the `b x s x d` all-reduce size does not decrease with more TP GPUs).

---

## 9. Communication Overhead Analysis

### 9.1 Latency vs Bandwidth

The paper does not provide an explicit analytical formula separating latency and bandwidth components, but the empirical results reveal the regime:

- **Intra-node (NVLink)**: 300 GB/s bandwidth between GPUs within a DGX-2H server. TP all-reduces operate in the **bandwidth-limited regime** for large hidden sizes.
- **Inter-node (InfiniBand)**: 100 GB/s interconnect between servers (8 InfiniBand adapters per server). Data-parallel all-reduces operate here.

The strong scaling results (Table 8) show diminishing returns: 2 GPUs give 1.64x (82% efficiency) but 8 GPUs only give 2.98x (37% efficiency). This indicates that at TP=8 for a model that fits on 1 GPU, the communication overhead is very significant -- roughly 63% of the time is spent on communication, not computation.

For models that *need* TP (do not fit on 1 GPU), this overhead is the price of admission.

### 9.2 Why TP Should Stay Within a Node

The paper uses NVLink exclusively for TP communication. At 300 GB/s bidirectional bandwidth (NVLink on V100 DGX-2H), the all-reduce of a `[b, s, d]` tensor takes:

```
T_allreduce = 2 x (N_tp - 1) / N_tp x (b x s x d x beta) / BW_nvlink
```

For b=8, s=1024, d=3072, beta=2 (fp16), N_tp=8:
- Tensor size: 8 x 1024 x 3072 x 2 = ~48 MB
- Ring all-reduce volume: 2 x 7/8 x 48 MB = ~84 MB
- Time at 300 GB/s: ~0.28 ms per all-reduce
- 4 all-reduces per layer: ~1.12 ms per layer in communication

If this were done over InfiniBand at 100 GB/s instead, it would be 3x slower: ~3.36 ms per layer, which would be catastrophic for training throughput.

---

## 10. Hybrid Model and Data Parallelism

From Appendix B.1:

- Model parallel groups and data parallel groups are orthogonal
- GPUs within the same server form model parallel groups (e.g., GPUs 1-8 = model parallel group 1)
- GPUs at the same position across different model parallel groups form data parallel groups (e.g., GPU 1, GPU 9, ..., GPU 505 = data parallel group 1)

```
Total GPUs = N_tp x N_dp
```

For the 8.3B model: 8 (TP) x 64 (DP) = 512 GPUs.

During backpropagation:
- Weight gradient all-reduces happen within each data parallel group
- Model parallel all-reduces happen within each model parallel group
- These are independent and can potentially overlap

### 10.1 Random Number Generation

A subtle but important implementation detail:
- **Residual connection dropout**: Seed is synchronized across all model parallel workers (same seed) to ensure identical dropout masks, since the residual is computed identically by all TP ranks.
- **Dropout within model parallel regions** (e.g., attention dropout): Uses a *separate* random number generator uniquely seeded per model parallel worker, so each GPU applies different dropout to its own shard.

---

## 11. Training Configuration Details

### 11.1 Model Configurations (GPT-2)

From Table 2:

| Params | Layers | Hidden | Attn Heads | Hidden/Head | GPUs | Time/Epoch |
|--------|--------|--------|-----------|-------------|------|------------|
| 355M | 24 | 1024 | 16 | 64 | 64 | 0.86 days |
| 2.5B | 54 | 1920 | 20 | 96 | 128 | 2.27 days |
| 8.3B | 72 | 3072 | 24 | 128 | 512 | 2.10 days |

Note: The 8.3B model uses 24 attention heads (not 32 as in Table 1's scaling study). The 2.5B model has the same head count per GPU: 20/2=10 heads per TP rank (but it's listed as using 128 GPUs total, meaning TP + DP).

### 11.2 Training Setup
- Sequence length: 1024 (all GPT-2 models)
- Batch size: 512 (global)
- Iterations: 300K
- Mixed precision training (fp16) with dynamic loss scaling
- Activation checkpointing after every transformer layer
- Adam optimizer with weight decay lambda=0.01
- Global gradient norm clipping at 1.0
- Dropout: 0.1 everywhere
- Weight initialization: `W ~ N(0, 0.02)`, residual layer weights scaled by `1/sqrt(2*N)` where N is the number of transformer layers

### 11.3 BERT Configurations

From Table 4:

| Params | Layers | Hidden | Attn Heads | GPUs |
|--------|--------|--------|-----------|------|
| 336M | 24 | 1024 | 16 | 128 |
| 1.3B | 24 | 2048 | 32 | 256 |
| 3.9B | 48 | 2560 | 40 | 512 |

---

## 12. BERT Layer Normalization Placement

A significant finding for the calculator's accuracy model (not memory, but architecture correctness):

The original BERT architecture (Figure 7a) places LayerNorm after the residual addition:
```
x = x + Attention(LayerNorm(x))   # Post-LN: LayerNorm inside residual
```

Megatron found this causes training instability for models larger than BERT-Large (336M). They propose Pre-LN (Figure 7b):
```
x = x + Attention(LayerNorm(x))   # Pre-LN: LayerNorm before attention/MLP
```

With Pre-LN, the 752M model trains stably. The original architecture fails at 752M. This is relevant to the calculator because **Pre-LN vs Post-LN does not change the parameter count or memory** -- LayerNorm has the same number of parameters either way -- but it affects whether training will actually converge at scale.

---

## 13. What Is Unique / Non-Obvious

### 13.1 Fused Cross-Entropy with Parallel Logits
The optimization of computing cross-entropy loss in a distributed manner (each TP rank handles its vocabulary slice, only scalar losses are communicated) is a huge communication saving that most calculators do not model. The communication reduction is from `O(b x s x V)` to `O(b x s)` -- a factor of V reduction (50K-130K+).

### 13.2 Duplicated Computation vs Communication Trade-off
The deliberate choice to duplicate LayerNorm and dropout computations rather than partitioning them is a design principle: cheap element-wise operations are duplicated to avoid all-reduce overhead. This means the `4d` LayerNorm parameters per layer are stored on every TP rank -- a negligible memory overhead for large d, but a real one for small models.

### 13.3 Communication is on Hidden Dimension, Not Model Size
A key insight: TP communication volume is `O(b x s x d)` per all-reduce, independent of total model size (number of layers). Adding more layers only adds more all-reduces (linear scaling), but each all-reduce's size depends only on `b x s x d`. This means TP communication overhead per step scales linearly with the number of layers assigned to the GPU (`L/N_pp`).

### 13.4 Strong Scaling is Inefficient
The strong scaling results (Table 8) show only 2.98x speedup with 8 GPUs for a model that fits on 1 GPU. This is important for the calculator's throughput estimation: TP helps with memory but hurts throughput per GPU. The calculator should model this degradation.

### 13.5 Head Count Affects TP Efficiency
Table 7 shows 82% efficiency with 16 heads vs 77% with 32 heads (same model, same TP degree). More heads mean smaller per-head GEMMs, which have worse compute efficiency (lower arithmetic intensity). The calculator could optionally warn if `heads_per_gpu = a / N_tp` is very small (e.g., < 4).

### 13.6 Vocabulary Padding is Multiplicative
The padding formula `V_padded = ceil(V / (128 x N_tp)) x (128 x N_tp)` can add significant parameters for large N_tp and awkward V values. For V=32000 with N_tp=8, the padding is: ceil(32000/1024) x 1024 = 32 x 1024 = 32768 (+768 entries = +768*d parameters in the embedding).

---

## 14. Relevance to Our Calculator Spec

### 14.1 Already Covered in Spec

Our spec (Section 5.6) already captures the following correctly:
- 4 all-reduce operations per layer per training step (2 forward + 2 backward)
- Communication volume formula: `Comm_tp_per_layer = 4 x 2 x (N_tp - 1) / N_tp x b x s x d x beta`
- TP backward all-gather buffer estimate
- Constraint that N_tp divides attention heads and KV heads
- Vocabulary padding formula `ceil(V / (128 x N_tp)) x (128 x N_tp)`
- TP communication to ZeRO-3 communication ratio
- Constraint that N_tp should be <= 8 (GPUs per node)
- Constraint that N_tp divides d_ff

### 14.2 Potential Additions or Refinements

1. **Duplicated LayerNorm parameters**: The spec says `M_params_per_gpu ≈ Psi_params / N_tp` with the note "not all layers split perfectly." It could be more precise by noting that LayerNorm params (`4d` per layer for 2 LayerNorms with scale+bias) are duplicated. For a model with `L` layers: `4dL` parameters are duplicated, meaning the exact per-GPU parameter count is `(Psi_total - L*4d) / N_tp + L*4d`. For large models this is negligible (4d is ~16K for d=4096, vs ~12M per layer split), but for completeness it could be noted.

2. **Fused cross-entropy communication savings**: The spec mentions TP communication volume per layer but does not model the output/embedding layer communication. The Megatron optimization (fused cross-entropy) avoids the `b x s x V` all-gather that would otherwise be needed. If the calculator models per-layer communication, it should note that the output layer does NOT add `b x s x V` to TP communication when fused cross-entropy is used (which is standard in Megatron-LM and most modern frameworks).

3. **Strong scaling efficiency warning**: The spec could add a note that TP primarily helps with memory, not throughput -- the strong scaling efficiency for a 1.2B model (that fits on 1 GPU) is only 37% at TP=8. The calculator's throughput model should account for this if it estimates per-GPU effective FLOPS with TP enabled.

4. **Attention head count effect on TP efficiency**: Table 7 shows 82% vs 77% scaling efficiency depending on head count with the same TP degree. More heads per GPU = smaller GEMMs per head = worse compute efficiency. The calculator could optionally warn when `d / a` (head dimension) is very small (e.g., < 64) or when `a / N_tp` (heads per GPU) is very small (e.g., < 2).

5. **Embedding all-reduce**: The spec does not explicitly note that the input embedding requires an additional all-reduce (the `g` operator) when TP is active. This is one extra all-reduce per forward pass and one extra all-reduce per backward pass, beyond the per-layer counts. For models with many layers this is negligible, but for shallow models it's a real cost.

### 14.3 No Changes Needed

The paper does not provide formulas for:
- Per-GPU activation memory with TP (this comes from the later Korthikanti et al., 2022 paper, already referenced in our spec)
- ZeRO integration (orthogonal; covered by our spec via DeepSpeed references)
- LoRA or other PEFT techniques (postdates this paper)
- Sequence parallelism (introduced in the follow-up Megatron-LM v2 paper)
- Expert parallelism / MoE (not addressed)

---

## 15. Key Formulas Summary for Calculator

| Formula | Expression | Source |
|---------|-----------|--------|
| TP communication per layer | `4 x 2(N_tp-1)/N_tp x b x s x d x beta` | Figure 4 |
| Vocab padding | `ceil(V/(128 x N_tp)) x 128 x N_tp` | Section 5.1 |
| Per-GPU params (approx) | `Psi / N_tp` | Implied by splitting |
| Per-GPU params (exact, per layer) | `(4d^2 + 2*d*d_ff)/N_tp + 4d` | Weight split + duplicated LN |
| Heads per GPU constraint | `a % N_tp == 0` | Section 3 |
| d_ff divisibility constraint | `d_ff % N_tp == 0` | Implied by column split |
| Total GPUs | `N_tp x N_dp x N_pp` | Appendix B.1 |
| Pipeline bubble (1F1B) | `(N_pp-1)/(num_microbatches + N_pp - 1)` | Not in this paper; included for context |
| Weak scaling efficiency (TP only, 8 GPU) | ~77% | Figure 5, Table 1 |
| Strong scaling efficiency (TP 8 GPU, 1.2B) | ~37% | Table 8 |
