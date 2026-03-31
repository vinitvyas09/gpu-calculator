# Deep Dive: PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)

**Source**: https://arxiv.org/abs/2204.02311
**Authors**: Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al. (Google Research)
**Date**: April 2022 (v5 October 2022)
**Relevance**: Defines MFU (Model FLOPs Utilization), provides the per-token FLOPs formula, reports efficiency benchmarks at scale

---

## 1. Model Architecture Details (Table 1)

| Model     | Layers | Heads | d_model | d_head | d_ff    | Parameters (B) | Vocab Size |
|-----------|--------|-------|---------|--------|---------|----------------|------------|
| PaLM 8B  | 32     | 16    | 4096    | 256    | 16384   | 8.63           | 256,000    |
| PaLM 62B | 64     | 32    | 8192    | 256    | 32768   | 62.50          | 256,000    |
| PaLM 540B| 118    | 48    | 18432   | 256    | 73728   | 540.35         | 256,000    |

**Critical relationships**:
- `d_ff = 4 * d_model` for all sizes
- `d_head = 256` for all sizes (this is constant, NOT d_model / n_heads)
- For PaLM 540B: `n_heads * d_head = 48 * 256 = 12,288`, but `d_model = 18,432`. These are NOT equal.
- This means PaLM uses a **non-standard dimension mapping** where d_model != n_heads * d_head.
- Sequence length: 2048 for all models
- Vocabulary: 256k tokens (SentencePiece)

**Architectural modifications from standard Transformer**:
- **SwiGLU activation**: Uses `Swish(xW) * xV` for MLP. Requires **3 matrix multiplications** in the MLP (not 2), but d_ff is sized for compute-equivalence.
- **Parallel layers**: `y = x + MLP(LayerNorm(x)) + Attention(LayerNorm(x))` instead of serial. ~15% faster training at large scales because MLP and Attention input matmuls can be fused.
- **Multi-Query Attention**: Q is projected to [k, h] per head, but K and V projections are **shared across heads** to [1, h]. Neutral for training speed, significant savings for inference.
- **RoPE embeddings**: Rotary Position Embeddings instead of absolute/relative.
- **No biases**: No biases in any dense kernels or layer norms (improves training stability).
- **Shared input-output embeddings**: Same matrix for input embedding and output projection.

---

## 2. The Per-Token FLOPs Formula (The "PaLM Formula")

This is defined in **Appendix B** and is the canonical formula we should adopt for MFU calculation.

### 2.1 Non-attention FLOPs

For a dense Transformer decoder-only model with N parameters:

```
Non-attention matmul FLOPs per token = 6N
```

**Derivation**: Each matmul FLOP is counted as a multiply-add = 2 FLOPs per parameter.
- Forward pass: 2N FLOPs per token (each parameter participates in one matmul)
- Backward pass: 4N FLOPs per token (2x for recomputing gradients w.r.t. inputs, 2x for gradients w.r.t. weights)
- Total: 2N + 4N = 6N

This comes from Kaplan et al. (2020).

### 2.2 Self-Attention FLOPs

The matmuls in dense self-attention add:

```
Attention FLOPs per token = 6 * L * H * (2 * Q * T)
                          = 12 * L * H * Q * T
```

Where:
- `L` = number of layers
- `H` = number of attention heads
- `Q` = attention head dimension (d_head, which is 256 for PaLM)
- `T` = sequence length

**The factor of 6 comes from**: 2 for multiply-add, times 3 for {forward, backward-input-grad, backward-weight-grad}. Wait -- attention QK^T and scores*V have no "weight grad" (they are activation-activation products). Let me be more precise:

The attention FLOPs per token come from two operations per layer:
1. `QK^T`: matrix of shape [T, d_head] x [d_head, T] = [T, T]. This is 2*T*d_head FLOPs per query position, per head.
2. `scores * V`: matrix of shape [T, T] x [T, d_head] = [T, d_head]. This is 2*T*d_head FLOPs per query position, per head.

Per token, per head, per layer: `2 * (2 * T * Q)` = `4 * T * Q` in forward.
Backward doubles this (gradients for both inputs): `4 * T * Q * 2` = `8 * T * Q`.
Total per token, per head, per layer: `4*T*Q + 8*T*Q = 12*T*Q`.
Over all L layers, H heads: `12 * L * H * Q * T`.

### 2.3 Combined Formula

```
Total FLOPs per token = 6N + 12*L*H*Q*T
```

The paper explicitly notes this attention term is "a much smaller value for large models" -- because N grows as d_model^2 while the attention term grows as d_model * T (and T is fixed at 2048, much smaller than d_model for large models).

### 2.4 Concrete Values (Table 22)

| Model | FLOPs/token (non-attn+attn) | FLOPs/token (non-attn+attn+remat) |
|-------|-----------------------------|------------------------------------|
| 8B    | 0.0550 TFLOP               | 0.0561 TFLOP                       |
| 62B   | 0.388 TFLOP                | 0.392 TFLOP                        |
| 540B  | 3.28 TFLOP                 | 4.10 TFLOP                         |

**Key observation**: For 8B, rematerialization adds only 2% overhead. For 540B, it adds **25% overhead**. This is because PaLM 540B uses aggressive rematerialization while the 8B and 62B models do not.

### 2.5 Verification of the Formula

For PaLM 540B:
- N = 540.35e9
- L = 118, H = 48, Q = 256, T = 2048
- Non-attention: 6 * 540.35e9 = 3.242e12 = 3.242 TFLOP
- Attention: 12 * 118 * 48 * 256 * 2048 = 12 * 118 * 48 * 256 * 2048
  = 12 * 118 * 48 * 524288
  = 12 * 118 * 25,165,824
  = 12 * 2,969,567,232
  = 35,634,806,784
  = 0.0356 TFLOP
- Total: 3.242 + 0.036 = 3.278 TFLOP ~= 3.28 TFLOP (matches table)

This confirms: for 540B, the attention FLOPs are only ~1.1% of the total. The 6N approximation is highly accurate for large models.

---

## 3. MFU (Model FLOPs Utilization) -- THE Key Metric

### 3.1 Definition

From Section 4.1 and Appendix B:

> MFU is the ratio of the observed throughput (tokens-per-second) relative to the theoretical maximum throughput of a system operating at peak FLOPs.

> Crucially, the "theoretical maximum" throughput only accounts for the required operations to compute the forward+backward passes, and **not rematerialization**.

### 3.2 Formula

```
MFU = R / (6N + 12*L*H*Q*T)
```

Where `R` is the achieved throughput in tokens-per-second, and the denominator is the model FLOPs per token divided by the peak hardware FLOPs per second. More precisely:

```
MFU = (observed_tokens_per_second * FLOPs_per_token) / (num_accelerators * peak_FLOPs_per_accelerator)
```

Or equivalently:

```
         observed_tokens_per_second
MFU = ─────────────────────────────────────────────────
       peak_total_FLOPs / (6N + 12LHQT)
```

Where `peak_total_FLOPs = num_accelerators * peak_FLOPs_per_accelerator`.

The paper writes it as:

```
        P
R = ─────────────────
    (6N + 12LHQT)
```

Where P is the total theoretical peak matmul throughput of the system in FLOPs/second.

### 3.3 Why MFU and Not HFU

The paper makes a critical argument for MFU over HFU:

1. **HFU is implementation-dependent**: Different compilers/frameworks make different rematerialization choices, changing how many FLOPs the hardware actually executes. Two systems training the same model could report very different HFU numbers.

2. **HFU rewards wasted computation**: A system that rematerializes everything (more total FLOPs) could report higher HFU than a system that cleverly avoids rematerialization.

3. **MFU is architecture-only**: The numerator is simply observed tokens/sec, the denominator depends only on model architecture and hardware specs. No implementation details leak in.

4. **MFU is the "true" efficiency**: The goal of training is tokens/sec, not FLOPs/sec. MFU directly measures how close you are to the theoretical maximum tokens/sec.

### 3.4 The MFU vs. HFU distinction for rematerialization

- **MFU numerator**: actual observed tokens/second
- **MFU denominator**: theoretical max tokens/second assuming ONLY the mandatory forward+backward FLOPs (6N + 12LHQT per token)
- **HFU numerator**: analytically estimated hardware FLOPs/second (includes rematerialization FLOPs)
- **HFU denominator**: peak hardware FLOPs/second

So for PaLM 540B:
- MFU: 46.2% (using 3.28 TFLOP/token)
- HFU: 57.8% (using 4.10 TFLOP/token, which includes rematerialization)

The relationship: `HFU = MFU * (FLOPs_with_remat / FLOPs_without_remat)`
- `57.8% ~= 46.2% * (4.10 / 3.28) = 46.2% * 1.25 = 57.75%` -- matches.

---

## 4. Reported Efficiency Numbers

### 4.1 MFU Comparison Across Models (Table 3)

| Model              | Parameters | Accelerators     | MFU   |
|--------------------|-----------|------------------|-------|
| GPT-3              | 175B      | V100             | 21.3% |
| Gopher             | 280B      | 4096 TPU v3      | 32.5% |
| Megatron-Turing NLG| 530B      | 2240 A100        | 30.2% |
| PaLM               | 540B      | 6144 TPU v4      | 46.2% |

### 4.2 PaLM 540B Throughput Details

- **Average training throughput**: 238.3K tokens/sec at batch size 2048 (the final batch size)
- **Hardware**: 6144 TPU v4 chips across 2 pods (3072 chips per pod)
- **Peak matmul TFLOP/s per TPU v4**: 275 TFLOP/s (this is implied from the MFU calculation)

**MFU verification**:
```
MFU = (238.3e3 tokens/sec * 3.28e12 FLOPs/token) / (6144 * 275e12 FLOPs/sec)
    = (238.3e3 * 3.28e12) / (6144 * 275e12)
    = 7.816e17 / 1.6896e18
    = 0.4626
    = 46.3%  (matches 46.2%)
```

**Without self-attention**:
- MFU without self-attention = 45.7%
- MFU with self-attention = 46.2%

### 4.3 Megatron-Turing NLG MFU Derivation (for comparison)

The paper explains how they calculated MFU for MT-NLG 530B:
- 2240 A100 GPUs with 312 peak matmul TFLOP/s each
- Training throughput: 65.43K tokens/sec (1920 * 2048 / 60.1 seconds per step)
- MFU = (65.43 * 6 * 530) / (312 * 2240) = 29.7% without attention, 30.2% with attention

### 4.4 PaLM 540B MFU variants

- **Without self-attention**: 45.7%
- **With self-attention**: 46.2%
- **HFU (with rematerialization)**: 57.8%

### 4.5 Two-Pod Scaling Efficiency

- Single pod throughput: baseline
- Two-pod throughput: 1.95x of single pod (97% of perfect 2x weak scaling)
- Each pair of hosts exchanges ~1.3 GB of gradients per step
- Aggregate burst: 81 Tbps across all hosts

---

## 5. Parallelism Configuration

### 5.1 PaLM 540B Configuration

- **Total chips**: 6144 TPU v4 (2 pods of 3072 each)
- **Model parallelism**: 12-way (within each pod, weight tensors partitioned across 12 chips)
- **Data parallelism**: 256-way fully sharded (within each pod, 3072/12 = 256)
- **Pipeline parallelism**: NONE -- this is a key achievement
- **Cross-pod parallelism**: 2-way data parallelism at the pod level (each pod gets half the batch)

This is described as "2D finalized" (Xu et al., 2021):
- Each TPU v4 Pod has a full copy of model parameters
- Weight tensors are partitioned over 3072 chips using 12-way model parallelism and 256-way fully sharded data parallelism
- During forward pass: weights are all-gathered over the data parallel axis
- One fully sharded activation tensor is saved per layer
- During backward pass: activations are rematerialized (rest of them)

### 5.2 Why No Pipeline Parallelism

The paper explicitly argues against pipeline parallelism at this scale:
1. **Pipeline bubble**: Many devices idle while filling/emptying the pipeline
2. **Memory bandwidth**: Weights must be reloaded from memory for each micro-batch
3. **Software complexity**: Added implementation complexity

Instead, they use the Pathways system to achieve pod-level data parallelism.

### 5.3 PaLM 540B Rematerialization Strategy

- 75% of non-attention forward pass FLOPs are recomputed during backward pass
- ALL attention forward pass FLOPs are recomputed during backward pass
- This trades compute for memory, enabling larger batch sizes
- The 8B and 62B models do NOT use rematerialization
- The choice of rematerialization increases feasible batch size, which increases training throughput despite the extra compute

---

## 6. Training Configuration

### 6.1 Optimizer

- **Adafactor** (Shazeer & Stern, 2018) without factorization (effectively Adam with parameter scaling)
- Learning rate: 10^-2, decayed at rate 1/sqrt(k) where k is step number
- Momentum: beta_1 = 0.9
- Second moment: beta_2 = 1.0 - k^(-0.8) (NOT the standard 0.99; varies with step number)
- Weight decay: dynamic, lr^2.0 during training
- Gradient clipping: global norm, value 1.0
- No dropout during pretraining

### 6.2 Batch Size Schedule

All models use increasing batch sizes during training:

| Model     | Batch Size Schedule                          |
|-----------|----------------------------------------------|
| PaLM 8B  | 256 -> 512                                   |
| PaLM 62B | 512 -> 1024                                  |
| PaLM 540B| 512 (1M tokens) -> 1024 (2M) -> 2048 (4M)   |

For PaLM 540B:
- BS 512 until step 50k
- BS 1024 until step 115k
- BS 2048 until step 255k (training complete)

Batch sizes listed are in number of sequences. With sequence length 2048:
- BS 512 = 1M tokens per batch
- BS 1024 = 2M tokens per batch
- BS 2048 = 4M tokens per batch (~4.2M tokens)

### 6.3 Sequence Length

- Fixed at 2048 for all models
- Input examples concatenated and split into exactly 2048 tokens (no padding)
- `[eod]` token separates documents

### 6.4 Training Tokens

| Model     | Training Tokens |
|-----------|----------------|
| PaLM 8B  | 780B           |
| PaLM 62B | 795B           |
| PaLM 540B| 780B           |

All models trained for exactly 1 epoch of the dataset. The 62B model trained slightly longer due to an oversight in checkpoint selection.

---

## 7. Total Compute (Table 22)

| Model | FLOPs/token (no remat) | FLOPs/token (with remat) | Total Train FLOPs    | PetaFLOP/s-days |
|-------|------------------------|--------------------------|----------------------|-----------------|
| 8B    | 0.0550 TFLOP           | 0.0561 TFLOP             | 4.29 x 10^22        | 497             |
| 62B   | 0.388 TFLOP            | 0.392 TFLOP              | 3.08 x 10^23        | 3,570           |
| 540B  | 3.28 TFLOP             | 4.10 TFLOP               | 2.56 x 10^24        | 29,600          |

**Note on footnote 20**: The paper also mentions a simpler approximate FLOP count formula used elsewhere:
```
Total Training FLOPs ~ 2 * 3 * N * T_tokens = 6 * N * T_tokens
```
Where 2 is for multiply-add, 3 is one for forward plus two for backward, N is model size, T_tokens is total training tokens. This simpler formula ignores attention FLOPs entirely.

**Verification for PaLM 540B**:
```
Total FLOPs = FLOPs_per_token * num_tokens = 3.28e12 * 780e9 = 2.558e24 ~ 2.56e24 (matches)
```

### 7.1 Training Duration

From Appendix B:
- PaLM 540B was trained on 6144 TPU v4 chips for **1200 hours**
- Also trained on 3072 TPU v4 chips for **336 hours** (including downtime and repeated steps)
- Power per TPU v4 chip: 378.5W (measured system power)

### 7.2 Training FLOP Count (Table 21)

For cross-model comparison:
| Model       | Parameters | Tokens | Training FLOPs (Zettaflops) |
|-------------|-----------|--------|-----------------------------|
| PaLM 8B    | 8B        | 780B   | 37.4                        |
| PaLM 62B   | 62B       | 795B   | 295.7                       |
| PaLM 540B  | 540B      | 780B   | 2527.2                      |
| Chinchilla | 70B       | 1400B  | 588.0                       |
| Gopher     | 280B      | 300B   | 504.0                       |

---

## 8. d_model vs n_heads * d_head -- The Critical Subtlety

This is the most important non-obvious detail for our calculator.

For PaLM 540B:
- `d_model = 18432`
- `n_heads = 48`
- `d_head = 256`
- `n_heads * d_head = 48 * 256 = 12,288`
- **d_model != n_heads * d_head** (18432 != 12288)

This means:
1. The Q/K/V projection matrices are NOT square. Q projection is [d_model, n_heads * d_head] = [18432, 12288] for standard MHA (or [18432, n_heads * d_head] for Q and [18432, d_head] for K, V in multi-query attention).
2. The output projection is [n_heads * d_head, d_model] = [12288, 18432].
3. The parameter count from attention is DIFFERENT from a model where d_model = n_heads * d_head.

With multi-query attention, the projection sizes per layer are:
- Q: [d_model, n_heads * d_head] = [18432, 12288] -- projects to all heads
- K: [d_model, d_head] = [18432, 256] -- shared across heads
- V: [d_model, d_head] = [18432, 256] -- shared across heads
- O: [n_heads * d_head, d_model] = [12288, 18432] -- output projection

**How this affects FLOPs**:
The "6N" approximation lumps all parameters together. The attention parameters and MLP parameters all contribute their share to N, and each parameter contributes 6 FLOPs per token. So the formula handles this correctly as long as N is the actual parameter count (540.35B), not a derived count based on architectural formulas.

The separate attention term `12*L*H*Q*T` counts the QK^T and scores*V products, which are independent of d_model -- they only depend on d_head, n_heads, and sequence length. So this term is also correct regardless of the d_model vs n_heads*d_head relationship.

**Impact on our calculator**: When computing N from architecture, we must be careful:
- MLP parameters per layer: d_model * d_ff * 3 (for SwiGLU: W_gate, W_up, W_down; where d_ff is the intermediate size) or d_model * d_ff * 2 (for standard FFN: W_up, W_down). PaLM uses SwiGLU with 3 matrices but d_ff = 4 * d_model, so MLP params = 3 * d_model * 4 * d_model = 12 * d_model^2 per layer.
- For standard GeLU FFN: MLP params = 2 * d_model * d_ff = 8 * d_model^2 per layer (when d_ff = 4 * d_model).
- Attention parameters per layer (multi-query): d_model * (n_heads * d_head) + 2 * d_model * d_head + (n_heads * d_head) * d_model
  = d_model * n_heads * d_head + 2 * d_model * d_head + n_heads * d_head * d_model
  = 2 * d_model * n_heads * d_head + 2 * d_model * d_head
- Attention parameters per layer (standard MHA): d_model * (n_heads * d_head) * 4 = 4 * d_model * n_heads * d_head

---

## 9. Training Instability (Section 5.1)

For the 540B model:
- **~20 loss spikes** observed during training
- Occurred at highly irregular intervals, sometimes late in training
- NOT observed in 8B or 62B models
- Gradient clipping was enabled but did not prevent spikes

**Mitigation**: Restart from checkpoint ~100 steps before spike, skip 200-500 data batches.

**Root cause analysis**: Spikes were NOT caused by "bad data" per se. The same data batches did not cause spikes when training from a different earlier checkpoint. The spikes occur due to the combination of specific data batches WITH a particular model parameter state.

---

## 10. What's Unique / Non-Obvious

### 10.1 MFU is the PaLM paper's key contribution for efficiency measurement

Before PaLM, the standard metric was HFU. The PaLM paper explicitly argues that MFU is superior and should be the standard. This has been widely adopted since.

### 10.2 The per-token FLOPs formula includes both N-dependent and attention terms

Many calculators use only `6N`. The PaLM formula is `6N + 12LHQT`. For large models the attention term is negligible (~1% for 540B), but for small models or very long sequences it becomes significant.

### 10.3 Rematerialization multiplier varies dramatically with model size

- 8B: 1.02x (rematerialization adds 2%)
- 62B: 1.01x (rematerialization adds 1%)
- 540B: 1.25x (rematerialization adds 25%)

This is because the 540B model uses aggressive rematerialization to fit in memory, while the smaller models do not need it.

### 10.4 The "parallel layers" formulation gives 15% speedup

This is specific to PaLM's architecture but suggests that MFU/HFU numbers are affected by architectural choices that change the compute graph, not just parallelism and hardware.

### 10.5 Multi-query attention affects parameter count but not attention FLOPs

The QK^T and scores*V FLOPs are the same regardless of whether K/V are shared across heads. The savings from multi-query attention come from reduced parameters and reduced memory for KV cache during inference, not from reduced training FLOPs.

### 10.6 The implicit peak FLOP/s for TPU v4

The MFU calculation implies TPU v4 has 275 TFLOP/s peak matmul throughput:
```
MFU = 0.462 = (238.3e3 * 3.28e12) / (6144 * P)
P = (238.3e3 * 3.28e12) / (6144 * 0.462)
P = 7.816e17 / 2838.5
P = 2.753e14 = 275.3 TFLOP/s
```

This is the **bf16 peak matmul TFLOP/s** for TPU v4.

### 10.7 The simple 6*N*T formula for total training FLOPs

Footnote 20 gives: `Total FLOPs ~ 2 * 3 * N * T = 6NT` where T is total tokens. This is the most commonly cited approximation and is what Chinchilla scaling laws use. The "2" is for multiply-add per parameter, and "3" is one forward pass + two backward passes (one for activation gradients, one for weight gradients).

### 10.8 No pipeline parallelism at 540B scale

PaLM achieved better efficiency than Megatron-Turing NLG (which used pipeline parallelism) without using PP at all. This is a strong data point that PP introduces overhead (pipeline bubbles) that can be avoided with the right system architecture.

### 10.9 Batch size affects MFU

The paper uses batch size 2048 for reporting the 46.2% MFU number. Larger batch sizes enable larger matrix multiplications, which increases hardware utilization. The batch size schedule (512 -> 1024 -> 2048) means MFU varies during training.

---

## 11. Formulas Summary for Our Calculator

### Primary: Per-token FLOPs (for MFU denominator)

```
FLOPs_per_token = 6*N + 12*L*H*Q*T

Where:
  N = total model parameters
  L = number of layers
  H = number of attention heads
  Q = attention head dimension (d_head)
  T = sequence length
```

### Primary: MFU

```
MFU = (observed_tokens_per_sec * FLOPs_per_token) / (num_chips * peak_FLOPS_per_chip)
```

Or equivalently:
```
MFU = observed_tokens_per_sec / theoretical_max_tokens_per_sec

where theoretical_max_tokens_per_sec = (num_chips * peak_FLOPS_per_chip) / FLOPs_per_token
```

### Primary: HFU

```
HFU = (observed_tokens_per_sec * FLOPs_per_token_with_remat) / (num_chips * peak_FLOPS_per_chip)

HFU = MFU * (FLOPs_per_token_with_remat / FLOPs_per_token_without_remat)
```

### Secondary: Total Training FLOPs

```
Total_FLOPs = FLOPs_per_token * total_training_tokens
            ~ 6 * N * total_training_tokens  (approximate, ignoring attention)
```

### Secondary: Training Time Estimate

```
Training_time_seconds = Total_FLOPs / (num_chips * peak_FLOPS_per_chip * MFU)
```

Or:
```
Training_time_seconds = total_tokens / observed_tokens_per_sec
                      = total_tokens / (num_chips * peak_FLOPS_per_chip * MFU / FLOPs_per_token)
```

---

## 12. Gaps / What This Paper Does NOT Cover

1. **No per-GPU memory breakdown**: The paper does not report memory usage per chip (optimizer states, activations, parameters, gradients).
2. **No communication overhead formulas**: While they mention 1.3 GB per host pair per step and 81 Tbps aggregate, there are no general formulas for communication volume.
3. **No ZeRO-style sharding analysis**: The "fully sharded data parallelism" is mentioned but not analyzed in terms of memory savings formulas.
4. **No LoRA or PEFT**: Paper predates the widespread adoption of these techniques.
5. **No training loss values reported**: The paper reports downstream task performance but does not include training loss curves or final training loss values in the text or tables. (Training loss curves may exist in figures but exact values are not given.)
6. **TPU-specific**: All numbers are for TPU v4. Mapping to GPU architectures requires adjusting peak FLOP/s values.
7. **No mixed-precision analysis**: The paper does not discuss bf16 vs fp32 or their memory implications. TPU v4 natively uses bf16 for matmuls.

---

## 13. Relevant File Paths

The PDF of the paper is cached at:
`/Users/vinitvyas09/.claude/projects/-Users-vinitvyas09-code-personal-llm-training-gpu-calculator/d91b4879-75e5-4f21-8298-32855ec485d1/tool-results/webfetch-1774994792125-hro16a.pdf`
