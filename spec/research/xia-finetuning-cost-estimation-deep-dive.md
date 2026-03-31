# Deep Dive: Understanding the Performance and Estimating the Cost of LLM Fine-Tuning

**Paper**: Xia, Y., Kim, J., Chen, Y., Ye, H., Kundu, S., Hao, C., & Talati, N. (2024). "Understanding the Performance and Estimating the Cost of LLM Fine-Tuning." IEEE International Symposium on Workload Characterization (IISWC), pp. 210-223.
**URL**: https://arxiv.org/abs/2408.04693
**Code**: https://github.com/stsxxx/finetune
**Published**: IISWC 2024

---

## 1. Executive Summary

This paper provides an empirical characterization and analytical cost model for fine-tuning MoE (Mixture-of-Experts) LLMs on a single GPU. The two primary contributions are:

1. **Profiling study** of sparse vs. dense MoE fine-tuning (Mixtral-8x7B with QLoRA and BlackMamba with full fine-tuning), covering memory, throughput, execution time breakdown, GPU SM/memory utilization, and expert load distribution.
2. **Analytical model** with two equations to estimate (a) maximum batch size given GPU memory and (b) throughput given batch size, which together yield a cost estimate.

**Scope**: Single-GPU fine-tuning only. Does NOT cover multi-GPU, pretraining, pipeline parallelism, tensor parallelism, or ZeRO. The analytical model is empirically calibrated (curve-fitted) rather than derived from first principles.

**Relevance to our spec**: Moderate. The paper provides useful empirical data and a cost estimation framework, but the formulas are semi-empirical (requiring coefficient fitting per model/GPU/dataset combination). The MoE-specific insights (sparsity effects on memory and throughput) are the most unique contributions. The paper does NOT provide the kind of first-principles memory formulas (weights + optimizer states + gradients + activations) that our calculator needs.

---

## 2. Analytical Model: Maximum Batch Size (Equation 1)

### 2.1 Formula

```
Max_BSZ = floor( C0 * (GPU_mem - model_mem) / (seq_len * ((1 - C1) + C1 * sparsity)) )
```

### 2.2 Parameters

| Parameter | Description | Source |
|-----------|-------------|--------|
| `C0` | Scaling coefficient (model-dependent) | Empirically fitted |
| `C1` | MoE coefficient (model-dependent) | Empirically fitted |
| `GPU_mem` | Total GPU memory in GB | Hardware spec |
| `model_mem` | Model weight memory footprint in GB | Empirically measured |
| `seq_len` | Sequence length | Dataset property |
| `sparsity` | Fraction of experts activated (e.g., 2/8 = 0.25 for top-2-of-8) | Model config |

### 2.3 Fitted Coefficient Values

| Model | C0 | C1 |
|-------|----|----|
| Mixtral-8x7B (QLoRA, 4-bit) | 82 | 0.95 |
| BlackMamba (full fine-tuning) | 83 | 0.88 |

**CRITICAL DISCREPANCY**: The paper states C0=82 for Mixtral, but the actual code (`analytical_capacity/plot.py`) uses **C0=62 and C1=0.89** with **seq_len=512** hardcoded. The paper's coefficients may correspond to a different sequence length normalization. This is a significant inconsistency between paper and code.

From the code:
```python
C0 = 62
C1 = 0.89
model_mem = 23.35
seq_length = 512
sparsity = 0.25
yy1 = [C0*(xx[i]-model_mem)/(seq_length*((1-C1)+C1*sparsity)) for i in range(len(xx))]
```

### 2.4 What model_mem Represents

For Mixtral (47B params, 23.35GB): this is the 4-bit quantized QLoRA weight footprint.
- 47B params * 4 bits / 8 bits_per_byte = ~23.5 GB -- matches 23.35GB.

For BlackMamba (2.8B params, 5.6GB): this is the full fp16 weight footprint.
- 2.8B * 2 bytes = 5.6 GB -- exact match.

The paper does NOT provide a formula for model_mem; it is measured empirically. This means it implicitly includes LoRA adapter overhead, optimizer states, gradient memory, and base CUDA context -- or alternatively, C0 absorbs these into its scaling.

### 2.5 How Sparsity Affects Memory

The term `((1 - C1) + C1 * sparsity)` creates a weighted blend:
- When sparsity=1.0 (dense, all experts active): denominator factor = 1.0
- When sparsity=0.25 (top-2-of-8): denominator factor = (1-0.95) + 0.95*0.25 = 0.05 + 0.2375 = 0.2875 (for Mixtral)

This means sparse MoE reduces the per-batch memory by ~3.5x (1/0.2875) compared to dense, which directly translates to supporting ~3.5x larger batch sizes. The empirical data confirms this: Mixtral-CS dense supports bsz=2, sparse supports bsz=8 (4x ratio).

### 2.6 Validation Against Ground Truth

From the code, three ground truth points for Mixtral sparse on MATH (seq_len=512):

| GPU | Memory (GB) | Measured Max BSZ | Model Prediction |
|-----|-------------|------------------|-----------------|
| A100-40GB | 40 | 6 | ~6 |
| A40 | 48 | 8 | ~8 |
| A100/H100-80GB | 80 | 27 | ~19 (paper shows line through point) |

### 2.7 Limitations

- Coefficients are model-specific and must be re-fitted for each new model.
- Does NOT decompose memory into components (weights, gradients, optimizer states, activations).
- Does NOT account for CUDA memory fragmentation, padding, or peak transient allocations.
- Does NOT distinguish between QLoRA, LoRA, and full fine-tuning memory analytically.
- Sequence length interaction is simplistic (linear in denominator), but real memory scaling with seq_len is sublinear for attention (FlashAttention) and varies by layer type.

---

## 3. Analytical Model: Throughput Estimation (Equation 2)

### 3.1 Formula (Paper Version)

```
Throughput = C2 * log(batch_size / (sparsity * C3)) + C4
```

### 3.2 Formula (Actual Code Implementation)

The code reveals the actual implementation is DIFFERENT from the paper's equation. The code uses:

```python
# The adjustment transforms batch_size by sparsity before fitting
x_adjust = x ** (1 / (sparsity * coeff))

# Then fits: Throughput = a + b * log(x_adjust)
# Which expands to: Throughput = a + b * log(batch_size^(1/(sparsity*coeff)))
# = a + (b / (sparsity * coeff)) * log(batch_size)
```

So the ACTUAL formula is:

```
Throughput = C4 + C2 * log( batch_size^(1 / (sparsity * C3)) )
           = C4 + (C2 / (sparsity * C3)) * log(batch_size)
```

Where `C3` (called `coeff` in code) is NOT an MoE attenuation coefficient per the paper -- it is an exponent applied to batch_size.

### 3.3 Fitted Coefficient Values from Code

The throughput model is fitted via `scipy.optimize.curve_fit` using `lambda t,a,b: a+b*numpy.log(t)` where `t = batch_size^(1/(sparsity*coeff))`.

**A40 GPU** (from analytical_throughput/plot.py):

| Model-Dataset | coeff (C3) | Note |
|--------------|------------|------|
| Mixtral-CS | 1.8 | Attention-based MoE |
| Mixtral-MATH | 1.8 | Attention-based MoE |
| BlackMamba-CS | 2.5 | State-space MoE |
| BlackMamba-MATH | 2.5 | State-space MoE |

**A100-40GB, A100-80GB, H100** (from analytical_gpu/plot.py and analytical_yuhan/plot.py):

All use coeff=1.8 for Mixtral across all GPUs. The C2 and C4 values are fitted per GPU-model-dataset combination via curve_fit (exact values not stored, computed at runtime).

### 3.4 Raw Throughput Data (queries/second, A40 GPU)

**Mixtral-CS** (median seq_len=79):

| Batch Size | Sparsity | Throughput (qps) |
|-----------|----------|-----------------|
| 1 | 1.0 (dense) | 0.321 |
| 2 | 1.0 (dense) | 0.514 |
| 1 | 0.25 (sparse) | 0.341 |
| 2 | 0.25 | 0.657 |
| 3 | 0.25 | 0.926 |
| 4 | 0.25 | 1.135 |
| 5 | 0.25 | 1.306 |
| 6 | 0.25 | 1.460 |
| 7 | 0.25 | 1.579 |
| 8 | 0.25 | 1.655 |

**Mixtral-MATH** (median seq_len=174):

| Batch Size | Sparsity | Throughput (qps) |
|-----------|----------|-----------------|
| 1 | 1.0 (dense) | 0.283 |
| 1 | 0.25 (sparse) | 0.335 |
| 2 | 0.25 | 0.622 |
| 3 | 0.25 | 0.833 |
| 4 | 0.25 | 1.005 |

**BlackMamba-CS** (median seq_len=79):

| Batch Size | Sparsity | Throughput (qps) |
|-----------|----------|-----------------|
| 1 | 1.0 (dense) | 2.324 |
| 2 | 1.0 | 4.436 |
| 3 | 1.0 | 5.922 |
| 4 | 1.0 | 6.995 |
| 5 | 1.0 | 7.768 |
| 6 | 1.0 | 7.878 |
| 1 | 0.25 (sparse) | 2.408 |
| 2 | 0.25 | 4.587 |
| 6 | 0.25 | 10.462 |
| 10 | 0.25 | 12.897 |
| 15 | 0.25 | 14.024 |
| 20 | 0.25 | 14.874 |

**BlackMamba-MATH** (median seq_len=174):

| Batch Size | Sparsity | Throughput (qps) |
|-----------|----------|-----------------|
| 1 | 1.0 (dense) | 2.219 |
| 2 | 1.0 | 3.232 |
| 3 | 1.0 | 5.321 |
| 1 | 0.25 (sparse) | 2.174 |
| 2 | 0.25 | 4.570 |
| 5 | 0.25 | 8.964 |
| 8 | 0.25 | 11.097 |
| 10 | 0.25 | 11.612 |

**Mixtral on other GPUs** (Mixtral-CS, sparse):

A100-40GB:
| BSZ | Throughput |
|-----|-----------|
| 1 (dense) | 0.389 |
| 1 (sparse) | 0.449 |
| 2 | 0.815 |
| 3 | 1.151 |

A100-80GB:
| BSZ | Throughput |
|-----|-----------|
| 1 (dense) | 0.415 |
| 5 (dense) | 0.953 |
| 1 (sparse) | 0.474 |
| 5 | 1.708 |
| 10 | 2.305 |
| 17 | 2.740 |

H100:
| BSZ | Throughput |
|-----|-----------|
| 1 (dense) | 0.547 |
| 5 (dense) | 1.657 |
| 1 (sparse) | 0.672 |
| 5 | 2.168 |
| 9 | 3.153 |
| 15 | 4.666 |
| 17 | 4.899 |

### 3.5 RMSE Validation Results

| Model-Dataset-GPU | RMSE |
|-------------------|------|
| Mixtral-CS (A40) | 0.05 |
| Mixtral-MATH (A40) | 0.02 |
| Mamba-CS (A40) | 0.79 |
| Mamba-MATH (A40) | 0.42 |
| Mixtral-CS-A100-40GB | 0.03 |
| Mixtral-CS-A100-80GB | 0.09 |
| Mixtral-CS-H100 | 0.55 |

### 3.6 Throughput Measurement Methodology

From the code (`throughput.py`):
```python
tp = 1000 / tune_time  # queries per second = 1000 test queries / total runtime
```

Throughput is measured as 1000 queries divided by total training runtime (extracted from PyTorch trainer logs as `train_runtime`). This includes all overhead (data loading, gradient accumulation, optimizer step).

---

## 4. Cost Estimation Model

### 4.1 Formula (Implicit)

The paper does not state a formal cost equation, but it is derivable:

```
Total_Cost = (num_queries * num_epochs / Throughput) / 3600 * GPU_hourly_rate
```

Where:
- `num_queries` = dataset size (number of prompt+answer pairs)
- `num_epochs` = training epochs
- `Throughput` = queries/second (from Equation 2)
- `GPU_hourly_rate` = cloud rental cost in $/hr

### 4.2 Cost Validation Data (Table IV)

Fine-tuning Mixtral on GSM8K (1.3K queries) with sparse MoE, 10 epochs:

| GPU | Memory | Max BSZ | Throughput (qps) | Cloud $/hr | Total Cost ($) |
|-----|--------|---------|------------------|-----------|----------------|
| A40 | 48GB | 4 | 1.01 | 0.79 | 32.7 |
| A100-80GB | 80GB | 17 | 2.74 | 1.67 | 25.4 |
| H100-80GB | 80GB | 17 | 4.90 | 2.10 | 17.9 |

Verification: For H100: (1300 * 10 / 4.90) / 3600 * 2.10 = 2653.06 / 3600 * 2.10 = 0.7370 * 2.10 = $1.55. This does NOT match $17.9.

Re-examining: The paper says "we extract 1000 examples from each dataset and fine-tuned... 10 epochs." If using full MATH dataset (14K queries): (14000 * 10 / 4.90) / 3600 * 2.10 = 28571.4 / 3600 * 2.10 = 7.9365 * 2.10 = $16.67. Close to $17.9.

**Cloud GPU Pricing** (CUDO Compute, as of 2024):

| GPU | $/hr |
|-----|------|
| A40 | 0.79 |
| A100-80GB | 1.67 |
| H100-80GB | 2.10 |

**Large-scale projection**: Fine-tuning Mixtral (sparse) on OpenOrca (2M queries) with H100 = $3,460.

### 4.3 GPU Specifications in Code

The code includes a broader GPU dictionary (not all used in the paper):

```python
GPU_dict = {
    "H100":  [80, 12.29],   # [memory_GB, ...]
    "A100":  [40, 4.097],
    "V100":  [16, 3.06],
    "K80":   [12, 0.9],
    "L4":    [24, 0.8048],
    "A10G":  [16, 1.006],
    "T4":    [16, 0.526],
    "M60":   [8,  0.75],
}
```

The second value in each entry appears to be a performance scaling factor or TFLOPS rating, though this is not explicitly documented. For reference: H100=12.29, A100=4.097 suggests these may be some normalized compute metric (not raw TFLOPS -- H100 SXM is 989 TFLOPS for FP8).

---

## 5. MoE-Specific Findings

### 5.1 Memory Impact of Sparsity

Sparse MoE (activating k-of-N experts) reduces intermediate activation memory proportionally to the active expert fraction. This allows larger batch sizes:

| Config | Mixtral-CS Max BSZ | Mixtral-MATH Max BSZ |
|--------|-------------------|---------------------|
| Dense (8/8) | 2 | 1 |
| Sparse (2/8) | 8 | 3 |

Ratio: 4x for CS (seq_len=79), 3x for MATH (seq_len=174). The ratio is sub-linear due to non-MoE components (attention, normalization) consuming fixed memory regardless of sparsity.

### 5.2 Throughput Impact of Sparsity

At the same batch size, sparse is faster than dense (fewer FLOPs per forward/backward pass):
- Mixtral-CS bsz=2: Dense=0.514 qps, Sparse=0.657 qps (1.28x)
- Combined with higher max batch size: Sparse bsz=8 = 1.655 qps vs Dense bsz=2 = 0.514 qps (3.22x)

### 5.3 MoE Layer Dominates Execution Time

The MoE layer accounts for **85% of overall execution time** on average across all configurations. Within the MoE layer:
- Matrix multiplications (W1, W2, W3) are the dominant kernels
- De-quantization (for QLoRA 4-bit) is significant at low batch sizes
- Router operations (softmax, topk, router matmul) are relatively cheap

### 5.4 Expert Load Imbalance After Fine-Tuning

Fine-tuning changes the expert routing distribution:

| Model | Dataset | Variance Before | Variance After | Change |
|-------|---------|----------------|---------------|--------|
| Mixtral | HE | 55.5 | 112.3 | +102% |
| Mixtral | GS | 21.2 | 79.2 | +274% |
| BlackMamba | HE | 150.7 | 93.3 | -38% |
| BlackMamba | GS | 186.5 | 187.9 | +1% |

For Mixtral, Expert 3 becomes dominantly utilized after fine-tuning. This has implications for expert parallelism efficiency.

### 5.5 Accuracy: Sparse vs Dense

From the convergence data (accuracy after 10 epochs):

| Model | Dataset | Dense Peak | Sparse Peak | Gap |
|-------|---------|-----------|-------------|-----|
| Mixtral | HE | 0.85 | 0.76 | 9pp (but sparse peaks similarly, drops due to overfitting) |
| Mixtral | GS | 0.57 | 0.58 | 0pp (sparse matches dense) |
| BlackMamba | HE | 0.36 | 0.33 | 3pp |
| BlackMamba | GS | 0.046 | 0.035 | ~1pp (both very low) |

Key insight: Sparse fine-tuning achieves comparable accuracy to dense, supporting the cost-efficiency argument for sparse MoE.

---

## 6. Hardware Utilization Data

### 6.1 SM Utilization (Mixtral MoE Layer, A40)

From the profiling data (columns: Dense-bsz1, Dense-bsz10, Sparse-bsz1, Sparse-bsz10, Sparse-bsz32):

| Kernel | Dense(1) | Dense(10) | Sparse(1) | Sparse(10) | Sparse(32) |
|--------|----------|-----------|-----------|------------|------------|
| matmul(w2) | 57.35% | 92.18% | 32.66% | 69.46% | 85.10% |
| w2_dequant | 82.99% | 82.99% | 82.99% | 83.00% | 82.99% |
| matmul(w3) | 51.71% | 90.64% | 35.82% | 78.47% | 87.57% |
| matmul(w1) | 51.80% | 90.59% | 35.84% | 78.42% | 87.54% |
| matmul(router) | 12.31% | 24.70% | 12.37% | 25.03% | 37.49% |
| **time_weighted** | **69.61%** | **87.53%** | **62.85%** | **76.80%** | **83.66%** |

Key observations:
- De-quantization kernels have constant ~83% SM utilization regardless of batch size
- Matrix multiplication kernels scale from ~50% to ~90% utilization with batch size
- Sparse is lower at same batch size but achieves similar peak utilization at its maximum batch size

### 6.2 DRAM Bandwidth Utilization (Mixtral MoE Layer, A40)

| Kernel | Dense(1) | Dense(10) | Sparse(1) | Sparse(10) | Sparse(32) |
|--------|----------|-----------|-----------|------------|------------|
| matmul(w2) | 92.13% | 64.21% | 92.50% | 71.88% | 65.80% |
| w2_dequant | 69.49% | 69.43% | 69.43% | 69.43% | 69.42% |
| matmul(w1) | 88.16% | 63.89% | 88.89% | 74.89% | 67.54% |
| **time_weighted** | **76.52%** | **65.39%** | **76.18%** | **71.56%** | **67.73%** |

Key observations:
- DRAM bandwidth utilization DECREASES with batch size (parameters loaded once, shared across batch)
- Small batch sizes are memory-bound; large batch sizes become compute-bound
- The crossover point is where SM utilization saturates and DRAM utilization drops

### 6.3 Sequence Length Sensitivity (BlackMamba, A40)

DRAM utilization across sequence lengths (Dense row, Sparse row):

| | SeqLen=64 | SeqLen=128 | SeqLen=256 | SeqLen=512 | SeqLen=1024 |
|-------|----------|-----------|-----------|-----------|------------|
| Dense | 72.78% | 72.77% | 71.78% | 71.71% | 71.80% |
| Sparse | 70.01% | 69.74% | 69.34% | 69.60% | 69.23% |

Latency across sequence lengths (seconds per step):

| | SeqLen=64 | SeqLen=128 | SeqLen=256 | SeqLen=512 | SeqLen=1024 |
|-------|----------|-----------|-----------|-----------|------------|
| Dense | 6.86 | 6.93 | 6.11 | 6.09 | 6.13 |
| Sparse | 6.93 | 6.88 | 6.92 | 6.90 | 6.90 |

**Surprising finding**: Latency is nearly constant across sequence lengths because batch size is adjusted to fill memory. Longer sequences = smaller batch size = similar total tokens per step.

---

## 7. Fine-Tuning Configuration Details

### 7.1 Mixtral-8x7B (QLoRA)

From the training scripts:
- **Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Quantization**: 4-bit (`--quantization_bit 4`)
- **LoRA rank**: 16 (`--lora_rank 16`)
- **LoRA targets**: w1, w2, w3, gate (MoE expert weights + router)
- **Precision**: bf16
- **FlashAttention**: v2 enabled (`--flash_attn`)
- **Gradient checkpointing**: enabled (saves memory, increases backward pass time)
- **Max sequence length**: 1024 (`--cutoff_len 1024`)
- **Learning rate**: 5e-5
- **Optimizer**: AdamW (default in HuggingFace Trainer)
- **Framework**: LLaMA-Factory

### 7.2 BlackMamba (Full Fine-Tuning)

- **Method**: Full fine-tuning (all parameters trainable)
- **No quantization**: fp16/bf16 weights
- **No FlashAttention**: uses standard attention (Mamba layers replace attention)
- **Gradient checkpointing**: not mentioned

### 7.3 Sparsity Control

Sparsity is controlled by modifying `num_experts_per_tok` in the model config JSON:
- Dense: `num_experts_per_tok = 8` (all 8 experts)
- Sparse: `num_experts_per_tok = 2` (top-2 gating)

---

## 8. What This Paper Does NOT Cover

These are areas where the paper is silent or explicitly defers to future work:

1. **No first-principles memory model**: No decomposition into weights + optimizer states + gradients + activations. The "model_mem" is measured, not computed.
2. **No multi-GPU support**: "extending this model to multi-GPU systems is left for future exploration"
3. **No communication overhead modeling**: No formulas for data parallel, tensor parallel, or pipeline parallel communication.
4. **No activation memory formulas**: Activations are implicitly captured in the C0 coefficient, not modeled.
5. **No LoRA rank ablation**: Rank is fixed at 16; no study of how rank affects memory or performance.
6. **No gradient accumulation analysis**: All experiments use gradient_accumulation_steps=1.
7. **No comparison with other memory estimators** (DNNMem, LLMem, etc.).
8. **No FLOP counting or MFU**: No theoretical FLOP analysis or model FLOP utilization metric.
9. **No peak memory vs steady-state analysis**.
10. **No CUDA fragmentation/padding considerations**.

---

## 9. Key Insights Relevant to Our Calculator Spec

### 9.1 Directly Usable

1. **MoE memory reduction factor**: The formula `((1-C1) + C1*sparsity)` where C1~0.9 provides a good approximation for how MoE sparsity reduces activation memory. For top-k-of-N experts with sparsity=k/N, the memory reduction for the MoE portion is approximately `sparsity * C1 + (1-C1)`.

2. **Throughput saturates logarithmically with batch size**: The log(batch_size) relationship for throughput is empirically validated and provides a reasonable model for diminishing returns.

3. **MoE layer is 85% of compute**: When building a FLOP model for MoE architectures, the MoE/FFN layer dominates execution time even more than in dense models.

4. **QLoRA 4-bit memory**: 47B params at 4-bit = 23.35 GB is consistent with expected: `47e9 * 0.5 bytes = 23.5 GB`.

5. **Cloud GPU cost data points** (2024 CUDO Compute pricing).

### 9.2 Insights to Inform Design

1. **Memory-bound to compute-bound transition**: At small batch sizes, throughput scales nearly linearly; at large batch sizes, it saturates. A calculator should indicate when a user's configuration is memory-bound vs compute-bound.

2. **Expert load imbalance**: For MoE models, fine-tuning can increase routing imbalance by 2-4x, which affects expert parallelism efficiency.

3. **Sequence length has minimal impact on per-step latency when batch size fills memory**: This is because shorter sequences allow larger batch sizes, keeping total token count per step roughly constant.

### 9.3 What NOT to Adopt

1. **The empirical coefficients (C0, C1, C2, C3, C4)**: These are model-specific and require re-fitting for each new model. They are not generalizable.

2. **The throughput formula**: It requires curve fitting on actual measurement data for each model-GPU combination, which defeats the purpose of a calculator.

3. **The "model_mem" as a black box**: Our calculator should decompose this into components (quantized weights, LoRA adapters, optimizer states, gradients, CUDA context).

---

## 10. Comparison with LLMem (Kim et al., 2024)

The paper explicitly cites LLMem [48] as related work. Key differences:

| Aspect | Xia et al. (this paper) | LLMem (Kim et al.) |
|--------|------------------------|---------------------|
| Scope | Fine-tuning (QLoRA + full) | Fine-tuning (full only) |
| Memory model | Empirical (model_mem measured) | First-principles (decomposed) |
| Throughput model | Yes (logarithmic) | No |
| Cost model | Yes | No |
| MoE support | Yes (core focus) | No |
| Multi-GPU | No | Yes (Colossal-AI/ZeRO-3) |
| Error rate | Not reported for memory | 1.6% single-GPU, 3.0% multi-GPU |
| Generalizable | Requires re-fitting per model | Generalizable from model architecture |

---

## 11. Key Files in Repository

For reference, the most implementation-relevant files in the repository:

- `/tmp/xia_finetune/analytical_model/analytical_capacity/plot.py` -- Max batch size model implementation with actual C0=62, C1=0.89
- `/tmp/xia_finetune/analytical_model/analytical_throughput/plot.py` -- Throughput model with curve fitting, raw data for all 4 model-dataset combinations on A40
- `/tmp/xia_finetune/analytical_model/analytical_gpu/plot.py` -- Throughput model validation on A100-40GB, A100-80GB, H100 with raw data
- `/tmp/xia_finetune/analytical_model/analytical_yuhan/plot.py` -- Early development version with explicit coefficient labels
- `/tmp/xia_finetune/LLaMA-Factory/throughput.py` -- How throughput is measured (1000 queries / runtime)
- `/tmp/xia_finetune/LLaMA-Factory/mixtral_tp.sh` -- Training configuration (QLoRA rank=16, 4-bit, bf16, flash_attn, targets=w1,w2,w3,gate)
- `/tmp/xia_finetune/analytical_model/data/tradeoff.csv` -- Full convergence accuracy data across all 10 epochs

---

## 12. Summary Assessment for Our Spec

**Overall value: LOW-MODERATE for formulas, HIGH for empirical data.**

The paper's analytical model is semi-empirical and requires per-model calibration, which is not what a general-purpose calculator needs. However, it provides:

1. Valuable empirical throughput data across multiple GPUs and configurations
2. The MoE sparsity-memory relationship (activation memory scales with `((1-C1) + C1*sparsity)`)
3. Confirmation that throughput scales logarithmically with batch size (diminishing returns)
4. Cloud GPU pricing benchmarks
5. Expert load imbalance data for MoE fine-tuning
6. QLoRA memory verification (4-bit quantization: bytes_per_param = 0.5)

The paper does NOT contribute first-principles memory formulas, FLOP counting, multi-GPU formulas, or communication overhead models that are the core of our calculator spec.
