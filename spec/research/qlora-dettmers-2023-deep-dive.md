# Deep Dive: QLoRA — Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)

**Paper**: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
**Published**: NeurIPS 2023
**Authors**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (University of Washington)
**Code**: [github.com/artidoro/qlora](https://github.com/artidoro/qlora) and [github.com/TimDettmers/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

---

## 1. Executive Summary

QLoRA reduces the average memory requirements of finetuning a 65B parameter model from >780 GB of GPU memory to <48 GB without degrading runtime or predictive performance compared to a 16-bit fully finetuned baseline. It achieves this through three innovations:

1. **4-bit NormalFloat (NF4)** — an information-theoretically optimal quantization data type for normally distributed weights
2. **Double Quantization (DQ)** — quantizing the quantization constants themselves, saving 0.373 bits per parameter (~3 GB for a 65B model)
3. **Paged Optimizers** — using NVIDIA unified memory to handle GPU memory spikes by automatically paging optimizer states to CPU

The key insight: the frozen base model is stored in 4-bit NF4, but all computation (forward pass, backward pass) happens in BFloat16 after dequantization. Only the LoRA adapter weights are trainable and receive gradients/optimizer states.

---

## 2. NF4 (4-bit NormalFloat) Quantization

### 2.1 Core Concept

NF4 is based on **Quantile Quantization** — it ensures each quantization bin has an equal number of values assigned from the input tensor. Since pretrained neural network weights are approximately normally distributed (verified by Shapiro-Wilk test on 7B LLaMA — only 7.5% of neurons are non-normally distributed at 5% significance threshold, about 2.5% above the false-positive rate), the optimal quantization levels can be precomputed from the standard normal distribution N(0,1).

### 2.2 NF4 Construction Formula (Paper Equation 4)

The 2^k quantization values q_i for a k-bit NormalFloat data type are computed as:

```
q_i = (1/2) * (Q_X(i / (2^k + 1)) + Q_X((i + 1) / (2^k + 1)))
```

Where Q_X(.) is the quantile function (inverse CDF) of the standard normal distribution N(0,1).

To ensure a discrete zero representation (important for zero-initialized padding and other zero-valued elements), NF4 uses an **asymmetric** construction:
- 2^(k-1) = 8 quantization ranges for the negative part
- 2^(k-1) + 1 = 9 quantization ranges for the positive part
- Unify both sets and remove one of the duplicate zeros
- Result: 16 values total (8 negative + 0 + 7 positive values)

### 2.3 Exact NF4 Values (from Appendix H)

The 16 exact NF4 quantization levels are:

```
[-1.0, -0.6961928009986877, -0.5250730514526367,
 -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
 -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
 0.24611230194568634, 0.3379152417182922, 0.44070982933044434,
 0.5626170039176941, 0.7229568362236023, 1.0]
```

Distribution: 8 negative values + 0 + 7 positive values = 16 total values.

### 2.4 Bytes per Parameter (NF4, No Double Quantization)

Each parameter is stored as a 4-bit index into the NF4 lookup table. Two 4-bit values are packed into one uint8 byte for PyTorch compatibility.

```
Base storage = 4 bits / parameter = 0.5 bytes / parameter
```

### 2.5 Quantization Constants Overhead (Without Double Quantization)

Block-wise quantization divides the weight tensor into contiguous blocks of size B (default B=64 in bitsandbytes). Each block has one quantization constant (absmax value) stored as FP32 (32 bits):

```
Quantization constant overhead = 32 bits / B = 32/64 = 0.5 bits per parameter
```

**Total without DQ** = 4 + 0.5 = **4.5 bits per parameter = 0.5625 bytes per parameter**

In bytes:
```
M_base_nf4 = Psi * (4/8 + 4/64) = Psi * (0.5 + 0.0625) = 0.5625 * Psi bytes
```

### 2.6 Quantization and Dequantization Formulas

**Quantization** (Paper Equation 1):
```
X^Int8 = round((127 / absmax(X^FP32)) * X^FP32) = round(c^FP32 * X^FP32)
```
Where c is the quantization constant (quantization scale).

**Dequantization** (Paper Equation 2):
```
dequant(c^FP32, X^Int8) = X^Int8 / c^FP32 = X^FP32
```

For NF4, the quantized values are 4-bit indices into the NF4 lookup table rather than raw integers, but the scaling by absmax is the same.

**Full QLoRA dequantization with double quantization** (Paper Equation 6):
```
doubleDeQuant(c_1^FP32, c_2^k-bit, W^k-bit) = dequant(dequant(c_1^FP32, c_2^k-bit), W^4bit) = W^BF16
```

This means: first dequantize the second-level quantization constants c_2 back to FP32, then use them to dequantize the weights.

---

## 3. Double Quantization (DQ)

### 3.1 The Problem

With block size 64, each block requires one FP32 quantization constant. This adds 32/64 = 0.5 bits per parameter overhead. For a 65B model: 0.5 * 65e9 / 8 = ~4.06 GB just for quantization constants.

### 3.2 How Double Quantization Works

Double Quantization treats the FP32 quantization constants c_1^FP32 from the first quantization as inputs to a second quantization:

1. Group the FP32 quantization constants into blocks of size 256
2. Since quantization constants are always positive, **subtract the mean** from each block to center values around zero (enabling symmetric quantization)
3. Quantize these centered constants to FP8 (8-bit Floats) — the second-level quantization constants c_2^FP8
4. Store the second-level constants c_1^FP32 (one per 256 blocks of the first level)

### 3.3 Memory Savings Formula

**Before DQ**: 32/64 = 0.5 bits per parameter (for first-level FP32 quantization constants)

**After DQ**:
```
Second-level quant constants (FP8):  8/64 = 0.125 bits per parameter
Third-level quant constants (FP32):  32/(64 * 256) = 0.001953... bits per parameter

Total DQ overhead = 8/64 + 32/(64 * 256) = 0.125 + 0.00195 = 0.127 bits per parameter
```

**Savings = 0.5 - 0.127 = 0.373 bits per parameter**

For a 65B model: 0.373 * 65e9 / 8 = **~3.03 GB saved**

### 3.4 Total Bytes Per Parameter (NF4 + Double Quantization)

```
NF4 weights:          4.000 bits
DQ overhead:          0.127 bits
                    --------
Total:                4.127 bits per parameter = 0.5159 bytes per parameter
```

Rounding for practical use: **~0.52 bytes per parameter** or approximately **4.13 bits per parameter**.

The spec's existing approximation of "~0.5 bytes per parameter + ~0.01 * Psi overhead" maps to:
- Base: 0.5 bytes/param (the 4-bit NF4 weights)
- Overhead: ~0.016 bytes/param (the DQ quantization constants = 0.127 bits = 0.0159 bytes)

More precisely:
```
M_base_qlora = Psi * (4 + 0.127) / 8 = Psi * 0.5159 bytes
```

Or equivalently: **M_base_qlora ≈ 0.52 * Psi bytes**

### 3.5 Compression Ratio

Compared to FP16 (2 bytes/param):
```
Compression ratio = 2.0 / 0.5159 = 3.877x
```

This is slightly less than the theoretical 4x due to quantization constant overhead. Without DQ, the ratio is 2.0 / 0.5625 = 3.556x. The commonly cited "3.76x" compression ratio corresponds to NF4 without DQ but with the absmax overhead counted as FP16 (16-bit) instead of FP32.

---

## 4. Paged Optimizers

### 4.1 Mechanism

Paged Optimizers use the **NVIDIA unified memory** feature (`cudaMallocManaged`) which enables automatic page-to-page transfers between CPU and GPU for error-free GPU processing when the GPU occasionally runs out of memory.

From the paper: "The feature works like regular memory paging between CPU RAM and the disk. We use this feature to allocate paged memory for the optimizer states which are then automatically evicted to CPU RAM when the GPU runs out-of-memory and paged back into GPU memory when the memory is needed in the optimizer update step."

### 4.2 When They Matter

Paged optimizers are critical for fitting 33B and 65B QLoRA tuning on a single 24 GB and 48 GB GPU respectively. Memory spikes occur when processing a mini-batch with a long sequence length during gradient checkpointing.

The paper notes: "paged optimizers provide the same training speed as regular optimizers" with batch size 16. Training speed is identical for typical configurations; slowdowns only occur during actual page-in/page-out events.

### 4.3 Memory Impact

Paged optimizers do NOT reduce peak theoretical memory — they provide a safety valve by offloading optimizer states to CPU when GPU memory spikes temporarily exceed capacity. The effect is that training does not OOM even when transient peaks exceed GPU VRAM.

For our calculator: paged optimizers should be modeled as enabling the user to slightly exceed GPU VRAM limits (by ~10-20%) without OOM, at the cost of potential throughput reduction during spike events. The calculator should note that paged optimizers are available and can handle transient spikes.

---

## 5. QLoRA Forward Pass Definition

The paper defines the QLoRA forward pass for a single linear layer (Paper Equation 5):

```
Y^BF16 = X^BF16 * doubleDeQuant(c_1^FP32, c_2^k-bit, W^NF4) + X^BF16 * L_1^BF16 * L_2^BF16
```

Key points:
- Storage data type: 4-bit NF4 (for base model W)
- Computation data type: BF16 (for all forward/backward computation)
- Weight W^NF4 is dequantized to W^BF16 before the matmul
- LoRA adapters L_1 and L_2 are stored and computed in BF16
- Gradients are computed only for the LoRA parameters, not for the base weights W

For parameter updates: only dE/dL (gradients w.r.t. adapter weights) are needed, not dE/dW (gradients w.r.t. 4-bit weights). However, computing dE/dL requires dX/dW, which involves dequantizing W^NF4 to W^BF16 during the backward pass.

---

## 6. LoRA Configuration Details

### 6.1 Hyperparameter Search Space (Appendix C.2)

The paper searched over:
- **LoRA dropout**: {0.0, 0.05, 0.1}
- **LoRA rank r**: {8, 16, 32, 64, 128, 256}
- **LoRA target modules**: {key+query, all attention layers, all FFN layers, attention + FFN output layers, **all layers**}
- **LoRA alpha**: kept fixed (alpha is always proportional to learning rate)

### 6.2 Key Finding: Apply LoRA to ALL Linear Layers

This is one of the paper's most important practical findings (Appendix D.2 and Figure 5):

> "When using the standard practice of applying LoRA to query and value attention projection matrices [28], we are not able to replicate full finetuning performance for large base models. We find that the most critical LoRA hyperparameter is how many LoRA adapters are used in total and that LoRA on all linear transformer block layers are required to match full finetuning performance."

This means applying LoRA to: **Q, K, V, O (attention) + gate_proj, up_proj, down_proj (SwiGLU FFN)** = 7 modules per layer for LLaMA-style models.

### 6.3 Rank Does Not Matter (When LoRA is on All Layers)

From Figure 4 and the paper: "LoRA r is unrelated to final performance if LoRA is used on all layers." The performance of specific LoRA r values appears to be independent of other hyperparameters. This was tested with r = {8, 16, 32, 64} on LLaMA 7B.

### 6.4 Exact Hyperparameters Used (Table 8 and Appendix D.3)

For the main Guanaco experiments:
- **LoRA r = 64**
- **LoRA alpha = 16**
- **LoRA modules**: all linear layers of the base model
- **LoRA dropout**: 0.1 for 7B/13B, 0.05 for 33B/65B
- **Quantization**: NF4 with double quantization
- **Compute dtype**: BF16
- **Optimizer**: Adam with beta2 = 0.999
- **Max grad norm**: 0.3
- **Learning rate**: 2e-4 (7B/13B), 1e-4 (33B/65B)
- **Schedule**: constant learning rate
- **Batch size**: 16 (7B/13B), 32 (33B), 64 (65B "All" dataset)

For T5 models on Super-Natural Instruction data (Appendix C.3):
- LoRA r = 16 for small/medium/large T5
- LoRA r = 64 for T5 xl and xxl models
- LoRA alpha = 64 in all experiments
- No LoRA dropout

---

## 7. Memory Breakdown (Figure 8 — Critical Data)

Figure 8 provides the exact memory breakdown for QLoRA training of LLaMA models at batch size 1, sequence length 512, with gradient checkpointing. All numbers are in MB.

### 7.1 Memory Breakdown Table (from Figure 8)

| Component        | 7B (6.9 GB total) | 13B (11.3 GB total) | 33B (24.7 GB total) | 65B (45.0 GB total) |
|------------------|--------------------|----------------------|----------------------|----------------------|
| **Model (NF4)**  | 5,046 MB           | 8,476 MB             | 19,302 MB            | 37,074 MB            |
| **Adapters**     | 298 MB             | 450 MB               | 897.5 MB             | 1,440 MB             |
| **Weight gradient** | (included in input gradient) | (included) | (included) | (included) |
| **Optimizer**    | 1,152 MB           | 1,800 MB             | 3,510 MB             | 5,760 MB             |
| **Input gradient** | (shown separately) | (shown separately) | (shown separately) | (shown separately) |
| **Total**        | ~6,900 MB          | ~11,300 MB           | ~24,700 MB           | ~45,000 MB           |

Note: The "input gradient" component is the activation gradient memory; "weight gradient" is the gradient for LoRA adapter weights. With gradient checkpointing, the input gradients are relatively small (they dominate only as a fraction of the adapter component).

### 7.2 Key Observations from Figure 8

1. The **base model in NF4** is by far the dominant memory consumer (73-82% of total)
2. **Optimizer states** are the second largest component (13-16%)
3. **LoRA adapters** are a small fraction (4-6%)
4. **Activation/gradient memory** is small with gradient checkpointing at batch size 1, seq_len 512

### 7.3 Validation of Per-Parameter Bytes

For the 7B model (6.7B parameters):
- Model storage: 5,046 MB / 6.7B params = 0.753 bytes/param -> This is higher than the 0.52 bytes/param theoretical. The extra may include PyTorch tensor metadata overhead and alignment.
- Actually, LLaMA 7B has 6.738B parameters. 5046 MB = 5.046 GB. 5.046 / 6.738 = 0.749 bytes/param. This likely includes the non-quantized components (e.g., layernorm parameters, embedding layer which may not be quantized to NF4).

For the 65B model:
- Model storage: 37,074 MB / 65B params = 0.570 bytes/param -> closer to the theoretical 0.52 bytes/param, since the non-quantized components (layernorm, embeddings) become a smaller fraction at larger scales.

### 7.4 Implied Adapter Sizes

For LLaMA 7B with LoRA r=64 on all linear layers:
- Per LLaMA 7B layer: d=4096, d_ff=11008, 7 modules (Q,K,V,O,gate,up,down)
- Adapter params per layer: 64 * (11 * 4096 + 3 * 11008) = 4,997,120
- Total adapter params (32 layers): 4,997,120 * 32 = 159,907,840 ≈ 160M
- In BF16: 160M * 2 = 320 MB (close to the 298 MB figure, with differences likely from exact target modules and implementation details)

For LLaMA 65B with LoRA r=64:
- d=8192, d_ff=22016, 80 layers, 7 modules per layer
- Adapter params per layer: 64 * (11 * 8192 + 3 * 22016) = 9,994,240
- Total: 9,994,240 * 80 = 799,539,200 ≈ 800M
- In BF16: 800M * 2 = 1,599 MB (vs. 1,440 MB in Figure 8, with differences likely from exact target modules and implementation details)

---

## 8. Memory Footprint from Appendix J

Appendix J provides critical detail on memory composition:

> "For a 7B LLaMA model trained on FLAN v2 with a batch size of 1, with LoRA weights equivalent to commonly used 0.2% of the original model weights [28, 37], the LoRA input gradients have a memory footprint of **567 MB** while the LoRA parameters take up only **26 MB**."

> "With gradient checkpointing [9], the input gradients reduce to an average of **18 MB** per sequence making them more memory intensive than all LoRA weights combined."

> "In comparison, the 4-bit base model consumes **5,048 MB** of memory."

### 8.1 Implications for the Calculator

The paper explicitly states that for LoRA/QLoRA:
- **Most memory comes from activation gradients**, not from LoRA parameters
- With gradient checkpointing, input gradients drop dramatically (567 MB -> 18 MB per sequence for 7B)
- LoRA parameter memory is negligible compared to the base model
- "Aggressively reducing the amount of LoRA parameter yields only minor memory benefits"
- "This means we can use more adapters without significantly increasing the overall training memory footprint"

---

## 9. Activation Memory Details

### 9.1 Are Activations in BF16?

**Yes.** The paper explicitly states (Section 3, page 4):

> "QLoRA has one low-precision storage data type, in our case usually 4-bit, and one computation data type that is usually BFloat16. In practice, this means whenever a QLoRA weight tensor is used, we dequantize the tensor to BFloat16, and then perform a matrix multiplication in 16-bit."

All activations during the forward and backward pass are computed and stored in BF16 (2 bytes per element), identical to standard 16-bit training. The 4-bit quantization only affects the **storage** of frozen base model weights, not the working precision of computations.

### 9.2 Gradient Checkpointing

All QLoRA experiments use gradient checkpointing. The paper mentions this is essential for fitting large models:

> "We use NF4 QLoRA with double quantization and paged optimizers to prevent memory spikes during gradient checkpointing."

The activation memory formulas from Korthikanti et al. (2022) apply unchanged since all activations are in BF16. With gradient checkpointing, activation memory is reduced from the full per-layer storage to just storing layer inputs (2 * s * b * d bytes per checkpointed layer).

---

## 10. Model Size vs GPU Memory Table (from Table 4 and Table 12)

### 10.1 QLoRA (4-bit) Memory Footprint

From Table 4 (which shows the "Size" column = total GPU memory used during inference/deployment, not training):

| Model    | Params | Model Bits | GPU Memory |
|----------|--------|------------|------------|
| Guanaco  | 65B    | 4-bit      | 41 GB      |
| Guanaco  | 33B    | 4-bit      | 21 GB      |
| Vicuna   | 13B    | 16-bit     | 26 GB      |
| Guanaco  | 13B    | 4-bit      | 10 GB      |
| Guanaco  | 7B     | 4-bit      | 5 GB       |

These are **inference** memory numbers (no optimizer states, no gradients). For training, add optimizer states, adapters, and activation gradients as shown in Figure 8.

### 10.2 Full Fine-tuning Memory (Referenced)

The paper states that "regular 16-bit finetuning of a LLaMA 65B parameter model requires more than **780 GB** of GPU memory."

Verification: 65B * 16 bytes/param (fp32 master + bf16 params + fp32 gradients + fp32 Adam m + fp32 Adam v = 2+4+4+4+4 = 18 bytes) = 1,170 GB for model states alone. The 780 GB figure likely assumes bf16 gradients (16 bytes/param): 65B * 16 = 1,040 GB, or possibly bf16 gradients without master weights in some configurations. The >780 GB figure is conservative and likely accounts for a simpler memory model.

---

## 11. NF4 vs Other 4-bit Data Types (Table 2 and Table 3)

### 11.1 Perplexity Comparison (Table 2)

Pile Common Crawl mean perplexity for 125M to 13B OPT, BLOOM, LLaMA, and Pythia models:

| Data Type       | Mean PPL |
|-----------------|----------|
| Int4            | 34.34    |
| Float4 (E2M1)  | 31.07    |
| Float4 (E3M0)  | 29.48    |
| **NFloat4 + DQ** | **27.41** |

NF4 with double quantization significantly outperforms all other 4-bit formats.

### 11.2 MMLU Accuracy (Table 3)

Mean 5-shot MMLU test accuracy for LLaMA 7-65B models finetuned with adapters on Alpaca and FLAN v2:

| Data Type   | 7B  | 13B  | 33B  | 65B  | Mean |
|-------------|-----|------|------|------|------|
| BFloat16    | 38.4-45.6 | 47.2-50.6 | 57.7-60.5 | 61.8-62.5 | **53.0** |
| Float4      | 37.2-44.0 | 47.3-50.0 | 55.9-58.5 | 61.3-63.3 | 52.2 |
| NFloat4 + DQ | 39.0-44.5 | 47.5-50.7 | 57.3-59.2 | 61.8-63.9 | **53.1** |

NF4 with DQ fully recovers BFloat16 (16-bit) LoRA MMLU performance. Float4 lags by about 1 percentage point.

---

## 12. Training Throughput and Speed

### 12.1 Paged Optimizer Throughput

From the paper (page 5-6):

> "We do, however, perform an analysis of the runtime of paged optimizers for 65B models on 48GB GPUs and find that with a batch size of 16, paged optimizers provide the same training speed as regular optimizers."

Paging slowdowns only occur when processing mini-batches with long sequence lengths, which is rare with typical training configurations.

### 12.2 QLoRA vs Full Fine-tuning Wall Clock

The paper does not provide direct wall-clock comparisons between QLoRA and 16-bit full fine-tuning or 16-bit LoRA. However:

- **Guanaco 65B** (the best model) was trained in **24 hours on a single professional GPU** (48 GB, likely A6000 or similar)
- **Guanaco 33B** can be trained on 24 GB consumer GPUs **in less than 12 hours**
- These are on the OASST1 dataset (9,846 examples)

### 12.3 Throughput Penalty from Dequantization

The dequantization overhead (NF4 -> BF16 before each matmul) adds computation to every forward and backward pass. External benchmarks (not from the paper itself) report approximately **1.3-1.75x wall-clock time** compared to equivalent 16-bit LoRA fine-tuning, depending on model size and hardware. The paper itself states training speed matches regular optimizers at batch size 16, suggesting the overhead is modest for practical configurations.

### 12.4 NF4 Inference Speedup (Appendix A, Figure 3)

For single-batch inference comparing NF4 to 16-bit float:
- **RTX 4090**: 3.5-4.0x speedup across 7B-65B
- **A40/RTX 3090**: 2.9-3.3x speedup
- **A100**: 1.1-1.5x speedup
- **RTX 2080 Ti**: 1.2-1.3x speedup
- **RTX 6000**: 1.2-1.5x speedup

The large speedup on RTX 3090/4090/A40 is due to the memory bandwidth bottleneck being more severe on consumer GPUs. A100 sees less speedup due to its higher memory bandwidth.

---

## 13. Loading Memory Floor

### 13.1 The Problem

The paper does NOT explicitly discuss a "loading memory floor" where the model must be loaded in fp16/bf16 before quantization. However, this is a known practical constraint in the bitsandbytes implementation:

When loading a model with `load_in_4bit=True` in HuggingFace Transformers:
1. The model is first loaded in fp16/bf16 on CPU
2. Each layer is then quantized to NF4 and moved to GPU
3. The transient peak CPU memory is ~2*Psi bytes

For GPU memory, the loading is done layer-by-layer, so the peak GPU memory during loading is approximately the NF4 model size plus one layer's worth of fp16 weights being quantized:

```
M_loading_peak_gpu ≈ M_nf4_total + max_layer_params * 2 bytes
```

This is much less than loading the entire model in fp16 on GPU. The spec's existing note about "max(M_total_qlora, 2*Psi)" as the loading floor is overly conservative for GPU memory — the layer-by-layer loading means the GPU peak is closer to M_nf4 + one_layer_fp16. The 2*Psi figure applies to **CPU memory**, not GPU memory.

### 13.2 Corrected Loading Floor

For the calculator:
- **CPU memory floor**: 2 * Psi bytes (full model in fp16 on CPU before quantization)
- **GPU memory floor during loading**: M_nf4 + max(Psi_layer * 2) (one layer in fp16 being quantized at a time)
- **GPU memory floor during training**: M_nf4 + adapter_states + activation_memory (the steady-state QLoRA memory)

---

## 14. Block Size Details

### 14.1 First-Level Block Size

- **CUDA implementations**: block size = **64** elements
- **ROCm (warp64) implementations**: block size = **128** elements
- This is the smallest block size used in bitsandbytes

Each block of 64 parameters shares one quantization constant (absmax value).

### 14.2 Second-Level Block Size (Double Quantization)

- Block size = **256** for the second quantization of quantization constants
- Quantization constants are quantized from FP32 to **FP8** (8-bit Floats)
- No performance degradation observed for 8-bit quantization of constants (per Dettmers and Zettlemoyer, 2022)

### 14.3 Memory Overhead at Different Block Sizes

| Block Size (B) | Without DQ (bits/param) | With DQ (bits/param) | Total bits/param |
|----------------|------------------------|---------------------|-----------------|
| 32             | 32/32 = 1.0            | 8/32 + 32/(32*256) = 0.254 | 4.254 |
| 64 (default)   | 32/64 = 0.5            | 8/64 + 32/(64*256) = 0.127 | 4.127 |
| 128            | 32/128 = 0.25          | 8/128 + 32/(128*256) = 0.063 | 4.063 |
| 256            | 32/256 = 0.125         | 8/256 + 32/(256*256) = 0.031 | 4.031 |

Smaller block sizes give finer-grained quantization (better quality) but higher overhead. Block size 64 is the sweet spot used by default.

---

## 15. Total Training Memory Formula

### 15.1 Complete QLoRA Memory Model

```
M_total_qlora = M_base_model + M_lora_params + M_lora_gradients + M_lora_optimizer + M_activations + M_framework

Where:
  M_base_model       = Psi * (4 + 0.127) / 8  bytes       (NF4 + DQ overhead; ≈ 0.52 * Psi)
  M_lora_params      = Psi_lora * 2            bytes       (BF16 adapter weights)
  M_lora_gradients   = Psi_lora * 2            bytes       (BF16 gradients for adapters)
  M_lora_optimizer   = Psi_lora * 12           bytes       (fp32 master + Adam m + Adam v)
  M_activations      = standard BF16 activation memory     (same as full model, see Section 5.3 of spec)
  M_framework        = ~2-5 GB                             (CUDA context, allocator overhead)

Simplified:
  M_total_qlora ≈ 0.52 * Psi + 16 * Psi_lora + M_activations + M_framework
```

### 15.2 Adapter Parameter Count

For a LLaMA-style model with SwiGLU and LoRA on all linear layers:

```
Psi_lora = r * sum(input_dim + output_dim for each adapted matrix copy)

Where:
  r = LoRA rank
  d = hidden dimension
  d_ff = feed-forward intermediate dimension
  L = number of layers
```

For MHA models with attention-only targets:
```
Psi_lora_attention_mha = L * r * 8*d
```

For MHA models with all-linear SwiGLU targets:
```
Psi_lora_swiglu_mha = L * r * (11*d + 3*d_ff)
```

For GQA models where K and V projections are smaller, with `d_kv = d * a_kv / a`:
```
Psi_lora_attention_gqa = L * r * (6*d + 2*d_kv)
Psi_lora_swiglu_gqa    = L * r * (9*d + 2*d_kv + 3*d_ff)
```

The common `2 * r * d * M_modules * L` shortcut is exact only when every adapted matrix is `d x d`. It undercounts all-linear SwiGLU adapters because `gate_proj`, `up_proj`, and `down_proj` use `d_ff`.

### 15.3 Worked Example: LLaMA 7B QLoRA

```
Model: LLaMA 7B (Psi = 6.738B, d = 4096, L = 32, d_ff = 11008)
LoRA: r = 64, alpha = 16, all 7 linear layers per block

M_base_model  = 6.738e9 * 0.52 = 3.504 GB
  (Paper shows 5,048 MB ≈ 5.05 GB — difference is non-quantized params like layernorm, embeddings)

Psi_lora = 32 * 64 * (11 * 4096 + 3 * 11008) = 159,907,840 ≈ 160M (2.37% of base)
M_lora_total  = 160M * 16 = 2.56 GB (adapters + grads + optimizer)

M_activations (batch=1, seq=512, gradient checkpointing):
  Per layer: 2 * 512 * 1 * 4096 = 4.2 MB (checkpointed)
  32 layers: 32 * 4.2 MB = 134 MB
  Plus recomputation working memory (1 layer): ~270 MB (34*s*b*d bytes)

Total ≈ 3.5 + 2.56 + 0.4 + ~2.0 (framework) ≈ 8.5 GB
  (Paper: 6.9 GB total; exact parity requires paper-specific implementation details such as target modules, trainable parameter storage, and paged optimizer behavior)
```

### 15.4 Worked Example: LLaMA 65B QLoRA

```
Model: LLaMA 65B (Psi = 65B, d = 8192, L = 80)
LoRA: r = 64, all 7 linear layers

M_base_model  = 65e9 * 0.52 = 33.8 GB
  (Paper shows 37,074 MB ≈ 37.1 GB — includes non-quantized components)

Psi_lora = 80 * 64 * (11 * 8192 + 3 * 22016) = 799,539,200 ≈ 800M
M_lora_total  = 800M * 16 = 12.8 GB

M_activations (batch=1, seq=512, gradient checkpointing):
  Per layer: 2 * 512 * 1 * 8192 = 8.4 MB (checkpointed)
  80 layers / pipeline: 80 * 8.4 = 672 MB

Total ≈ 33.8 + 12.8 + 0.67 + ~2.0 ≈ 49.3 GB
  (Paper: 45.0 GB total; exact parity requires paper-specific implementation details such as target modules, trainable parameter storage, and paged optimizer behavior)
```

---

## 16. What Is Unique / Non-Obvious

### 16.1 NF4 Requires Normally Distributed Weights

NF4 is optimal ONLY for weights that follow a normal distribution. The paper verified this holds for LLaMA (Appendix I), but models trained with different techniques or architectures that produce non-normal weight distributions would not benefit equally from NF4. The calculator should note this assumption.

### 16.2 Rank Doesn't Matter — Module Coverage Does

Contrary to common practice (applying LoRA to Q/V only with high rank), QLoRA finds that **applying LoRA to all linear layers** is the critical factor, not the rank. This means the calculator's default should apply LoRA to all linear layers (M_modules = 7 for SwiGLU, M_modules = 6 for standard FFN) rather than attention-only (M_modules = 4).

### 16.3 Data Quality > Data Quantity

The paper found that a 9k sample dataset (OASST1) outperformed a 450k sample dataset (FLAN v2, subsampled) on chatbot performance. Differences between datasets were up to 40x larger (1.5-8.0 MMLU points) than differences from increasing dataset size or epochs (0.0-0.5 MMLU points).

### 16.4 Non-Quantized Components

Not all parameters in the model are quantized to NF4. Typically, layernorm parameters, embedding layers, and the output head remain in their original precision (BF16 or FP32). This explains why the actual model memory in Figure 8 is higher than the theoretical 0.52 * Psi calculation. The calculator should account for this:

```
M_base_actual = Psi_quantized * 0.52 + Psi_non_quantized * 2.0

Where Psi_non_quantized includes:
  - Token embeddings: V * d (e.g., 32000 * 4096 = 131M for LLaMA 7B)
  - Output projection: V * d (if untied)
  - LayerNorm/RMSNorm: L * 2 * d (typically negligible)
```

For LLaMA 7B: Psi_non_quantized ≈ 131M * 2 (if embeddings untied) = ~262M. In BF16: 524 MB.
Quantized: (6738M - 262M) * 0.52 ≈ 3,367 MB.
Non-quantized: 262M * 2 = 524 MB.
Total model: 3,367 + 524 = 3,891 MB ≈ 3.9 GB.

Paper shows 5,048 MB, which suggests additional overhead from PyTorch tensor bookkeeping, alignment, and potentially some parameters stored at higher precision.

### 16.5 The 33B Model Does NOT Quite Fit on 24 GB Without Paged Optimizers

From Figure 8 and Appendix J: "We see that the 33B model does not quite fit into a 24 GB and that paged optimizers are needed to train it." The 33B model at batch size 1 and seq length 512 uses 24.7 GB total, which exceeds 24 GB available VRAM. Paged optimizers handle the overflow by spilling to CPU.

### 16.6 Double Quantization Has Negligible Impact on Quality

Table 2 and Table 3 confirm that adding double quantization does NOT degrade performance — NFloat4 + DQ achieves the same or better perplexity and MMLU accuracy as NFloat4 alone. This makes DQ a pure memory optimization with no quality tradeoff.

### 16.7 8-bit Quantization (QLoRA Int8) Also Works

Table 7 shows QLoRA with Int8 quantization also matches 16-bit LoRA and full finetuning performance, but with only 2x compression instead of 4x. For the calculator, this suggests supporting 8-bit QLoRA as well:
```
M_base_int8 = Psi * 1.0 bytes  (8-bit weights, plus quantization constant overhead)
```

---

## 17. Spec Change Recommendations

### 17.1 Refine the QLoRA Memory Formula

The existing spec (Section 10.1) has:
```
M_total_qlora ≈ 0.55*Psi + 16 * Psi_lora + M_activations
```

This should be refined to:
```
M_total_qlora = M_base_nf4 + 16 * Psi_lora + M_activations + M_framework

Where M_base_nf4 is computed as:
  Psi_quantized = Psi - Psi_non_quantized
  Psi_non_quantized = V * d * (1 + untied) + L * n_norm_params  (embeddings + layernorms)
  
  Without DQ: M_base_nf4 = Psi_quantized * 0.5625 + Psi_non_quantized * 2.0
  With DQ:    M_base_nf4 = Psi_quantized * 0.5159 + Psi_non_quantized * 2.0
```

For Quick Mode, the simplified 0.55*Psi approximation is acceptable (it includes non-quantized overhead implicitly). For Detailed Mode, the component-based formula above is more accurate.

### 17.2 Add Double Quantization Toggle

The calculator should expose a "Double Quantization" toggle (default ON) that switches between:
- DQ ON: 4.127 bits/param = 0.516 bytes/param (quantized portion)
- DQ OFF: 4.5 bits/param = 0.5625 bytes/param (quantized portion)

DQ saves approximately 0.373 bits per parameter = 0.0466 bytes per parameter. For a 65B model: ~3 GB savings.

### 17.3 Expand LoRA Module Defaults

Change the default LoRA module count from 4 (attention only) to **7** (all linear layers for SwiGLU) for QLoRA configurations, based on the paper's finding that all-linear-layer LoRA is critical for matching full finetuning performance. The existing spec already mentions M_modules=7 as "common for SwiGLU models" but should note it is strongly recommended by the QLoRA paper.

### 17.4 Add Paged Optimizer Note

Add to the QLoRA section: "Paged optimizers (using NVIDIA unified memory) can handle transient memory spikes by automatically paging optimizer states to CPU RAM. This allows training when peak memory briefly exceeds GPU VRAM (typically by up to 10-20%) at the cost of occasional throughput reduction. Available only on NVIDIA GPUs with CUDA unified memory support."

### 17.5 Update QLoRA Throughput Penalty

The existing spec says "approximately 1.75x wall-clock time compared to equivalent LoRA fine-tuning." The QLoRA paper itself does not report this number. External benchmarks suggest 1.3-1.75x depending on hardware and configuration. The 1.75x figure is a reasonable conservative estimate. Consider noting the range: "1.3-1.75x wall-clock time compared to equivalent BF16 LoRA fine-tuning, depending on hardware (consumer GPUs see less overhead than datacenter GPUs due to being more memory-bandwidth-bound)."

### 17.6 Correct Loading Memory Floor

The current spec states: "The effective minimum GPU memory for QLoRA is therefore max(M_total_qlora, 2*Psi)."

This is incorrect for GPU memory. The model is loaded layer-by-layer, so the GPU never needs to hold the full fp16 model. The correct loading floor is:
- **CPU memory**: 2 * Psi (full model in fp16 before quantization)
- **GPU memory during loading**: M_nf4 + max_layer_size_fp16 (one layer being quantized at a time)
- **GPU memory during training**: M_total_qlora (the steady-state formula)

The loading floor on GPU is NOT 2*Psi. It is closer to M_nf4 + one_layer_fp16. For a 7B model: ~3.5 GB + ~0.4 GB = ~3.9 GB GPU during loading, not 14 GB.

### 17.7 Add NF4 Exact Values as Constant

The calculator should include the 16 NF4 quantization levels as a reference constant (for documentation and potential visualization):

```typescript
const NF4_LEVELS = [
  -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,
  0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
];
```

### 17.8 Add Block Size Parameter

The calculator should include quantization block size as an advanced parameter:
- Default: 64 (CUDA)
- Alternative: 128 (ROCm)

The overhead formula generalizes as:
```
Without DQ: overhead = 32 / block_size  bits per parameter
With DQ:    overhead = 8 / block_size + 32 / (block_size * 256)  bits per parameter
```

---

## 18. Summary of All Numbers

### Bytes Per Parameter

| Component | Bytes/Param |
|-----------|-------------|
| NF4 weights (no overhead) | 0.500 |
| NF4 + quantization constants (no DQ, B=64) | 0.5625 |
| NF4 + DQ (B=64, default) | 0.5159 (~0.52) |
| Non-quantized components (embeddings, layernorm) | 2.0 (BF16) |
| LoRA adapter parameters (BF16) | 2.0 |
| LoRA adapter gradients (BF16) | 2.0 |
| LoRA optimizer (fp32 master + Adam m + v) | 12.0 |
| **LoRA total (params + grads + optimizer)** | **16.0** |

### Memory for LLaMA Models (QLoRA, batch=1, seq=512, gradient checkpointing)

| Model | Total Memory (Paper) | Base Model | Adapters | Optimizer | Gradients |
|-------|---------------------|------------|----------|-----------|-----------|
| 7B    | 6.9 GB              | 5,046 MB   | 298 MB   | 1,152 MB  | Included  |
| 13B   | 11.3 GB             | 8,476 MB   | 450 MB   | 1,800 MB  | Included  |
| 33B   | 24.7 GB             | 19,302 MB  | 897 MB   | 3,510 MB  | Included  |
| 65B   | 45.0 GB             | 37,074 MB  | 1,440 MB | 5,760 MB  | Included  |

### Key Quantization Parameters

| Parameter | Value |
|-----------|-------|
| NF4 quantization bit-width | 4 bits |
| NF4 quantization levels | 16 (8 neg + 0 + 7 pos) |
| First-level block size | 64 (CUDA) / 128 (ROCm) |
| First-level quantization constant | FP32 (32 bits) |
| Second-level block size (DQ) | 256 |
| Second-level quantization constant | FP8 (8 bits) |
| DQ overhead per parameter | 0.127 bits |
| DQ savings per parameter | 0.373 bits |
| DQ savings for 65B model | ~3 GB |

### LoRA Configuration (Paper's Defaults)

| Parameter | Value |
|-----------|-------|
| Rank r | 64 |
| Alpha | 16 |
| Target modules | All linear layers (7 for SwiGLU) |
| Dropout | 0.1 (7B/13B), 0.05 (33B/65B) |
| Compute dtype | BF16 |
| Quantization | NF4 + Double Quantization |
