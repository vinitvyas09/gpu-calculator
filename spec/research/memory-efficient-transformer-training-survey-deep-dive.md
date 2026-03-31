# Deep Dive: A Survey on Memory-Efficient Transformer-Based Model Training in AI for Science

**Paper**: Tian et al., "A survey on memory-efficient transformer-based model training in AI for science" (2025)
**ArXiv**: https://arxiv.org/abs/2501.11847v2
**Published**: Frontiers of Computer Science, 2025, DOI: 10.1007/s11704-025-50302-6
**Authors**: Kaiyuan Tian, Linbo Qiao, Baihui Liu, Gongqingjian Jiang, Shanshan Li, Dongsheng Li (NUDT, China)

---

## Executive Summary

This is a survey paper (not a source of new formulas) that systematically categorizes memory-efficient training techniques across three levels: algorithm, system, and hardware-software co-optimization. Its primary value to the GPU calculator spec is:

1. **Validation** of existing spec formulas (model states 16P, ZeRO stages, etc.)
2. **Taxonomy completeness check** identifying a few techniques the spec does not cover
3. **Quantitative benchmarks** from Table 3 that can validate parallelism efficiency estimates
4. **Memory-efficient optimizer landscape** covering techniques beyond AdamW that could expand the spec's optimizer table

The paper does NOT introduce new formulas. It surveys existing work. Most content is already well-covered by the spec. Below I detail what is new, what refines existing content, and what confirms existing content.

---

## 1. Formulas

### 1.1 Model States Memory (ALREADY COVERED)

The paper states the identical breakdown as the spec:

> "the memory overhead of model parameters is 2P + 4P = 6P bytes. The memory overhead of gradients is 2P bytes, and optimizer states consume 4P + 4P = 8P bytes. Thus, the total memory overhead is 16P bytes."

**Status**: Already in spec Section 5.1. The paper uses bf16 grads (total 16P), matching one of the spec's two modes (16P with bf16 grads, 18P with fp32 grads).

### 1.2 ZeRO Memory Reduction (ALREADY COVERED)

The paper states:

> "Rajbhandari et al. introduced Zero Redundancy Optimizer (ZeRO), which shards redundant model states across devices, reducing memory overhead up to 1/N_d"

Table 3 confirms: ZeRO-1 = 4x, ZeRO-2 = 8x, ZeRO-3 = N_d x memory reduction.

**Status**: Already in spec Section 5.2. The 4x/8x numbers for ZeRO-1/2 are specific to a particular configuration (16P base, N_dp sufficient). These match the spec's formulas when evaluated at appropriate N_dp.

### 1.3 Activation Memory Complexity (ALREADY COVERED)

The paper mentions O(n^2) attention memory complexity and references Korthikanti et al. for selective checkpointing.

> "Korthikanti et al. suggest that for transformer models, it is unnecessary to set checkpoints for the entire transformer layer for recomputation. Instead, selective recomputation can be performed by identifying components of the transformer layer where activations cost substantial memory yet require less computation (e.g., the computation of Q x K^T, subsequent softmax, dropout, and attention score multiplies V)."

**Status**: Already in spec Section 5.3 (Korthikanti formula with coefficients 34 and 5as/d).

### 1.4 Adafactor Low-Rank Decomposition (REFINEMENT)

> "Assuming the model parameter matrix is denoted as W in R^{m x n}, by applying low-rank decomposition on the second-order moment estimates of the gradients, Adafactor can reduce memory consumption from O(m * n) to O(m + n)."

**Status**: The spec lists Adafactor at 12 bytes/param in the optimizer table (Section 5.1) but does not explain the low-rank factorization mechanism. This is a useful detail for a tooltip/explanation but does not change the formula.

### 1.5 Pipeline Parallelism Activation Memory (ALREADY COVERED)

> "In GPipe, each device must retain activations of N_m micro-batches. To reduce memory consumption, researchers introduced the 1F1B schedule... the activations of micro-batches required to store on each device decrease from N_m to N_d."

**Status**: Already in spec Section 5.7 (pipeline bubble formula and microbatch constraints).

### 1.6 Tensor Parallelism Partitioning (ALREADY COVERED)

> "W_1 is partitioned column-wise, while W_2 is partitioned row-wise. This results in the parameter matrices stored on each device having the shapes: W_1'(h, 4h/N_d) and W_2'(4h/N_d, h)."

And: "4 all-reduce operations per transformer layer."

**Status**: Already in spec Section 5.6 (TP communication volume: "4 all-reduce operations per training step").

---

## 2. Memory-Saving Techniques

### 2.1 Quantization-Aware Training (QAT) -- PARTIALLY NEW

The paper covers several QAT approaches not in the spec:

**a) BitNet (1-bit weights + quantized activations)**
> "BitNet replaced standard linear layers with BitLinear modules using 1-bit weights and quantized activations... BitNet obeys a scaling law akin to full-precision Transformers."

**b) BitNet b1.58 (ternary weights: -1, 0, +1)**
> "7.2x memory savings and 71.4x energy efficiency... matched or exceeded the performance of FP16 LLaMA models starting from 3B parameters while offering up to 4.1x decoding latency speedup."

**c) LLM-QAT (4-bit weights, 8-bit activations)**
> "7.3x speedup with comparable accuracy... requires no access to the original pre-training data."

**d) EfficientQAT (2-bit on Llama-2-70B)**
> "Peak memory ~ 34 GB, <3 points accuracy degradation compared to the FP16 model."

**Status**: The spec covers FP8 mixed precision (Section 5.1) and QLoRA (Section 10.1) but does NOT cover training-time quantization below FP8 (4-bit, 2-bit, 1-bit weight training). These are primarily relevant for extremely resource-constrained scenarios or inference-optimized training.

**Recommendation for spec**: Add a brief note in Section 5.1 under "FP8 training note" mentioning that sub-FP8 quantized training (BitNet, EfficientQAT) exists but is not widely used for standard pretraining. For the calculator, the primary impact would be a new optimizer/precision mode: "QAT 4-bit" at approximately 0.5-1 byte/param for weights. However, these are experimental and not supported by standard frameworks (DeepSpeed, Megatron-LM), so they may be out of scope.

### 2.2 Memory-Efficient Optimizers -- PARTIALLY NEW

**a) SM3** (Anil et al.)
> "Reduces the memory requirements for storing optimizer states through a parameter cover mechanism. It involves dividing parameters into multiple subsets, with each subset maintaining only one variable to approximate the second-order statistics of all parameters within it."
> "Memory overhead is slightly lower than that of Adafactor."

**Status**: NOT in spec. Could be added to the optimizer table at approximately 10-12 bytes/param.

**b) CAME** (Luo et al.)
> "Uses non-negative matrix factorization to the confidence matrix, CAME further decreases memory overhead... CAME has shown superior performance compared to Adafactor, while being comparable to it in terms of memory efficiency."

**Status**: NOT in spec. Similar memory footprint to Adafactor (~12 bytes/param) but better training stability.

**c) Lion** (Chen et al.)
> "Unlike most adaptive optimizers, Lion solely tracks momentum to calculate updates, leading to reduced memory overhead and a consistent update magnitude."

**Status**: Already in spec Section 5.1 optimizer table at 12 bytes/param.

**d) Adam-mini** (Zhang et al.)
> "Partitions the parameters of components such as Q, K, V, and MLP into blocks and assigns a learning rate to each block, reducing the resource requirement by 90% to 99%... achieving savings of 45% to 50%."

**Status**: NOT in spec. This is a significant finding. Adam-mini achieves near-AdamW performance while halving optimizer memory. The memory model:
```
Adam-mini: 2 + 2 + 4 + ~2 = ~10 bytes/param (vs AdamW's 16-18)
```
The ~2 at the end is because instead of storing per-parameter v (variance) at 4 bytes, it stores per-block variance at much lower cost. The exact savings depend on block structure.

**Recommendation**: Add Adam-mini to the optimizer table in Section 5.1. Conservative estimate: 10-12 bytes/param.

**e) Zeroth-Order Optimizers (MeZO, DeepZero, ZO-AdaMU)** -- NEW CATEGORY

> "MeZO was the first to use a zeroth-order optimizer for fine-tuning memory optimization of LLMs. By introducing a memory-efficient zeroth-order optimizer, the fine-tuning of LLM can be achieved with only forward passes, significantly lowering memory demands. As a comparison, MeZO can train a 3B model on a single A100 80GB GPU, while traditional optimizers can only train a 270M model under the same hardware budget."

Memory model for zeroth-order optimizers:
```
MeZO: 2P bytes (parameters only, no gradients, no optimizer states)
     + small perturbation vector (negligible)
```
This is dramatically lower (2 bytes/param vs 16-18) but at the cost of much slower convergence.

**Status**: NOT in spec. This is an entirely new optimizer category. For the calculator, it would enable training much larger models on limited hardware, but with a convergence penalty that makes it primarily useful for fine-tuning, not pretraining.

**Recommendation**: Add zeroth-order optimizers as a new entry in the optimizer table:

| Optimizer | Bytes/Param | Breakdown | Notes |
|-----------|-------------|-----------|-------|
| MeZO (zeroth-order) | ~2 | 2 (param only) | Forward-only; no grads/optimizer states; very slow convergence; fine-tuning only |

### 2.3 Approximate Attention -- PARTIALLY NEW

**a) Linformer**
> "Reduces attention complexity from O(n^2) to O(n) by projecting the key and value matrices into a lower-dimensional space using learned linear projections."

**b) Performer**
> "Approximates softmax attention through kernel methods, replacing the softmax operation with a feature map transformation. It introduces the FAVOR+ mechanism, enabling linear-time and linear-memory attention."

**Status**: The spec covers Flash Attention (Section 5.3) which is an exact attention optimization. Linformer and Performer are approximate attention methods that change the model architecture, not just the training efficiency. These are NOT applicable to standard transformer training and are out of scope for the calculator (which estimates resources for a given architecture, not recommending architecture changes).

### 2.4 Gradient Accumulation -- ALREADY COVERED

The paper provides pseudocode (Algorithm 1) for gradient accumulation. The spec covers this via the gradient accumulation steps G parameter.

**Status**: Already in spec (G parameter in batch size formula B = b x G x N_dp).

### 2.5 Offloading Methods -- ALREADY COVERED

The paper summarizes three offloading approaches (Table 4):

| Method | Feature | Performance |
|--------|---------|-------------|
| SwapAdvisor | Optimize scheduling, memory allocation | 53-99% of ideal throughput (single-GPU) |
| ZeRO-Offload | CPU for weight updates | 1.62x larger model vs SwapAdvisor; 4.5x vs Megatron; 7.8x vs ZeRO-2 on single DGX-2 |
| ZeRO-Infinity | CPU + NVMe memory | 1T model on single DGX-2 node without MP |

**Status**: Already well-covered in spec Section 5.2 (CPU/NVMe offloading). The SwapAdvisor approach is not in the spec but is a single-GPU optimization that's less relevant for the large-scale training the calculator targets.

### 2.6 MPress -- NEW

> "MPress is a novel system designed to break the GPU memory wall for billion-scale model training on multi-GPU servers. It combines inter-operator parallelism with a new D2D swap technique to efficiently transfer tensors between GPUs, leveraging high-bandwidth NVLink connections. MPress dynamically selects the best memory-saving strategies based on tensor live intervals and available spare memory."
> "MPress can train larger models compared to recomputation baseline (3.7x for BERT and 1.7x for GPT) while maintaining high training throughput, and achieves 1.4-2.3x speedups compared to ZeRO-Series baselines under the same memory reduction."

**Status**: NOT in spec. This is a specialized system optimization that dynamically swaps tensors between GPUs via NVLink during training. It's an advanced optimization that doesn't fit neatly into the calculator's current framework (which assumes static memory allocation). Likely out of scope for v1 but worth noting as a future consideration.

### 2.7 FlashAttention Variants -- ALREADY COVERED with REFINEMENTS

**FlashAttention-1**: "Reduces memory consumption of self-attention from O(n^2) to O(n)... achieving up to 3.5x training time speedup."

**FlashAttention-2**: "2x speedup over FlashAttention and reaches up to 73% of the theoretical peak FLOPS on A100 GPU."

**FlashAttention-3**: "1.5-2.0x speedup over FlashAttention-2 (85% utilization in BF16, 1.3 PFLOP/s in FP8) while reducing FP8 error by 2.6x."

**Status**: The spec covers Flash Attention memory impact (Section 5.3) and mentions FlashAttention. The specific throughput numbers for FA-2 and FA-3 are useful calibration data but don't change formulas.

**New data point for spec**: FlashAttention-3 achieves 85% utilization in BF16 on Hopper architecture. This suggests that attention kernels are NOT the MFU bottleneck on modern hardware.

---

## 3. Taxonomy / Categorization

The paper organizes memory-efficient techniques into three levels (Figure 2):

### Level 1: Algorithm (Section 3.1)
- **Compression**: Mixed Precision Training, Quantization-Aware Training
- **Memory Efficient Optimizers**: First-Order (Adafactor, SM3, CAME, Lion, Adam-mini), Zeroth-Order (MeZO, DeepZero, ZO-AdaMU)
- **Gradient Checkpointing**: Full, Selective
- **Gradient Accumulation**
- **Approximate Attention**: Linformer, Performer

### Level 2: System (Section 3.2)
- **Distributed Training**:
  - Data Parallelism: PyTorch DDP, ZeRO-1/2/3, FSDP
  - Tensor Parallelism: Megatron-LM (1D-TP), 2D-TP, 2.5D-TP, 3D-TP
  - Pipeline Parallelism: GPipe, 1F1B, Interleaved 1F1B, Zero Bubble
  - Sequence Parallelism: Ring Self-Attention, Megatron SP, Ulysses, Ring Attention
- **Offloading**: SwapAdvisor, ZeRO-Offload, ZeRO-Infinity

### Level 3: Hardware-Software Co-Optimization (Section 3.3)
- FlashAttention, FlashAttention-2, FlashAttention-3, MPress

### Gaps Revealed by Taxonomy

Comparing this taxonomy against the spec:

| Technique | In Paper | In Spec | Gap? |
|-----------|----------|---------|------|
| Mixed Precision | Yes | Yes | No |
| QAT (sub-FP8) | Yes | Partial (FP8 only) | Minor -- spec covers FP8; sub-FP8 is experimental |
| SM3, CAME, Adam-mini | Yes | No | **Yes** -- optimizer table should be expanded |
| MeZO (zeroth-order) | Yes | No | **Yes** -- entirely new category |
| Gradient Accumulation | Yes | Yes | No |
| Approximate Attention | Yes | No | No gap -- out of scope (architecture change) |
| 2D/2.5D/3D TP | Yes | No | **Possible gap** -- see Section 4 below |
| Zero Bubble PP | Yes | Mentioned | No |
| Ulysses SP | Yes | Not explicitly | Minor -- spec covers Megatron SP but not Ulysses |
| Ring Attention | Yes | Not explicitly | Minor -- spec mentions long-context but not Ring Attention by name |
| SwapAdvisor | Yes | No | No gap -- single-GPU optimization, out of scope |
| MPress | Yes | No | No gap -- advanced system optimization, out of scope |

---

## 4. Quantitative Benchmarks

### 4.1 Tensor Parallelism Efficiency (Table 3) -- NEW CALIBRATION DATA

| TP Method | Memory Consumption | Throughput | Notes |
|-----------|-------------------|------------|-------|
| 1D-TP (Megatron-LM) | N_d x reduction | 77% throughput (8 GPUs) | Standard, in spec |
| 2D-TP | 70% of 1D-TP memory (4 GPUs) | Lower communication volume | Not in spec |
| 2.5D-TP | 56% of 1D-TP memory (8 GPUs) | Splits input data | Not in spec |
| 3D-TP | 35% of 1D-TP memory (8 GPUs) | More process groups | Not in spec |

**Status**: The spec only covers 1D-TP (Megatron-LM style). Multi-dimensional TP (2D, 2.5D, 3D) is a research area that further reduces per-device memory but with increased implementation complexity. The 77% throughput figure for 1D-TP at 8 GPUs is a useful calibration point.

**Recommendation**: The spec's MFU guidelines already implicitly account for TP communication overhead. However, the 77% throughput at 8 GPUs for 1D-TP could be used to refine the throughput scoring heuristic in Section 9. The 2D/2.5D/3D TP methods are not widely available in standard frameworks and are probably out of scope for v1.

### 4.2 Pipeline Parallelism Efficiency (Table 3) -- REFINES EXISTING

| PP Method | Memory | Throughput | Notes |
|-----------|--------|------------|-------|
| GPipe | N_d x reduction | 3.5x throughput improvement | High activation memory |
| 1F1B | Reduces stored microbatches from N_m to N_d | Baseline for modern PP | Already in spec |
| Interleaved 1F1B | 1.23x memory vs 1F1B | 15% throughput increase | Already in spec |
| Zero Bubble | 1.08x memory vs 1F1B | 11% throughput increase | Mentioned in spec |

**Status**: The interleaved 1F1B numbers (1.23x memory, 15% throughput) validate the spec's formula in Section 5.7:
```
Activation memory multiplier = 1 + (N_pp - 1) / (N_pp x VP)
```
The Zero Bubble schedule (1.08x memory, 11% throughput vs 1F1B) is a newer approach that achieves better memory-throughput tradeoff than interleaved 1F1B.

**New insight**: Zero Bubble PP achieves almost all the throughput benefit of interleaved 1F1B (11% vs 15% improvement) at much lower memory cost (1.08x vs 1.23x). This might be worth noting in the spec as an alternative PP schedule.

### 4.3 Sequence Parallelism Efficiency (Table 3) -- NEW CALIBRATION DATA

| SP Method | Memory | Throughput | Notes |
|-----------|--------|------------|-------|
| Ring Self-Attention (RSA) | 71% of 1D-TP memory | 97% of 1D-TP throughput (s=2048) | Not in spec by name |
| Megatron SP | 67% of 1D-TP memory | 6% speedup vs 1D-TP | In spec |
| Ulysses | - | 1.51x throughput vs Megatron SP; 1.99x vs RSA (s=8K, 32 GPUs) | Not in spec |
| Ring Attention | - | Enables training 7B-65B on >4M tokens | Not in spec |

**Status**: The spec covers Megatron SP (Section 5.3) well. The Ulysses and Ring Attention methods are relevant for very long sequences (>8K tokens) and could be noted as alternatives. The quantitative comparison (Ulysses 1.51x faster than Megatron SP) is useful for long-context training scenarios.

**Recommendation**: Add a brief note in Section 5.3 that for very long sequences (>8K tokens), DeepSpeed Ulysses or Ring Attention may provide better throughput than Megatron SP, with Ulysses showing approximately 1.5x throughput advantage at 8K sequence length on 32 GPUs.

### 4.4 Offloading Performance (Table 4) -- REFINES EXISTING

| Method | Performance |
|--------|-------------|
| SwapAdvisor | 53-99% of ideal throughput (single-GPU) |
| ZeRO-Offload | 1.62x larger model vs SwapAdvisor; 4.5x vs Megatron; 7.8x vs ZeRO-2 on single DGX-2 |
| ZeRO-Infinity | 1T model on single DGX-2 node without MP; super-linear scalability 64->512 GPUs |

**Specific ZeRO-Infinity numbers**: 
> "On 32 Nvidia V100 DGX-2 nodes, ZeRO-infinity supports Transformer models with up to 32T parameters, which is 50x the maximum size accommodated by 3D-parallelism (around 650B parameters)."

**Status**: Already covered in spec Section 5.2, but the 50x scale advantage over 3D parallelism is a useful quantitative benchmark.

### 4.5 Real-World Training Configurations (Table 1) -- NEW VALIDATION DATA

This table maps real models to their training configurations and memory costs. Useful for validating the calculator:

| Model | Params | Memory Cost (est.) | Optimizations Used |
|-------|--------|-------------------|-------------------|
| AlphaFold 2 | 93M | 1.45 GB | DP, mixed-precision, GC |
| xTrimoPGLM | 100B | 1.56 TB | DP (ZeRO-1), PP (1F1B), TP, mixed-precision, GC |
| ESM3 | 1.4B/7B/98B | 1.53 TB | DP (FSDP), mixed-precision |
| Med-PaLM | 540B | 8.44 TB | DP (ZeRO-3), TP, GC |
| GeoGalactica | 30B | 480 GB | DP+PP+TP+SP |
| K2 | 7B | 112 GB | DP (ZeRO-3), mixed-precision, GA |

**Validation check against spec formulas**:
- K2 (7B, ZeRO-3): Spec predicts 16 x 7B / N_dp bytes for model states. At 112 GB total, this implies model states + activations + overhead on a cluster. 16 x 7B = 112 GB unsharded, confirming the 16P baseline.
- Med-PaLM (540B): 16 x 540B = 8.64 TB, reported as 8.44 TB. Close match (within 3%).

**Status**: These serve as validation data points. The calculator could use these as internal test cases.

---

## 5. Novel Insights

### 5.1 Scientific Model Memory Challenges -- OUT OF SCOPE BUT INTERESTING

The paper identifies three unique memory drivers for scientific models:
1. **Parameter growth**: 240-fold every 2 years
2. **Long sequences**: DNA, protein sequences, climate time series demand O(n^2) attention
3. **Diverse data structures**: High-resolution images, multimodal data increase activation memory

The third point is interesting: multimodal models may have activation memory profiles that differ significantly from text-only transformers. The spec's Korthikanti formula assumes text-only attention. For vision-language models, patch embeddings and cross-attention add activation terms not in the current formula.

**Status**: Out of scope for v1 (calculator targets text LLMs), but worth noting for future extension.

### 5.2 Cubic Activation Memory for Certain Architectures -- INTERESTING

> "AlphaFold 2 model having a relatively small parameter size (about 93M), the memory consumed by intermediate activations is substantial due to its architecture, and peak memory usage increases cubically with the length of the input sequence."

This is specific to AlphaFold's architecture (pair representation attention over sequence pairs), not standard transformers. Standard transformer attention is O(s^2) for activations, not O(s^3).

**Status**: Out of scope. Standard transformer activation memory is O(s^2) per the Korthikanti formula.

### 5.3 Framework Adoption Gap -- ADVISORY

The paper notes a significant finding relevant to the calculator's UX:

> "Despite the widespread application of transformers in scientific fields, there is a notable lack of memory efficiency in their training processes. Most works adopt one or more common memory-saving techniques (e.g., mixed precision training, DP), but more advanced strategies remain underutilized."

From Table 1, many real models use only DP + mixed precision, missing opportunities from activation checkpointing, ZeRO-2/3, TP, SP, etc. The calculator's recommendation engine could highlight when users are leaving significant memory savings on the table.

### 5.4 Zero Bubble PP Decomposition -- REFINES EXISTING

> "Qi et al. suggest splitting the backward pass into calculating the gradients of activations and the gradients of parameters (denote the two calculations as B and W respectively). Theoretically, one could complete B passes of all layers before performing W passes for each layer."

> "Qi et al. proposed an analytic framework to decompose existing pipeline schedules into building blocks and found that peak memory usage of a pipeline schedule is highly related to the lifespan of the building blocks."

**Status**: The spec mentions Zero Bubble in Table 3 context but doesn't detail the B/W decomposition. This is primarily an implementation detail of the PP scheduler, not a memory formula change. The calculator could note that Zero Bubble PP exists as an advanced schedule option that reduces bubble overhead to near-zero at moderate memory cost (1.08x vs 1F1B).

### 5.5 Ulysses SP Workflow Detail -- NEW

> "Jacobs et al. proposed Ulysses, a method that divides the input along sequence dimension onto all devices and conducts all-to-all communication on the Query (Q), Key (K), and Value (V) matrices before performing attention computation. This ensures that each device operates with disjoint attention heads."

Key difference from Megatron SP: Ulysses requires the number of attention heads to be an integer multiple of the SP degree, while Megatron SP requires the sequence length to be divisible by SP degree. Ulysses communication volume decreases linearly as SP degree increases, while in Megatron-LM it is independent of TP degree.

**Status**: Not explicitly in spec. The constraint that Ulysses requires attention heads divisible by SP degree is similar to TP's constraint. Could be added to Section 9 constraints if Ulysses SP is supported.

---

## 6. Findings Summary Table

| Finding | Category | Spec Status | Recommendation |
|---------|----------|-------------|----------------|
| Model states = 16P (bf16 grads) | Formula | Already covered | None |
| ZeRO-1/2/3 memory reduction | Formula | Already covered | None |
| Korthikanti activation formula | Formula | Already covered | None |
| Adafactor low-rank explanation | Detail | Partially covered | Add tooltip detail |
| SM3 optimizer | Technique | **Not in spec** | Add to optimizer table (~10-12 bytes/param) |
| CAME optimizer | Technique | **Not in spec** | Add to optimizer table (~12 bytes/param) |
| Adam-mini optimizer | Technique | **Not in spec** | **Add to optimizer table (~10 bytes/param, 45-50% savings vs AdamW)** |
| MeZO (zeroth-order) | Technique | **Not in spec** | **Add new optimizer category (~2 bytes/param, fine-tuning only)** |
| BitNet / BitNet b1.58 / EfficientQAT | Technique | Not in spec | Note as experimental; out of scope for v1 |
| 1D-TP at 77% throughput (8 GPUs) | Benchmark | New calibration | Use for throughput scoring validation |
| 2D/2.5D/3D TP | Technique | Not in spec | Out of scope for v1 (not in standard frameworks) |
| Interleaved 1F1B: 1.23x mem, 15% throughput | Benchmark | Already covered | Validates spec formula |
| Zero Bubble: 1.08x mem, 11% throughput | Benchmark | Partially covered | Add as PP schedule option note |
| Ulysses SP: 1.51x throughput vs Megatron SP | Benchmark | **Not in spec** | Note for long-context (>8K) scenarios |
| Ring Attention: >4M token sequences | Technique | Not in spec | Note for extreme long-context |
| MPress (D2D NVLink swapping) | System | Not in spec | Out of scope for v1 |
| SwapAdvisor (53-99% throughput) | System | Not in spec | Out of scope (single-GPU) |
| FlashAttention-3 (85% utilization BF16) | Benchmark | Partially covered | Note FA-3 Hopper perf |
| Med-PaLM 540B = 8.44 TB memory | Validation | New test case | Add to validation test cases |
| K2 7B = 112 GB memory | Validation | New test case | Add to validation test cases |

---

## 7. Recommended Spec Changes (Priority-Ordered)

### HIGH PRIORITY (material impact on calculator accuracy/features)

1. **Add Adam-mini to optimizer table** (Section 5.1): ~10 bytes/param, 45-50% memory savings vs AdamW with comparable performance. This is a production-ready optimizer that meaningfully changes memory estimates.

2. **Add zeroth-order optimizer category** (Section 5.1): MeZO at ~2 bytes/param for fine-tuning only. This enables entirely different memory profiles for post-training (Section 10).

### MEDIUM PRIORITY (useful additions)

3. **Add SM3 and CAME to optimizer table** (Section 5.1): Both at ~12 bytes/param, alternatives to Adafactor with better training stability.

4. **Note Ulysses SP throughput advantage** (Section 5.3): For sequences >8K tokens, Ulysses provides ~1.5x throughput over Megatron SP. Add as a brief note.

5. **Add Zero Bubble PP schedule option** (Section 5.7): 1.08x memory vs 1F1B, 11% throughput increase. Better memory-throughput tradeoff than interleaved 1F1B.

### LOW PRIORITY (edge cases or validation only)

6. **Add validation test cases** from Table 1 (Med-PaLM 540B = 8.44 TB, K2 7B = 112 GB).

7. **Note FlashAttention-3 performance** (85% utilization on Hopper) in Section 5.3.

8. **Note sub-FP8 QAT** (BitNet, EfficientQAT) exists but is experimental; out of scope for calculator v1.
