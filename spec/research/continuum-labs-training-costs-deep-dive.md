# Continuum Labs "Transformer Training Costs" -- Deep Dive for LLM Training GPU Calculator

**Source:** https://training.continuumlabs.ai/infrastructure/data-and-memory/transformer-training-costs
**Date:** 2026-03-31

---

## 0. Executive Summary

This blog post by Continuum Labs is a **comprehensive reference page** covering transformer training cost estimation end-to-end: compute formulas, memory breakdown, activation memory with three recomputation strategies, ZeRO stages, 3D parallelism, and inference memory. It synthesizes material from the key papers (Kaplan 2020, Chinchilla 2022, Korthikanti 2022) into a single practical reference.

**Verdict for our spec**: The post covers almost exactly the same ground as our spec's Sections 4-6 and 5.1-5.3. There are **no novel formulas or recomputation strategies** beyond what the spec already contains. However, there are several useful cross-validation points and a few minor items worth noting. The post does NOT contain any content that would require spec changes.

---

## 1. Activation Memory Formulas (Three Recomputation Strategies)

The post presents exactly three activation memory formulas. All three originate from Korthikanti et al. (2022, arXiv:2205.05198).

### 1.1 No Recomputation (Store Everything)

```
M_activations = s * b * h * L * (10 + 24/t + 5*a*s / (h*t))  bytes
```

Variables:
- s = sequence length
- b = micro-batch size per GPU
- h = hidden dimension
- L = number of layers
- a = number of attention heads
- t = tensor parallelism degree

**Comparison with our spec (Section 5.3)**: Our spec uses the equivalent formula:
```
M_act_layer = s * b * d * (34 + 5 * a * s / d)  bytes
```
where d = h (hidden dimension). The difference is that our spec's `34` equals the blog's `10 + 24/t` when `t = 1` (no tensor parallelism): `10 + 24/1 = 34`. The blog's formula already includes the tensor parallelism factor `t` in the `24/t` term, while our spec handles TP in a separate formula variant.

**STATUS: Already covered.** Our spec's Section 5.3 has this formula with and without TP. The blog does not add anything new.

### 1.2 Selective Recomputation

```
M_activations = s * b * h * L * (10 + 24/t)  bytes
```

This drops the `5*a*s/(h*t)` term (the attention score matrix).

**Comparison with our spec**: Our spec's selective checkpointing formula is:
```
M_act_layer = s * b * d * (10 + 24/N_tp + 5 * a * s / (d * N_tp))  bytes
```
This is actually the formula for selective checkpointing WITH sequence parallelism. The blog's selective formula (`10 + 24/t`) matches the Korthikanti formula that drops the `5as/d` term, which is the attention score matrix that gets recomputed.

**STATUS: Already covered.** Our spec Section 5.3 has this. The blog does not explain WHAT selective recomputation drops -- it merely presents the formula without derivation, referencing arXiv:2205.05198 for details. Our spec already provides the derivation context (the `5as/d` term is the attention score matrix for Q*K^T, softmax, and dropout).

### 1.3 Full Recomputation

```
M_activations = 2 * s * b * h * L  bytes
```

Only stores the layer input (2 bytes per element in half precision).

**Comparison with our spec**: Our spec Section 5.3 has:
```
M_act_layer = 2 * s * b * d  bytes  (store only layer input)
```
Total = L_active * M_act_layer / N_pp. Identical.

**STATUS: Already covered.**

### 1.4 Key Assumption the Blog States

> "We assume no sequence parallelism is being used."

This is explicitly stated. Our spec handles both the with-SP and without-SP cases, making our coverage strictly broader.

### 1.5 No Novel Recomputation Strategies

The blog covers exactly the standard three: None, Selective, Full. It does NOT cover:
- Block-level partial recomputation (NeMo `recompute_method="block"`) -- our spec covers this
- CPU activation offloading -- our spec covers this
- Any hybrid or fine-grained strategy

---

## 2. Compute Cost Formulas

### 2.1 Base Compute

```
C ≈ tau * T = 6PD
C_forward ≈ 2 * P * D
C_backward ≈ 4 * P * D
```

Where:
- C = total compute (FLOPs)
- tau = aggregate throughput = (No. of GPUs) * (Actual FLOPs per GPU)
- T = training time in seconds
- P = number of parameters
- D = dataset size in tokens

**STATUS: Already in spec Section 4.1 (C = 6*Psi*D) and Section 6.1 (T = C / (N_gpu * F_peak * MFU)).**

### 2.2 Recomputation Compute Overhead

The blog states:

> "The additional recomputation necessary also depends on the selectivity of the method, but it's bounded above by a full additional forward pass. Hence the updated cost of the forward pass is given by: 2PD <= C_forward <= 4PD."

This gives total training compute ranging from 6PD (no recomputation) to 8PD (full recomputation), which matches our spec Section 6.1's table:

| Mode | Compute | Overhead |
|------|---------|----------|
| None | 6PsiD | Baseline |
| Selective | ~6PsiD * (1 + s/(18h)) | 2-7% |
| Full | 8PsiD | 33% |

The blog does NOT provide the selective recomputation overhead formula `(1 + s/(18h))` that our spec derives from Korthikanti Appendix A. It only gives the bounds.

**STATUS: Already covered. Our spec is more precise for selective overhead.**

### 2.3 Chinchilla Scaling Law

The blog mentions `D = 20P` as the compute-optimal relationship and recommends:

> "We do not recommend training a LLM for less than 200B tokens."

**STATUS: Already in spec Section 4.3 (D_optimal ~ 20*Psi) and Section 4.3 practical minimum note.**

---

## 3. Training Memory Components

### 3.1 Model Parameters

| Precision | Bytes/param |
|-----------|-------------|
| int8 | 1 |
| fp16/bf16 | 2 |
| fp32 | 4 |
| Mixed (fp16/bf16 + fp32) | 2 (compute) + 4 (optimizer master copy) |

**STATUS: Already in spec Section 5.1.**

### 3.2 Optimizer States

The blog provides three optimizer configurations:

**AdamW (vanilla mixed precision)**: 12 bytes/param
- fp32 copy of weights: 4 bytes/param
- Momentum (m): 4 bytes/param
- Variance (v): 4 bytes/param

**8-bit optimizers (bitsandbytes)**: 6 bytes/param
- fp32 copy: 4 bytes/param
- Momentum: 1 byte/param
- Variance: 1 byte/param

**SGD with momentum**: 8 bytes/param
- fp32 copy: 4 bytes/param
- Momentum: 4 bytes/param

**Comparison with our spec**: Our spec Section 5.1 has a more complete optimizer table including AdamW (16 or 18 total), AdamW 8-bit (12 total), SGD+momentum (12 total), SGD no momentum (8 total), Adafactor (12 total), Lion (12 total), LAMB (16-18 total), and FP8 (14 total).

**DISCREPANCY NOTE**: The blog's "8-bit optimizer" formula of 6 bytes/param includes ONLY the optimizer states (4 fp32 master + 1 momentum + 1 variance). It excludes the 2 bytes for bf16 parameters and 2 bytes for gradients that our spec includes in the total. Our spec's "AdamW + 8-bit states" at 12 bytes/param (2+2+4+2+2) correctly includes parameters and gradients. These are NOT contradictions -- they are counting different things (optimizer-only vs. total model states).

Similarly, the blog's SGD "8 bytes/param" is optimizer-only (4 master + 4 momentum), while our spec's SGD+momentum at 12 bytes/param includes the bf16 params and gradients (2+2+4+4).

The blog's AdamW "12 bytes/param" is optimizer-only. Our spec's total is 18 (with fp32 grads) or 16 (with bf16 grads).

**STATUS: Already covered. No new information.**

### 3.3 Gradients

- fp32: 4 bytes/param
- fp16: 2 bytes/param

**STATUS: Already in spec Section 5.1 (beta_grad = 2 or 4).**

### 3.4 Total Training Memory

```
Total Memory_Training = Model Memory + Optimizer Memory + Activation Memory + Gradient Memory
```

**STATUS: Already in spec Section 5.5 (M_total = M_model_states + M_activations + M_temporary + M_communication).**

### 3.5 Inference Memory

```
Total Memory_Inference ~ 1.2 * Model_Memory
```

The 1.2x factor accounts for KV cache, temporary buffers, and framework overhead. The blog notes the overhead is "typically <= 20%".

**STATUS: Not directly in our spec since we focus on training. This is a simple inference heuristic. Not needed for our calculator.**

---

## 4. ZeRO Sharded Optimizers

### 4.1 ZeRO-1

```
Total Memory = Model Memory + Optimizer_Memory / N_gpu + Activation Memory + Gradient Memory
```

### 4.2 ZeRO-2

```
Total Memory = Model Memory + Activation Memory + (Optimizer Memory + Gradient Memory) / N_gpu
```

### 4.3 ZeRO-3

```
Total Memory = Activation Memory + (Model Memory + Optimizer Memory + Gradient Memory) / N_gpu + ZeRO-3_Live_Params
```

ZeRO-3 Live Params: Controlled by DeepSpeed config options `stage3_max_live_parameters`, `stage3_max_reuse_distance`, `stage3_prefetch_bucket_size`, `stage3_param_persistence_threshold`. These control how many parameters reside on GPU at any time; larger values = more memory but less communication.

**Comparison with our spec**: Our spec Section 5.2 uses more precise formulations with explicit byte counts:
- ZeRO-0: Phi*Psi
- ZeRO-1: (2 + beta_grad)*Psi + K_opt*Psi/N_dp
- ZeRO-2: 2*Psi + (beta_grad + K_opt)*Psi/N_dp
- ZeRO-3: Phi*Psi/N_dp

Our spec is strictly more precise and also covers K_opt as a derived quantity that auto-adapts to any optimizer. The blog uses informal "Model Memory" / "Optimizer Memory" / "Gradient Memory" placeholders rather than byte-level formulas.

**STATUS: Already covered. Our spec is more precise.**

### 4.4 ZeRO-2 Sharding Clarification

The blog's ZeRO-2 formula groups optimizer and gradient memory together and divides by N_gpu. This matches the standard formulation: ZeRO-2 shards both optimizer states AND gradients.

**NOTE**: In my initial fetch, the AI extraction suggested ZeRO-2 only shards gradients, with optimizer unsharded. Re-reading confirms the blog actually shards both (optimizer + gradient) / N_gpu, which is correct and matches our spec.

**STATUS: Consistent with spec.**

---

## 5. 3D Parallelism

### 5.1 Model Memory with Parallelism

```
M_model_with_parallelism ~ Model Memory / (PP * TP)
```

### 5.2 Gradient Memory with Parallelism

```
M_gradients_with_parallelism ~ Gradient Memory / PP
```

Note: Gradients are divided only by PP, not by TP. This is because gradient accumulation in pipeline parallelism reduces gradient memory per stage, while TP does not reduce gradient memory (each TP rank stores a partial gradient that matches its partial parameter set).

**Comparison with our spec**: Our spec Section 5.6 states `M_params_per_gpu ~ Psi_params / N_tp` for TP, and Section 5.7 handles PP. The blog's formula is consistent: model memory divides by both PP and TP, gradient memory only by PP.

**WAIT -- this is actually inconsistent.** If model memory divides by PP*TP, then gradients (which are the same shape as parameters) should also divide by PP*TP. The blog's formula `Gradient Memory / PP` is incorrect or describes a specific implementation detail. In standard Megatron-LM with TP, each TP rank holds 1/N_tp of the parameters AND 1/N_tp of the gradients. The blog may be simplifying or describing a specific ZeRO+PP interaction where gradient accumulation across pipeline stages creates this asymmetry.

Our spec handles this more carefully through the per-GPU memory formulas that account for both TP and PP effects on all memory components.

**STATUS: Already covered. The blog's gradient formula may be imprecise.**

### 5.3 Data Parallelism Degree

```
DP_Degree = N_gpu / (PP * TP)
```

**STATUS: Already in spec Section 9 constraints: N_dp = N_gpu / (N_tp * N_pp).**

### 5.4 Combined ZeRO-1 + 3D Parallelism + Activation Partitioning

```
Total Memory = Model Memory / (PP * TP) + Optimizer Memory / N_gpu + Activation Memory / TP + Gradient Memory / PP
```

This is the blog's most complex formula. It combines:
- ZeRO-1 for optimizer state sharding (divided by total N_gpu)
- TP for activation partitioning (Activation Memory / TP)
- PP for gradient and model sharding
- TP for model weight sharding

**Comparison with our spec**: Our spec handles each component separately through the ZeRO (Section 5.2), TP (Section 5.6), and PP (Section 5.7) sections. The combined effect is derived from composing these individual formulas. The blog's combined formula is a useful sanity check but does not add new information.

**STATUS: Already covered through composition of individual spec formulas.**

---

## 6. Engineering Performance Benchmarks

The blog provides these throughput benchmarks on A100 GPUs:

| Framework | TFLOP/s per A100 | Notes |
|-----------|-------------------|-------|
| GPT-NeoX (normal attention) | 150 | |
| GPT-NeoX (Flash Attention) | 180 | +20% from Flash Attention |
| Megatron-DS | 137-163 | Range |
| General guideline | ~120 | "You should always be able to achieve approximately 120 TFLOP/s/A100" |
| Alert threshold | <115 | "If you are seeing below 115 TFLOP/s/A100 something is probably going wrong" |

**Comparison with our spec**: Our spec Section 6.3 uses MFU percentages rather than absolute TFLOP/s. Converting: A100 BF16 peak = 312 TFLOP/s.
- 150 TFLOP/s = 48% MFU
- 180 TFLOP/s = 58% MFU
- 120 TFLOP/s = 38% MFU
- 115 TFLOP/s = 37% MFU

These fall within our spec's "Medium model (1B-10B), 8-64 GPUs: 35-45% MFU" guideline range. The Flash Attention number (58% MFU) is notably high and may be HFU rather than MFU, or may reflect a well-optimized configuration.

**STATUS: Useful as validation data points. The 120 TFLOP/s "minimum achievable" threshold (38% MFU on A100) aligns with our spec's lower-end guidelines.**

### 6.1 EleutherAI Parallelism Preference

The blog notes:

> "We train with pipeline and tensor parallelism along with ZeRO-1. This is because we find ZeRO-3 to be too communication-heavy for our hardware at large scales."

**STATUS: Already captured in our spec Section 9 (ZeRO Stage Selection Heuristic, which recommends the lowest ZeRO stage that fits) and the ZeRO+PP compatibility table.**

---

## 7. Compute Units and Notation

The blog defines three compute unit conventions:
- **FLOP-seconds**: FLOPs/s * seconds
- **GPU-hours**: Number of GPUs * hours
- **PetaFLOP-days**: 10^15 * 24 * 3600 total FLOPs = 8.64 * 10^19 FLOPs

**STATUS: Not explicitly in our spec. The PetaFLOP-days convention is used by OpenAI but is not needed for our calculator. No action required.**

---

## 8. Pipeline Parallelism Activation Note

The blog mentions:

> "[Pipeline parallelism requires] that all GPUs store the activations for all micro-batches in-flight, which becomes significant for large models."

This refers to the fact that with PP, multiple micro-batches are simultaneously in different pipeline stages, each requiring its own set of activations. Our spec addresses this in Section 5.7 with the interleaved scheduling activation memory multiplier.

**STATUS: Already covered in spec Section 5.7.**

---

## 9. ZeRO + PP Compatibility

The blog notes:

> "Pipeline parallelism incompatible with ZeRO-2/3 gradient sharding (difficult efficiency maintenance)."

**STATUS: Already in spec Section 9 constraints table (ZeRO-2/3 + PP = No).**

---

## 10. References Cited by the Blog

The blog references three papers:
1. **arXiv:2001.08361** -- Kaplan et al. (2020), "Scaling Laws for Neural Language Models"
2. **arXiv:2203.15556** -- Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models" (Chinchilla)
3. **arXiv:2205.05198** -- Korthikanti et al. (2022), "Reducing Activation Recomputation in Large Transformer Models"

All three are already in our spec's reference papers list (Section in research-sources.md).

---

## 11. What This Blog Does NOT Cover (Gaps vs. Our Spec)

The blog is notable for what it omits relative to our spec:

1. **No PaLM per-token formula** (6Psi + 12Lds) -- uses only the simplified 6PD
2. **No GQA/MQA support** -- activation formulas use generic `a` (attention heads) without distinguishing KV heads
3. **No sequence parallelism** -- explicitly assumed to be off
4. **No Flash Attention memory impact** -- not mentioned
5. **No SwiGLU/GeGLU FFN variants** -- not discussed
6. **No MoE models** -- not covered
7. **No LoRA/QLoRA/PEFT** -- not covered
8. **No post-training (SFT/DPO/PPO/GRPO)** -- not covered
9. **No cost estimation** -- no pricing or $/GPU-hour
10. **No failure-adjusted training time** -- not covered
11. **No output logits memory** -- not covered
12. **No communication buffer sizing** -- mentioned but not quantified
13. **No FSDP/ZeRO equivalence** -- not discussed
14. **No CPU/NVMe offloading** -- not covered
15. **No block-level partial recomputation** -- only None/Selective/Full
16. **No d_ff correction** for non-standard FFN widths
17. **No PyTorch AMP FP32 precision caveat** (36+6 vs 34+5 coefficients)
18. **No interleaved pipeline scheduling**

---

## 12. Verdict: Impact on Spec

**No spec changes required.**

Every formula in the Continuum Labs blog is already present in our spec, and in most cases our spec has more precise and more general formulations. The blog is a good educational reference but contributes no novel formulas, recomputation strategies, or edge cases that our spec does not already cover.

**Useful as cross-validation**:
- The activation memory formulas match our spec's Korthikanti-derived formulas exactly (accounting for notation differences h=d, t=N_tp)
- The ZeRO formulas are consistent with our spec's Section 5.2
- The throughput benchmarks (120-180 TFLOP/s on A100) validate our MFU guideline ranges
- The 8-bit optimizer breakdown (4+1+1 = 6 bytes optimizer-only) is consistent with our 12 bytes total (2 param + 2 grad + 4 master + 2 momentum + 2 variance)

The only minor note is the blog's `M_gradients / PP` formula (gradients sharded only by PP, not by TP) which differs from the expected behavior where gradients match parameter sharding. This could reflect a specific implementation choice but is not well-explained in the blog. Our spec handles this correctly through per-component formulas.

---

## 13. Action Items

1. Mark `[ ] Transformer Training Costs (Continuum Labs)` as `[x]` in research-sources.md -- reviewed, no changes needed
2. No spec modifications required
