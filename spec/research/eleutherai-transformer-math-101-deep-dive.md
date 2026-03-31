# Deep Dive: Transformer Math 101 (EleutherAI) + Latent Space Podcast

**Source**: Anthony, Biderman, Schoelkopf (EleutherAI), "Transformer Math 101" blog post, and Latent Space podcast episode "The Mathematics of Training LLMs"
**URLs**:
- Blog post: https://blog.eleuther.ai/transformer-math/
- Podcast: https://www.latent.space/p/transformers-math
**Type**: Technical blog post + podcast interview
**Authors**: Quentin Anthony, Stella Biderman, Hailey Schoelkopf
**Published**: April 18, 2023; last modified October 8, 2024
**Date reviewed**: 2026-03-31

---

## 1. Executive Summary

This is the most widely-referenced practitioner's guide to LLM training memory and compute estimation. It consolidates formulas from OpenAI's scaling laws paper, DeepMind's Chinchilla, the ZeRO paper, and Megatron's activation recomputation paper into a single unified reference. The blog post focuses primarily on **memory estimation** (model weights, optimizer states, gradients, activations) and **compute estimation** (C = 6PD), while the podcast discussion adds practical heuristics around GPU utilization thresholds, parallelism strategy selection, and failure modes.

The key contribution is the unified "total training memory" formula combining all parallelism strategies (ZeRO stages, tensor parallelism, pipeline parallelism) into a single expression, plus the practical heuristic that 115 TFLOP/s per A100 is the minimum acceptable utilization threshold.

---

## 2. Core Compute Formula

### 2.1 The C = 6PD approximation

```
C = tau * T = 6 * P * D
```

Where:
- `C` = total compute in floating-point operations (FLOPs)
- `tau` = aggregate throughput = (number of GPUs) x (actual FLOPs per GPU)
- `T` = training time in seconds
- `P` = number of model parameters
- `D` = dataset size in tokens

**Breakdown:**
- Forward pass: C_forward = 2PD
- Backward pass: C_backward = 4PD (2x the forward pass)
- Total: C = C_forward + C_backward = 6PD

**Source**: Proposed and experimentally validated in OpenAI's scaling laws paper (arXiv:2001.08361) and DeepMind's Chinchilla paper (arXiv:2203.15556).

**Training time derivation** (algebraic rearrangement, not explicitly in the blog but implied):

```
T = C / tau = 6PD / (N_gpu * FLOPS_per_gpu)
```

### 2.2 Activation recomputation impact on compute

When activation checkpointing is used, the forward pass is repeated (partially or fully) during the backward pass:

```
2PD <= C_forward <= 4PD
```

- No recomputation: C_forward = 2PD, total C = 6PD
- Full recomputation: C_forward effectively doubled during backward, total C up to 8PD
- Selective recomputation: somewhere in between

### 2.3 Chinchilla optimal scaling

The blog states: "compute optimal language model has a number of parameters and a dataset size that satisfies the approximation D = 20P"

With D = 20P substituted: C = 6P * 20P = **120P^2**

Additional recommendation: "We do not recommend training a LLM for less than 200B tokens."

---

## 3. Memory Formulas

### 3.1 Inference Memory

**Model weights only:**

| Precision | Bytes per parameter | Formula |
|-----------|-------------------|---------|
| INT8      | 1                 | memory_model = 1 * P |
| FP16/BF16 | 2                 | memory_model = 2 * P |
| FP32      | 4                 | memory_model = 4 * P |

**Total inference memory with overhead:**

```
Total_Memory_Inference = 1.2 * Model_Memory
```

The 1.2x factor accounts for "additional overhead during the actual forward pass. In our experience this overhead is <= 20%."

### 3.2 Training Memory -- Model Parameters

- Pure FP32 training: `memory_model = 4 * P` bytes
- Pure FP16 training: `memory_model = 2 * P` bytes
- Mixed precision (FP16 compute + FP32 master): `memory_model = 2 * P` bytes (the FP32 copy is counted in optimizer states)

### 3.3 Training Memory -- Optimizer States

**AdamW (standard mixed-precision training):**

```
memory_optimizer = 12 * P  bytes
```

Components:
- FP32 master copy of parameters: 4 bytes/param
- FP32 momentum (first moment): 4 bytes/param
- FP32 variance (second moment): 4 bytes/param

**8-bit Adam (bitsandbytes):**

```
memory_optimizer = 6 * P  bytes
```

Components:
- FP32 master copy: 4 bytes/param
- INT8 momentum: 1 byte/param
- INT8 variance: 1 byte/param

**SGD with momentum:**

```
memory_optimizer = 8 * P  bytes
```

Components:
- FP32 master copy: 4 bytes/param
- FP32 momentum: 4 bytes/param

### 3.4 Training Memory -- Gradients

```
memory_gradients = 4 * P  bytes   (FP32 gradients)
memory_gradients = 2 * P  bytes   (FP16/BF16 gradients)
```

### 3.5 Training Memory -- Activations

These formulas come from Korthikanti et al. (arXiv:2205.05198). Variables:
- `s` = sequence length
- `b` = micro-batch size per GPU
- `h` = hidden dimension
- `L` = number of transformer layers
- `a` = number of attention heads
- `t` = tensor parallelism degree (1 if not used)

**No recomputation:**

```
memory_activations = s * b * h * L * (10 + 24/t + 5*a*s / (h*t))  bytes
```

**Selective recomputation:**

```
memory_activations = s * b * h * L * (10 + 24/t)  bytes
```

**Full recomputation:**

```
memory_activations = 2 * s * b * h * L  bytes
```

Note: These assume FP16 activation storage (2 bytes per element) with dropout masks at 1 byte per element. No sequence parallelism.

### 3.6 Total Training Memory (No Parallelism)

```
Total_Memory_Training = Model_Memory + Optimizer_Memory + Activation_Memory + Gradient_Memory
```

For standard mixed-precision training with AdamW (FP16 model + FP32 optimizer + FP16 gradients):

```
Total_Memory_Training = 2*P + 12*P + Activation_Memory + 2*P
                      = 16*P + Activation_Memory  bytes
```

The blog sometimes quotes **18 bytes per parameter** for the static component (model + optimizer + gradients), which corresponds to:
- FP16 model weights: 2 bytes/param
- FP32 master copy + momentum + variance: 12 bytes/param
- FP32 gradients: 4 bytes/param
- Total static: 2 + 12 + 4 = 18 bytes/param

However, the more common mixed-precision setup uses FP16 gradients (2 bytes/param), yielding 16 bytes/param static. The 18 figure includes FP32 gradients.

### 3.7 KV Cache Memory (Inference)

From the podcast: "two times the number of layers, times the number of heads, times the dimension of each head"

```
KV_cache = 2 * L * n_heads * d_head * bytes_per_element
```

Per token, per layer: stores both a key and value vector, each of size (n_heads * d_head).

The podcast notes this can be "comparable or larger than the model in some cases" for long contexts.

---

## 4. Distributed Training Memory Formulas

### 4.1 ZeRO Stages

**ZeRO-0** (disabled, standard data parallelism):

```
Total = Model_Memory + Optimizer_Memory + Activation_Memory + Gradient_Memory
```

**ZeRO-1** (optimizer state sharding, also called P_os):

```
Total = Model_Memory + Optimizer_Memory/N_dp + Activation_Memory + Gradient_Memory
```

**ZeRO-2** (optimizer + gradient sharding, also called P_os+g):

```
Total = Model_Memory + (Optimizer_Memory + Gradient_Memory)/N_dp + Activation_Memory
```

**ZeRO-3** (full sharding = optimizer + gradient + parameter sharding, also called P_os+g+p):

```
Total = (Model_Memory + Optimizer_Memory + Gradient_Memory)/N_dp + Activation_Memory + ZeRO3_Live_Params
```

Where N_dp = number of GPUs in the data-parallel group, and ZeRO3_Live_Params is the additional memory for parameters currently being used (controlled by DeepSpeed configuration parameters):
- `stage3_max_live_parameters`
- `stage3_max_reuse_distance`
- `stage3_prefetch_bucket_size`
- `stage3_param_persistence_threshold`

**ZeRO-R** (activation partitioning, can be combined with ZeRO stages):

```
Total = Model_Memory + Optimizer_Memory/N_dp + Activation_Memory + Gradient_Memory
```

(ZeRO-R partitions activations across data-parallel replicas.)

### 4.2 Data Parallelism Degree

```
N_dp = N_gpu / (N_pp * N_tp)
```

Where N_pp = pipeline parallelism degree, N_tp = tensor parallelism degree, N_gpu = total GPUs.

### 4.3 Tensor and Pipeline Parallelism

**Model memory with parallelism:**

```
memory_model_parallel = Model_Memory / (N_pp * N_tp)
```

**Gradient memory with pipeline parallelism:**

```
memory_gradients_parallel = Gradient_Memory / N_pp
```

### 4.4 Combined 3D Parallelism + ZeRO-1 + Activation Partitioning

The blog provides a combined formula:

```
Total = Model_Memory / (N_pp * N_tp)
      + Optimizer_Memory / N_gpu
      + Activation_Memory / N_tp
      + Gradient_Memory / N_pp
```

**Important caveats stated in the blog:**
1. Pipeline parallelism does NOT reduce the memory footprint of activations.
2. Pipeline parallelism requires ALL GPUs to store activations for ALL micro-batches in-flight, which becomes significant for large models.
3. GPUs need to temporarily store additional communication buffers required by parallelism schemes.

### 4.5 Compatibility Notes

- "Pipeline parallelism and tensor parallelism are compatible with all stages of ZeRO."
- However: "it's difficult to maintain efficiency when combining pipeline parallelism with ZeRO-2/3's gradient sharding... DeepSpeed currently forbids it."
- "Tensor parallelism... is complementary to all stages of ZeRO."

### 4.6 EleutherAI's Practical Strategy

"For the majority of Eleuther's work, we train with pipeline and tensor parallelism along with ZeRO-1. This is because we find ZeRO-3 to be too communication-heavy for our hardware at large scales, and instead use pipeline parallelism across nodes along with tensor parallelism within nodes."

---

## 5. GPU Utilization Thresholds and Performance Baselines

### 5.1 Achieved FLOPs Benchmarks (A100)

| Configuration | Achieved TFLOP/s per A100 |
|--------------|--------------------------|
| GPT-NeoX (standard attention) | 150 |
| GPT-NeoX (Flash Attention) | 180 |
| Megatron-DS (range) | 137-163 |
| General minimum expectation | 120 |
| Red flag threshold | < 115 |

"You should always be able to achieve approximately 120 TFLOP/s/A100."

"If you are seeing below 115 TFLOP/s/A100 there is probably something wrong."

### 5.2 V100 Baseline

From the podcast: 30-40 TFLOP/s on V100.

### 5.3 MFU vs HFU (from podcast discussion and referenced papers)

**Model FLOPs Utilization (MFU):**

```
MFU = (Model_FLOPs_per_step * steps_per_second) / Peak_Hardware_FLOPS
```

Where Model_FLOPs_per_step uses the theoretical 6PD formula (ignoring recomputation).

**Hardware FLOPs Utilization (HFU):**

```
HFU = (Actual_FLOPs_per_step * steps_per_second) / Peak_Hardware_FLOPS
```

Where Actual_FLOPs_per_step includes any recomputation overhead.

**Key distinction**: If activation recomputation is used, HFU > MFU because more FLOPs are actually executed than the theoretical minimum. If no recomputation, HFU = MFU.

---

## 6. Practical Heuristics and Rules of Thumb

### 6.1 GPU Cost (from podcast, 2023 pricing)

- Retail (cloud): $4-8/hour per GPU
- Wholesale/reserved: $1-2/hour per GPU

### 6.2 Parallelism Strategy Selection (from podcast)

1. Find the minimum number of GPUs needed to fit a single model instance in memory.
2. Calculate expected training time at that GPU count.
3. Add more GPUs (via data parallelism) only if time exceeds the acceptable threshold.
4. Each GPU doubling theoretically halves training time but introduces synchronization overhead and failure risk.

### 6.3 Tensor Parallelism Placement

- Tensor parallelism should be used WITHIN a node (uses fast NVLink/NVSwitch interconnect).
- Tensor parallelism should NOT span across nodes (inter-node bandwidth too low for the per-operation synchronization TP requires).

### 6.4 Scaling Limits (from podcast)

- For ~20B parameter models on the Summit supercomputer: performance degrades significantly beyond ~100-200 GPUs due to synchronization costs.
- "The more GPUs you have, the more likely things break."

### 6.5 Heterogeneous Clusters (from podcast)

- System performance is limited by the slowest GPU's VRAM (determines what fits).
- Training speed is limited by the slowest GPU/interconnect in the cluster.
- Performance "degrades significantly" with heterogeneous hardware.

### 6.6 Optimizer Memory Dominance (from podcast)

"Optimizer memory dominates over model weight memory in most training scenarios."

For AdamW: optimizer = 12 bytes/param vs model = 2 bytes/param (6:1 ratio).

### 6.7 Precision Recommendations

- FP16: requires loss scaling for numerical stability; limited dynamic range due to small exponent bits.
- BF16: preferred for A100+ hardware; eliminates loss scaling; better dynamic range.
- Quantized training (from podcast): "Even if you had infinite VRAM, you would still want a quantized model, just a bigger model that's quantized" -- deep learning's stochasticity makes precision beyond certain thresholds irrelevant.

### 6.8 Minimum Dataset Size

"We do not recommend training a LLM for less than 200B tokens."

---

## 7. Referenced Tools and Libraries

- **Megatron-LM**: Tensor + pipeline parallelism framework (NVIDIA)
- **DeepSpeed**: ZeRO optimizer sharding (Microsoft)
- **bitsandbytes**: 8-bit Adam optimizer (reduces optimizer memory from 12 to 6 bytes/param)
- **Flash Attention**: Efficient attention kernels (Dao et al.)
- **RWKV**: Linear attention alternative (eliminates quadratic KV cache scaling)
- **APEX/AMP**: Mixed precision training utilities

---

## 8. Referenced Papers

1. OpenAI Scaling Laws: arXiv:2001.08361
2. Chinchilla Scaling Laws (DeepMind): arXiv:2203.15556
3. ZeRO paper: arXiv:1910.02054
4. Activation Recomputation (Korthikanti et al.): arXiv:2205.05198
5. FSDP (Facebook): engineering.fb.com/2021/07/15/open-source/fsdp/
6. Transformer Inference Arithmetic: kipp.ly/blog/transformer-inference-arithmetic/

---

## 9. What is Unique or Non-Obvious

1. **The 18 vs 16 bytes/param ambiguity**: The "18 bytes per parameter" figure that is widely quoted includes FP32 gradients (4 bytes), but many implementations actually use FP16 gradients (2 bytes), giving 16 bytes/param. The blog lists both possibilities but does not clearly flag which is standard. Our calculator should default to 16 bytes/param (FP16 gradients) but allow the user to select FP32 gradients.

2. **ZeRO-3 live parameter overhead**: ZeRO-3's per-GPU memory is NOT simply `total/N_gpu` -- there is an additional "live parameters" overhead controlled by DeepSpeed configuration. This is a term that most simplified calculators omit.

3. **Pipeline parallelism does NOT reduce activation memory**: This is a common misconception. PP distributes model layers across GPUs, but each GPU must store activations for ALL micro-batches currently in-flight (the pipeline fill). This can be substantial.

4. **ZeRO-3 + pipeline parallelism incompatibility**: DeepSpeed forbids combining ZeRO-2/3 gradient sharding with pipeline parallelism due to efficiency loss. This is a practical constraint our calculator should enforce.

5. **The C = 6PD approximation accuracy**: Validated to within 3-10% across model sizes from 73M to 6.8B parameters (per Chinchilla paper Table A4, referenced in the blog). The approximation is slightly less accurate for smaller models where embedding parameters are a larger fraction of total parameters.

6. **The 115 TFLOP/s A100 red flag threshold**: This is a very practical diagnostic -- if a training run is achieving less than 115 TFLOP/s per A100, it indicates a software/configuration problem, not a hardware limitation.

7. **Training time linear scaling caveat**: The formula T = C/tau assumes "1000 GPUs for 1 hour costs the same as 1 GPU for 1000 hours." In practice, communication overhead means scaling is sublinear, especially beyond ~100-200 GPUs.

8. **Inference 1.2x overhead factor**: The blog provides a practical 20% overhead factor for inference memory beyond just model weights, based on empirical experience.

---

## 10. Gaps and Limitations

This source does NOT provide:
- Worked examples of training time calculations
- Batch size selection heuristics or gradient accumulation formulas
- Communication volume formulas for different parallelism strategies
- Pipeline bubble overhead formulas (covered in Narayanan 2021, which we have separately)
- MFU/HFU formal definitions with formulas (covered in Korthikanti 2022, which we have separately)
- Sequence parallelism details (covered in Korthikanti 2022)
- LoRA/QLoRA/adapter memory formulas
- Multi-node scaling efficiency models
- Per-layer FLOPs breakdown (covered in Chinchilla appendix, which we have separately)

---

## 11. Relevance to Our Calculator Spec

### Already covered by our existing deep dives:
- Activation memory formulas (Korthikanti deep dive has the authoritative version with all sub-components)
- ZeRO memory formulas (ZeRO paper deep dive)
- Pipeline bubble overhead (Megatron-LM Narayanan 2021 deep dive)
- C = 6PD validation (Chinchilla deep dive has Table A4)

### Potentially new or reinforcing for our spec:
- **8-bit Adam optimizer memory formula** (6 bytes/param): We should verify this is in our spec. The bitsandbytes 8-bit optimizer reduces memory from 12 to 6 bytes/param. This is distinct from QLoRA.
- **SGD with momentum memory formula** (8 bytes/param): For completeness, our calculator should support optimizers beyond just AdamW.
- **Inference memory 1.2x overhead factor**: If we support inference memory estimation, this is a useful empirical factor.
- **GPU utilization thresholds**: 120 TFLOP/s expected, 115 TFLOP/s minimum for A100. Useful as default FLOP/s values in our calculator.
- **EleutherAI's practical parallelism strategy**: TP within nodes + PP across nodes + ZeRO-1 as the default recommended configuration.
- **ZeRO-3 + PP incompatibility constraint**: Our calculator should warn or prevent this combination.
- **KV cache formula for inference**: Useful if we extend to inference memory estimation.
