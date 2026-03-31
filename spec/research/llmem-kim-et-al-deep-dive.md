# Deep Dive: LLMem -- Estimating GPU Memory Usage for Fine-Tuning Pre-Trained LLMs

**Paper**: Kim, T., Wang, Y., Chaturvedi, V., Gupta, L., Kim, S., Kwon, Y., & Ha, S. (2024). "LLMem: Estimating GPU Memory Usage for Fine-Tuning Pre-Trained LLMs." IJCAI 2024, pp. 6324-6330.
**URL**: https://arxiv.org/abs/2404.10933
**Code**: https://github.com/taehokim20/LLMem
**Framework**: Built on Colossal-AI (Gemini plugin, ZeRO-3 style chunk-based memory management)

---

## 1. Executive Summary

LLMem is a peak GPU memory estimator for fine-tuning (not pretraining) of transformer-based decoder LLMs. It achieves 1.6% error on single-GPU and 3.0% average error on multi-GPU setups, compared to DNNMem's 42.6% average error. The key insight is that existing memory estimators (DNNMem, TSplit) drastically underestimate memory because they fail to account for three things specific to LLM fine-tuning with modern frameworks:

1. **Chunk-based memory management** (Colossal-AI/ZeRO) where parameters and gradients share the same allocated chunk with CUDA page alignment
2. **Separate memory allocation patterns** for the transformer body vs. the language modeling head (lm_head)
3. **First-iteration optimizer state allocation** (momentum and variance tensors are allocated on the first training step, not at initialization)

**Scope limitation**: The paper covers only full fine-tuning. It does NOT cover LoRA, QLoRA, PEFT methods, or pretraining. It does NOT cover pipeline parallelism.

---

## 2. Core Memory Formula (Single GPU)

### 2.1 Peak Memory Equation

```
m_peak_single = m_base + m_p + m_os + m_out + m_lm
```

Where:
- `m_base` = Initial GPU memory (CUDA context + chunk manager initialization)
- `m_p` = Memory for parameters (fp16) + parameter copy (fp32), with gradient fp16 sharing the fp16 space
- `m_os` = Memory for optimizer states (Adam momentum fp32 + variance fp32)
- `m_out` = Peak memory from output tensors (layer/embedding outputs kept for gradient checkpointing)
- `m_lm` = Memory consumed by the language modeling head and loss calculation

### 2.2 Notation Table

| Symbol | Description |
|--------|-------------|
| m_base | Initially used GPU memory (CUDA context + chunk manager) |
| embed_p | Input embedding parameter count |
| lm_p | Language modeling head parameter count (bytes) |
| cs | Chunk size (determined by chunk manager to minimize waste) |
| bs | Batch size |
| sl | Sequence length |
| other_p | Remaining parameter count (everything except embed and lm_head) |
| B_16 | 2 bytes (fp16/bf16) |
| B_32 | 4 bytes (fp32) |
| cu_p | CUDA memory page size = 2 * 1024^2 bytes (2 MiB) |
| m_p | GPU memory for param/gradient fp16 and param fp32 |
| m_p,16 | GPU memory used by param fp16 only |
| m_p,32 | GPU memory used by param fp32 only |
| m_os | GPU memory for optimizer states (momentum fp32 + variance fp32) |
| e_n | Number of embedding layers |
| l_n | Number of transformer layers |
| o_n | Model output features (hidden dimension) |
| m_out | Peak GPU memory from output tensors |
| dict_n | Embedding dictionary size (vocabulary size) |
| m_lm | GPU memory for lm_head with loss calculation |
| gpu_n | Number of GPUs |
| dp_n, tp_n | Number of GPUs for DP or TP |
| m_total | Total GPU memory capacity |

---

## 3. Detailed Formula Breakdown

### 3.1 Parameter Memory (m_p)

```
m_p = ceil( (embed_p + ceil(other_p / cs) * cs) * (B_16 + B_32) / cu_p ) * cu_p
```

**Key insight -- parameter/gradient memory sharing**: During fine-tuning, param fp16 goes through forward and backward passes, then is converted to gradient fp16. Consequently, param fp16 and gradient fp16 share the same GPU memory space. This means you do NOT add separate gradient memory -- the gradient overwrites the parameter in-place within the chunk.

**Chunk-based allocation details**:
- The chunk manager determines the optimal chunk size (`cs`) to minimize GPU memory waste based on the model's parameter distribution
- The embedding parameters (`embed_p`) are managed separately from the rest because the input embedding is huge and has a large dictionary
- The remaining parameters (`other_p`) are rounded up to the nearest chunk size multiple: `ceil(other_p / cs) * cs`
- Both fp16 and fp32 copies are allocated in chunk-size units
- The `(B_16 + B_32)` = (2 + 4) = 6 bytes per parameter: this covers the fp16 working copy AND the fp32 master copy, but NOT gradients (which share the fp16 space)
- Everything is then rounded up to the CUDA memory page size (`cu_p = 2 MiB`)

**From the source code** (`size_estimator.py`, lines 104-150):
```python
def param_bytes(self):
    mods = list(self.model.modules())
    param_sizes = []
    for i in range(1, len(mods)):
        if not 'Embedding' in mods[i]._get_name():
            if not mods[i]._get_name() in ['Linear']:
                continue
        m = mods[i]
        p = list(m.parameters())
        for j in range(len(p)):
            param_sizes.append(np.array(p[j].size()))

    total_bytes = 0
    for i in range(len(param_sizes)):
        s = param_sizes[i]
        bytes = np.prod(np.array(s)) * self.bytes  # self.bytes = 2 (fp16)
        # Round up to CUDA page size
        if bytes % self.base_size != 0:
            bytes = int(bytes / self.base_size) * self.base_size + self.base_size

        if i == len(param_sizes) - 1:
            # Last param (lm_head) -- not chunk-managed
            total_bytes += bytes
        else:
            # Chunk-based mixed-precision: param/gradient fp16
            if not self.tp:
                if self.gpu_n > 1:  # ZeRO-3 style
                    bytes = bytes * (self.gpu_n - 1) / self.gpu_n
                    if bytes % self.base_size != 0:
                        bytes = int(bytes / self.base_size) * self.base_size + self.base_size
                    total_bytes += bytes
            elif self.tp and self.gpu_n > 1 and self.tp != self.gpu_n:
                # Hybrid DP+TP
                bytes = bytes * ((self.gpu_n - self.tp) / self.gpu_n - 1 / self.gpu_n)
                if bytes % self.base_size != 0:
                    bytes = int(bytes / self.base_size) * self.base_size + self.base_size
                total_bytes += bytes

            # Optimizer parameters (fp32 master copy)
            bytes = np.prod(np.array(s)) * self.bytes * 2  # *2 = fp32 (4 bytes)
            bytes = bytes / self.gpu_n
            if bytes % self.base_size != 0:
                bytes = int(bytes / self.base_size) * self.base_size + self.base_size

            # Momentum fp32 + variance fp32
            total_bytes += 2 * bytes
```

**Important detail from code**: The optimizer states (momentum + variance) are computed per-parameter-tensor, each rounded to CUDA page boundaries independently. This is because they are allocated based on actual tensor sizes, not chunk sizes (see Section 3.2).

### 3.2 Optimizer States Memory (m_os)

```
m_os = SUM over t in {E, L} of: ceil( t_p * (B_32 + B_32) / cu_p ) * cu_p
```

Where:
- `t` iterates over operator types: E = Embedding, L = Linear
- `t_p` = parameter size of operator t
- `(B_32 + B_32)` = 8 bytes per parameter (4 bytes momentum + 4 bytes variance)

**Critical timing insight**: Momentum fp32 and variance fp32 tensors are NOT allocated during initialization. They consume GPU memory based on actual tensor size (not chunk size) and are allocated only during the first fine-tuning iteration. After first allocation, the memory is retained until fine-tuning completes.

**Why per-operator granularity matters**: The system allocates GPU memory based on the actual size of each momentum fp32 and variance fp32 tensor. Because Bias and LayerNorm parameters are very small, their GPU memory consumption can fit within other memory fragmentation. Therefore, only Embedding and Linear operator parameters are counted.

### 3.3 Output Tensor Memory (m_out)

```
m_out = ceil( (e_n + l_n) * (bs * sl * o_n) * B_16 / cu_p ) * cu_p
```

Where:
- `e_n` = number of embedding layers (typically 2: token + positional for OPT-style; 1 for LLaMA-style)
- `l_n` = number of transformer layers
- `o_n` = model output features (hidden dimension d)
- This is stored in fp16

**Gradient checkpointing interaction**: The paper states: "PyTorch provides gradient checkpointing as an option to save memory during fine-tuning. We support estimating GPU memory usage due to each operator's input/output tensors considering gradient checkpointing." The key observation is that "layer and embedding outputs of the transformer model are kept in GPU memory for efficient gradient checkpointing, which minimizes the increase in fine-tuning time."

This means: with gradient checkpointing enabled, each layer's output activation (not the full internal activations) is retained. This is essentially storing 1 tensor of shape `[bs, sl, d]` per layer, in fp16, which is exactly what the formula captures.

**From source code** (`size_estimator.py`, lines 161-172):
```python
def calc_output_bytes(self):
    total_bytes = 0
    for i in range(0, len(self.inout_sizes)):
        self.inout_sizes[i][0] = self.real_bs  # update batch size
    for i in range(1, len(self.inout_sizes)-1):
        s = self.inout_sizes[i]
        bytes = np.prod(np.array(s)) * self.bytes  # self.bytes = 2 (fp16)
        if bytes % self.base_size != 0:
            bytes = int(bytes / self.base_size) * self.base_size + self.base_size
        total_bytes += bytes
```

### 3.4 Language Modeling Head Memory (m_lm)

```
m_lm = ceil(bs * sl * dict_n * B / cu_p) * cu_p
     + 2 * ceil(bs * (sl - 1) * dict_n * B / cu_p) * cu_p
     + lm_p
```

Where:
- `dict_n` = vocabulary size
- `B` = either B_16 (2 bytes) or B_32 (4 bytes) depending on model type
- `lm_p` = lm_head parameter memory in bytes
- The first term: logits tensor `[bs, sl, vocab_size]`
- The second term (with factor 2): shifted logits for loss calculation
- `lm_p`: the lm_head weight parameters themselves

**Why the factor of 2 and (sl-1)**: The lm_head converts transformer outputs into logits. Then, for causal LM loss calculation, the logits are shifted by one position (predicting the next token). The value obtained by shifting the sequence length of the logits by one space is stored in a separate temporary variable and used for the loss calculation. Both the original and shifted versions must coexist in memory simultaneously, hence the factor of 2 on the `(sl-1)` term.

**Model-specific precision (lm_fp32 flag)**: Some models (CodeGen, GPT-Neo) compute the lm_head in fp32 rather than fp16. The code handles this:
```python
# From dp_real.py, lines 341-342:
lm_fp32 = False
if ('codegen' in model_args.model_name_or_path) or ('neo' in model_args.model_name_or_path):
    lm_fp32 = True
```

When `lm_fp32=True`, the bytes per element doubles (B = B_32 = 4 instead of B_16 = 2).

**From source code** (`size_estimator.py`, lines 184-210):
```python
# lm_head and loss function
last_part = 0
s = self.inout_sizes[-1]  # shape: [bs, sl, vocab_size]
if self.lm_fp32:
    bytes = np.prod(np.array(s)) * self.bytes * 2  # fp32
else:
    bytes = np.prod(np.array(s)) * self.bytes       # fp16
if bytes % self.base_size != 0:
    bytes = int(bytes / self.base_size) * self.base_size + self.base_size
last_part += bytes

s[1] -= 1  # sl -> sl-1 for shifted logits
if self.lm_fp32:
    bytes = np.prod(np.array(s)) * self.bytes * 2
else:
    bytes = np.prod(np.array(s)) * self.bytes
if bytes % self.base_size != 0:
    bytes = int(bytes / self.base_size) * self.base_size + self.base_size
last_part += bytes * 2  # factor of 2 for shifted logits
```

---

## 4. Multi-GPU Memory Formulas

### 4.1 Conventional Data Parallelism (CDP)

Peak memory = single GPU peak memory (no savings):
```
m_peak_cdp = m_peak_single
```
Each GPU holds the entire model. Performance score for method selection: `(bs - 1) * gpu_n * 1.5` (the 1.5x factor accounts for ZeRO-3 communication overhead relative to CDP).

### 4.2 Advanced Data Parallelism (ADP) -- ZeRO Stage 3

```
m_peak_dp = m_base + m_p,16 + (m_p,32 + m_os) / gpu_n + m_out + m_lm
```

**Critical insight**: `m_p,16` (the fp16 parameters) is NOT divided by `gpu_n`. This is because during computation, each GPU must have ALL values of param fp16 gathered (all-gather operation) before the forward/backward pass can proceed. Only `m_p,32` (fp32 master copy) and `m_os` (optimizer states) are partitioned across GPUs.

This differs from the theoretical ZeRO-3 formula which suggests everything is sharded. In practice, during the computation phase, the full fp16 parameters must be materialized on each GPU.

**From the code** (`size_estimator.py`, lines 128-134):
```python
# For ZeRO-3 (gpu_n > 1, not TP):
# The fp16 param bytes are reduced by (gpu_n-1)/gpu_n, meaning
# only 1/gpu_n of param fp16 is stored BETWEEN computation phases.
# But during computation, all params are gathered.
bytes = bytes * (self.gpu_n - 1) / self.gpu_n
```

Wait -- the code actually computes `bytes * (gpu_n - 1) / gpu_n` for the fp16 params in ZeRO-3 mode. This represents the additional memory needed to gather params from other GPUs: each GPU already has `1/gpu_n` of params, and must gather the remaining `(gpu_n-1)/gpu_n`. The paper's formula `m_base + m_p,16 + (m_p,32 + m_os)/gpu_n` accounts for the full fp16 params being present during computation (the all-gather phase).

### 4.3 Tensor Parallelism (1D TP)

```
m_peak_tp = m_base + (m_p + m_os) / gpu_n + m_out + m_lm + m_back_tp
```

Where the backward all-gather buffer is:
```
m_back_tp = ceil( l_n * (bs * sl * o_n) * (tp_n - 1) / tp_n * B_16 / cu_p ) * cu_p
```

**Why m_back_tp exists**: In tensor parallelism, each GPU computes only a partial result for each linear operation. During the backward pass, an all-gather operation collects partial outputs from all GPUs, and a temporary buffer is needed to hold the gathered values. The buffer size is proportional to `(tp_n - 1) / tp_n` because each GPU already has its own `1/tp_n` portion and needs the remaining `(tp_n - 1)/tp_n` from other GPUs. This buffer is needed for every layer.

**Key difference from ADP**: With TP, both parameters AND optimizer states are divided by `gpu_n` (not just optimizer states). This is because TP physically splits each parameter tensor, so each GPU only holds `1/tp_n` of each weight matrix at all times.

### 4.4 Hybrid DP+TP

```
m_peak_dp_tp = m_peak_dp - (m_p,16 * tp_n) / gpu_n + m_back_tp
```

This combines ADP (for model state sharding) with TP (for parameter splitting). The `m_p,16 * tp_n / gpu_n` subtraction removes the portion of fp16 params that TP handles.

---

## 5. Distributed Method Selection Algorithm

### Algorithm 1: Optimal Fine-Tuning Method Decision

```
Input: Pre-trained model M, gpu_n, sl
Output: Selected fine-tuning method and optimal bs

1. eval = [0, 0, 0, 0], bs_list = [0, 0, 0, 0]
2. Measure total GPU memory capacity (m_total)
3. For each method i in {0=CDP, 1=ADP, 2=TP, 3=DP+TP}:
   a. Set up method config
   b. bs = 1, compute m_base, m_p, m_os
   c. Compute m_out and m_lm
   d. Repeat bs = bs + 1 until m_peak > m_total
   e. Compute performance score:
      - CDP (i=0): eval[0] = (bs-1) * gpu_n * 1.5
      - ADP (i=1): eval[1] = (bs-1) * gpu_n
      - TP  (i=2): eval[2] = bs - 1
      - DP+TP (i=3): eval[3] = (bs-1) * dp_n
   f. bs_list[i] = bs - 1
4. Select method with maximum eval score
5. Tie-breaking order: CDP > ADP > TP > DP+TP
6. If all scores are 0, suggest CPU offloading
```

**Performance scoring rationale**:
- CDP processes `(bs-1) * gpu_n` samples per step but ZeRO-3 communication adds 1.5x overhead, so effective throughput is `(bs-1) * gpu_n * 1.5` (higher is better for CDP since it avoids ZeRO overhead)
- ADP processes `(bs-1) * gpu_n` samples per step (data parallel scaling)
- TP processes `bs-1` samples per step (only one data stream, all GPUs work on same batch)
- DP+TP processes `(bs-1) * dp_n` samples per step

---

## 6. CUDA Memory Alignment Details

### 6.1 CUDA Page Size

The paper uses `cu_p = 2 * 1024^2` = 2 MiB as the CUDA memory page size. In the source code:
```python
self.base_size = 2 * 1024 * 1024  # 2 MiB
```

Every memory allocation is rounded up to this boundary:
```python
if bytes % self.base_size != 0:
    bytes = int(bytes / self.base_size) * self.base_size + self.base_size
```

### 6.2 PyTorch Internal Alignment

The paper notes: "PyTorch aligns with multiples of 512 bytes for internal tensor fragmentation, and DNNMem treats the buffer size as a constant (64 MB by default) as memory block management."

LLMem uses the 2 MiB page size instead of either of these, which is the CUDA memory page granularity used by Colossal-AI's Gemini plugin for chunk allocation.

### 6.3 Chunk Size Determination

From the Colossal-AI code (`search_utils.py`), the chunk size search:
1. Classifies parameters by data-parallel degree
2. Filters out extremely large parameters (> mean + 3*std)
3. Searches for optimal chunk size that minimizes wasted space
4. The search starts from `max(max_parameter_size, min_chunk_size)` and searches in intervals of `search_interval_byte` (typically the hidden dimension)
5. For each candidate chunk size, computes unused bytes across all parameter groups
6. Selects the chunk size with minimum waste

---

## 7. What Existing Methods Get Wrong (and LLMem Fixes)

### 7.1 DNNMem Failures

DNNMem (Gao et al., 2020) shows an average error of **42.6%** when applied to LLM fine-tuning. Three specific failures:

1. **Does not handle mixed precision**: Fine-tuning uses fp16 parameters + fp32 master copies. DNNMem assumes uniform precision, underestimating the dual-copy overhead.

2. **Does not consider chunk-based memory management**: In Colossal-AI's Gemini plugin (and similar ZeRO implementations), parameters and gradients share the same memory chunk. DNNMem separately allocates memory for parameters AND gradients, double-counting.

3. **Overlooks first-iteration optimizer state memory**: Momentum and variance tensors are allocated during the first training iteration (not at model init time). DNNMem's pre-execution estimation misses this allocation entirely.

### 7.2 Specific Error Rates

| Model | LLMem Error | DNNMem Error |
|-------|-------------|--------------|
| OPT-125m | 0.4% | 36.5% |
| OPT-350m | 1.6% | 42.5% |
| bloom-560m | 1.6% | 34.3% |
| codegen-350M | 0.8% | 57.1% |
| **Average** | **~1.1%** | **~42.6%** |

The irony: DNNMem consistently **underestimates** memory (not overestimates). This is worse from a practical standpoint because it would falsely suggest that training will fit in memory when it won't.

---

## 8. Experimental Validation

### 8.1 Hardware and Software

- **GPU**: Tesla V100 16GB (total capacity: 16,384 MB)
- **GPU count**: 4 GPUs on CloudLab
- **Framework**: Colossal-AI with Gemini plugin
- **PyTorch**: 2.0.1
- **CUDA**: 11.7
- **Optimizer**: HybridAdam (Colossal-AI's fused Adam)
- **Precision**: fp16
- **Gradient checkpointing**: Enabled for all experiments
- **Dataset**: Stanford Alpaca (52K instruction-following samples)
- **Sequence length**: 512 (maximum)

### 8.2 Models Tested

OPT (125m, 350m, 1.3b, 2.7b), BLOOM (560m, 1b1, 3b), CodeGen (350M, 2B), BioGPT (Large), GPT-BigCode (santacoder), GPT-Neo (1.3B), LLaMA (7b).

### 8.3 Single-GPU Results (Table 2)

| Model | LLMem (MB) | DNNMem (MB) | Ground Truth (MB) | LLMem Error |
|-------|-----------|-----------|-----------------|------------|
| OPT-125m | 16,314 | 10,402 | 16,378 | 0.4% |
| OPT-350m | 16,004 | 9,354 | 16,264 | 1.6% |
| bloom-560m | 16,578 | 10,726 | 16,324 | 1.6% |
| codegen-350M | 16,236 | 6,910 | 16,100 | 0.8% |

### 8.4 Multi-GPU Method Selection (Table 3)

| Model | LLMem Selection | 4DP Time(s) | 2DP+2TP Time(s) | 4TP Time(s) |
|-------|-----------------|-------------|------------------|-------------|
| OPT-1.3b | 4DP | **688** | 1,616 | 2,186 |
| OPT-2.7b | 4TP | OOM | 8,174 | **6,038** |
| bloom-1b1 | 4DP | **680** | 1,724 | 2,631 |
| bloom-3b | 4TP | OOM | OOM | **14,495** |
| BioGPT-Large | 4DP | **1,022** | 3,315 | 4,773 |
| codegen-2B-nl | 4TP | OOM | 6,314 | **6,244** |
| gpt_bigcode | 4DP | **651** | 1,292 | 1,652 |
| gpt-neo-1.3B | 4DP | **768** | 1,686 | 2,372 |
| llama-7b | CPU offload | OOM | OOM | OOM |

Bold = fastest valid option. LLMem correctly identifies the best method in all cases.

### 8.5 Multi-GPU Error Rates

**ADP (ZeRO-3) error rates on 4 GPUs** (from Figure 8):
- OPT-350m: ~0.8%
- OPT-1.3b: ~1.8%
- bloom-560m: ~0.9%
- bloom-1b1: ~1.6%
- codegen-350M: ~1.1%
- GPTBigCode-1.12b: ~6.0%
- Average: ~3.0%

**TP and DP+TP error rates on 4 GPUs** (from Figure 9):
- OPT-2.7b: 4TP ~1.1%, 2DP+2TP ~2.1%
- bloom-1b7: ~1.5%
- codegen-2B: ~2.5%
- BioGPT-Large: ~3.6%
- gpt-neo-1.3B: ~0.9%, 2DP+2TP ~7.4%

The errors increase on multi-GPU due to: (1) memory usage gaps between GPUs, (2) memory allocator characteristics of larger models, and (3) temporary buffer allocation variability.

---

## 9. Novel Insights and Implementation-Relevant Details

### 9.1 Gradient/Parameter Memory Sharing

The single most important insight for a memory calculator: in chunk-based memory managers (ZeRO/FSDP), **param fp16 and gradient fp16 share the same allocated memory**. The gradient overwrites the parameter in-place during the backward pass. This means the standard formula of "2 bytes for params + 2 bytes for gradients = 4 bytes per parameter" is WRONG for chunk-based systems. The correct accounting is 2 bytes for the shared fp16 chunk (holding either params or gradients at any given time) + 4 bytes for fp32 master copy = 6 bytes, not 8.

### 9.2 m_base Measurement (Not Calculated)

The paper does NOT provide a formula for `m_base`. It is measured empirically at runtime:
```python
# From dp_real.py, lines 266-274:
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
total_nvml = int(info.total / (1024 * 1024))
used_nvml = int(info.used / (1024 * 1024))
cuda_context_mem = used_nvml - GPUtil.getGPUs()[dist.get_rank()].memoryUsed
framework_initial_mem = GPUtil.getGPUs()[dist.get_rank()].memoryUsed

# After model loading and booster.boost():
booster_chunk_mem = GPUtil.getGPUs()[dist.get_rank()].memoryUsed
m_pbase = booster_chunk_mem + cuda_context_mem - (after_get_output - prev_get_output)
```

For a calculator that doesn't have access to the actual GPU, `m_base` would need to be estimated. Typical values from the data: for V100 with Colossal-AI, CUDA context alone is ~300-800 MB, and chunk manager overhead adds more depending on model size.

### 9.3 lm_head Memory is Treated Differently from Transformer Body

The language modeling head (final linear projection to vocabulary) does NOT use chunk-based memory management. It uses "real-size-based" memory allocation. This is because:
1. The lm_head produces a very large output tensor (shape `[bs, sl, vocab_size]`)
2. The loss calculation requires both the original logits AND a shifted version simultaneously
3. The temporary memory for loss computation is significant when vocabulary is large (e.g., 128K for LLaMA 3)

### 9.4 Per-Operator Optimizer State Rounding

Optimizer states (momentum, variance) are allocated per-tensor, each independently rounded to CUDA page boundaries. This is different from parameters which are grouped into chunks. The sum of individually-rounded allocations can be significantly larger than a single rounded allocation of the total, especially for models with many small layers.

### 9.5 Model-Specific lm_head Precision

Some model architectures compute the lm_head in fp32 rather than fp16 (CodeGen, GPT-Neo). This doubles the lm_head memory requirement. A general calculator should either auto-detect this or allow users to specify it.

### 9.6 The Vocabulary Size Dominates lm_head Memory

For large vocabulary models, `m_lm` can be the dominant memory term during the loss computation phase. Consider LLaMA 3 with V=128,256, bs=1, sl=512:
- Logits: 1 * 512 * 128,256 * 2 = ~125 MB
- Shifted logits (x2): 2 * 1 * 511 * 128,256 * 2 = ~250 MB  
- Total m_lm portion (just tensors): ~375 MB

With larger batch sizes, this grows linearly with bs.

---

## 10. Applicability to Our Calculator Spec

### 10.1 What LLMem Adds That Our Spec Currently Misses

1. **CUDA memory page alignment rounding**: Our spec uses exact byte calculations. Real GPU memory is allocated in 2 MiB pages. For small models or per-tensor accounting, the rounding error is significant (up to several hundred MB). This is mentioned nowhere in our current spec.

2. **lm_head/loss computation peak memory**: Our spec accounts for activation memory using the Korthikanti formula but does NOT separately model the peak memory from the loss computation, where `[bs, sl, vocab_size]` logits plus shifted logits must coexist. For large vocabularies, this is non-trivial.

3. **Parameter/gradient sharing in ZeRO**: Our spec correctly uses Phi*Psi for model states, but does not note that in chunk-based implementations (FSDP, DeepSpeed ZeRO), gradient memory and parameter memory may overlap rather than being additive. This could lead to slight overestimation.

4. **First-iteration memory spike from optimizer states**: Our spec mentions DeepSpeed's initialization spike but not the optimizer state allocation spike on the first training step.

5. **Per-operator rounding of optimizer states**: Small but cumulative source of memory overhead that exact calculations miss.

### 10.2 What Our Spec Already Covers Better

1. **Activation memory formulas**: Our spec has the Korthikanti formula (34 + 5*a*s/d coefficient), Flash Attention corrections, selective checkpointing, and AMP autocast corrections. LLMem uses a simpler output-tensor-only model that only works with gradient checkpointing enabled.

2. **Pipeline parallelism**: Our spec covers PP with interleaved scheduling and bubble overhead. LLMem does not cover PP at all.

3. **LoRA/QLoRA/PEFT**: Not covered by LLMem at all.

4. **MoE models**: Not covered by LLMem.

5. **Different optimizers**: LLMem only covers Adam. Our spec covers SGD, Adafactor, Lion, LAMB, etc.

6. **ZeRO stages 1 and 2**: LLMem only covers ZeRO-3 (Colossal-AI Gemini) and basic DP. Our spec covers all stages.

7. **Communication overhead modeling**: Our spec has detailed communication volume formulas. LLMem uses the heuristic 1.5x multiplier for ZeRO-3.

### 10.3 Specific Formula Improvements to Consider

**1. Loss computation peak memory term**: Add to our spec:
```
M_loss_peak = bs * sl * V * beta        # logits tensor
            + 2 * bs * (sl-1) * V * beta # shifted logits for loss (x2)
            + V * d * beta               # lm_head weight parameters
```
This is transient (only during the loss computation) but sets the peak.

**2. CUDA memory page alignment**: Add a note or option:
```
M_aligned = ceil(M_raw / 2 MiB) * 2 MiB
```
Applied per allocation, not to the total. This is most relevant for the "accuracy refinement" mode of our calculator.

**3. ZeRO-3 all-gather materialization**: During computation in ZeRO-3/FSDP, full fp16 parameters must be gathered. Our spec already captures this via the FSDP mixed precision formula (`M_fsdp_mixed`), but we could make it more explicit.

### 10.4 Limitations of LLMem

1. **Framework-specific**: Heavily tied to Colossal-AI's Gemini plugin. The chunk sizes, memory management patterns, and allocation strategies are specific to this framework.

2. **Only V100 16GB validation**: All experiments on a single GPU type. Modern training uses A100/H100 with different memory characteristics.

3. **No pretraining support**: Fine-tuning only.

4. **No PEFT support**: Full fine-tuning only.

5. **Gradient checkpointing assumed**: All formulas assume gradient checkpointing is enabled. Without it, the activation memory model would need revision.

6. **m_base requires measurement**: The base memory cannot be calculated analytically -- it must be measured on the actual hardware/framework combination.

7. **Limited model architectures**: Validated on decoder-only transformers (OPT, BLOOM, CodeGen, GPT-Neo, LLaMA, BioGPT, GPT-BigCode). Encoder-decoder or encoder-only architectures are mentioned (BERT) but not fully validated.

---

## 11. Key Takeaway for Our Calculator

The most practically useful insight from LLMem is **NOT** its specific formulas (which are framework-specific) but rather its identification of memory components that standard calculators underestimate:

1. **Loss computation memory** (`m_lm`) as a distinct, potentially dominant peak memory contributor -- especially for large-vocabulary models
2. **CUDA allocation granularity** causing real memory to exceed theoretical calculations by a nontrivial amount
3. **Param/gradient memory sharing** in modern chunk-based ZeRO implementations
4. **Optimizer state allocation timing** (first iteration, not initialization) as a peak memory contributor

For our calculator, the most actionable addition would be the `m_lm` (loss computation peak memory) term, which is vocabulary-size-dependent and not captured by standard activation memory formulas.

---

## Relevant Source Files

- `/tmp/LLMem/size_estimator.py` -- Core memory estimation implementation (317 lines)
- `/tmp/LLMem/dp_real.py` -- Data parallel training + estimation invocation
- `/tmp/LLMem/tp_real.py` -- Tensor parallel training + estimation invocation  
- `/tmp/LLMem/colossalai/zero/gemini/chunk/search_utils.py` -- Chunk size search algorithm
- `/tmp/LLMem/colossalai/zero/gemini/chunk/chunk.py` -- Chunk memory management implementation
