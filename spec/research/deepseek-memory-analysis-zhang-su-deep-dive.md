# Deep Dive: Memory Analysis on the Training Course of DeepSeek Models

**Paper**: Zhang, P. & Su, L. (2025). "Memory Analysis on the Training Course of DeepSeek Models." arXiv:2502.07846v1, Baichuan-Inc.
**URL**: https://arxiv.org/abs/2502.07846
**Date**: February 11, 2025
**Focus**: Theoretical GPU memory analysis for training DeepSeek-v2/v3 MoE models with 3D parallelism + Expert Parallelism + ZeRO

---

## 1. Executive Summary

This paper presents a detailed, bottom-up GPU memory consumption analysis for training DeepSeek-v3 (671B total parameters, 61 transformer layers, 256 routed experts per MoE layer). It is authored by researchers at Baichuan-Inc (not DeepSeek), and the training configurations discussed are illustrative, not DeepSeek's actual production settings. The analysis covers:

- Layer-level parameter counting for MoE + Multi-head Latent Attention (MLA) architecture
- Per-device static parameter memory under combined TP/PP/EP/ETP parallelism
- ZeRO (os, os+g, os+g+params) memory with separate sharding for MoE vs non-MoE parameters
- Detailed activation memory formulas for MLA and MoE layers, with and without recomputation
- Temporal buffer and fragmentation overhead

The paper's primary value for a GPU calculator is:
1. MLA-specific activation memory formulas (not found in standard transformer calculators)
2. MoE activation memory formulas that account for expert routing and token distribution
3. Explicit treatment of how MoE params and non-MoE params use DIFFERENT ZeRO sharding denominators (DP vs EDP)
4. Concrete worked examples with DeepSeek-v3 architecture that can serve as validation targets

**Disclaimer from authors**: "The training policies discussed in this report are not representative of DeepSeek's official configurations."

---

## 2. Architecture: DeepSeek-v3 Structure Configuration

### 2.1 Notation Table (Table 1 in paper)

| Symbol | Meaning | DeepSeek-v3 Value |
|--------|---------|-------------------|
| h | Hidden dimension | 7,168 |
| h_E | MoE intermediate dimension (per expert) | 2,048 |
| h_F | Non-MoE intermediate dimension (dense FFN) | 18,432 |
| d_h | Dimension per attention head (for RoPE) | 128 |
| n_h | Number of attention heads | 128 |
| d_cq | Query compression dimension (LoRA rank for Q) | 1,536 |
| d_hr | Per-head RoPE dimension for Q/K | 64 |
| d_c | Key-value compression dimension (LoRA rank for KV) | 512 |
| N | Number of routed experts per MoE layer | 256 |
| N_s | Number of shared experts per MoE layer | 1 |
| l | Number of transformer layers | 61 |
| v | Vocabulary size | 129,280 |

### 2.2 Layer Architecture (Hybrid Dense + MoE)

DeepSeek-v3 has a hybrid layer structure:
- **Layer 0**: Embedding + MLA + standard dense FFN (h_F=18,432) + LayerNorm
- **Layers 1-2**: MLA + standard dense FFN + LayerNorm
- **Layers 3-59**: MLA + MoE (256 routed experts + 1 shared expert) + LayerNorm (57 MoE layers)
- **Layer 60**: MLA + MoE + Head (output projection)

Key: Only 3 layers use dense FFN; the remaining 58 layers use MoE. The first layer also contains the embedding.

### 2.3 Multi-head Latent Attention (MLA) Architecture

MLA is a compressed attention mechanism unique to DeepSeek-v2/v3. Instead of standard Q/K/V projections, it uses low-rank compression:

**MLA computation flow** (equations 37-47 in paper):

Query path:
```
c_Q = W^DQ * h_t           # Compress hidden to d_cq dimensions [d_cq, h]
[q_C_1, ..., q_C_nh] = W^UQ * c_Q   # Decompress to per-head Q [d_h * n_h, d_cq]
q_R = RoPE(W^QR * c_Q)     # RoPE component [d_hr * n_h, d_cq]
q_{i,j} = [q_C_j, q_R_j]  # Concatenate content + RoPE queries
```

Key-Value path:
```
c_KV = W^DKV * h_t          # Compress hidden to d_c dimensions [d_c, h]
[k_C_1, ..., k_C_nh] = W^UK * c_KV  # Decompress to per-head K [d_h * n_h, d_c]
k_R = RoPE(W^KR * h_t)      # RoPE component (note: from h_t directly) [d_hr, h]
k_{i,j} = [k_C_j, k_R_j]   # Concatenate content + RoPE keys
[v_C_1, ..., v_C_nh] = W^UV * c_KV  # Decompress to per-head V [d_h * n_h, d_c]
```

Attention:
```
o_{t,j} = sum_i Softmax(q^T_{t,j} * k_{i,j} / sqrt(d_h + d_hr)) * v_C_{i,j}
u_t = W^O * [o_{t,1}, o_{t,2}, ..., o_{t,nh}]   # Output projection [h, d_h * n_h]
```

### 2.4 MLA Parameter Matrices (Table 2 in paper)

| Component | Matrix | Shape | DeepSeek-v3 |
|-----------|--------|-------|-------------|
| MLA | W^DQ | [d_cq, h] | [1536, 7168] |
| MLA | W^UQ | [d_h * n_h, d_cq] | [16384, 1536] |
| MLA | W^QR | [d_hr * n_h, d_cq] | [8192, 1536] |
| MLA | W^DKV | [d_c, h] | [512, 7168] |
| MLA | W^UK | [d_h * n_h, d_c] | [16384, 512] |
| MLA | W^KR | [d_hr, h] | [64, 7168] |
| MLA | W^UV | [d_h * n_h, d_c] | [16384, 512] |
| MLA | W^O | [h, d_h * n_h] | [7168, 16384] |
| MoE expert | gate_proj | [h, h_E] | [7168, 2048] |
| MoE expert | up_proj | [h, h_E] | [7168, 2048] |
| MoE expert | down_proj | [h_E, h] | [2048, 7168] |

MLA total parameters per layer = 187,107,328 (compared to standard MHA which would be ~4*h^2 = ~205M for h=7168 with MHA). MLA is slightly more parameter-efficient.

### 2.5 MoE Expert Structure

Each MoE layer contains:
- **Router/Gate**: weight shape [N, h] = [256, 7168] = 1,835,008 params
- **N routed experts**: each with 3 projections (gate_proj, up_proj, down_proj), each expert = 3 * h * h_E = 3 * 7168 * 2048 = 44,040,192 params
- **N_s shared experts**: same structure as routed expert, replicated on all ranks
- Per MoE layer total: router + 256 routed experts + 1 shared expert = 1,835,008 + 257 * 44,040,192 = 11,320,164,352 params

---

## 3. Model Parameter Counting

### 3.1 Layer-Level Counting (Table 3 in paper)

| Layer(s) | Components | Parameters Per Layer | Total GB (BF16) |
|----------|-----------|---------------------|-----------------|
| Layer 0 | Embedding [129280,7168] + MLA + MLP [3*7168*18432] + LN | 1.5B | 2.8 |
| Layers 1-2 | MLA + MLP [3*7168*18432] + LN | 0.58B each | 1.1 each |
| Layers 3-59 | MLA + Gate [256,7168] + MoE [3*7168*2048*257] + LN | 11.5B each | 21.44 each |
| Layer 60 | MLA + Gate + MoE + LN + Head [7168,129280] | 12.4B | 23.16 |
| **Total** | | **671B** | **1,250 GB** |

Key observations:
- LayerNorm parameters per layer: 2*h + 2*d_cq + 2*d_c = 2*7168 + 2*1536 + 2*512 = 16,384 (the "2*" is for the two RMSNorm operations per transformer block, each having scale parameters for both the hidden state and the compressed attention dimensions)
- Wait -- the paper says LN shape is "2*7168+1536+512" = 16,384. This means per layer there are RMSNorm weights of size h (before attention) + h (before FFN/MoE) + d_cq + d_c = 7168 + 7168 + 1536 + 512 = 16,384.
- Word embeddings are NOT tied with the output head (separate parameters).

### 3.2 Pipeline Stage Assignment (Table 4)

With PP=16, the paper uses DeepSeek's configuration:
- **Stage 0**: 4 layers (layers 0-3), 14.16B params, 26 GB
- **Stages 1-14**: 4 layers each (all MoE), 46B params each, 86 GB each
- **Stage 15**: 1 layer (layer 60, with head), 12.4B params, 23 GB
- **Total**: 61 layers, 671B params, 1,250 GB

The bottleneck stage is stages 1-14 at 86 GB each (before parallelism sharding).

---

## 4. Static Parameter Memory Per Device (Section 3 in paper)

### 4.1 Reference Parallelism Configuration (Table 5)

| Parallelism | Value |
|-------------|-------|
| DP | 32 |
| TP | 2 |
| PP | 16 |
| EP | 8 |
| ETP (expert tensor parallelism) | 1 |
| EDP (expert data parallelism) | 8 |

Relationship: Total GPUs = DP * TP * PP = 32 * 2 * 16 = 1,024. Also EP * ETP * EDP = 8 * 1 * 8 = 64, and DP * TP / EP / ETP = 32 * 2 / 8 / 1 = 8 = EDP.

### 4.2 RMSNorm Parameters (Section 3.1)

RMSNorm is NOT partitioned by TP -- it is replicated across all TP ranks:
```
RMSNorm per layer = 2*h + d_cq + d_c = 2*7168 + 1536 + 512 = 16,384 params
RMSNorm total (4 layers per stage) = 16,384 * 4 = 65,536 params
Memory = 65,536 * 2 bytes = 131,072 bytes (128 KB)
```

### 4.3 MLA Parameters with TP (Section 3.2)

With TP=2, the following MLA matrices are SPLIT across TP ranks:
- W^UQ, W^UK, W^UV, W^O (column/row parallel)

The following are REPLICATED on each TP rank (not split):
- W^DQ, W^DKV, W^QR, W^KR

**TP-partitioned parameters per rank** (for 4 layers in a PP stage):
```
(16384*1536 + 16384*512*2 + 7168*16384) * 4 / 2 = 318,767,104
```
Breakdown: (W^UQ: 16384*1536) + (W^UK: 16384*512) + (W^UV: 16384*512) + (W^O: 7168*16384) = 79,691,776 per layer, times 4 layers, divided by TP=2.

**Replicated parameters per rank** (for 4 layers):
```
(1536*7168 + 512*7168 + 8192*1536 + 64*7168) * 4 = 110,886,912
```
Breakdown: (W^DQ: 1536*7168) + (W^DKV: 512*7168) + (W^QR: 8192*1536) + (W^KR: 64*7168) per layer.

**Total MLA per rank**: 318,767,104 + 110,886,912 = 429,654,016 params
**Memory**: 429,654,016 * 2 = 859,308,032 bytes (819.5 MB)

### 4.4 MoE Parameters with EP (Section 3.3)

Under PP16 @ EP8 @ ETP1 configuration:
- Router parameters (gate) are NOT partitioned by EP -- replicated on all ranks: 256 * 7168 = 1,835,008 per layer
- 256 routed experts / EP8 = 32 routed experts per EP rank
- 1 shared expert is replicated on ALL ranks (not distributed by EP)
- With ETP=1, individual expert matrices are not split by tensor parallelism

**Experts per EP rank** (for 4 layers in a PP stage):
```
4 layers * (32 routed + 1 shared) = 132 experts
Expert params = 132 * 3 * 7168 * 2048 = 5,813,305,344
```

**Router per rank** (for 4 layers):
```
1,835,008 * 4 = 7,340,032
```

**Total MoE per rank**: 5,813,305,344 + 7,340,032 = 5,820,645,376 params
**Memory**: 5,820,645,376 * 2 = 11,641,290,752 bytes (10.84 GB)

### 4.5 Total Static Parameters Per Device (Table 6)

| Module | Params Per Device | Bytes | GB |
|--------|-------------------|-------|-----|
| RMSNorm 1&2 | 65,536 | 131,072 | ~0 |
| MLA | 429,654,016 | 859,308,032 | 0.82 |
| Non-MoE Part | 429,719,552 | 859,439,104 | 0.82 |
| MoE | 5,820,645,376 | 11,641,290,752 | 10.84 |
| **Total** | **6,250,364,928** | **12,500,729,856** | **11.64** |

This is a critical result: from 671B total params, each GPU stores only 6.25B params (11.64 GB in BF16) under PP16@TP2@EP8@ETP1.

### 4.6 General Formula for Static Parameters Per Device

The paper implicitly uses:
```
Params_per_device = Params_non_moe / (TP * PP) + Params_moe / (EP * ETP * PP)
                  + Params_replicated  (RMSNorm, router, shared experts)
```

Where:
- Non-MoE params (MLA weights that are TP-split) are divided by TP
- Non-MoE replicated params (MLA weights not split, RMSNorm) are kept full
- MoE routed expert params are divided by EP * ETP
- MoE shared expert params and router params are replicated
- Everything is divided by PP (pipeline stages)

More precisely for DeepSeek-v3's configuration:
```
Params_MLA_per_device = (TP_split_MLA_params / TP + replicated_MLA_params) * layers_per_stage
Params_MoE_per_device = (N/EP + N_s) * params_per_expert * layers_per_stage
                      + router_params * layers_per_stage
Params_RMSNorm_per_device = rmsnorm_params_per_layer * layers_per_stage

Total = Params_MLA + Params_MoE + Params_RMSNorm
```

---

## 5. ZeRO Memory Analysis (Section 4)

### 5.1 Data Types (Table 7)

| Component | Format | Bytes Per Param/Value |
|-----------|--------|----------------------|
| Weights | BF16 | 2 |
| Activations | BF16 | 2 |
| Gradients | FP32 | 4 |
| Optimizer - Copy of parameters | FP32 | 4 |
| Optimizer - Momentum | BF16 | 2 |
| Optimizer - Variance | BF16 | 2 |

**IMPORTANT DIFFERENCE FROM STANDARD CALCULATORS**: This paper uses BF16 momentum and BF16 variance (2+2=4 bytes), NOT the standard FP32 momentum and FP32 variance (4+4=8 bytes). The total optimizer state is 4+2+2 = 8 bytes per parameter, not the standard 12 bytes per parameter. This is specific to DeepSpeed's implementation choice for DeepSeek training and differs from the standard AdamW mixed precision accounting.

Total per-parameter cost WITHOUT ZeRO:
```
Weights: 2 bytes (BF16)
Gradients: 4 bytes (FP32)  
Optimizer: 4 (master FP32 copy) + 2 (BF16 momentum) + 2 (BF16 variance) = 8 bytes
TOTAL = 2 + 4 + 8 = 14 bytes per parameter (not the standard 18)
```

But the paper then calculates baseline gradient and optimizer memory using different multipliers:
- Gradients (no ZeRO): 6,250,364,928 * 4 = 23.3 GB (i.e., 4 bytes per param for FP32 gradients)
- Optimizer (no ZeRO): 6,250,364,928 * 8 = 46.6 GB (i.e., 8 bytes per param for optimizer states: 4 FP32 copy + 2 BF16 m + 2 BF16 v)

Wait -- that does not match. 6,250,364,928 * 8 / (1024^3) = 46.6 GB. But the paper says "23.3 GB (6,250,364,928 x 4) for optimizer states, and 46.6 GB (6,250,364,928 x 8) for gradients." That seems like a typo -- the paper text at the bottom of page 6 says "23.3 GB (6,250,364,928 x 4) for optimizer states, and 46.6 GB (6,250,364,928 x 8) for gradients" but then uses the opposite multipliers in the ZeRO formulas. Let me re-read.

Actually, examining Table 8 carefully: with ZeRO=None, Static Parameters=11.64 GB, Gradients=23.3 GB, Optimizer=46.6 GB, P+G+O=81.54 GB. 

11.64 + 23.3 + 46.6 = 81.54. Checks out.

Gradients at 23.3 GB: 6,250,364,928 * 4 / 1024^3 = 23.28 GB. So gradients = 4 bytes each (FP32).
Optimizer at 46.6 GB: 6,250,364,928 * 8 / 1024^3 = 46.57 GB. So optimizer = 8 bytes each.

So the optimizer states = 8 bytes per param (master weights FP32 = 4 + momentum BF16 = 2 + variance BF16 = 2).

### 5.2 ZeRO with Separate MoE vs Non-MoE Sharding

The critical insight: MoE parameters and non-MoE parameters have DIFFERENT data parallelism degrees:
- Non-MoE parameters: sharded across DP = 32 GPUs
- MoE parameters: sharded across EDP = 8 GPUs (because EP already distributes experts)

The per-device parameter counts are:
- Non-MoE params per device = 429,719,552 (MLA + RMSNorm)
- MoE params per device = 5,820,645,376

**ZeRO-os (Stage 1) -- optimizer states sharded**:
```
Optimizer per device = (non_moe_params/DP + moe_params/EDP) * 8
                     = (429,719,552/32 + 5,820,645,376/8) * 8
                     = (13,428,736 + 727,580,672) * 8
                     = 741,009,408 * 8
                     = 5,928,075,264 bytes = 5.52 GB
```

Static params unchanged at 11.64 GB. Gradients unchanged at 23.3 GB.
Total = 11.64 + 23.3 + 5.52 = 40.46 GB

**ZeRO-os+g (Stage 2) -- optimizer states + gradients sharded**:
```
Gradients per device = (429,719,552/32 + 5,820,645,376/8) * 4 = 2.76 GB
Optimizer per device = 5.52 GB (same as above)
```
Static params unchanged at 11.64 GB.
Total = 11.64 + 2.76 + 5.52 = 19.92 GB

**ZeRO-os+g+params (Stage 3) -- everything sharded**:
```
Params per device = (429,719,552/32 + 5,820,645,376/8) * 2 = 1.38 GB
Gradients per device = 2.76 GB
Optimizer per device = 5.52 GB
```
Total = 1.38 + 2.76 + 5.52 = 9.66 GB

### 5.3 ZeRO Summary Table (Table 8)

| ZeRO Level | Static Params | Gradients | Optimizer | Total (P+G+O) |
|------------|--------------|-----------|-----------|----------------|
| None | 11.64 GB | 23.3 GB | 46.6 GB | 81.54 GB |
| os (Stage 1) | 11.64 GB | 23.3 GB | 5.52 GB | 40.46 GB |
| os+g (Stage 2) | 11.64 GB | 2.76 GB | 5.52 GB | 19.92 GB |
| os+g+params (Stage 3) | 1.38 GB | 2.76 GB | 5.52 GB | 9.66 GB |

### 5.4 Generalized ZeRO Formula for MoE Models

```
ZeRO-os:
  M_params     = (P_nonmoe + P_moe) * 2
  M_gradients  = (P_nonmoe + P_moe) * bytes_grad
  M_optimizer  = (P_nonmoe/DP + P_moe/EDP) * bytes_opt

ZeRO-os+g:
  M_params     = (P_nonmoe + P_moe) * 2
  M_gradients  = (P_nonmoe/DP + P_moe/EDP) * bytes_grad
  M_optimizer  = (P_nonmoe/DP + P_moe/EDP) * bytes_opt

ZeRO-os+g+params:
  M_params     = (P_nonmoe/DP + P_moe/EDP) * 2
  M_gradients  = (P_nonmoe/DP + P_moe/EDP) * bytes_grad
  M_optimizer  = (P_nonmoe/DP + P_moe/EDP) * bytes_opt
```

Where:
- P_nonmoe = non-MoE parameters per device (already divided by TP and PP)
- P_moe = MoE parameters per device (already divided by EP, ETP, and PP)
- DP = data parallelism degree
- EDP = expert data parallelism degree = DP * TP / (EP * ETP)
- bytes_grad = 4 (FP32 gradients)
- bytes_opt = 8 (FP32 master + BF16 momentum + BF16 variance) or 12 (standard FP32 Adam)

---

## 6. Activation Memory Analysis (Section 5)

### 6.1 Configuration for Activation Analysis (Table 9)

| Symbol | Meaning | Value |
|--------|---------|-------|
| b | Micro-batch size | 1/2/4 |
| s | Sequence length | 4096 |
| N_r | Routed experts per token | 8 |
| N | Total experts per MoE layer | 256 |
| E_token | Avg tokens per expert = b*s*N_r/N | b*s*8/256 |
| SP | Sequence parallelism | On, 2 (= TP) |
| CP | Context parallelism | 1 |
| AC | Activation checkpointing | None or Full |

### 6.2 MLA Activation Memory (Section 5.1)

The paper provides a detailed activation flow diagram (Figure 2) tracing every intermediate tensor through the MLA computation.

**Without any parallelism (no TP, no SP), single layer, in bytes**:
```
M_MLA_no_parallel = 4bsh + 2bs(d_cq + d_c) + 4bs(d_h + d_hr)*n_h + 2bs(d_h*n_h)
                  + 5bn_h*s^2 + 2bs(d_h*n_h) + bsh
```

Breaking down each term:
- `4bsh`: Two RMSNorm inputs (2*2bsh = 4bsh -- one before attention, stored for backward)
- `2bs*d_cq`: Compressed query c_Q output
- `2bs*d_c`: Compressed KV c_KV output  
- `4bs(d_h + d_hr)*n_h`: Decompressed Q and K (with RoPE component) -- 2bs*(d_h+d_hr)*n_h for Q, same for K
- `2bs*d_h*n_h`: V values
- `5bn_h*s^2`: Attention scores (pre-softmax logits 2, softmax output 2, dropout mask 1) = 5*b*n_h*s^2 bytes
- `2bs*d_h*n_h`: Attention output (before output projection)
- `bsh`: Dropout mask after output projection

**With TP=2 and SP=2 (TP2@SP2@CP1), single layer**:
```
M_1^A = 4bsh/2 + 2bs(d_cq + d_c) + 4bs(d_h + d_hr)*n_h/2 + 2bs(d_h*n_h)/2
      + 5bn_h*s^2/2 + 2bs(d_h*n_h)/2 + bsh/2
```

Key: the term `2bs(d_cq + d_c)` is NOT divided by SP/TP because the corresponding weights (W^DQ, W^DKV, W^QR, W^KR) are replicated across all TP ranks. Everything else divides by 2 (= TP = SP).

**For a 4-layer PP stage (the paper's standard unit)**:
```
4*M_1^A = 10bsh + 8bs(d_cq + d_c) + 16bs*d_h*n_h + 8bs*d_hr*n_h + 10bn_h*s^2
```

**With full recomputation, single layer (TP2@SP2@CP1)**:
```
M_2^A = 2bsh/2 = bsh  (only store the layer input before RMSNorm)
```

**For 4 layers with full recomputation**:
```
4*M_2^A = 4bsh
```

### 6.3 MoE Activation Memory (Section 5.2)

**Token distribution formula**:
```
E_token = b * s * N_r / N
```
This assumes balanced expert load. For DeepSeek-v3: E_token = b * 4096 * 8 / 256 = 128*b tokens per expert.

**Without recomputation, single MoE layer (SP2@EP8@ETP1), in bytes**:
```
M_1^E = 4bsh/2 + 4bsN + 2bsN_r + 32*(3*E_token*h + 8*E_token*h_E) + 1*(3bsh + 8bsh_E)
```

The `4bsh/2` is the RMSNorm input (with SP). The `4bsN` is router logits (2bsN) + router probabilities (2bsN). The `2bsN_r` is the dispatch mask. The `32*...` is for the 32 routed experts per rank (EP=8), each processing E_token tokens with gate_proj, up_proj activations. The `1*...` is the shared expert.

Simplified:
```
M_1^E = 5bsh + 4bsN + 2bsN_r + bs*N_r/N*(96h + 256h_E) + 8bsh_E
```

Wait, let me re-derive from the paper's simplification. The paper says:
```
M_1^E = 4bsh/2 + 4bsN + 2bsN_r + 32*(3*E_token*h + 8*E_token*h_E) + 1*(3bsh + 8bsh_E)
```

Where the 32 routed experts each process E_token = bs*N_r/N tokens:
```
32 * (3*E_token*h + 8*E_token*h_E) = 32 * E_token * (3h + 8h_E)
= 32 * (bs*N_r/N) * (3h + 8h_E)
= bs * N_r/N * 32 * (3h + 8h_E)
```
But wait, 32 = N/EP = 256/8. So this is: bs * N_r/N * (N/EP) * (3h + 8h_E) = bs * N_r/EP * (3h + 8h_E).

Hmm, but the paper simplifies to:
```
M_1^E = 5bsh + 4bsN + 2bsN_r + bs*N_r/N*(96h + 256h_E) + 8bsh_E
```

After further simplification, the paper gives the general formula for a 4-layer PP stage:
```
4*M_1^E = 20bsh + 16bsN + 8bsN_r + 4bs*N_r/N*(96h + 256h_E) + 32bsh_E
```

**With full recomputation, single MoE layer**:
```
M_2^E = bsh + 2bsN_r  (store only layer input + dispatch masks for expert routing)
```
The dispatch mask (2bsN_r) is retained even with full recomputation because it is needed for the router during backward.

**For 4 layers with full recomputation**:
```
4*M_2^E = 4bsh + 8bsN_r
```

### 6.4 Total Activation Memory Per Device (Table 10)

| Component | No Recomputation (AC=None) | Full Recomputation (AC=Full) |
|-----------|---------------------------|------------------------------|
| MLA | 10bsh + 8bs(d_cq + d_c) + 16bs*d_h*n_h + 8bs*d_hr*n_h + 10bn_h*s^2 | 4bsh |
| MoE | 20bsh + 16bsN + 8bsN_r + 4bs*N_r/N*(96h + 256h_E) + 32bsh_E | 4bsh + 8bsN_r |
| **Total (4 layers)** | **4*(M_1^A + M_1^E)** | **8bsh + 8bsN_r** |

### 6.5 Numerical Example (DeepSeek-v3 with b=1, s=4096)

For AC=Full, 4-layer stage:
```
Total = 8*1*4096*7168 + 8*1*4096*8
      = 234,881,024 + 262,144
      = 235,143,168 bytes
      ≈ 224 MB
```

For AC=None with b=1, s=4096, the activation memory would be vastly larger due to the s^2 attention term and expert activations.

---

## 7. Temporal Buffers and Fragmentation (Section 6)

The paper identifies two additional memory overhead factors:

1. **Memory fragmentation**: Typically ranges from **5% to 30%** of total allocated memory. This introduces overhead that must be considered in training analysis.

2. **Temporary communication buffers**: Required for inter-GPU data transfer, generally occupy between **0.8 GB to 2 GB** per device, depending on parallel configuration and communication patterns.

The paper does not provide detailed formulas for these overheads, offering only the ranges above.

---

## 8. What Is Novel / Unique in This Paper

### 8.1 MLA-Specific Activation Formulas

No other calculator or paper provides activation memory formulas specifically for DeepSeek's Multi-head Latent Attention. The key differences from standard MHA:
- Additional compressed representation tensors (c_Q at d_cq dimensions, c_KV at d_c dimensions)
- Some MLA weight matrices are NOT split by TP (W^DQ, W^DKV, W^QR, W^KR are replicated), so their activation outputs are also not split by SP
- The RoPE component adds separate d_hr-dimensional activations per head

### 8.2 Separate ZeRO Sharding for MoE vs Non-MoE

The paper explicitly computes ZeRO memory using DIFFERENT denominators:
- Non-MoE params sharded by DP
- MoE params sharded by EDP (= DP * TP / EP / ETP)

This is mentioned briefly in our existing spec (Section 5.2, "MoE + ZeRO interaction") but this paper provides the complete worked-through calculation with actual numbers.

### 8.3 Expert Token Distribution in Activation Memory

The formula `E_token = bs * N_r / N` for average tokens per expert affects activation memory proportionally. This means MoE activation memory scales with the number of routed experts per token (topk) divided by total experts, not just with hidden dimension.

### 8.4 Router Activation Memory Is Kept During Recomputation

Even with full activation checkpointing, the MoE dispatch mask (`2bsN_r` bytes) is retained, not recomputed. This is because the expert routing decision must be preserved exactly for the backward pass.

### 8.5 Shared Expert Replication

Shared experts (N_s=1 for DeepSeek-v3) are replicated on ALL EP ranks, not distributed. Their parameters, gradients, and optimizer states appear on every rank. This must be accounted for separately from routed experts.

### 8.6 BF16 Optimizer States

The paper uses BF16 for momentum and variance (2+2=4 bytes) instead of the standard FP32 (4+4=8 bytes), giving 8 bytes/param for optimizer states instead of the standard 12. This is a memory optimization that should be offered as an option in the calculator.

### 8.7 Expert Tensor Parallelism (ETP)

The paper introduces the concept of ETP -- tensor parallelism applied WITHIN individual experts. With ETP=1 (as in their case study), expert matrices are not split. With ETP>1, each expert's gate_proj/up_proj/down_proj would be split across ETP ranks, further reducing per-device expert memory.

### 8.8 TP Behavior for MLA Weights

Not all MLA matrices follow the standard column/row parallel pattern:
- **Column parallel** (split along output dim): W^UQ, W^UK, W^UV (these decompress from low-rank to per-head)
- **Row parallel** (split along input dim): W^O
- **NOT parallelized** (replicated): W^DQ, W^DKV, W^QR, W^KR (these compress from hidden to low-rank)

This means TP does NOT achieve a clean 1/TP reduction for MLA parameters. The replicated portion (W^DQ, W^DKV, W^QR, W^KR) represents about 26% of MLA parameters for DeepSeek-v3.

---

## 9. Relevance to GPU Calculator Spec

### 9.1 What Should Be Adopted

1. **MoE ZeRO formula with separate sharding denominators**: The spec already mentions this concept but the paper provides the complete formula. Adopt the generalized formula from Section 5.4 above.

2. **MoE activation memory formulas**: The spec currently uses the Korthikanti formula (`sbh * (34 + 5as/d)`), which was derived for dense transformers. For MoE models, the FFN activation term must be replaced with the expert routing + expert compute terms from Section 6.3.

3. **Router dispatch mask preserved under recomputation**: When full activation checkpointing is enabled for MoE, add `2bsN_r` bytes per MoE layer for the dispatch mask. The spec does not currently account for this.

4. **Shared expert replication**: The spec's formula `Experts per GPU = E / N_ep` should note that shared experts are replicated, not distributed.

5. **Expert Tensor Parallelism (ETP)**: Add ETP as an additional parallelism dimension for MoE models. The formula becomes `Expert params per device = (N/EP/ETP + N_s) * params_per_expert`.

6. **BF16 optimizer states option**: Add as an optimizer variant in the spec's Table (Section 5.1): "AdamW mixed (BF16 m+v)" at 14 bytes/param (2 weights + 4 grad + 4 master + 2 m + 2 v).

7. **DeepSeek-v3 as validation target**: The paper provides exact numbers (11.64 GB params per device, 9.66 GB total with ZeRO-3, etc.) that can be used to validate the calculator's MoE support.

### 9.2 What Should NOT Be Adopted

1. **MLA-specific formulas**: These are highly specific to DeepSeek's architecture and not generalizable. The calculator should use standard MHA/GQA/MQA activation formulas. MLA support could be a future extension.

2. **The specific parallelism configuration (DP=32, TP=2, PP=16, EP=8)**: This is illustrative, not prescriptive.

3. **Memory fragmentation range of 5-30%**: The spec already has a better-calibrated 10% fragmentation buffer with the 1.04x alignment multiplier. The paper's 5-30% range is too wide to be actionable.

### 9.3 Spec Gaps This Paper Highlights

1. **No ETP (Expert Tensor Parallelism)**: The spec has EP but not ETP. Some MoE training configurations use TP within experts.

2. **No EDP (Expert Data Parallelism)**: The spec does not explicitly define EDP = DP * TP / (EP * ETP). This relationship is important for correct ZeRO sharding of MoE parameters.

3. **MoE activation memory is underspecified**: The spec's activation formula does not account for router activations (logits, probabilities, dispatch masks) or the per-expert token distribution.

4. **No MLA in parameter counting presets**: DeepSeek-v3 is listed as a preset but its MLA parameter structure is not modeled (the spec uses standard attention formulas).

---

## 10. Validation Data Points

The paper provides these concrete numbers for cross-validation:

| Metric | Value | Configuration |
|--------|-------|---------------|
| Total model params | 671B (671,000,000,000 = 187,107,328*61 layers of MLA + MoE + embedding + head) | DeepSeek-v3 |
| Params per device (BF16) | 6,250,364,928 (11.64 GB) | PP16@TP2@EP8@ETP1 |
| Model states, no ZeRO | 81.54 GB | Same config |
| Model states, ZeRO-1 | 40.46 GB | DP=32, EDP=8 |
| Model states, ZeRO-2 | 19.92 GB | DP=32, EDP=8 |
| Model states, ZeRO-3 | 9.66 GB | DP=32, EDP=8 |
| MLA params per layer | 187,107,328 | - |
| MoE params per layer (with router) | 11,320,164,352 | 256 routed + 1 shared |
| Expert params (single expert) | 44,040,192 (3*7168*2048) | SwiGLU with h_E=2048 |
| Activation (4 layers, AC=Full, b=1, s=4096) | ~224 MB | TP2@SP2@EP8 |

---

## 11. Key Formulas Summary (For Calculator Implementation)

### Formula 1: MoE Parameter Count Per Device
```
P_per_device = P_attn_tp_split / TP * L_stage
             + P_attn_replicated * L_stage
             + P_norm * L_stage
             + (N/EP/ETP + N_s) * P_single_expert * L_stage_moe
             + P_router * L_stage_moe
             + P_embedding  (if first/last stage)
```

### Formula 2: ZeRO with MoE (Separate Sharding)
```
P_nonmoe_per_device = (attention + norm params already divided by TP, for the PP stage)
P_moe_per_device = (expert + router params already divided by EP/ETP, for the PP stage)
EDP = DP * TP / (EP * ETP)

ZeRO-1: M = (P_nonmoe + P_moe)*2 + (P_nonmoe + P_moe)*grad_bytes + (P_nonmoe/DP + P_moe/EDP)*opt_bytes
ZeRO-2: M = (P_nonmoe + P_moe)*2 + (P_nonmoe/DP + P_moe/EDP)*grad_bytes + (P_nonmoe/DP + P_moe/EDP)*opt_bytes
ZeRO-3: M = (P_nonmoe/DP + P_moe/EDP)*2 + (P_nonmoe/DP + P_moe/EDP)*grad_bytes + (P_nonmoe/DP + P_moe/EDP)*opt_bytes
```

### Formula 3: MoE Activation Memory (General, No Recomputation)
For each MoE FFN layer, replace the standard FFN activation with:
```
M_act_moe_layer = M_router + M_dispatch + M_expert_compute + M_shared_expert + M_combine
Where:
  M_router = 4bsN      (router logits + probs, BF16)
  M_dispatch = 2bsN_r   (dispatch mask)
  M_expert_compute = (N/EP) * E_token * (3h + 8h_E) * 2   (per-rank expert activations in BF16)
                   = bs * N_r / EP * (6h + 16h_E)          (substituting E_token = bs*N_r/N, times N/EP experts)
  M_shared_expert = bs * (3h + 8h_E) * 2                   (shared expert, full token set)
                  = bs * (6h + 16h_E)
```

### Formula 4: MoE Activation Memory (Full Recomputation)
```
M_act_moe_recomp = 2bsh + 2bsN_r   (per layer: layer input + dispatch mask)
```
The dispatch mask is always retained (not recomputed) because expert routing must be preserved.

### Formula 5: Average Tokens Per Expert
```
E_token = b * s * N_r / N
```
Where N_r = topk (experts routed per token), N = total experts.
