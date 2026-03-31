# Deep Dive: Comprehensive Performance Modeling and System Design Insights for Foundation Models (Subramanian et al., 2024)

**Paper**: [arXiv:2410.00273](https://arxiv.org/abs/2410.00273)
**Code**: [github.com/ShashankSubramanian/transformer-perf-estimates](https://github.com/ShashankSubramanian/transformer-perf-estimates)
**Authors**: Shashank Subramanian, Ermal Rrapaj, Peter Harrington, Smeet Chheda, Steven Farrell, Brian Austin, Samuel Williams, Nicholas Wright, Wahid Bhimji (Lawrence Berkeley National Laboratory / NERSC)
**Published**: October 2024, SC'24

## Summary

This paper provides an analytical performance model for transformer training that systematically counts FLOPs, memory accesses (bytes), communication volume, and HBM memory consumption for every operation in a transformer layer. It then uses a roofline model to predict execution time, and performs brute-force search over all possible parallelism configurations to find the optimal strategy. The model was validated against Megatron-LM on 512 A100 GPUs (Perlmutter) with 2-15% error for GPT3-175B and 2-26% error for ViT-32K.

This is one of the most granular publicly available performance models for transformer training. Unlike most calculators that use aggregate "6 * P * T" FLOPs formulas, this paper models each sub-operation individually with separate forward and backward pass accounting.

---

## 1. Per-Operation FLOPs Formulas

### 1.1 General Matrix Multiplication Primitive

For C = A * B where A in R^(m x k), B in R^(k x n), C in R^(m x n):

```
FLOPs_fwd = (2k - 1) * m * n
```

Note: The code counts multiply and add separately. Multiply FLOPs = k*m*n, Add FLOPs = (k-1)*m*n. Total = (2k-1)*m*n.

### 1.2 QKV Projection (Linear Layer)

Fused QKV: Y = X * W_qkv where X in R^(b, l, e), W_qkv in R^(e, 3e)

```
FLOPs_fwd = b * l * 3e * (e + (e-1)) = b * l * 3e * (2e - 1)
```

Backward pass for a Linear layer Y = X * W:
- dL/dX = dL/dY * W^T: FLOPs = b * l * e * (f + (f-1)) = b*l*e*(2f-1)
- dL/dW = X^T * dL/dY: FLOPs = e * f * (b*l + (b*l-1)) = e*f*(2*b*l - 1)

```
FLOPs_bwd = b*l*e*(2f-1) + e*f*(2*b*l - 1)
```

### 1.3 Attention Logits (Q * K^T)

A = Q * K^T where Q, K in R^(b, h, l, q), A in R^(b, h, l, l), q = e/h

```
FLOPs_fwd = b * h * l * l * (q + (q-1)) = b * h * l^2 * (2q - 1)
```

Backward:
- dL/dQ = dL/dA * K: same structure
- dL/dK = dL/dA^T * Q: same structure

```
FLOPs_bwd = 2 * b * h * l * q * (l + (l-1)) = 2 * b * h * l * q * (2l - 1)
```

### 1.4 Attend (A * V)

S = A * V where A in R^(b, h, l, l), V in R^(b, h, l, q), S in R^(b, h, l, q)

```
FLOPs_fwd = b * h * l * q * (l + (l-1)) = b * h * l * q * (2l - 1)
```

Backward:
- dL/dA = dL/dS * V^T: FLOPs = b * h * l * l * (q + (q-1))
- dL/dV = A^T * dL/dS: FLOPs = b * h * l * q * (l + (l-1))

```
FLOPs_bwd = b*h*l^2*(2q-1) + b*h*l*q*(2l-1)
```

### 1.5 Output Projection

Y = S * W_p where S in R^(b, l, e), W_p in R^(e, e)

```
FLOPs_fwd = b * l * e * (2e - 1)
FLOPs_bwd = b*l*e*(2e-1) + e*e*(2*b*l - 1)
```

### 1.6 MLP (two linear layers)

FC1: Z = X * W1, where W1 in R^(e, f), f = 4e
FC2: O = GeLU(Z) * W2, where W2 in R^(f, e)

```
FLOPs_fwd_fc1 = b * l * f * (2e - 1)
FLOPs_fwd_fc2 = b * l * e * (2f - 1)
FLOPs_bwd_fc1 = b*l*e*(2f-1) + e*f*(2*b*l - 1)
FLOPs_bwd_fc2 = b*l*f*(2e-1) + f*e*(2*b*l - 1)
```

### 1.7 GeLU Activation

```
FLOPs_fwd = 8 * m     (where m = number of elements = b * l * f)
FLOPs_bwd = 13 * m
```

Source: Based on Google Research ELECTRA FLOPs computation (the 8x and 13x account for erf, tanh, and other ops in GeLU).

### 1.8 Softmax

For input in R^(b, h, l, l):

```
FLOPs_fwd = b * h * l * l * 2 + b * h * l * (l-1)
           = b*h*l*l*(exp + mult) + b*h*l*(l-1)*(add for normalization)
FLOPs_bwd = 2*b*h*l*l + b*h*l*(l-1) + b*h*l*l
```

### 1.9 Dropout

```
FLOPs_fwd = m * 1    (one multiply per element with random mask)
FLOPs_bwd = m * 1
```

### 1.10 LayerNorm

Forward pass (4 sub-operations: mean, variance, normalize, scale+shift):

```
FLOPs_fwd = 2*b*l*e     (mean: reduce + divide)
          + 3*b*l*e      (variance: subtract, square, add+divide)
          + 2*b*l*e      (normalize: subtract, divide)
          + 2*b*l*e      (scale and shift: multiply by gamma, add beta)
          = 9 * b * l * e   (approximately)
```

Backward:

```
FLOPs_bwd = 3*b*l*e + 9*b*l*e = 12*b*l*e  (approximately)
```

(The code notes: "little rough calcs, pretty sure some constant factor is off")

### 1.11 FlashAttention (Fused Logit-Attend)

The FusedLA class combines logits + softmax + dropout + attend into one fused operation.

Forward FLOPs (sum of all sub-operations):
```
FLOPs_fwd = b*h*l^2*(2q-1)                           # logits
          + b*h*l^2*2 + b*h*l*(l-1)                   # softmax (adjusted by tensor_core_factor)
          + b*h*l^2                                     # dropout (adjusted by tensor_core_factor)
          + b*h*l*q*(2l-1)                              # attend
```

Backward FLOPs (includes recomputation of forward attention):
```
FLOPs_bwd = [standard backward for logits, softmax, dropout, attend]
          + [recomputed forward for logits, softmax, dropout]  # FlashAttention recomputes attention in backward
```

**Critical detail**: The code applies a `tensor_core_factor = matrix_flops_fp16 / vector_flops_fp16` correction to softmax and dropout FLOPs because they do NOT use tensor cores. This factor converts them to equivalent tensor-core FLOPs for the roofline model. For A100: tensor_core_factor = 312/78 = 4x.

---

## 2. Per-Operation Memory Access (Bytes Read/Written from HBM)

All in FP16 (element_size = 2 bytes), mask uses 1 byte (mask_element_size).

### 2.1 Linear Layer (forward)

```
mem_fwd = activation_in + activation_out + weights
        = (b*l*e)*2 + (b*l*f)*2 + (e*f)*2   bytes
```

### 2.2 Linear Layer (backward)

```
mem_bwd = weights_grad + xgrad + num_bwd_ops*ygrad + weights + activation_buffer
        = (e*f)*2 + (b*l*e)*2 + 2*(b*l*f)*2 + (e*f)*2 + (b*l*e)*2   bytes
```

### 2.3 Logits (Q*K^T)

```
mem_fwd = 2*(b*h*l*q)*2 + (b*h*l*l)*2   bytes  (read Q,K; write A)
mem_bwd = 2*(b*h*l*q)*2 + 2*(b*h*l*l)*2 + activation_buffer   bytes
```

### 2.4 Attend (A*V)

```
mem_fwd = (b*h*l*l)*2 + (b*h*l*q)*2 + (b*h*l*q)*2   bytes  (read A,V; write S)
mem_bwd = 3*(b*h*l*q)*2 + (b*h*l*l)*2 + buffers   bytes
```

### 2.5 FlashAttention (FusedLA)

Forward:
```
mem_fwd = 3*(b*h*l*q)*2       # Q, K, V
        + (b*h*l)*2            # softmax stats
        + (b*h*l*q)*2          # output
```

Activation buffer (stored for backward):
```
activation_buffer = 3*(b*h*l*q)*2     # Q, K, V
                  + (b*h*l)*2          # RNG states for dropout
                  + (b*h*l)*2          # softmax stats
                  + (b*h*l*q)*2        # output (for flash backward)
```

Key: FlashAttention does NOT store the b*h*l*l attention matrix. This is the quadratic memory saving.

### 2.6 Softmax

```
mem_fwd = 2*(b*h*l*l)*2   bytes  (read + write)
activation_buffer = (b*h*l*l)*2   bytes
mem_bwd = 2*(b*h*l*l)*2 + activation_buffer   bytes
```

### 2.7 GeLU

```
mem_fwd = 2*m*2   bytes  (read input, write output)
activation_buffer = m*2   bytes  (store input for backward)
mem_bwd = 2*m*2 + activation_buffer   bytes
```

### 2.8 Dropout

```
mem_fwd = m*2 + m*1 + m*2   bytes  (input + mask + output; mask uses 1 byte)
activation_buffer = m * 1   bytes  (store mask)
mem_bwd = 2*m*2 + activation_buffer   bytes
```

### 2.9 LayerNorm

```
mem_fwd = (b*l*e)*2 + 2*(b*l)*2 + (b*l*e)*2 + 2*e*2   bytes
         (input + mean,std + output + gamma,beta)
activation_buffer = (b*l*e)*2   bytes
weights_mem = 2*e*2   bytes  (gamma and beta)
```

---

## 3. Communication Cost Formulas

### 3.1 General Communication Time Model

```python
def get_time_comm(vol, n_gpus, comm_type, topology):
    # topology = number of GPUs in fast (NVLink) domain for this communicator
    # n_gpus = total GPUs in communicator
    nodes = n_gpus // topology  # number of NVLink domains
    
    # Ring correction factor
    if comm_type in ['allreduce', 'allgather', 'reducescatter']:
        correction = (n_gpus - 1) / n_gpus
    elif comm_type in ['reduce', 'broadcast']:
        correction = 1
    
    if topology == 1:  # all InfiniBand
        t_comm = ls * (n_gpus - 1) + correction * vol / bs
    elif nodes == 1:   # all NVLink
        t_comm = lf * (n_gpus - 1) + correction * vol / bf
    else:              # multi-node with both networks
        num_rings = nic_factor * topology  # NICs per node
        t1 = correction * vol / (num_rings * bs)   # IB bottleneck
        t2 = correction * vol / bf                   # NVLink bottleneck
        t_comm = max(t1, t2)
        if comm_type == 'allreduce':
            t_comm *= 2
        t_comm += ls*(nodes-1) + lf*(n_gpus - nodes)  # latencies
```

Where:
- ls, lf = InfiniBand and NVLink latencies
- bs = ib_bandwidth * ib_eff (effective IB bandwidth)
- bf = nvlink_bandwidth * nvlink_eff (effective NVLink bandwidth)
- nic_factor = number of NICs per GPU

Key insight: The effective slow (IB) bandwidth scales with `nic_factor * topology` because each GPU in the NVLink domain has its own NIC. Larger NVLink domains lead to higher effective inter-node bandwidth.

### 3.2 Point-to-Point (Pipeline Parallel)

```
if all IB:   t_comm = ls * (n-1) + vol / bs
if NVLink:   t_comm = lf * (n-1) + vol / bf
```

### 3.3 Communication Volumes by Parallelism Type

#### 1D Tensor Parallelism

Per transformer layer, communication consists of:
- 2 AllGather operations (before LayerNorm in SA and MLP): volume = b*l*e each
- 2 ReduceScatter operations (after output projection and MLP fc2): volume = b*l*e each

**Total per layer** = 4 * b*l*e bytes

**Critical insight**: Communication volume is INDEPENDENT of tensor parallel degree n_t.

#### 2D Tensor Parallelism (SUMMA, n_t = n_1 * n_2)

Communication per layer uses Broadcast and Reduce operations:
- Broadcasts of activation slices: volume scales as b*(l/n_2)*e or b*(l/n_2)*(e/n_1)
- Broadcasts of weight slices: volume scales as e*(f/n_1) or e*e/(n_1)
- Communication volume scales with one GPU dimension (reduced by partitioning)

#### 2D Sequence/Context Parallelism

For FlashAttention with sequence parallelism:
- Forward: AllGather KV tensors: volume = 2 * b * (h/m1) * l * q * element_size
- Backward: AllGather + ReduceScatter of KV: same volume

### 3.4 Data Parallelism Communication

Implements distributed optimizer (ReduceScatter gradients + AllGather weights):
- ReduceScatter of weight gradients: overlapped with backward pass of last microbatch
- AllGather of weights: overlapped with forward pass of first microbatch
- Volume = total_weight_memory_of_all_local_layers (per one layer: wts_mem)

Overlap model:
```python
# t_comm = time for one collective (RS or AG)
# t_compute_overlap = time for one fwd or bwd pass of one layer
if overlap:
    t_total = t_comm + max(t_comm - t_compute_overlap, 0) * (num_layers - 1)
else:
    t_total = t_comm * num_layers
```

### 3.5 Pipeline Parallel Communication

- Point-to-point sends/receives of activation maps
- Volume = activation_buffer of LayerNorm input (b * l_local * e * element_size)
- Overlapped with computation (set to 0 when overlap=True in code)

Pipeline bubble time:
```
t_bubble = (n_p - 1) * (t_fwd_one_microbatch + t_bwd_one_microbatch)
```

Uses 1F1B (one-forward-one-backward) non-interleaved schedule.

---

## 4. Memory Estimation Formulas

### 4.1 Weight Memory

Per transformer layer (1D TP with n_t GPUs):
```
weights_per_layer = (e * 3e/n_t) + (e/n_t * e) + (e * 4e/n_t) + (4e/n_t * e) + biases + layernorm_params
                  = ~12*e^2/n_t + small_terms   (in elements)
```

Weight memory in bytes = weights_per_layer * element_size (2 bytes for FP16)

### 4.2 Weight Gradient Memory

Same size as weights (stored in same precision):
```
wts_grad = wts   (same bytes)
```

### 4.3 Optimizer State Memory

Adam optimizer with distributed optimizer (sharded across data parallel group):
```
wts_optimizer_states = 6 * (wts / (dp * seqp))
```

This 6x factor represents: 2 bytes for FP32 copy of weights (master weights) + 4 bytes for momentum + 4 bytes for variance = 10 bytes; but since wts is already in 2-byte FP16, the ratio is 10/2 = 5... The code actually uses 6x which corresponds to 12 bytes total per parameter / 2 bytes per FP16 weight = 6x. This gives 12 bytes per parameter for optimizer states, divided by dp*seqp.

**Reconciliation**: 12 bytes = 4 (FP32 master weight) + 4 (momentum) + 4 (variance). The factor 6 arises because wts is measured in FP16 bytes (2 per param), so 12/2 = 6.

### 4.4 Activation Memory

Per layer, per microbatch, each operation stores an `activation_buffer` for the backward pass:

**Self-Attention (with FlashAttention)**:
```
activation_buffer_sa = (b*l*e/n_t)*2           # QKV input buffer (stored by QKV Linear)
                     + 3*(b*h_local*l*q)*2      # Q, K, V for flash backward
                     + (b*h_local*l)*2           # dropout RNG states
                     + (b*h_local*l)*2           # softmax stats
                     + (b*h_local*l*q)*2         # flash output
                     + (b*l*e/n_t)*2             # vproj input buffer
                     + (b*l/n_t*e)*1             # dropout mask (1 byte)
                     + (b*l/n_t*e)*2             # layernorm buffer
```

**Self-Attention (without FlashAttention, with attention recompute)**:
- Q, K buffers stored: 2*(b*h_local*l*q)*2
- Attention matrix: (b*h_local*l*l)*2  (NOT stored if remat=True)
- V buffer: (b*h_local*l*q)*2

**MLP**:
```
activation_buffer_mlp = (b*l*e)*2              # fc1 input buffer
                      + (b*l*f/n_t)*2           # GeLU input buffer
                      + (b*l*f/n_t)*1           # dropout mask
                      + (b*l*e/n_t)*2           # fc2 input buffer (this is ReduceScattered)
                      + (b*l/n_t*e)*1           # dropout mask
                      + (b*l/n_t*e)*2           # layernorm buffer
```

### 4.5 Total HBM Memory

```
total_mem = wts * (depth/pp)                           # all local layers
          + wts_grad * (depth/pp)                       # same
          + 6 * wts * (depth/pp) / (dp * seqp)         # optimizer states
          + (acts_per_layer * (depth/pp)) * mem_factor  # activations
```

Where:
```
mem_factor = min(pp, number_micro_batches)   # 1F1B schedule stores at most pp microbatches
```

Feasibility check: `total_mem <= hbm_capacity`

---

## 5. Roofline Model for Execution Time

### 5.1 Per-Operation Time

```python
def get_time(flops, mem, comm, ...):
    t_comp = 20e-6 + flops / (hardware_flops * 0.8)  # 20us kernel launch overhead + compute
    t_mem = mem / hbm_bandwidth
    t_compute = max(t_comp, t_mem)  # roofline: whichever is slower
    t_comm = get_time_comm(comm, ...)
    t_total = t_compute + t_comm
```

Key details:
- Uses tensor core FLOPs for matrix operations, vector FLOPs for element-wise ops
- Applies 0.8 efficiency factor to hardware peak FLOPs
- 20 microsecond kernel launch overhead per operation
- Communication time is NOT overlapped with compute (added serially) except for specific cases (SUMMA, DP)

### 5.2 Total Training Iteration Time

```python
t_fwd = (sum of all layer fwd times)    # 1 microbatch, 1 layer
t_bwd = (sum of all layer bwd times)    # 1 microbatch, 1 layer
t_per_layer = (t_fwd + t_bwd) * number_micro_batches
t_all_layers = t_per_layer * (depth / pp)
t_pp_comm = pipeline_parallel_comm_time
t_bubble = (pp - 1) * (t_all_layers / number_micro_batches)  # bubble
t_dp_comm = data_parallel_comm_time
t_total = t_all_layers + t_pp_comm + t_bubble + t_dp_comm
```

### 5.3 SUMMA Communication Overlap Model

For SUMMA-based 2D parallelism, communication is pipelined with compute:

```python
def compute_time_summa(flops, mem, comms, ...):
    t_compute = max(flops/hardware_flops, mem/hbm_bw)
    t_for_one_comm_set = sum(get_time_comm(c/n_b, ...) for c in comms)  # n_b = SUMMA block size
    t_comm = t_for_one_comm_set + max(t_for_one_comm_set * n_b - t_compute, 0)  # overlap
    return t_compute + t_comm
```

---

## 6. Hardware Configuration Parameters

All values in consistent units: FLOPs in TFLOPS, bandwidth in GB/s, capacity in GB, latency in seconds, element_size in GB.

| Parameter | A100 | H200 | B200 | B200-next |
|-----------|------|------|------|-----------|
| matrix_flops_fp16 (TFLOPS) | 312 | 989.4 | 2500 | 3750 |
| vector_flops_fp16 (TFLOPS) | 78 | 133.8 | 339 | 508.5 |
| vector_flops_fp32 (TFLOPS) | 19.5 | 66.9 | 169 | 253.5 |
| hbm_bandwidth (GB/s) | 1555 | 4800 | 8000 | 16000 |
| hbm_capacity (GB) | 80 | 141 | 192 | 384 |
| nvlink_bandwidth (GB/s) | 300 | 450 | 900 | 3600 |
| nvlink_eff | 0.7 | 0.7 | 0.7 | 0.8 |
| nvlink_latency (s) | 2.5e-6 | 2.5e-6 | 2.5e-6 | 5e-6 |
| ib_bandwidth (GB/s) | 25 | 50 | 100 | 200 |
| ib_eff | 0.7 | 0.7 | 0.7 | 0.8 |
| ib_latency (s) | 5e-6 | 5e-6 | 5e-6 | 1e-5 |
| nvlink_size (default) | 4 | 64 | 4 | 64 |
| nic_factor | 1 | 1 | 1 | 1 |
| element_size (GB) | 2e-9 | 2e-9 | 2e-9 | 2e-9 |
| mask_element_size (GB) | 1e-9 | 1e-9 | 1e-9 | 1e-9 |

Note: The code stores element_size as 2e-9 GB (= 2 bytes for FP16). All memory calculations multiply element counts by this value, so results are in GB.

---

## 7. Model Configurations Used for Validation

```python
gpt3_175B = {'l': 2048, 'e': 12288, 'h': 96, 'depth': 96}      # f = 4*e = 49152
gpt3_1T   = {'l': 2048, 'e': 25600, 'h': 160, 'depth': 128}    # f = 4*e = 102400
vit_era5  = {'l': 64800, 'e': 12288, 'h': 64, 'depth': 48}     # f = 4*e = 49152
```

Validation results on Perlmutter (512 A100 GPUs):
- GPT3-175B optimal: (n_t=4, n_p=16, n_d=8, b_m=1), 11% error
- GPT3-175B sub-optimal configs: 4-15% error range
- ViT-32K near-optimal: (n_1=2, n_2=4, n_p=4, n_d=16, b_m=1), 2% error
- ViT-32K sub-optimal configs: 11-26% error range

---

## 8. Parallelism Strategies Modeled

### 8.1 Configuration Space

Total GPUs decomposed as: n = n_1 * n_2 * n_p * n_d

Each parallelism dimension can be assigned to fast (NVLink) or slow (IB) network:
- NVLink assignment: (nv_1, nv_2, nv_dp, nv_pp) where nv_1 * nv_2 * nv_dp * nv_pp <= nvlink_size

### 8.2 Three TP Strategies

1. **1D Tensor Parallelism**: Partitions attention heads and MLP hidden dim across n_t GPUs. Communication = AllGather + ReduceScatter per sub-layer.

2. **2D SUMMA Tensor Parallelism**: Uses n_t = n_1 * n_2 grid. Partitions both heads/hidden dim (n_1) and sequence (n_2). Communication via Broadcast + Reduce in SUMMA iterations.

3. **2D Sequence/Context Parallelism**: Partitions heads (n_1) and sequence (n_2). Sequence parallel with AllGather/ReduceScatter for KV in attention. Requires additional weight gradient AllReduce across sequence parallel group.

### 8.3 Feasibility Constraints

```
n_2 must divide l (sequence length) evenly
n_d must divide global_batch_size
n_p must divide depth
h must be divisible by n_1 (for head partitioning)
e must be divisible by n_1 (for embedding partitioning)
3e/n_t >= 128 (minimum local tensor dimension for efficiency)
l/n_2 >= 128 (minimum local sequence length)
```

---

## 9. What Is Unique / Non-Obvious

### 9.1 Separate Forward and Backward FLOPs

Most calculators use "backward = 2x forward" approximation. This code computes exact backward FLOPs for every operation. The backward pass has TWO matrix multiplications per Linear layer (xgrad and wgrad), each with different dimensions, plus bias gradient reduction.

### 9.2 Memory Access Modeling (Not Just FLOPs)

The code models bytes read/written from HBM for every operation and uses a roofline model. Operations like softmax, dropout, and LayerNorm are memory-bound, not compute-bound. The roofline approach takes `max(t_compute, t_memory)` per operation.

### 9.3 Tensor Core vs. Vector FLOPs Distinction

Matrix multiplications use tensor cores; element-wise operations (softmax, GeLU, dropout, LayerNorm) use vector units. The code tracks `use_tensor_cores` per operation and selects the appropriate hardware FLOPs rate. For A100: tensor cores = 312 TFLOPS, vector = 78 TFLOPS (4x difference).

In FusedLA, the softmax and dropout FLOPs within the fused kernel are scaled by `tensor_core_factor = matrix_flops / vector_flops` to account for this when computing time.

### 9.4 Dual-Network Communication Model

Models two-tier network topology (NVLink intra-node + IB inter-node) with different latencies and bandwidths. The effective IB bandwidth scales with `nic_factor * topology` (number of NICs per NVLink domain). This is critical for large-scale training where some communication groups span nodes.

### 9.5 SUMMA Communication-Compute Overlap

For 2D SUMMA parallelism, the code models pipelined communication where broadcasts overlap with compute. The formula accounts for partial overlap:
```
t_comm = t_one_comm_set + max(t_one_comm_set * n_b - t_compute, 0)
```

### 9.6 FlashAttention Activation Buffer Accounting

The FlashAttention (FusedLA) class carefully accounts for what IS stored:
- Q, K, V tensors (3 * b*h*l*q * 2 bytes)
- RNG states for dropout (b*h*l * 2 bytes, not full mask)
- Softmax statistics (b*h*l * 2 bytes, not full l*l matrix)
- Output tensor (b*h*l*q * 2 bytes)

And what is NOT stored (recomputed in backward):
- Attention matrix A in R^(b*h*l*l) - the quadratic memory saving

### 9.7 Pipeline Schedule Memory Factor

The 1F1B pipeline schedule limits activation storage:
```
mem_factor = min(pp, number_micro_batches)
```
Instead of storing activations for all microbatches, only pp microbatches worth are stored.

### 9.8 Kernel Launch Overhead

The code includes a 20 microsecond kernel launch overhead per operation:
```
t_comp = 20e-6 + flops / (hardware_flops * 0.8)
```

This is non-trivial for small operations (biases, dropout, activation functions) and represents a realistic lower bound on kernel execution time.

### 9.9 80% Hardware Efficiency Assumption

The code applies a flat 0.8 efficiency factor to peak hardware FLOPS:
```
hardware_flops *= 0.8  # some avg efficiency
```

This accounts for instruction overhead, warp scheduling inefficiencies, and other real-world factors that prevent achieving peak.

### 9.10 Distributed Optimizer Memory Savings

Optimizer states (12 bytes/param) are sharded across `dp * seqp` GPUs:
```
optimizer_mem = 6 * weights_mem / (dp * seqp)
```

For sequence parallelism, the seqp dimension also participates in optimizer sharding because weights are replicated across the sequence parallel group.

---

## 10. What This Paper Does NOT Cover

- No ZeRO Stage 1/2/3 modeling (only distributed optimizer which is similar to ZeRO-1)
- No LoRA or PEFT modeling
- No MoE (Mixture of Experts)
- No gradient accumulation as separate concept (uses microbatching)
- No embedding layer or final output/loss layer FLOPs (only models transformer block layers)
- No BF16/FP8/INT8 precision options (fixed at FP16)
- No activation checkpointing options beyond FlashAttention recomputation
- No CPU offloading
- No vocabulary/embedding parallelism

---

## 11. Relevance to GPU Calculator Spec

### Directly Adoptable

1. **Per-operation FLOPs formulas** with exact (2k-1)*m*n counting instead of 2*m*n*k approximation
2. **Roofline-based time estimation** distinguishing compute-bound vs memory-bound operations
3. **Dual-network communication model** for realistic multi-node training time
4. **Activation buffer formulas** for each operation, especially FlashAttention
5. **Optimizer state formula**: 12 bytes/param divided by dp*seqp
6. **Pipeline bubble formula**: (pp-1) * (t_fwd + t_bwd) per microbatch
7. **1F1B activation memory**: stored for min(pp, num_microbatches) microbatches
8. **Hardware parameter table** for A100, H200, B200

### Needs Adaptation

1. The code assumes FP16 everywhere - would need extension for BF16/FP8/FP32
2. No ZeRO stage modeling - need to add from other sources
3. Communication model is NVIDIA-specific (NVLink/IB) - may want to generalize
4. The brute-force search is useful for validation but too slow for a web calculator - need closed-form or heuristic
5. GeLU FLOPs (8x forward, 13x backward) is a useful detail but may need verification for SwiGLU

### Key Formulas to Extract

The most calculator-relevant formulas from this paper are:

**Total FLOPs per transformer layer (forward)**:
```
= 2*b*l*(2e-1)*3e    # QKV projection
+ 2*b*h*l^2*(2q-1)   # logits + attend  (or FlashAttention)
+ b*l*e*(2e-1)        # output projection
+ b*l*f*(2e-1)        # MLP fc1
+ b*l*e*(2f-1)        # MLP fc2
+ smaller terms       # LayerNorm, GeLU, dropout, softmax
```

**Total activation memory per layer per microbatch (with FlashAttention)**:
```
= 4*(b*h_local*l*q)*2 + 2*(b*h_local*l)*2    # FlashAttention buffers
+ (b*l*e_local)*2 * 2                          # QKV and vproj input buffers
+ masks and dropout buffers
+ (b*l*f_local)*2                               # GeLU buffer
+ MLP input/output buffers
+ LayerNorm buffers
```

