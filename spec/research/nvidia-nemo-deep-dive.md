# NVIDIA NeMo Ecosystem -- Deep Dive for LLM Training GPU Calculator

**Repos examined:**
- `NVIDIA/NeMo` (aka `NVIDIA-NeMo/NeMo`) -- core framework (speech/TTS/LLM)
- `NVIDIA/NeMo-Framework-Launcher` -- AutoConfigurator for training HP search
- `NVIDIA/NeMo-Aligner` -- RLHF (PPO, DPO, REINFORCE, KTO, SPIN, RS)

**Date:** 2026-03-31

---

## 1. Key Finding: No MemoryProfileCallback Exists

The originally cited `MemoryProfileCallback` **does not exist** in any of the three NeMo repos. There is no dedicated memory profiling callback or analytical memory estimation utility anywhere in the NeMo codebase. The repos do not contain any formulas for estimating GPU memory consumption (model states, activations, optimizer states, etc.). This is a documentation/source misattribution.

What NeMo *does* have is a `log_tflops_per_sec_per_gpu` flag in the experiment manager config, but the actual FLOPs calculation for that logging is deferred to Megatron-Core (not present in the NeMo repo itself).

---

## 2. Model Size (Parameter Count) Formulas

### 2.1 GPT-3 / LLaMA / Decoder-only Models

**File:** `/tmp/nemo_launcher/auto_configurator/autoconfig/utils.py`, lines 48-59

```python
model_size = (
    12 * num_layers * hidden_size**2
    * (
        1
        + (13 / (12 * hidden_size))
        + ((vocab_size + seq_length) / (12 * num_layers * hidden_size))
    )
    / 1e9
)
```

**Expanded form:**
```
P = 12 * L * h^2 * [1 + 13/(12h) + (V + s)/(12Lh)]
  = 12*L*h^2 + 13*L*h + h*(V + s)
```

Where:
- L = num_layers
- h = hidden_size
- V = vocab_size
- s = seq_length

This breaks down to:
- `12*L*h^2` = self-attention (Q,K,V projections + output projection) + MLP (two linear layers with 4h intermediate)
- `13*L*h` = layer norm parameters and biases
- `h*V` = embedding/output projection
- `h*s` = positional embeddings

### 2.2 T5/mT5 (Encoder-Decoder)

**File:** `/tmp/nemo_launcher/auto_configurator/autoconfig/utils.py`, lines 61-74

```python
proj_size = att_heads * kv_channels
model_size = (
    2 * num_layers * 1.5 * ffn_size
    + 3 * num_layers * proj_size
    + hidden_size * (
        2
        + 4 * num_layers * 1.5 * ffn_size
        + num_layers * (21 + 12 * proj_size)
        + seq_length
        + vocab_size
    )
) / 1e9
```

### 2.3 BERT

**File:** `/tmp/nemo_launcher/auto_configurator/autoconfig/utils.py`, lines 76-84

```python
model_size = (
    num_layers * (
        ffn_size
        + hidden_size * (4 * hidden_size + 3 * att_heads + 2 * ffn_size + 6)
    )
    + hidden_size * (vocab_size + seq_length + hidden_size + 5)
) / 1e9
```

---

## 3. FLOPs (Throughput) Calculation Formulas

### 3.1 GPT-3 / Decoder-only TFLOPS

**File:** `/tmp/nemo_launcher/auto_configurator/autoconfig/scripts/compare_throughput.py`, lines 273-286

```python
model_flops = (
    (
        24 * gbs * enc_seq_len * hs * hs
        + 4 * gbs * enc_seq_len * enc_seq_len * hs
    )
    * (3 * layers)
    + (6 * gbs * enc_seq_len * hs * vocab)
) / time_per_step
```

**Formula notation:**
```
FLOPs_per_step = 3 * [(24*B*s*h^2 + 4*B*s^2*h) * L] + 6*B*s*h*V
```

Where:
- The factor 3 accounts for forward + backward (2x forward for backward)
- `24*B*s*h^2` = attention Q/K/V projections + output projection + MLP (2 layers)
- `4*B*s^2*h` = attention score computation (QK^T and attention*V)
- `6*B*s*h*V` = embedding layer (forward + backward = 3x the 2*B*s*h*V forward pass)

This is the standard Megatron-LM formula from "Efficient Large-Scale Language Model Training on GPU Clusters."

### 3.2 BERT TFLOPS

```python
model_flops = (
    72 * gbs * layers * enc_seq_len * hs * hs
    * (1 + (enc_seq_len / (6 * hs)) + (vocab / (12 * hs * layers)))
) / time_per_step
```

**Formula:** `FLOPs = 72*B*L*s*h^2 * [1 + s/(6h) + V/(12hL)]`

### 3.3 T5/mT5 TFLOPS (Encoder-Decoder, detailed breakdown)

```python
# Encoder self-attention + MLP
flops_self_attn_enc = 8*B*s_enc*h^2 + 4*B*s_enc^2*h
flops_mlp_enc = 6*B*s_enc*h*ffn_h  # GeGLU: two gemms for h -> ffn_h

# Decoder self-attention + cross-attention + MLP
flops_self_attn_dec = 8*B*s_dec*h^2 + 4*B*s_dec^2*h
flops_cross_attn_dec = 4*B*s_enc*h^2 + 4*B*s_dec*h^2 + 4*B*s_enc*s_dec*h
flops_mlp_dec = 6*B*s_dec*h*ffn_h

# Total per step (3x for fwd + bwd)
FLOPs = 3 * [(enc_layer + dec_layer) * (L/2) + 2*B*s_dec*h*V]
```

**Unique detail:** The T5 FLOPs calculation explicitly separates encoder self-attention, decoder self-attention, cross-attention, and MLP components. The MLP uses factor 6 (not 8) because GeGLU needs two gemms for h->ffn_h rather than the standard two gemms for h->4h->h.

---

## 4. Training Time Estimation

### 4.1 Estimate Model Size from Time Budget

**File:** `/tmp/nemo_launcher/auto_configurator/autoconfig/base_config.py`, lines 94-104

```python
model_size_in_b = (
    model_penalty
    * (max_training_days * 3600 * 24 * gpu_count * tflops_per_gpu * 1e12)
    / (8 * num_tokens_in_b * 1e9)
    / 1e9
)
```

**Formula:**
```
P = (T_days * 86400 * N_gpu * TFLOPS_per_gpu * 1e12) / (8 * T_tokens * 1e9) / 1e9
```

Where:
- The factor of 8 comes from: each token requires ~6*P FLOPs for forward pass, and ~2x for backward = ~8*P total (note: this approximation does NOT use the standard 6*P, it uses 8*P -- presumably including embedding layer overhead)
- `model_penalty` = 0.87 for mT5 (slower due to cross-attention), 1.0 otherwise

### 4.2 Estimate Training Time from Model Size

```python
training_days = (
    model_penalty
    * (model_size_in_b * 1e9 * 8 * num_tokens_in_b * 1e9)
    / (3600 * 24 * gpu_count * tflops_per_gpu * 1e12)
)
```

**Inverse formula:**
```
T_days = (P * 8 * T_tokens) / (86400 * N_gpu * TFLOPS_per_gpu)
```

**Note:** `model_penalty` for estimation is 1.15 for mT5 (asymmetric: 0.87 for size estimation, 1.15 for time estimation).

### 4.3 Max Training Steps

```python
max_steps = int((num_tokens_in_b * 1e9) / (seq_length * gbs))
```

---

## 5. AutoConfigurator: Parallelism Recommendation Heuristics

The NeMo AutoConfigurator uses a **brute-force grid search with empirical heuristic bounds** rather than analytical memory estimation. It does NOT calculate memory usage -- instead, it defines search spaces based on model size and GPU memory, launches actual training runs, and selects the fastest configuration.

### 5.1 Architecture of the Search

1. Given model_size_in_b and gpu_memory_gb (40 or 80), select a search grid of (TP, PP, CP, EP, MBS) values
2. Generate all valid combinations (respecting divisibility constraints)
3. Launch short training runs for each combination
4. Measure throughput and select the optimal configuration

### 5.2 Heuristic Parallelism Bounds (80GB GPU, seq_len=2048)

From `/tmp/nemo_launcher/auto_configurator/autoconfig/training_config.py`:

| Model Size    | TP         | PP        | MBS       | Min MP | Max MP |
|---------------|------------|-----------|-----------|--------|--------|
| <= 1B         | [1,2]      | [1]       | [1,2,3,4,6,8] | 1   | 8      |
| <= 4B         | [1,2,4]    | [1]       | [1,2,3,4,6,8] | 1   | 8      |
| <= 8B         | [1,2,4]    | [1]       | [1,2,3,4,6,8] | 1   | 8      |
| <= 13B        | [1,2,4,8]  | [1]       | [1,2,3,4,6,8] | 1   | 8      |
| <= 23B        | [1,2,4]    | [1..4]    | [1,2,4]   | 4      | 8      |
| <= 45B        | [2,4,8]    | [1..4]    | [1,2,4]   | 8      | 32     |
| <= 95B        | [2,4,8]    | [1..8]    | [1,2,4,8] | 8      | 64     |
| <= 130B       | [2,4,8]    | [1..16]   | [1,2,4,8] | 16     | 128    |
| <= 195B       | [8]        | [4..16]   | [1,2,4]   | 32     | 256    |
| <= 395B       | [8]        | [8..32]   | [1,2,4]   | 64     | 512    |
| <= 790B       | [8]        | [8..100]  | [1,2,4]   | 128    | 1024   |
| <= 1100B      | [8]        | [16..130] | [1,2,4]   | 256    | 2048   |

**Key takeaway for our calculator:** These tables encode NVIDIA's operational experience with what parallelism configs actually work on A100-80GB GPUs. They can serve as validation bounds or default recommendations.

### 5.3 Default GBS/TP/PP Recommendations (80GB GPU, seq_len=2048)

From `/tmp/nemo_launcher/auto_configurator/autoconfig/base_config.py`:

| Model Size | GBS  | TP | PP |
|------------|------|----|----|
| <= 1B      | 256  | 1  | 1  |
| <= 4B      | 1024 | 1  | 1  |
| <= 8B      | 2048 | 2  | 1  |
| <= 13B     | 2048 | 4  | 1  |
| <= 20.6B   | 2048 | 8  | 1  |
| <= 45.6B   | 2048 | 8  | 2  |
| <= 123.6B  | 2048 | 8  | 4  |
| <= 196.6B  | 2048 | 8  | 8  |
| <= 392.2B  | 2048 | 8  | 16 |
| <= 735B    | 2048 | 8  | 32 |
| <= 1100B   | 2048 | 8  | 64 |

### 5.4 Sequence Length Impact on Parallelism

The configurator adjusts recommendations based on sequence length. For longer sequences:
- TP increases earlier (e.g., TP=2 needed even for 1B model at seq_len=16384)
- PP needed at smaller model sizes
- MBS forced to 1 earlier
- GBS decreases proportionally

Tables exist for seq_len in {2048, 4096, 8192, 16384, 32768}.

### 5.5 Hidden Size / Attention Head Mapping

From `/tmp/nemo_launcher/auto_configurator/autoconfig/utils.py`, lines 116-150:

| Model Size | Hidden Size | Attention Heads | Learning Rate |
|------------|-------------|-----------------|---------------|
| < 0.25B   | 768         | 12              | 6e-4          |
| < 0.5B    | 1024        | 16              | 3e-4          |
| < 1B      | 1536        | 16              | 2.5e-4        |
| < 2B      | 2048        | 16              | 2e-4          |
| < 3B      | 2560        | 32              | 1.6e-4        |
| < 4.5B   | 3072        | 32              | 1.4e-4        |
| < 8B      | 4096        | 32              | 1.2e-4        |
| < 15B     | 5120        | 40              | 1e-4          |
| < 25B     | 6144        | 48              | 1e-4          |
| < 52B     | 8192        | 64              | 0.8e-4        |
| < 105B    | 10240       | 80              | 0.7e-4        |
| < 205B    | 12288       | 96              | 0.6e-4        |
| < 405B    | 20480       | 128             | 0.5e-4        |
| < 805B    | 20480       | 128             | 0.4e-4        |
| < 1105B   | 25600       | 160             | 0.3e-4        |

**Key insight:** Learning rate scales inversely with model size (roughly proportional to `1/sqrt(hidden_size)`).

### 5.6 Validity Constraints for Parallelism Configs

From `/tmp/nemo_launcher/auto_configurator/autoconfig/training_config.py`, lines 164-181:

```python
# A config is valid if ALL of these hold:
mod_gbs = gbs % (mbs * num_gpus / model_parallelism) == 0  # GBS divisible by per-GPU batch
mod_att_heads = att_heads % tp == 0                          # Attention heads divisible by TP
mod_layers = (multiplier * num_layers) % pp == 0             # Layers divisible by PP
min_model_parallel <= (tp * pp * cp * ep) <= max_model_parallel  # Total parallelism in range
# For CP/EP: one must be divisible by the other
mod_cp // mod_ep == mod_cp or mod_ep // mod_cp == mod_ep
```

---

## 6. Activation Checkpointing Configuration

### 6.1 Checkpoint Granularity

The AutoConfigurator explores activation checkpointing with three parameters:
- `activations_checkpoint_num_layers`: how many layers to checkpoint per pipeline stage
- `num_micro_batches_with_partial_activation_checkpoints`: for partial checkpointing across micro-batches in PP
- `activations_checkpoint_layers_per_pipeline`: layers to checkpoint per pipeline stage

### 6.2 Checkpoint Multiplier Heuristic

From `_set_activations_checkpoint_params`:

```python
# Base: 4 // pp layers at a time
act_multiple = 4 // pp

# For "block" method, increase with model size:
if 1B <= model_size < 11.3B: act_multiple = 8 // pp
if 11.3B <= model_size < 26B: act_multiple = 16 // pp
if 26B <= model_size < 60B:  act_multiple = 16 // pp
if 60B <= model_size:        act_multiple = 32 // pp
```

### 6.3 Interleaved Pipeline Scheduling

When PP > 2, virtual pipeline parallelism is enabled:
```python
virtual_pipelines = num_layers // pp
```

This changes activation checkpointing parameters to work with the interleaved schedule.

---

## 7. RLHF / Post-Training Memory Architecture (NeMo-Aligner)

### 7.1 PPO Training Memory Components

NeMo-Aligner PPO requires **4 models simultaneously** (though not all on same GPUs):

1. **Actor (Policy) Model** -- the model being trained
   - Full model weights + optimizer states + gradients
   - Generates rollouts (inference mode) then trains (training mode)
   - Supports `offload_adam_states: True` to offload optimizer to CPU during generation

2. **Critic Model** -- value function estimator
   - Full model weights + optimizer states + gradients
   - Runs on separate GPU group (separate nodes)
   - Initialized from reward model weights

3. **Reward Model** -- scores generated responses
   - Inference only (no gradients/optimizer states needed)
   - Can share GPU group with critic (`combine_rm_and_critic_server: True`)

4. **Reference Policy** -- for KL penalty computation
   - Inference only weights
   - Actor stores reference weights and swaps them for KL computation via `cpu_weight_swap`

**Memory optimization strategies in NeMo-Aligner PPO:**
- `offload_adam_states`: Offloads distributed Adam optimizer states to CPU during generation phase
- `cpu_weight_swap`: Swaps model weights between CPU and GPU for reference policy computation
- `clear_memory()`: Explicit `gc.collect()` + `torch.cuda.empty_cache()` between phases
- `combine_rm_and_critic_server`: Merges reward model and critic onto same GPUs

### 7.2 DPO Training Memory Components

DPO requires **2 model copies**:
1. **Training model** -- full weights + optimizer states + gradients
2. **Reference model** -- inference-only weights (stored in CPU, swapped in via `cpu_weight_swap`)

Key DPO memory detail: Each training step processes BOTH chosen AND rejected sequences together:
```python
# DPO processes paired data: chosen + rejected in same batch
# Effective memory = 2x a normal training step for sequence storage
chosen_tokens, rejected_tokens  # both padded to same length
chosen_labels, rejected_labels
```

The reference policy logprobs are computed BEFORE training begins for each batch (`augment_dataloader` method), then the reference model weights are swapped back to CPU.

### 7.3 REINFORCE Training

Similar to PPO but simpler:
- Actor model (train) + Reward model (inference)
- Reference policy for KL (CPU weight swap)
- `num_rollouts_per_prompt: 4` -- generates multiple completions per prompt
- No critic model needed (uses reward directly, no value function)

### 7.4 Memory-Saving Patterns in Post-Training

1. **Adapter Control for Reference Policy:**
```python
@contextmanager
def adapter_control(model):
    """Temporarily disable adapters and re-enable them after the operation"""
    # Disable LoRA adapters -> model behaves as reference policy
    # Re-enable after computing reference logprobs
```
When using LoRA for DPO/PPO, the reference policy IS the base model (with adapters disabled), so no separate reference model copy is needed. This is a **massive memory saving**.

2. **Distributed Adam Offloading:**
```python
def offload_distributed_adam(state_dict, force_clear_memory=False):
    for state_bucket in state_dict["state"]["buckets"]:
        dist_adam_load_state_bucket_into_device(state_bucket, device="cpu")
    torch.cuda.synchronize()
    # ... do generation ...
    for state_bucket in state_dict["state"]["buckets"]:
        dist_adam_load_state_bucket_into_device(state_bucket, device=torch.cuda.current_device())
```

---

## 8. LoRA / QLoRA Implementation Details

### 8.1 LoRA Architecture

The `ParallelLinearAdapter` class implements LoRA with Megatron tensor parallelism:
- `linear_in`: ColumnParallelLinear (in_features -> rank)
- `linear_out`: RowParallelLinear (rank -> out_features)
- Supports scaling via `alpha / rank`
- Optional layer norm, activation function, dropout

**Target modules supported:**
```python
'attention_qkv'  # Q, K, V projection (fused)
'attention_dense' # attention output projection
'mlp_fc1'        # MLP first layer (h -> 4h)
'mlp_fc2'        # MLP second layer (4h -> h)
'attention'      # qkv + dense
'mlp'            # fc1 + fc2
'all'            # everything
```

### 8.2 QLoRA (NF4 Quantization)

NeMo implements QLoRA with NF4 quantization using NVIDIA ModelOpt:
- Base weights quantized to NF4 (4-bit) with block_size=64
- Double quantization with scale_block_size=256
- LoRA adapters remain in bf16/fp16
- Weight quantization happens on GPU, base weights stored as quantized tensors
- Dequantization on-the-fly during forward pass

**Memory implication:** Base model weights go from 2 bytes (bf16) to ~0.5 bytes (4-bit), approximately 4x reduction in weight memory. LoRA adapter weights remain full precision.

### 8.3 LoRA Memory Formula (derivable from architecture)

For LoRA rank `r`, applied to a weight matrix of size (in_features x out_features):
```
LoRA_params = r * (in_features + out_features)
LoRA_memory = LoRA_params * bytes_per_param
```

For QLoRA:
```
Base_weight_memory = (in_features * out_features * 4) / 8  # 4-bit
LoRA_weight_memory = r * (in_features + out_features) * 2  # bf16
Optimizer_memory = r * (in_features + out_features) * 8    # Adam: 2 fp32 states
```

---

## 9. Miscellaneous Training Config Details

### 9.1 Warmup Steps and Learning Rate Schedule

```python
warmup_steps = int(0.0015 * max_steps)  # 0.15% of total steps
constant_steps = int(0.166 * max_steps)  # 16.6% constant after warmup
min_lr = lr * 0.1                        # 10x decay at end
init_method_std = 0.64 / sqrt(hidden_size)  # Weight initialization
```

### 9.2 Precision

Default: `bf16` everywhere (both NeMo and NeMo-Aligner).
`megatron_amp_O2: True` is recommended for pipeline parallelism to avoid explicit casting.

### 9.3 Optimizer Configuration

Default optimizer for post-training: `distributed_fused_adam` with:
- `bucket_cap_mb: 200` -- communication bucket size
- `contiguous_grad_buffer: True` -- reduces memory fragmentation
- `overlap_grad_sync: False` -- can be enabled for overlapping

---

## 10. What is Unique / Non-Obvious

1. **8x multiplier, not 6x**: NeMo uses `8 * P * T` for training time estimation (not the commonly cited `6 * P * T`). This accounts for embedding layer FLOPs overhead.

2. **Asymmetric model penalty for mT5**: 0.87x for size estimation, 1.15x for time estimation -- empirically tuned.

3. **No analytical memory estimation**: Despite being NVIDIA's own framework, NeMo does NOT include memory estimation formulas. The AutoConfigurator works by empirical grid search -- it launches actual training runs and picks the fastest one that doesn't OOM.

4. **Sequence length dramatically affects parallelism**: The heuristic tables show that doubling sequence length from 2048 to 4096 roughly doubles the minimum TP needed. At 32K context, even a 1B model needs TP=2.

5. **LoRA as memory optimization for reference policy**: In post-training (DPO/PPO), using LoRA means the reference policy is simply the base model with adapters disabled -- no separate model copy needed. This is a ~2x memory savings for the weight component.

6. **PPO optimizer offloading**: During generation (which is the majority of wall-clock time in RLHF), optimizer states can be offloaded to CPU, freeing ~2x the model size in GPU memory.

7. **Virtual pipeline parallelism threshold**: Enabled automatically when PP > 2, with `virtual_pipelines = num_layers / pp`.

8. **T5 FLOPs uses factor 6 for MLP (not 8)**: Because GeGLU activation requires two separate gemms for h->ffn_h, the factor is 6 instead of the standard 8 used for two-layer MLPs.

---

## 11. Relevant File Paths

- `/tmp/nemo_launcher/auto_configurator/autoconfig/utils.py` -- Model size formulas, param recommendations
- `/tmp/nemo_launcher/auto_configurator/autoconfig/base_config.py` -- Training time estimation, GBS/TP/PP defaults
- `/tmp/nemo_launcher/auto_configurator/autoconfig/training_config.py` -- Parallelism grid search bounds
- `/tmp/nemo_launcher/auto_configurator/autoconfig/scripts/compare_throughput.py` -- FLOPs/TFLOPS formulas
- `/tmp/nemo_aligner/nemo_aligner/algorithms/ppo.py` -- PPO training loop
- `/tmp/nemo_aligner/nemo_aligner/algorithms/dpo.py` -- DPO training loop
- `/tmp/nemo_aligner/nemo_aligner/utils/utils.py` -- Memory management utilities (offload, swap, clear)
- `/tmp/nemo_repo/nemo/collections/common/modules/adapters/parallel_adapters.py` -- LoRA implementation
- `/tmp/nemo_repo/nemo/collections/common/modules/adapters/qlora.py` -- QLoRA NF4 implementation
- `/tmp/nemo_aligner/examples/nlp/gpt/conf/gpt_ppo_actor.yaml` -- PPO actor config
- `/tmp/nemo_aligner/examples/nlp/gpt/conf/gpt_ppo_critic.yaml` -- PPO critic config
