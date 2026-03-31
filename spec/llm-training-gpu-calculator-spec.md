# LLM Training GPU Calculator — Full Implementation Specification

You are building an interactive **GPU Requirement Calculator for LLM Training** — a tool that estimates the number of GPUs, memory breakdown, training time, recommended parallelism strategy, and cloud cost for training large language models. It covers two phases: **Pretraining** and **Post-Training** (no inference). This will be added to an existing Next.js portfolio site as an interactive tool.

Your job: produce a **detailed, step-by-step implementation plan** for this tool based on the specification below.

## Reference Documents

- **Competitive analysis**: The attached document lists features from existing open-source GPU calculators. Use it to identify gaps, baseline features, and opportunities to exceed the state of the art.
- **This specification**: Contains all domain formulas, GPU specs, and codebase patterns. Treat it as the source of truth — do not hallucinate or invent formulas beyond what's here.

---

## 1. Portfolio Tech Stack & Architecture

### Stack
- **Next.js 15** + **React 19** + **TypeScript** (strict mode)
- **Tailwind CSS 4** with OKLch color system (CSS variables in `app/globals.css`)
- **Framer Motion** for animations
- **next-themes** for dark/light mode (`.dark` class on `<html>`)
- **Lucide React** for icons
- No external charting libraries — canvas/SVG for custom visualizations

### Existing Tool Pattern

**Tool metadata** (`lib/utils/tools.ts`):
```typescript
export interface Tool {
  slug: string
  title: string
  summary: string
  category: string
  tags: string[]
  relatedPost?: string
}

const tools: Tool[] = [
  {
    slug: "mlp-playground",
    title: "Neural Network Playground",
    summary: "Build, train, and visualize multi-layer perceptrons in real time...",
    category: "Interactive Visualization",
    tags: ["neural-networks", "deep-learning", "interactive"],
    relatedPost: "mlp",
  },
]
```

**Tool page** (`app/tools/mlp-playground/page.tsx`):
- Breadcrumb back to `/tools`
- Category badge, title, description
- Dynamic embed component (SSR disabled)
- Related blog post link at bottom

**Dynamic embed wrapper** (`app/tools/mlp-playground/mlp-playground-embed.tsx`):
```typescript
"use client"
import dynamic from "next/dynamic"
const Component = dynamic(() => import("../../../components/mlp/tensorflow-playground"), {
  ssr: false,
  loading: () => <LoadingSkeleton />,
})
export default function Embed() { return <Component /> }
```

### Component Patterns (MUST follow)

**Dark/light mode** — every interactive component uses this exact pattern:
```typescript
const [mounted, setMounted] = useState(false)
useEffect(() => setMounted(true), [])
const { resolvedTheme } = useTheme()
const isDark = mounted && resolvedTheme === "dark"

const colors = useMemo(() => ({
  bg: isDark ? "#1a1a2e" : "#f8f9fa",
  text: isDark ? "#e0e0e0" : "#1a1a2e",
  // ... full palette
}), [isDark])
```

**File conventions**: kebab-case files, PascalCase components, `"use client"` for interactive components, path aliases (`@/lib/`, `@/components/`).

**State management**: `useState` for UI state, `useRef` for expensive objects, `useCallback` for heavy computations, `useMemo` for derived values. No external state libraries.

---

## 2. Notation

These symbols are used consistently throughout all formulas:

| Symbol | Meaning |
|--------|---------|
| Ψ (or N) | Total model parameters |
| Ψ_active | Active parameters per token (= Ψ for dense models; < Ψ for MoE) |
| D | Total training tokens |
| d | Hidden dimension (d_model) |
| L | Number of transformer layers |
| a (or n_heads) | Number of attention heads |
| a_kv (or n_kv) | Number of KV heads (GQA/MQA) |
| d_kv | KV head dimension = d / a (same as per-head dim; total KV width = a_kv × d_kv) |
| d_ff | FFN intermediate dimension |
| V | Vocabulary size |
| s | Sequence length |
| b | Micro-batch size (per GPU) |
| B | Global batch size |
| G | Gradient accumulation steps |
| E | Number of experts (MoE models) |
| topk | Experts activated per token (MoE models) |
| N_dp | Data parallel degree |
| N_tp | Tensor parallel degree |
| N_pp | Pipeline parallel degree |
| N_ep | Expert parallel degree (MoE models) |
| N_gpu | Total GPUs = N_dp × N_tp × N_pp |
| β | Bytes per parameter in compute precision (2 for bf16, 4 for fp32) |
| β_grad | Bytes per gradient element (2 for bf16, 4 for fp32) |
| Φ | Total bytes per parameter for model states = 2 + β_grad + 12 (AdamW mixed) |
| MFU | Model FLOPS Utilization (uses ideal 6ΨD FLOPs) |
| HFU | Hardware FLOPS Utilization (uses actual executed FLOPs, including recomputation) |

---

## 3. Model Parameter Count

### 3.1 From Architecture (Detailed Mode)

For a decoder-only transformer (GPT/LLaMA style):

**Per-layer parameters:**

Self-attention (with GQA support):
```
Ψ_attn = d² + 2 × d × (d × a_kv / a) + d²
       = 2d² × (1 + a_kv / a)
```
- MHA (a_kv = a): 4d²
- GQA (a_kv = a/4): 2.5d²  
- MQA (a_kv = 1): ≈ 2d²

FFN:
```
Standard (expansion ratio r, typically 4):
  Ψ_ffn = 2 × r × d² = 8d²  (for r=4)

SwiGLU (3 projections, typically r = 8/3):
  Ψ_ffn = 3 × r × d² = 8d²  (for r=8/3)

Or directly: Ψ_ffn = 2 × d × d_ff  (standard)
            Ψ_ffn = 3 × d × d_ff  (SwiGLU/GeGLU)
```

Layer normalization:
```
LayerNorm: 2 × 2d = 4d per layer (scale + bias, 2 norms)
RMSNorm:  2 × d = 2d per layer (scale only, 2 norms)
```

**Non-layer parameters:**
```
Token embedding:          V × d
Positional embedding:     s × d  (learned, e.g., GPT-2/3; 0 for RoPE models like LLaMA/Mistral)
Output projection:        V × d  (0 if tied with embedding)
Final layer norm:         d or 2d
```

**Total:**
```
Ψ = L × (Ψ_attn + Ψ_ffn + Ψ_norm) + V × d × (1 + untied) + Ψ_pos + d
```
Where Ψ_pos = s × d for learned positional embeddings, 0 for RoPE/ALiBi.

### 3.2 Quick Estimate

For a standard MHA transformer with 4× FFN expansion:
```
Ψ ≈ 12 × L × d²
```
Accurate to within ~5% for most models (embedding/norm terms are small).

### 3.3 Model Architecture Presets

The calculator should include these presets (user can also enter custom values):

| Model | d | L | a | a_kv | d_ff | V | Params | FFN |
|-------|---|---|---|------|------|---|--------|-----|
| GPT-2 Small | 768 | 12 | 12 | 12 | 3,072 | 50,257 | 124M | Standard |
| GPT-2 XL | 1,600 | 48 | 25 | 25 | 6,400 | 50,257 | 1.56B | Standard |
| LLaMA 7B | 4,096 | 32 | 32 | 32 | 11,008 | 32,000 | 6.7B | SwiGLU |
| LLaMA 2 13B | 5,120 | 40 | 40 | 40 | 13,824 | 32,000 | 13B | SwiGLU |
| LLaMA 2 70B | 8,192 | 80 | 64 | 8 | 28,672 | 32,000 | 70B | SwiGLU+GQA |
| Mistral 7B | 4,096 | 32 | 32 | 8 | 14,336 | 32,000 | 7.2B | SwiGLU+GQA |
| LLaMA 3 8B | 4,096 | 32 | 32 | 8 | 14,336 | 128,256 | 8B | SwiGLU+GQA |
| LLaMA 3 70B | 8,192 | 80 | 64 | 8 | 28,672 | 128,256 | 70.6B | SwiGLU+GQA |
| LLaMA 3.1 405B | 16,384 | 126 | 128 | 8 | 53,248 | 128,256 | 405B | SwiGLU+GQA |
| GPT-3 175B | 12,288 | 96 | 96 | 96 | 49,152 | 50,257 | 175B | Standard |
| Qwen 2.5 72B | 8,192 | 80 | 64 | 8 | 29,568 | 152,064 | 72.7B | SwiGLU+GQA |
| DeepSeek V3 671B | 7,168 | 61 | 128 | — | — | 129,280 | 671B | MoE (256E) |

### 3.4 Mixture-of-Experts (MoE) Models

MoE models replace some or all dense FFN layers with a set of expert FFN sub-networks plus a gating (router) network. Key parameters:

```
E         = total number of experts per MoE layer
topk      = experts activated per token (typically 1-2)
L_moe     = number of MoE layers (may be < L; e.g., every 2nd layer)
L_dense   = L - L_moe  (remaining dense FFN layers)
```

**Parameter count:**
```
Per MoE layer:
  Ψ_experts = E × Ψ_ffn            (E copies of the FFN block)
  Ψ_router  = d × E                 (gating linear layer)
  Ψ_attn    = same as dense layer

Per dense layer:
  Ψ_dense_layer = Ψ_attn + Ψ_ffn + Ψ_norm

Total:
  Ψ_total = L_dense × Ψ_dense_layer
          + L_moe × (Ψ_attn + Ψ_experts + Ψ_router + Ψ_norm)
          + Ψ_embedding
```

**Active parameters** (for compute estimation):
```
Ψ_active = L_dense × Ψ_dense_layer
         + L_moe × (Ψ_attn + topk × Ψ_ffn + Ψ_router + Ψ_norm)
         + Ψ_embedding
```

Use Ψ_active (not Ψ_total) in the `C = 6 × Ψ × D` compute formula.

**Memory**: All E experts must be stored in memory (parameters, gradients, optimizer states) even though only topk are active per token. Use Ψ_total for memory calculations. Expert Parallelism (EP) can shard experts across GPUs to reduce per-GPU memory:
```
Experts per GPU = E / N_ep
```
Where N_ep is the expert parallel degree. EP is typically combined with TP and DP.

---

## 4. Compute Estimation (FLOPS)

### 4.1 Simplified (Recommended Default)

For a full training run over D tokens:
```
C = 6 × Ψ × D
```
Breakdown:
- Forward pass: 2ΨD (each parameter involved in ~2 FLOPs per token)
- Backward pass: 4ΨD (gradient computation ≈ 2x forward)

**PaLM per-token formula** (adds the quadratic attention correction):
```
FLOPs_per_token = 6Ψ + 12 × L × d × s
```
The first term is the standard `6N` model FLOPs; the second term (`12Lds`) accounts for the attention score and value reduction matmuls (`Q·K^T` and `scores·V`), which scale with sequence length rather than parameter count. For training over D tokens: `C = (6Ψ + 12Lds) × D`.

**When to use which formula:**
- **Rule of thumb**: `6ΨD` is accurate when `d > s/12`. This condition holds for most large models at standard context lengths (e.g., 175B at s=4096 has <3% from quadratic terms).
- When `d <= s/12`, the quadratic attention term becomes significant: e.g., 175B at s=32768 has ~31% from quadratic terms; models under ~13B can exceed 30% even at moderate context lengths.
- For long-context training (s >= 32K), the `12Lds` term can exceed the `6Ψ` term and must not be ignored. At s=128K with a 7B-class model, the attention term is roughly 5x the parameter term.
- The calculator should always use the PaLM formula and display the attention overhead percentage so users understand the cost of long sequences. It should also check `d > s/12` and flag when the simplified `6ΨD` would be inaccurate.

**MoE models**: For Mixture-of-Experts architectures, Ψ in this formula should be the **active parameters** (parameters routed per token), not the total parameter count. For example, DeepSeek V3 has 671B total parameters but only ~37B active parameters per token, so `C = 6 × 37B × D`.

### 4.2 Per-Layer Detailed

Per transformer layer, per token, forward pass:
```
Attention QKV (MHA): 6 × d²          (3 projections × 2d² each)
Attention QKV (GQA): 2d² × (1 + 2 × a_kv/a)  (Q is full, K/V are reduced)
Attention scores:    2 × s × d        (Q·K^T)
Attention softmax:   3 × a × s        (exp + sum + divide per head; typically negligible)
Attention values:    2 × s × d        (scores · V)
Output projection:   2 × d²
FFN (standard):      16 × d²          (2 linear layers × 4d expansion)
FFN (SwiGLU):        24 × d²          (3 linear layers × 8/3 d expansion, but extra for gating)
```

**GQA FLOPs impact**: For GQA models, the QKV cost drops significantly. For example, LLaMA 2 70B (a_kv/a = 8/64 = 1/8) has QKV FLOPs of 2.5d^2 per token instead of 6d^2. This reduces total per-layer FLOPs by ~15% compared to MHA.

```
Total per layer per token (standard MHA):  24d² + 4sd + 3as
Total per layer per token (SwiGLU + GQA):  2d²(1 + 2a_kv/a) + 4sd + 3as + 2d² + 3 × 2 × d × d_ff
```

The `3as` softmax term is from the DeepMind/Chinchilla method (Hoffmann et al., 2022, Appendix F). It is negligible for typical configs (<0.01% of layer FLOPs) and can be dropped in practice. The simplified `C = 6ΨD` formula follows Kaplan et al. (2020) and omits both the softmax and the sequence-dependent attention terms.

For the simplified formula `C = 6ΨD`, GQA is already accounted for via the reduced parameter count Ψ.

Full model forward, B tokens:
```
C_fwd = B × L × (per-layer FLOPs) + 2BdV  (embedding lookup)
C_total = 3 × C_fwd  (forward + backward)
```

The `4sd` term is the attention quadratic cost — significant for long sequences (s > d).

### 4.3 Chinchilla-Optimal Tokens

Hoffmann et al. (2022) found compute-optimal training uses:
```
D_optimal ≈ 20 × Ψ
```
The calculator should display this as a recommendation alongside user-specified D.

**Caveat**: The 20× ratio is an approximation that holds in the ~10²²–10²⁴ FLOPs regime. The true compute-optimal D/N ratio varies with compute budget because the power-law exponents for model size and data are not equal (alpha ≠ beta in the Chinchilla parametric fit). At significantly larger or smaller scales, the optimal ratio shifts. For a training calculator this approximation is sufficient, but the UI should present it as a guideline, not a hard rule.

In practice, many teams deliberately over-train on tokens to improve inference efficiency (smaller model, more data). LLaMA 3 trained 8B on 15T tokens (≈ 1875× Chinchilla ratio). The calculator should show the Chinchilla ratio: `D / (20 × Ψ)`.

**Practical minimum**: Regardless of Chinchilla optimality, models trained on fewer than ~200B tokens tend to produce poor results. The calculator should warn when D < 200B tokens, even if the Chinchilla ratio is satisfied (e.g., a small model where 20x Psi < 200B).

---

## 5. Memory Estimation

This is the core of the calculator. Memory has four components:

```
M_total = M_model_states + M_activations + M_temporary + M_communication
```

### 5.1 Model States (Parameters + Gradients + Optimizer)

**Mixed precision with AdamW** (the standard):

| Component | Bytes/Param | Purpose |
|-----------|-------------|---------|
| Parameters (bf16) | 2 | Forward/backward computation |
| Gradients (β_grad) | 2 or 4 | Accumulated during backward (see note) |
| Master weights (fp32) | 4 | High-precision copy for optimizer updates |
| Adam 1st moment m (fp32) | 4 | Running mean of gradients |
| Adam 2nd moment v (fp32) | 4 | Running mean of squared gradients |
| **Total** | **16 or 18** | — |

**Gradient precision note**: Frameworks differ on gradient accumulation precision. DeepSpeed/Megatron default to fp32 gradients (β_grad=4, total 18Ψ). PyTorch FSDP and some HuggingFace configs use bf16 gradients (β_grad=2, total 16Ψ). The calculator should default to **fp32 gradients (18Ψ)** as the safer estimate and allow users to select bf16 gradients (16Ψ).

Let Φ = total bytes per parameter = 2 + β_grad + 12 (for AdamW mixed precision), so Φ = 18 (fp32 grads) or 16 (bf16 grads).

So: **M_model_states = ΦΨ bytes** (mixed precision AdamW, Φ = 18 default)

**Other optimizers:**

| Optimizer | Bytes/Param | Breakdown |
|-----------|-------------|-----------|
| AdamW fp32 | 16 | 4 (param) + 4 (grad) + 4 (m) + 4 (v) |
| AdamW mixed (fp32 grads) | 18 | 2+4+4+4+4 = 18 |
| AdamW mixed (bf16 grads) | 16 | 2+2+4+4+4 = 16 |
| AdamW FP8 mixed precision | 14 | 1+1+4+4+4 = 14 |
| AdamW + 8-bit states | 12 | 2+2+4+2+2 = 12 |
| SGD + momentum (mixed) | 12 | 2+2+4+4 = 12 |
| SGD (no momentum, mixed) | 8 | 2+2+4 = 8 |
| Adafactor | 12 | 2+2+4+4 (row+col factors instead of full m,v) |

**FP8 training note**: FP8 mixed precision stores parameters and gradients in fp8 (1 byte each) but master weights and optimizer states remain in fp32. The memory savings over bf16 mixed precision are modest (14 vs 16 bytes/param, ~12% reduction in model states). The primary benefit of FP8 is compute throughput (2x FLOPS on supported hardware), not memory reduction.

**Checkpoint (storage) size**: Training checkpoints saved to disk contain fp32 master weights + Adam m + Adam v (gradients are not saved). For AdamW:
```
Checkpoint size = 12 × Ψ bytes  (4 + 4 + 4 per parameter)
```
This is distinct from live training memory (16-18 bytes/param) because gradients are recomputed on resume. PyTorch checkpoint files include metadata overhead of ~3-5% above the theoretical size. The calculator should display checkpoint size as an output for storage planning (e.g., LLaMA 7B checkpoint = 12 x 6.7B = 80.4 GB per save).

### 5.2 ZeRO Partitioning

ZeRO (Rajbhandari et al., 2020) shards model states across N_dp GPUs. The formulas below use Φ from Section 5.1 (18 with fp32 grads, 16 with bf16 grads):

| Stage | Memory per GPU | What's Sharded |
|-------|---------------|----------------|
| ZeRO-0 | ΦΨ | Nothing |
| ZeRO-1 | (2 + β_grad)Ψ + 12Ψ/N_dp | Optimizer states |
| ZeRO-2 | 2Ψ + (β_grad + 12)Ψ/N_dp | Optimizer states + gradients |
| ZeRO-3 | ΦΨ/N_dp | Everything (params + grads + optimizer) |

With fp32 grads (Φ=18): ZeRO-0 = 18Ψ, ZeRO-1 = 6Ψ + 12Ψ/N_dp, ZeRO-2 = 2Ψ + 16Ψ/N_dp, ZeRO-3 = 18Ψ/N_dp.
With bf16 grads (Φ=16): ZeRO-0 = 16Ψ, ZeRO-1 = 4Ψ + 12Ψ/N_dp, ZeRO-2 = 2Ψ + 14Ψ/N_dp, ZeRO-3 = 16Ψ/N_dp.

**DeepSpeed gradient upcasting note**: DeepSpeed's FusedAdam upcasts all gradients from fp16 to fp32 during the optimizer step, meaning both copies coexist briefly. This adds 2 bytes/param to the sharded portion in ZeRO-2, making it `2Ψ + 18Ψ/N_dp` instead of the theoretical `2Ψ + 16Ψ/N_dp`. The formulas above follow the ZeRO paper's accounting (one gradient copy). For DeepSpeed-specific estimates, add 2Ψ/N_dp to the ZeRO-2 formula. This transient overhead does not affect ZeRO-3 (which shards everything uniformly) or ZeRO-1 (gradients are unsharded).

**Important**: ZeRO-3 adds communication for parameter gathering during forward/backward. This increases communication overhead by ~50% over ZeRO-1.

### 5.3 Activation Memory

Activations are intermediate values stored during forward pass for use in backward pass.

**Per transformer layer** (Korthikanti et al., 2022):

No checkpointing (store everything):
```
M_act_layer = s × b × d × (34 + 5 × a × s / d) bytes
```

Full activation checkpointing (recompute each layer):
```
M_act_layer = 2 × s × b × d bytes  (store only layer input)
```
Cost: ~33% more compute (recompute forward during backward)

Selective activation checkpointing:
```
M_act_layer = s × b × d × (10 + 24/N_tp + 5 × a × s / (d × N_tp)) bytes
```

With Flash Attention (avoids materializing s×s attention matrix):
```
M_act_layer = s × b × d × (10 + 24/N_tp) bytes  (the 5as/d term disappears)
```

**Total activation memory:**
```
M_activations = L_active × M_act_layer / N_pp
```
Where L_active = layers assigned to this pipeline stage = L / N_pp

**Note**: With gradient accumulation G steps, activation memory is for ONE micro-batch, not the full global batch.

### 5.4 Temporary Buffers & Communication

Rough estimate (use as fallback):
```
M_communication ≈ 0.05 × (M_model_states + M_activations)  (5% overhead)
```

More precisely, allocate concrete buffer sizes used by DeepSpeed/Megatron:
- **DP all-reduce buffer**: ~2Ψ × β bytes (ring all-reduce double-buffer)
- **ZeRO allgather bucket**: 500M elements x β bytes (~1 GB in bf16, ~2 GB in fp32)
- **ZeRO-3 max live params**: 1B elements x β bytes (~2 GB in bf16) — parameters gathered for the current layer during forward/backward
- **TP all-reduce**: small, within-layer activations
- **PP send/receive**: s × b × d × β per stage boundary

For ZeRO-2/3 workloads, a practical estimate is **3-5 GB** for communication buffers rather than the 5% heuristic.

### 5.5 Total Memory per GPU

```
M_gpu = M_model_states(ZeRO) + M_activations + M_communication + M_framework_overhead
```

Where M_framework_overhead ≈ 2-5 GB (CUDA context, framework buffers, memory allocator). Empirically, Megatron-DeepSpeed uses ~5 GB; lighter frameworks like bare PyTorch FSDP use ~2 GB.

**Usable GPU memory** = Total VRAM × 0.90 (leave 10% buffer for fragmentation).

### 5.6 Tensor Parallelism Effect on Memory

TP splits weight matrices across N_tp GPUs within a layer:
```
M_params_per_gpu ≈ Ψ_params / N_tp  (approximately — not all layers split perfectly)
M_optimizer_per_gpu ≈ proportional reduction
M_activations: reduced by factor in the 24/N_tp term
```

TP constraint: N_tp ≤ GPUs per node (typically 8) because it requires NVLink bandwidth.

### 5.7 Pipeline Parallelism Effect

PP distributes layers across stages:
```
Layers per stage = L / N_pp
M_params_per_gpu = Ψ × (layers_per_stage / L)  (proportional to layers held)
M_activations_per_gpu = per-layer activation × layers_per_stage
```

PP overhead (pipeline bubble):
```
Bubble fraction = (N_pp - 1) / (num_microbatches + N_pp - 1)
```
Rule of thumb: need num_microbatches ≥ 4 × N_pp to keep bubble < 20%.

---

## 6. Training Time Estimation

### 6.1 Core Formula

```
T_seconds = C / (N_gpu × F_peak × MFU)
```

Where:
- C = total FLOPS (from Section 4), **adjusted for activation recomputation** (see below)
- F_peak = peak FLOPS per GPU in the relevant precision (bf16 for mixed precision)
- MFU = Model FLOPS Utilization

**Activation recomputation adjustment**: The base formula `C = 6ΨD` assumes no activation recomputation. When activation checkpointing is enabled, the forward pass is recomputed during backward, increasing total compute:

| Checkpointing Mode | Compute Formula | Overhead |
|---------------------|-----------------|----------|
| None | C = 6ΨD | Baseline |
| Selective | C ≈ 7ΨD | ~17% more |
| Full | C = 8ΨD | 33% more |

The calculator must apply this multiplier when the user selects activation checkpointing. Full recomputation doubles the forward pass cost (2ΨD becomes 4ΨD), giving 4ΨD + 4ΨD = 8ΨD total.

### 6.2 MFU vs HFU

**MFU (Model FLOPS Utilization)** measures achieved throughput against peak hardware FLOPS using the *ideal* model FLOPs (6ΨD), regardless of implementation choices like activation checkpointing:
```
MFU = (6Ψ × tokens_per_second) / F_peak
```

**HFU (Hardware FLOPS Utilization)** measures the same ratio but using *actual executed* FLOPs, including recomputation overhead. With full activation checkpointing, the executed FLOPs are 8ΨD instead of 6ΨD:
```
HFU = (actual_FLOPs_per_second) / F_peak
```

MFU is the fairer comparison metric because it is independent of implementation choices. A system with activation checkpointing will always show higher HFU than MFU (since it does more work per token), but this does not mean it is faster. The calculator should use **MFU** (based on 6ΨD) for its utilization display and training time formula, and apply the activation checkpointing overhead separately (Section 6.1).

### 6.3 MFU Guidelines

MFU depends on model size, batch size, parallelism, and hardware:

| Scenario | Typical MFU |
|----------|-------------|
| Small model (<1B), 1-8 GPUs | 25-35% |
| Medium model (1B-10B), 8-64 GPUs | 35-45% |
| Large model (10B-100B), 64-512 GPUs | 40-50% |
| Very large (100B+), 512+ GPUs | 45-55% |
| State-of-the-art (PaLM-scale) | 55-65% |

The calculator should provide a default MFU based on model size and GPU count, with a slider for user override (range: 10-70%).

### 6.4 Communication Overhead

For more precise time estimates, subtract communication time:

DP all-reduce per step:
```
T_dp_comm = 2 × Ψ × β / (bandwidth_inter_node × N_dp)
```

PP bubble per step:
```
T_pp_bubble = bubble_fraction × T_compute_per_step
```

Effective throughput:
```
Tokens/sec = B / T_step = B / (T_compute + T_communication)
```

---

## 7. GPU Hardware Specifications

Embed these as selectable presets. Users should also be able to enter custom GPU specs.

| GPU | VRAM (GB) | BF16 TFLOPS | FP8 TFLOPS | Mem BW (GB/s) | NVLink BW (GB/s) | TDP (W) |
|-----|-----------|-------------|------------|---------------|-------------------|---------|
| V100 32GB | 32 | 125 | — | 900 | 300 | 300 |
| A100 40GB | 40 | 312 | — | 1,555 | 600 | 400 |
| A100 80GB | 80 | 312 | — | 2,039 | 600 | 400 |
| H100 SXM | 80 | 989 | 1,979 | 3,350 | 900 | 700 |
| H100 NVL | 94 | 989 | 1,979 | 3,350 | 900 | 800 |
| H200 SXM | 141 | 989 | 1,979 | 4,800 | 900 | 700 |
| B200 | 192 | 2,250 | 4,500 | 8,000 | 1,800 | 1,000 |
| GB200 NVL72 | 384 | 4,500 | 9,000 | 16,000 | 1,800 | 2,700 |
| MI300X | 192 | 1,307 | 2,614 | 5,300 | — | 750 |

**GPUs per node**: Typically 8 for NVIDIA (DGX), 8 for AMD. This constrains max TP degree.

---

## 8. Cost Estimation

```
Cost = N_gpu × T_hours × price_per_GPU_hour
```

Provide default pricing presets (user can override):

| GPU | Approx. On-Demand ($/hr/GPU) |
|-----|------------------------------|
| V100 32GB | $1.50 - $2.50 |
| A100 80GB | $2.50 - $4.00 |
| H100 SXM | $3.00 - $5.00 |
| H200 SXM | $4.00 - $6.00 |
| B200 | $5.00 - $8.00 |

The calculator should accept a custom $/GPU/hr input and show total estimated cost.

---

## 9. Parallelism Recommendation Engine

Given memory constraints and GPU count, recommend a parallelism strategy:

### Decision Logic

```
1. Calculate M_model_states = ΦΨ (no parallelism; Φ from Section 5.1)
2. If M_model_states fits in one GPU (with room for activations):
   → Use DP only. N_dp = N_gpu.
3. Else, try ZeRO stages (using Φ-based formulas from Section 5.2):
   a. ZeRO-1: (2 + β_grad)Ψ + 12Ψ/N_dp — fits? Use this.
   b. ZeRO-2: 2Ψ + (β_grad + 12)Ψ/N_dp — fits? Use this.
   c. ZeRO-3: ΦΨ/N_dp — fits? Use this.
4. If ZeRO alone doesn't fit, add model parallelism:
   → Add TP (start with N_tp = 2, increase to 4, 8)
   → Recalculate per-GPU memory with TP
   → Combine with ZeRO-1 (preferred) or ZeRO-2/3 (no PP constraint yet)
5. If TP=8 still insufficient:
   → Add PP (start with N_pp = 2, increase as needed)
   → Each stage holds fewer layers → less memory
   → CONSTRAINT: PP requires ZeRO-0 or ZeRO-1 only (see compatibility table below).
     If ZeRO-2/3 was selected in step 3/4, downgrade to ZeRO-1 when adding PP.
6. Remaining GPUs become DP:
   N_dp = N_gpu / (N_tp × N_pp)
```

### Minimum GPU Memory Floor (Largest Layer)

Even with ZeRO-3 sharding across arbitrarily many GPUs, a single transformer layer's parameters must be fully gathered on one GPU during forward and backward passes. During backward, both the gathered parameters and their gradients coexist in memory. This sets an absolute minimum VRAM requirement:

```
Ψ_largest_layer = Ψ_attn + Ψ_ffn + Ψ_norm  (single transformer block)
M_min_gpu = Ψ_largest_layer × 2β              (β bytes for gathered params + β bytes for gradients)
```

In bf16 (β=2), this is 4 bytes per parameter in the largest layer. For example, LLaMA 70B has ~1.1B params per layer, so the minimum per-GPU memory is ~4.4 GB in bf16 (2.2 GB params + 2.2 GB gradients). In practice, activations and working memory push this higher. Display this as an output: "Minimum GPU VRAM (even with full sharding)".

### Constraints
- N_tp must divide both a (attention heads) and a_kv (KV heads) evenly. For GQA models, a_kv is the binding constraint (e.g., LLaMA 2 70B has a_kv=8, so N_tp must divide 8)
- N_tp ≤ 8 (GPUs per node, NVLink requirement)
- N_pp must divide L (layers) evenly
- N_dp × N_tp × N_pp = N_gpu (for dense models)
- N_dp × N_tp × N_pp × N_ep = N_gpu (for MoE models; N_ep must divide E evenly)
- Global batch size B = b × G × N_dp
- **ZeRO + Pipeline Parallelism compatibility**: ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism (gradient sharding conflicts with PP's gradient accumulation across stages). Only ZeRO-0 or ZeRO-1 can be combined with PP. The calculator must enforce this constraint:

| ZeRO Stage | + TP | + PP |
|------------|------|------|
| ZeRO-0 | Yes | Yes |
| ZeRO-1 | Yes | Yes |
| ZeRO-2 | Yes | **No** |
| ZeRO-3 | Yes | **No** |

When PP is active, the recommendation engine must not select ZeRO-2 or ZeRO-3. Conversely, if ZeRO-2/3 is needed for memory, PP cannot be used.

### Output
Display the recommended configuration: N_dp × N_tp × N_pp, ZeRO stage, and estimated pipeline bubble overhead.

---

## 10. Phase 2: Post-Training

Post-training covers everything after pretraining: supervised fine-tuning and preference alignment. The key difference is **multiple models may be in memory simultaneously**.

### 10.1 Supervised Fine-Tuning (SFT)

**Full fine-tuning**: Identical to pretraining memory (16Ψ + activations). Dataset is smaller so compute is less.

**LoRA** (Low-Rank Adaptation):
```
Base model (frozen, bf16):  2Ψ bytes  (no gradients, no optimizer)
LoRA adapter parameters:    Ψ_lora = 2 × r × d × M_modules × L
  where r = rank (8-64), M_modules = adapted modules per layer (typically 4: Q,K,V,O)
LoRA gradients (bf16):      2 × Ψ_lora
LoRA optimizer (fp32):      12 × Ψ_lora  (master + Adam m + v)
Activations:                Same as full model (entire model runs forward/backward)

M_total_lora = 2Ψ + 16 × Ψ_lora + M_activations
```

Example: 7B model, rank 16, 4 modules/layer, 32 layers:
```
Ψ_lora = 2 × 16 × 4096 × 4 × 32 = 16.8M  (0.24% of base model)
Memory: 2 × 7B + 16 × 16.8M = 14GB + 0.27GB = ~14.3GB + activations
```

**QLoRA** (Quantized LoRA):
```
Base model (4-bit NF4):    ~0.5Ψ bytes + ~0.01Ψ overhead (quantization constants)
LoRA adapters + optimizer: 16 × Ψ_lora (same as LoRA)
Activations:               Computed in bf16 (dequantize → compute → re-quantize)

M_total_qlora ≈ 0.55Ψ + 16 × Ψ_lora + M_activations
```

### 10.2 Direct Preference Optimization (DPO)

Two models in memory:
```
Policy model (trainable):   16Ψ (full) or 2Ψ + 16Ψ_lora (LoRA)
Reference model (frozen):   2Ψ
Activations:                2× normal (forward through both models for chosen + rejected)
DPO log-prob storage:       2 × B × s × 4 bytes (chosen + rejected)

M_total_dpo = 18Ψ + 2 × M_activations  (full fine-tuning)
            = 4Ψ + 16Ψ_lora + 2 × M_activations  (LoRA)
```

Compute per step: ~8ΨB tokens (6Ψ policy train + 2Ψ reference forward).

### 10.3 PPO / RLHF

The most memory-intensive method — up to **4 models simultaneously**:

```
Actor (policy, trainable):      16Ψ_actor + M_act
Critic (value model, trainable): 16Ψ_critic + M_act
Reference model (frozen):       2Ψ_ref
Reward model (frozen):           2Ψ_reward

PPO rollout buffers (per sample):
  - log_probs:   s × 4 bytes
  - values:      s × 4 bytes
  - advantages:  s × 4 bytes
  - returns:     s × 4 bytes
  Total: ~16 × s bytes per sample, × batch size
```

**Peak memory** (if all models same size Ψ):
```
M_peak_ppo = 16Ψ + 16Ψ + 2Ψ + 2Ψ + activations + buffers
           = 36Ψ + activations + buffers
```

With model offloading (load/unload between phases):
```
Generation phase: 2Ψ_actor (inference) + M_kv_cache
Scoring phase:    2Ψ_reward (inference)
Training phase:   16Ψ_actor + 16Ψ_critic + 2Ψ_ref + activations
Peak = max of above = 34Ψ + activations
```

**KV cache during generation** (applies to PPO and GRPO generation phases):
```
M_kv_cache = batch × 2 × L × a_kv × d_kv × s_gen × β_cache
```
Where the factor of 2 is for K and V tensors, `a_kv × d_kv` is the per-layer KV width (equals `d` for MHA, smaller for GQA/MQA), `s_gen` is the generation sequence length, and `β_cache` is bytes per element (2 for bf16). For GQA models, the KV cache shrinks proportionally to `a_kv / a`.

**Common optimization**: Critic is smaller than actor (e.g., half the layers), and reference model shares architecture but is frozen.

Compute per PPO step (K PPO epochs):
```
C_ppo_step = 2Ψ × generated_tokens  (generation)
           + 2Ψ_reward × scored_tokens  (reward)
           + K × (6Ψ_actor + 6Ψ_critic + 2Ψ_ref) × batch_tokens  (training)
```

### 10.4 GRPO (Group Relative Policy Optimization)

Simpler than PPO — no critic model, uses group-relative advantages:
```
Policy model (trainable):   16Ψ + M_act
Reference model (frozen):   2Ψ

M_total_grpo = 18Ψ + M_activations
```

Key difference: generates G completions per prompt, so the generation-phase KV cache (see formula in Section 10.3) scales with G:
```
Generation KV cache: M_kv_cache with batch = G × num_prompts
```

Where G is typically 4-16 completions per prompt.

### 10.5 Post-Training Compute

| Method | FLOPS per Token | Notes |
|--------|----------------|-------|
| SFT | 6Ψ | Same as pretraining |
| DPO | 8Ψ | Policy train (6Ψ) + reference forward (2Ψ) |
| PPO | ~20Ψ per step | Generation + reward + multi-epoch training |
| GRPO | ~10Ψ per step | Generation + policy train (no critic) |

Post-training datasets are much smaller (10K-1M examples vs. trillions of tokens for pretraining), so total compute is orders of magnitude less.

---

## 11. Feature Requirements

### 11.1 Input Modes

**Quick Mode**: User enters total parameter count (e.g., "7B") + total tokens + GPU type → instant estimate.

**Detailed Mode**: User specifies full architecture (d, L, a, a_kv, d_ff, V) + all training config → precise breakdown.

**Preset Mode**: User selects from model presets (Section 3.3) → auto-fills architecture.

### 11.2 Pretraining Calculator Features

**Inputs:**
1. Model specification (preset, quick, or detailed)
2. Dataset size D (tokens) — show Chinchilla-optimal recommendation
3. Training precision (fp32 / bf16 / fp16 / fp8)
4. Optimizer (AdamW / AdamW 8-bit / SGD+momentum / Adafactor)
4a. Gradient accumulation precision (fp32 default / bf16) — affects Φ and ZeRO formulas
5. Micro-batch size b
6. Sequence length s
7. Gradient accumulation steps G
8. Activation checkpointing (none / selective / full)
9. Flash Attention (on/off)
10. GPU type (preset or custom specs)
11. Target training time (optional — for computing minimum GPUs)
12. Number of GPUs (optional — for computing training time)
13. MFU override (slider, 10-70%, with smart default)
14. Parallelism: auto-recommend OR manual (N_tp, N_pp, N_dp, ZeRO stage)
15. Cost per GPU-hour (with cloud provider presets)

**Outputs:**
1. Total parameter count (computed from architecture)
2. Total FLOPS
3. Chinchilla ratio (D / 20Ψ) and recommendation
4. **Memory breakdown per GPU** (visual bar/pie):
   - Parameters
   - Gradients
   - Optimizer states
   - Activations
   - Communication buffers
   - Framework overhead
   - Free headroom
5. Minimum GPUs needed (memory-constrained)
6. Minimum GPU VRAM floor (largest transformer block — see Section 9)
7. Recommended parallelism strategy (N_dp × N_tp × N_pp, ZeRO stage)
8. Pipeline bubble overhead %
9. Estimated training time (days/hours)
10. Estimated tokens/second throughput
11. Estimated total cost ($)
12. Global batch size (computed: b × G × N_dp)
13. Checkpoint size (12Ψ bytes for AdamW -- see Section 5.1) for storage planning
14. Attention overhead percentage (12Lds / 6Ψ -- see Section 4.1) to flag long-context cost

### 11.3 Post-Training Calculator Features

**Inputs:**
1. Base model (preset or parameter count)
2. Method: SFT / DPO / PPO / GRPO
3. Fine-tuning approach: Full / LoRA / QLoRA
4. LoRA config (if applicable): rank r, alpha, target modules
5. For PPO: critic model size, reward model size
6. For GRPO: group size G
7. Dataset size (examples)
8. Epochs
9. Sequence length, batch size
10. GPU type and count

**Outputs:**
1. Memory breakdown per GPU (showing all models):
   - Trainable model(s) with optimizer states
   - Frozen model(s) in inference mode
   - LoRA adapter overhead (if applicable)
   - Activations
   - PPO buffers (if applicable)
2. Number of GPUs needed
3. Estimated training time
4. Estimated cost

### 11.4 Cross-Phase Features

- **GPU Comparison**: side-by-side comparison of 2-3 GPU types for the same workload
- **Summary Dashboard**: combined pretraining + post-training requirements
- **Export/Share**: copy results as formatted text or JSON

---

## 12. UI/UX Requirements

### 12.1 Layout

The calculator lives at `/tools/gpu-calculator`. It should be a single-page tool with two main sections (tabs or scroll sections):

1. **Pretraining** section
2. **Post-Training** section

Each section has:
- **Left panel**: Input controls (sliders, dropdowns, number inputs)
- **Right panel**: Results and visualizations

On mobile: stack vertically (inputs on top, results below).

### 12.2 Visual Design

Follow the portfolio's existing aesthetic:
- Clean, minimal design with generous whitespace
- OKLch color palette from CSS variables
- Display font for headings (`var(--font-display)` with `fontVariationSettings`)
- Smooth transitions between states (Framer Motion)
- Subtle gradient backgrounds on card surfaces

### 12.3 Key Visualizations

1. **Memory breakdown bar**: Horizontal stacked bar showing what fills GPU VRAM
   - Each segment colored distinctly (parameters, gradients, optimizer, activations, free)
   - Show percentage and absolute GB on hover
   - Animate when values change
   - Red warning indicator if memory exceeds GPU capacity

2. **GPU utilization indicator**: Circular gauge or bar showing memory fill %
   - Green (<70%), yellow (70-90%), red (>90%)

3. **Parallelism layout**: Simple grid showing how GPUs are organized
   - DP × TP × PP visualization
   - Highlight which dimension each GPU belongs to

4. **Cost timeline** (optional): How cost accumulates over training time

### 12.4 Interactivity

- Inputs should update results in real-time (no submit button)
- Use debouncing for number inputs (300ms)
- Sliders for continuous values (MFU, learning rate)
- Dropdowns for categorical choices (GPU type, optimizer, model preset)
- Number inputs for precise values (parameter count, tokens, batch size)
- Collapsible "Advanced" sections for less common parameters
- Tooltip/info icons explaining each input

### 12.5 Responsive Design

- Desktop: side-by-side layout (inputs | results)
- Tablet: stacked with compact controls
- Mobile: single column, results below inputs
- Breakpoints: match existing site (640px, 1024px)

---

## 13. Implementation File Structure

```
app/tools/gpu-calculator/
  page.tsx                          # Tool page (metadata, layout, breadcrumb)
  gpu-calculator-embed.tsx          # Dynamic import wrapper (ssr: false)

components/gpu-calculator/
  gpu-calculator.tsx                # Main calculator component
  types.ts                          # TypeScript interfaces
  constants.ts                      # GPU specs, model presets, defaults
  formulas/
    compute.ts                      # FLOPS calculations
    memory.ts                       # Memory estimation (all components)
    parallelism.ts                  # Parallelism recommendation engine
    cost.ts                         # Cost estimation
    post-training.ts                # SFT/DPO/PPO/GRPO specific formulas
  components/
    pretraining-panel.tsx           # Pretraining inputs + results
    post-training-panel.tsx         # Post-training inputs + results
    memory-breakdown-bar.tsx        # Stacked bar visualization
    gpu-utilization-gauge.tsx       # Memory fill indicator
    parallelism-layout.tsx          # DP×TP×PP grid visualization
    model-selector.tsx              # Preset/quick/detailed model input
    gpu-selector.tsx                # GPU type selector with specs
    results-summary.tsx             # Combined results dashboard
    input-controls.tsx              # Shared input components (sliders, dropdowns)
```

Register the tool in `lib/utils/tools.ts`:
```typescript
{
  slug: "gpu-calculator",
  title: "LLM Training GPU Calculator",
  summary: "Estimate GPU requirements for LLM training — compute memory breakdown, parallelism strategy, training time, and cost across pretraining and post-training phases.",
  category: "Planning & Estimation",
  tags: ["llm", "training", "gpu", "compute", "distributed-training"],
  relatedPost: undefined,  // or link to a future blog post
}
```

---

## 14. Validation & Edge Cases

### Input Validation
- Parameter count: must be positive, warn if < 1M or > 10T
- Tokens: must be positive, warn if Chinchilla ratio < 1 or > 200
- Micro-batch size: must be ≥ 1
- Sequence length: must be positive, typical range 512 - 131,072
- GPU count: must be ≥ 1, warn if > 100,000
- TP must divide attention heads evenly
- PP must divide layers evenly
- N_dp × N_tp × N_pp must equal N_gpu

### Edge Cases
- Model too large for ANY single GPU → must use TP or ZeRO-3
- Activations dominate memory (very long sequences, large batch)
- Pipeline bubble > 50% → warn user to increase micro-batches
- ZeRO-3 communication overhead makes training slower → warn
- QLoRA base model quantized to 4-bit but activations still bf16
- PPO with all 4 models same size → needs 36× parameter bytes
- MoE models: total params ≠ active params — use Ψ_active for compute (Section 3.4, 4.1) but Ψ_total for memory (Section 5.1); Expert Parallelism may be needed

### Numerical Safety
- All calculations in JavaScript `number` (64-bit float) — adequate for up to ~10^15
- Display large numbers with appropriate units (M, B, T for parameters; GFLOPS, TFLOPS, PFLOPS)
- Avoid integer overflow: 405B × 16 bytes = 6.48TB — fine in float64

---

## 15. Test Cases

Use these to validate the calculator produces correct results.

**Note**: These test cases use Φ=16 (bf16 gradients). With fp32 gradients (Φ=18, the default), model state numbers increase by ~12%. Both should be supported by the calculator.

### Test 1: LLaMA 7B Pretraining
```
Input: Ψ=6.7B, D=1T tokens, bf16 mixed precision, AdamW (bf16 grads), H100 SXM
Memory per GPU (no parallelism):
  Model states: 6.7B × 16 = 107.2 GB  → doesn't fit in 80GB!
  (With fp32 grads: 6.7B × 18 = 120.6 GB → even more reason for sharding)
  → Needs at least ZeRO-1 with N_dp ≥ 2, or TP=2

With 8× H100 (1 node), ZeRO-1, DP=8 (bf16 grads):
  Model states/GPU: 4×6.7B + 12×6.7B/8 = 26.8 + 10.05 = 36.85 GB
  Activations (s=4096, b=2, full ckpt): 2×4096×2×4096 = 67MB/layer × 32 = 2.1 GB
  Total: ~39 GB ✓ fits in 80GB

Compute: 6 × 6.7B × 1T = 40.2 × 10²¹ FLOPS = 40.2 ZFLOPS
Time: 40.2e21 / (8 × 989e12 × 0.45) = 40.2e21 / 3.56e15 = 11.3e6 seconds ≈ 131 days
```

### Test 2: LLaMA 70B Pretraining
```
Input: Ψ=70B, D=2T tokens, 256× H100 SXM
Model states: 70B × 16 = 1,120 GB total

With TP=8, PP=4, DP=8, ZeRO-1:
  Params per GPU (TP×PP): 70B / (8×4) = 2.19B → 35 GB model states
  Optimizer per GPU (ZeRO-1, DP=8): 12 × 2.19B / 8 ≈ 3.3 GB
  Total model states: ~38.3 GB + activations
  Should fit in 80GB ✓

Compute: 6 × 70B × 2T = 840 ZFLOPS
Time: 840e21 / (256 × 989e12 × 0.50) ≈ 6.6e6 sec ≈ 77 days
```

### Test 3: 7B SFT with LoRA
```
Input: Ψ=7B, LoRA rank=16, 4 modules/layer, 32 layers, 1× H100
Ψ_lora = 2 × 16 × 4096 × 4 × 32 = 16.8M
Base model: 2 × 7B = 14 GB
LoRA states: 16 × 16.8M = 268 MB
Activations: ~2-4 GB (s=2048, b=4)
Total: ~16-18 GB  → fits on single H100 ✓
```

### Test 4: PPO on 70B
```
Input: actor=70B, critic=70B, ref=70B, reward=70B
Peak memory: 36 × 70B = 2,520 GB
Minimum GPUs (H100 80GB, ZeRO-3): 2520 / (80 × 0.9) = 35 GPUs minimum
Realistically with activations: 64-128 GPUs
```

---

## 16. Deliverable

Produce a **detailed implementation plan** that includes:

1. **Component architecture**: How to structure the React component tree, state flow, and data flow between formulas and UI
2. **Implementation phases**: Ordered steps to build incrementally (start with pretraining memory → add compute → add parallelism → add post-training → add visualizations)
3. **File-by-file breakdown**: What each file contains, key functions, types
4. **Formula implementation**: How to translate the math above into TypeScript functions with clear types
5. **UI component specs**: For each visualization component, describe what it renders, its props, and interaction behavior
6. **State management design**: What state lives where, how inputs flow to computed outputs
7. **Testing strategy**: How to verify correctness against the test cases
8. **Integration steps**: How to register the tool and create the route in the existing portfolio

Prioritize correctness of the formulas and clarity of the memory breakdown visualization — these are the core value of the tool.
