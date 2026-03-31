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
| D | Total training tokens (may include repeated data) |
| U | Unique training tokens (U ≤ D; defaults to D when all data is unique) |
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
| N_sp | Sequence parallel degree (= N_tp when enabled; see Section 5.3) |
| N_gpu | Total GPUs = N_dp × N_tp × N_pp |
| β | Bytes per parameter in compute precision (2 for bf16, 4 for fp32) |
| β_grad | Bytes per gradient element (2 for bf16, 4 for fp32) |
| Φ | Total bytes per parameter for model states = 2 + β_grad + 12 (AdamW mixed) |
| N_inst | Number of instances = ceil(N_gpu / GPUs_per_node) |
| f | Instance failure rate (failures per instance per day; see Section 6.5) |
| f_checkpoint | Checkpoint saves per day (see Section 6.5) |
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
| GPT-2 Medium | 1,024 | 24 | 16 | 16 | 4,096 | 50,257 | 350M | Standard |
| GPT-2 Large | 1,280 | 36 | 20 | 20 | 5,120 | 50,257 | 774M | Standard |
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
         + Ψ_output_proj
```
Where `Ψ_output_proj = V × d` (the lm_head matmul). The input embedding (`V × d`) is excluded because it is a table lookup, not a matrix multiplication, contributing zero FLOPs (see Section 4.1). For tied embeddings, `Ψ_output_proj` shares weights with the input embedding but is still used as a matmul. For untied embeddings, the input embedding parameters appear only in `Ψ_total` (for memory) but not in `Ψ_active` (for compute).

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

**FLOP counting convention**: This spec counts 1 FLOP = 1 floating-point operation. A fused multiply-add (FMA/MAC) counts as **2 FLOPs** (one multiply + one add). The factor of 6 in `6ΨD` is therefore `2 FLOPs/MAC × 3 passes` (1 forward + 2 backward). Some tools — notably Facebook's fvcore and PyTorch's `thop` — report MACs (multiply-accumulate operations) instead of FLOPs, producing numbers exactly **half** of ours for the same computation. When comparing this calculator's output to external tools, check whether they report FLOPs or MACs.

**PaLM per-token formula** (adds the quadratic attention correction):
```
FLOPs_per_token = 6Ψ + 12 × L × d × s
```
The first term is the standard `6N` model FLOPs; the second term (`12Lds`) accounts for the attention score and value reduction matmuls (`Q·K^T` and `scores·V`), which scale with sequence length rather than parameter count. For training over D tokens: `C = (6Ψ + 12Lds) × D`.

**Important**: The `d` in the attention term `12Lds` is the total Q/K/V projection width (`n_heads × d_head`), not necessarily `d_model`. For most models these are equal, but some architectures (notably PaLM 540B, where `d_model=18432` but `n_heads × d_head = 48 × 256 = 12288`) use a smaller projection width than the model hidden dimension. When `d_model != n_heads × d_head`, use `n_heads × d_head` for the attention term. The calculator should derive `d` for the attention term from the model's head configuration rather than assuming it equals `d_model`.

**When to use which formula:**
- **Rule of thumb**: `6ΨD` is accurate when `d > s/12`. This condition holds for most large models at standard context lengths (e.g., 175B at s=4096 has <3% from quadratic terms).
- When `d <= s/12`, the quadratic attention term becomes significant: e.g., 175B at s=32768 has ~31% from quadratic terms; models under ~13B can exceed 30% even at moderate context lengths.
- For long-context training (s >= 32K), the `12Lds` term can exceed the `6Ψ` term and must not be ignored. At s=128K with a 7B-class model, the attention term is roughly 5x the parameter term.
- The calculator should always use the PaLM formula and display the attention overhead percentage so users understand the cost of long sequences. It should also check `d > s/12` and flag when the simplified `6ΨD` would be inaccurate.

**Embedding exclusion from FLOPs**: The input embedding layer is a table lookup (indexing into a `V x d` matrix), not a matrix multiplication, and contributes **zero FLOPs**. The output projection (lm_head) IS a matmul and contributes `2 x d x V` FLOPs per token (forward). When using the simplified `6ΨD` formula, Ψ should ideally exclude the input embedding parameters for FLOPs accuracy. For **tied embeddings** (input embedding = lm_head), the shared `V x d` weight IS used as a matmul in the output projection, so no correction is needed. For **untied embeddings**, the input embedding's `V x d` parameters are pure lookups and should be subtracted: use `Ψ_flops = Ψ - V x d` in the compute formula. MosaicML's LLM Foundry makes this correction explicitly. In practice, for large models (7B+) the input embedding is <2% of total parameters, so the overcount is small; for models under 1B it can be 5-10%.

**Small-model accuracy of 6ND**: Even when the sequence length condition (`d > s/12`) is satisfied, the `6ND` approximation underestimates total FLOPs for small models because the logit output projection (`2sdV` for forward) is proportionally larger relative to Ψ. Empirical ratios of exact-to-6ND FLOPs: ~1.10 at 300M, ~1.04 at 1B, ~1.00 at 7B. The calculator should note that `6ND` may underestimate by up to 10% for models under 1B parameters. Above ~7B, the approximation is essentially exact (excluding the attention quadratic term, which is handled separately by the PaLM formula).

**MoE models**: For Mixture-of-Experts architectures, Ψ in this formula should be the **active parameters** (parameters routed per token), not the total parameter count. For example, DeepSeek V3 has 671B total parameters but only ~37B active parameters per token, so `C = 6 × 37B × D`.

**MoE load balance overhead**: In practice, MoE expert routing is not perfectly uniform -- some experts receive more tokens than others, and the slowest expert determines step time. This adds a load balance overhead to MoE compute:
```
C_moe_effective = C_moe × load_balance_factor
```
Where `load_balance_factor` typically ranges from 1.05 to 1.2 (5-20% overhead). Default to **1.1** (10% overhead). A factor of 1.0 assumes perfect load balancing (theoretical minimum). The calculator should apply this multiplier to the MoE portion of compute (expert FLOPs only, not attention or dense layers) and expose it as an advanced input for MoE models.

**Note on alternative attention FLOPs formulas**: Some implementations (notably karpathy/llm.c, following Kaplan et al. 2020 Section 2.1) use `6LCT` for the attention term instead of `12Lds`. The factor of 6 vs 12 arises because `12Lds` separately counts both attention matmuls (Q*K^T and scores*V), each contributing `2sd` forward FLOPs per layer (x 3 for fwd+bwd = `12sd` total), while the `6LCT` variant uses a different convention that effectively halves the attention cost. The `12Lds` formulation from PaLM is the standard per-matmul accounting and is what this calculator uses.

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
C_fwd = B × L × (per-layer FLOPs) + 2BdV  (output projection / lm_head; embedding lookup is 0 FLOPs — pure memory index)
C_total = 3 × C_fwd  (forward + backward)
```

The `4sd` term is the attention quadratic cost — significant for long sequences (s > d).

**MoE per-layer FLOPs**: For MoE layers, the FFN FLOPs are replaced by expert FLOPs (only the active experts) plus router FLOPs:
```
Per MoE layer, per token, forward pass:
  Router:              2 × d × E           (linear gating layer)
  Active experts (standard FFN):  topk × 4 × d × d_ff   (topk experts, each with 2 linear layers)
  Active experts (SwiGLU FFN):    topk × 6 × d × d_ff   (topk experts, each with 3 projections)
  Attention:           same as dense layer
```
The router cost is small relative to expert FLOPs (typically <0.5% of per-layer compute) but should be included for completeness. The simplified `C = 6 × Ψ_active × D` formula accounts for this implicitly since `Ψ_active` includes `Ψ_router` (Section 3.4).

### 4.3 Chinchilla Scaling Law and Compute-Optimal Tokens

#### Loss Prediction Formula

Hoffmann et al. (2022) fit a parametric loss model over 400+ training runs:
```
L(N, D) = E + A / N^alpha + B / D^beta
```
Where N is model parameters, D is training tokens, and the fitted coefficients (Chinchilla, Table A.3) are:
```
alpha = 0.34,  beta = 0.28,  A = 406.4,  B = 410.7,  E = 1.69
```
The three terms represent: irreducible loss (E), underfitting from model size (A/N^alpha), and underfitting from data (B/D^beta). All loss values (both Kaplan and Chinchilla) are in **nats** (natural log base); to convert to bits, divide by ln(2) ≈ 0.693. The calculator should use this formula to display **predicted training loss** for the user's chosen (N, D) combination, labeled in nats.

#### Quick Rule: D_optimal ≈ 20N

The widely-cited approximation for compute-optimal training:
```
D_optimal ≈ 20 × Ψ
```

**More accurate power-law fit**: The Chinchilla Approach 2 empirical data shows the D/N ratio is not constant at 20x but increases systematically with model size (from ~19x at 400M to ~27x at 1T parameters). A power-law fit to this data gives:
```
D_optimal = 8.62 × N^1.041
```
where N and D are both in raw counts (not billions). The 20x rule is only accurate near 1B parameters; at 10B+ the ratio is 22-30x. The calculator should use this power-law fit for its Chinchilla-optimal recommendation and display the simple 20x rule as a secondary reference.

#### Exact Compute-Optimal Allocation

Given a total compute budget C (in FLOPs), the closed-form compute-optimal model size and token count are:
```
N*(C) = ((alpha*A) / (beta*B))^(1/(alpha+beta)) * (C/6)^(beta/(alpha+beta))
D*(C) = ((beta*B) / (alpha*A))^(1/(alpha+beta)) * (C/6)^(alpha/(alpha+beta))
```
Because alpha != beta, the optimal D/N ratio is **not constant** -- it grows slowly with compute budget. The 20x rule is the ratio at roughly 10^22-10^24 FLOPs. The calculator can use these formulas to give a more precise Chinchilla-optimal recommendation when the user specifies a compute budget or GPU-hours target.

#### Coefficient Sensitivity Caveat

The fitted coefficients vary significantly with the training regime used to fit them. Sardana et al. (2024, "Beyond Chinchilla-Optimal") show:

| Data Range (tok/param) | alpha | beta | A | B | E |
|---|---|---|---|---|---|
| <= 100 | 0.08 | 0.13 | 7.199 | 25.97 | 0.17 |
| <= 250 | 0.13 | 0.16 | 14.23 | 39.54 | 0.98 |
| <= 500 | 0.13 | 0.16 | 17.07 | 35.80 | 0.95 |
| All Data (up to 10,000x) | 0.18 | 0.24 | 33.66 | 138.9 | 1.45 |
| Chinchilla (original, ~20x) | 0.34 | 0.28 | 406.4 | 410.7 | 1.69 |

The coefficients shift substantially depending on which token/parameter ratio range the fitting data covers. The Chinchilla coefficients were fit on runs near the compute-optimal frontier (~20x). At moderate overtraining (<=100x), alpha and beta are roughly half the Chinchilla values; the loss curve is flatter but still monotonically decreasing out to 10,000x with no observed data saturation (Sardana et al. tested 47 models from 150M-6B params). The calculator should select the row matching the user's D/N ratio for loss prediction, defaulting to the "All Data" row when the ratio exceeds 500x. At extreme over-training ratios (like LLaMA 3's 1875x), the original Chinchilla coefficients overestimate the benefit of additional data and underestimate achievable loss. Additionally, there is a known internal inconsistency within the Chinchilla paper itself: minimizing the Approach 3 parametric loss function L(N,D) with these coefficients does not reproduce the Approach 2 empirical compute-optimal points (e.g., Approach 3 predicts D_opt=14.4B for N=400M, while Approach 2 measures D_opt=9.2B). This has been confirmed by the authors. The calculator should present loss predictions as estimates, not ground truth, and note reduced accuracy at D/N ratios far from 20x.

In practice, many teams deliberately over-train on tokens to improve inference efficiency (smaller model, more data). LLaMA 3 trained 8B on 15T tokens (≈ 1875× Chinchilla ratio). The calculator should show the Chinchilla ratio: `D / (20 × Ψ)`.

**Practical minimum**: Regardless of Chinchilla optimality, models trained on fewer than ~200B tokens tend to produce poor results. The calculator should warn when D < 200B tokens, even if the Chinchilla ratio is satisfied (e.g., a small model where 20x Psi < 200B).

#### Inference-Aware Compute-Optimal Scaling

Standard Chinchilla minimizes loss for a given training compute budget. Sardana et al. (2024, arXiv:2401.00448, ICML 2024) formalize the more practical question: minimize **total lifetime cost** (training + inference) for a target loss level. The key insight is that inference cost scales with N (model size) but not D_tr (training tokens), so the optimal model is smaller and trained longer than Chinchilla predicts.

**Core formula (FLOP-based objective):**
```
N*, D_tr* = argmin_{N, D_tr | L(N, D_tr) = l}  [6*N*D_tr + 2*N*D_inf]
```
Where `6*N*D_tr` is training FLOPs and `2*N*D_inf` is total lifetime inference FLOPs (forward-pass only, summed over all inference requests). When D_inf = 0, this reduces to standard Chinchilla. When D_inf > 0, the optimum shifts to smaller N and larger D_tr; there is no closed-form solution and it must be solved numerically.

**Practical guidance -- inference-optimal vs Chinchilla-optimal sizes:**

| Target Loss | Chinchilla N | Inference-Optimal N | N Ratio | Training Tokens Multiplier | Total Cost Savings |
|---|---|---|---|---|---|
| 2.53 | 1B | 327M | 33% | 5.5x | 50% |
| 2.13 | 7B | 2.90B | 41% | 3.4x | 34% |
| 1.96 | 30B | 8.58B | 29% | 7.8x | 58% |
| 1.89 | 70B | 21.5B | 31% | 6.3x | 54% |

The pattern is consistent: inference-optimal models are **30-40% of Chinchilla size**, trained on **3-8x more tokens**, yielding **34-58% total cost savings** over the model's lifetime. This is the formal justification for the overtraining trend visible in the Token-to-Parameter Ratio Reference table below. The calculator should display this table as advisory context when the user's D/N ratio significantly exceeds 20x, confirming that deliberate overtraining is a rational cost optimization, not a departure from scaling laws.

#### Independent Replication

Epoch AI (2024, arXiv:2404.10102) independently replicated the Chinchilla result, finding a compute-optimal ratio of 25.6:1 -- slightly above 20:1 and consistent with the power-law fit in this spec (which predicts the ratio increases with scale above 1B parameters). This confirms the Chinchilla coefficients remain the best available baseline.

#### Token-to-Parameter Ratio Reference

The compute-optimal D/N ratio has shifted dramatically over time as the field learned to overtrain smaller models for inference efficiency. The calculator should display where the user's chosen ratio falls in this landscape:

| Year | Model / Paper | D/N Ratio | Notes |
|------|--------------|-----------|-------|
| 2020 | Kaplan / GPT-3 | 1.7:1 | Undertrained by modern standards |
| 2022 | Chinchilla | 20:1 | Compute-optimal baseline |
| 2024 | Epoch AI replication | 25.6:1 | Independent confirmation |
| 2024 | DeepSeek | 30:1 | Data-quality dependent |
| 2024 | LLaMA 3 (8B) | 1,875:1 | Deliberate overtraining for inference |
| 2025 | Qwen3-0.6B | 60,000:1 | 0.6B params on 36T tokens; current extreme |

The range from 20:1 (compute-optimal) to 1,875:1+ (inference-optimized) is not a sign of disagreement -- it reflects different optimization targets. The calculator should show the Chinchilla-optimal ratio as the default recommendation and clearly label departures as deliberate overtraining.

#### MoE-Specific Scaling Note

For MoE models (Section 3.4), the optimal tokens-to-active-parameters ratio behaves differently than for dense models. Per the Warsaw MoE paper (2024, arXiv:2402.07871), the ratio **decreases** with scale for MoE (opposite of dense models): from ~44:1 at 6.4B active parameters down to ~8:1 at 64T active parameters. The calculator should note this when the user configures an MoE architecture: the standard Chinchilla 20:1 rule (calibrated on dense models) may overestimate the optimal token count for large MoE models.

#### Historical Context: Kaplan vs Chinchilla

The Chinchilla scaling law superseded the earlier Kaplan et al. (2020) scaling law, which recommended a different compute-optimal allocation. Kaplan found `N_opt proportional to C^0.73`, meaning most additional compute should go to model size with relatively little to training data (train large, train short). Chinchilla found `N_opt proportional to C^0.50`, meaning compute should be split roughly equally between model size and data (the "20x rule"). The practical difference is significant: at a given compute budget, Kaplan would recommend a larger model trained on fewer tokens, while Chinchilla recommends a smaller model trained on more tokens. The Chinchilla result is now the accepted standard because it was validated on a much larger set of training runs (400+ vs Kaplan's narrower range) and correctly predicted that models like GPT-3 175B were significantly undertrained on data. The calculator uses Chinchilla.

### 4.4 Critical Batch Size

The **critical batch size** B_crit is the batch size (in tokens) at which training is equally efficient in terms of compute and time. It was introduced by McCandlish et al. (2018) and quantified for language models by Kaplan et al. (2020).

```
B_crit(L) = B_star / L^(1/alpha_B)
```

Where L is the training loss and the fitted coefficients are:
```
Kaplan et al.: B_star = 2.0 × 10^8 tokens, alpha_B = 0.21 (exponent 1/alpha_B ≈ 4.76)
```

**What it means practically:**
- **B < B_crit**: Training is *time-inefficient* -- you could train faster with larger batches without meaningful compute waste. The gradient signal-to-noise ratio is low, so each step's update is noisy relative to its cost.
- **B > B_crit**: Training is *compute-inefficient* -- larger batches give diminishing returns. You are spending more FLOPs per unit of loss reduction than necessary.
- **B ≈ B_crit**: The sweet spot -- near-optimal trade-off between wall-clock time and total compute.

The critical batch size grows as loss decreases (i.e., as training progresses or as models get better). For a well-trained large model at low loss, B_crit can be in the millions of tokens. For example, at loss L=3.0, B_crit ≈ 60K tokens; at L=2.0, B_crit ≈ 2M tokens.

**Compute efficiency at non-optimal batch size** (Kaplan et al., 2020): When training at batch size B != B_crit, the minimum compute C_min that would achieve the same result at the optimal batch size is:
```
C_min = C / (1 + B / B_crit(L))
```
The dual formula for minimum training steps:
```
S_min = S / (1 + B_crit(L) / B)
```
At B = B_crit, C_min = C/2 (the theoretical minimum overhead). The **compute overhead percentage** is `B / (B + B_crit) * 100%`. At B = 10 * B_crit, the overhead is ~91% -- the user is spending 11x more compute than necessary. At B = B_crit/10, overhead is only ~9% but training takes ~11x longer than it needs to.

**Calculator use**: Given the user's chosen global batch size B (in tokens: `b × s × G × N_dp`) and the predicted training loss from Section 4.3, the calculator should display: (1) whether B is above or below B_crit, and (2) the compute overhead percentage from the C_min formula above. For example: "Your batch size of 4M tokens is 2x B_crit -- you are using ~67% more compute than the theoretical minimum, but training ~33% faster than at B_crit." This is advisory only -- many practical constraints (memory, hardware utilization, training stability) override the theoretical optimum.

### 4.5 Data-Constrained Scaling (Data Repetition)

When unique training data is limited, teams often train for multiple epochs (repeating data). Muennighoff et al. (2023, "Scaling Data-Constrained Language Models") ran 400 training experiments and found that data repetition has sharply diminishing returns.

#### Key Concepts

```
U = unique training tokens (the actual dataset size)
D = total training tokens (including repeats)
Epochs = D / U
```

When the user specifies both D and U (or equivalently, D and epochs), the calculator can assess data efficiency.

#### Diminishing Returns from Repetition

The value of repeated data follows an exponential saturation curve. The practical thresholds are:

| Epochs (D/U) | Data Utilization | Recommendation |
|---|---|---|
| 1 (no repeats) | 100% efficient | Ideal |
| Up to ~4 | Near-full value | Acceptable -- repetition is almost as good as unique data |
| 4-40 | Rapidly diminishing | Warning -- significant compute waste on repeated data |
| Beyond ~40 | Essentially zero marginal value | Strong warning -- additional epochs provide no meaningful loss improvement |

**Maximum effective data**: Regardless of how many times data is repeated, the effective contribution saturates at approximately **16x the unique token count**. Training on D = 100 x U yields roughly the same loss as D = 16 x U. This means there is an absolute ceiling on how much a limited dataset can be "stretched" through repetition.

#### Calculator Use

The calculator should:
1. Accept an optional "unique tokens U" input (defaults to D, meaning no repetition)
2. Compute epochs = D / U
3. Display a **data repetition warning** when epochs > 4, escalating at epochs > 40
4. When epochs > 16, note that the user is past the effective data ceiling and additional training tokens provide negligible benefit
5. For the Chinchilla loss prediction (Section 4.3), note that it assumes unique data -- predictions become unreliable when D >> U because repeated tokens contribute less than fresh tokens to loss reduction

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

**FP16 vs BF16 precision note**: FP16 training requires **loss scaling** to prevent gradient underflow, because FP16 has only ~40 powers of 2 in dynamic range (vs FP32's ~264). PyTorch's `GradScaler` handles this automatically via dynamic loss scaling, but this may cause occasional skipped optimizer steps when the scale is too high. **BF16 has the same exponent range as FP32**, so loss scaling is not needed — BF16 is the preferred half-precision format on Ampere+ GPUs. This distinction does not affect memory formulas but is important advisory information: the calculator should note when the user selects FP16 that loss scaling is required and BF16 is preferred if supported by their hardware.

**AMP autocast vs explicit bf16 mode**: The table above describes **explicit bf16 mode** (used by Megatron-LM, DeepSpeed, and FSDP with mixed precision), where parameters are stored in bf16 with a separate fp32 master copy, yielding 16-18 bytes/param. A different memory model exists: **standard PyTorch AMP** (`torch.amp.autocast`) keeps all parameters in fp32 (4 bytes each) and casts activations to bf16/fp16 on-the-fly inside each operation. In AMP autocast mode, no separate master weights are needed (the fp32 parameters ARE the master weights), so the cost is 4 (param=master) + 4 (grad, fp32) + 8 (Adam m+v) = **16 bytes/param** -- or 14 bytes/param with bf16 gradients. Memory savings in AMP autocast come entirely from reduced activation memory, not from parameter storage. This calculator assumes **explicit bf16 mode** because it is standard for the large-scale distributed training scenarios (Megatron-LM, DeepSpeed, FSDP) the calculator targets. Users doing single-GPU or HuggingFace Trainer runs with `torch.amp.autocast` should note that their parameter memory will be 4 bytes/param (fp32) rather than 2 bytes/param (bf16), but their total model states will be comparable (14-16 vs 16-18 bytes/param).

Let Φ = total bytes per parameter = 2 + β_grad + 12 (for AdamW mixed precision), so Φ = 18 (fp32 grads) or 16 (bf16 grads).

So: **M_model_states = ΦΨ bytes** (mixed precision AdamW, Φ = 18 default)

**Other optimizers:**

| Optimizer | Bytes/Param | Breakdown |
|-----------|-------------|-----------|
| AdamW fp32 | 16 | 4 (param) + 4 (grad) + 4 (m) + 4 (v) |
| AdamW mixed (fp32 grads) | 18 | 2+4+4+4+4 = 18 |
| AdamW mixed (bf16 grads) | 16 | 2+2+4+4+4 = 16 |
| AdamW mixed (no master weights) | 12 | 2+2+4+4 = 12 (update bf16 params directly; used by llm.c) |
| AdamW FP8 mixed precision | 14 | 1+1+4+4+4 = 14 |
| AdamW + 8-bit states | 12 | 2+2+4+2+2 = 12 |
| SGD + momentum (mixed) | 12 | 2+2+4+4 = 12 |
| SGD (no momentum, mixed) | 8 | 2+2+4 = 8 |
| Adafactor | 12 | 2+2+4+4 (row+col factors instead of full m,v) |
| Lion (mixed) | 12 | 2+2+4+4 (momentum only, no variance term) |
| Adam-mini (mixed) | 10 | 2+2+4+~2 (block-diagonal Hessian reduces momentum to ~2 bytes/param) |
| LAMB (mixed) | 16-18 | Same as AdamW (m + v + master weights); used for large-batch pretraining |
| MeZO (zeroth-order) | 2 | 2+0+0 (forward-pass only; no gradients or optimizer states stored) |

**Adam-mini note**: Adam-mini (Zhang et al., 2024) exploits the block-diagonal structure of the Hessian in transformers to use a single learning rate per parameter block instead of per-parameter second moments. This reduces optimizer state memory by ~45-50% compared to AdamW while maintaining comparable training quality. It is production-ready and a good default recommendation when memory is tight but AdamW-level convergence is desired.

**MeZO note**: MeZO (Malladi et al., 2023) is a zeroth-order optimizer that estimates gradients via forward-pass perturbation, eliminating all gradient and optimizer state storage. At only 2 bytes/param (the model weights in bf16), it enables fine-tuning models ~10x larger than standard optimizers on the same hardware (e.g., fine-tuning a 30B model on a single A100 vs. ~3B with AdamW). However, MeZO is **fine-tuning only** -- it is not suitable for pretraining due to slow convergence. The calculator should offer MeZO only in the post-training section (Section 10) and grey it out for pretraining.

**FP8 training note**: The 14 bytes/param row above assumes parameters and gradients are explicitly stored in fp8 format (1 byte each), as with Microsoft's MS-AMP backend. However, the most common FP8 implementation -- NVIDIA TransformerEngine in its native mode -- does **not** reduce memory: the model remains in bf16/fp32 in memory, and FP8 is used only inside compute kernels (matmuls). In this mode, memory consumption is identical to bf16 mixed precision (16-18 bytes/param). Only specialized backends like MS-AMP that actually store weight and gradient tensors in fp8 achieve the 14 bytes/param figure. The calculator should default FP8 to **no memory savings** (same as bf16 mixed precision) and offer an "FP8 weight storage" toggle for the 14 bytes/param mode. The primary benefit of FP8 is compute throughput (2x FLOPS on supported hardware), not memory reduction.

**Checkpoint (storage) size**: Training checkpoints saved to disk contain fp32 master weights + Adam m + Adam v (gradients are not saved). For AdamW:
```
Checkpoint size = 12 × Ψ bytes  (4 + 4 + 4 per parameter)
```
This is distinct from live training memory (16-18 bytes/param) because gradients are recomputed on resume. PyTorch checkpoint files include metadata overhead of ~3-5% above the theoretical size. The calculator should display checkpoint size as an output for storage planning (e.g., LLaMA 7B checkpoint = 12 x 6.7B = 80.4 GB per save).

### 5.2 ZeRO Partitioning

ZeRO (Rajbhandari et al., 2020) shards model states across N_dp GPUs. The formulas below use Φ from Section 5.1 (18 with fp32 grads, 16 with bf16 grads).

Let **K_opt** = optimizer state bytes per parameter (the portion sharded in ZeRO-1). K_opt = Φ - 2 - β_grad. For AdamW mixed precision: K_opt = 12 (master weights + m + v). For SGD+momentum mixed: K_opt = 8. For SGD no momentum: K_opt = 4. This ensures ZeRO formulas auto-adapt to any optimizer from the table in Section 5.1.

| Stage | Memory per GPU | What's Sharded |
|-------|---------------|----------------|
| ZeRO-0 | ΦΨ | Nothing |
| ZeRO-1 | (2 + β_grad)Ψ + K_opt·Ψ/N_dp | Optimizer states |
| ZeRO-2 | 2Ψ + (β_grad + K_opt)·Ψ/N_dp | Optimizer states + gradients |
| ZeRO-3 | ΦΨ/N_dp | Everything (params + grads + optimizer) |

With AdamW fp32 grads (Φ=18, K_opt=12): ZeRO-0 = 18Ψ, ZeRO-1 = 6Ψ + 12Ψ/N_dp, ZeRO-2 = 2Ψ + 16Ψ/N_dp, ZeRO-3 = 18Ψ/N_dp.
With AdamW bf16 grads (Φ=16, K_opt=12): ZeRO-0 = 16Ψ, ZeRO-1 = 4Ψ + 12Ψ/N_dp, ZeRO-2 = 2Ψ + 14Ψ/N_dp, ZeRO-3 = 16Ψ/N_dp.
With SGD+momentum mixed (Φ=12, K_opt=8): ZeRO-0 = 12Ψ, ZeRO-1 = 4Ψ + 8Ψ/N_dp, ZeRO-2 = 2Ψ + 10Ψ/N_dp, ZeRO-3 = 12Ψ/N_dp.

**DeepSpeed gradient upcasting note**: DeepSpeed's FusedAdam upcasts all gradients from fp16 to fp32 during the optimizer step, meaning both copies coexist briefly. This adds 2 bytes/param to the sharded portion in ZeRO-2, making it `2Ψ + 18Ψ/N_dp` instead of the theoretical `2Ψ + 16Ψ/N_dp`. The formulas above follow the ZeRO paper's accounting (one gradient copy). For DeepSpeed-specific estimates, add 2Ψ/N_dp to the ZeRO-2 formula. This transient overhead does not affect ZeRO-3 (which shards everything uniformly) or ZeRO-1 (gradients are unsharded).

**ZeRO communication volume per training step** (Rajbhandari et al., 2020, Section 4): The communication cost of each ZeRO stage relative to standard data-parallel all-reduce (~2Psi bytes):
```
Baseline DP (all-reduce):        reduce-scatter: Ψ  +  all-gather: Ψ       = ~2Ψ total
ZeRO-1 (shard optimizer):       reduce-scatter: Ψ  +  all-gather: Ψ       = ~2Ψ total (same as baseline)
ZeRO-2 (shard optimizer+grads): reduce-scatter: Ψ  +  all-gather: Ψ       = ~2Ψ total (same as baseline)
ZeRO-3 (shard everything):      reduce-scatter: Ψ  +  all-gather: Ψ (fwd) + all-gather: Ψ (bwd) = ~3Ψ total (1.5x baseline)
```
ZeRO-1 and ZeRO-2 have **identical** communication volume to standard data parallelism -- the memory savings come from redistributing what is stored, not from reducing communication. ZeRO-3 adds two extra all-gather operations (one in forward, one in backward) to reconstruct full parameters on each GPU, increasing total volume by 50%.

**Parameter divisibility**: ZeRO requires the total parameter count to be evenly divisible by N_dp for clean sharding. Some frameworks (e.g., llm.c) silently disable ZeRO if `Ψ % N_dp != 0`; others pad parameters automatically. The calculator should warn when the parameter count is not evenly divisible by the data parallel degree.

**FSDP-to-ZeRO equivalence**: PyTorch FSDP (Fully Sharded Data Parallel) implements the same sharding strategies as DeepSpeed ZeRO under different names. The calculator should accept either terminology:

| FSDP Strategy | ZeRO Equivalent | What is Sharded |
|---|---|---|
| NO_SHARD | ZeRO-0 (DDP) | Nothing |
| SHARD_GRAD_OP | ZeRO Stage 2 | Optimizer states + gradients |
| FULL_SHARD | ZeRO Stage 3 | Optimizer states + gradients + parameters |
| HYBRID_SHARD | ZeRO++ Stage 3 | Everything within node; replicated across nodes |
| HYBRID_SHARD_ZERO2 | ZeRO++ Stage 2 | Optimizer + gradients within node; replicated across nodes |

**FSDP mixed precision memory model**: FSDP handles mixed precision differently from standard AMP. In standard AMP, both the fp32 master weights and the low-precision working copy coexist in GPU memory, costing `(K_full + K_low) x Psi` bytes. FSDP instead keeps local shards at full precision and materializes the unsharded (all-gathered) parameters transiently in low precision:
```
M_fsdp_mixed = K_full x (Psi / F) + K_low x max(Psi_fsdp_unit)
```
Where `F` is the sharding factor (number of FSDP ranks, typically `N_dp`), `K_full` is bytes per parameter at full precision (e.g., 4 for fp32), `K_low` is bytes at reduced precision (e.g., 2 for bf16), and `max(Psi_fsdp_unit)` is the parameter count of the largest FSDP wrapping unit (see "FSDP wrapping granularity" note in Section 5.4). This saves memory vs standard AMP because only one FSDP unit's worth of full parameters is ever materialized in low precision at a time, rather than the entire model. The calculator should use this formula when FSDP + mixed precision is selected, instead of the standard AMP memory model.

**HYBRID_SHARD (intra-node sharding)**: HYBRID_SHARD shards model states within each node (across `N_dp_intra = GPUs_per_node`, typically 8) but replicates full model states across nodes. This reduces inter-node communication at the cost of less memory savings than full ZeRO-3:
```
HYBRID_SHARD (ZeRO++ Stage 3):   M_per_gpu = ΦΨ / N_dp_intra
HYBRID_SHARD_ZERO2 (ZeRO++ Stage 2): M_per_gpu = 2Ψ + (β_grad + K_opt)·Ψ / N_dp_intra
```
Where `N_dp_intra = GPUs_per_node` (typically 8). For example, a 7B model with HYBRID_SHARD on 8-GPU nodes: `18 × 7B / 8 = 15.75 GB` per GPU for model states (vs. `18 × 7B / 64 = 1.97 GB` with full ZeRO-3 across 64 GPUs). HYBRID_SHARD is preferred for multi-node training with slow inter-node interconnect because it dramatically reduces cross-node traffic. The cross-host communication volume per GPU for HYBRID_SHARD is:
```
Cross-host traffic per GPU = 2M x (W - 1) / (G x W)
```
Where `M` = model bytes, `W` = world size (total GPUs), `G` = GPUs per host. This is `G` times less cross-host traffic than DDP's `2M x (W-1)/W`, since intra-node all-reduce stays on NVLink and only the inter-node reduction crosses the network.

**DeepSpeed initialization memory spike**: DeepSpeed creates flat fp32 parameter buffers during model preparation (before sharding), causing a transient peak of `4 × Ψ / N_dp` bytes above steady-state. This spike occurs only during initialization and is released once sharding completes. PyTorch FSDP without mixed precision can operate entirely in bf16 (no fp32 upcast), avoiding this spike. The calculator should note this transient cost for DeepSpeed users.

**CPU memory initialization constraint**: Standard model initialization materializes the full model in fp32 on CPU before distributing to GPUs. This requires `4 × Ψ` bytes of CPU memory per node (e.g., a 405B model needs ~1.6 TB CPU RAM). When `4 × Ψ > CPU_memory_per_node`, standard initialization fails. The workaround is partitioned initialization (ZeRO-Infinity's `remote_device="cpu"` or PyTorch's `device="meta"` with deferred materialization), which initializes parameters shard-by-shard. The calculator should warn when `4 × Ψ` exceeds 80% of typical CPU memory per node (e.g., 1 TB for standard DGX nodes, 2 TB for high-memory nodes).

**CPU and NVMe offloading** (ZeRO-Offload / ZeRO-Infinity): ZeRO can offload model states to CPU memory or NVMe storage, trading throughput for reduced GPU memory. Offload availability varies by ZeRO stage:

| Offload Type | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|
| CPU offload (optimizer states) | Yes | Yes | Yes |
| CPU offload (parameters) | No | No | Yes |
| NVMe offload | No | No | Yes |
| Partial offload (ratio 0.0-1.0) | No | No | Yes |

The memory impact by offload configuration:

| Offload Target | GPU Memory | CPU Memory | Throughput Impact |
|---|---|---|---|
| None | Full model states | Minimal | Fastest |
| CPU (optimizer only) | Params + grads: `(2 + β_grad)Ψ` | Optimizer states: `K_opt·Ψ` | Moderate slowdown |
| CPU (optimizer + params, ZeRO-3 only) | Minimal (working buffers only) | Full model states: `ΦΨ` | Significant slowdown |
| NVMe (optimizer only, ZeRO-3 only) | Params + grads: `(2 + β_grad)Ψ` | Buffer only | Slow |
| NVMe (all, ZeRO-3 only) | Minimal (working buffers only) | Buffer only | Slowest |

The calculator should include CPU offloading as an option that modifies the ZeRO memory formulas: when CPU offload is enabled for optimizer states, subtract `K_opt·Ψ/N_dp` from per-GPU memory (ZeRO-2) or subtract the optimizer portion of `ΦΨ/N_dp` (ZeRO-3). Parameter and NVMe offloading are only available with ZeRO-3 -- the calculator should enforce this constraint and not offer these options for ZeRO-1 or ZeRO-2. NVMe offloading further reduces CPU memory requirements but is rarely used in practice due to extreme throughput penalties. The calculator should display a throughput warning when any offloading is enabled.

**Offload throughput efficiency formula** (Rajbhandari et al., 2021, ZeRO-Infinity): The throughput degradation from offloading can be estimated from the arithmetic intensity of the offloaded component relative to PCIe bandwidth and GPU compute:
```
offload_efficiency = (AIT × bw_pcie) / (AIT × bw_pcie + F_peak)
```
Where `AIT` is the arithmetic intensity (FLOPs per byte transferred) of the offloaded component, `bw_pcie` is PCIe bandwidth per GPU (typically 12-32 GB/s), and `F_peak` is peak GPU TFLOPS. The AIT values for each offloadable component:
```
Parameters + gradients:  AIT = s × b          (each byte participates in s×b FLOPs)
Optimizer states:        AIT = s × b / 4      (updated once per 4 micro-steps on average)
```
Example: V100 (70 TFLOPS), PCIe Gen3 (12 GB/s), s=1024, b=4, optimizer offload:
`AIT_opt = 1024×4/4 = 1024 FLOPS/byte`. `efficiency = (1024 × 12e9) / (1024 × 12e9 + 70e12) = 12.3e12 / 82.3e12 ≈ 15%`. Single-GPU offloading is heavily bottlenecked by PCIe bandwidth. However, offload efficiency improves with larger batch×sequence (higher AIT), faster PCIe (Gen4/Gen5), and multi-GPU setups where aggregate PCIe bandwidth scales with GPU count. For example, 8 GPUs with aggregate 96 GB/s: efficiency rises to ~58%. The calculator should display this efficiency estimate when offloading is enabled to set realistic throughput expectations.

**MoE + ZeRO interaction**: When combining ZeRO with Expert Parallelism, expert (MLP) parameters and non-expert (attention, layernorm) parameters use different sharding denominators because EP already distributes experts across GPUs:
```
Non-expert params: sharded across N_dp GPUs (standard ZeRO)
Expert params:     sharded across N_dp / N_ep GPUs (fewer GPUs share each expert)
```
For example, with N_dp=64 and N_ep=8, attention weights are sharded 64-way but each expert's MLP is sharded only 8-way. The calculator should apply ZeRO formulas separately to expert and non-expert parameter groups when both EP and ZeRO are active.

### 5.3 Activation Memory

Activations are intermediate values stored during forward pass for use in backward pass.

**Per transformer layer** (Korthikanti et al., 2022):

No checkpointing (store everything):
```
M_act_layer = s × b × d × (34 + 5 × a × s / d) bytes
```
The constant 34 = 11 (attention: Q,K,V,softmax output, attention dropout, attention output projection, two layernorm inputs/outputs) + 19 (MLP: up-projection input/output, down-projection input/output, activation function, dropout) + 4 (two LayerNorm: 2 norms x 2 tensors each). The `5as/d` term is the attention score matrix (s x s per head, stored for Q*K^T, softmax, and dropout).

Full activation checkpointing (recompute each layer):
```
M_act_layer = 2 × s × b × d bytes  (store only layer input)
```
Cost: ~33% more compute (recompute forward during backward)

**Transient recomputation working memory**: The `2 × s × b × d` figure above is the *stored* checkpoint memory. During the backward pass, when a checkpointed layer is recomputed, its full activations must be temporarily materialized in GPU memory. This transient working memory equals one layer's full (non-checkpointed) activation memory and cannot be offloaded:
```
M_recomp_working = s × b × d × (34 + 5 × a × s / d) bytes   (per-layer checkpointing, ci=1)
```
For checkpoint intervals spanning multiple layers (ci > 1, i.e., checkpointing every ci-th layer), the working memory scales with ci since all intermediate layers must be recomputed:
```
M_recomp_working = ci × s × b × d × (34 + 5 × a × s / d) bytes
```
This working memory is transient (freed after each layer's backward completes) but sets a hard floor on per-GPU VRAM alongside the minimum GPU memory floor from Section 9. When Flash Attention is enabled, the `5as/d` term disappears from this formula as well. The calculator should include `M_recomp_working` (with ci=1) as part of the peak memory estimate when full activation checkpointing is selected.

Block-level partial recomputation (NeMo `recompute_method="block"` with `recompute_num_layers=N`): checkpoints the first N layers per pipeline stage fully, remaining layers store all activations:
```
M_activations_stage = N_recomp × (2 × s × b × d) + (L_per_stage - N_recomp) × M_act_full_layer
```
This is a practical intermediate that lets users recompute only as many layers as needed to fit in memory. The calculator should support this as a "partial" checkpointing option where the user specifies N_recomp.

Selective activation checkpointing:
```
M_act_layer = s × b × d × (10 + 24/N_tp + 5 × a × s / (d × N_tp)) bytes
```

**PyTorch AMP FP32 precision caveat**: The Korthikanti coefficients (34 for linear terms, 5 for attention) assume all activations are stored in the compute precision (bf16/fp16). Under PyTorch's `torch.cuda.amp.autocast`, two operations are promoted to FP32 for numerical stability: (1) **softmax** outputs are saved in FP32 (4 bytes instead of 2), adding an extra `b*a*s^2` bytes per layer (changing the attention coefficient from 5 to 6), and (2) **layer norm** inputs are saved in FP32, adding `2*s*b*d` bytes per layer (changing the linear coefficient from 34 to 36). The corrected formula under AMP autocast is:
```
M_act_layer = s × b × d × (36 + 6 × a × s / d) bytes  (PyTorch AMP autocast)
```
This was empirically validated against `torch.cuda.max_memory_allocated()` on GPT-2 small (A100), achieving 1.15% error (Rees, erees.dev). The difference is ~6% more activation memory than the Korthikanti formula predicts. Megatron-LM and frameworks that use explicit bf16 storage (not autocast) match the original coefficients (34, 5). The calculator should use the **Korthikanti coefficients (34, 5)** as the default (they match the widely-used Megatron-LM implementation) and offer an "AMP autocast" toggle that applies the corrected coefficients (36, 6) for users training with standard PyTorch AMP.

**d_ff correction for activation memory**: The constant `24` in the selective checkpointing and SP formulas assumes `d_ff = 4d` (standard FFN). Megatron-LM parameterizes this as `4 × (d_ff / d)`, making the FFN activation cost proportional to the actual intermediate dimension. For SwiGLU models where `d_ff != 4d`, replace `24` with `4 × d_ff / d` in these formulas. The calculator should use the actual `d_ff` value when available (Detailed/Preset modes) and fall back to `24` only in Quick Mode where `d_ff` is estimated.

With Flash Attention (avoids materializing s×s attention matrix):
```
M_act_layer = s × b × d × (10 + 24/N_tp) bytes  (the 5as/d term disappears)
```
Flash Attention still stores O(s) per-head log-sum-exp statistics for the backward pass. The precise replacement for the `5as/d` term is `4 × a × s × b / N_tp` bytes (per head: one float for row-max and one for log-sum-exp, times sequence length). For typical hidden dimensions this is negligible compared to the linear activation terms, but it grows with head count and sequence length.

**Flash Attention + selective checkpointing interaction**: Selective checkpointing saves activations needed for the attention and MLP forward passes but recomputes the attention score matrix. When Flash Attention is already enabled, the O(s^2) attention activations are never materialized in the first place, so the `5as/d` term is already eliminated. The selective checkpointing formula with Flash Attention collapses to the same `s*b*d*(10 + 24/N_tp)` as the Flash Attention formula above. In practice, selective checkpointing provides additional memory savings over Flash Attention only when Flash Attention is NOT used. The calculator should recognize this overlap: when both Flash Attention and selective checkpointing are enabled, use the Flash Attention formula rather than double-counting the savings.

**CPU activation offloading**: A third option beyond "store on GPU" and "recompute" is offloading activation tensors to CPU memory via PCIe. Both NeMo (`cpu_offloading_activations=True`) and PyTorch (`CheckpointPolicy.MUST_CPU_OFFLOAD`) support this. CPU offloading trades PCIe bandwidth for GPU memory, and is most effective when the recomputation cost exceeds the transfer time. The throughput overhead decreases with model hidden dimension because larger layers have higher arithmetic intensity, better overlapping transfer with compute (Rajbhandari et al., 2021): d=2K: ~25% slowdown, d=8K: ~10% slowdown, d=32K+: <2% slowdown. The calculator should note this option exists but treat it as an advanced toggle rather than a primary mode.

**Sequence parallelism** (Korthikanti et al., 2022): When used with tensor parallelism (N_sp = N_tp in Megatron-LM), LayerNorm and dropout activations are partitioned along the sequence dimension. This reduces the `10 × s × b × d` term (which covers LayerNorm inputs/outputs and dropout masks outside the TP-sharded regions) to `10 × s × b × d / N_tp`. The selective checkpointing formula with sequence parallelism becomes:
```
M_act_layer = s × b × d × (10/N_tp + 24/N_tp + 5 × a × s / (d × N_tp)) bytes
            = s × b × d × (34/N_tp + 5 × a × s / (d × N_tp)) bytes
```
Sequence parallelism is standard practice in Megatron-LM when TP is used and should be assumed enabled whenever N_tp > 1.

**Context parallelism** (Meta, Llama 3 scaling): Context parallelism (CP) shards the input sequence along the sequence dimension across N_cp GPUs. Each CP rank processes `s/N_cp` tokens. In all activation memory formulas above, replace `s` with `s/N_cp` when CP is active. The quadratic attention term benefits most because the attention score matrix is O(s^2):
```
M_act_layer = (s/N_cp) × b × d × (34 + 5 × a × (s/N_cp) / d) bytes  [no checkpointing, with CP]
```
With Flash Attention the `5a(s/N_cp)/d` term disappears as usual. CP communication cost is an all-gather of KV tensors per layer (forward) and reduce-scatter of KV gradients (backward). Because communication is O(s) while attention compute is O(s^2), CP overhead shrinks with longer sequences, making it most effective at 32K+ sequence lengths. When to use CP: when sequence length causes activation memory pressure and the micro-batch size would otherwise drop to 1. The trigger heuristic is `GBS / (N_gpu / (N_tp × N_pp)) <= 1` at long sequence lengths, indicating DP alone cannot maintain throughput. CP should only be enabled when `s/N_cp` still exceeds a minimum chunk size (~2K tokens) to maintain sufficient arithmetic intensity per rank.

**Total activation memory:**
```
M_activations = L_active × M_act_layer / N_pp
```
Where L_active = layers assigned to this pipeline stage = L / N_pp

**Note**: With gradient accumulation G steps, activation memory is for ONE micro-batch, not the full global batch.

**Output logits tensor**: The per-layer formulas above (Korthikanti et al.) cover only transformer layer activations. During loss computation, the full output logits tensor is materialized as a non-layer activation:
```
M_output_logits = b × s × V × β bytes
```
For large vocabularies this can exceed per-layer activation memory. Examples: LLaMA 7B (V=32K, b=4, s=2048, bf16) = ~0.5 GB; LLaMA 3 8B (V=128K, same config) = ~2 GB. The calculator should add this to the total activation memory. Note that **chunked cross-entropy loss** (used by Liger Kernel, HuggingFace, and others) avoids materializing the full logits tensor by fusing the projection and loss computation in vocabulary-sized chunks, effectively eliminating this cost. The calculator should include an option to disable this component when fused/chunked loss is enabled.

### 5.4 Temporary Buffers & Communication

Rough estimate (use as fallback):
```
M_communication ≈ 0.05 × (M_model_states + M_activations)  (5% overhead)
```

More precisely, allocate concrete buffer sizes used by DeepSpeed/Megatron:
- **DP all-reduce buffer**: ~2Ψ × β bytes (ring all-reduce double-buffer)
- **ZeRO allgather bucket**: 500M elements x β bytes (~1 GB in bf16, ~2 GB in fp32)
- **ZeRO-3 parameter prefetch**: During forward/backward, ZeRO-3 must allgather the full (unsharded) parameters of the current layer. The prefetch buffer holds one full transformer layer's unsharded weights:
  ```
  M_prefetch_fwd = max(Ψ_embedding, Ψ_largest_layer) × β
  M_prefetch_bwd ≈ 2 × Ψ_largest_layer × β  (current + next prefetched layer)
  ```
  For example, LLaMA 70B has ~1.1B params/layer, so the prefetch buffer is ~2.2 GB in bf16 per layer during forward, ~4.4 GB during backward.
- **FSDP AllGather rate limiter**: FSDP limits concurrent AllGather operations to at most 2 in flight at any time to prevent CUDA allocator over-allocation (without this limit, T5-11B sees up to 5x slowdown from `cudaMalloc` retries). This means the peak AllGather buffer memory is bounded by:
  ```
  M_allgather_peak = 2 x max(Psi_fsdp_unit) x K bytes
  ```
  Where `max(Psi_fsdp_unit)` is the largest FSDP wrapping unit's parameter count and `K` is bytes per parameter at the materialized precision. The calculator should use this as the AllGather buffer contribution for FSDP, replacing the backward prefetch formula above (which assumes 2 layers; the rate limiter is the actual constraint).
- **FSDP wrapping granularity**: The "largest layer" term in the prefetch buffer formulas is more precisely the largest FSDP wrapping unit, which users control. Wrapping at the transformer block level (default) makes each block one unit. Finer wrapping (e.g., per-attention/per-FFN) reduces the prefetch buffer and peak memory but increases the number of AllGather operations, reducing throughput. The calculator should default to per-transformer-block wrapping (i.e., `max(Psi_fsdp_unit) = Psi_largest_layer`) but note that users can trade throughput for memory by wrapping more finely.
- **Peak logit memory during loss backward**: At the start of the backward pass through the loss function, a new FP32 gradient tensor for the logits is allocated while the forward-pass logits (stored in mixed precision) still reside in memory. Both coexist briefly, creating a peak:
  ```
  M_logits_peak = M_output_logits + 4 × b × s × V / N_tp
  ```
  Where `M_output_logits` is from Section 5.3 (the forward logits, typically in bf16+fp32 = ~6 bytes/element under AMP) and the second term is the FP32 gradient tensor. For GPT-2 small (V=50,304, b=12, s=1024): the peak logit allocation alone is ~2.5 GB. For large vocabularies (V=128K+) this is the dominant temporary buffer. The calculator should use `M_logits_peak` (not just `M_output_logits`) when computing peak memory, and note that chunked cross-entropy loss (Section 5.3) eliminates this spike entirely.
- **TP all-reduce**: small, within-layer activations
- **PP send/receive**: s × b × d × β per stage boundary

For ZeRO-2/3 workloads, communication buffer memory depends on bucket sizes. When `overlap_comm = true` is enabled, DeepSpeed allocates:
```
M_overlap_comm = 4.5 × (allgather_bucket_size + reduce_bucket_size) × β
```
**Important**: `overlap_comm` defaults to `False` for ZeRO stages 0-2 and `True` only for ZeRO stage 3 (per DeepSpeed's config validator). The overlap buffer cost only applies when `overlap_comm` is explicitly enabled or when using ZeRO-3 (where it is on by default). Raw DeepSpeed default bucket sizes are 5x10^8 elements each, giving `4.5 x 10^9 x 2 = ~9 GB` in bf16. However, when using **HuggingFace Trainer** (the most common DeepSpeed integration path), bucket sizes are auto-calculated from the model's hidden dimension:
```
reduce_bucket_size   = hidden_size^2
prefetch_bucket_size = 0.9 × hidden_size^2
param_persistence_threshold = 10 × hidden_size
```
For hidden_size=4096: reduce_bucket = 16.7M elements (~32 MB in bf16) vs the raw default of 500M elements (~1 GB in bf16). With HF Trainer auto-config, `M_overlap_comm` drops from ~9 GB to ~0.6 GB. The calculator should use `hidden_size^2` as the default bucket size (matching HF Trainer behavior) and allow users to override with raw DeepSpeed defaults if they are configuring DeepSpeed directly. For ZeRO-3, the prefetch buffer formula above (Section 5.4) provides the dominant communication cost. With large vocabularies (128K+), add M_logits_peak to these estimates.

**torch.compile overhead**: When using `torch.compile` (increasingly common in PyTorch training), the compiler creates additional graph representations and optimized kernel caches that persist in GPU memory. Estimate approximately **10% of model weights** (`0.1 × Ψ × β` bytes) as additional overhead. The calculator should include this as an optional toggle (off by default).

### 5.5 Total Memory per GPU

```
M_gpu = M_model_states(ZeRO) + M_activations + M_communication + M_framework_overhead
```

**Peak memory refinement**: The additive formula above is a conservative upper bound. In practice, activations and gradients do not fully coexist -- during the backward pass, activations are freed as gradients accumulate. A tighter peak estimate (validated by cli99/llm-analysis to within 5% of Megatron-LM measurements) models this as:
```
M_peak = M_weights + M_optimizer + max(M_activations, M_gradients)
       + max(M_prefetch, M_logits_peak) + M_framework_overhead
```
Where `M_prefetch` is the ZeRO-3 parameter gather buffer and `M_logits_peak` is the peak logit memory during loss backward (both from Section 5.4). The calculator should use the simpler additive formula as the default conservative estimate and may optionally display the tighter peak for advanced users.

Where M_framework_overhead ≈ 2-5 GB (CUDA context, framework buffers, memory allocator). Empirically, Megatron-DeepSpeed uses ~5 GB; lighter frameworks like bare PyTorch FSDP use ~2 GB.

**CUDA allocator alignment overhead**: PyTorch's caching memory allocator rounds every tensor allocation up to 2 MiB (2 x 1024^2 bytes) boundaries. The actual memory consumed by a tensor is:
```
actual_bytes = ceil(theoretical_bytes / 2_MiB) x 2_MiB
```
For large tensors (weight matrices), the waste is negligible. For small tensors (LayerNorm parameters, biases, per-head buffers), the overhead can exceed 100% of the theoretical size. In aggregate, this alignment waste adds **~3-5% to total GPU memory** beyond what the formulas predict. The calculator should apply a 1.04x multiplier to the final memory estimate (before comparing to GPU capacity) to account for allocator alignment. This is distinct from the 10% fragmentation buffer below, which covers runtime fragmentation; the alignment overhead is a deterministic rounding cost on every allocation.

**Usable GPU memory** = Total VRAM x 0.90 (leave 10% buffer for fragmentation). This 10% assumes the framework uses contiguous pre-allocated memory buffers for defragmentation (as DeepSpeed and Megatron-LM do). Without defragmentation, interleaving of short-lived tensors (discarded activations) and long-lived tensors (checkpointed activations, gradients) can cause OOM with 30%+ of VRAM technically free. For users training with vanilla PyTorch (no DeepSpeed/Megatron memory management), consider a 20% buffer instead.

### 5.6 Tensor Parallelism Effect on Memory

TP splits weight matrices across N_tp GPUs within a layer:
```
M_params_per_gpu ≈ Ψ_params / N_tp  (approximately — not all layers split perfectly)
M_optimizer_per_gpu ≈ proportional reduction
M_activations: reduced by factor in the 24/N_tp term
```

TP constraint: N_tp ≤ GPUs per node (typically 8) because it requires NVLink bandwidth.

**TP communication volume**: Each transformer layer requires exactly **4 all-reduce operations per training step** -- 2 in the forward pass (one after the attention output projection, one after the FFN second linear layer) and 2 corresponding all-reduces in the backward pass. Each all-reduce transfers `b × s × d × β` bytes. The total TP communication per layer per step is:
```
Comm_tp_per_layer = 4 × 2 × (N_tp - 1) / N_tp × b × s × d × β
```
(The factor `2 × (N_tp - 1) / N_tp` is the ring all-reduce cost per operation.) For the full model: `Comm_tp_total = L / N_pp × Comm_tp_per_layer`. This grows linearly with sequence length and batch size, making TP communication-bound for very long sequences.

**TP backward all-gather buffer**: During the backward pass with tensor parallelism, each GPU must all-gather partial activation outputs from other TP ranks to compute correct gradients. This requires a temporary buffer per layer. The buffer is transient (freed after each layer's backward), so only one layer's worth is live at a time:
```
M_tp_backward_peak = (b x s x d) x (N_tp - 1) / N_tp x beta
```
For LLaMA 7B (d=4096) with TP=4, b=4, s=4096, bf16: `4 x 4096 x 4096 x 0.75 x 2` = ~96 MB per layer, but only one layer is active at a time. At TP=2 the factor is 0.5 instead of 0.75. The calculator should include this in the communication/temporary buffer estimate when N_tp > 1.

### 5.7 Pipeline Parallelism Effect

PP distributes layers across stages:
```
Layers per stage = L / N_pp
```

**Most-loaded stage** (the memory bottleneck): With PP, the first and last pipeline stages hold the embedding and output projection weights in addition to their share of transformer layers. The calculator should estimate the most-loaded stage, not the average:
```
Ψ_transformer_per_stage = Ψ_transformer / N_pp
Ψ_most_loaded_stage = Ψ_transformer_per_stage + Ψ_embedding  (embedding + output head)
Ψ_per_gpu = Ψ_most_loaded_stage / N_tp
```
For tied embeddings, `Ψ_embedding = V × d`; for untied, `Ψ_embedding = 2 × V × d`. For models with large vocabularies (V=128K+), the embedding can be significant -- e.g., LLaMA 3 8B's embedding is ~525M params, adding ~1 GB in bf16 to the bottleneck stage beyond the uniform `Ψ/N_pp` estimate.

**Embedding-aware PP partitioning**: When the embedding layer is comparable in size to a transformer block (common with large vocabularies), it can be treated as an equivalent pipeline stage for load balancing. This changes the divisibility constraint from `L % N_pp == 0` to `(L + 2) % N_pp == 0` (counting the input embedding and output projection as two additional "virtual layers"). For example, BLOOM-176B used this approach: with 70 transformer layers and PP=12, `70 % 12 != 0` but `(70 + 2) % 12 == 0`, enabling even partitioning by assigning the embedding/output layers as dedicated stages. The calculator should check both `L % N_pp == 0` and `(L + 2) % N_pp == 0` when validating PP configurations, and suggest the embedding-aware option when the standard constraint fails but the embedding-aware one passes.

```
M_params_per_gpu = Ψ_per_gpu × β  (model weights on bottleneck GPU)
M_activations_per_gpu = per-layer activation × layers_per_stage
```

PP overhead (pipeline bubble):
```
Bubble fraction = (N_pp - 1) / (num_microbatches + N_pp - 1)
```
**Hard minimum (1F1B schedule)**: The standard 1F1B (one-forward-one-backward) pipeline schedule requires `num_microbatches >= N_pp - 1`. Below this threshold the pipeline cannot be filled and the schedule fails. AFAB (all-forward-all-backward) has no such constraint but stores all micro-batch activations simultaneously, greatly increasing memory. The calculator should enforce `num_microbatches >= N_pp - 1` as a hard constraint when PP is active and warn the user if violated.

Rule of thumb: need num_microbatches >= 4 x N_pp to keep bubble < 20%.

**Higher efficiency thresholds** (validated by BLOOM-176B training): For 90% pipeline efficiency, need `num_microbatches >= 8 × N_pp`; for 94% efficiency, need `num_microbatches >= 16 × N_pp`. The 4x rule above is the minimum for acceptable efficiency; production training runs typically target 8-16x.

Interleaved (virtual pipeline) schedule: Megatron-LM supports splitting each pipeline stage into multiple virtual stages (VP chunks). This reduces the bubble at the cost of more in-flight microbatches:
```
Bubble fraction (interleaved) = (N_pp - 1) / (VP × num_microbatches + N_pp - 1)
```
Where VP = virtual_pipeline_chunks (typically 2-8). The bubble shrinks by a factor of ~VP compared to non-interleaved.

However, interleaved scheduling increases peak activation memory because more microbatches are simultaneously in-flight:
```
Activation memory multiplier = 1 + (N_pp - 1) / (N_pp × VP)
```
The calculator should apply this multiplier to activation memory when interleaved PP is selected. For example, with PP=4 and VP=2, the multiplier is `1 + 3/8 = 1.375` (37.5% more activation memory). This trades memory for reduced pipeline bubble.

---

## 6. Training Time Estimation

### 6.1 Core Formula

```
T_seconds = C / (N_gpu × F_peak × MFU)
```

Where:
- C = total FLOPS (from Section 4), **adjusted for activation recomputation** (see below)
- F_peak = peak **dense matmul** FLOPS per GPU at training precision (BF16 for mixed precision; see Section 6.2 for precision guidance)
- MFU = Model FLOPS Utilization

**Activation recomputation adjustment**: The base formula `C = 6ΨD` assumes no activation recomputation. When activation checkpointing is enabled, the forward pass is recomputed during backward, increasing total compute:

| Checkpointing Mode | Compute Formula | Overhead |
|---------------------|-----------------|----------|
| None | C = 6ΨD | Baseline |
| Selective | C = 6ΨD × (1 + s/(18h)) | 2-7% (s/h dependent) |
| Full | C = 8ΨD | 33% more |

The calculator must apply this multiplier when the user selects activation checkpointing. Full recomputation doubles the forward pass cost (2ΨD becomes 4ΨD), giving 4ΨD + 4ΨD = 8ΨD total.

**Selective overhead detail** (Korthikanti et al., Table 2): The previous "~17% / 7*Psi*D" estimate was too conservative. The actual selective recomputation overhead depends on the sequence-length-to-hidden-dimension ratio `s/h`:
```
Overhead ratio = (1 + 2s/(9h)) / (1 + s/(6h)) - 1
```
For typical large models: GPT-3 175B (s=2048, h=12288) = **0.97%** overhead; a model with s=4096, h=4096 = **4.8%** overhead. The simplified per-token formula is `C_selective = 6ΨD × (1 + s/(18h))`, derived from Korthikanti Appendix A. This also gives the HFU/MFU ratio: `HFU/MFU = 1 + s/(18h)` when using selective checkpointing. The calculator should use `s/(18h)` as the overhead factor rather than a fixed 17%.

**Empirical wall-clock overhead**: The 33% compute overhead for full recomputation is the theoretical minimum assuming perfect overlap of recomputation with backward. Measured per-layer on a 22B model (A100, Korthikanti et al. Table 4): full recompute = **39% overhead** (19.6ms baseline vs 27.2ms), selective = **7% overhead** (19.6ms vs 20.9ms), selective + sequence parallelism = **4% overhead** (19.6ms vs 20.3ms). The per-layer overhead exceeds the theoretical 33% because recomputation disrupts backward-pass communication overlap. End-to-end wall-clock overhead is typically ~30-39% per-layer for full recompute, and ~1.5-1.65x overall (HuggingFace benchmarks, gpu_poor). The calculator should use **1.33x for FLOP estimation** (since actual FLOPs increase by exactly 33%) but may optionally note a ~1.4-1.65x wall-clock penalty in the training time estimate to set more realistic expectations.

### 6.2 MFU vs HFU

**MFU (Model FLOPS Utilization)** measures achieved throughput against peak hardware FLOPS using the *ideal* model FLOPs per token, regardless of implementation choices like activation checkpointing:
```
MFU = tokens_per_second × (6Ψ + 12Lds) / (N_gpu × F_peak)
```
Where `6Ψ + 12Lds` is the per-token model FLOPs from Section 4.1 (the PaLM formula). For large models at standard context lengths, the `12Lds` attention term is <1% of `6Ψ`, so the simplified `MFU ≈ (6Ψ × tokens_per_second) / (N_gpu × F_peak)` is often used. PaLM reports both variants (46.2% with attention term, 45.7% without for PaLM 540B). The calculator should use the full formula for consistency with the compute estimation in Section 4.1.

**F_peak precision**: F_peak must be the **BF16 dense matmul** peak TFLOPS (e.g., 989 TFLOPS for H100 SXM). Do NOT use FP8 peak (1,979 TFLOPS for H100) or structured sparsity peak -- either would halve the MFU number and produce a 2x error in training time estimates. On current GPU hardware, "BF16 peak" and "peak matmul TFLOPS" are the same number; this is the value listed in the GPU specs table (Section 7).

**HFU (Hardware FLOPS Utilization)** measures the same ratio but using *actual executed* FLOPs, including recomputation overhead. With full activation checkpointing, the executed FLOPs are 8ΨD instead of 6ΨD:
```
HFU = (actual_FLOPs_per_second) / (N_gpu × F_peak)
```

MFU is the fairer comparison metric because it is independent of implementation choices. A system with activation checkpointing will always show higher HFU than MFU (since it does more work per token), but this does not mean it is faster. The calculator should use **MFU** (based on 6ΨD) for its utilization display and training time formula, and apply the activation checkpointing overhead separately (Section 6.1).

**Common mistake -- omitting MFU from the time formula**: Using peak FLOPS without an MFU factor produces dramatic underestimates. For example, HyperCLOVA 82B trained on 1024 A100s: a naive estimate using raw peak FLOPS gives ~2.7 days, but actual training took 13.4 days -- a **5x underestimate**. The training time formula in Section 6.1 avoids this by dividing by MFU, which is essential for realistic estimates.

### 6.3 MFU Guidelines

MFU depends on model size, batch size, parallelism, and hardware:

| Scenario | Typical MFU |
|----------|-------------|
| Small model (<1B), 1-8 GPUs | 25-35% |
| Medium model (1B-10B), 8-64 GPUs | 35-45% |
| Large model (10B-100B), 64-512 GPUs | 40-50% |
| Very large (100B+), 512+ GPUs | 35-45% |
| State-of-the-art (PaLM-scale) | 45-55% |

**Note on state-of-the-art MFU**: PaLM 540B achieved **46.2% MFU** on 6144 TPUv4 chips. The commonly cited 57.8% figure is HFU (which includes rematerialization overhead), not MFU. This distinction matters: using 57.8% as an MFU target would underestimate training time by ~25%. The 45-55% range reflects actual MFU from the best-optimized large-scale runs.

**Historical MFU reference points** (PaLM paper, Table 11): MFU has improved significantly over time as training stacks have matured. These published MFU values for 100B+ models show the progression:

| Model | Year | Params | Hardware | MFU |
|-------|------|--------|----------|-----|
| GPT-3 | 2020 | 175B | ~10K V100 | 21.3% |
| Gopher | 2021 | 280B | 4096 TPUv3 | 32.5% |
| MT-NLG | 2022 | 530B | 2240 A100 | 30.2% |
| PaLM | 2022 | 540B | 6144 TPUv4 | 46.2% |

The jump from ~21-32% (2020-2021) to 46% (PaLM) reflects advances in parallelism strategies and software optimization rather than hardware alone. These numbers validate the "Very large (100B+): 35-45%" guideline as appropriate for well-optimized modern stacks, while showing that less-optimized or older setups can fall to 20-30%.

**Llama 3 405B MFU reference points** (H100 SXM, Meta 2024): These measurements show how MFU varies with cluster size and sequence length for a single model on current-generation hardware:

| GPUs | Parallelism | Seq Length | TFLOP/s per GPU | MFU |
|------|-------------|------------|-----------------|-----|
| 8,192 | TP8/PP16/DP64 | 8K | 430 | 43% |
| 16,384 | TP8/PP16/DP128 | 8K | 400 | 41% |
| 16,384 | TP8/CP16/PP16/DP8 | 128K | 380 | 38% |

Key takeaways: (1) MFU drops ~2% when doubling from 8K to 16K GPUs at the same sequence length, due to increased communication overhead; (2) long-context training (128K) reduces MFU by a further ~3% due to context parallelism overhead; (3) even at 16K GPUs, a well-optimized stack achieves 38-43% MFU, validating the "Very large (100B+)" guideline range of 35-45%.

**Additional MFU reference points**: DeepSeek-v3 training achieves an estimated ~20-30% MFU (no official number published; SemiAnalysis estimate), notably below the Llama 3 range despite being a state-of-the-art model -- likely due to MoE routing overhead and different hardware/interconnect choices. As a theoretical ceiling, even pure matrix multiplication kernels achieve only **70-80% MFU** on modern GPUs, meaning the ~40% achieved by large-scale training represents roughly half of the hardware's realistic maximum.

**What MFU includes**: MFU as used in the training time formula (Section 6.1) is a single efficiency factor that captures *all* sources of throughput loss relative to peak hardware FLOPS. This includes not only raw compute utilization but also communication overhead (DP all-reduce, TP all-reduce, PP bubbles), data loading and preprocessing stalls, checkpointing I/O, memory allocator overhead, and kernel launch latency. Some calculators model these as separate "overhead" or "scaling" multipliers on top of MFU, but this risks double-counting. The calculator should use MFU as a single comprehensive efficiency knob rather than stacking multiple inefficiency factors.

**Why MFU is 30-55% -- sources of throughput loss**: MFU falls well below 100% due to a cascade of inefficiencies, each compounding on the last. Empirical A100 measurements (Bahdanau, 2023) illustrate this progression:
1. **Tensor core utilization gap** (~76% of peak): Even pure matrix multiplications achieve only ~76% of peak TFLOPS due to memory latency, warp scheduling, and tile quantization effects. Highly optimized matmul kernels (cuBLAS, custom CUDA) reach 70-80% MFU as an empirical ceiling, establishing an upper bound that no real training workload can exceed.
2. **Non-matmul operations** (~60% of peak): Memory-bandwidth-bound operations like LayerNorm, softmax, activation functions (ReLU/SiLU/GELU), and residual additions cannot saturate tensor cores and add ~15-20% throughput loss on top of the matmul gap.
3. **Framework and kernel launch overhead** (down to 20-40% of peak): Unoptimized implementations (e.g., naive HuggingFace GPT-2) suffer from Python overhead, kernel launch latency, and unfused operations. Optimized frameworks (Megatron-LM, NeMo) recover much of this through kernel fusion, efficient scheduling, and CUDA Graphs (which convert a sequence of kernel launches into a DAG launched once, eliminating per-kernel launch overhead). **Important**: `nvidia-smi` reports *kernel utilization* (fraction of time any kernel is running), NOT MFU. A GPU showing 100% `nvidia-smi` utilization may still have very low MFU if the running kernels are inefficient or memory-bound. Do not conflate `nvidia-smi` utilization with compute efficiency.
4. **Distributed communication**: DP all-reduce, TP all-reduce, and PP bubbles add idle time proportional to cluster size and interconnect bandwidth.
5. **I/O and system overhead**: Data loading, checkpointing, and memory allocator fragmentation contribute the remaining loss.

The gap between items 2-3 explains why framework choice matters so much: a well-optimized stack (Megatron-LM) achieves 40-55% MFU, while a naive implementation on the same hardware may see 15-25%.

**Arithmetic intensity and compute vs. memory boundedness**: Arithmetic intensity is the ratio of FLOPs performed to bytes of memory moved (FLOPs/byte). Modern GPUs have a compute-to-bandwidth ratio of ~150-300 FLOPs/byte (e.g., H100 SXM: 989 TFLOPS BF16 / 3.35 TB/s HBM = ~295 FLOPs/byte). When a workload's arithmetic intensity exceeds this ratio, it is *compute-bound* and can approach peak FLOP/s; when it falls below, it is *memory-bandwidth-bound* and MFU drops proportionally. Strategies to increase arithmetic intensity include larger batch sizes, kernel fusion (reducing intermediate memory traffic), and algorithmic rewrites like Flash Attention's online softmax (which restructures the computation to perform more FLOPs per byte moved between HBM and SRAM).

**Small micro-batch MFU degradation**: The MFU ranges above assume reasonably large micro-batch sizes (b >= 4). When micro-batch size is very small (b = 1-2) and the model's hidden dimension is modest, individual matmuls become memory-bandwidth-bound rather than compute-bound because the arithmetic intensity (FLOPs per byte loaded) drops below the GPU's compute-to-bandwidth ratio. This can reduce MFU to well below the guideline ranges -- in extreme cases by 2-5x. The calculator should warn when b <= 2 that MFU may be significantly lower than the default estimate.

The calculator should provide a default MFU based on model size and GPU count, with a slider for user override (range: 10-70%).

**Profiling tools for MFU diagnosis**: For users seeking to measure or diagnose MFU in their actual training runs, the key tools are: (1) NVIDIA DCGM with `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (fraction of time Tensor Cores are active) and `DCGM_FI_PROF_DRAM_ACTIVE` (HBM bandwidth utilization) -- these directly indicate whether a workload is compute-bound or memory-bound; (2) PyTorch Profiler traces, which show idle gaps between kernel launches on CUDA streams; (3) `nvidia-smi` for basic kernel occupancy (but see the warning above about its limitations). Measuring MFU itself typically requires pen-and-paper analysis: compute model FLOPs per iteration from the architecture, measure wall-clock time per iteration, and divide.

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

### 6.5 Failure-Adjusted Training Time

At large scale (1000+ GPUs), hardware failures are frequent enough to materially extend training time. The following closed-form formula (from JGalego/llm-calc) accounts for recovery overhead and lost work between checkpoints:

```
T_actual = T_theory / [1 - f × N_inst × (t_recovery + 1/(2 × f_checkpoint))]
```

Where:
- `T_theory` = theoretical training time from Section 6.1 (in days)
- `f` = instance failure rate (failures per instance per day; default **0.01** = 1%)
- `N_inst` = number of instances = ceil(N_gpu / GPUs_per_node)
- `t_recovery` = time to detect, restart, and reload after a failure (default **1 hour**, expressed in days = 1/24)
- `f_checkpoint` = checkpoint saves per day (default **24** = hourly)
- `1/(2 × f_checkpoint)` = average lost work between checkpoints (half the checkpoint interval)

The denominator captures the self-reinforcing nature of failures: longer training means more failures, which means even longer training. The formula diverges (training becomes infeasible) when the failure overhead approaches 100% of available time.

**Impact by scale:**

| N_gpu | N_inst (8 GPU/node) | Daily failures | Overhead per failure | Training time multiplier |
|-------|---------------------|----------------|----------------------|--------------------------|
| 64 | 8 | 0.08 | 1.5 hrs | ~1.005x (negligible) |
| 256 | 32 | 0.32 | 1.5 hrs | ~2% |
| 1,024 | 128 | 1.28 | 1.5 hrs | ~8% |
| 4,096 | 512 | 5.12 | 1.5 hrs | ~32% |
| 16,384 | 2,048 | 20.5 | 1.5 hrs | ~130% (if feasible) |

The calculator should:
1. Compute and display the failure-adjusted training time alongside the theoretical time when N_gpu >= 256
2. Expose failure rate, recovery time, and checkpoint frequency as advanced inputs
3. Warn when the denominator drops below 0.5 (training time more than doubles due to failures)
4. Note that at extreme scale (16K+ GPUs), the 1% failure rate assumption may understate reality -- Meta reported hundreds of interruptions during LLaMA 3 405B training on 16K H100s

---

## 7. GPU Hardware Specifications

Embed these as selectable presets. Users should also be able to enter custom GPU specs.

| GPU | VRAM (GB) | BF16 TFLOPS | TF32 TFLOPS | FP8 TFLOPS | Mem BW (GB/s) | NVLink BW (GB/s) | TDP (W) |
|-----|-----------|-------------|-------------|------------|---------------|-------------------|---------|
| V100 32GB | 32 | 125 | — | — | 900 | 300 | 300 |
| A100 PCIe 80GB | 80 | 312 | 156 | — | 2,039 | — | 300 |
| A100 40GB | 40 | 312 | 156 | — | 1,555 | 600 | 400 |
| A100 80GB | 80 | 312 | 156 | — | 2,039 | 600 | 400 |
| H100 PCIe 80GB | 80 | 989 | 378 | 1,979 | 2,039 | — | 350 |
| H100 SXM | 80 | 989 | 495 | 1,979 | 3,350 | 900 | 700 |
| H100 NVL | 94 | 989 | 495 | 1,979 | 3,350 | 900 | 800 |
| H200 SXM | 141 | 989 | 495 | 1,979 | 4,800 | 900 | 700 |
| B200 | 192 | 2,250 | 1,125 | 4,500 | 8,000 | 1,800 | 1,000 |
| GB200 NVL72 | 384 | 4,500 | 2,250 | 9,000 | 16,000 | 1,800 | 2,700 |
| MI250X | 128 | 383 | — | — | 3,276 | — | 560 |
| MI300X | 192 | 1,307 | — | 2,614 | 5,300 | — | 750 |
| L40S | 48 | 362 | 183 | — | 864 | — | 350 |
| RTX 4090 | 24 | 165 | 83 | — | 1,008 | — | 450 |
| RTX 4080 | 16 | 97 | 49 | — | 717 | — | 320 |
| RTX 3090 | 24 | 71 | 36 | — | 936 | — | 350 |
| RTX 3060 12GB | 12 | 25 | 13 | — | 360 | — | 170 |

Note: Consumer GPU BF16 TFLOPS listed above are tensor core rates (with sparsity disabled). Consumer GPUs lack BF16 support prior to Ampere (30-series); the RTX 3090/3060 values are FP16 tensor core rates. PCIe variants lack NVLink, so TP across PCIe GPUs uses PCIe bandwidth (~64 GB/s for Gen5) instead. The calculator should warn when N_tp > 1 is selected with a PCIe GPU.

**TF32 (TensorFloat-32) note**: TF32 is a **compute mode**, not a storage format — it uses 19-bit precision internally in tensor cores (10-bit mantissa of FP16, 8-bit exponent of FP32) but all tensors remain stored as FP32 (4 bytes/element) in memory. TF32 is **enabled by default** in PyTorch 1.12+ on Ampere and newer NVIDIA GPUs. When the user selects "FP32 training" on an Ampere+ GPU, the calculator should use the TF32 TFLOPS rate (not the non-tensor-core FP32 rate) for training time estimation, since this reflects the actual default behavior. Without this adjustment, FP32 training time estimates would be approximately **8x too pessimistic**. TF32 does not affect memory calculations — all tensors remain in FP32 at 4 bytes per element. Pre-Ampere GPUs (V100) and AMD GPUs do not support TF32; for those, the calculator should use the non-tensor-core FP32 rate (e.g., V100: 15.7 TFLOPS). For consumer GPUs (RTX 30xx/40xx), TF32 TFLOPS listed are tensor core rates.

**Dense vs sparse TFLOPS warning**: NVIDIA's official spec sheets frequently headline **structured sparsity (2:4) TFLOPS**, which are exactly **2x the dense TFLOPS**. For example, the H100 SXM is often quoted at 1,979 BF16 TFLOPS -- that is the sparsity rate; the dense rate is 989 TFLOPS. All values in the table above are **dense TFLOPS**, which is what training workloads achieve (2:4 sparsity requires specially pruned weight matrices and is not used during standard training). When users enter custom GPU specs, the calculator should validate against known dense values and warn if the entered TFLOPS appears to be a sparsity-inflated figure (i.e., roughly 2x a known dense value). Using sparsity TFLOPS in the training time formula would underestimate wall-clock time by 2x.

**GPUs per node**: Typically 8 for NVIDIA (DGX), 8 for AMD. This constrains max TP degree. Consumer/workstation GPUs (L40S, RTX 4090, RTX 3090) are typically 1-2 per node without NVLink.

**Inter-node bandwidth defaults** (for communication overhead estimation):
- InfiniBand HDR: 200 GB/s (A100-era clusters)
- InfiniBand NDR: 400 GB/s (H100-era clusters)
- The calculator should default to 200 GB/s and allow user override.

### Apple Silicon (Unified Memory)

Apple Silicon chips use **unified memory** shared between CPU and GPU -- there is no separate VRAM. The "Max Memory" column is the maximum configurable unified RAM for that chip; users may have less depending on their configuration. These chips are relevant for LoRA/QLoRA fine-tuning and small-model pretraining on consumer hardware. Only the Max and Ultra tiers have sufficient memory for meaningful LLM training work.

| Chip | FP16 TFLOPS | Max Memory (GB) | Mem BW (GB/s) |
|------|-------------|-----------------|---------------|
| M1 Max | 10.4 | 64 | 400 |
| M1 Ultra | 21.0 | 128 | 800 |
| M2 Max | 13.6 | 96 | 400 |
| M2 Ultra | 27.2 | 192 | 800 |
| M3 Max | 14.2 | 128 | 400 |
| M3 Ultra | 28.0 | 512 | 800 |
| M4 Max | 16.0 | 128 | 546 |

Note: Apple Silicon has no BF16 tensor core support; all values are FP16. These chips lack NVLink or multi-GPU interconnect, so parallelism is limited to single-device strategies (no TP/PP). The calculator should treat Apple Silicon as single-GPU only (N_tp=1, N_pp=1, N_dp=1) and use the user-selected memory configuration (not the max) as available VRAM. The M3 Ultra's 512 GB unified memory is notable -- it can hold a full 70B model in bf16 (140 GB) with room for optimizer states, enabling full fine-tuning of large models on a single device.

---

## 8. Cost Estimation

### 8.1 Compute Cost (Primary)

```
Cost_compute = N_gpu × T_hours × price_per_GPU_hour
```

This is the dominant cost component for most training runs.

Provide default pricing presets (user can override):

| GPU | Approx. On-Demand ($/hr/GPU) |
|-----|------------------------------|
| V100 32GB | $1.50 - $2.50 |
| A100 80GB | $2.50 - $4.00 |
| H100 SXM | $3.00 - $5.00 |
| H200 SXM | $4.00 - $6.00 |
| B200 | $5.00 - $8.00 |

**Reference cloud instances** (representative on-demand pricing; prices change frequently -- the calculator should let users override):

| Provider | Instance | GPU | Count | VRAM/GPU | $/hr |
|----------|----------|-----|-------|----------|------|
| AWS | p4d.24xlarge | A100 | 8 | 40 GB | $24.15 |
| AWS | g5.xlarge | A10G | 1 | 24 GB | $1.01 |
| GCP | a2-highgpu-1g | A100 | 1 | 40 GB | $2.95 |
| GCP | g2-standard-4 | L4 | 1 | 24 GB | $0.99 |
| Azure | Standard_NC8as_T4_v3 | T4 | 1 | 16 GB | $0.90 |
| Lambda | gpu_1x_a100_sxm5 | A100 | 1 | 80 GB | $1.99 |

The calculator should accept a custom $/GPU/hr input and show total estimated cost.

### 8.2 Checkpoint Storage Cost

Training checkpoints accumulate over a run. Using the checkpoint size from Section 5.1 (12Ψ bytes for AdamW):
```
num_checkpoints = ceil(T_actual_days × f_checkpoint)
total_checkpoint_storage = num_checkpoints × 12Ψ bytes
avg_storage = total_checkpoint_storage / 2  (linear accumulation → average over run is half the peak)
Cost_storage = price_per_GB_month × (avg_storage_GB + dataset_GB) × (T_actual_days / 30.25)
```

Default storage price: **$0.023/GB/month** (AWS S3 standard). The calculator should expose this as an advanced input. For large models, checkpoint storage is significant: a 70B model saving hourly checkpoints over 90 days accumulates ~2,160 checkpoints x 840 GB each = ~1.8 PB peak storage.

### 8.3 Failure Overhead Cost

When using the failure-adjusted training time from Section 6.5, the additional training time translates directly to additional compute cost:
```
Cost_failure_overhead = N_gpu × (T_actual - T_theory) × price_per_GPU_hour
```

This includes both the recovery time (GPUs idle but allocated) and recomputation cost (re-doing work since last checkpoint). At scale, this is substantial: for a 4,096-GPU run with ~32% failure overhead, the failure cost is roughly a third of the base compute cost.

### 8.4 Total Cost

```
Cost_total = Cost_compute + Cost_storage + Cost_failure_overhead
```

The calculator should display each component separately so users can see the cost breakdown. For small-scale runs (<256 GPUs), `Cost_failure_overhead` is negligible and can be omitted from the display. `Cost_storage` is always a small fraction of `Cost_compute` but useful for storage planning at scale.

---

## 9. Parallelism Recommendation Engine

Given memory constraints and GPU count, recommend a parallelism strategy:

### Decision Logic

```
1. Calculate M_model_states = ΦΨ (no parallelism; Φ from Section 5.1)
2. If M_model_states fits in one GPU (with room for activations):
   → Use DP only. N_dp = N_gpu.
3. Else, try ZeRO stages (using Φ-based formulas from Section 5.2):
   a. ZeRO-1: (2 + β_grad)Ψ + K_opt·Ψ/N_dp — fits? Use this.
   b. ZeRO-2: 2Ψ + (β_grad + K_opt)·Ψ/N_dp — fits? Use this.
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
6. If sequence length is very long (>32K) and activation memory still exceeds
   GPU capacity even with checkpointing:
   → Add CP (start with N_cp = 2, increase to 4, 8, 16)
   → Replaces s with s/N_cp in activation memory formulas (Section 5.3)
   → CP trades DP parallelism for sequence sharding (DP shrinks as CP grows)
7. Remaining GPUs become DP:
   N_dp = N_gpu / (N_tp × N_cp × N_pp)
```

### Minimum GPU Memory Floor (Largest Layer)

Even with ZeRO-3 sharding across arbitrarily many GPUs, a single transformer layer's parameters must be fully gathered on one GPU during forward and backward passes. During backward, both the gathered parameters and their gradients coexist in memory. This sets an absolute minimum VRAM requirement:

```
Ψ_largest_layer = Ψ_attn + Ψ_ffn + Ψ_norm  (single transformer block)
M_min_gpu = Ψ_largest_layer × 2β              (β bytes for gathered params + β bytes for gradients)
```

**Closed-form for the largest layer** (Rajbhandari et al., 2021): The largest single weight matrix in a standard transformer is the FFN up-projection (d -> d_ff). The minimum working memory for gathering its parameters and gradients is:
```
Standard FFN (d_ff = 4d):   M_min_gpu = 16 × d² bytes     (in bf16: 4d² params × 2 + 4d² grads × 2)
SwiGLU FFN (arbitrary d_ff): M_min_gpu = 4 × d × d_ff bytes (in bf16: d×d_ff params × 2 + d×d_ff grads × 2)
```
These closed-form expressions let the calculator compute the floor from just `d` and `d_ff` without enumerating all layer parameters. The calculator should flag when `M_min_gpu > GPU_VRAM × 0.8` (leaving 20% for framework overhead and activations), as this indicates that even with full ZeRO-3 sharding, per-GPU memory pressure from the largest single operator will be severe.

In bf16 (β=2), this is 4 bytes per parameter in the largest layer. For example, LLaMA 70B has ~1.1B params per layer, so the minimum per-GPU memory is ~4.4 GB in bf16 (2.2 GB params + 2.2 GB gradients). In practice, activations and transient recomputation working memory (Section 5.3) push this higher. Display this as an output: "Minimum GPU VRAM (even with full sharding)".

### ZeRO Stage Selection Heuristic

When multiple ZeRO stages fit in memory, the recommendation engine should prefer the stage that maximizes throughput. The speed-vs-memory tradeoff for ZeRO stages is well-established:

| Fastest (throughput) | Most memory efficient |
|---|---|
| ZeRO-1 | ZeRO-3 + CPU offload |
| ZeRO-2 | ZeRO-3 |
| ZeRO-2 + CPU offload | ZeRO-2 + CPU offload |
| ZeRO-3 | ZeRO-2 |
| ZeRO-3 + CPU offload | ZeRO-1 |

The recommendation engine should select the **lowest ZeRO stage** (fewest sharding) that fits in GPU memory, since lower stages have less communication overhead and higher throughput. Only escalate to a higher stage when the lower one does not fit. CPU offloading should be a last resort -- it enables training that would otherwise be impossible, but at a significant throughput penalty.

### TP vs ZeRO-3 Communication Tradeoff

The ratio of tensor parallelism communication to ZeRO-3 communication per training step (Rajbhandari et al., 2020):
```
TP_comm / ZeRO3_comm = (b x s) / (3 x d)
```
When `b x s >> 3 x d` (large batch, long sequences), ZeRO-3 has lower total communication volume than TP. However, this is a volume comparison only -- TP communication travels over NVLink (intra-node, ~900 GB/s on H100), while ZeRO-3 communication often traverses inter-node interconnect (~50-400 GB/s). The recommendation engine should prefer TP within a node (where NVLink is available) and ZeRO across nodes, unless the model is too small to benefit from TP (few attention heads) or the cluster has uniformly high-bandwidth interconnect.

### Multi-Node Parallelism Guidance

When training spans multiple nodes, interconnect bandwidth determines the optimal parallelism strategy:

- **Multi-node with slow interconnect** (e.g., Ethernet, InfiniBand HDR): Prefer HYBRID_SHARD (ZeRO++ Stage 3) or standard ZeRO with PP. Keep TP strictly within a single node. Use DP or PP across nodes since they have lower communication bandwidth requirements than TP.
- **Multi-node with fast interconnect** (InfiniBand NDR 400 Gb/s+): Standard ZeRO-3 / FULL_SHARD across all GPUs is viable. TP can extend across nodes only if NVSwitch or equivalent high-bandwidth fabric is available (rare outside DGX SuperPOD configurations).
- **Single node**: All parallelism strategies are viable. TP up to 8 (full node) is standard. HYBRID_SHARD is unnecessary (equivalent to FULL_SHARD within one node).

The calculator should default to TP-within-node and flag when the user's configuration would place TP across node boundaries.

**TP degree tuning**: Using fewer than the maximum TP ranks per node can improve throughput. BLOOM-176B found TP=4 outperformed TP=8 by 19% on 8-GPU nodes, because smaller TP degree reduces per-layer all-reduce volume and increases the amount of work per GPU (better arithmetic intensity). The optimal TP degree depends on model size, interconnect topology, and the tradeoff between communication overhead and per-GPU memory pressure. The calculator should not assume TP=8 is always optimal; when memory permits, TP=4 or TP=2 with higher DP may yield better throughput.

**Parallelism dimension ordering** (Meta, Llama 3 scaling): Dimensions with higher communication demands should be placed on higher-bandwidth interconnects. The recommended ordering from innermost (highest bandwidth) to outermost (lowest bandwidth) is:
1. **TP** (innermost): 4 all-reduces per layer, requires NVLink (intra-node)
2. **CP**: all-gather KV per layer, high bandwidth demand but less than TP
3. **PP**: point-to-point between 2 ranks per stage boundary, lowest per-layer sync requirement
4. **DP/FSDP** (outermost): one all-reduce per training step, overlapped with backward compute

This ordering means that for an 8-GPU-per-node cluster: TP ranks share a node, CP ranks span adjacent nodes with fast interconnect, PP and DP span the remaining topology. The calculator should validate that the user's parallelism configuration respects this bandwidth hierarchy and warn when high-communication dimensions (TP, CP) are mapped to low-bandwidth interconnects.

### Constraints
- N_tp must divide both a (attention heads) and a_kv (KV heads) evenly. For GQA models, a_kv is the binding constraint (e.g., LLaMA 2 70B has a_kv=8, so N_tp must divide 8)
- N_tp must divide d_ff evenly (FFN weight columns are split across TP ranks). For SwiGLU models with non-standard d_ff values, this is an additional binding constraint beyond attention head divisibility.
- N_tp ≤ 8 (GPUs per node, NVLink requirement)
- N_pp must divide L (layers) evenly, or `(L + 2) % N_pp == 0` when embedding-aware partitioning is used (see Section 5.7)
- **Hidden dimension alignment (wave/tile quantization)**: `d % 128 == 0` for efficient GPU tensor core utilization. Modern GPUs process matmuls in tiles (e.g., 128x128 on A100/H100); misaligned hidden dimensions cause partial tiles that waste SM cycles. BLOOM-176B measured a **38% throughput improvement** from proper alignment (94 to 131 TFLOPs on a 200B model) by ensuring `d` is divisible by the LCM of the tile size and TP degree. The calculator should warn when `d % 128 != 0` in Detailed Mode and auto-round to the nearest multiple of 128 in Quick Mode. All model presets in Section 3.3 already satisfy this constraint.
- **Vocab size padding for TP**: When tensor parallelism is active, Megatron-LM pads the vocabulary size to be divisible by `128 × N_tp` so the embedding and output projection can be evenly split. The padded size is `ceil(V / (128 × N_tp)) × (128 × N_tp)`. The calculator should use this padded V for parameter counting and memory estimation when N_tp > 1. For example, V=128,256 with N_tp=8 pads to 129,024 (+768 entries).
- N_dp × N_tp × N_cp × N_pp = N_gpu (for dense models; N_cp = 1 when context parallelism is not used)
- N_dp × N_tp × N_cp × N_pp × N_ep = N_gpu (for MoE models; N_ep must divide E evenly)
- Global batch size B = b × G × N_dp
- **1F1B microbatch minimum**: When pipeline parallelism is active with the standard 1F1B schedule, `num_microbatches >= N_pp - 1` (hard minimum; see Section 5.7). The calculator should validate this and warn when violated.
- **ZeRO + Pipeline Parallelism compatibility**: ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism (gradient sharding conflicts with PP's gradient accumulation across stages). Only ZeRO-0 or ZeRO-1 can be combined with PP. The calculator must enforce this constraint:

| ZeRO Stage | + TP | + PP |
|------------|------|------|
| ZeRO-0 | Yes | Yes |
| ZeRO-1 | Yes | Yes |
| ZeRO-2 | Yes | **No** |
| ZeRO-3 | Yes | **No** |

When PP is active, the recommendation engine must not select ZeRO-2 or ZeRO-3. Conversely, if ZeRO-2/3 is needed for memory, PP cannot be used.

**FSDP + Pipeline Parallelism**: FULL_SHARD (ZeRO-3) must not be used with PP because parameters are freed after forward and must be AllGathered again for every micro-batch in the PP schedule, which is prohibitively expensive. The choice between ZeRO-1 and ZeRO-2 semantics depends on the micro-batch count relative to pipeline depth (Meta, Llama 3 scaling):
```
if micro_batch_per_dp_rank >= 2 × N_pp: use FSDP ZeRO-1 + interleaved 1F1B schedule
if micro_batch_per_dp_rank <  2 × N_pp: use FSDP ZeRO-2 (SHARD_GRAD_OP) + AFAB schedule
```
ZeRO-1 retains unsharded gradients across micro-batches, avoiding extra communication but using more memory. ZeRO-2 re-shards gradients after each micro-batch, saving memory at the cost of additional reduce-scatter operations. The calculator should apply this heuristic when FSDP + PP is active: default to ZeRO-1 (higher throughput) and fall back to ZeRO-2 only when the batch size is too small to fill the pipeline efficiently.

### Throughput Scoring for Strategy Selection

When multiple parallelism configurations fit in memory, the recommendation engine should rank them by estimated training throughput. A simple scoring heuristic (derived from LLMem, IJCAI-24):
```
Score(DP only)          = max_batch x N_gpu x 1.5    (1.5x bonus: no parameter gathering overhead)
Score(ZeRO-3 / FSDP)   = max_batch x N_gpu           (parameter all-gather overhead ~50% more comm than DP)
Score(TP only)          = max_batch                    (no data parallelism -- only one copy of data)
Score(DP + TP)          = max_batch x N_dp             (TP within node, DP across nodes)
```
Where `max_batch` is the largest micro-batch that fits in GPU memory for that configuration. The 1.5x multiplier reflects the empirically observed ~50% communication overhead penalty of ZeRO-3 relative to standard data parallelism (consistent with the note in Section 5.2). The engine should compute scores for all feasible configurations and recommend the highest-scoring one. This is a throughput heuristic, not a precise model -- actual throughput depends on interconnect bandwidth, batch size, and model architecture. The calculator should display 2-3 top configurations when scores are within 20% of each other, letting users choose based on their priorities (memory headroom vs. throughput).

### Output
Display the recommended configuration: N_dp x N_tp x N_cp x N_pp, ZeRO stage, and estimated pipeline bubble overhead.

---

## 10. Phase 2: Post-Training

Post-training covers everything after pretraining: supervised fine-tuning and preference alignment. The key difference is **multiple models may be in memory simultaneously**.

### 10.1 Supervised Fine-Tuning (SFT)

**Full fine-tuning**: Identical to pretraining memory (16Ψ + activations). Dataset is smaller so compute is less.

**LoRA** (Low-Rank Adaptation):
```
Base model (frozen, bf16):  2Ψ bytes  (no gradients, no optimizer)
LoRA adapter parameters:    Ψ_lora = 2 × r × d × M_modules × L
  where r = rank (8-64), M_modules = adapted modules per layer:
    - 4 (attention only: Q, K, V, O) — conservative default
    - 7 (attention + FFN: Q, K, V, O, gate, up, down) — common for SwiGLU models
LoRA gradients (bf16):      2 × Ψ_lora
LoRA optimizer (fp32):      12 × Ψ_lora  (master + Adam m + v)
Activations:                Same as full model (entire model runs forward/backward)

M_total_lora = 2Ψ + 16 × Ψ_lora + M_activations
```

Example: 7B SwiGLU model, rank 16, 32 layers:
```
With M_modules=4 (attention only):
  Ψ_lora = 2 × 16 × 4096 × 4 × 32 = 16.8M  (0.24% of base model)
  Memory: 2 × 7B + 16 × 16.8M = 14GB + 0.27GB = ~14.3GB + activations

With M_modules=7 (attention + SwiGLU FFN):
  Ψ_lora = 2 × 16 × 4096 × 7 × 32 = 29.4M  (0.42% of base model)
  Memory: 2 × 7B + 16 × 29.4M = 14GB + 0.47GB = ~14.5GB + activations
```

**QLoRA** (Quantized LoRA):

"4-bit quantization" maps to several concrete formats: NF4 (bitsandbytes, used by QLoRA), GPTQ-4bit, and AWQ-4bit. Similarly, "8-bit" maps to LLM.int8() (bitsandbytes), GPTQ-8bit, and AWQ-8bit. The calculator should accept a quantization bit-width (4 or 8) and display the corresponding format names for clarity.

```
Base model (4-bit NF4):    ~0.5Ψ bytes + ~0.01Ψ overhead (quantization constants)
LoRA adapters + optimizer: 16 × Ψ_lora (same as LoRA)
Activations:               Computed in bf16 (dequantize → compute → re-quantize)

M_total_qlora ≈ 0.55Ψ + 16 × Ψ_lora + M_activations
```

**QLoRA loading memory floor**: During model loading, the full model must be loaded in bf16/fp16 (~2Ψ bytes) before quantization to NF4. This creates a transient peak memory of ~2Ψ that exceeds the steady-state QLoRA memory for small LoRA configurations. The effective minimum GPU memory for QLoRA is therefore `max(M_total_qlora, 2Ψ)`. For example, a 7B model requires ~14 GB just to load before quantization, even though steady-state QLoRA training uses only ~4-5 GB for the base model. The calculator should report this loading floor as a warning when it exceeds the training memory.

**QLoRA throughput penalty**: QLoRA training is slower than standard LoRA due to the dequantize-compute-requantize overhead in each forward and backward pass. Empirical measurements show approximately **1.75x wall-clock time** compared to equivalent LoRA fine-tuning. The calculator should apply this penalty when estimating QLoRA training time.

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

**LoRA-as-reference-policy optimization**: When fine-tuning with LoRA, the frozen base model already serves as the reference policy -- disabling the LoRA adapter produces reference model outputs without loading a separate copy. This eliminates the 2Ψ reference model entirely (used by NeMo-Aligner and TRL):
```
M_total_dpo_lora_ref = 2Ψ + 16Ψ_lora + 2 × M_activations  (LoRA with shared reference)
```
This saves 2Ψ bytes compared to the standard LoRA DPO formula above (e.g., ~14 GB for a 7B model). The calculator should default to this optimization when LoRA/QLoRA is selected for DPO or GRPO, and show the savings compared to a separate reference model.

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
Where the factor of 2 is for K and V tensors, `a_kv × d_kv` is the per-layer KV width (equals `d` for MHA, smaller for GQA/MQA), `s_gen` is the generation sequence length, and `β_cache` is bytes per element (2 for bf16). For GQA models, the KV cache shrinks proportionally to `a_kv / a`. Note that `β_cache` can differ from the model's compute precision -- for example, INT8 KV cache (`β_cache = 1`) is commonly used with bf16 model weights in serving frameworks like vLLM and TGI. The calculator should allow independent selection of KV cache precision.

**Common optimization**: Critic is smaller than actor (e.g., half the layers), and reference model shares architecture but is frozen.

**LoRA-as-reference-policy**: As with DPO (Section 10.2), when the actor uses LoRA, the base model serves as the reference policy by disabling the adapter. This eliminates the 2Ψ_ref term from the training phase:
```
Training phase (LoRA actor): 2Ψ_actor + 16Ψ_lora + 16Ψ_critic + activations  (no separate ref)
Peak (LoRA actor) = max(generation, scoring, training) — saves 2Ψ vs separate reference
```

Compute per PPO step (K PPO epochs):
```
C_ppo_step = 2Ψ × generated_tokens  (generation)
           + 2Ψ_reward × scored_tokens  (reward)
           + K × (6Ψ_actor + 6Ψ_critic + 2Ψ_ref) × batch_tokens  (training)
```

**Generation phase wall-clock time** (applies to PPO and GRPO):

Generation is almost always the bottleneck in RL training loops. Autoregressive decode is *memory-bandwidth-bound* at small batch sizes because each token requires loading all model weights (~2Ψ bytes) to perform only ~2Ψ FLOPs -- an arithmetic intensity of ~1 FLOP/byte, far below modern GPUs which need ~150-300 FLOPs/byte to saturate compute.

```
T_generation = T_prefill + n_gen_tokens × T_decode_per_token

T_prefill = (2 × Ψ × s_prompt) / (F_peak × N_gpus)          [compute-bound]

T_decode_per_token = max(
    (2 × Ψ × β) / (BW_mem × N_gpus),                        [memory-bound term]
    (2 × Ψ × batch_gen) / (F_peak × N_gpus)                  [compute-bound term]
)
```

Where `BW_mem` is GPU memory bandwidth (e.g., 2.0 TB/s for A100-80GB, 3.35 TB/s for H100 SXM), `β` is bytes per parameter (2 for bf16), `batch_gen` is total concurrent generations, and `F_peak` is peak GPU FLOPS. Apply ~0.87-0.90 efficiency factor to `BW_mem` in practice.

The crossover batch size where decode transitions from memory-bound to compute-bound:
```
B_threshold = F_peak / BW_mem

| GPU         | BW_mem (TB/s) | F_peak (TFLOPS bf16) | B_threshold |
|-------------|---------------|----------------------|-------------|
| A100-80GB   | 2.0           | 312                  | ~156        |
| H100 SXM    | 3.35          | 989                  | ~295        |
| H200 SXM    | 4.8           | 989                  | ~206        |
```

Below `B_threshold`, throughput scales linearly with batch size at near-zero cost -- the calculator should flag this as an optimization opportunity when `batch_gen << B_threshold`.

**Maximum concurrent generations** (memory constraint):
```
max_batch_gen = (M_gpu_available - 2 × Ψ × β) / (M_kv_per_token × s_gen)

M_kv_per_token = 2 × a_kv × d_kv × L × β_cache   (bytes; factor of 2 for K and V)
```

Where `M_gpu_available` is GPU memory after framework overhead per GPU (accounting for TP/ZeRO sharding of weights), and `s_gen` is the maximum generation sequence length. This determines whether a given PPO batch size or GRPO group size `G` fits in memory during the generation phase.

### 10.4 GRPO (Group Relative Policy Optimization)

Simpler than PPO — no critic model, uses group-relative advantages:
```
Policy model (trainable):   16Ψ + M_act
Reference model (frozen):   2Ψ

M_total_grpo = 18Ψ + M_activations
             = 2Ψ + 16Ψ_lora + M_activations  (LoRA with shared reference — see Section 10.2)
```

Key difference: generates G completions per prompt, so the generation-phase KV cache (see formula in Section 10.3) scales with G:
```
Generation KV cache: M_kv_cache with batch = G × num_prompts
```

Where G is typically 4-16 completions per prompt.

**GRPO generation feasibility**: The `max_batch_gen` formula from Section 10.3 constrains the effective group size. If `G × num_prompts_per_batch > max_batch_gen`, the generation phase must be split into multiple rounds, increasing wall-clock time. The calculator should warn when G exceeds `max_batch_gen / num_prompts_per_batch` and estimate the resulting slowdown. Use the `T_generation` formulas from Section 10.3 to estimate GRPO generation wall-clock time with `batch_gen = G × num_prompts_per_batch`.

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

Quick Mode needs to infer architecture details (for activation memory, KV cache, etc.) from just a parameter count. Use this lookup table to estimate heads (a) and layers (L), then solve for hidden dimension (d):

| Param Range | a (heads) | L (layers) |
|------------|-----------|------------|
| < 5B       | 32        | 24         |
| 5B - 10B   | 32        | 32         |
| 10B - 24B  | 40        | 40         |
| 24B - 55B  | 64        | 48         |
| >= 55B     | 64        | 80         |

Then solve for d from the standard parameter formula `Psi = 12 * L * d^2` (Section 3.2):
```
d = sqrt(Psi / (12 * L))
```
Round d to the nearest multiple of 128 (common alignment in real architectures). Set d_ff = round(8/3 * d) (SwiGLU default), a_kv = a (assume MHA), V = 32,000. This gives a reasonable architecture for activation memory and parallelism constraint calculations. The simplification introduces ~5-10% error in memory estimates compared to real architectures, which is acceptable for Quick Mode.

**Detailed Mode**: User specifies full architecture (d, L, a, a_kv, d_ff, V) + all training config → precise breakdown.

**Preset Mode**: User selects from model presets (Section 3.3) → auto-fills architecture.

### 11.2 Pretraining Calculator Features

**Inputs:**
1. Model specification (preset, quick, or detailed)
2. Dataset size D (tokens) — show Chinchilla-optimal recommendation
2a. Unique tokens U (optional, defaults to D) — for data repetition analysis (Section 4.5)
3. Training precision (fp32 / bf16 / fp16 / fp8)
4. Optimizer (AdamW / AdamW 8-bit / Adam-mini / SGD+momentum / Adafactor / LAMB)
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
9. Estimated training time (days/hours), with failure-adjusted time shown alongside when N_gpu >= 256 (Section 6.5)
10. Estimated tokens/second throughput
11. Estimated cost breakdown: compute cost, checkpoint storage cost, failure overhead cost, and total (Section 8)
12. Global batch size (computed: b × G × N_dp)
13. Checkpoint size (12Ψ bytes for AdamW -- see Section 5.1) for storage planning
14. Attention overhead percentage (12Lds / 6Ψ -- see Section 4.1) to flag long-context cost
15. Predicted training loss (from Chinchilla parametric formula -- see Section 4.3) with caveat on accuracy at extreme over-training ratios
16. Maximum micro-batch size (computed from free GPU memory after model states: `b_max = floor(free_memory / bytes_per_sequence)` where `bytes_per_sequence` is the per-sequence activation cost)
17. Data repetition analysis (when U < D): epochs, data utilization warning, effective data ceiling (Section 4.5)
18. MoE sparsity metrics (when MoE model is selected): sparsity ratio (`Ψ_active / Ψ_total`), efficiency gain (`Ψ_total / Ψ_active`), and load balance overhead applied (Section 4.1) -- helps users understand the compute vs. memory tradeoff
19. Batch size efficiency: B vs B_crit comparison and compute overhead percentage from C_min formula (Section 4.4) -- e.g., "Your batch size is 2x B_crit: ~67% compute overhead, ~33% faster than optimal"

### 11.3 Post-Training Calculator Features

**Inputs:**
1. Base model (preset or parameter count)
2. Method: SFT / DPO / PPO / GRPO
3. Fine-tuning approach: Full / LoRA / QLoRA / MeZO
4. LoRA config (if applicable): rank r, alpha, target modules
4a. Trainable parameter percentage (for partial layer freezing beyond LoRA, e.g., "train only the last N layers"). Defaults to 100% for full fine-tuning, computed automatically for LoRA/QLoRA. Affects gradient and optimizer memory proportionally: only the trainable fraction incurs gradient (β_grad bytes/param) and optimizer state (12 bytes/param) costs.
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
- Unique tokens U: must be ≤ D, must be positive; warn when D/U > 4 (diminishing returns) or D/U > 40 (wasteful)

### Edge Cases
- Model too large for ANY single GPU → must use TP or ZeRO-3
- Activations dominate memory (very long sequences, large batch)
- Pipeline bubble > 50% → warn user to increase micro-batches
- ZeRO-3 communication overhead makes training slower → warn
- QLoRA base model quantized to 4-bit but activations still bf16
- PPO with all 4 models same size → needs 36× parameter bytes
- MoE models: total params ≠ active params — use Ψ_active for compute (Section 3.4, 4.1) but Ψ_total for memory (Section 5.1); Expert Parallelism may be needed
- Data repetition: when epochs > 4 (D/U > 4), warn about diminishing returns; when epochs > 40, warn that additional training is essentially wasted compute (Section 4.5)

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

### Test 5: ZeRO Paper Table I Validation (Rajbhandari et al., 2020)
```
Input: Ψ=7.5B, AdamW mixed precision (Φ=16, K_opt=12, bf16 grads), N_dp=64

ZeRO-1: (2+2)×7.5e9 + 12×7.5e9/64 = 30.0B + 1.41B = 31.4 GB → paper: 31.4 GB ✓
ZeRO-2: 2×7.5e9 + (2+12)×7.5e9/64 = 15.0B + 1.64B = 16.6 GB → paper: 16.6 GB ✓
ZeRO-3: 16×7.5e9/64 = 1.88 GB                                 → paper: 1.9 GB  ✓

Input: Ψ=1T, AdamW mixed precision (Φ=16, bf16 grads), N_dp=1024
ZeRO-3: 16×1e12/1024 = 15.6 GB                                → paper: 15.6 GB ✓
```
These cases validate that the ZeRO formulas produce results matching the original paper's Table I.

### External Benchmarks (for validation)

These real-world measurements from karpathy/llm.c can be used to validate the calculator's throughput and cost estimates:
```
GPT-2 124M on 8xA100-80GB:  ~300ms/step, ~94 min for 10B tokens
GPT-2 1558M on 8xH100-SXM:  ~2.8s/step, ~24 hrs for 32B tokens
Reference cost rates: 8xA100 ~ $14/hr, 8xH100 ~ $28/hr
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
