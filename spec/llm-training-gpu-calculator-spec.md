# LLM Training GPU Calculator — Full Implementation Specification

You are building an interactive **GPU Requirement Calculator for LLM Training** — a tool that estimates the number of GPUs, memory breakdown, training time, recommended parallelism strategy, and cloud cost for training large language models. It covers two phases: **Pretraining** and **Post-Training** (no inference). This will be added to an existing Next.js portfolio site as an interactive tool.

Your job: produce a **detailed, step-by-step implementation plan** for this tool based on the specification below.

## Reference Documents

- **Competitive analysis**: The attached document lists features from existing open-source GPU calculators. Use it to identify gaps, baseline features, and opportunities to exceed the state of the art.
- **This specification**: Contains all domain formulas, GPU specs, and codebase patterns. Treat it as the source of truth — do not hallucinate or invent formulas beyond what's here.

---

## 1. Portfolio Tech Stack & Architecture

### Stack
- Use the **latest stable versions at implementation time** for **Next.js**, **React**, **React DOM**, and **TypeScript** (strict mode)
- Use the **latest stable version at implementation time** of **Tailwind CSS 4** with OKLch color system (CSS variables in `app/globals.css`)
- Use the **latest stable version at implementation time** of **Framer Motion** for animations
- Use the **latest stable version at implementation time** of **next-themes** for dark/light mode (`.dark` class on `<html>`)
- Use the **latest stable version at implementation time** of **Lucide React** for icons
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
| B_seq | Global batch size in sequences/samples = b × G × N_dp |
| B_tok | Global batch size in tokens = B_seq × s = b × s × G × N_dp |
| G | Gradient accumulation steps |
| E | Number of experts (MoE models) |
| topk | Experts activated per token (MoE models) |
| N_dp | Data parallel degree |
| N_tp | Tensor parallel degree |
| N_cp | Context parallel degree |
| N_pp | Pipeline parallel degree |
| N_ep | Expert parallel degree (MoE models) |
| N_edp | Expert data parallel degree = N_dp × N_tp × N_cp / N_ep (MoE + ZeRO; Section 5.2) |
| E_s | Number of shared (always-active) experts (MoE models; 0 for most architectures) |
| N_sp | Sequence parallel degree (= N_tp when enabled; see Section 5.3) |
| N_gpu | Total GPUs / world size. Dense: N_dp × N_tp × N_cp × N_pp. MoE with EP: N_dp × N_tp × N_cp × N_pp × N_ep. Set N_cp = 1 and N_ep = 1 when unused. |
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

GQA validity constraint: `a` must be evenly divisible by `a_kv` so each KV head serves an integer number of query heads. Configurations such as `a=12, a_kv=8` are invalid even though both values are positive. Quick/detailed mode should warn critically and avoid treating fractional GQA groups as valid.

If the user sets an explicit per-head projection width `d_head`, it must be positive and finite. Invalid explicit head dimensions are configuration errors; the calculator should not silently fall back to `d / a` because that hides malformed PaLM-style architecture inputs.

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

When MoE is enabled, `E`, `topk`, and `L_moe` must be positive integers, `topk <= E`, `L_moe <= L`, `E_s` must be a non-negative integer, and `load_balance_factor >= 1`. Invalid MoE architecture inputs should be treated as configuration errors rather than clamped into a dense-model fallback.

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

**Shared experts**: Some MoE architectures (DeepSeek-v2/v3, Snowflake Arctic) include `E_s` shared experts that are always active for every token alongside the `topk` routed experts. Shared experts are **replicated on all EP ranks**, not distributed, because every token must access them. The per-GPU expert count with shared experts is:
```
Routed experts per GPU = E / N_ep
Total expert params per GPU = (E / N_ep + E_s) × Ψ_ffn
```
NOT `(E + E_s) / N_ep`. When shared experts are present, `Ψ_active` also increases: replace `topk × Ψ_ffn` with `(topk + E_s) × Ψ_ffn` in the active parameter formula.

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
- **Rule of thumb**: `6ΨD` is accurate when `d > s/12`. This condition holds for most large models at standard context lengths (e.g., 175B at s=2048 has <3% from quadratic terms; at s=4096 it rises to ~5%).
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

When a preset provides calibrated total and active parameter counts, the load-balance multiplier must use the calibrated active routed-expert parameters (`L_moe × topk × Ψ_ffn_expert` after active-count scaling), not the all-routed-expert memory parameter count. Shared experts are always active and are not load-balanced routed capacity.

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
FFN (SwiGLU):        6 × d × d_ff     (3 linear layers; = 16d² when d_ff = 8d/3 [LLaMA], = 24d² when d_ff = 4d [PaLM])
```

**GQA FLOPs impact**: For GQA models, the QKV cost drops significantly. For example, LLaMA 2 70B (a_kv/a = 8/64 = 1/8) has QKV FLOPs of 2.5d^2 per token instead of 6d^2. This reduces total per-layer FLOPs by ~15% compared to MHA.

```
Total per layer per token (standard MHA):  24d² + 4sd + 3as
Total per layer per token (SwiGLU + GQA):  2d²(1 + 2a_kv/a) + 4sd + 3as + 2d² + 3 × 2 × d × d_ff
```

The `3as` softmax term is from the DeepMind/Chinchilla method (Hoffmann et al., 2022, Appendix F). It is negligible for typical configs (<0.01% of layer FLOPs) and can be dropped in practice. The simplified `C = 6ΨD` formula follows Kaplan et al. (2020) and omits both the softmax and the sequence-dependent attention terms.

For the simplified formula `C = 6ΨD`, GQA is already accounted for via the reduced parameter count Ψ.

Full model forward, B_tok tokens:
```
C_fwd = B_tok × L × (per-layer FLOPs) + 2B_tok dV  (output projection / lm_head; embedding lookup is 0 FLOPs — pure memory index)
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
Where N is model parameters, D is training tokens. N is the logical trainable model size for scaling-law purposes: use the unpadded dense parameter count before tensor-parallel vocabulary padding, and use active parameters for MoE models. Implementation padding affects FLOPs, memory, and checkpoint storage, but should not change the model-size term in the scaling-law loss. The calculator should use the **corrected Epoch AI coefficients** (Besiroglu et al., 2024, arXiv:2404.10102), which fix a convergence failure in the original Chinchilla Approach 3 fitting:
```
Corrected (Epoch AI):  alpha = 0.3478,  beta = 0.3658,  A = 482.01,  B = 2085.43,  E = 1.8172
Original (Hoffmann):   alpha = 0.34,    beta = 0.28,    A = 406.4,   B = 410.7,    E = 1.69
```
The original published coefficients have a known fitting error: the L-BFGS optimizer stopped before convergence, and the rounded parameters introduce substantial bias. The corrected coefficients achieve lower loss for 90% of the original training observations, and a likelihood ratio test rejects the original fit at p < 10^-135. Critically, the corrected exponents (alpha ~= beta ~= 0.35) restore consistency with Chinchilla Approaches 1 and 2 (both of which find near-equal ~0.5 scaling exponents for N and D), whereas the original coefficients (alpha=0.34, beta=0.28) produce a spurious asymmetry that overpredicts the data term's contribution.

The three terms represent: irreducible loss (E), underfitting from model size (A/N^alpha), and underfitting from data (B/D^beta). All loss values (both Kaplan and Chinchilla) are in **nats** (natural log base); to convert to bits, divide by ln(2) ≈ 0.693. The calculator should use this formula to display **predicted training loss** for the user's chosen (N, D) combination, labeled in nats. **Note**: The loss prediction is calibrated to the MassiveText dataset. While the scaling exponents (alpha, beta) generalize across datasets (validated on C4 and GitHub), the absolute loss values (A, B, E) are dataset-specific. Loss predictions should be presented as relative guidance, not absolute targets for a user's specific data mix.

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
With the corrected Epoch AI coefficients (alpha=0.3478, beta=0.3658), the allocation exponents become:
```
a = beta/(alpha+beta) = 0.3658/0.7136 = 0.513  (for N_opt ~ C^a)
b = alpha/(alpha+beta) = 0.3478/0.7136 = 0.487  (for D_opt ~ C^b)
```
These near-equal exponents (~0.5/~0.5) are consistent with Chinchilla Approaches 1 and 2 (which found a=0.50, b=0.50 and a=0.49, b=0.51 respectively). With the original coefficients (alpha=0.34, beta=0.28), the exponents were a=0.45, b=0.55 -- a spurious asymmetry that suggested allocating more compute to data than to model size. The corrected coefficients confirm the D/N ratio grows only slowly with compute budget, and the 20x rule is approximately correct across a wide range of scales. The calculator can use these formulas to give a more precise Chinchilla-optimal recommendation when the user specifies a compute budget or GPU-hours target.

#### Coefficient Sensitivity Caveat

The fitted coefficients vary significantly with the training regime used to fit them. Sardana et al. (2024, "Beyond Chinchilla-Optimal") show:

| Data Range (tok/param) | alpha | beta | A | B | E |
|---|---|---|---|---|---|
| <= 100 | 0.08 | 0.13 | 7.199 | 25.97 | 0.17 |
| <= 250 | 0.13 | 0.16 | 14.23 | 39.54 | 0.98 |
| <= 500 | 0.13 | 0.16 | 17.07 | 35.80 | 0.95 |
| All Data (up to 10,000x) | 0.18 | 0.24 | 33.66 | 138.9 | 1.45 |
| Chinchilla (corrected Epoch AI, ~20x) | 0.3478 | 0.3658 | 482.01 | 2085.43 | 1.8172 |
| Chinchilla (original published, ~20x) | 0.34 | 0.28 | 406.4 | 410.7 | 1.69 |

The coefficients shift substantially depending on which token/parameter ratio range the fitting data covers. The Chinchilla coefficients were fit on runs near the compute-optimal frontier (~20x). At moderate overtraining (<=100x), alpha and beta are roughly half the Chinchilla values; the loss curve is flatter but still monotonically decreasing out to 10,000x with no observed data saturation (Sardana et al. tested 47 models from 150M-6B params). The calculator should select the row matching the user's actual training D/N ratio for loss prediction, defaulting to the "All Data" row when the ratio exceeds 500x. If data is repeated (Section 4.5), the effective-token cap applies only to the `B / D^beta` data term; it must not reclassify the run into a lower D/N coefficient regime. At extreme over-training ratios (like LLaMA 3's 1875x), the original Chinchilla coefficients overestimate the benefit of additional data and underestimate achievable loss. The calculator should present loss predictions as estimates, not ground truth, and note reduced accuracy at D/N ratios far from 20x.

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

Epoch AI (Besiroglu et al., 2024, arXiv:2404.10102) independently replicated the Chinchilla result, finding a compute-optimal ratio of 25.6:1 -- slightly above 20:1 and consistent with the power-law fit in this spec (which predicts the ratio increases with scale above 1B parameters). Critically, they also identified and corrected the Approach 3 fitting error (see Loss Prediction Formula above), providing the corrected coefficients this calculator uses. The corrected coefficients are the basis for the scaling exponent comparison across all three Chinchilla approaches:

| Approach | a (N_opt ~ C^a) | b (D_opt ~ C^b) |
|---------|-----------------|-----------------|
| 1. Min over training curves | 0.50 (CI: 0.488-0.502) | 0.50 (CI: 0.501-0.512) |
| 2. IsoFLOP profiles | 0.49 (CI: 0.462-0.534) | 0.51 (CI: 0.483-0.529) |
| 3. Parametric loss (corrected) | 0.513 | 0.487 |
| 3. Parametric loss (original published) | 0.46 | 0.54 |
| Kaplan et al. | 0.73 | 0.27 |

All three Chinchilla approaches now agree on near-equal (~0.5/~0.5) scaling, confirming the corrected coefficients as the best available baseline.

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

The **critical batch size** B_crit is the batch size in tokens (`B_tok`) at which training is equally efficient in terms of compute and time. It was introduced by McCandlish et al. (2018) and quantified for language models by Kaplan et al. (2020).

```
B_crit(L) = B_star / L^(1/alpha_B)
```

Where L is the training loss and the fitted coefficients are:
```
Kaplan et al.: B_star = 2.0 × 10^8 tokens, alpha_B = 0.21 (exponent 1/alpha_B ≈ 4.76)
```

**What it means practically:**
- **B_tok < B_crit**: Training is *time-inefficient* -- you could train faster with larger batches without meaningful compute waste. The gradient signal-to-noise ratio is low, so each step's update is noisy relative to its cost.
- **B_tok > B_crit**: Training is *compute-inefficient* -- larger batches give diminishing returns. You are spending more FLOPs per unit of loss reduction than necessary.
- **B_tok ≈ B_crit**: The sweet spot -- near-optimal trade-off between wall-clock time and total compute.

The critical batch size grows as loss decreases (i.e., as training progresses or as models get better). For a well-trained large model at low loss, B_crit can be in the millions of tokens. Plugging into the formula above: at loss L=3.5, B_crit ≈ 460K tokens; at L=3.0, B_crit ≈ 1M tokens; at L=2.5, B_crit ≈ 2.4M tokens; at L=2.0, B_crit ≈ 7.4M tokens.

**Compute efficiency at non-optimal batch size** (Kaplan et al., 2020): When training at token batch size `B_tok != B_crit`, the minimum compute C_min that would achieve the same result at the optimal batch size is:
```
C_min = C / (1 + B_tok / B_crit(L))
```
The dual formula for minimum training steps:
```
S_min = S / (1 + B_crit(L) / B_tok)
```
At `B_tok = B_crit`, `C_min = C/2`. There are two useful derived views, and the calculator should label them separately:
- **Compute multiplier above optimum**: `C / C_min = 1 + B_tok / B_crit`
- **Wasted-compute fraction of the actual run**: `1 - C_min / C = B_tok / (B_tok + B_crit)`

For example, at `B_tok = 10 × B_crit`, the run uses `11×` the theoretical minimum compute and ~91% of the actual compute is overhead. At `B_tok = B_crit / 10`, the compute overhead above optimum is only 10%, but training takes ~11× more steps than the minimum-step schedule.

**Calculator use**: Given the user's chosen token batch size `B_tok = b × s × G × N_dp` and the predicted training loss from Section 4.3, the calculator should display: (1) whether `B_tok` is above or below `B_crit`, (2) the compute multiplier above optimum, and (3) the wasted-compute fraction. For example: "Your token batch of 4M is 2x B_crit -- you are using 3x the theoretical minimum compute, and 67% of the compute in this run is overhead relative to the optimum." This is advisory only -- many practical constraints (memory, hardware utilization, training stability) override the theoretical optimum.

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
| AdamW mixed | 16-18 | 2 + β_grad + 4 + 4 + 4 (bf16/fp32 grads) |
| AdamW mixed (bf16 optimizer states) | 12-14 | 2 + β_grad + 4 + 2 + 2 (used by DeepSeek-v3) |
| AdamW mixed (no master weights) | 12-14 | 2 + β_grad + 4 + 4 (update bf16 params directly; used by llm.c) |
| AdamW FP8 mixed precision | 14 | 1+1+4+4+4 = 14 |
| AdamW + 8-bit states (bitsandbytes) | 10-12 | 2 + β_grad + 4 + 1 + 1 (fp32 master + int8 m + int8 v) |
| SGD + momentum (mixed) | 12-14 | 2 + β_grad + 4 + 4 |
| SGD (no momentum, mixed) | 8-10 | 2 + β_grad + 4 |
| Adafactor | 12-14 | 2 + β_grad + 4 + 4 (row+col factors instead of full m,v) |
| Lion (mixed) | 12-14 | 2 + β_grad + 4 + 4 (momentum only, no variance term) |
| Adam-mini (mixed) | 10-12 | 2 + β_grad + 4 + ~2 (block-diagonal Hessian reduces momentum to ~2 bytes/param) |
| LAMB (mixed) | 16-18 | Same as AdamW (m + v + master weights); used for large-batch pretraining |
| MeZO (zeroth-order) | 2 | 2+0+0 (forward-pass only; no gradients or optimizer states stored) |

**Adam-mini note**: Adam-mini (Zhang et al., 2024) exploits the block-diagonal structure of the Hessian in transformers to use a single learning rate per parameter block instead of per-parameter second moments. This reduces optimizer state memory by ~45-50% compared to AdamW while maintaining comparable training quality. It is production-ready and a good default recommendation when memory is tight but AdamW-level convergence is desired.

**MeZO note**: MeZO (Malladi et al., 2023) is a zeroth-order optimizer that estimates gradients via forward-pass perturbation, eliminating all gradient and optimizer state storage. At only 2 bytes/param (the model weights in bf16), it enables fine-tuning models ~10x larger than standard optimizers on the same hardware (e.g., fine-tuning a 30B model on a single A100 vs. ~3B with AdamW). However, MeZO is **fine-tuning only** -- it is not suitable for pretraining due to slow convergence. The calculator should offer MeZO only in the post-training section (Section 10) and grey it out for pretraining.

**FP8 training note**: The 14 bytes/param row above assumes parameters and gradients are explicitly stored in fp8 format (1 byte each), as with Microsoft's MS-AMP backend. However, the most common FP8 implementation -- NVIDIA TransformerEngine in its native mode -- does **not** reduce memory: the model remains in bf16/fp32 in memory, and FP8 is used only inside compute kernels (matmuls). In this mode, memory consumption is identical to bf16 mixed precision (16-18 bytes/param). Only specialized backends like MS-AMP that actually store weight and gradient tensors in fp8 achieve the 14 bytes/param figure. The calculator should default FP8 to **no memory savings** (same as bf16 mixed precision) and offer an "FP8 weight storage" toggle for the 14 bytes/param mode. The primary benefit of FP8 is compute throughput (2x FLOPS on supported hardware), not memory reduction.

**Checkpoint (storage) size**: Training checkpoints saved to disk contain persistent restart state, not live gradients. For default mixed-precision AdamW, common framework checkpoints persist the model parameter tensor plus fp32 master weights and Adam moments:
```
Checkpoint size = 14 × Ψ bytes  (2 + 4 + 4 + 4 per parameter)
```
This is distinct from live training memory (16-18 bytes/param) because gradients are recomputed on resume and are not persisted. For all optimizer variants, count the resolved parameter tensor plus optimizer states (`parameterBytes + K_opt`) and omit live gradients. PyTorch checkpoint files include metadata overhead of ~3-5% above the theoretical size. The calculator should display checkpoint size with a 1.04x file-overhead factor for storage planning (e.g., LLaMA 7B tensor payload = 14 x 6.7B = 93.8 GB, displayed planning size ≈ 97.6 GB per save).

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

**Context/sequence parallelism and optimizer sharding interaction**: Tensor parallelism already partitions dense weights across `N_tp`, so Megatron-style sequence parallelism (`N_sp = N_tp`) should not add another optimizer-state shard factor. It partitions activation regions along the sequence dimension, not an independent replica of dense weights. Context parallelism (`N_cp`) is different: it shards tokens while leaving dense weights duplicated across CP ranks, and Megatron folds CP ranks into the DP communication group for dense weight gradients and distributed optimizer state. Therefore, when `N_cp > 1`, replace the ZeRO/FSDP state shard degree `N_dp` with `N_dp × N_cp`. Do **not** multiply model-state sharding by `N_tp` merely because sequence parallelism is enabled. Subramanian et al.'s `seqp` optimizer divisor applies to their separate 2D sequence/context axis with replicated weights; in this calculator that role is represented by `N_cp`, not the Megatron sequence-parallel toggle.

**Parameter divisibility**: ZeRO requires the sharded parameter groups to be evenly divisible by the effective state shard degree for clean sharding. For dense parameters this is `N_dp × N_cp`; for expert parameters it is `N_edp`. Some frameworks (e.g., llm.c) silently disable ZeRO if the parameter count is not divisible by the shard degree; others pad parameters automatically. The calculator should warn when the parameter count is not evenly divisible by the effective state shard degree.

**FSDP-to-ZeRO equivalence**: PyTorch FSDP (Fully Sharded Data Parallel) implements the same sharding strategies as DeepSpeed ZeRO under different names. The calculator should accept either terminology:

| FSDP Strategy | ZeRO Equivalent | What is Sharded |
|---|---|---|
| NO_SHARD | ZeRO-0 (DDP) | Nothing |
| — (no native equivalent) | ZeRO Stage 1 | Optimizer states only (use DeepSpeed) |
| SHARD_GRAD_OP | ZeRO Stage 2 | Optimizer states + gradients |
| FULL_SHARD | ZeRO Stage 3 | Optimizer states + gradients + parameters |
| HYBRID_SHARD | ZeRO++ Stage 3 | Everything within node; replicated across nodes |
| HYBRID_SHARD_ZERO2 | ZeRO++ Stage 2 | Optimizer + gradients within node; replicated across nodes |

**FSDP mixed precision memory model**: FSDP handles mixed precision differently from standard AMP. In standard AMP, both the fp32 master weights and the low-precision working copy coexist in GPU memory, costing `(K_full + K_low) x Psi` bytes. FSDP instead keeps local shards at full precision and materializes the unsharded (all-gathered) parameters transiently in low precision:
```
M_fsdp_mixed_peak = K_full x (Psi / F) + 2 x K_low x max(Psi_fsdp_unit)
```
Where `F` is the sharding factor (number of FSDP ranks, typically `N_dp`), `K_full` is bytes per parameter at full precision (e.g., 4 for fp32), `K_low` is bytes at reduced precision (e.g., 2 for bf16), and `max(Psi_fsdp_unit)` is the parameter count of the largest FSDP wrapping unit (see "FSDP wrapping granularity" note in Section 5.4). The factor of 2 reflects PyTorch FSDP's AllGather rate limiter, which permits up to two unsharded wrapping units in flight at peak. This saves memory vs standard AMP because FSDP materializes only a small number of full low-precision units transiently, rather than keeping a persistent low-precision copy of the entire model. The calculator should use this formula when FSDP + mixed precision is selected, instead of the standard AMP memory model.

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

**DeepSpeed initialization memory spike**: DeepSpeed can create flat fp32 parameter buffers during model preparation before ZeRO sharding is fully applied. The transient GPU peak is therefore `4 × Ψ_local_before_ZeRO` bytes above steady-state, where `Ψ_local_before_ZeRO` is the rank-local model partition after TP/PP/EP placement but before ZeRO/FSDP state sharding. For pure data parallelism this is `4 × Ψ`, not `4 × Ψ / N_dp`. This spike occurs only during initialization and is released once sharding completes. PyTorch FSDP without mixed precision can operate entirely in bf16 (no fp32 upcast), avoiding this spike. The calculator should note this transient cost for DeepSpeed users and recommend ZeRO-init/partitioned initialization when steady-state fits but initialization would OOM.

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
`AIT_opt = 1024×4/4 = 1024 FLOPS/byte`. `efficiency = (1024 × 12e9) / (1024 × 12e9 + 70e12) = 12.3e12 / 82.3e12 ≈ 15%`. Single-GPU offloading is heavily bottlenecked by PCIe bandwidth. Offload efficiency improves with larger batch×sequence (higher AIT), faster PCIe/CXL links, or better effective host-link bandwidth per GPU.

Use a per-GPU bandwidth and per-GPU FLOPS pair when estimating offload efficiency. Aggregate PCIe bandwidth alone should not be compared to a single GPU's compute rate: if both compute and offload traffic scale with the number of identical GPUs, the efficiency ratio is unchanged unless effective host-link bandwidth per GPU improves. The calculator should display this per-GPU planning estimate when offloading is enabled to set realistic throughput expectations.

**MoE + ZeRO interaction**: When combining ZeRO with Expert Parallelism, expert (MLP) parameters and non-expert (attention, layernorm) parameters use different sharding denominators because EP already distributes experts across GPUs:
```
Non-expert params: sharded across N_dp × N_cp GPUs (standard ZeRO/FSDP replica group)
Expert params:     sharded across N_edp GPUs, where N_edp = N_dp × N_cp × N_tp / N_ep
```
Router weights are non-expert parameters for this purpose: they are not expert MLP weights and should use the standard non-expert ZeRO/FSDP replica group. The `N_tp` factor appears because within each TP group, each GPU holds a shard of the same expert weights, so TP ranks sharing an expert also participate in expert data parallelism. The `N_cp` factor appears because context-parallel ranks duplicate weights. For example, with N_dp=32, N_cp=1, N_tp=2, and N_ep=8: attention weights are sharded 32-way (N_dp × N_cp), but each expert's MLP is sharded 8-way (N_edp = 32×1×2/8 = 8). The calculator should apply ZeRO formulas separately to expert and non-expert parameter groups when both EP and ZeRO are active (Zhang & Su, 2025).

### 5.3 Activation Memory

Activations are intermediate values stored during forward pass for use in backward pass.

**Per transformer layer** (Korthikanti et al., 2022):

No checkpointing (store everything):
```
M_act_layer = s × b × d × (34 + 5 × a × s / d) bytes
```
The constant 34 = 11 (attention: Q,K,V,softmax output, attention dropout, attention output projection, two layernorm inputs/outputs) + 19 (MLP: up-projection input/output, down-projection input/output, activation function, dropout) + 4 (two LayerNorm: 2 norms x 2 tensors each). The `5as/d` term is the attention score matrix (s x s per head, stored for Q*K^T, softmax, and dropout).

With tensor parallelism (N_tp > 1, no checkpointing, **sequence parallelism disabled**) the 34 coefficient decomposes into TP-split and non-TP-split components (Korthikanti et al. 2022, Equation 2):
```
M_act_layer = s × b × d × (10 + 24/N_tp + 5 × a × s / (d × N_tp)) bytes
```
The **10sbd** term covers activations that are replicated across all TP ranks (not split): 2 LayerNorm inputs (4sbd in fp16), QKV shared input before TP split (2sbd), first MLP linear input before TP split (2sbd), attention dropout mask (sbd at 1 byte/elem), and MLP dropout mask (sbd at 1 byte/elem). These operations occur outside the TP-sharded regions (after all-reduce outputs or before TP splits) and cannot be partitioned. The **24sbd/N_tp** term covers activations inside the TP-sharded attention and MLP blocks: Q, K, V, output projection input (8sbd), GeLU input and second MLP linear input (16sbd for d_ff=4d). The **5as²b/(d×N_tp)** attention score term is also split because attention heads are distributed across TP ranks (a/N_tp heads per rank).

Full activation checkpointing (recompute each layer):
```
M_act_layer = 2 × s × b × d bytes  (store only layer input)
```
Cost: ~33% more compute (recompute forward during backward)

**Transient recomputation working memory**: The `2 × s × b × d` figure above is the *stored* checkpoint memory. During the backward pass, when a checkpointed layer is recomputed, its full activations must be temporarily materialized in GPU memory. This transient working memory equals one layer's active non-checkpointed activation formula (`M_act_full_layer`, with the same TP/SP/GQA/d_ff/Flash/precision settings) and cannot be offloaded. For standard MHA with `d_ff=4d` and no TP:
```
M_recomp_working = M_act_full_layer
                 = s × b × d × (34 + 5 × a × s / d) bytes   (per-layer checkpointing, ci=1)
```
For checkpoint intervals spanning multiple layers (ci > 1, i.e., checkpointing every ci-th layer), the working memory scales with ci since all intermediate layers must be recomputed:
```
M_recomp_working = ci × M_act_full_layer
```
This working memory is transient (freed after each layer's backward completes) but sets a hard floor on per-GPU VRAM alongside the minimum GPU memory floor from Section 9. The calculator should include `M_recomp_working` (with ci=1 for per-layer full checkpointing) as part of the peak memory estimate when full activation checkpointing is selected.

Block-level partial recomputation (NeMo `recompute_method="block"` with `recompute_num_layers=N`): checkpoints the first N layers per pipeline stage fully, remaining layers store all activations:
```
M_activations_stage = N_recomp × (2 × s × b × d) + (L_per_stage - N_recomp) × M_act_full_layer
```
Where `M_act_full_layer` means the active non-full-checkpoint stored-activation formula for the same TP/SP/GQA/d_ff/Flash/precision setting. This is a practical intermediate that lets users recompute only as many layers as needed to fit in memory. The calculator should support this as a "partial" checkpointing option where the user specifies N_recomp.

When exact MoE layer positions inside a pipeline stage are not modeled, partial recomputation must take the conservative peak over feasible checkpointed dense/MoE layer counts rather than averaging dense and MoE activation costs. This prevents understating memory when the recomputed block covers cheaper dense layers while larger MoE layers still store full activations.

**Optimal checkpoint interval** (Narayanan et al., 2021): For interval-based checkpointing every `c` layers out of `l` layers per pipeline stage, total activation memory is approximately `(l/c) × A_input + c × A_intermediate`, where `A_input = 2sbd` (the stored checkpoint) and `A_intermediate` is the full per-layer activation memory needed during recompute. The memory-optimal interval is:
```
c_optimal = sqrt(l × (A_input / A_intermediate))
```
In practice, because `A_intermediate` is much larger than `A_input`, this yields checkpointing every 1-2 transformer layers as optimal for typical model sizes. This interval formula is related but not identical to NeMo's block recomputation control: `recompute_num_layers` / `N_recomp` is the number of layers per pipeline stage whose activations are fully checkpointed and recomputed. The calculator's Partial mode should treat `N_recomp` as that block count, default conservatively to 1 layer per stage, and let users increase it when more memory reduction is needed.

Selective activation checkpointing rematerializes the attention-score tensor but keeps the linear activations needed for backward. For the standard MHA, `d_ff=4d` case, use the following **stored activation** formulas:
```
M_act_layer = s × b × d × 34 bytes                                 (N_tp = 1)
M_act_layer = s × b × d × (10 + 24/N_tp) bytes                    (N_tp > 1, SP disabled)
M_act_layer = s × b × d × (34 / N_tp) bytes                       (N_tp > 1, SP enabled)
```

This ordering removes the ambiguity between checkpointing mode and TP/SP mode:
1. Pick the checkpointing mode (`none`, `selective`, `full`, or `partial`).
2. Pick the TP/SP layout. When `N_tp = 1`, use the dense formulas. When `N_tp > 1`, this calculator should assume **sequence parallelism is enabled by default** unless the user explicitly disables it in advanced settings.
3. Apply Flash Attention and AMP autocast corrections to the active formula rather than stacking independent formulas on top of each other.

**PyTorch AMP FP32 precision caveat**: The Korthikanti coefficients (34 for linear terms, 5 for attention) assume all activations are stored in the compute precision (bf16/fp16). Under PyTorch's `torch.cuda.amp.autocast`, two operations are promoted to FP32 for numerical stability: (1) **softmax** outputs are saved in FP32 (4 bytes instead of 2), adding an extra `b*a*s^2` bytes per layer (changing the attention coefficient from 5 to 6), and (2) **layer norm** inputs are saved in FP32, adding `2*s*b*d` bytes per layer (changing the linear coefficient from 34 to 36). The corrected formula under AMP autocast is:
```
M_act_layer = s × b × d × (36 + 6 × a × s / d) bytes  (PyTorch AMP autocast)
```
This was empirically validated against `torch.cuda.max_memory_allocated()` on GPT-2 small (A100), achieving 1.15% error (Rees, erees.dev). The difference is ~6% more activation memory than the Korthikanti formula predicts. Megatron-LM and frameworks that use explicit bf16 storage (not autocast) match the original coefficients (34, 5). The calculator should use the **Korthikanti coefficients (34, 5)** as the default (they match the widely-used Megatron-LM implementation) and offer an "AMP autocast" toggle that applies the corrected coefficients (36, 6) for users training with standard PyTorch AMP. When Flash Attention or selective checkpointing removes the attention-score tensor, only the LayerNorm correction remains, so AMP autocast adds `+2 × s × b × d` bytes to the active formula rather than reintroducing an `O(s^2)` term.

**GQA and d_ff correction for activation memory**: The dense-MHA `24` term in the TP/SP formulas is `8` attention-linear activations plus `16` FFN activations for a standard `d_ff = 4d` MLP. Do not replace the whole `24` with an FFN-only term. Instead decompose the linear coefficient:
```
attention_linear = 4 × (query_width / d) + 4 × (kv_width / d)
ffn_linear       = 4 × (d_ff / d)
```
For standard MHA, `query_width = kv_width = d`, so `attention_linear = 8`; for GQA/MQA, `kv_width = d × a_kv / a`, so K/V activation storage shrinks while Q/O storage remains full width. The no-checkpoint stored activation formula generalizes to:
```
M_act_layer = s × b × d × (10 + attention_linear + ffn_linear + 5 × a × s / d) bytes                 (N_tp = 1)
M_act_layer = s × b × d × (10 + (attention_linear + ffn_linear)/N_tp + 5 × a × s / (d × N_tp)) bytes  (N_tp > 1, SP disabled)
M_act_layer = s × b × d × ((10 + attention_linear + ffn_linear)/N_tp + 5 × a × s / (d × N_tp)) bytes  (N_tp > 1, SP enabled)
```
For Flash Attention or selective checkpointing, remove only the quadratic attention-score term. The calculator should use actual `d_ff`, query width, and KV width when available (Detailed/Preset modes) and fall back to heuristic Quick Mode values otherwise.

With Flash Attention (avoids materializing s×s attention matrix), remove the quadratic attention-score term from whichever stored-activation formula is active. For the standard MHA, `d_ff=4d` case, the formulas reduce to:
```
M_act_layer = s × b × d × 34 bytes                                 (N_tp = 1)
M_act_layer = s × b × d × (10 + 24/N_tp) bytes                    (N_tp > 1, SP disabled)
M_act_layer = s × b × d × (34 / N_tp) bytes                       (N_tp > 1, SP enabled)
```
Flash Attention still stores O(s) per-head statistics for the backward pass. The precise replacement for the `5as/d` term is `4 × a × s × b / N_tp` bytes (per head: one fp32 float per row storing the combined logsumexp statistic `L_i = m_i + log(l_i)`, where `m_i` is the row-max and `l_i` is the sum of exponentials; FlashAttention-1 stored these as two separate values but FlashAttention-2 merged them into a single scalar). For typical hidden dimensions this is negligible compared to the linear activation terms, but it grows with head count and sequence length.

**Flash Attention + selective checkpointing interaction**: Selective checkpointing and Flash Attention both remove the `O(s^2)` attention-score tensor from the stored-activation estimate. When both are enabled, use the Flash Attention formula above and do **not** apply any additional stored-activation reduction beyond it. Any remaining difference between the two modes is throughput, not peak activation memory.

**CPU activation offloading**: A third option beyond "store on GPU" and "recompute" is offloading activation tensors to CPU memory via PCIe. Both NeMo (`cpu_offloading_activations=True`) and PyTorch (`CheckpointPolicy.MUST_CPU_OFFLOAD`) support this. CPU offloading trades PCIe bandwidth for GPU memory, and is most effective when the recomputation cost exceeds the transfer time. The throughput overhead decreases with model hidden dimension because larger layers have higher arithmetic intensity, better overlapping transfer with compute (Rajbhandari et al., 2021): d=2K: ~25% slowdown, d=8K: ~10% slowdown, d=32K+: <2% slowdown. The calculator should note this option exists but treat it as an advanced toggle rather than a primary mode.

**Sequence parallelism** (Korthikanti et al., 2022): When used with tensor parallelism (N_sp = N_tp in Megatron-LM), LayerNorm and dropout activations are partitioned along the sequence dimension. This reduces the `10 × s × b × d` term (which covers LayerNorm inputs/outputs and dropout masks outside the TP-sharded regions) to `10 × s × b × d / N_tp`. The no-checkpointing formula with sequence parallelism becomes:
```
M_act_layer = s × b × d × (10/N_tp + 24/N_tp + 5 × a × s / (d × N_tp)) bytes
            = s × b × d × (34/N_tp + 5 × a × s / (d × N_tp)) bytes
```
Sequence parallelism is standard practice in Megatron-LM when TP is used and should be assumed enabled whenever `N_tp > 1`, unless the user explicitly disables it for framework-specific reasons.

**Context parallelism** (Meta, Llama 3 scaling): Context parallelism (CP) shards the input sequence along the sequence dimension across N_cp GPUs. Each CP rank processes `s/N_cp` tokens. In all activation memory formulas above, replace `s` with `s/N_cp` when CP is active. The quadratic attention term benefits most because the attention score matrix is O(s^2):
```
M_act_layer = (s/N_cp) × b × d × (34 + 5 × a × (s/N_cp) / d) bytes  [no checkpointing, with CP]
```
With Flash Attention the `5a(s/N_cp)/d` term disappears as usual. CP communication cost is an all-gather of KV tensors per layer (forward) and reduce-scatter of KV gradients (backward). Because communication is O(s) while attention compute is O(s^2), CP overhead shrinks with longer sequences, making it most effective at 32K+ sequence lengths. When to use CP: when sequence length causes activation memory pressure and the micro-batch size would otherwise drop to 1. The trigger heuristic is `B_seq / (N_gpu / (N_tp × N_pp)) <= 1` at long sequence lengths, indicating DP alone cannot maintain throughput. CP should only be enabled when `s/N_cp` still exceeds a minimum chunk size (~2K tokens) to maintain sufficient arithmetic intensity per rank.

**MoE activation memory**: The per-layer activation formulas above assume one dense FFN block per token on the local rank. In MoE layers, each token creates `topk` routed expert assignments, distributed across the expert-parallel ranks that own those experts. The local FFN activation scale is therefore `topk / N_ep` on average, not `topk / E` unless `N_ep = E` (one expert per EP rank). Apply the MoE load-balance factor to routed experts only; shared experts are present on every EP rank and are active for every token.
```
routed_scale = (topk / N_ep) × load_balance_factor
M_act_moe_layer = M_act_non_ffn + M_act_ffn_expert × (routed_scale + E_s)
```
Where `M_act_non_ffn` covers attention, LayerNorm, and dropout activations (the `10 × s × b × d` component plus attention-linear and `5as²b/d` attention-score terms), and `M_act_ffn_expert` covers one expert FFN/MLP activation footprint using the `ffn_linear = 4 × d_ff/d` coefficient above. For Mixtral 8x7B with no expert parallelism (`topk=2`, `N_ep=1`), each rank can hold activations for two routed expert calls per token. With `N_ep=8`, the average routed FFN activation per EP rank is 25% of a single expert FFN, before load-balance overhead. For dense layers in the same model (if L_dense > 0), use the standard formula unchanged. Note that this applies to *activation* memory only -- model states (parameters, gradients, optimizer states) must store all E experts regardless of sparsity unless Expert Parallelism shards them (Section 3.4).

When full activation checkpointing is used on an MoE layer, the expert block can be recomputed but the router dispatch mask must remain resident so backward uses the exact same expert assignment:
```
M_act_moe_full_checkpoint = 2 × s × b × d + 2 × b × s × topk bytes
```
Apply context parallelism by replacing `s` with `s / N_cp`. Shared experts do not change this dispatch-mask term; they are always active and are represented in the expert FFN activation term above when activations are not fully recomputed.

**Total activation memory:**
```
M_act_per_stage = L_dense_per_stage × M_act_layer + L_moe_per_stage × M_act_moe_layer
M_activations = M_act_per_stage × min(N_pp, num_microbatches)
```
Where `L_dense_per_stage` and `L_moe_per_stage` are the dense and MoE layers assigned to this pipeline stage (`L / N_pp` for uniform distribution). For pure dense models, this simplifies to `(L / N_pp) × M_act_layer × min(N_pp, num_microbatches)`.

**1F1B in-flight microbatch factor**: The `min(N_pp, num_microbatches)` term accounts for the number of microbatches whose activations must be simultaneously stored during the 1F1B pipeline schedule. In 1F1B, the pipeline ramps up by executing forward passes on successive microbatches before any backward pass begins. At steady state, each pipeline stage holds activations for `min(N_pp, num_microbatches)` microbatches (Subramanian et al., 2024; Narayanan et al., 2021). When `num_microbatches >= N_pp` (the normal case), this equals `N_pp`. When `num_microbatches < N_pp` (the pipeline is not fully filled), it equals `num_microbatches`. Without pipeline parallelism (`N_pp = 1`), this factor is 1 and the formula reduces to the non-PP case. This factor is critical for memory estimation -- for example, with PP=8, one stage stores activations for 8 concurrent microbatches, not 1.

**Note**: Each microbatch's activation memory is for ONE micro-batch (`b` sequences), not the full global batch.

**Output logits tensor**: The per-layer formulas above (Korthikanti et al.) cover only transformer layer activations. During loss computation, the full output logits tensor is materialized as a non-layer activation:
```
M_output_logits = b × (s / N_cp) × (V_padded / N_tp) × β bytes
```
Where `V_padded` is the tensor-parallel padded vocabulary size from Section 9. For `N_tp > 1`, this assumes Megatron-style vocab-parallel cross entropy, where each TP rank keeps only its vocabulary shard rather than all-gathering full logits. For `N_cp > 1`, each CP rank holds logits only for its local sequence shard. Frameworks that gather full logits before loss can require up to `N_tp × N_cp` more memory for this term. For large vocabularies this can exceed per-layer activation memory. Examples with `N_tp=N_cp=1`: LLaMA 7B (V=32K, b=4, s=2048, bf16) = ~0.5 GB; LLaMA 3 8B (V=128K, same config) = ~2 GB. The calculator should add this to the total activation memory. Note that **chunked cross-entropy loss** (used by Liger Kernel, HuggingFace, and others) avoids materializing the full logits tensor by fusing the projection and loss computation in vocabulary-sized chunks, effectively eliminating this cost. The calculator should include an option to disable this component when fused/chunked loss is enabled.

### 5.4 Temporary Buffers & Communication

Rough estimate (use as fallback when framework-specific bucket sizes are unknown):
```
M_temporary + M_communication ≈ 0.05 × (M_model_states + M_activations)  (5% overhead)
```

More precisely, allocate concrete buffer sizes used by DeepSpeed/Megatron:
- **DP gradient-reduction volume**: ~2Ψ × β bytes per step (ring all-reduce communication volume). This is a throughput term, **not** a resident VRAM buffer; do not add the full `2Ψ × β` to `M_communication` for memory estimates.
- **ZeRO allgather bucket**: 500M elements x β bytes (~1 GB in bf16, ~2 GB in fp32)
- **ZeRO-3 parameter prefetch**: During forward/backward, ZeRO-3 must allgather the full (unsharded) parameters of the current layer. The prefetch buffer holds one full transformer layer's unsharded weights:
  ```
  M_prefetch_fwd = max(Ψ_embedding, Ψ_largest_layer) × β
  M_prefetch_bwd ≈ (Ψ_largest_layer + min(Ψ_largest_layer, prefetch_bucket_size)) × β
  ```
  The backward term reflects the current materialized layer plus parameters fetched ahead, capped by DeepSpeed's `stage3_prefetch_bucket_size`. Raw DeepSpeed defaults use 50M prefetch elements; HuggingFace auto config commonly uses `0.9 × hidden_size²`. For example, LLaMA 70B has ~1.1B params/layer, so the current layer is ~2.2 GB in bf16; with a 50M-element prefetch bucket, the extra fetch-ahead residency is capped near 0.1 GB rather than another full layer.
- **FSDP AllGather rate limiter**: FSDP limits concurrent AllGather operations to at most 2 in flight at any time to prevent CUDA allocator over-allocation (without this limit, T5-11B sees up to 5x slowdown from `cudaMalloc` retries). This means the peak AllGather buffer memory is bounded by:
  ```
  M_allgather_peak = 2 x max(Psi_fsdp_unit) x K bytes
  ```
  Where `max(Psi_fsdp_unit)` is the largest FSDP wrapping unit's parameter count and `K` is bytes per parameter at the materialized precision. The calculator should use this as the AllGather buffer contribution for FSDP, replacing the backward prefetch formula above (which assumes 2 layers; the rate limiter is the actual constraint).
- **FSDP wrapping granularity**: The "largest layer" term in the prefetch buffer formulas is more precisely the largest FSDP wrapping unit, which users control. Wrapping at the transformer block level (default) makes each block one unit. Finer wrapping (e.g., per-attention/per-FFN) reduces the prefetch buffer and peak memory but increases the number of AllGather operations, reducing throughput. The calculator should default to per-transformer-block wrapping (i.e., `max(Psi_fsdp_unit) = Psi_largest_layer`) but note that users can trade throughput for memory by wrapping more finely.
- **Peak logit memory during loss backward**: At the start of the backward pass through the loss function, a new FP32 gradient tensor for the logits is allocated while the forward-pass logits (stored in mixed precision) still reside in memory. Both coexist briefly, creating a peak:
  ```
  M_logits_peak = M_output_logits + 4 × b × (s / N_cp) × (V_padded / N_tp)
  ```
  Where `M_output_logits` is from Section 5.3 and the second term is the FP32 gradient tensor for the local logits shard. For GPT-2 small (V=50,304, b=12, s=1024): the peak logit allocation alone is ~2.5 GB. For large vocabularies (V=128K+) this is the dominant temporary buffer. The calculator should use `M_logits_peak` (not just `M_output_logits`) when computing peak memory, and note that chunked cross-entropy loss (Section 5.3) eliminates this spike entirely.
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
For hidden_size=4096: reduce_bucket = 16.7M elements (~32 MB in bf16) vs the raw default of 500M elements (~1 GB in bf16). With HF Trainer auto-config, `M_overlap_comm` drops from ~9 GB to ~0.6 GB. The calculator should use `hidden_size^2` as the default bucket size (matching HF Trainer behavior) and allow users to override with raw DeepSpeed defaults if they are configuring DeepSpeed directly. The `param_persistence_threshold` default should be treated as an implementation assumption, not a custom memory-control input, unless per-module ZeRO-3 persistence is explicitly modeled. For ZeRO-3, the prefetch buffer formula above (Section 5.4) provides the dominant communication cost. With large vocabularies (128K+), add M_logits_peak to these estimates.

**torch.compile overhead**: When using `torch.compile` (increasingly common in PyTorch training), the compiler creates additional graph representations and optimized kernel caches that persist in GPU memory. Estimate approximately **10% of model weights** (`0.1 × Ψ × β` bytes) as additional overhead. The calculator should include this as an optional toggle (off by default).

### 5.5 Total Memory per GPU

```
M_gpu = M_model_states(ZeRO) + M_activations + M_temporary + M_communication + M_framework_overhead
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
Ψ_boundary_stage = max(Ψ_input_embedding + Ψ_positional, Ψ_output_projection + Ψ_final_norm)
Ψ_most_loaded_stage = Ψ_transformer_per_stage + Ψ_boundary_stage
Ψ_per_gpu = Ψ_most_loaded_stage / N_tp
```
For tied embeddings, the shared word embedding can participate in both the first and last pipeline stages, but the bottleneck stage adds one `V × d` boundary matrix, not two. For untied embeddings, the first stage carries the input embedding and the last stage carries the output projection, so the bottleneck adds the larger boundary side rather than their sum. For models with large vocabularies (V=128K+), this boundary can be significant -- e.g., LLaMA 3 8B's embedding is ~525M params, adding ~1 GB in bf16 to a boundary stage beyond a uniform `Ψ/N_pp` estimate.

**Embedding-aware PP partitioning**: When the embedding layer is comparable in size to a transformer block (common with large vocabularies), it can be treated as an equivalent pipeline stage for load balancing. This changes the divisibility constraint from `L % N_pp == 0` to `(L + 2) % N_pp == 0` (counting the input embedding and output projection as two additional "virtual layers"). For example, BLOOM-176B used this approach: with 70 transformer layers and PP=12, `70 % 12 != 0` but `(70 + 2) % 12 == 0`, enabling even partitioning by assigning the embedding/output layers as dedicated stages. The calculator should check both `L % N_pp == 0` and `(L + 2) % N_pp == 0` when validating PP configurations, and suggest the embedding-aware option when the standard constraint fails but the embedding-aware one passes. **Quantified benefit** (Meta, Llama 3 405B): Removing one transformer layer each from the first and last PP stages (to compensate for embedding/output head overhead) yielded 5 GB lower peak memory and 6.5% higher TFLOPs per GPU compared to uniform layer distribution, and eliminated the need for activation checkpointing at 8K sequence length.

```
M_params_per_gpu = Ψ_per_gpu × β  (model weights on bottleneck GPU)
M_activations_per_gpu = per-layer activation × layers_per_stage
```

PP overhead (pipeline bubble):
```
Bubble fraction = (N_pp - 1) / (num_microbatches + N_pp - 1)
```
**Convention note**: This formula gives the bubble as a fraction of **total wall-clock time** (idle time / total time), which is the correct metric for a calculator estimating training duration. Some references, including the original Megatron-LM paper (Narayanan et al., 2021), use `(p-1)/m` instead, which is the fraction of **ideal compute time** wasted. The two conventions diverge when microbatches are comparable to pipeline stages (e.g., p=8, m=16: this spec gives 30.4%, the paper convention gives 43.75%) but converge at large microbatch counts (p=8, m=64: 9.9% vs 10.9%).
**Hard minimum (1F1B schedule)**: The standard 1F1B (one-forward-one-backward) pipeline schedule requires `num_microbatches >= N_pp - 1`. Below this threshold the pipeline cannot be filled and the schedule fails. AFAB (all-forward-all-backward) has no such constraint but stores all micro-batch activations simultaneously, greatly increasing memory. The calculator should enforce `num_microbatches >= N_pp - 1` as a hard constraint when PP is active and warn the user if violated.

Rule of thumb: need num_microbatches >= 4 x N_pp to keep bubble < 20%.

**Higher efficiency thresholds** (validated by BLOOM-176B training): For 90% pipeline efficiency, need `num_microbatches >= 8 × N_pp`; for 94% efficiency, need `num_microbatches >= 16 × N_pp`. The 4x rule above is the minimum for acceptable efficiency; production training runs typically target 8-16x.

Interleaved (virtual pipeline) schedule: Megatron-LM supports splitting each pipeline stage into multiple virtual stages (VP chunks). This reduces the bubble at the cost of more in-flight microbatches:
```
Bubble fraction (interleaved) = (N_pp - 1) / (VP × num_microbatches + N_pp - 1)
```
Where VP = virtual_pipeline_chunks (typically 2-8). The bubble shrinks by a factor of ~VP compared to non-interleaved.

**Interleaved PP communication overhead**: The interleaved schedule increases PP point-to-point communication volume by a factor of VP, because each virtual stage boundary requires a send/receive operation. Total PP communication becomes `VP x s x b x d x beta` per pair of consecutive pipeline ranks per step (vs `s x b x d x beta` for non-interleaved). At large batch sizes where the pipeline bubble is already small, this added communication can negate the bubble reduction, making interleaved scheduling perform worse than standard 1F1B. The calculator should display the net effect: reduced bubble minus increased communication overhead.

**Interleaved microbatch divisibility constraint** (Narayanan et al., 2021): The interleaved schedule requires `num_microbatches % N_pp == 0` (num_microbatches must be evenly divisible by the pipeline parallel degree). This is a **stronger** constraint than the 1F1B minimum of `num_microbatches >= N_pp - 1`. The interleaved schedule assigns virtual stages in a round-robin pattern across pipeline ranks, so the total microbatch count must divide evenly to ensure each rank processes the same number of microbatches per virtual stage. The calculator should enforce this divisibility constraint when interleaved PP is selected and suggest rounding the microbatch count up to the nearest multiple of N_pp.

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
- C = total FLOPS from Section 4 — always use **ideal model FLOPs** (6ΨD or the PaLM formula). Do NOT adjust C for activation recomputation (see critical note below).
- F_peak = peak **dense matmul** FLOPS per GPU at training precision (BF16 for mixed precision; see Section 6.2 for precision guidance)
- MFU = Model FLOPS Utilization

**Critical: Do NOT adjust C for activation recomputation when using MFU.** MFU is defined relative to ideal model FLOPs (6ΨD) and already captures throughput loss from recomputation — a system using full activation checkpointing produces fewer tokens/sec (because it does more work per token), so its MFU is naturally lower. Adjusting C to 8ΨD while also using MFU would double-count the recomputation overhead. This is confirmed by the PaLM paper (Chowdhery et al., 2022, Appendix B), which defines MFU using "only the required operations to compute the forward+backward passes, and not rematerialization." The equivalent HFU-based formula (T = 8ΨD / (N_gpu × F_peak × HFU) for full recompute) gives the same answer because HFU > MFU by the recomputation factor.

**Implementation rule**: The calculator's default wall-clock estimate should use **one efficiency path only**:
1. Use ideal model FLOPs `C`.
2. Use `MFU` as the single throughput knob.
3. Choose or recommend an `MFU` value that already reflects the selected checkpointing, framework, and communication strategy.

Do **not** multiply the final time estimate by an additional activation-checkpointing slowdown on top of MFU. The empirical checkpointing overhead numbers below are calibration guidance for choosing MFU defaults or for user override, not extra multipliers to apply after the time formula.

For pipeline-parallel layouts, the default MFU should be adjusted before entering the time formula rather than applied as a separate runtime multiplier:
```
eta_pipeline = 1                                           [N_pp <= 1]
eta_pipeline = (VP_eff × num_microbatches) / (VP_eff × num_microbatches + N_pp - 1)
MFU_default_effective = MFU_tier_default × eta_pipeline × eta_recompute
```
Where `VP_eff = VP` for interleaved 1F1B and `VP_eff = 1` for non-interleaved or AFAB schedules. User-entered MFU overrides are treated as already end-to-end effective MFU and should not be schedule-adjusted again.

For default MFU only, apply a recomputation calibration factor so checkpointing affects wall-clock estimates through MFU rather than by inflating C:
```
eta_recompute = 1                                      [none]
eta_recompute = 1 / (1 + s/(6d))                       [selective, unless Flash Attention already removes attention scores]
eta_recompute = 1 / (1 + (N_recomp / L_stage)/3)       [partial block recomputation, capped at N_recomp <= L_stage]
eta_recompute = 0.75                                   [full recomputation; 6ΨD ideal / 8ΨD executed]
```
Manual MFU overrides are assumed to already include recomputation and must not receive this factor.

**Activation recomputation reference** (informational — for understanding HFU/MFU ratios, NOT for adjusting C in the training time formula):

| Checkpointing Mode | Actual Executed FLOPs | HFU/MFU Ratio |
|---------------------|----------------------|---------------|
| None | 6ΨD | 1.0 |
| Selective | 6ΨD × (1 + s/(6d)) | 1 + s/(6d) (Korthikanti et al. 2022, Eq. 9) |
| Full | 8ΨD | ~1.33 |

The selective recomputation HFU/MFU ratio `1 + s/(6d)` was verified against Korthikanti et al. Table 5: for GPT-3 175B (s=2048, d=12288), predicted ratio = 1.028, measured MFU=51.4% / HFU=52.8% giving ratio = 1.027. For typical large models this overhead is 1-5%; for long sequences (s >> d) it can be significant.

**Empirical wall-clock overhead**: The 33% compute overhead for full recomputation is the theoretical minimum assuming perfect overlap of recomputation with backward. Measured per-layer on a 22B model (A100, Korthikanti et al. Table 4): full recompute = **39% overhead** (19.6ms baseline vs 27.2ms), selective = **7% overhead** (19.6ms vs 20.9ms), selective + sequence parallelism = **4% overhead** (19.6ms vs 20.3ms). The per-layer overhead exceeds the theoretical 33% because recomputation disrupts backward-pass communication overlap. End-to-end wall-clock overhead is typically ~30-39% per-layer for full recompute, and ~1.5-1.65x overall (HuggingFace benchmarks, gpu_poor). Use these numbers to justify lower MFU defaults for checkpointed configurations; do not stack them as extra runtime multipliers after applying MFU.

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

MFU is the fairer comparison metric because it is independent of implementation choices. A system with activation checkpointing will always show higher HFU than MFU (since it does more work per token), but this does not mean it is faster. The calculator should use **MFU** (based on 6ΨD) for its utilization display and training time formula, and reflect checkpointing through lower MFU defaults or user override rather than a second runtime multiplier.

**Common mistake -- omitting MFU from the time formula**: Using peak FLOPS without an MFU factor produces dramatic underestimates. For example, HyperCLOVA 82B trained on 1024 A100s: a naive estimate using raw peak FLOPS gives ~2.7 days, but actual training took 13.4 days -- a **5x underestimate**. The training time formula in Section 6.1 avoids this by dividing by MFU, which is essential for realistic estimates.

**FP8 effective TFLOPS for training time estimation**: When FP8 training is selected (NVIDIA TransformerEngine mode), the calculator should use effective TFLOPS rather than the raw FP8 peak. FP8 kernels achieve **~1.3-1.5x BF16 throughput** in practice, not the theoretical 2x, due to quantization/dequantization overhead, incomplete kernel coverage (LayerNorm, softmax, loss computation remain in higher precision), and unchanged communication costs. The speedup is model-size-dependent (NVIDIA NeMo benchmarks on H100, 32 GPUs):

| Model Size | Measured FP8/BF16 Speedup | Source |
|---|---|---|
| < 10B (e.g., LLaMA 3 8B) | ~1.30x | NVIDIA NeMo FP8 blog |
| 10-100B (e.g., LLaMA 3 70B) | ~1.43x | NVIDIA NeMo FP8 blog |
| 100B+ (e.g., LLaMA 3.1 405B) | ~1.53x | NVIDIA NeMo FP8 blog |

The calculator should apply the effective TFLOPS as:
```
F_effective_fp8 = BF16_TFLOPS × fp8_speedup_factor
```
Where `fp8_speedup_factor` defaults to **1.3** (conservative, matches the most common model sizes users train) and is exposed as an advanced input with range 1.0-2.0. The calculator should NOT use the raw FP8 peak TFLOPS from Section 7 (e.g., 1,979 for H100) directly in the training time formula -- doing so would underestimate training time by ~50%.

**FP8 memory impact**: NVIDIA TransformerEngine FP8 does **NOT** reduce model state memory -- parameters remain in bf16/fp32, and FP8 is used only inside compute kernels (matmuls). Memory consumption is identical to bf16 mixed precision (16-18 bytes/param for AdamW). Only the MS-AMP backend (Microsoft) stores weights and gradients in FP8 format, achieving the 14 bytes/param figure from Section 5.1. The calculator should default FP8 to **no memory savings** (same as bf16 mixed precision) unless the user explicitly selects "FP8 weight storage (MS-AMP)" mode.

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
2. **Non-matmul operations** (~60% of peak): Memory-bandwidth-bound operations like LayerNorm, softmax, activation functions (ReLU/SiLU/GELU), and residual additions cannot saturate tensor cores and add ~15-20% throughput loss on top of the matmul gap. The hardware reason: these operations execute on **vector (non-tensor-core) units** whose peak throughput is ~4x lower than tensor core throughput (e.g., A100: ~78 TFLOPS vector vs 312 TFLOPS tensor cores in BF16; H100: ~250 vs 989 TFLOPS). Even if these operations were compute-bound (most are memory-bound), they could never exceed ~25% of the peak FLOPS used in MFU denominators.
3. **Framework and kernel launch overhead** (down to 20-40% of peak): Unoptimized implementations (e.g., naive HuggingFace GPT-2) suffer from Python overhead, kernel launch latency, and unfused operations. Optimized frameworks (Megatron-LM, NeMo) recover much of this through kernel fusion, efficient scheduling, and CUDA Graphs (which convert a sequence of kernel launches into a DAG launched once, eliminating per-kernel launch overhead). **Important**: `nvidia-smi` reports *kernel utilization* (fraction of time any kernel is running), NOT MFU. A GPU showing 100% `nvidia-smi` utilization may still have very low MFU if the running kernels are inefficient or memory-bound. Do not conflate `nvidia-smi` utilization with compute efficiency.
4. **Distributed communication**: DP all-reduce, TP all-reduce, and PP bubbles add idle time proportional to cluster size and interconnect bandwidth.
5. **I/O and system overhead**: Data loading, checkpointing, and memory allocator fragmentation contribute the remaining loss.

The gap between items 2-3 explains why framework choice matters so much: a well-optimized stack (Megatron-LM) achieves 40-55% MFU, while a naive implementation on the same hardware may see 15-25%.

**Arithmetic intensity and compute vs. memory boundedness**: Arithmetic intensity is the ratio of FLOPs performed to bytes of memory moved (FLOPs/byte). Modern GPUs have a compute-to-bandwidth ratio of ~150-300 FLOPs/byte (e.g., H100 SXM: 989 TFLOPS BF16 / 3.35 TB/s HBM = ~295 FLOPs/byte). When a workload's arithmetic intensity exceeds this ratio, it is *compute-bound* and can approach peak FLOP/s; when it falls below, it is *memory-bandwidth-bound* and MFU drops proportionally. Strategies to increase arithmetic intensity include larger batch sizes, kernel fusion (reducing intermediate memory traffic), and algorithmic rewrites like Flash Attention's online softmax (which restructures the computation to perform more FLOPs per byte moved between HBM and SRAM).

**Small micro-batch MFU degradation**: The MFU ranges above assume reasonably large micro-batch sizes (b >= 4). When micro-batch size is very small (b = 1-2) and the model's hidden dimension is modest, individual matmuls become memory-bandwidth-bound rather than compute-bound because the arithmetic intensity (FLOPs per byte loaded) drops below the GPU's compute-to-bandwidth ratio. This can reduce MFU to well below the guideline ranges -- in extreme cases by 2-5x. The calculator should warn when b <= 2 that MFU may be significantly lower than the default estimate.

The calculator should provide a default MFU based on model size, GPU count, and selected pipeline schedule efficiency, with a slider for user override (range: 1-70%). The override value represents end-to-end MFU, including any pipeline bubble and communication idle time.

**Profiling tools for MFU diagnosis**: For users seeking to measure or diagnose MFU in their actual training runs, the key tools are: (1) NVIDIA DCGM with `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (fraction of time Tensor Cores are active) and `DCGM_FI_PROF_DRAM_ACTIVE` (HBM bandwidth utilization) -- these directly indicate whether a workload is compute-bound or memory-bound; (2) PyTorch Profiler traces, which show idle gaps between kernel launches on CUDA streams; (3) `nvidia-smi` for basic kernel occupancy (but see the warning above about its limitations). Measuring MFU itself typically requires pen-and-paper analysis: compute model FLOPs per iteration from the architecture, measure wall-clock time per iteration, and divide.

### 6.4 Communication Overhead

For diagnostic or explicit step-time models, communication time can be estimated
separately:

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
Tokens/sec = B_tok / T_step = B_tok / (T_compute + T_communication)
```

**Default calculator convention**: Do not add this communication model on top of
the MFU-based training-time formula in Section 6.1. MFU already includes
communication stalls and pipeline idle time. These formulas are for explaining
or replacing the all-in MFU assumption when a user supplies a calibrated
step-time model.

**EP all-to-all communication cost** (MoE models only): Expert Parallelism requires two all-to-all operations per MoE layer per forward pass — one to dispatch token hidden states to the GPU holding the assigned expert, and one to combine processed results back. Per MoE layer, the communication volume per direction (dispatch or combine) is (MegaScale-MoE, ByteDance 2025):
```
V_ep_a2a = (topk / N_ep) × b × s × d × (N_ep - 1) / N_ep × β bytes
```
Where `topk/N_ep` reflects that each token routes to `topk` experts out of `N_ep` groups, and `(N_ep - 1)/N_ep` is the standard all-to-all scaling factor (each GPU sends data to `N_ep - 1` peers). The full per-step EP communication for the forward pass is:
```
V_ep_forward = L_moe × 2 × V_ep_a2a     (2 = dispatch + combine per layer)
V_ep_total = 2 × V_ep_forward             (forward + backward)
```
The backward pass mirrors the forward with two additional all-to-all operations per layer (gradient dispatch and gradient combine). Total EP communication per training step: `4 × L_moe × V_ep_a2a`.

EP communication can be a significant bottleneck: Megatron Core documentation reports EP all-to-all consuming 30-40% of training time without optimization. NVIDIA recommends overlapping all-to-all with computation (`--overlap-moe-expert-parallel-comm` in Megatron-LM). When `topk > N_ep`, the system can switch from all-to-all to all-gather + reduce-scatter (ring-based), which is more efficient for high-topk models like DeepSeek-V3 (topk=8) with small EP groups.

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

This formula assumes at least one recoverable checkpoint is retained. If `checkpoint_retention = 0`, scheduled checkpoint writes do not provide a usable recovery point, so nonzero failure-rate runs should treat average lost work as unbounded and mark failure-adjusted training time as divergent.

**Impact by scale:**

| N_gpu | N_inst (8 GPU/node) | Daily failures | Overhead per failure | Denominator loss | Training time multiplier |
|-------|---------------------|----------------|----------------------|------------------|--------------------------|
| 64 | 8 | 0.08 | 1.5 hrs | 0.5% | ~1.005x (negligible) |
| 256 | 32 | 0.32 | 1.5 hrs | 2.0% | ~1.02x |
| 1,024 | 128 | 1.28 | 1.5 hrs | 8.0% | ~1.09x |
| 4,096 | 512 | 5.12 | 1.5 hrs | 32.0% | ~1.47x |
| 16,384 | 2,048 | 20.5 | 1.5 hrs | 128.0% | diverges |

The calculator should:
1. Compute and display the failure-adjusted training time alongside the theoretical time when N_gpu >= 256
2. Expose failure rate, recovery time, and checkpoint frequency as advanced inputs
3. Warn when the denominator drops below 0.5 (training time more than doubles due to failures)
4. Warn when failure recovery is enabled but checkpoint retention is 0
5. Note that at extreme scale (16K+ GPUs), the 1% failure rate assumption may understate reality -- Meta reported hundreds of interruptions during LLaMA 3 405B training on 16K H100s

---

## 7. GPU Hardware Specifications

Embed these as selectable presets. Users should also be able to enter custom GPU specs.

| GPU | VRAM (GB) | BF16 TFLOPS | TF32 TFLOPS | FP8 TFLOPS | Mem BW (GB/s) | NVLink BW (GB/s) | TDP (W) |
|-----|-----------|-------------|-------------|------------|---------------|-------------------|---------|
| V100 32GB | 32 | 125 | — | — | 900 | 300 | 300 |
| A100 PCIe 80GB | 80 | 312 | 156 | — | 2,039 | — | 300 |
| A100 40GB | 40 | 312 | 156 | — | 1,555 | 600 | 400 |
| A100 80GB | 80 | 312 | 156 | — | 2,039 | 600 | 400 |
| H100 PCIe 80GB | 80 | 756 | 378 | 1,513 | 2,039 | — | 350 |
| H100 SXM | 80 | 989 | 495 | 1,979 | 3,350 | 900 | 700 |
| H100 NVL | 94 | 835 | 418 | 1,671 | 3,900 | 600 | 400 |
| H200 SXM | 141 | 989 | 495 | 1,979 | 4,800 | 900 | 700 |
| B200 (HGX) | 180 | 2,250 | 1,125 | 4,500 | 8,000 | 1,800 | 1,000 |
| B200 (NVL72) | 186 | 2,500 | 1,250 | 5,000 | 8,000 | 1,800 | 1,200 |
| MI250X | 128 | 383 | — | — | 3,276 | — | 560 |
| MI300X | 192 | 1,307.4 | 653.7 | 2,614.9 | 5,300 | — | 750 |
| L40S | 48 | 362 | 183 | 733 | 864 | — | 350 |
| RTX 4090 | 24 | 165.2 | 82.6 | 330.3 | 1,008 | — | 450 |
| RTX 4080 | 16 | 97.5 | 48.7 | 194.9 | 716.8 | — | 320 |
| RTX 3090 | 24 | 71 | 36 | — | 936 | — | 350 |
| RTX 3060 12GB | 12 | 25 | 13 | — | 360 | — | 170 |
| T4 | 16 | 65 | — | — | 320 | — | 70 |
| A10G | 24 | 70 | 35 | — | 600 | — | 300 |
| L4 | 24 | 121 | 60 | 242 | 300 | — | 72 |

Note: Consumer GPU BF16 TFLOPS listed above are tensor core rates (with sparsity disabled). Pre-Ampere GPUs (V100, T4) lack BF16 hardware support; their values are FP16 tensor core rates. The calculator should warn when BF16 precision is selected with a pre-Ampere GPU (requires FP16 with loss scaling instead). PCIe variants lack NVLink, so TP across PCIe GPUs uses PCIe bandwidth (~64 GB/s for Gen5) instead. The calculator should warn when N_tp > 1 is selected with a PCIe GPU.

**B200 variant note**: The B200 ships in two power/performance configurations: the HGX variant (1,000W, 180 GB, used in DGX B200 systems) and the NVL72 variant (1,200W, 186 GB, higher clocks). The GB200 NVL72 system contains 72 B200 GPUs paired into 36 Grace-Blackwell Superchips (each = 1 Grace CPU + 2 B200 GPUs). NVIDIA's official "per GPU" specs for the NVL72 refer to per-Superchip (2 B200 dies), not per individual die — the per-die values above are the correct inputs for this calculator. DGX B200 lists 1,440 GB HBM3e and 64 TB/s bandwidth across 8 GPUs, so the per-GPU HGX value is 180 GB and 8 TB/s. The full NVL72 system has ~13.4 TB total GPU memory and all 72 GPUs connected via NVLink in a single domain. Note: early NVIDIA announcements (GTC 2024) cited 192 GB for B200; shipped products have 180 GB (HGX) or 186 GB (NVL72).

**TF32 (TensorFloat-32) note**: TF32 is a **compute mode**, not a storage format — it uses 19-bit precision internally in matrix/tensor cores (10-bit mantissa of FP16, 8-bit exponent of FP32) but all tensors remain stored as FP32 (4 bytes/element) in memory. TF32 is **enabled by default** in PyTorch 1.12+ on Ampere and newer NVIDIA GPUs. AMD CDNA 3 accelerators such as MI300X also expose TF32 matrix throughput; MI200-series accelerators do not. When the user selects "FP32 training" on hardware with TF32 support, the calculator should use the TF32 TFLOPS rate (not the non-tensor-core/vector FP32 rate) for training time estimation, since this reflects the actual default behavior. Without this adjustment, FP32 training time estimates would be several times too pessimistic. TF32 does not affect memory calculations — all tensors remain in FP32 at 4 bytes per element. Pre-Ampere GPUs (V100, T4) and AMD MI200-series GPUs should use the non-tensor-core/vector FP32 rate (e.g., V100: 15.7 TFLOPS). For consumer GPUs (RTX 30xx/40xx), TF32 TFLOPS listed are tensor core rates.

**Dense vs sparse TFLOPS warning**: NVIDIA's official spec sheets frequently headline **structured sparsity (2:4) TFLOPS**, which are exactly **2x the dense TFLOPS**. For example, the H100 SXM is often quoted at 1,979 BF16 TFLOPS -- that is the sparsity rate; the dense rate is 989 TFLOPS. All values in the table above are **dense TFLOPS**, which is what training workloads achieve (2:4 sparsity requires specially pruned weight matrices and is not used during standard training). When users enter custom GPU specs, the calculator should validate against known dense values and warn if the entered TFLOPS appears to be a sparsity-inflated figure (i.e., roughly 2x a known dense value). Using sparsity TFLOPS in the training time formula would underestimate wall-clock time by 2x.

**GPUs per node**: Typically 8 for NVIDIA (DGX), 8 for AMD. This constrains max TP degree. Consumer/workstation GPUs (L40S, RTX 4090, RTX 3090) are typically 1-2 per node without NVLink.

**Inter-node bandwidth defaults** (for communication overhead estimation):
- InfiniBand HDR: 200 Gb/s link rate, about 25 GB/s before protocol overhead (A100-era clusters)
- InfiniBand NDR: 400 Gb/s link rate, about 50 GB/s before protocol overhead (H100-era clusters)
- The calculator should default to HDR 200 (25 GB/s) and allow a GB/s override.
- The default MFU-based training-time estimate should not stack a separate bandwidth multiplier unless the user switches to an explicit communication/step-time model.

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

Note: Apple Silicon has no BF16 or TF32 tensor-core mode. The Apple table values are FP32-class GPU throughput figures used as the non-TF32 FP32 fallback in this calculator, not NVIDIA-style tensor-core rates that can be divided by 8 to recover FP32 throughput. Apple Silicon FP16 estimates should be treated as approximate unless a separate FP16 throughput value is supplied. These chips lack NVLink or multi-GPU interconnect, so parallelism is limited to single-device strategies (no TP/PP). The calculator should treat Apple Silicon as single-GPU only (N_tp=1, N_pp=1, N_dp=1) and use the user-selected memory configuration (not the max) as available VRAM. The M3 Ultra's 512 GB unified memory is notable -- it can hold a full 70B model in bf16 (140 GB) with room for optimizer states, enabling full fine-tuning of large models on a single device.

---

## 8. Cost Estimation

### 8.1 Compute Cost (Primary)

```
Cost_compute = N_gpu × T_theory_hours × price_per_GPU_hour
```

Here `T_theory_hours = 24 × T_theory_days` from Section 6.5. This is the dominant cost component for most training runs.

Provide default pricing presets (user can override):

| GPU | Approx. On-Demand ($/hr/GPU) |
|-----|------------------------------|
| V100 32GB | $1.50 - $2.50 |
| A100 80GB | $2.50 - $4.00 |
| H100 SXM | $3.00 - $5.00 |
| H200 SXM | $4.00 - $6.00 |
| B200 | $5.00 - $8.00 |

**Reference cloud instances** (representative on-demand pricing; prices change frequently -- the calculator should let users override). AWS rows use us-east-1 Linux shared On-Demand rates; GCP rows use Iowa (`us-central1`) on-demand rates; Azure uses East US Linux retail pricing; Lambda uses public on-demand instance pricing:

| Provider | Instance | GPU | Count | VRAM/GPU | $/hr |
|----------|----------|-----|-------|----------|------|
| AWS | p4d.24xlarge | A100 | 8 | 40 GB | $21.957642 |
| AWS | g5.xlarge | A10G | 1 | 24 GB | $1.006 |
| GCP | a2-highgpu-1g | A100 | 1 | 40 GB | $3.673385 |
| GCP | g2-standard-4 | L4 | 1 | 24 GB | $0.706832276 |
| Azure | Standard_NC8as_T4_v3 | T4 | 1 | 16 GB | $0.752 |
| Lambda | gpu_1x_a100_sxm4 | A100 SXM | 1 | 40 GB | $1.99 |

The calculator should accept a custom $/GPU/hr input and show total estimated cost.

### 8.2 Checkpoint Storage Cost

Training checkpoints accumulate over a run. Let `checkpoint_bytes_per_param` be the persisted restart state from Section 5.1. Gradients are recomputed after resume and are not persisted. Count the resolved parameter tensor plus optimizer states (`parameterBytes + K_opt`). Default mixed-precision AdamW is `14Ψ` bytes (bf16 parameters + fp32 master + Adam moments), bf16-state AdamW is `10Ψ`, no-master mixed AdamW is `10Ψ`, and bitsandbytes 8-bit AdamW is `8Ψ`.
```
checkpoint_span = T_actual_days × f_checkpoint
num_checkpoints = ceil(checkpoint_span)
checkpoint_retention = user-configurable limit on saved checkpoints (default: 5)
checkpoint_payload_size = checkpoint_bytes_per_param × Ψ
checkpoint_size = 1.04 × checkpoint_payload_size
peak_checkpoint_storage = min(num_checkpoints, checkpoint_retention) × checkpoint_size
avg_checkpoint_count = if checkpoint_span <= 0
  then 0
  else (
    sum_{i=0}^{floor(checkpoint_span)-1} min(i, checkpoint_retention)
    + (checkpoint_span - floor(checkpoint_span)) × min(floor(checkpoint_span), checkpoint_retention)
  ) / checkpoint_span
avg_storage = avg_checkpoint_count × checkpoint_size
Cost_storage = price_per_GB_month × (avg_storage_GB + dataset_GB) × (T_actual_days / 30.25)
```
Here `avg_checkpoint_count` is time-weighted over the training run. Checkpoints are assumed to save at cadence boundaries, with a terminal save included in `num_checkpoints` and peak storage but contributing no additional in-run storage duration. `dataset_GB` is an optional user-provided dataset/object-store footprint and defaults to 0 GB when omitted.

**Checkpoint retention**: In practice, training frameworks limit the number of checkpoints retained on disk. HuggingFace Trainer provides `save_total_limit` (default: None/unlimited); DeepSpeed has no native equivalent -- retention must be handled at the training script or framework wrapper level. The calculator defaults to `checkpoint_retention = 5` as a practical estimate (a commonly used value that balances recovery flexibility against storage cost). When `checkpoint_retention` is set, older checkpoints are deleted as new ones are saved, capping peak storage at `checkpoint_retention × checkpoint_size` bytes rather than growing indefinitely. The calculator should expose this as an advanced input.

Default storage price: **$0.023/GB/month** (AWS S3 standard). The calculator should expose this and dataset storage size as advanced inputs. For large models, checkpoint storage is significant even with retention limits: a 70B model with `checkpoint_retention = 5` has peak storage of 5 × 840 GB = 4.2 TB. Without retention limits, the same model saving hourly checkpoints over 90 days would accumulate ~2,160 checkpoints × 840 GB each = ~1.8 PB peak storage.

### 8.3 Failure Overhead Cost

When using the failure-adjusted training time from Section 6.5, let `T_actual_hours = 24 × T_actual_days`. The additional training time then translates directly to additional compute cost:
```
Cost_failure_overhead = N_gpu × (T_actual_hours - T_theory_hours) × price_per_GPU_hour
```

This includes both the recovery time (GPUs idle but allocated) and recomputation cost (re-doing work since last checkpoint). At scale, this is substantial: for a 4,096-GPU run with ~32% failure overhead, the failure cost is roughly a third of the base compute cost.

### 8.4 Total Cost

```
Cost_total = Cost_compute + Cost_storage + Cost_failure_overhead
```

The calculator should display each component separately so users can see the cost breakdown. For small-scale runs (<256 GPUs), `Cost_failure_overhead` is negligible and can be omitted from the display. `Cost_storage` is always a small fraction of `Cost_compute` but useful for storage planning at scale.

If the UI also shows the fully adjusted compute spend, label it explicitly as `Cost_compute_actual = N_gpu × T_actual_hours × price_per_GPU_hour` and do **not** add `Cost_failure_overhead` a second time.

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
   → PCIe GPU check: When the selected GPU lacks NVLink (PCIe-only GPUs: RTX
     4090/4080/3060, L40S, T4, L4, A10G, A100 PCIe), prefer ZeRO-3 over TP.
     TP requires 4 synchronous all-reduces per layer on the critical path;
     at PCIe Gen4 unidirectional bandwidth (~32 GB/s) vs NVLink (~900 GB/s
     on H100 SXM), TP communication becomes a severe bottleneck (~28x slower).
     Only use TP on PCIe GPUs if ZeRO-3 cannot fit the model. The RTX 3090
     supports a limited 2-GPU NVLink bridge (112.5 GB/s) — TP=2 is acceptable
     on paired RTX 3090s but still far slower than data-center NVLink.
   → For NVLink-equipped GPUs: Add TP (start with N_tp = 2, increase to 4, 8)
   → Recalculate per-GPU memory with TP
   → Combine with ZeRO-1 (preferred) or ZeRO-2/3 (no PP constraint yet)
4a. For MoE models, if total expert params exceed per-GPU memory after TP:
   → Add EP (Expert Parallelism). Sizing heuristic:
     a. Start with N_ep = 1 (no EP)
     b. If expert memory per GPU = (E / N_ep) × Ψ_ffn × Φ exceeds available
        memory after TP, increase N_ep
     c. N_ep must satisfy: E % N_ep == 0 (experts divide evenly)
     d. Prefer N_ep = E when feasible (one expert per GPU eliminates local
        token permutation overhead — Megatron Core recommendation)
     e. CONSTRAINT: N_ep × N_tp ≤ GPUs_per_node (keep EP × TP within NVLink
        domain for performance — Megatron Core, NeMo Framework)
     f. When scaling beyond one node, prefer PP over expanding EP/TP across
        nodes (inter-node all-to-all is expensive)
     g. For MoE layers, prefer EP over TP: EP provides larger local matrix
        sizes and lower communication overhead than TP for expert computation
   → N_dp = N_gpu / (N_tp × N_cp × N_pp × N_ep)
   → Expert data parallel degree: N_edp = N_dp × N_cp × N_tp / N_ep (for ZeRO
     interaction; see Section 5.2)
5. If TP=8 still insufficient:
   → Add PP (start with N_pp = 2, increase as needed)
   → Each stage holds fewer layers → less memory
   → CONSTRAINT: branch on framework before selecting PP.
     - DeepSpeed: PP requires ZeRO-0 or ZeRO-1 only (see compatibility table below).
       If ZeRO-2/3 was selected in step 3/4, downgrade to ZeRO-1 when adding PP.
     - FSDP: FULL_SHARD / ZeRO-3 is disallowed with PP; SHARD_GRAD_OP / ZeRO-2
       is allowed only under the AFAB conditions in the FSDP + PP subsection below.
6. If sequence length is very long (>32K) and activation memory still exceeds
   GPU capacity even with checkpointing:
   → Add CP. Default sizing: N_cp = seq_len / 8192 (Meta, Llama 3 heuristic).
     E.g., 128K → CP=16, 32K → CP=4. Clamp to powers of 2; minimum N_cp = 1.
   → Replaces s with s/N_cp in activation memory formulas (Section 5.3)
   → CP trades DP parallelism for sequence sharding (DP shrinks as CP grows)
7. Remaining GPUs become DP:
   dense: N_dp = N_gpu / (N_tp × N_cp × N_pp)
   MoE:   N_dp = N_gpu / (N_tp × N_cp × N_pp × N_ep)
```

### Minimum GPU Memory Floor (Largest Layer)

Even with ZeRO-3 sharding across arbitrarily many GPUs, a single transformer layer's parameters must be fully gathered on one GPU during forward and backward passes. During backward, both the gathered parameters and their gradients coexist in memory. This sets an absolute minimum VRAM requirement:

```
Ψ_largest_layer = Ψ_attn + Ψ_ffn + Ψ_norm  (single transformer block)
M_min_gpu = Ψ_largest_layer × (β + β_grad)    (β bytes for gathered params + β_grad bytes for gradients)
```

**Closed-form for the largest layer** (Rajbhandari et al., 2021): The largest single weight matrix in a standard transformer is the FFN up-projection (d -> d_ff). The minimum working memory for gathering its parameters and gradients is:
```
Standard FFN (d_ff = 4d):    M_min_gpu = 4 × d² × (β + β_grad) bytes
SwiGLU FFN (arbitrary d_ff): M_min_gpu = d × d_ff × (β + β_grad) bytes
```
These closed-form expressions let the calculator compute the floor from just `d`, `d_ff`, `β`, and `β_grad` without enumerating all layer parameters. The calculator should flag when `M_min_gpu > GPU_VRAM × 0.8` (leaving 20% for framework overhead and activations), as this indicates that even with full ZeRO-3 sharding, per-GPU memory pressure from the largest single operator will be severe.

With bf16 parameters and bf16 gradients, this is 4 bytes per parameter in the largest layer. With bf16 parameters and fp32 gradients (the default estimate), it is 6 bytes per parameter. For example, a 1.1B-parameter largest layer requires ~4.4 GB with bf16 grads or ~6.6 GB with fp32 grads before adding activations and framework overhead. Display this as an output: "Minimum GPU VRAM (even with full sharding)".

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
When `b x s >> 3 x d` (large batch, long sequences), ZeRO-3 has lower total communication volume than TP. However, this is a volume comparison only -- TP communication travels over NVLink (intra-node, ~900 GB/s on H100), while ZeRO-3 communication often traverses inter-node interconnect. A single 400 Gb/s NDR rail is only about 50 GB/s before protocol overhead; higher aggregate bandwidth requires multiple NICs/rails. The recommendation engine should prefer TP within a node (where NVLink is available) and ZeRO across nodes, unless the model is too small to benefit from TP (few attention heads) or the cluster has uniformly high-bandwidth interconnect.

### Multi-Node Parallelism Guidance

When training spans multiple nodes, interconnect bandwidth determines the optimal parallelism strategy:

- **Multi-node with slow interconnect** (e.g., Ethernet, InfiniBand HDR): Prefer HYBRID_SHARD (ZeRO++ Stage 3) or standard ZeRO with PP. Keep TP strictly within a single node. Use DP or PP across nodes since they have lower communication bandwidth requirements than TP.
- **Multi-node with fast interconnect** (InfiniBand NDR 400 Gb/s+): Standard ZeRO-3 / FULL_SHARD across all GPUs is viable. TP can extend across nodes only if NVSwitch or equivalent high-bandwidth fabric is available (rare outside DGX SuperPOD configurations).
- **Single node**: All parallelism strategies are viable. TP up to 8 (full node) is standard. HYBRID_SHARD is unnecessary (equivalent to FULL_SHARD within one node).

The calculator should default to TP-within-node and flag when the user's configuration would place TP across node boundaries.

**TP degree tuning**: Using fewer than the maximum TP ranks per node can improve throughput. BLOOM-176B found TP=4 outperformed TP=8 by 19% on 8-GPU nodes, because smaller TP degree reduces per-layer all-reduce volume and increases the amount of work per GPU (better arithmetic intensity). The underlying mechanism is that high TP degree splits weight matrices into smaller GEMMs, which can push computation from compute-bound to memory-bandwidth-bound (Narayanan et al., 2021). For models >= 18.4B parameters, TP=8 was empirically optimal; for smaller models (particularly < 10B), TP=4 or TP=2 can yield higher throughput because the per-GPU GEMMs remain large enough to saturate tensor cores. The optimal TP degree depends on model size, interconnect topology, and the tradeoff between communication overhead and per-GPU memory pressure. The calculator should not assume TP=8 is always optimal; when memory permits, TP=4 or TP=2 with higher DP may yield better throughput.

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
- Global batch size in sequences: `B_seq = b × G × N_dp`
- Global batch size in tokens: `B_tok = b × s × G × N_dp`
- **1F1B microbatch minimum**: When pipeline parallelism is active with the standard 1F1B schedule, `num_microbatches >= N_pp - 1` (hard minimum; see Section 5.7). The calculator should validate this and warn when violated.
- **Interleaved PP microbatch divisibility**: When using the interleaved (virtual pipeline) schedule, `num_microbatches % N_pp == 0` (must be evenly divisible by pipeline parallel degree; see Section 5.7). This is stronger than the 1F1B minimum. The calculator should enforce this when interleaved PP is selected.
- **ZeRO + Pipeline Parallelism compatibility**: DeepSpeed ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism (gradient sharding conflicts with PP's gradient accumulation across stages). Only ZeRO-0 or ZeRO-1 can be combined with PP in DeepSpeed. The calculator must enforce this constraint:

| ZeRO Stage | + TP | + PP (DeepSpeed) | + PP (FSDP) |
|------------|------|------------------|-------------|
| ZeRO-0 | Yes | Yes | Yes |
| ZeRO-1 | Yes | Yes | N/A (no FSDP equivalent*) |
| ZeRO-2 | Yes | **No** | Yes (SHARD_GRAD_OP + AFAB schedule only**) |
| ZeRO-3 | Yes | **No** | **No** |

\* PyTorch FSDP has no native ZeRO-1 equivalent (shard optimizer states only). The minimum FSDP sharding level is SHARD_GRAD_OP (ZeRO-2). Users needing ZeRO-1 must use DeepSpeed.
\** FSDP's SHARD_GRAD_OP can be combined with PP using the AFAB schedule under specific conditions (see FSDP + PP subsection below). This is an FSDP-specific capability that does not apply to DeepSpeed ZeRO-2.

When PP is active with DeepSpeed, the recommendation engine must not select ZeRO-2 or ZeRO-3. Conversely, if ZeRO-2/3 is needed for memory, PP cannot be used (unless using FSDP with SHARD_GRAD_OP + AFAB).

**FSDP + Pipeline Parallelism**: FULL_SHARD (ZeRO-3) must not be used with PP because parameters are freed after forward and must be AllGathered again for every micro-batch in the PP schedule, which is prohibitively expensive. The choice between replicated-gradient 1F1B semantics and ZeRO-2-style AFAB semantics depends on the micro-batch count relative to pipeline depth (Meta, Llama 3 scaling):
```
if micro_batch_per_dp_rank >= 2 × N_pp: use FSDP NO_SHARD in this calculator, or an implementation-specific optimizer-only FSDP mode when available, with interleaved 1F1B
if micro_batch_per_dp_rank <  2 × N_pp: use FSDP ZeRO-2 (SHARD_GRAD_OP / HYBRID_SHARD_ZERO2) + AFAB schedule
```
The replicated-gradient path retains unsharded gradients across micro-batches, avoiding extra communication but using more memory. PyTorch-style FSDP has no native optimizer-only ZeRO-1 strategy, so this calculator represents that high-throughput path as FSDP `NO_SHARD`; users needing optimizer-state sharding with PP should use DeepSpeed ZeRO-1 or an implementation that explicitly exposes FSDP optimizer-only sharding. ZeRO-2 re-shards gradients after each micro-batch, saving memory at the cost of additional reduce-scatter operations. The calculator should apply this heuristic when FSDP + PP is active: default to `NO_SHARD`/optimizer-only semantics for 1F1B and fall back to `SHARD_GRAD_OP`/`HYBRID_SHARD_ZERO2` only when the batch size is too small to fill the pipeline efficiently and AFAB is being modeled.

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

**Note on Φ in this section**: The formulas below use Φ=16 (bf16 gradients, AdamW mixed precision) for concrete examples. With fp32 gradients (Φ=18, the pretraining default from Section 5.1), trainable model memory increases by ~12% (e.g., 16Ψ becomes 18Ψ, 36Ψ becomes 40Ψ). The calculator should use the user's selected gradient precision throughout.

**Post-training activation convention**: Unless stated otherwise, `M_activations` includes checkpointed transformer activations plus the output-logit peak from Section 5.3. For trainable language-model passes, count both the mixed-precision logits tensor and the transient fp32 logits gradient. For inference-only or MeZO-style forward passes, count only the forward logits tensor. Critic/value heads in PPO do not materialize vocabulary-sized logits; scale only their transformer activation component.

### 10.1 Supervised Fine-Tuning (SFT)

**Full fine-tuning**: Identical to pretraining memory (16Ψ + activations). Dataset is smaller so compute is less.

**LoRA** (Low-Rank Adaptation):
```
Base model (frozen, bf16):  2Ψ bytes  (no gradients, no optimizer)
LoRA adapter parameters:    Ψ_lora = r × Σ_adapted_matrices(input_dim + output_dim)
  where r = rank (8-64), summed over each adapted matrix copy:
    - 4 (attention only: Q, K, V, O)
    - 7 (attention + FFN: Q, K, V, O, gate, up, down) — recommended default (QLoRA paper shows all-linear is required to match full finetuning quality)
LoRA gradients (bf16):      2 × Ψ_lora
LoRA optimizer (fp32):      12 × Ψ_lora  (master + Adam m + v)
Activations:                Same as full model (entire model runs forward/backward)

M_total_lora = 2Ψ + 16 × Ψ_lora + M_activations
```

For dense LLaMA/SwiGLU models with MHA and LoRA on Q, K, V, O, gate, up, and down:
```
Ψ_lora = L × r × (11d + 3d_ff)
```
For GQA, K/V projections are narrower. With `d_kv = d × a_kv / a`:
```
Attention only:             Ψ_lora = L × r × (6d + 2d_kv)
Attention + SwiGLU FFN:     Ψ_lora = L × r × (9d + 2d_kv + 3d_ff)
```

The shorthand `2 × r × d × M_modules × L` is exact only when every adapted matrix is `d × d`. It is exact for attention-only MHA, but undercounts SwiGLU FFN adapters because FFN matrices use `d_ff`, not `d`.

Example: 7B SwiGLU MHA model, rank 16, 32 layers, d=4096, d_ff=11008:
```
With attention-only targets (Q, K, V, O):
  Ψ_lora = 2 × 16 × 4096 × 4 × 32 = 16.8M  (0.24% of base model)
  Memory: 2 × 7B + 16 × 16.8M = 14GB + 0.27GB = ~14.3GB + activations

With all-linear targets (Q, K, V, O, gate, up, down):
  Ψ_lora = 32 × 16 × (11 × 4096 + 3 × 11008) = 40.0M  (0.57% of base model)
  Memory: 2 × 7B + 16 × 40.0M = 14GB + 0.64GB = ~14.6GB + activations
```

**QLoRA** (Quantized LoRA):

"4-bit quantization" maps to several concrete formats: NF4 (bitsandbytes, used by QLoRA), GPTQ-4bit, and AWQ-4bit. Similarly, "8-bit" maps to LLM.int8() (bitsandbytes), GPTQ-8bit, and AWQ-8bit. The calculator should accept a quantization bit-width (4 or 8) and display the corresponding format names for clarity.

```
Base model (4-bit NF4):    Ψ_quantized × 0.5159 bytes + Ψ_non_quantized × b_weight
Base model (8-bit):        Ψ_quantized × 1.01 bytes + Ψ_non_quantized × b_weight
LoRA adapters + optimizer: 16 × Ψ_lora (same as LoRA)
Activations:               Computed in bf16/fp16/fp32 (dequantize → compute → re-quantize)

Ψ_non_quantized = embeddings + output projection + positional embeddings + all norms
b_weight = 2 bytes for bf16/fp16/fp8 compute, 4 bytes for fp32 compute

M_total_qlora = M_quantized_base + 16 × Ψ_lora + M_activations
```

If architecture fields are invalid or unavailable, the calculator falls back to the coarse estimate `M_quantized_base ≈ 0.55Ψ` for 4-bit NF4. This approximation implicitly absorbs non-quantized embeddings/norms and quantization metadata.

**QLoRA loading memory**: HuggingFace/bitsandbytes loads models layer-by-layer: each layer is loaded in fp16 on CPU, quantized to NF4, then the NF4 version is moved to GPU. The GPU never holds the full fp16 model -- its peak during loading is approximately `M_nf4_total + max_layer_size × 2` (the accumulated NF4 model plus one layer in fp16 at a time), which is only slightly above steady-state. However, CPU memory requires ~2Ψ bytes to hold the full fp16 model during this process. The calculator should warn about the CPU memory requirement (~2Ψ) when it exceeds available system RAM.

**QLoRA throughput penalty**: QLoRA training is slower than standard LoRA due to the dequantize-compute-requantize overhead in each forward and backward pass. Empirical measurements show approximately **1.75x wall-clock time** compared to equivalent LoRA fine-tuning. The calculator should apply this penalty when estimating QLoRA training time.

### 10.2 Direct Preference Optimization (DPO)

Two models in memory:
```
Policy model (trainable):   16Ψ (full) or 2Ψ + 16Ψ_lora (LoRA)
Reference model (frozen):   2Ψ
Activations:                2× normal (forward through both models for chosen + rejected)
DPO log-prob storage:       2 × B_seq × s × 4 bytes (chosen + rejected)

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

batch_local = ceil(batch_gen / N_gpus)

T_prefill = (2 × Ψ × s_prompt × batch_local) / F_peak            [compute-bound]

T_decode_per_token = max(
    (Ψ × β) / BW_mem,                                            [memory-bound term]
    (2 × Ψ × batch_local) / F_peak                               [compute-bound term]
)
```

Where `BW_mem` is per-GPU memory bandwidth (e.g., 2.0 TB/s for A100-80GB, 3.35 TB/s for H100 SXM), `β` is bytes per parameter (2 for bf16), `batch_gen` is total concurrent generations, `batch_local` is the fullest per-GPU generation batch under data-parallel serving, and `F_peak` is peak GPU FLOPS. The memory-bound decode term streams one copy of the model weights per token (`Ψ × β` bytes, about `2Ψ` bytes for bf16), not two copies. More data-parallel GPUs increase total generation throughput by serving more sequences concurrently; they do not reduce per-token memory-bound latency for a single local decode step. Apply ~0.87-0.90 efficiency factor to `BW_mem` in practice.

The crossover batch size where decode transitions from memory-bound to compute-bound:
```
B_threshold = β × F_peak / (2 × BW_mem)

| GPU         | BW_mem (TB/s) | F_peak (TFLOPS bf16) | B_threshold |
|-------------|---------------|----------------------|-------------|
| A100-80GB   | 2.0           | 312                  | ~156        |
| H100 SXM    | 3.35          | 989                  | ~295        |
| H200 SXM    | 4.8           | 989                  | ~206        |
```

For bf16/fp16 weights (`β=2`), this simplifies to `B_threshold = F_peak / BW_mem`, matching the table above. Below `B_threshold`, throughput scales linearly with batch size at near-zero cost -- the calculator should flag this as an optimization opportunity when the per-GPU generation batch is much smaller than `B_threshold`.

**Maximum concurrent generations** (memory constraint):
```
max_batch_gen = N_gpus × floor((M_gpu_available_per_gpu - Ψ × β) / (M_kv_per_token × s_gen))

M_kv_per_token = 2 × a_kv × d_kv × L × β_cache   (bytes; factor of 2 for K and V)
```

Where `M_gpu_available_per_gpu` is GPU memory after framework overhead per GPU, and `s_gen` is the maximum generation sequence length. For methods whose training phase already keeps more than `Ψ × β` bytes of resident model state on each GPU, use that larger resident state footprint instead of the simplified `Ψ × β` term. This determines whether a given PPO batch size or GRPO group size `G` fits in memory during the generation phase.

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

Where G is an integer group size, typically 4-16 completions per prompt.

**GRPO generation feasibility**: The `max_batch_gen` formula from Section 10.3 constrains the effective group size. If `G × num_prompts_per_batch > max_batch_gen`, the generation phase must be split into multiple rounds, increasing wall-clock time. The calculator should warn when G exceeds `max_batch_gen / num_prompts_per_batch` and estimate the resulting slowdown. Use the `T_generation` formulas from Section 10.3 to estimate GRPO generation wall-clock time with `batch_gen = G × num_prompts_per_batch`.

### 10.5 Post-Training Compute

| Method | FLOPS per Token | Notes |
|--------|----------------|-------|
| SFT | 6Ψ | Same as pretraining |
| DPO | 8Ψ | Policy train (6Ψ) + reference forward (2Ψ) |
| PPO | ~20Ψ per step | Generation + reward + multi-epoch training |
| GRPO | ~10Ψ per step | Generation + policy train (no critic) |

Post-training datasets are much smaller (10K-1M examples vs. trillions of tokens for pretraining), so total compute is orders of magnitude less.

For MoE policy models, apply the Section 4.1 load-balance factor to the routed-expert portion of non-generation policy training and reference/scoring forwards. Shared experts are always active and are not load-balanced routed capacity. PPO/GRPO generation wall-clock is estimated separately with the autoregressive decode model from Section 10.3, so do not hide generation memory-bandwidth limits by folding them into a single inflated active-parameter count.

### 10.6 Post-Training Parallelism

Post-training methods (DPO, PPO, GRPO) involve multiple models with different roles (trainable vs. frozen) and different execution phases (training vs. generation). This requires parallelism strategies distinct from pretraining.

**Frozen model placement strategies** (in order of preference):

1. **Precompute and discard** (DPO only): TRL's `precompute_ref_log_probs=True` runs one forward pass over the dataset to cache reference log-probs, then discards the reference model entirely. This eliminates 2Ψ from GPU memory. Only applicable when the reference model's outputs are fixed (DPO, not PPO/GRPO where the policy changes during generation).

2. **Replicate on GPU if memory permits**: When the frozen model fits in per-GPU memory alongside the trainable model, keep it resident. This avoids ZeRO-3 all-gather overhead on every forward pass. For LoRA fine-tuning where the base model is already loaded (serving as both trainable base and reference), this comes at zero additional cost (Section 10.2).

3. **Shard with ZeRO-3/FSDP**: When the frozen model does not fit per-GPU, shard it across DP ranks. Each forward pass through the frozen model triggers parameter all-gathers, adding communication overhead. DeepSpeed-Chat's Hybrid Engine mitigates this by switching frozen models from ZeRO-3 to TP-only during inference phases.

4. **CPU offload**: NeMo-Aligner and DeepSpeed-Chat support offloading frozen models (reference, reward) to CPU memory and swapping them in when needed. This trades PCIe bandwidth latency for GPU memory savings.

**TP configuration for frozen models**: Frozen models do NOT need to share the same TP configuration as the policy model. Modern frameworks use different parallelism strategies for different models and phases: OpenRLHF uses vLLM's auto TP for generation while actor/critic use ZeRO-3; DeepSpeed-Chat's Hybrid Engine switches between ZeRO (training) and TP (inference) for the same model. However, if frozen models are colocated on the same GPU group and share NCCL process groups, matching TP simplifies implementation.

**Generation phase parallelism** (applies to PPO rollouts and GRPO group generation):

- **TP reduces generation latency**: All GPUs process the same token in parallel (intra-layer parallelism), reducing per-token decode time proportionally to TP degree. This is the standard approach: NeMo-Aligner reshards from TP+PP (training) to TP-only (generation), achieving a **3.87x speedup** in PPO training (NeMo-Aligner, 2024).

- **PP is problematic for autoregressive decode**: During generation, each decode step produces a single token that must traverse all pipeline stages sequentially. With only 1 microbatch in flight, the pipeline bubble fraction becomes `(N_pp - 1) / N_pp` — e.g., 75% bubbles with 4 stages, 87.5% with 8 stages. This is catastrophic compared to training where `num_microbatches >> N_pp`. The calculator should warn when PP is active and the workload includes a generation phase (PPO, GRPO), and recommend resharding to TP-only for generation when possible.

- **Recommended generation parallelism**: Use TP (up to 8 within a node) for generation latency. If the model does not fit within a single node's GPUs with TP alone, use ZeRO-3 parameter gathering rather than PP. Reserve PP exclusively for the training phase of PPO/GRPO.

---

## 11. Feature Requirements

### 11.1 Input Modes

**Quick Mode**: User enters total parameter count (e.g., "7B") + total tokens + GPU type → instant estimate.

Quick Mode needs to infer architecture details (for activation memory, KV cache, logits memory, and TP divisibility) from just a parameter count. Use this lookup table to estimate heads (a) and layers (L), then solve for hidden dimension (d):

| Param Range | a (heads) | L (layers) |
|------------|-----------|------------|
| < 500M     | 12        | 12         |
| 500M - 2B  | 16        | 24         |
| 2B - 5B    | 32        | 24         |
| 5B - 10B   | 32        | 32         |
| 10B - 24B  | 40        | 40         |
| 24B - 55B  | 64        | 48         |
| >= 55B     | 64        | 80         |

Then solve for d from the standard parameter formula `Psi = 12 * L * d^2` (Section 3.2):
```
d = sqrt(Psi / (12 * L))
```
Round d to the nearest multiple of 128 (common alignment in real architectures). Then infer a coarse architecture family:
- **Dense GPT-style heuristic** (`Psi < 2B`): `d_ff = 4d`, `a_kv = a`, `V = 50,000`
- **Modern open-weight heuristic** (`Psi >= 2B`): `d_ff = round(8/3 * d)`, `a_kv = largest divisor of a that is <= 8`, `V = 128,000`

This gives a reasonable architecture for coarse activation memory and parallelism calculations, but it is intentionally approximate. Expect roughly **10-20% error** for dense 32K-50K-vocab style models and potentially **20-40% error** for KV-cache or logits-dominated estimates on modern GQA / 128K-vocab families. Use Preset or Detailed Mode for purchase decisions, exact fit-checking, or long-context planning.

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
8. Activation checkpointing (none / selective / full / partial)
9. Flash Attention (on/off)
10. GPU type (preset or custom specs)
11. Target training time (optional — for computing minimum GPUs)
12. Number of GPUs (optional — for computing training time)
13. MFU override (slider, 10-70%, with smart default)
14. Parallelism: auto-recommend OR manual (N_tp, N_cp, N_pp, N_dp, ZeRO/FSDP stage)
15. Cost per GPU-hour (with cloud provider presets)

**Advanced Inputs** (collapsible section — these are needed by formulas but have sensible defaults):
16. Context parallelism degree N_cp (default: 1; auto-set by recommendation engine for long sequences)
17. Expert parallelism degree N_ep (default: 1; auto-set for MoE models)
18. MoE parameters: E (total experts), topk (active experts per token), L_moe (MoE layers), E_s (shared experts, default 0), load_balance_factor (default: 1.1) — shown only when MoE architecture is selected
19. Partial checkpointing depth N_recomp (shown when checkpointing mode = partial) — block recompute layer count per pipeline stage, default 1
20. Virtual Pipeline chunks VP (default: 1; for interleaved PP schedule)
21. Framework choice: Megatron-LM / DeepSpeed / FSDP / HF Trainer (default: DeepSpeed) — affects communication bucket sizes (Section 5.4), PP compatibility (Section 9), and activation memory coefficients (Section 5.3)
22. Sequence parallelism toggle (default: auto/on when N_tp > 1)
23. AMP autocast toggle (default: off; use explicit bf16 mode by default)
24. CPU offloading mode (none / optimizer-only / optimizer+params for ZeRO-3) — only valid where supported in Section 5.2
25. ZeRO communication bucket mode: HF auto / raw DeepSpeed defaults / custom bucket sizes; include `overlap_comm` toggle when applicable
26. Inter-node bandwidth preset/override: HDR 200 Gb/s (~25 GB/s) / NDR 400 Gb/s (~50 GB/s) / custom GB/s; tracked for communication assumptions and diagnostics, not stacked on top of MFU by default
27. torch.compile toggle (default: off) — adds ~10% of model weights as overhead (Section 5.4)
28. Chunked cross-entropy toggle (default: off) — eliminates output logits tensor from activation memory (Section 5.3)
29. FP8 options (shown when precision = fp8): effective kernel speedup factor (default: 1.3) and weight storage mode (TransformerEngine default / MS-AMP)
30. KV cache precision for post-training generation phases: bf16 / fp16 / int8 (default: bf16)
31. Checkpoint retention count (default: 5) — caps peak checkpoint storage (Section 8.2)
32. Failure rate f (default: 0.01 failures/instance/day), recovery time t_recovery (default: 1 hour), checkpoint frequency f_checkpoint (default: 24/day) — for failure-adjusted training time (Section 6.5); shown when N_gpu >= 256

**Outputs:**
1. Total parameter count (computed from architecture)
2. Total FLOPS
3. Chinchilla ratio (D / 20Ψ), the scaling-law parameter-count basis used for Ψ, and recommendation
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
7. Recommended parallelism strategy (`N_dp × N_tp × N_cp × N_pp`, plus `× N_ep` for MoE, and ZeRO/FSDP stage)
8. Pipeline bubble overhead %
9. Estimated training time (days/hours), with failure-adjusted time shown alongside when N_gpu >= 256 (Section 6.5)
10. Estimated tokens/second throughput
11. Estimated cost breakdown: compute cost, checkpoint storage cost, failure overhead cost, and total (Section 8)
12. Global batch size (computed as both `B_seq = b × G × N_dp` sequences and `B_tok = b × s × G × N_dp` tokens)
13. Checkpoint size (optimizer-specific persisted state, 14Ψ bytes for default mixed AdamW -- see Section 5.1) for storage planning
14. Attention overhead percentage (12Lds / 6Ψ -- see Section 4.1) to flag long-context cost
15. Predicted training loss (from Chinchilla parametric formula -- see Section 4.3) with caveat on accuracy at extreme over-training ratios
16. Maximum micro-batch size (computed from free GPU memory after model states: `b_max = floor(free_memory / bytes_per_sequence)` where `bytes_per_sequence` is the per-sequence activation cost)
17. Data repetition analysis (when U < D): epochs, data utilization warning, effective data ceiling (Section 4.5)
18. MoE sparsity metrics (when MoE model is selected): sparsity ratio (`Ψ_active / Ψ_total`), efficiency gain (`Ψ_total / Ψ_active`), and load balance overhead applied (Section 4.1) -- helps users understand the compute vs. memory tradeoff
19. Batch size efficiency: `B_tok` vs `B_crit` comparison, compute multiplier above optimum, and wasted-compute fraction from Section 4.4 -- e.g., "Your token batch is 2x B_crit: 3x compute vs optimum, 67% of actual compute is overhead"

### 11.3 Post-Training Calculator Features

**Inputs:**
1. Base model (preset or parameter count)
2. Method: SFT / DPO / PPO / GRPO
3. Fine-tuning approach: Full / LoRA / QLoRA / MeZO
4. LoRA config (if applicable): rank r, alpha, target modules
4a. Trainable parameter percentage (for partial layer freezing beyond LoRA, e.g., "train only the last N layers"). Defaults to 100% for full fine-tuning, computed automatically for LoRA/QLoRA. Affects gradient and optimizer memory proportionally: only the trainable fraction incurs gradient (`β_grad` bytes/param) and optimizer-state (`K_opt` bytes/param, optimizer-specific) costs.
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
- Tokens: must be positive. Warn if the Chinchilla ratio is < 1 (severely undertrained), > 500 (far beyond the regime where the standard Chinchilla coefficients are reliable), and escalate the warning at > 5,000. Do not reject large overtraining ratios outright.
- Micro-batch size: must be ≥ 1
- Sequence length: must be positive, typical range 512 - 131,072
- GPU count: must be ≥ 1, warn if > 100,000
- TP must divide attention heads, KV heads, and d_ff evenly where applicable
- PP must divide layers evenly, or pass the embedding-aware partitioning rule from Section 5.7
- For dense models: `N_dp × N_tp × N_cp × N_pp = N_gpu`
- For MoE models: `N_dp × N_tp × N_cp × N_pp × N_ep = N_gpu`
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

With TP=8, PP=4, DP=8, ZeRO-1 (bf16 grads):
  Params per GPU (TP×PP): 70B / (8×4) = 2.19B
  Params + grads (unsharded, bf16): (2+2) × 2.19B = 8.75 GB
  Optimizer (ZeRO-1, sharded across DP=8): 12 × 2.19B / 8 = 3.28 GB
  Total model states: ~12 GB + activations
  Should fit in 80GB with substantial headroom ✓

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

These real-world measurements can be used to validate the calculator's throughput and cost estimates:
```
GPT-2 124M on 8xA100-80GB:  ~300ms/step, ~94 min for 10B tokens  (karpathy/llm.c)
GPT-2 1558M on 8xH100-SXM:  ~2.8s/step, ~24 hrs for 32B tokens  (karpathy/llm.c)
Reference cost rates: 8xA100 ~ $14/hr, 8xH100 ~ $28/hr
```

**Llama 3 405B training** (Meta, 2024): The largest publicly documented single-model training run with detailed cost/time data:
```
Model:          405B params, 15.6T tokens, 3.8×10^25 total FLOPs
Hardware:       16,384 H100-80GB SXM (700W TDP), RoCE interconnect
GPU-hours:      30.84M (all models: 39.3M)
Est. cost:      ~$62M at ~$2/GPU-hr
Effective time: >90% despite ~1 failure every 2.8 hours on 16K GPUs
MFU:            38-43% depending on stage (see Section 6.3)
```
The calculator should be able to reproduce the ~30.84M GPU-hours figure given the 405B architecture, 15.6T tokens, and 38-43% MFU on H100 SXM. This serves as the primary large-scale validation target.

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
