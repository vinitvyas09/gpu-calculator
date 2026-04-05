# GPU Calculator Implementation Plan

This document guides the phased implementation of the LLM Training GPU Calculator. The spec (`spec/llm-training-gpu-calculator-spec.md`) is the source of truth for all formulas, constants, and requirements. **Do not duplicate formulas here** — reference spec sections instead.

**Current state**: Zero implementation code exists. Spec and research only.

---

## Architecture Overview

Two clean layers:
1. **Calculation engine** (`components/gpu-calculator/formulas/`): Pure TypeScript functions, zero UI dependencies.
2. **UI layer** (`components/gpu-calculator/components/`): React components that call the engine.

### File structure (spec Section 13):
```
app/tools/gpu-calculator/
  page.tsx                          # Tool page (metadata, layout, breadcrumb)
  gpu-calculator-embed.tsx          # Dynamic import wrapper (ssr: false)

components/gpu-calculator/
  gpu-calculator.tsx                # Main calculator component
  types.ts                          # TypeScript interfaces
  constants.ts                      # GPU specs, model presets, defaults
  formulas/
    compute.ts                      # Parameter counting + FLOPs (Sections 3, 4)
    memory.ts                       # Memory estimation (Section 5, 10)
    parallelism.ts                  # Parallelism recommendation engine (Section 9)
    cost.ts                         # Training time + cost estimation (Sections 6, 8)
    post-training.ts                # SFT/DPO/PPO/GRPO specifics (Section 10)
  components/
    pretraining-panel.tsx           # Pretraining inputs + results
    post-training-panel.tsx         # Post-training inputs + results
    memory-breakdown-bar.tsx        # Stacked bar visualization
    gpu-utilization-gauge.tsx       # Memory fill indicator
    parallelism-layout.tsx          # DP×TP×PP grid visualization
    model-selector.tsx              # Preset/quick/detailed model input
    gpu-selector.tsx                # GPU type selector with specs
    results-summary.tsx             # Combined results dashboard
    input-controls.tsx              # Shared input components

lib/utils/tools.ts                  # Tool registry (add gpu-calculator entry)
```

---

## Phase 1: Foundation (Types + Constants + Scaffolding)

**Goal**: Project setup, TypeScript types, hardware/model constants, page scaffolding.

**Spec sections to read**: 1 (stack & patterns), 2 (notation), 3.3 (model presets), 5.1 (optimizer table), 7 (GPU specs), 11.1-11.3 (inputs/outputs), 13 (file structure)

### What to build:

**1. Initialize Next.js project**: Next.js 15, React 19, TypeScript strict, Tailwind CSS 4, Framer Motion, next-themes, Lucide React.

**2. `types.ts`** — All interfaces for the entire calculator. Key types:
- `ModelArchitecture` — d, L, a, a_kv, d_ff, V, s, ffnType, normType, posEmbedding, tiedEmbeddings
- `MoEConfig` — E, topk, L_moe, E_s, loadBalanceFactor
- `GPUSpec` — name, vram, tflops (bf16/tf32/fp8), mem bandwidth, nvlink bandwidth, tdp, gpus_per_node
- `TrainingConfig` — all inputs from spec Section 11.2 (precision, optimizer, batch size, seq length, grad accum, checkpointing, flash attention, parallelism, MFU, cost, failure params, etc.)
- `PostTrainingConfig` — method, approach (full/LoRA/QLoRA/MeZO), LoRA config, PPO config, GRPO config
- `ParallelismConfig` — N_tp, N_pp, N_dp, N_cp, N_ep, zeroStage, framework, sequenceParallelism, VP
- Output types: `ParameterCounts`, `ComputeEstimate`, `MemoryBreakdown`, `ParallelismRecommendation`, `TrainingTimeEstimate`, `CostEstimate`, `CalculatorOutput`
- `Warning` — severity, category, message

Read the spec's notation table (Section 2) and input/output lists (Sections 11.2, 11.3) carefully to make sure every symbol and every input has a corresponding type field.

**3. `constants.ts`** — All static data tables:
- GPU specs: all 22 GPUs from spec Section 7 table + 7 Apple Silicon chips
- Model presets: all 10 models from spec Section 3.3 table (GPT-2 Small through DeepSeek V3)
- Optimizer profiles: map each optimizer to its bytes-per-param breakdown (spec Section 5.1 table). Derive Φ and K_opt for both fp32 and bf16 gradient precision.
- Quick Mode lookup table: param ranges → heads/layers (spec Section 11.1)
- Chinchilla coefficients: all rows from spec Section 4.3 sensitivity table
- MFU defaults: from spec Section 6.3 table
- Cloud pricing presets: from spec Section 8.1
- Default values for every TrainingConfig field

**4. Page scaffolding**:
- `page.tsx` — following the tool page pattern from spec Section 1
- `gpu-calculator-embed.tsx` — dynamic import wrapper with SSR disabled
- `gpu-calculator.tsx` — skeleton component with dark/light mode pattern (spec Section 1) and tab navigation
- Register tool in `lib/utils/tools.ts` (spec Section 13)

### Validation:
- `npm run build` succeeds
- Page renders at `/tools/gpu-calculator`
- Dark/light mode works
- Types compile under strict mode

---

## Phase 2: Core Calculation Engine

**Goal**: Implement ALL formulas as pure TypeScript functions. No React, no UI.

Three sub-tasks that CAN run in parallel (separate files, no conflicts):

### Phase 2A: Parameter Counting + Compute Estimation

**File**: `formulas/compute.ts`
**Spec sections to read**: 3 (all), 4 (all)

**Functions to implement**:
- `calculateParameterCount(arch, moe)` → ParameterCounts — Section 3.1 (dense), 3.4 (MoE active vs total params, shared experts)
- `estimateParametersQuick(totalParams)` → ModelArchitecture — Section 3.2 + 11.1 Quick Mode heuristic
- `calculateFLOPs(params, config, arch, moe)` → ComputeEstimate — Section 4.1 PaLM formula (`6Ψ + 12Lds`), MoE load balance, embedding exclusion
- `calculateChinchillaAnalysis(params, tokens, uniqueTokens)` — Section 4.3 loss prediction, optimal allocation, coefficient row selection by D/N ratio
- `calculateCriticalBatchSize(loss, batchTokens)` — Section 4.4
- `analyzeDataRepetition(totalTokens, uniqueTokens)` — Section 4.5

**Key correctness points**:
- Use `Ψ_active` (not `Ψ_total`) in compute formula for MoE (Section 4.1)
- Embedding exclusion: input embedding = 0 FLOPs (lookup, not matmul). For untied embeddings, subtract V×d from Ψ_flops (Section 4.1)
- The `d` in attention term `12Lds` is the Q/K/V projection width (`n_heads × d_head`), not necessarily d_model (Section 4.1)
- Select Chinchilla coefficient row based on D/N ratio (Section 4.3 sensitivity table)

### Phase 2B: Memory Estimation

**File**: `formulas/memory.ts`
**Spec sections to read**: 5 (all), 10 (all for post-training memory)

**Functions to implement**:
- `getOptimizerProfile(optimizer, gradPrecision)` → Φ, K_opt, β_grad — from Section 5.1 table
- `calculateModelStateMemory(params, config)` — Section 5.1 + 5.2 (ZeRO stages 0-3, HYBRID_SHARD, MoE+ZeRO interaction, SP+optimizer sharding)
- `calculateActivationMemory(arch, config, moe)` — Section 5.3. Must handle all combinations of:
  - Checkpointing: none / selective / full / partial
  - TP/SP layout: N_tp=1 / N_tp>1 without SP / N_tp>1 with SP
  - Flash Attention: on/off (removes O(s²) attention term)
  - AMP autocast: on/off (coefficients 34→36, 5→6)
  - Context parallelism: replace s with s/N_cp
  - MoE layers: FFN activation scales by topk/E
  - Output logits tensor, transient recomp working memory
- `calculateCommunicationBuffers(params, config, arch)` — Section 5.4 (prefetch buffers, logit peak, overlap comm, torch.compile, TP backward buffer)
- `calculateTotalMemoryPerGPU(...)` → MemoryBreakdown — Section 5.5 (apply 1.04x CUDA alignment, framework overhead)
- `calculateLoRAMemory(...)`, `calculateQLoRAMemory(...)`, `calculateDPOMemory(...)`, `calculatePPOMemory(...)`, `calculateGRPOMemory(...)` — Section 10

**Key correctness points**:
- The d_ff correction for activation memory: the spec says "replace 24 with `4 × d_ff / d`" but the 24 = 8 (attention TP-split) + 16 (FFN TP-split). The correct interpretation is replace the FFN portion: `8 + 4 × d_ff / d`. Verify: for d_ff=4d this gives 8+16=24 ✓
- Post-training formulas in the spec use Φ=16 (bf16 grads) as examples. Implementation must use the user's actual Φ from their optimizer/gradient precision choice.
- Usable GPU memory = VRAM × 0.9 (or 0.8 for vanilla PyTorch)

### Phase 2C: Training Time + Cost

**File**: `formulas/cost.ts`
**Spec sections to read**: 6 (all), 8 (all), 10.3 (generation time), 10.5 (post-training compute)

**Functions to implement**:
- `calculateTrainingTime(compute, config)` → TrainingTimeEstimate — Section 6.1. Use BF16 TFLOPS for F_peak; FP8 uses bf16 × speedup factor (NOT raw FP8 TFLOPS); FP32 on Ampere+ uses TF32 TFLOPS.
- `calculateFailureAdjustedTime(theoreticalDays, config)` — Section 6.5 closed-form formula
- `calculateCost(time, config)` → CostEstimate — Section 8 (compute + storage + failure overhead)
- `getDefaultMFU(params, numGPUs)` — Section 6.3 table lookup
- `calculatePostTrainingCompute(method, params, config)` — Section 10.5 (SFT: 6Ψ, DPO: 8Ψ, PPO: ~20Ψ, GRPO: ~10Ψ)
- `calculateGenerationTime(params, config, batchGen, nTokens, sPrompt)` — Section 10.3 (prefill + decode, memory-bound vs compute-bound)

**Key correctness point**: Do NOT adjust C for activation recomputation when using MFU. MFU already captures checkpointing throughput loss. Section 6.1 emphasizes this in bold.

### Phase 2 Validation:
Run against spec Section 15 test cases:
- Test 1: LLaMA 7B — model states = 36.85 GB/GPU, compute = 40.2 ZFLOPS, time ≈ 131 days
- Test 2: LLaMA 70B — model states ≈ 12 GB, time ≈ 77 days
- Test 3: 7B LoRA — Ψ_lora = 16.8M, total ≈ 16-18 GB
- Test 5: ZeRO paper Table I — all 4 cases match

---

## Phase 3: Parallelism Recommendation Engine

**File**: `formulas/parallelism.ts`
**Spec sections to read**: 9 (all), 5.2 (ZeRO+PP compatibility table), 5.7 (PP constraints)
**Depends on**: Phase 2B memory functions (needs to check "does this config fit?")

**Functions to implement**:
- `recommendParallelism(params, arch, config, gpu, numGPUs, moe)` → ParallelismRecommendation — the full decision tree from Section 9 (DP → ZeRO → TP → EP → PP → CP)
- Constraint validators: TP divisibility (a, a_kv, d_ff), PP divisibility (L or L+2), ZeRO+PP compatibility, world size, microbatch minimums, hidden dim alignment, vocab padding
- `calculatePipelineBubble(N_pp, microbatches, VP)` — Section 5.7
- `scoreConfigurations(configs)` — Section 9 throughput scoring heuristic

**Key correctness points**:
- ZeRO-2/3 incompatible with PP in DeepSpeed (Section 9 compatibility table)
- PCIe-only GPUs: prefer ZeRO-3 over TP (Section 9 PCIe check)
- Prefer lowest ZeRO stage that fits (ZeRO-1 > ZeRO-2 > ZeRO-3 for throughput)
- MoE: N_ep × N_tp ≤ gpus_per_node; prefer EP over TP for expert layers

### Validation:
- LLaMA 7B on 8× H100 → ZeRO-1 + DP=8
- LLaMA 70B on 256× H100 → TP=8, PP=4, DP=8, ZeRO-1
- Test 4: PPO 70B → min ~35 GPUs for H100

---

## Phase 4: UI Input Components

**Spec sections to read**: 11.2 (pretraining inputs), 11.3 (post-training inputs), 12 (all UI/UX)
**Depends on**: Phase 1 only. Can start in parallel with Phase 2/3.

**Files to build**:
- `input-controls.tsx` — NumberInput (debounced 300ms), SliderInput, SelectInput, ToggleInput, CollapsibleSection, TooltipIcon. All with dark/light mode.
- `model-selector.tsx` — Quick/Preset/Detailed tabs. Quick = param count + tokens. Preset = dropdown from constants. Detailed = full architecture form + MoE toggle.
- `gpu-selector.tsx` — Grouped dropdown (Datacenter/Consumer/AMD/Apple Silicon). Shows specs on selection. Custom GPU option. Warns for PCIe + TP and pre-Ampere + BF16.
- `pretraining-panel.tsx` — All inputs from spec Section 11.2 (core inputs always visible, advanced inputs in collapsible section)
- `post-training-panel.tsx` — All inputs from spec Section 11.3

### Validation:
- All inputs render in dark and light mode
- Conditional fields show/hide correctly (MoE, LoRA, failure params)
- No strict TypeScript errors

---

## Phase 5: UI Output Components + Visualizations

**Spec sections to read**: 11.2 outputs list, 12.3 (visualizations), 12.2 (visual design)
**Depends on**: Phase 1 only. Can start in parallel with Phase 2/3.

**Files to build**:
- `memory-breakdown-bar.tsx` — Horizontal stacked bar (SVG, no charting libraries). Segments: params, grads, optimizer, activations, buffers, overhead, free. Hover tooltips. Framer Motion animation. Red warning if exceeds capacity.
- `gpu-utilization-gauge.tsx` — Circular/bar gauge. Green <70%, yellow 70-90%, red >90%.
- `parallelism-layout.tsx` — Grid showing DP×TP×PP (+ EP for MoE). Colored per dimension.
- `results-summary.tsx` — Dashboard displaying all outputs from spec Section 11.2 (19 items). Includes warnings panel with severity styling.

### Validation:
- Renders with mock data matching Test Case 1
- Animations work
- Responsive: desktop side-by-side, mobile stacked (spec Section 12.5, breakpoints 640px / 1024px)

---

## Phase 6: Integration + Wiring + Polish

**Spec sections to read**: 12.4 (interactivity), 14 (validation), 15 (test cases)
**Depends on**: All previous phases.

**What to build**:
- State management in `gpu-calculator.tsx`: useState for inputs, useMemo for all derived calculations (synchronous, pure math). No submit button — real-time updates.
- Calculation pipeline: inputs → debounce → resolveArchitecture → calculateParams → calculateFLOPs → recommendParallelism → calculateMemory → calculateTime → calculateCost
- Input validation: all rules from spec Section 14
- Responsive layout: desktop >1024px side-by-side, tablet 640-1024px stacked, mobile <640px single column
- Export: "Copy as text" (markdown), "Copy as JSON"

### Validation — run ALL test cases from spec Section 15:
1. LLaMA 7B: model states = 36.85 GB/GPU, compute = 40.2 ZFLOPS, time ≈ 131 days
2. LLaMA 70B: TP=8 PP=4 DP=8, model states ≈ 12 GB, time ≈ 77 days
3. 7B LoRA: Ψ_lora = 16.8M, total ≈ 16-18 GB
4. PPO 70B: peak 2,520 GB, min 35 GPUs
5. ZeRO paper Table I: all 4 cases match
6. External: Llama 3 405B → ~30.84M GPU-hours at 38-43% MFU

---

## Dependency Graph

```
Phase 1 (Foundation)
  ├── Phase 2A (Compute)  ──┐
  ├── Phase 2B (Memory)   ──┼── Phase 3 (Parallelism) ──┐
  ├── Phase 2C (Cost)     ──┘                            │
  ├── Phase 4 (UI Inputs)  ─────────────────────────────┼── Phase 6 (Integration)
  └── Phase 5 (UI Outputs) ────────────────────────────-┘
```

- 2A, 2B, 2C can run in parallel
- 4 and 5 can run in parallel with 2 and 3
- 3 must wait for 2B
- 6 must wait for everything
