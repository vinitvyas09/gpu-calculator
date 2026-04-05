# GPU Calculator Implementation — Agent Prompts

Copy-paste one prompt at a time into a Claude Code session with `@sw-implementer`. Wait for it to finish and validate before moving to the next.

**Order**: Phase 1 → 2A/2B/2C (can be parallel) → 3 → 4 & 5 (can be parallel) → 6

The `sw-implementer` agent has the `frontend-design` skill built in — it applies design guidelines automatically when building UI components (Phases 4, 5, 6).

---

## Phase 1: Foundation

```
@sw-implementer

You are implementing Phase 1 of the GPU calculator.

Read the implementation plan: spec/implementation-plan.md (Phase 1 section)
Read these spec sections: 1 (stack & patterns), 2 (notation), 3.3 (model presets), 5.1 (optimizer table), 7 (GPU specs + Apple Silicon), 11.1-11.3 (all inputs and outputs), 13 (file structure)

Your job for this phase:

1. Initialize a Next.js 15 project in this repo with: React 19, TypeScript strict, Tailwind CSS 4, Framer Motion, next-themes, Lucide React. Do NOT delete the existing spec/ directory or plan.md.

2. Create components/gpu-calculator/types.ts — define ALL TypeScript interfaces for the entire calculator. Every input from spec Section 11.2 and 11.3 must have a corresponding field. Every output must have a type. Read the notation table (Section 2) carefully. The plan lists the key types needed.

3. Create components/gpu-calculator/constants.ts — embed ALL static data tables from the spec:
   - All 22 GPU specs from Section 7 table + all 7 Apple Silicon chips
   - All 10 model presets from Section 3.3 table
   - Optimizer memory profiles from Section 5.1 table (derive Φ and K_opt for both fp32 and bf16 grad precision)
   - Quick Mode lookup table from Section 11.1
   - Chinchilla coefficients from Section 4.3 sensitivity table (all rows)
   - MFU defaults from Section 6.3
   - Cloud pricing from Section 8.1
   - Sensible defaults for every TrainingConfig field

4. Create the page scaffolding:
   - app/tools/gpu-calculator/page.tsx (follow pattern from spec Section 1)
   - app/tools/gpu-calculator/gpu-calculator-embed.tsx (dynamic import, ssr: false)
   - components/gpu-calculator/gpu-calculator.tsx (skeleton with dark/light mode pattern from Section 1, tab nav)
   - Register in lib/utils/tools.ts

5. Verify: npm run build succeeds, types compile under strict mode.

Commit when done.
```

---

## Phase 2A: Parameter Counting + Compute Estimation

```
@sw-implementer

You are implementing Phase 2A of the GPU calculator.

Read the implementation plan: spec/implementation-plan.md (Phase 2A section)
Read these spec sections: 3 (all of Section 3), 4 (all of Section 4)
Read the existing types: components/gpu-calculator/types.ts
Read the existing constants: components/gpu-calculator/constants.ts

Your job: Create components/gpu-calculator/formulas/compute.ts

Implement these pure TypeScript functions (no React, no DOM):

1. calculateParameterCount(arch, moe) → ParameterCounts
   - Dense: Section 3.1 (attention with GQA, FFN standard/SwiGLU, norms, embeddings)
   - MoE: Section 3.4 (total params, active params, shared experts)
   - Active params EXCLUDE input embedding (lookup, not matmul) but INCLUDE output projection

2. estimateParametersQuick(totalParams) → ModelArchitecture
   - Section 3.2 quick estimate + Section 11.1 Quick Mode lookup table
   - Solve d = sqrt(Ψ / (12 × L)), round to nearest multiple of 128

3. calculateFLOPs(params, config, arch, moe) → ComputeEstimate
   - PaLM formula: C = (6Ψ_active + 12Lds) × D (Section 4.1)
   - d in attention term = n_heads × d_head (not necessarily d_model)
   - MoE: use Ψ_active, apply load_balance_factor to expert FLOPs only
   - Compute attention overhead percentage

4. calculateChinchillaAnalysis(params, tokens, uniqueTokens)
   - Loss prediction: L(N,D) = E + A/N^alpha + B/D^beta (Section 4.3)
   - Select coefficient row by D/N ratio from sensitivity table
   - Power-law optimal: D_optimal = 8.62 × N^1.041
   - Compute-optimal allocation formulas

5. calculateCriticalBatchSize(loss, batchTokens) — Section 4.4

6. analyzeDataRepetition(totalTokens, uniqueTokens) — Section 4.5

Validate against spec Section 15:
- Test 1: 6 × 6.7e9 × 1e12 = 4.02e22 FLOPs
- Test 2: 6 × 70e9 × 2e12 = 8.4e23 FLOPs

Run npx tsc --noEmit to verify. Commit when done.
```

---

## Phase 2B: Memory Estimation

```
@sw-implementer

You are implementing Phase 2B of the GPU calculator.

Read the implementation plan: spec/implementation-plan.md (Phase 2B section)
Read these spec sections: 5 (ALL of Section 5 — this is long, read it all), 10 (all of Section 10 for post-training memory)
Read the existing types: components/gpu-calculator/types.ts
Read the existing constants: components/gpu-calculator/constants.ts

Your job: Create components/gpu-calculator/formulas/memory.ts

This is the most complex formula file. Implement these pure functions:

1. getOptimizerProfile(optimizer, gradPrecision) → { Φ, K_opt, β_grad }
   - From Section 5.1 table. Must be exact — these propagate everywhere.

2. calculateModelStateMemory(params, config)
   - No sharding: Φ × Ψ (Section 5.1)
   - ZeRO 0-3: Section 5.2 formulas using Φ, K_opt, β_grad
   - HYBRID_SHARD: use N_dp_intra = gpus_per_node
   - SP + optimizer sharding: replace N_dp with N_dp × N_sp (Section 5.2)
   - MoE + ZeRO: shard expert and non-expert params separately (N_dp vs N_edp)
   - TP effect: Ψ_per_gpu ≈ Ψ/N_tp
   - PP effect: use most-loaded stage = Ψ_transformer/N_pp + Ψ_embedding (Section 5.7)

3. calculateActivationMemory(arch, config, moe)
   - Handle ALL combinations of: checkpointing mode × TP/SP layout × Flash Attention × AMP autocast × Context Parallelism × MoE
   - Base coefficients: 34 linear, 5*a*s/d attention (Korthikanti). AMP autocast: 36, 6.
   - d_ff correction: the 24 in TP formulas = 8 (attention) + 16 (FFN for d_ff=4d). Correct FFN portion to 4*d_ff/d, giving 8 + 4*d_ff/d.
   - Flash Attention: drops O(s²) attention term
   - CP: replace s with s/N_cp everywhere
   - MoE layers: FFN activation × topk/E
   - Total: per-stage × min(N_pp, num_microbatches) + output logits tensor
   - Transient recomp working memory for full checkpointing

4. calculateCommunicationBuffers(params, config, arch) — Section 5.4

5. calculateTotalMemoryPerGPU(...) → MemoryBreakdown
   - Sum all components. Apply 1.04x CUDA alignment. Framework overhead (2-5 GB).
   - Usable VRAM = gpu.vram × 0.9 (or 0.8 for vanilla PyTorch)
   - Min GPU floor = Ψ_largest_layer × (β + β_grad)

6. Post-training memory functions:
   - calculateLoRAMemory, calculateQLoRAMemory (Section 10.1)
   - calculateDPOMemory (Section 10.2, including LoRA-as-reference optimization)
   - calculatePPOMemory (Section 10.3, including KV cache for generation)
   - calculateGRPOMemory (Section 10.4)

Validate against spec Section 15:
- Test 1: ZeRO-1 DP=8 bf16 grads → 4×6.7e9 + 12×6.7e9/8 = 36.85 GB
- Test 3: LoRA 7B → Ψ_lora = 16.8M, total ≈ 16-18 GB
- Test 5: ZeRO paper Table I — all 4 cases must match exactly

Run npx tsc --noEmit to verify. Commit when done.
```

---

## Phase 2C: Training Time + Cost

```
@sw-implementer

You are implementing Phase 2C of the GPU calculator.

Read the implementation plan: spec/implementation-plan.md (Phase 2C section)
Read these spec sections: 6 (all of Section 6), 8 (all of Section 8), 10.3 (generation time), 10.5 (post-training compute)
Read the existing types: components/gpu-calculator/types.ts
Read the existing constants: components/gpu-calculator/constants.ts

Your job: Create components/gpu-calculator/formulas/cost.ts

Implement these pure functions:

1. calculateTrainingTime(compute, config) → TrainingTimeEstimate
   - T = C / (N_gpu × F_peak × MFU) — Section 6.1
   - F_peak: BF16 TFLOPS for bf16; TF32 for fp32 on Ampere+; bf16 × fp8SpeedupFactor for fp8 (NOT raw FP8 TFLOPS)
   - CRITICAL: Do NOT adjust C for activation recomputation. MFU already captures it.
   - MFU: use override if provided, else auto-default from Section 6.3 table

2. calculateFailureAdjustedTime(theoreticalDays, config)
   - Section 6.5 formula. Check for divergence (denominator ≤ 0).

3. calculateCost(time, config) → CostEstimate
   - Compute cost, checkpoint storage cost, failure overhead cost — Section 8

4. getDefaultMFU(params, numGPUs) — Section 6.3 table

5. calculatePostTrainingCompute(method, params, config) — Section 10.5

6. calculateGenerationTime(params, config, batchGen, nTokens, sPrompt)
   - Section 10.3: prefill (compute-bound) + decode (memory-bound vs compute-bound)

Validate:
- Test 1: 4.02e22 / (8 × 989e12 × 0.45) ≈ 131 days
- Test 2: 8.4e23 / (256 × 989e12 × 0.50) ≈ 77 days

Run npx tsc --noEmit to verify. Commit when done.
```

---

## Phase 3: Parallelism Recommendation Engine

```
@sw-implementer

You are implementing Phase 3 of the GPU calculator.

Read the implementation plan: spec/implementation-plan.md (Phase 3 section)
Read these spec sections: 9 (ALL of Section 9), 5.2 (ZeRO+PP compatibility table), 5.7 (PP constraints + bubble)
Read the existing code:
  - components/gpu-calculator/types.ts
  - components/gpu-calculator/constants.ts
  - components/gpu-calculator/formulas/memory.ts (you need to call memory functions to check "does this config fit?")

Your job: Create components/gpu-calculator/formulas/parallelism.ts

Implement:

1. recommendParallelism(params, arch, config, gpu, numGPUs, moe) → ParallelismRecommendation
   - Full decision tree from Section 9: DP → ZeRO stages → TP → EP → PP → CP
   - At each step, call memory functions to check if the config fits in GPU VRAM
   - Prefer lowest ZeRO stage that fits (throughput order)
   - PCIe GPUs: prefer ZeRO-3 over TP
   - MoE: add EP, respecting E%N_ep==0 and N_ep×N_tp ≤ gpus_per_node
   - DeepSpeed: ZeRO-2/3 incompatible with PP
   - Long sequences >32K: add CP with N_cp = seq_len/8192

2. Constraint validators:
   - validateTPDivisibility(N_tp, a, a_kv, d_ff)
   - validatePPDivisibility(N_pp, L) — check both L%N_pp==0 and (L+2)%N_pp==0
   - validateZeroPPCompatibility(zeroStage, N_pp, framework)
   - validateWorldSize(config)
   - validateMicrobatches(numMicrobatches, N_pp, VP)
   - validateHiddenDimAlignment(d) — d%128==0
   - calculateVocabPadding(V, N_tp) — ceil(V/(128×N_tp))×(128×N_tp)

3. calculatePipelineBubble(N_pp, microbatches, VP) — Section 5.7

4. scoreConfigurations(configs, ...) — Section 9 throughput scoring

Validate:
- 7B on 8×H100 → ZeRO-1 + DP=8
- 70B on 256×H100 → TP=8, PP=4, DP=8, ZeRO-1
- PPO 70B → needs ~35+ GPUs minimum

Run npx tsc --noEmit to verify. Commit when done.
```

---

## Phase 4: UI Input Components

```
@sw-implementer

You are implementing Phase 4 of the GPU calculator. This is a UI phase — apply your frontend-design skill for polished, distinctive components.

Read the implementation plan: spec/implementation-plan.md (Phase 4 section)
Read these spec sections: 1 (component patterns — dark/light mode, file conventions), 11.2 (pretraining inputs list), 11.3 (post-training inputs list), 12 (all UI/UX requirements)
Read the existing code:
  - components/gpu-calculator/types.ts
  - components/gpu-calculator/constants.ts
  - components/gpu-calculator/gpu-calculator.tsx (skeleton from Phase 1)

Build these files. All must be "use client", follow the dark/light mode pattern from spec Section 1, and use the types from types.ts.

1. components/gpu-calculator/components/input-controls.tsx
   - NumberInput (debounced 300ms per spec Section 12.4), SliderInput, SelectInput, ToggleInput, CollapsibleSection, TooltipIcon

2. components/gpu-calculator/components/model-selector.tsx
   - Three tabs: Quick (param count + tokens) | Preset (dropdown from MODEL_PRESETS) | Detailed (full architecture form + MoE toggle)

3. components/gpu-calculator/components/gpu-selector.tsx
   - Grouped dropdown: Datacenter NVIDIA / Consumer NVIDIA / AMD / Apple Silicon
   - Shows key specs on selection. Custom GPU option.
   - Warn for PCIe + TP>1 and pre-Ampere + BF16

4. components/gpu-calculator/components/pretraining-panel.tsx
   - All inputs from spec Section 11.2. Core inputs always visible, advanced inputs (items 16-32) in a collapsible section.

5. components/gpu-calculator/components/post-training-panel.tsx
   - All inputs from spec Section 11.3. Conditional fields for LoRA, PPO, GRPO.

Verify: npm run build succeeds, components render in both themes. Commit when done.
```

---

## Phase 5: UI Output Components + Visualizations

```
@sw-implementer

You are implementing Phase 5 of the GPU calculator. This is a UI phase — apply your frontend-design skill. The memory breakdown bar is the hero visualization.

Read the implementation plan: spec/implementation-plan.md (Phase 5 section)
Read these spec sections: 1 (component patterns), 11.2 (outputs list — items 1-19), 12.2 (visual design), 12.3 (key visualizations)
Read the existing code:
  - components/gpu-calculator/types.ts (especially MemoryBreakdown, CalculatorOutput, Warning)

Build these files. "use client", dark/light mode, Framer Motion for animations, pure SVG for charts (NO external charting libraries — spec Section 1).

1. components/gpu-calculator/components/memory-breakdown-bar.tsx
   - Horizontal stacked bar chart (SVG). Segments: parameters, gradients, optimizer states, activations, buffers, overhead, free headroom.
   - Each segment distinctly colored. Percentage + absolute GB on hover. Framer Motion animate transitions. Red warning if exceeds GPU capacity.

2. components/gpu-calculator/components/gpu-utilization-gauge.tsx
   - Circular gauge or progress bar. Green <70%, yellow 70-90%, red >90%. Animated.

3. components/gpu-calculator/components/parallelism-layout.tsx
   - Grid visualization of DP×TP×PP (+ EP for MoE). Colored by dimension.

4. components/gpu-calculator/components/results-summary.tsx
   - Dashboard showing ALL 19 output items from spec Section 11.2.
   - Warnings panel with severity-based styling (info/warning/error).
   - Post-training results section.
   - Responsive: desktop side-by-side, mobile stacked (breakpoints 640px, 1024px per spec Section 12.5)

Verify: npm run build succeeds, components accept the right prop types. Commit when done.
```

---

## Phase 6: Integration + Wiring

```
@sw-implementer

You are implementing Phase 6 (final phase) of the GPU calculator. This is both integration and UI — apply your frontend-design skill for the final layout and polish.

Read the implementation plan: spec/implementation-plan.md (Phase 6 section)
Read these spec sections: 12.4 (interactivity), 12.5 (responsive design), 14 (validation & edge cases), 15 (ALL test cases)
Read ALL existing code — this phase wires everything together:
  - components/gpu-calculator/types.ts
  - components/gpu-calculator/constants.ts
  - components/gpu-calculator/formulas/compute.ts
  - components/gpu-calculator/formulas/memory.ts
  - components/gpu-calculator/formulas/cost.ts
  - components/gpu-calculator/formulas/parallelism.ts
  - components/gpu-calculator/gpu-calculator.tsx
  - All UI components in components/gpu-calculator/components/

Your job: Wire inputs → calculation engine → outputs in the main component.

1. State management in gpu-calculator.tsx:
   - useState for all input state
   - useMemo chains for derived calculations: resolveArchitecture → calculateParams → calculateFLOPs → recommendParallelism → calculateMemory → calculateTime → calculateCost
   - All calculations are synchronous pure math — no async needed
   - Number inputs debounced at 300ms. No submit button — real-time updates.
   - When parallelism = "auto", use recommendParallelism(). When manual, use user values.

2. Input validation — implement ALL rules from spec Section 14:
   - Param count positive, warn <1M or >10T
   - Tokens positive, warn Chinchilla ratio <1 or >500 or >5000
   - TP divides a, a_kv, d_ff
   - PP divides L (or L+2)
   - World size constraint
   - 1F1B microbatch minimum
   - etc.

3. Responsive layout — spec Section 12.5:
   - Desktop >1024px: side-by-side (inputs | results)
   - Tablet 640-1024px: stacked compact
   - Mobile <640px: single column

4. Export: "Copy as text" (markdown), "Copy as JSON"

5. VALIDATE against ALL test cases from spec Section 15:
   - Test 1: LLaMA 7B — states=36.85 GB/GPU, compute=40.2 ZFLOPS, time≈131 days
   - Test 2: LLaMA 70B — TP=8 PP=4 DP=8, states≈12 GB, time≈77 days
   - Test 3: 7B LoRA — Ψ_lora=16.8M, total≈16-18 GB
   - Test 4: PPO 70B — peak 2,520 GB, min 35 GPUs
   - Test 5: ZeRO paper — all 4 cases match
   - External: Llama 3 405B → ~30.84M GPU-hours at 38-43% MFU

Fix any integration issues. Verify npm run build succeeds. Commit when done.
```

---

## Notes

- **Parallel phases**: 2A, 2B, 2C can run simultaneously in separate terminals. 4 and 5 can run alongside 2 and 3.
- **Sequential deps**: Phase 3 needs Phase 2B (memory functions). Phase 6 needs everything.
- **If an agent gets stuck**: it should read the spec section more carefully. The spec has all formulas.
- **If types need updating**: any phase can update types.ts, but keep changes minimal and backward-compatible.
