# Appendix D — Warning routing, results relocation, export surface, a11y

*Generated 2026-06-05 from line-anchored code analysis. Part of spec/ux-redesign-plan.md.*

# APPENDIX D — Warning Routing + Results Relocation + Export/A11y

*All line refs are `components/gpu-calculator/gpu-calculator.tsx` unless prefixed. Verified against the 6,657-line file as of this recon. The warning predicates are math/validation logic and are **untouchable** — only `severity` label, placement, and tone change (plan §7).*

---

## D.1 Warning system: how it's assembled today

Every `Warning` (type at `types.ts:384-396`: `{severity: "info"|"warning"|"critical", category: "memory"|"precision"|"parallelism"|"compute"|"data"|"hardware"|"cost"|"generation", message: string}`) ends up in **one flat array** on the output object (`output.warnings`, `types.ts:469` / `:486`). There are **201 `.push()` sites** total (deduping to ~120 distinct messages). They are produced by these emitters and merged in this order:

| Emitter (fn) | Lines | Pushes | Feeds |
|---|---|---|---|
| `addPrecisionSupportWarnings` | 327-388 | 5 | pretraining + post-training |
| `addFP8KernelSpeedupWarnings` | 431-455 | 2 | both |
| `addCustomGPUThroughputWarnings` | 456-551 | 6 (+ nested helper loops at :504, :544) | both |
| generic field validators `addIntegerCountWarning`/`addPositiveIntegerWarning`/`addOptionalPositiveNumberWarning`/`addOptionalPositiveIntegerWarning`/`addNonNegativeIntegerWarning` | 553-647 | 5 (category is a **parameter** — see D.3) | both |
| `addParameterScaleWarnings` | 648-672 | 2 | both |
| `addPostTrainingInputWarnings` | 740-1408 | **50** | post-training only |
| `addArchitectureDimensionWarnings` | 1465-1547 | 3 (+ enum loop :1469) | both |
| `addKVHeadValidationWarnings` | 1548-1593 | 4 | both |
| `addManualStateShardDivisibilityWarnings` | 2489-2562 | 2 | pretraining (manual) |
| `generateInputWarnings` | 3713-4762 | **102** | pretraining only |
| Inline in component memos: DeepSpeed transient spike (`5561`), memory-fit (`5871`, `5885`), target-time/GPU mismatch (`5857`, `5979`), token-vs-batch (`5892`, `5900`, `5907`, `5926`, `5951`, `5957`, `5990`, `6001`) | 5561, 5855-6001 | ~14 | pretraining |
| Inline post-training memos: partial-batch (`6275`, `6281`), PPO/GRPO step (`6294`, `6307`), generation (`6341`, `6358`, `6381`) | 6275-6381 | ~7 | post-training |

**Rendering today:** all of these are dumped, unsorted-by-location, into a single `WarningsPanel` (`results-summary.tsx:218-265`) rendered **dead last** in both result columns (`results-summary.tsx:634` pretraining, `:778` post-training). The panel sorts by severity (`critical→warning→info`, `:220-223`) and color-codes via `SEVERITY_META` (`:173-216`). **`SEVERITY_META.critical.label === "Error"` (`:203`) — this is the string the plan renames to "Critical." (UI only; the separate `WARNING_LABEL` at `gpu-calculator.tsx:4780` feeds the text export and stays frozen — see D.7.)**

> **Severity distribution:** 114 critical, 30 warning, 54 info (3 are computed by ternary at the call site). **Category distribution:** compute 69, parallelism 36, memory 27, data 17, hardware 15, cost 14, precision 12, generation 6.

---

## D.2 Routing rule (the only thing the implementer needs to internalize)

The plan (§3 VerdictBand, component-plan row `results-summary.tsx`, Phase 1 + Phase 5) defines routing **by severity, then category**. You do **not** need to route 201 messages by hand — route by this matrix:

| Severity | NEW SURFACE | Mechanism |
|---|---|---|
| **`critical`** | **Verdict band** (`verdict-band.tsx`) — surfaced at top, sticky. Plan §3 line 47: "critical warnings surface here." | Filter `output.warnings.filter(w => w.severity === "critical")`, render in/under the sticky band. The `!memory.fits` criticals (D.4) additionally drive the amber band + "Fix for me." |
| **`warning`** | **Owning layer**: ⚠-count chip on that layer's header (closed state) **+** inline callout inside the layer body (open state). Map by `category` → layer via **D.5 table**. | Group `warnings.filter(w => w.severity==="warning")` by `category`; each layer reads its own slice for chip count + inline render. |
| **`info`** | **Footnote area** of the owning layer (de-emphasized) — plan: "info → footnote area." Same category→layer map. | `warnings.filter(w => w.severity==="info")`, grouped by category, rendered as small print at the bottom of the owning layer. |

Keep the existing `SEVERITY_META` color tokens (`results-summary.tsx:173-216`) — just relabel `critical` from `"Error"` → `"Critical"` and relocate the render. The dedup key used by the current panel is `${severity}-${category}-${index}` (`:234`); preserve uniqueness when splitting across surfaces.

**Important:** critical warnings frequently mean "estimates are disabled" (the message text says so, e.g. `:795`, `:818`, `:841`, `:1121`, `:4116`). When any critical is present the verdict band must reflect a degraded/blocked state, not a green ✓. This dovetails with Phase 5's "keep last-valid output dimmed + fix-field banner" behavior.

---

## D.3 The category→layer map (warning category → owning Layer per plan §3)

`Warning.category` maps cleanly onto the 8-layer IA. This is the table that lets `warning` + `info` route to the right layer header chip / footnote:

| `Warning.category` | Owning Layer (plan §3) | Layer # |
|---|---|---|
| `memory` | Memory & feasibility | **Layer 1** |
| `cost` | Cost detail & failures (failure-model + checkpoint cost warnings) | **Layer 7** |
| `parallelism` | Parallelism | **Layer 3** |
| `precision` | Precision & optimizer | **Layer 5** |
| `data` | Data & scaling | **Layer 6** |
| `compute` | Mixed — see note below | **Layer 4 / 5 / 8** |
| `hardware` | GPU-related → Essentials/verdict (GPU is in Essentials); custom-GPU spec errors → Layer 5/7 | Essentials + L5/L7 |
| `generation` | Post-training only (PPO/GRPO rollout) → Layer 2 (Performance) + the post-training method section | L2 + Essentials method |

**`compute` is the one ambiguous category** (69 pushes, the biggest bucket) because it's overloaded. Sub-route by message origin:
- Architecture validity (`addArchitectureDimensionWarnings`, `addKVHeadValidationWarnings` :1465-1593; messages like `:1540` "Hidden dimension d must be divisible…", `:1580` KV-head, `:1586` GQA grouping) → **Layer 4 (Model architecture)**.
- MoE-structure (`:3974`-`:4034`, `:3935`; "MoE total experts E…") → **Layer 8 (MoE)**.
- Optimizer/micro-batch/MFU (`:4116` pretraining-optimizer-invalid, `:4195` micro-batch≤2, `:4207` MFU override, `:1123` post-training optimizer-invalid) → **Layer 5 (Precision & optimizer)**.
- Quick-mode / preset caveats (`:3780`, `:3816`, `:867`, `:883`, `:987`, `:1013`) → **Layer 4** or **Essentials** (model preset notes area).
- Token-vs-batch step rounding (`:5951`, `:5957`, `:6275`, `:6281`, `:6294`, `:6307`) carry category `data`/`compute` → **Layer 6 (Data)** for token-ratio, **Layer 2 (Performance)** for step-count.

Since all criticals go to the verdict band regardless, this sub-routing only matters for `warning`/`info` severities — a smaller set (the compute `warning`/`info` items are listed in D.6).

---

## D.4 Critical warnings that drive `!memory.fits` (verdict band amber + "Fix for me")

The verdict band's amber/"Fix for me" state is **NOT** keyed off the warning array — it reads **`output.memory.fits`** directly (`types.ts:367` / `:376`; plan §3 line 47 and `verdict-band.tsx` spec). The corresponding critical *warnings* that explain the failure live here and should render **inside/under the band**:

| Line | Variants | Trigger | Drives |
|---|---|---|---|
| **5871** (pretraining) | 3-way ternary: `totalExceedsUsable` → "Per-GPU memory (X GB) exceeds usable capacity (Y GB)." / `floorExceedsUsable` → "The largest parameter-unit working set (X GB) exceeds usable capacity…" / else generic non-fit | `!memoryBreakdown.fits` (`:5865`) | verdict amber; "Fix for me" sets `numGPUs := minGPUsNeeded` |
| **6319** (post-training) | data-parallel-mode variant: "Per-GPU memory (X GB) exceeds usable capacity (Y GB). Split the global batch / add GPUs…" | `requiredGpuEstimate.mode==="data-parallel" && numGPUsNeeded!==null` | verdict amber; "Fix for me" → `numGPUsNeeded` |
| **5885** (pretraining, `warning`) | "The largest parameter-unit working set (X GB) exceeds 80% of raw GPU memory…" | `minVRAMFloor > gpuCapacity*0.8` (`:5864`) | amber advisory in Layer 1 chip |
| **5561** (pretraining, computed sev) | DeepSpeed transient init spike, `transientFits`/`fits` ternary | DeepSpeed init buffer | info/warning in Layer 1 |

These four are the memory-feasibility story. Note the post-training fit-message at **6319** is distinct from the pretraining one at **5871** — both must be wired to the band.

---

## D.5 Warning inventory (distinct messages, grouped by category)

Deduped representative list with line, severity, trigger (one line), and current message/template. Generic `${...}` are interpolation. **NEW SURFACE = per D.2 (critical→band; warning→layer chip+inline; info→layer footnote) into the layer from D.3.**

### memory → Layer 1 (criticals → band)
| Line | Sev | Trigger | Message (verbatim/template) |
|---|---|---|---|
| 859 | crit | `chunkedCrossEntropy` not bool | "Chunked cross-entropy must be true or false." |
| 1277 | crit | QLoRA bits ∉ {4,8} | "QLoRA quantization bits must be 4 or 8." |
| 4143 | crit | `torchCompile` not bool | "torch.compile must be true or false." |
| 4346 | crit | bad ZeRO comm bucket mode | "ZeRO communication bucket mode must be HF auto, DeepSpeed defaults, or custom." |
| 4353 | crit | `overlapComm` not bool | "ZeRO communication overlap must be true or false." |
| 4387 | crit | partial ckpt depth < 1 | "Partial checkpointing depth must be at least 1." |
| 4403 | crit | partial ckpt depth > max/stage | "Partial checkpointing depth must not exceed ${n} transformer layer(s) per pipeline stage for the current PP=${pp}…" |
| 4726 | crit | optimizer offload w/o ZeRO | "Optimizer offload requires ZeRO-1, ZeRO-2, ZeRO-3, or an equivalent FSDP sharding strategy." |
| 4733 | crit | param offload w/o ZeRO-3/FULL_SHARD | "Parameter offload requires ZeRO-3 or FSDP FULL_SHARD / HYBRID_SHARD." |
| **5871** | crit | `!memory.fits` (pretraining) | (3 variants — see D.4) → **VERDICT BAND** |
| **6319** | crit | `!memory.fits` (post-training, DP) | (see D.4) → **VERDICT BAND** |
| 3804 | warn | fp32 CPU init > node RAM | "Standard fp32 CPU initialization would materialize about ${n} TB of parameters per node before sharding…" |
| 4746 | warn | CPU offload enabled | "CPU offloading reduces GPU memory pressure but slows training…" |
| 5885 | warn | VRAM floor > 80% raw mem | (see D.4) |
| 875, 1061, 1184, 1213, 1227, 4174, 4181, 4188, 4561, 4752 | info | modeling-assumption notes (chunked-CE activation, DP-replica memory, LoRA RL reference-sharing, FSDP mixed-precision, AMP, MoE-under-PP, CPU-offload-scope) | (long modeling caveats) → **Layer 1 footnote** (FSDP/AMP/CPU ones may belong in Layer 5 footnote — route by sub-topic) |
| 1253, 5561 | (ternary) | QLoRA host-RAM / DeepSpeed transient spike | → Layer 1 |

### precision → Layer 5 (criticals → band)
| Line | Sev | Trigger | Message |
|---|---|---|---|
| 333 | crit | GPU lacks BF16, precision=bf16 | "${gpu.name} does not support BF16. Select FP16/FP32 or hardware with BF16 support; estimates are disabled…" |
| 381 | crit | GPU lacks FP8, precision=fp8 | "${gpu.name} does not support FP8 kernels…estimates are disabled…" |
| 440 | crit | FP8 speedup ∉ [1.0,2.0] | "FP8 kernel speedup factor must be between 1.0x and 2.0x." |
| 448 | crit | bad FP8 storage mode | "FP8 storage mode must be TransformerEngine or MS-AMP." |
| 1085 | warn | adamw-fp8 fallback | `adamWFP8FallbackMessage` (var; from `getAdamWFP8FallbackMessage` :389) → Layer 5 (also drives OverrideBadge per plan) |
| 341, 362, 371, 1094, 4159 | info | FP16 loss-scaling / FP32-TF32 / no-FP32-TFLOPS heuristic / FP8 storage info / FSDP grad-upcast | → Layer 5 footnote |

### parallelism → Layer 3 (criticals → band)
| Line | Sev | Trigger | Message |
|---|---|---|---|
| 4079 | crit | manual parallelism on single-device | "Manual multi-rank parallelism is unavailable on single-device hardware; estimates force DP=TP=PP=CP=EP=1." |
| 4149 | crit | bad framework enum | "Parallelism framework must be Megatron, DeepSpeed, PyTorch FSDP, or Hugging Face Trainer." |
| 4214 | crit | bad inter-node BW mode | "Inter-node bandwidth mode must be HDR, NDR, or a positive custom GB/s value." |
| 4424 | crit | non-positive parallel degrees | "Manual parallelism degrees must be positive finite values." |
| 4439 | crit | non-integer degrees | "Manual parallelism degrees must be integers." |
| 4464/4467/4496/4503/4521/4549/4646 | crit | divisibility validators (TP/PP/world-size/ZeRO-PP/TP-EP-SP/microbatch/CP) | message from helper: `tp.message`/`pp.message`/`ws.message`/`zp.message`/`tpEpSp.message`/`mb.message`/`cp.message` (from `validate*Divisibility`) → **BAND** |
| 4510 | crit | FSDP SHARD_GRAD_OP+PP outside AFAB | "FSDP SHARD_GRAD_OP / HYBRID_SHARD_ZERO2 with PP is only modeled under the AFAB fallback…" |
| 4571 | crit | TP > local HB group | "N_tp=${n} exceeds the local high-bandwidth group size of ${g}." |
| 4577 | crit | EP>1 on dense | "Expert parallelism is only valid for MoE models; set N_ep=1 for dense models." (also 5990 dup) |
| 4588 | crit | N_ep ∤ E | "N_ep=${n} must divide the total expert count E=${e}." |
| 4602 | crit | EDP not integer | "N_ep=${n} must divide ${...} (${num}) so expert data parallelism is an integer." |
| 4616 | crit | TP×EP > local HB group | "N_tp × N_ep must stay within the local high-bandwidth group (${g}) for expert traffic." |
| 4448 | warn | hidden-dim alignment | `hiddenAlignment.message` |
| 4655, 4666, 4676, 4705 | warn | CP<2K tok/rank, CP cross-node traffic, TP on PCIe, pipeline bubble high | (see messages) → Layer 3 chip+inline |
| 2538, 2555, 4469, 4475, 4489, 4527, 4534, 4626, 4682, 4689, 4711, 4717 | info | shard-padding, embedding-aware PP, vocab padding, AFAB notes, EP all-to-all, NVLink-bridge assumptions, bubble rule-of-thumb, ZeRO-3 overhead | → Layer 3 footnote |

### compute → Layers 4/5/8 (criticals → band) — *sub-route per D.3*
| Line | Sev | Trigger | Message / target layer |
|---|---|---|---|
| 749,768,781,795,803,818,829,841,849 | crit | base-model / method / approach / MoE validity (post-training) | various "must be…/estimates are disabled" → BAND |
| 898,963,1022,1134,1140,1146,1164,1171,1285,1294,1357,1394 | crit | post-training field validity (trainable %, seq len, batch, LoRA modules/rank/alpha, PPO) | → BAND |
| 1123 | crit | post-training optimizer/approach mismatch | 3-variant `message` var (MeZO/optimizer) → BAND |
| 1477,1540,1559,1571,1580,1586 | crit | architecture validity (tied emb, d÷heads, d_head, KV heads, GQA) | → BAND; (these belong to **Layer 4** for chip purposes) |
| 1470 | crit | invalid arch enum | `getInvalidArchitectureEnumMessages` loop var → Layer 4 |
| 3744,3750,3759,3769 | crit | pretraining param/mode/preset/quick validity | → BAND |
| 3942,3949 | crit | micro-batch / grad-accum < 1 | → BAND (Layer 5) |
| 3974-4034 | crit | MoE field validity (E, topk, L_moe, E_s, load-balance, FFN sizes) | → BAND (chip = **Layer 8**) |
| 4116 | crit | fine-tuning-only optimizer in pretraining | "${name} is fine-tuning only and is not a valid pretraining optimizer…" → BAND (Layer 5) |
| 4207 | crit | MFU override out of range | "MFU override must be greater than 0 and at most the calibrated 70% upper range." → BAND (Layer 5) |
| 658,666,3788,3794 | warn | param scale <1M / >10T | "…fewer than 1M parameters." / "…exceeds 10T parameters…" → Layer 4 |
| 912,5892,6307 | warn | partial-FT compute caveat / PaLM-formula attention / fewer independent items than GPUs | → Layer 5 / Layer 2 |
| 867,883,927,969,987,1000,1013,1266,1341,3780,3816,3935,3967,4195,6294 | info | modeling caveats (param-mode, MFU calibration, MeZO, seq-len range, MLA, sliding-window, QLoRA slowdown, GRPO reward, quick-mode, MoE FFN, micro-batch≤2, PPO step) | → owning layer footnote (Layer 4/5/8 per topic) |

### data → Layer 6 (criticals → band)
| Line | Sev | Trigger | Message |
|---|---|---|---|
| 941 | crit | dataset < 1 unit | "Dataset size must be at least 1 ${unit}." |
| 955 | crit | epochs ≤ 0 | "Epoch count must be positive." |
| 3835 | crit | total tokens ≤ 0 | "Total training tokens must be positive." |
| 3849 | crit | unique tokens ≤ 0 | "Unique token count must be positive." |
| 3902 | crit | >5000× Chinchilla | "Extreme overtraining (>5000x Chinchilla). Standard scaling law coefficients are not calibrated…" |
| 3916 | crit | excessive epochs | "Training for ${n} epochs — additional repetition is effectively wasted compute." |
| 3841,3876,3893,3909,3922 | warn | <200B tokens / <200B effective / below power-law / >500× / diminishing-returns epochs | → Layer 6 chip+inline |
| 3860,3928,5951,5957,6275,6281 | info | sub-epoch, MoE scaling guidance, tokens<batch (partial step), non-integer batch multiple | → Layer 6 footnote (step-count ones also relevant to Layer 2) |

### hardware → Essentials / verdict / Layer 5-7
| Line | Sev | Trigger | Message |
|---|---|---|---|
| 463 | crit | bad GPU input mode | "GPU input mode must be preset or custom." |
| 476,493,505,537 | crit | custom-GPU spec invalid (memory/TFLOPS/bandwidth/metadata/gpusPerNode) | "${label} GPU … must be positive…" → BAND; chip = Layer 5/7 (custom-GPU detail) |
| 1034 | crit | numGPUs < 1 | "GPU count must be at least 1." → BAND (GPU is in Essentials) |
| 1051 | crit | single-device GPU, numGPUs>1 | "${gpu.name} only supports single-device execution." → BAND |
| 4100 | crit | target days ≤ 0 | "Target training days must be positive when set." |
| 545 | warn | sparse-throughput helper messages | var `message` (from `getSparseThroughputWarningMessages`) |
| 4063 | warn | numGPUs > 100,000 | "GPU count exceeds 100,000." |
| 5979 | warn | target-time ≠ estimated-time | "Target training time is ${t} days, but the selected layout estimates ${e} days…" |
| 6001 | warn | 16K+ GPUs failure-rate caveat | "At 16K+ GPUs, the default 1% instance-day failure-rate assumption may understate…" |
| 4091, 5857 | info | target-days auto-only / target-start-vs-effective GPU count | → Essentials / Layer 3 footnote |

### cost → Layer 7 (criticals → band)
| Line | Sev | Trigger | Message |
|---|---|---|---|
| 1076 | crit | $/GPU-hr negative/non-finite | "Cost per GPU-hour must be a non-negative finite value." → BAND (also Essentials field) |
| 4233 | crit | cloud instance preset unresolved | "Selected cloud instance preset could not be resolved." |
| 4242 | crit | instance GPU ≠ hardware preset | "${provider} ${type} uses ${gpu}; clear the instance preset or switch the hardware preset to match." |
| 4272,4287,4296,4309 | crit | retention/storage-price/dataset-GB/failure-params negative | various "cannot be negative / must be non-negative finite…" |
| 4323,4338 | crit | failure recovery needs ckpt freq / retention | "Failure recovery needs a positive checkpoint frequency…" / "…at least one retained checkpoint…" |
| 5900 | crit | failure-adjusted time diverges | "Failure-adjusted training time diverges for the current failure rate, recovery time, checkpoint cadence, and cluster size." → BAND |
| 4259 | warn | instance bills N GPUs/instance | "${provider} ${type} bills ${n} GPUs per instance; … so compute cost includes the…" |
| 5907 | warn | failure overhead >2× time | "Failure overhead more than doubles training time (${m}x)…" |
| 5926 | warn | ckpt freq faster than step cadence | "Checkpoint frequency (${f}/day) is faster than the optimizer-step cadence…" → Layer 7 chip+inline |

### generation → Layer 2 + post-training method (criticals → band)
| Line | Sev | Trigger | Message |
|---|---|---|---|
| 1202 | crit | bad KV-cache precision | "KV cache precision must be BF16, FP16, or INT8." |
| 1315 | crit | GRPO group size < 2 | "GRPO group size must be an integer of at least 2." |
| 6341 | warn | generation capacity | `formatGenerationCapacityWarning(...)` (var from :3017) → Layer 2 |
| 1193, 6358, 6381 | info | prompt-prefill not modeled / DP serving replicas / decode bandwidth-bound | → Layer 2 footnote |

### Generic field validators (category is a PARAMETER) — `addIntegerCountWarning` etc. (553-647)
| Line | Sev | Message template | Routing |
|---|---|---|---|
| 560 | crit | "${label} must be an integer." | category passed by caller → route by that category (D.3) → BAND |
| 581 | crit | "${label} must be a positive integer." | same |
| 600 | crit | "${label} must be positive when set." | same |
| 619 | crit | "${label} must be a positive integer when set." | same |
| 640 | crit | "${label} must be a non-negative finite integer." | same |

These five are called repeatedly across the codebase with `(value, label, category)` — the `category` field on the produced Warning is what D.3 routes on, so the implementer does not special-case them.

---

## D.6 Results relocation table (results-summary.tsx → new layer homes)

Every visible metric/visualization in `results-summary.tsx`, its current line, and its new Layer per plan §3. **Stat label renames** column flags the plain-word-first relabel (Phase 6 §7; do the rename here so labels land correctly). All formatters (`formatCost`, `formatCount`, `formatDuration`, `formatFLOPs`, `formatMemory`, `formatMultiplier`, `formatPercent`, `formatFractionPercent`, imported `:28-37`) are **reused as-is**.

### Pretraining (`PretrainingResults` :369-637)

| Current block / Stat | Line | New Layer | Label rename |
|---|---|---|---|
| `MemoryBreakdownBar` | 384 | **L1** (full-width row + caption) | — |
| `GpuUtilizationGauge` | 388 | **L1** (full) + mini in verdict band | — |
| Stat "Effective GPUs" | 392 | **L1** | "Effective GPUs" (keep; plain enough) |
| Stat "Minimum GPUs Needed" | 397 | **L1** | keep |
| Stat "Minimum VRAM Floor" | 399 | **L1** | "Minimum VRAM per GPU" |
| Stat "Maximum Micro-Batch" | 404 | **L1** | "Max micro-batch (b) per GPU" |
| ResultCard "Model and Compute" | 413 | split → **L4 (arch params)** + **L6 (Chinchilla/loss)** | section title dissolves into layers |
| Stat "Model Parameters" | 415 | **L4** (also feeds verdict context) | "Model parameters (N)" |
| Stat "Total FLOPs" | 420 | **L2** (Performance) | "Total training FLOPs" |
| Stat "Chinchilla Ratio" | 425 | **L6** | "Tokens-per-parameter ratio (Chinchilla)" |
| Stat "Attention Overhead" | 430 | **L4** | "Attention FLOP overhead" |
| Stat "Predicted Loss" | 435 | **L6** | "Predicted loss" |
| Stat "MoE Sparsity" (conditional) | 444 | **L8** | "MoE sparsity" |
| Chinchilla Recommendation prose | 453-459 | **L6** ("Show your work" narrative, plan §3 L6) | — |
| ResultCard "Parallelism Strategy" | 463 | **L3** | — |
| `ParallelismLayout` mesh | 465 | **L3** (uncapped + caption per Phase 2) | — |
| Layout string (`formatParallelism`) | 472 | **L3** | — |
| Recommendation `strategyLabel` | 481 | **L3** | — |
| Stat "Pipeline Bubble" | 487 | **L3** | "Pipeline bubble (idle %)" |
| Stat "Inter-node Bandwidth" | 492 | **L3** | keep |
| `reasoning[]` bullets | 502-516 | **L3** (plan §3: "reasoning[] bullets") | — |
| ResultCard "Training Performance" | 521 | **L2** | — |
| Stat "Training Time" | 523 | **L2** + **verdict band** (wall-clock days) | "Training time" |
| Stat "Throughput" | 532 | **L2** | "Throughput (tokens/sec)" |
| Stat "Global Batch Size" | 537 | **L2** | "Global batch size (B)" |
| Stat "Batch Compute Multiplier" | 542 | **L2** (or L6) | "Batch compute multiplier" |
| Data Repetition callout | 549-569 | **L6** | — (uses `dataRepetition.severity` tone) |
| ResultCard "Cost Estimate" | 572 | **L2** (compute/total) + **L7** (storage/failure/ckpt detail) | — |
| Stat "Compute Cost" | 574 | **L2** | "Compute cost" |
| Stat "Storage Cost" | 584 | **L7** | "Storage cost" |
| Stat "Failure Overhead" | 589 | **L7** | "Failure overhead cost" |
| Stat "Total Cost" (highlight) | 594 | **verdict band** (total cost) + **L2** | "Total cost" |
| Stat "Checkpoint Size" | 598 | **L7** | keep |
| Stat "Checkpoint Storage" | 603 | **L7** | keep |
| ResultCard "MoE Metrics" (conditional) | 611-632 | **L8** | — |
| Stat "Sparsity Ratio" / "Efficiency Gain" / "Load Balance Factor" | 614/618/622 | **L8** | keep |
| `WarningsPanel` | 634 | **removed** — routed per D.2 | "Error"→"Critical" |

### Post-training (`PostTrainingResults` :639-781)

| Current block / Stat | Line | New Layer | Label rename |
|---|---|---|---|
| `MemoryBreakdownBar` | 667 | **L1** | — |
| `GpuUtilizationGauge` | 671 | **L1** + mini in verdict | — |
| Stat "GPUs Needed" + `gpuRequirementSub` | 675 (sub 655-661) | **L1** + verdict (N×GPU) | keep |
| Stat "Free Headroom" | 684 | **L1** | "Free VRAM headroom" |
| Stat "Working Set" | 689 | **L1** | "Peak working set per GPU" |
| ResultCard "Memory Line Items" (`PostTrainingMemoryItems`) | 699-703 (component :290-353) | **L1** (plan §3 L1: "post: MemoryLineItems") | — |
| ResultCard "Training Time" | 705 | **L2** | — |
| Stat "Estimated Time" | 707 | **L2** + verdict (wall-clock) | "Estimated time" |
| Stat "Throughput" | 716 | **L2** | "Throughput (tokens/sec)" |
| Stat `output.stepTimeLabel` (dynamic) | 721 | **L2** | keep (dynamic label) |
| Stat "Failure Multiplier" | 729 | **L2** / **L7** | keep |
| ResultCard "Cost Estimate" | 740 | **L2** (compute/total) + **L7** | — |
| Stat "Compute Cost" | 748 | **L2** | "Compute cost" |
| Stat "Storage Cost" / "Failure Overhead" (conditional) | 762/767 | **L7** | — |
| Stat "Total Cost" (highlight) | 774 | **verdict band** + **L2** | "Total cost" |
| `WarningsPanel` | 778 | **removed** — routed per D.2 | "Error"→"Critical" |

**Reused sub-components** (no relocation, just re-parented into layers): `MemoryBreakdownBar` (`./memory-breakdown-bar`), `GpuUtilizationGauge`, `ParallelismLayout`, `PostTrainingMemoryItems` (local `:290`), `ResultCard` (`:114`), `Stat` (`:143`). Helper formatters `formatStorageFootprint` (`:41`), `formatBatchRelation` (`:49`), `formatPretrainingParameterSub` (`:65`), `formatParallelism` (`:90`), `isPretraining` (`:110`) all move with their consumers.

---

## D.7 Export surface spec (load-bearing — do NOT touch)

There is **no `buildSummaryText`/`serializeCalculatorOutput`-named pair as the brief assumed**; the actual export builders are:

**JSON export — `serializeCalculatorOutput(output)` (`:4864-4876`):** `JSON.stringify(output, replacer, 2)` over the **entire** `CalculatorOutput`. Replacer converts non-finite numbers → `null` (`:4868`). Emits every field of `PretrainingOutput`/`PostTrainingOutput` including `warnings[]`. **Field list = the full type** (`types.ts:443-470` / `:477-487`).

**Text/Markdown export — `generatePretrainingMarkdown(o)` (`:4878-4957`)** and **`generatePostTrainingMarkdown(o)` (`:4959-4992`):** hand-built Markdown. Field list:
- *Pretraining:* Model params (+active +TP-padded), Total FLOPs, Chinchilla ratio, Predicted loss, Attention overhead; Memory per GPU (parameters/gradients/optimizerStates/activations/buffers/total/usable/fits/minVRAMFloor); Parallelism (strategyLabel/pipelineBubble/interNodeBandwidth/effectiveNumGPUs/minGPUsNeeded); Batch (global/maxMicroBatch/critical/computeMultiplier/wastedCompute); Data Repetition (via `formatDataRepetitionMarkdown` :4804); Training Time (theoretical/failure-adjusted/throughput/steps); Cost (compute/actual/storage/failure/checkpoints/dataset/total); Warnings (via `formatWarningsMarkdown` :4786).
- *Post-training:* Memory per GPU (+line items via `formatPostTrainingMemoryItemsMarkdown` :4818), GPUs Needed (via `formatPostTrainingGPURequirementMarkdown` :4845), Training Time (estimated/throughput/step labels), Cost (compute/total), Warnings.

**Confirmation:** the export builders read **only `CalculatorOutput`** (the memoized pipeline result), never the DOM or `results-summary.tsx`. **Relocating display does NOT touch these builders** — they live at `:4763-4992`, entirely outside the shell JSX (`:6446-6657`) and the results component. They satisfy hard-invariant §0.3 ("Export values are load-bearing"). The only export-adjacent shared symbol is `WARNING_LABEL` (`:4780`, `critical:"Error"`) used by `formatWarningsMarkdown`; the plan renames the **UI** label to "Critical" but the **export** string is part of the serialized contract — **leave `WARNING_LABEL` as-is** unless the owner explicitly wants the text export changed (parity protocol §6 diffs these snapshots; changing it would break parity).

**Export buttons today:** in the sticky results-pane header (`:6597-6650`) — "Text" button → `handleCopyText` (`:6407-6416`), "JSON" → `handleCopyJSON` (`:6418-6425`); both write to `navigator.clipboard` and flash a `copied` state (`:5032`). **New home:** plan §3 FOOTER — `[ Text ] [ JSON ]` alongside `[ Expand all ] [ Compact ⇄ ]`. Move the two buttons + the `copied` state + the two `useCallback` handlers verbatim into the footer; they already reference `pretrainingOutput`/`postTrainingOutput`/`currentOutput`/`activeTab` which remain in scope.

---

## D.8 A11y checklist for the new shell

Recon facts: **zero `aria-live`/`role="status"`/`role="alert"` exist anywhere** (grep clean). **No `useReducedMotion` import anywhere** — framer-motion is used in 8 components (input-controls 12 sites, memory-breakdown-bar 9, results-summary 7, gpu-selector 5, parallelism-layout 4, model-selector 2, gpu-calculator 2, gpu-utilization-gauge 1) with JS-driven `animate`/spring/`height:auto`. **No keyboard shortcuts exist** (only Enter-to-commit in NumberInput, `input-controls.tsx:336-337`).

### Heading hierarchy
- Today: `<h1>` lives in the page wrapper (`app/tools/gpu-calculator/page.tsx:42`), calculator header is `<h2>` (`gpu-calculator.tsx:6468`), result cards `<h3>` (`results-summary.tsx:134`).
- New shell (plan §3): **exactly one `<h1>`** in `HeroBar` ("How many GPUs to train an LLM?", Fraunces) on `app/page.tsx`. **`<h2>` per Layer** (each Layer title) — convert `ResultCard`'s `<h3>` (`results-summary.tsx:134`) and the panel section headers to `<h2>` when they become layer headers, or keep cards as `<h3>` *under* their `<h2>` layer. **Verdict band is not a heading** (it's a status region — see aria-live). Tabs are `<button>`s, not headings (keep `:6522-6544`).
- Delete the old `<h1>`/`<h2>` in the removed `app/tools/**` and the placeholder `app/page.tsx:16` `<h1>` (plan §1 root-only routing).

### aria-live (verdict updates — currently silent)
- The `VerdictBand` must be a polite live region so screen-reader users hear fits/cost/time/GPU changes: wrap the band's value row in `role="status"` + `aria-live="polite"` + `aria-atomic="true"`. This is **net-new** (nothing in the repo announces today).
- **Critical warnings** routed to the band (D.2) should be `role="alert"` (assertive) so blocking errors interrupt — distinct from the polite verdict numbers.
- Do **not** make every Layer summary-line a live region (would spam SR on each keystroke); only the verdict band + critical alerts announce.

### Focus order
- Logical DOM order = visual order in the single column (plan §2 kills the dual-pane `lg:overflow-y-auto` at `:6564/:6591`, which today creates two independent scroll/focus contexts). Order: HeroBar (theme toggle, Dense-view) → IntentRow expander → phase tabs → VerdictBand ("Fix for me" button, AssumptionsLedger chip) → Essentials controls → Layer headers (each a focusable disclosure button) → footer (Expand all / Compact / Text / JSON).
- Each `Layer` header is a `<button aria-expanded>` (extend `CollapsibleSection` which already has `aria-expanded`, `input-controls.tsx:672`). Collapsed-layer bodies must be removed from the tab order (the existing `AnimatePresence` unmounts content at `:704-711`, so closed layers are already out of focus order — preserve that).
- "Fix for me" (verdict) and "fix <field> ▸" (Phase 5 banner) must move focus to the culprit control after action (`scrollIntoView` + `.focus()`), and the culprit's layer must auto-open first.
- ⌘K palette (Phase 6 `settings-search.tsx`): on match, focus the control AND open its layer; trap focus while the palette is open; Esc closes and restores focus to the opener.

### prefers-reduced-motion (gap confirmed)
- `app/globals.css:177` neutralizes only CSS `animation`/`transition` durations via `!important`. It does **NOT** stop framer-motion's JS animations (`animate`, spring, `height:0→auto`), which run off `requestAnimationFrame`. So the toggle spring (`input-controls.tsx:565-569`), CollapsibleSection height/opacity (`:694-711`), tooltip pop (`:183-199`), and every `motion.section`/`motion.div` in results-summary/memory-bar/parallelism still animate under reduced-motion.
- **Fix (plan Phase 6):** add `const reduce = useReducedMotion()` (framer-motion hook) in **input-controls.tsx first** (plan explicitly flags it), then in the new layer/verdict/results components, and gate `animate`/`transition` (e.g. `transition={reduce ? { duration: 0 } : {...}}`, and skip the `height:auto` expand animation when `reduce`). This is net-new — no component imports it today.

### Keyboard map (all NEW — none exist today)
- **`e`** → Expand all / collapse all toggle (footer `[ Expand all ]`).
- **`c`** → Compact ⇄ density toggle (footer; the `density` UI state, plan component-plan).
- **`d`** → Dense view (HeroBar affordance; persisted — equivalent to expand-all + compact, plan Expert journey line 99).
- **`⌘K` / `Ctrl-K`** → settings-search palette (Phase 6).
- Implement single-key (e/c/d) shortcuts on a document-level `keydown` **guarded against typing context** (ignore when `event.target` is an `<input>/<select>/<textarea>`/`contenteditable`, and when a modifier other than the intended one is held) — the only existing `keydown` is the NumberInput Enter handler (`input-controls.tsx:336`), so there's no collision but the guard is mandatory to avoid hijacking text entry. Provide a visible affordance/tooltip for each (don't rely on hidden shortcuts alone).
- Tabs (Pretraining/Post-Training) should support arrow-key navigation per the WAI-ARIA tablist pattern; today they're plain `<button>`s (`:6522`) without `role="tablist"`/`role="tab"`/`aria-selected` — add these roles when rewriting the shell.

---

**Files read for this artifact:** `spec/ux-redesign-plan.md`; `components/gpu-calculator/gpu-calculator.tsx` (warnings :327-1594, :2489-2562, :3713-4762, :5561, :5855-6001, :6275-6381; export builders :4763-4992; copy handlers + shell JSX :6407-6657); `components/gpu-calculator/components/results-summary.tsx` (full, :1-782); `components/gpu-calculator/types.ts` (`Warning` :384-396, outputs :443-487); `components/gpu-calculator/components/input-controls.tsx` (motion sites); `app/globals.css:174-186`; `app/` tree + heading/metadata greps.

**Brief-name corrections for the implementing agent:** export builders are `serializeCalculatorOutput` + `generatePretrainingMarkdown`/`generatePostTrainingMarkdown` (not `buildSummaryText`); the line ranges in the original brief were stale (file is 6,657 lines — warnings span :327-6381, exports :4763-4992, shell JSX :6446-6657, not the :740-1407/:4864-4992 originally cited).
