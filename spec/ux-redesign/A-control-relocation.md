# Appendix A — Exhaustive control relocation table

*Generated 2026-06-05 from line-anchored code analysis. Part of spec/ux-redesign-plan.md.*

## Control Relocation Table — Pretraining + Post-Training (Calm Layers)

**Coverage:** Verified by grepping `<NumberInput|<SelectInput|<ToggleInput|<SliderInput|<CheckboxGroupInput` (JSX usages, excluding import blocks) across all four files. Editable-input counts: **pretraining-panel** 33 Number + 16 Select + 6 Toggle + 1 Slider = **56**; **post-training-panel** 15 Number + 8 Select + 1 Toggle + 1 CheckboxGroup = **25**; **model-selector** 12 Number + 6 Select + 2 Toggle = **20** (Quick/Preset readouts are `Stat`, listed separately); **gpu-selector** 7 Number + 6 Select + 4 Toggle = **17**. Plus **3 segmented mode-switches** (model Quick/Preset/Detailed, base-model Preset/By-Size, GPU Preset/Custom) + **1 parallelism Auto/Manual** `SelectInput`. Editable-control total = **122**; with the 24 read-only `Stat` rows in selectors (second table) the surface is ~146 user-facing slots. Every row below cites `file:line`. Files: `pretraining-panel.tsx` (PT), `post-training-panel.tsx` (POST), `model-selector.tsx` (MS), `gpu-selector.tsx` (GS).

### TABLE A — Pretraining tab

| # | Current label (exact) | Primitive | Defined at | Config field written | Visible when (exact) | NEW HOME | New label (plain-word-first) | Notes |
|---|---|---|---|---|---|---|---|---|
| P1 | *(segmented)* Quick / Preset / Detailed | TabButton ×3 | MS:329-338 (`MODE_TABS` MS:145-149) | `model.inputMode` (+ resets arch/moe on quick/preset) | always (Essentials model picker) | **Essentials** (model picker control) | "Quick · Preset · Detailed" | **detailedDraft fix applies** (plan §3, MS:208-210): `detailed` else-branch (MS:208) restores nothing — must shadow-copy arch/moe and restore on return to Detailed. Quick/Preset clobber `architecture`+`moe` (MS:194-207). |
| P2 | Total Parameters | NumberInput | MS:401 | `model.quickMode.totalParameters` → `resolveQuickMode` sets arch | `inputMode==="quick"` (MS:348) | **Essentials** | "Parameters" | Quick-mode only. Drives inferred-arch readout (Table B). compact/integer. |
| P3 | Training Tokens (D) | NumberInput | MS:413 | `totalTokens` (via `onQuickTokensChange`→`setTotalTokens` PT:341) | `inputMode==="quick"` AND `quickTokens!==undefined` (MS:411) | **Essentials** | "Total training tokens (D)" | Quick-mode duplicate of P15; syncs `uniqueTokens` when equal (PT:345-348). |
| P4 | Model Preset | SelectInput | MS:486 | `model.presetId` + arch + moe (MS:216-223) | `inputMode==="preset"` (MS:357) | **Essentials** | "Model" (searchable) | Make searchable/grouped; **render `preset.notes`** under picker (plan §3). |
| P5 | Hidden dim (d) | NumberInput | MS:598 | `architecture.d` (via `normalizeAttentionVariantHeads` MS:233-237) | `inputMode==="detailed"` (MS:365) | **Layer 4 Model architecture** | "Hidden size (d)" | Detailed-mode. step 64. |
| P6 | Layers (L) | NumberInput | MS:608 | `architecture.L` | detailed | **Layer 4** | "Layers (L)" | Caps `L_moe` (PT:1085) & checkpoint depth. |
| P7 | Attention heads | NumberInput | MS:617 | `architecture.a` (+ derives `a_kv`, MS:574-591) | detailed | **Layer 4** | "Attention heads (a)" | Coupling: sets `a_kv` per variant. |
| P8 | Head dim (d_head) | NumberInput | MS:626 | `architecture.d_head` | detailed | **Layer 4** | "Head size (d_head)" | Derived default = d/a (MS:628-629). |
| P9 | KV heads | NumberInput | MS:638 | `architecture.a_kv` | detailed | **Layer 4** | "Key/value heads (a_kv)" | **disabled unless `attentionVariant==="gqa"`** (MS:649); derived display for MHA/MQA/MLA (MS:639-647). |
| P10 | FFN dim (d_ff) | NumberInput | MS:656 | `architecture.d_ff` | detailed | **Layer 4** | "Feed-forward size (d_ff)" | Derived default depends on ffnType (MS:658). |
| P11 | Vocab size (V) | NumberInput | MS:667 | `architecture.V` | detailed | **Layer 4** | "Vocabulary size (V)" | — |
| P12 | FFN type | SelectInput | MS:680 | `architecture.ffnType` (`setFFNType` MS:297-320) | detailed | **Layer 4** | "Feed-forward type" | "moe" option couples to MoE enable (MS:298-305) → toggles Layer 8. |
| P13 | Norm type | SelectInput | MS:693 | `architecture.normType` | detailed | **Layer 4** | "Normalization" | Influences dense-FFN default (MS:42). |
| P14 | Positional encoding | SelectInput | MS:707 | `architecture.posEmbedding` | detailed | **Layer 4** | "Positional encoding" | — |
| P15 | Attention variant | SelectInput | MS:723 | `architecture.attentionVariant` (+ resets `a_kv` MS:728-729) | detailed | **Layer 4** | "Attention variant" | Controls P9 disabled-state. |
| P16 | Mixture of Experts | ToggleInput | MS:743 | `moe.enabled` + `architecture.ffnType` (`setMoeEnabled` MS:243-295) | detailed | **Layer 8 MoE** (control echoed in Essentials/Layer 4 to enable) | "Mixture of Experts (MoE)" | Seeds E/topk/L_moe/sizes (MS:266-292). Un-dims Layer 8. DECIDE: keep enable-toggle in Layer 4 (architecture) since it lives in the model block today; recommend **Layer 4** for the toggle, Layer 8 for all routing knobs. |
| P17 | Tied embeddings | ToggleInput | MS:751 | `architecture.tiedEmbeddings` | detailed | **Layer 4** | "Tied embeddings" | — |
| P18 | Dense FFN size | NumberInput | MS:776 | `moe.denseIntermediateSize` | detailed AND `moe.enabled` (MS:758) | **Layer 8 MoE** | "Dense FFN size" | **Duplicate** of P40 (same field). Two distant panels — merge into Layer 8 (plan §3 line 59). |
| P19 | Expert FFN size | NumberInput | MS:794 | `moe.expertIntermediateSize` | detailed AND `moe.enabled` (MS:758) | **Layer 8 MoE** | "Expert FFN size" | **Duplicate** of P41. Merge. |
| P20 | "MoE Overview" (collapsible) | CollapsibleSection | MS:759 | container only | detailed AND `moe.enabled` | **Layer 8 MoE** | (dissolve into Layer 8) | `defaultOpen` + badge `${E} experts`; wraps P18/P19. Becomes part of Layer 8 body. |
| P21 | Total tokens (D) | NumberInput | PT:480 | `totalTokens` (`setTotalTokens` PT:341, syncs uniqueTokens) | `!isQuickMode` i.e. `inputMode!=="quick"` (PT:478) | **Essentials** | "Total training tokens (D)" | In Quick mode this is hidden and edited via P3 (PT:503-511 explains). |
| P22 | Unique tokens (U) | NumberInput | PT:492 | `uniqueTokens` | always (PT:491) | **Layer 6 Data & scaling** | "Unique tokens (U)" | DECIDE: spec Essentials lists "total tokens" only; recommend **Layer 6** (Chinchilla/repetition context). Drives data-repetition math. |
| P23 | Precision | SelectInput | PT:519 | `precision` | always | **Layer 5 Precision & optimizer** | "Precision" | `fp8` reveals P49/P50; drives `effectiveOptimizerId` (PT:355). |
| P24 | Optimizer | SelectInput | PT:535 | `optimizer` | always | **Layer 5** | "Optimizer" | `adamw-fp8`→`adamw-mixed` substitution (PT:355-361) → **OverrideBadge** + ledger. |
| P25 | Gradient precision | SelectInput | PT:543 | `gradientPrecision` | always | **Layer 5** | "Gradient precision" | **disabled when `optimizerFixesGradientStorage`** (PT:550); option set + tooltip swap (PT:370-386). Derived value `gradientPrecisionValue` (PT:367). |
| P26 | Micro-batch size (b) | NumberInput | PT:555 | `microBatchSize` | always | **Layer 5** | "Micro-batch size (b)" | — |
| P27 | Sequence length (s) | NumberInput | PT:565 | `sequenceLength` | always | **Layer 4 Model architecture** | "Sequence length (s)" | step 128. Tracked to preset default (PT:175-198). Plan §3 lists seq len under Layer 4. |
| P28 | Grad accum steps (G) | NumberInput | PT:576 | `gradientAccumulationSteps` | always | **Layer 5** | "Gradient accumulation steps (G)" | — |
| P29 | Activation checkpointing | SelectInput | PT:588 | `activationCheckpointing` (+ `partialCheckpointDepth` PT:593-596) | always | **Layer 5** | "Activation checkpointing" | `"partial"` reveals P42 (PT:1354). |
| P30 | Flash Attention | ToggleInput | PT:612 | `flashAttention` | always | **Layer 4** | "Flash Attention" | Plan §3 lists flash attention under Layer 4. |
| P31 | *(segmented)* GPU Preset / Custom GPU | button ×2 | GS:126-144 | `hardware.inputMode` (`setMode` GS:69-78) | always (Essentials GPU picker) | **Essentials** (GPU picker control) | "Preset · Custom" | Custom reveals P56-P72 (CustomGPUForm). |
| P32 | GPU | SelectInput | GS:150 | `hardware.gpuId`+`gpu` (`setGPU` GS:80) + clears pricing preset (PT:204-247) | `inputMode==="preset"` (GS:147) | **Essentials** | "GPU" (searchable) | Make searchable over existing optgroups (plan §3). Coupling: switching GPU may re-sync `costPerGPUHour` to matching price preset (PT:222-240). |
| P33 | Number of GPUs | NumberInput | PT:637 | `hardware.numGPUs` (`setHw` PT:201) | always | **Essentials** | "Number of GPUs" | **disabled when `gpuCountDerivedFromTarget`** (PT:648); value swaps to `effectiveNumGPUs` (PT:638). Gates P46 (≥256). One-tap "Fix for me" writes this (`:= minGPUsNeeded`). |
| P34 | Target training days | NumberInput | PT:652 | `hardware.targetTrainingDays` (null when 0) | always | **Essentials** (beside #GPUs) | "Target training days (optional)" | RESOLVED: Essentials beside P33 (#GPUs) — tight coupling (PT:643-645); it derives/locks #GPUs when set. Must NOT go in Layer 2: Layer 2 is always-open output-only and the hard disclosure rule forbids inputs there. When set, "Fix for me" first clears this field, then writes numGPUs (plan §3). |
| P35 | Override MFU default | ToggleInput | PT:674 | `mfuOverride` (null↔`defaultMFU`) | always | **Layer 5** | "Override MFU estimate" | Gates P36. `defaultMFU` is a live `useMemo` (PT:421-434). OverrideBadge candidate. |
| P36 | MFU Override | SliderInput | PT:686 | `mfuOverride` | always (disabled unless override on) | **Layer 5** | "MFU override" | **disabled when `!hasMFUOverride`** (PT:695). Only `SliderInput` in app. |
| P37 | Cloud instance | SelectInput | PT:702 | `pricing.cloudInstanceId` (`setCloudInstance` PT:278-309) + switches GPU + `costPerGPUHour` | always | **Layer 7 Cost detail & failures** | "Cloud instance (optional)" | Heavy coupling: selecting an instance overwrites GPU preset + $/hr + failure GPU count (PT:294-308). Source-of-truth sub-label (plan §5/§3 L7). |
| P38 | Cloud pricing preset | SelectInput | PT:710 | `pricing.cloudPricingPresetId` + `costPerGPUHour` (PT:712-728) | always | **Layer 7** | "Pricing preset" | Couples with P39 (sets $/hr, clears instance). |
| P39 | Cost per GPU-hour | NumberInput | PT:734 | `pricing.costPerGPUHour` (clears preset+instance PT:736-742) | always | **Essentials** | "Cost per GPU-hour ($/hr)" | unit `$/hr`. Source-of-truth coupling w/ P37/P38 (plan §5). Also surfaced in Essentials per §3 line 50. |
| P40 | *(segmented)* Mode (Auto / Manual) | SelectInput | PT:754 | `parallelismMode` | always | **Layer 3 Parallelism** (header control) | "How to choose the GPU layout — Auto · Manual" | **Auto-open-on-manual** (plan §1, §3): switching to `manual` opens Layer 3 and persists. Drives `displayParallelism` (PT:417-420). |
| P41 | Framework | SelectInput | PT:828 | `parallelism.framework` (`setPar` PT:249-273) | `parallelismMode==="manual"` (PT:824) | **Layer 3** | "Framework" | `fsdp` derives `zeroStage` (PT:252-262). **Duplicate** of P52 (auto-mode copy). |
| P42 | Tensor parallel (N_tp) | NumberInput | PT:845 | `parallelism.N_tp` | `parallelismMode==="manual"` (PT:824) | **Layer 3** | "Tensor parallel (TP)" | Feeds GPU-selector PCIe warning (GS:91, `tpDegree`). |
| P43 | Pipeline parallel (N_pp) | NumberInput | PT:854 | `parallelism.N_pp` | manual | **Layer 3** | "Pipeline parallel (PP)" | Caps checkpoint depth (PT:147). |
| P44 | Data parallel (N_dp) | NumberInput | PT:863 | `parallelism.N_dp` | manual | **Layer 3** | "Data parallel (DP)" | — |
| P45 | ZeRO stage | SelectInput | PT:876 | `parallelism.zeroStage` | manual AND framework∈{deepspeed,hf_trainer} (PT:873-874) | **Layer 3** | "ZeRO stage" | Stage 3 forces overlap-comm (PT:454-458) → OverrideBadge. |
| P46 | FSDP strategy | SelectInput | PT:895 | `parallelism.fsdpStrategy` (→ derives zeroStage PT:260) | manual AND `framework==="fsdp"` (PT:893) | **Layer 3** | "FSDP strategy" | ZeRO-under-FSDP derivation (PT:117-131) → OverrideBadge/ledger. |
| P47 | Context parallel (N_cp) | NumberInput | PT:993 | `parallelism.N_cp` | inside Advanced AND `parallelismMode==="manual"` (PT:989) | **Layer 3** | "Context parallel (CP)" | Auto-mode shows read-only value (PT:944-947, Table B). |
| P48 | Expert parallel (N_ep) | NumberInput | PT:1003 | `parallelism.N_ep` | Advanced AND manual (PT:989) | **Layer 8 MoE** (per §3 "incl. EP") | "Expert parallel (EP)" | **disabled unless `moeEnabled`** (PT:1007). Plan §3 line 59: merge EP into Layer 8. DECIDE: also relevant to Layer 3 mesh — recommend **Layer 8** (spec is explicit) with mesh caption referencing it. |
| P49 | Virtual pipeline chunks (VP) | NumberInput | PT:1014 | `parallelism.VP` | Advanced AND manual (PT:989) | **Layer 3** | "Virtual pipeline chunks (VP)" | — |
| P50 | Sequence parallelism | SelectInput | PT:1024 | `parallelism.sequenceParallelism` | Advanced AND manual (PT:989) | **Layer 3** | "Sequence parallelism" | Auto-mode read-only (PT:977, Table B). |
| P51 | Total experts (E) | NumberInput | PT:1049 | `model.moe.E` | Advanced AND `moeEnabled` (PT:1044) | **Layer 8 MoE** | "Total experts (E)" | `moeEnabled` = `moe.enabled \|\| ffnType==="moe"` (PT:414-416). |
| P52 | Active experts (topk) | NumberInput | PT:1062 | `model.moe.topk` | Advanced AND `moeEnabled` | **Layer 8 MoE** | "Active experts per token (top-k)" | max = `E` (PT:1071). |
| P53 | MoE layers (L_moe) | NumberInput | PT:1076 | `model.moe.L_moe` | Advanced AND `moeEnabled` | **Layer 8 MoE** | "MoE layers (L_moe)" | max = `architecture.L` (PT:1085). |
| P54 | Shared experts (E_s) | NumberInput | PT:1090 | `model.moe.E_s` | Advanced AND `moeEnabled` | **Layer 8 MoE** | "Shared experts (E_s)" | min 0. |
| P55 | Load-balance factor | NumberInput | PT:1103 | `model.moe.loadBalanceFactor` | Advanced AND `moeEnabled` | **Layer 8 MoE** | "Load-balance factor" | range 1–2, step 0.05. |
| P56 | Dense FFN size | NumberInput | PT:1121 | `model.moe.denseIntermediateSize` | Advanced AND `moeEnabled` | **Layer 8 MoE** | "Dense FFN size" | **Duplicate** of P18 (same field). Merge — single control in Layer 8. |
| P57 | Expert FFN size | NumberInput | PT:1145 | `model.moe.expertIntermediateSize` | Advanced AND `moeEnabled` | **Layer 8 MoE** | "Expert FFN size" | **Duplicate** of P19. Merge. |
| P58 | Framework | SelectInput | PT:1180 | `parallelism.framework` | Advanced AND `parallelismMode==="auto"` (PT:1178) | **Layer 3** | "Framework" | **Duplicate** of P41 (manual copy). Show one Framework control in Layer 3 regardless of mode. |
| P59 | CPU offloading | SelectInput | PT:1199 | `cpuOffload` | always (inside Advanced) (PT:1198) | **Layer 5 Precision & optimizer** | "CPU offloading" | DECIDE: memory-state knob. Plan §3 lists offload nowhere explicitly; recommend **Layer 5** (optimizer/memory state) — alternatively Layer 1 controls. Coupling note in tooltip: param offload needs ZeRO-3/FSDP. |
| P60 | AMP autocast | ToggleInput | PT:1222 | `ampAutocast` | always (Advanced) | **Layer 5** | "AMP autocast" | Plan §3 line 56 lists AMP under Layer 5. |
| P61 | ZeRO communication buckets | SelectInput | PT:1231 | `zeroCommunication.mode` | always (Advanced) | **Layer 7 Cost detail & failures** | "ZeRO communication buckets" | `"custom"` reveals P63-P65 (PT:1264). Plan §3 line 58 puts ZeRO comm buckets in Layer 7. |
| P62 | Overlap communication | ToggleInput | PT:1253 | `zeroCommunication.overlapComm` | always (Advanced) | **Layer 3 Parallelism** | "Overlap communication" | **disabled + forced-on when `zero3ForcesOverlapComm`** (PT:1262); displays `effectiveOverlapComm` (PT:457) → OverrideBadge/ledger. DECIDE: comm-overlap is parallelism-coupled; recommend **Layer 3** (tied to ZeRO-3 derivation) though buckets sit in L7. |
| P63 | Allgather bucket (elements) | NumberInput | PT:1267 | `zeroCommunication.allgatherBucketSizeElements` | Advanced AND `zeroCommunication.mode==="custom"` (PT:1264) | **Layer 7** | "All-gather bucket (elements)" | — |
| P64 | Reduce bucket (elements) | NumberInput | PT:1282 | `zeroCommunication.reduceBucketSizeElements` | Advanced AND `mode==="custom"` | **Layer 7** | "Reduce bucket (elements)" | — |
| P65 | Prefetch bucket (elements) | NumberInput | PT:1297 | `zeroCommunication.prefetchBucketSizeElements` | Advanced AND `mode==="custom"` | **Layer 7** | "Prefetch bucket (elements)" | — |
| P66 | Inter-node bandwidth | SelectInput | PT:1317 | `interNodeBandwidth.mode` (`setInterNodeBandwidthMode` PT:317-330) | always (Advanced) | **Layer 3 Parallelism** | "Inter-node bandwidth" | `"custom"` reveals P67 (PT:1332). Plan §3 line 54 lists inter-node BW under Layer 3. |
| P67 | Custom bandwidth | NumberInput | PT:1334 | `interNodeBandwidth.customGBps` | Advanced AND `interNodeBandwidth.mode==="custom"` (PT:1332) | **Layer 3** | "Custom bandwidth (GB/s)" | unit GB/s. |
| P68 | Checkpointed layers/stage | NumberInput | PT:1356 | `partialCheckpointDepth` | Advanced AND `activationCheckpointing==="partial"` (PT:1354) | **Layer 5** | "Checkpointed layers per stage" | max = `maxCheckpointedLayersPerStage` (derived PT:459-460, PT:133-148). Tied to P29. Plan §3 line 56: "activation ckpt + depth" in Layer 5. |
| P69 | torch.compile | ToggleInput | PT:1372 | `torchCompile` | always (Advanced) | **Layer 5** | "torch.compile" | Plan §3 line 56. |
| P70 | Chunked cross-entropy | ToggleInput | PT:1380 | `chunkedCrossEntropy` | always (Advanced) | **Layer 5** | "Chunked cross-entropy" | Plan §3 line 56. |
| P71 | FP8 kernel speedup | NumberInput | PT:1391 | `fp8.kernelSpeedupFactor` | Advanced AND `precision==="fp8"` (PT:1388) | **Layer 5** | "FP8 kernel speedup" | "FP8 knobs" per §3 line 56. |
| P72 | FP8 storage mode | SelectInput | PT:1408 | `fp8.storageMode` | Advanced AND `precision==="fp8"` | **Layer 5** | "FP8 storage mode" | Feeds `effectiveOptimizerId` substitution (PT:359). |
| P73 | Checkpoint retention count | NumberInput | PT:1439 | `pricing.checkpointRetentionCount` | always (Advanced) | **Layer 7** | "Checkpoints to keep" | Plan §3 line 58. |
| P74 | Checkpoint freq | NumberInput | PT:1450 | `failureModel.checkpointFrequencyPerDay` | always (Advanced) | **Layer 7** | "Checkpoint frequency (/day)" | unit `/day`. Couples w/ failure recovery (must be >0 when failures on). |
| P75 | Storage price | NumberInput | PT:1469 | `pricing.storagePricePerGBMonth` | always (Advanced) | **Layer 7** | "Storage price ($/GB/mo)" | — |
| P76 | Dataset storage | NumberInput | PT:1480 | `pricing.datasetStorageGB` | always (Advanced) | **Layer 7** | "Dataset storage (GB)" | unit GB. |
| P77 | Failure rate | NumberInput | PT:1495 | `failureModel.failureRatePerInstancePerDay` | Advanced AND `effectiveNumGPUs >= 256` (PT:1492) | **Layer 7** | "Failure rate (/instance/day)" | Conditional on #GPUs ≥ 256. |
| P78 | Recovery time | NumberInput | PT:1515 | `failureModel.recoveryTimeHours` | Advanced AND `effectiveNumGPUs >= 256` (PT:1492) | **Layer 7** | "Recovery time (hours)" | Same conditional. |
| P79 | "Advanced Settings" (collapsible) | CollapsibleSection | PT:919 | container only | always | **dissolved** → its children distribute to Layers 3/5/7/8 | (removed) | badge `"17 options"` (PT:921). The single Advanced bucket is replaced by the layer stack; SubLabel groupings (PT:927/1046/1172/1350/1435) map to layer sections. |

**Pretraining CustomGPU sub-form (shared with Post-Training — appears when GPU mode = Custom, GS:158-159):**

| # | Current label | Primitive | Defined at | Config field | Visible when | NEW HOME | New label | Notes |
|---|---|---|---|---|---|---|---|---|
| G1 | Vendor | SelectInput | GS:275 | `hardware.gpu.vendor` | GPU `inputMode==="custom"` (GS:158) | **conditional-disclosure** under Essentials GPU picker → spills to **Layer 7**/Layer 1 detail | "Vendor" | DECIDE: custom-GPU form is bulky. Recommend a **"Custom GPU specs ▸" disclosure** opened from the Essentials picker (mirrors "Customize adapter" pattern); it is not one of the 8 layers. |
| G2 | Category | SelectInput | GS:288 | `hardware.gpu.category` | custom | same disclosure | "Category" | — |
| G3 | Memory type | SelectInput | GS:308 | `hardware.gpu.memoryType` | custom | same | "Memory type" | unified vs vram. |
| G4 | Half precision mode | SelectInput | GS:322 | `hardware.gpu.halfPrecisionFormat` | custom | same | "Half-precision format" | — |
| G5 | VRAM (GB) | NumberInput | GS:337 | `hardware.gpu.memoryGB` | custom | same | "VRAM (GB)" | Drives Layer 1 feasibility. |
| G6 | Dense BF16/FP16 TFLOPS | NumberInput | GS:344 | `hardware.gpu.halfPrecisionTFLOPS` | custom | same | "Dense BF16/FP16 (TFLOPS)" | — |
| G7 | Mem bandwidth (GB/s) | NumberInput | GS:352 | `hardware.gpu.memoryBandwidthGBps` | custom | same | "Memory bandwidth (GB/s)" | — |
| G8 | Dense TF32 TFLOPS | NumberInput | GS:359 | `hardware.gpu.tf32TFLOPS` (0→null) | custom | same | "Dense TF32 (TFLOPS)" | — |
| G9 | FP32 TFLOPS | NumberInput | GS:367 | `hardware.gpu.fp32TFLOPS` (0→null) | custom | same | "FP32 (TFLOPS)" | — |
| G10 | Dense FP8 TFLOPS | NumberInput | GS:375 | `hardware.gpu.fp8TFLOPS` (0→null) | custom | same | "Dense FP8 (TFLOPS)" | — |
| G11 | GPUs per node | NumberInput | GS:383 | `hardware.gpu.gpusPerNode` | custom | same | "GPUs per node" | — |
| G12 | Interconnect | SelectInput | GS:391 | `hardware.gpu.interconnect` | custom | same | "Interconnect" | PCIe+TP>1 → warning (GS:91). |
| G13 | Supports BF16 | ToggleInput | GS:405 | `hardware.gpu.supportsBF16` | custom | same | "Supports BF16" | — |
| G14 | Supports TF32 | ToggleInput | GS:411 | `hardware.gpu.supportsTF32` | custom | same | "Supports TF32" | — |
| G15 | Supports FP8 | ToggleInput | GS:417 | `hardware.gpu.supportsFP8` | custom | same | "Supports FP8" | Affects `effectiveOptimizerId` (PT:358). |
| G16 | Single-device only | ToggleInput | GS:423 | `hardware.gpu.singleDeviceOnly` | custom | same | "Single-device only" | — |

### TABLE A — Post-Training tab

| # | Current label | Primitive | Defined at | Config field | Visible when | NEW HOME | New label | Notes |
|---|---|---|---|---|---|---|---|---|
| T1 | *(segmented)* Preset / By Size | TabButton ×2 | MS:876-889 | `baseModel.inputMode` (`setMode` MS:836-854) | always (Essentials base-model picker) | **Essentials** (base-model control) | "Preset · By size" | `BaseModelSelector` (simpler than pretraining). No detailed mode → no detailedDraft issue here, but preset switch overwrites arch/moe (MS:841-867). |
| T2 | Base Model | SelectInput | MS:894 | `baseModel.presetId`+params+arch+moe (MS:856-867) | `inputMode==="preset"` (MS:892) | **Essentials** | "Base model" (searchable) | Switching syncs PPO/GRPO critic/reward param counts (POST:271-312). Render `preset.notes`. |
| T3 | Parameter Count | NumberInput | MS:902 | `baseModel.parameterCount` | `inputMode==="parameter-count"` (MS:900) | **Essentials** | "Parameters" | compact/integer. |
| T4 | Method | SelectInput | POST:435 | `method` | always | **Essentials** | "Method" | **disabled when `isMeZO`** (POST:444, forced to `sft`). Reveals PPO (T11-T13) / GRPO (T14-T15) blocks. Drives data-label set (POST:417). |
| T5 | Approach | SelectInput | POST:449 | `approach` (`setApproach` POST:344-370) | always | **Essentials** | "Approach" | Reveals LoRA block (T8-T10) when lora/qlora; sets optimizer to mezo (POST:351-357). Per plan §3: method extras reveal via Essentials "Customize adapter ▸" disclosure. |
| T6 | Trainable parameter % | NumberInput | POST:466 | `trainableParameterPercentage` (≥100→null) | `approach==="full" \|\| approach==="mezo"` (POST:463) | **conditional-disclosure** in Essentials | "Trainable parameters (%)" | unit %. Mutually exclusive with LoRA readout (T-readout below). |
| T7 | Rank (r) | NumberInput | POST:511 | `lora.rank` | `isLoRA` (approach∈{lora,qlora}) (POST:507) | **conditional-disclosure** "Customize adapter ▸" (Essentials) | "LoRA rank (r)" | Plan §3 line 65: LoRA reveals via "Customize adapter ▸". Feeds computed-footprint card (Table B). |
| T8 | Alpha | NumberInput | POST:521 | `lora.alpha` | `isLoRA` | "Customize adapter ▸" | "LoRA alpha" | — |
| T9 | Quantization bits | SelectInput | POST:531 | `lora.quantizationBits` | `approach==="qlora"` (POST:529) | "Customize adapter ▸" | "Quantization bits" | QLoRA only. |
| T10 | Target modules | CheckboxGroupInput | POST:549 | `lora.targetModules` | `isLoRA` (inside LoRA section POST:507) | "Customize adapter ▸" | "Target modules" | Only `CheckboxGroupInput` in app. Options `LORA_MODULE_OPTIONS` (POST:76-84). |
| T11 | Critic model params | NumberInput | POST:567 | `ppo.criticModelParameterCount` | `method==="ppo"` (POST:563) | **conditional-disclosure** (reveal on method=PPO) | "Critic model parameters" | Auto-synced from base model (POST:274-301). compact. |
| T12 | Reward model params | NumberInput | POST:584 | `ppo.rewardModelParameterCount` | `method==="ppo"` | PPO disclosure | "Reward model parameters" | Auto-synced (POST:278-301). |
| T13 | Update epochs | NumberInput | POST:601 | `ppo.updateEpochs` | `method==="ppo"` | PPO disclosure | "PPO update epochs" | range 1–32. |
| T14 | Group size (G) | NumberInput | POST:626 | `grpo.groupSize` | `method==="grpo"` (POST:622) | **conditional-disclosure** (reveal on method=GRPO) | "GRPO group size (G)" | min 2. |
| T15 | Reward model params | NumberInput | POST:637 | `grpo.rewardModelParameterCount` | `method==="grpo"` | GRPO disclosure | "Reward model parameters" | **Duplicate label** of T12 (different field: grpo vs ppo). 0 = rule-based. Auto-synced (POST:282-310). |
| T16 | Dataset size | NumberInput | POST:662 | `datasetSizeExamples` | always | **Essentials** | "Dataset size" | unit/tooltip vary by method (`getTrainingDataLabels` POST:207-251): examples/pairs/prompts. compact. |
| T17 | Epochs | NumberInput | POST:674 | `epochs` | always | **Essentials** | "Epochs" | — |
| T18 | Sequence length | NumberInput | POST:682 | `sequenceLength` | always | **Layer 4 Model architecture** | "Sequence length" | step 128; method-varying tooltip. DECIDE: spec post-training Essentials list omits seq len; recommend **Layer 4** to match pretraining, OR keep in a "Data" disclosure. |
| T19 | Batch size | NumberInput | POST:692 | `batchSize` | always | **Layer 5 Precision & optimizer** (or Data) | "Batch size" | method-varying tooltip. DECIDE: recommend **Layer 6 Data & scaling** to sit with dataset/epochs context; not in the 8-control Essentials list. |
| T20 | Precision | SelectInput | POST:712 | `precision` | always | **Layer 5** | "Precision" | `fp8` reveals T25/T26. |
| T21 | Optimizer | SelectInput | POST:727 | `optimizer` | always | **Layer 5** | "Optimizer" | **disabled + forced "mezo" when `isMeZO`** (POST:728-736); else `adamw-fp8`→`adamw-mixed` substitution (POST:378-384) → OverrideBadge. |
| T22 | Gradient precision | SelectInput | POST:740 | `gradientPrecision` | always | **Layer 5** | "Gradient precision" | **disabled when `optimizerFixesGradientStorage`** (POST:749); option/tooltip swap incl. MeZO "No gradients" (POST:393-411). |
| T23 | Chunked cross-entropy | ToggleInput | POST:753 | `chunkedCrossEntropy` | always | **Layer 5** | "Chunked cross-entropy" | — |
| T24 | KV cache precision | SelectInput | POST:761 | `kvCachePrecision` | always | **Layer 5** | "KV-cache precision" | Post-training-only field (constants.ts:1749). DECIDE: precision-family knob → Layer 5; acceptable. |
| T25 | FP8 kernel speedup | NumberInput | POST:776 | `fp8.kernelSpeedupFactor` | `precision==="fp8"` (POST:773) | **Layer 5** | "FP8 kernel speedup" | — |
| T26 | FP8 storage mode | SelectInput | POST:793 | `fp8.storageMode` | `precision==="fp8"` | **Layer 5** | "FP8 storage mode" | Feeds `effectiveOptimizerId` (POST:382). |
| T27 | *(segmented)* GPU Preset / Custom GPU | button ×2 | GS:126-144 | `hardware.inputMode` | always | **Essentials** | "Preset · Custom" | Same GS component as pretraining → reuses G1-G16 sub-form (no `tpDegree` here, GS:828). |
| T28 | GPU | SelectInput | GS:150 | `hardware.gpuId`+`gpu` | GPU `inputMode==="preset"` | **Essentials** | "GPU" (searchable) | Switching may re-sync `costPerGPUHour` (POST:334-339). |
| T29 | Number of GPUs | NumberInput | POST:833 | `hardware.numGPUs` (`setHw` POST:317) | always | **Essentials** | "Number of GPUs" | **Phase 4 documented exception**: default seeds from `1` (constants.ts:1741) to a fitting count so the tab opens green. One-tap "Fix for me" target. |
| T30 | Cost per GPU-hour | NumberInput | POST:842 | `costPerGPUHour` | always | **Essentials** | "Cost per GPU-hour ($/hr)" | unit `$/hr`. Source-of-truth coupling with GPU price preset (POST:98-106, 334-339). |

**Post-Training Section containers (no input, structural — all dissolve into Essentials disclosures / layers):** `Section "Base Model"` POST:422, `"Method & Approach"` POST:431, `"LoRA Configuration"` POST:508 (visible `isLoRA`), `"PPO Configuration"` POST:564 (`method==="ppo"`), `"GRPO Configuration"` POST:623 (`method==="grpo"`), `"Training Data"` POST:658, `"Training Setup"` POST:705, `"Hardware & Cost"` POST:820. Pretraining `Section` wrappers: `"Model"` PT:465, `"Training Data"` PT:476, `"Training Setup"` PT:515, `"Hardware"` PT:622, `"Parallelism"` PT:752.

---

## TABLE B — Read-only readouts (no `onChange`)

| # | Readout (label/title) | Defined at | Source values | Visible when | NEW HOME | Notes |
|---|---|---|---|---|---|---|
| R1 | Inferred-arch grid (d_model, Layers, Heads, d_ff, Vocab, Style) | MS:433-454 (6× `Stat`) + caption MS:455-461 | `architecture.d/L/a/d_ff/V`, `quickMode.family` | model `inputMode==="quick"` | **Essentials** (inline under Quick params) | "Approximate inference" caption stays; tiny readout under the picker. |
| R2 | Preset spec card (Parameters, d_model, Layers, Context, Heads, FFN, Attention; +Experts/Active if MoE) | MS:497-548 (7-9× `Stat`) | `preset.*` | model `inputMode==="preset"` | **Essentials** (under picker) + **`preset.notes` rendered** (plan §3/§4) | Today `notes` never rendered (plan §2). Surface them here. |
| R3 | GPU specs grid (VRAM, Dense half/TF32/FP8, Bandwidth, Interconnect, GPUs/node, TDP, Modes) | GS:189-257 (`GPUSpecsCard`, 9× `Stat`) | `gpu.*` | GPU `inputMode==="preset"` (GS:156) | **Essentials** (under GPU picker) or **Layer 1 Memory & feasibility** | Plan §3 doesn't pin it; recommend a compact strip in Essentials, full grid available in Layer 1. |
| R4 | GPU-selector inline warnings (PCIe+TP, BF16-unsupported, sparse-throughput) | GS:163-181 (`AnimatePresence` over `warnings` GS:89-111) | `getSparseThroughputWarningMessages`, PCIe/BF16 predicates | any GPU mode when predicates hit | **owning layer footnote + critical→Verdict band** (plan §5/Phase 1) | Severity routing: hardware-fatal (BF16 unsupported) → verdict band; advisories → Layer 1/3 footnote. Keep exact predicates. |
| R5 | "Live Recommendation" card (auto layout string, strategyLabel, Minimum GPUs, Pipeline Bubble) | PT:768-822 | `autoLayoutParts` (PT:438-453), `autoParallelismRecommendation.strategyLabel/minGPUs/pipelineBubbleFraction` | `parallelismMode==="auto"` (PT:768) | **Layer 3 Parallelism** (summary-line-when-closed + body) | Becomes the Layer 3 closed-summary ("Auto — TP=8, fits in 8 GPUs") + diff-vs-auto chip (plan §3 line 54). |
| R6 | Auto parallelism detail grid (Context Parallel, Expert Parallel, Virtual Pipeline, Sequence Parallel) + caption | PT:929-988 (4× value blocks) | `displayParallelism.N_cp/N_ep/VP/sequenceParallelism` | inside Advanced AND `parallelismMode==="auto"` (PT:928) | **Layer 3** (read-only when auto) | Mirrors manual inputs P47-P50; show as derived values when auto. |
| R7 | "Computed trainable footprint" card (adapter params + % of base) | POST:482-503 | `estimateLoRAParameterCount` (POST:412), `estimatedLoRAPercentage` (POST:413-416) | `isLoRA` AND both values non-null (POST:482) | **conditional-disclosure** "Customize adapter ▸" (with T7-T10) | Derived display; pairs with the LoRA inputs. |
| R8 | "MoE Overview" intro prose (sparse-routing notice) | MS:766-773 | static copy | model `inputMode==="detailed"` AND `moe.enabled` | **Layer 8 MoE** | The prose currently points users to "Advanced Settings"; reword to point at Layer 8 once knobs merge there. |
| R9 | "Total tokens edited from Quick tab" hint | PT:503-511 | static copy | `isQuickMode` (model `inputMode==="quick"`) | **Essentials** (near tokens) | Explains why P21 is hidden in Quick; keep as helper text. |
| R10 | "GPU count resolved from target" hint | PT:662-670 | static copy | `gpuCountDerivedFromTarget` | **Essentials** (near P33/P34) | Explains P33 disabled-state. |

**Verdict band / footprint card / memory bar** (these are produced in `gpu-calculator.tsx` + `results-summary.tsx` + `memory-breakdown-bar.tsx` + `gpu-utilization-gauge.tsx`, *outside the four files in scope*, so labels not enumerated here): per plan §3 they map to **VerdictBand** (sticky: `memory.fits`/`cost.totalCost`/`trainingTime`/GPU count) and **Layer 1** (`MemoryBreakdownBar` + gauge + minGPUs + maxMicroBatch; post-training MemoryLineItems) and **Layer 2** (days/tok-s/steps/global-batch/cost breakdown/checkpoints). Flagged for the implementing agent: those component files are the home of the "computed footprint card" and "live recommendation card as a verdict" — confirm against `results-summary.tsx:634/778` (warnings) and `gpu-calculator.tsx:6446-6657` (shell) when wiring.

---

### Key flags for the implementing agent

- **detailedDraft fix (P1):** `setMode` else-branch at MS:208-210 only sets `inputMode` and restores no architecture; Quick (MS:194) and Preset (MS:199-207) overwrite `architecture`+`moe`. A `detailedDraft` shadow copy must capture Detailed edits and restore them on return to Detailed; only overwrite `architecture` on explicit preset load (plan §3/§4, decision at plan line 34).
- **Duplicate field pairs to merge into Layer 8:** P18≡P56 (`moe.denseIntermediateSize`), P19≡P57 (`moe.expertIntermediateSize`) — same config field rendered in two panels (MS detailed + PT advanced). Render once in Layer 8.
- **Duplicate Framework control:** P41 (manual, PT:828) and P58 (auto, PT:1180) write the same `parallelism.framework`; collapse to one Layer 3 control.
- **OverrideBadge / AssumptionsLedger sources (plan §3/§5):** `effectiveOptimizerId` adamw-fp8→adamw-mixed (PT:355 / POST:378); ZeRO-3 forces overlap-comm (PT:454-458, control P62); ZeRO-stage-under-FSDP (PT:117-131, P46). These render at the control AND as a ledger row.
- **DECIDE rows summarized:** P16 (MoE enable → Layer 4 toggle, knobs to Layer 8); P22 (unique tokens → Layer 6); P34 (target days → Essentials beside #GPUs); P48 (EP → Layer 8 per spec, mesh-referenced in Layer 3); P59 (CPU offload → Layer 5); P62 (overlap-comm → Layer 3); G1-G16 (custom-GPU form → "Custom GPU specs ▸" disclosure, not a layer); T18 (seq len → Layer 4); T19 (batch size → Layer 6); T24 (KV-cache precision → Layer 5).
- **Phase 4 default exception:** `DEFAULT_POST_TRAINING_CONFIG.numGPUs` = `1` at constants.ts:1741 (vs pretraining `8` at :1676) — the single approved input-default change, applied to T29.
