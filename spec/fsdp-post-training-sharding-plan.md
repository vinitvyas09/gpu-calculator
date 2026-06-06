# FSDP Post-Training Sharding — Verified Findings & Implementation Plan

> **Scope:** add a "Distributed strategy" option to POST-TRAINING mode only:
> `ddp-replicated` (current behavior, DEFAULT) vs `fsdp-full-shard` (FSDP / ZeRO-3).
> **Existing math is sacred** — every existing formula keeps byte-identical outputs when the
> strategy is the default. Only a NEW sharded branch is added. Date: 2026-06-05.
>
> Provenance: 5 read-only codebase mappers + 5 web researchers; 27 load-bearing claims,
> each adversarially verified by 1–2 independent agents against primary sources
> (24 confirmed, 3 flagged with resolutions in §2). Code anchors re-verified first-hand.

---

## 0. Locked decisions (user-approved 2026-06-05)

| Decision | Choice | Rationale |
|---|---|---|
| FSDP usable-capacity factor | **0.8** (DDP keeps 0.9) | Matches pretraining FSDP semantics (memory.ts:2984-2989); measured FSDP reserved ≈ 1.10–1.12× allocated |
| All-gather working buffer | **2× largest wrapped block** | Matches shipped pretraining convention (memory.ts:2790-2794); FSDP paper's 1× refuted as idealized lower bound (default `BACKWARD_PRE` holds current+next unit) |
| QLoRA CPU offload | **Deferred** — model no-offload regime | Tooltip discloses that published 2×24GB results additionally use CPU offload (19.6 GB vs 35.6 GB no-offload) |
| ZeRO-1/2 granularity | Out of scope | Binary toggle: replicated vs FULL_SHARD |
| Time/cost comm overhead | Not modeled; FSDP-gated soft warning only | Memory calculator; honesty preserved via warning |
| Parity baselines | **Never touched** | Invariant checks live in a separate script, not in scripts/parity/ |

---

## 1. Verified FSDP memory model (what the formulas may rely on)

Every statement below was confirmed by independent adversarial verification. Claim ids
refer to the research record; cite these URLs in code comments next to each new term.

### 1.1 What shards ÷ world size W (persistent states)

- **`full-shard-shards-all-three`** (2 verifiers): FULL_SHARD / `fully_shard` == ZeRO-3 shards
  **parameters, gradients, AND optimizer states** by 1/W. Optimizer states stay sharded the
  whole loop (updated locally per rank); gradients reduce-scatter to shards; params all-gather
  transiently. Sources: `torch/distributed/fsdp/api.py` docstring ("Parameters, gradients, and
  optimizer states are sharded… The sharded optimizer states are updated locally per rank"),
  https://docs.pytorch.org/docs/2.12/fsdp.html, ZeRO paper https://arxiv.org/pdf/1910.02054.
- **`sharding-factor-F-world-size`**: sharding factor F = W under FULL_SHARD; F=1 ≡ DDP.
  1/W is the **persistent** footprint; instantaneous peak adds the working buffer (§1.2).
  Source: FSDP paper https://arxiv.org/pdf/2304.11277.
- **`frozen-shard-allgather`** (2 verifiers): **frozen** weights also store sharded (÷W) and
  all-gather transiently. (The frozen-FlatParameter resharding leak was FSDP1 < 2.1.0,
  fixed in PyTorch PR #101982 — current stable and FSDP2 unaffected.)
- **`optimizer-states-trainable-only-sharded`**: optimizer states exist only for trainable
  params; frozen base has none; trainable states shard ÷W.
- **`mixed-precision-sharded-fp32-masters`** (2 verifiers): with `param_dtype=bf16`, sharded
  persistent params stay in original precision and ARE the master copy — FSDP adds **no extra
  fp32-master memory**. The existing optimizer profiles (`parameterBytes`/`masterWeightBytes`/
  `kOpt`) already encode this; **do not add a master-weight term on top**.
- Cross-checks: DeepSpeed's own estimator `estimate_zero3_model_states_mem_needs` =
  `4·largest_layer_params + 18·P/N_gpus` (**no activations**) — reproduced numerically against
  DeepSpeed's published T5-3B tables by verifiers; ZeRO paper 16Ψ/N_d; EleutherAI
  Transformer-Math-101 (activations NOT sharded by ZeRO); HF: FULL_SHARD == ZeRO-3.
  The 12-vs-16-optimizer-bytes divergence between FSDP/DeepSpeed conventions is absorbed by
  **reusing the resolved optimizer profile** — no new byte constants.

### 1.2 The all-gather working buffer (the term naive calculators miss)

- **`transient-working-set-k-and-prefetch`** (2 verifiers): default
  `backward_prefetch=BACKWARD_PRE` holds **current + next unit params (+ current unit grads)**
  unsharded at peak → **k = 2** units. Wrapped unit = one transformer/decoder layer under
  `TRANSFORMER_BASED_WRAP`. FSDP2 always behaves BACKWARD_PRE-equivalent.
  Sources: `torch/distributed/fsdp/fully_sharded_data_parallel.py` (default BACKWARD_PRE,
  `backward_prefetch_limit = 1`), https://arxiv.org/pdf/2304.11277.
- **`deepspeed-stage3-max-live-default`** (2 verifiers): `stage3_max_live_parameters`
  default **1e9** params is the canonical cap on the unsharded live working set.
  Source: https://www.deepspeed.ai/docs/config-json/ (+ EleutherAI hardcodes 1e9).
- This buffer does **NOT** divide by W — it is the floor that GPUs cannot shrink.

### 1.3 Frozen + LoRA specifics

- **`grad-flatparam-fsdp1`** (2 verifiers): naive FSDP1 with frozen base + LoRA in one
  FlatParameter allocates gradients for the **entire block**. NOT modeled, because:
- **`peft-recipe-fsdp1`** (2 verifiers): the HF PEFT recipe (`use_orig_params=false` +
  `peft.utils.other.fsdp_auto_wrap_policy`) wraps trainable LoRA leaves separately →
  **no grad/optimizer memory for the frozen base**. Source:
  https://huggingface.co/docs/peft/accelerate/fsdp.
- **`fsdp2-per-param-no-frozen-grad`** (2 verifiers): FSDP2 allocates grads only for
  `requires_grad=True` — same conclusion without special wrapping.
  → Model: frozen base = sharded weights only; LoRA params/grads/optimizer sharded ÷W.

### 1.4 QLoRA under FSDP

- **`fsdp-shards-packed-4bit`** (2 verifiers): FSDP shards/all-gathers the **PACKED 4-bit**
  storage (0.5 B/param payload regardless of `bnb_4bit_quant_storage` dtype), never
  dequantized weights. Mainstream since bitsandbytes 0.43.0 (PR #970), wired through
  transformers/PEFT/TRL; requires `bnb_4bit_quant_storage` = float dtype matching model dtype.
  Sources: https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora,
  https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive,
  https://github.com/bitsandbytes-foundation/bitsandbytes/pull/970.
- **`nf4-double-quant-overhead`** (2 verifiers): NF4+double-quant = 4.127 bits = 0.5159 B/param
  — matches existing `NF4_DOUBLE_QUANT_BYTES_PER_PARAM = 0.5159` (memory.ts:1791) **exactly**.
- **`existing-calculator-constants-consistency`**: reuse `calculateQuantizedBaseModelBytes`
  unchanged (it already keeps embeddings/lm_head/norms unquantized) and divide **its result**
  by W. **No new bytes/param constant.**
- **`transient-dequant-buffer`** (sourced, NOT adversarially verified — re-verify before
  coding, see §9 step 0): dequantization to compute dtype is transient and per-unit
  (one block's weights in bf16 live during that block's matmuls).
  Sources: https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora,
  https://arxiv.org/abs/2305.14314.

---

## 2. Flagged claims and their resolutions

1. **`fsdp-peak-memory-model`** (uncertain + refuted): the FSDP paper's
   `Σψ/F + 1×maxψ` is an idealized **lower bound**; realizable default peak ≈ **2× largest unit**
   (BACKWARD_PRE; rate limiter caps two inflight all-gathers). → Resolved: use 2×,
   matching pretraining memory.ts:2790-2794. **User approved.**
2. **`peft-70b-qlora-2x24gb-no-offload`**: the **35.6 GB/GPU** figure is confirmed verbatim in
   PEFT docs; only the researcher's napkin derivation was wrong (used the 140GB-over-8-GPUs
   number). → Use the measured 35.6 GB; discard the derivation.
3. **`answerai-yi34b-qlora-2x3090-offload-pair`**: benchmark numbers (23.05 / 22.98 GiB)
   confirmed; the "~17 GB 4-bit weights per GPU" reading refuted — that is the **whole-model**
   4-bit size; the per-GPU shard is ~8.0–8.5 GiB. → Use corrected interpretation.

Carried open items: full-parameter FSDP 70B measured datapoint not found (full-SFT FSDP is the
least-calibrated branch — disclose); k can transiently spike to 3 (rare, not modeled);
wrap granularity assumed per-transformer-block.

---

## 3. Shared conventions for the new branch

- `N` = `config.hardware.numGPUs`. Post-training has no TP/PP/EP — N is the whole sharding world.
- FSDP active ⇔ `config.distributedStrategy === "fsdp-full-shard"` (**strict equality**:
  `undefined` from old persisted localStorage states ⇒ DDP. `safeParse` does NOT merge defaults —
  use-persisted-state.ts:21-27).
- All byte multipliers come from the **resolved optimizer profile** and existing helpers
  (`getPostTrainingWeightBytes`, `calculateTrainableModelStates`,
  `calculateQuantizedBaseModelBytes`). **Zero new byte-per-param constants.**
- MoE: existing post-training convention uses **TOTAL** params (all expert copies) for weights —
  FSDP shards all expert copies, so totals stay total (`calculateLoRAParamCountForArchitectureWithExpertCopies` precedent).

### 3.1 New term: all-gather working buffer `W_allgather` (NOT ÷N)

```
cappedUnitParams = min(largestTransformerBlockParams, 1e9)        // 1e9 cap ← stage3_max_live_parameters default
W_allgather       = 2 × cappedUnitParams × gatherBytesPerParam     // 2× ← BACKWARD_PRE peak, matches pretraining
  gatherBytesPerParam (full/LoRA/MeZO/DPO/PPO/GRPO) = getPostTrainingWeightBytes(config)   // bf16/fp16=2, fp32=4
  QLoRA variant:
    W_allgather_qlora = 2 × cappedUnitParams × quantizedBytesPerParam   // packed 4-bit gathers packed (≈0.5159)
                      + 1 × cappedUnitParams × 2                        // transient per-unit dequant to bf16 (§1.4, re-verify)
```

- `largestTransformerBlockParams` needs a **new post-training helper**: the existing
  `getLargestLayerParameterCount` (memory.ts:1144) takes pretraining `ParameterCounts`/
  `TrainingConfig`; post-training passes `ModelArchitecture` + `MoEConfig`. For MoE, one block
  includes ALL its expert copies (which is exactly why the 1e9 cap matters: a DeepSeek-V3 MoE
  block is ~11B params; block-level gathering would be absurd — the cap is the documented
  stand-in for the finer-grained (per-expert) wrapping every real MoE recipe uses).
- Surfaced in UI tooltip as documented assumptions: per-block auto-wrap; 2× prefetch peak;
  1e9-param live-set cap (DeepSpeed `stage3_max_live_parameters` default).

### 3.2 Capacity + finalizer

- FSDP branch fit check uses **0.8** usable capacity; DDP stays **0.9**. ALL post-training
  `usableCapacity` sites must route through one FSDP-aware helper — known sites:
  memory.ts:2304 (`getPostTrainingDataParallelSizingCapacity` region), 2346 (finalizer),
  3276, and the inline SFT-full/MeZO computations in gpu-calculator.tsx. Enumerate by
  grepping `* 0.9` before editing; miss one and fit verdicts disagree between panels.
- `W_allgather` enters the total via the finalizer's `peakWorkingSet` argument
  (total = `(parameters + gradients + optimizerStates + peakWorkingSet + frameworkOverhead) × 1.04`,
  memory.ts:2338-2344) and is surfaced via the existing `communicationBuffers` slot + an
  `items[]` line ("FSDP all-gather buffer") — **no new always-present output keys** (§5).
- SFT-full and MeZO (defined in gpu-calculator.tsx, NOT memory.ts) build their returns inline
  without the finalizer — apply the same arithmetic in their own style. Match each function's
  existing structure; do not refactor.

---

## 4. Per-approach formulas (DDP column = current code, untouched)

`/N` marks terms divided by numGPUs **only under FSDP**. Activations NEVER shard (ZeRO/FSDP
keep activations per-rank — Transformer-Math-101, EleutherAI cookbook). Framework overheads keep
their existing values (1e9 / 2e9 / 5e9). The ×1.04 scalar is unchanged.

| Approach (fn, anchor) | Sharded ÷N under FSDP | Stays per-GPU | Notes |
|---|---|---|---|
| **SFT-full** (`calculateSFTFullMemory`, gpu-calculator.tsx:3408) | trainable·parameterBytes, frozen·frozenBytes, gradients (trainable·betaGrad), optimizer (trainable·kOpt) | activations, 2e9 overhead, +`W_allgather` | least-calibrated branch (no measured datapoint) — disclose in tooltip |
| **MeZO** (`calculateMeZOMemory`, gpu-calculator.tsx:3493) | parameters (totalParams·wb) | logits-only activations, 1e9, +`W_allgather` | soft warning: gather floor usually wipes out MeZO's benefit |
| **LoRA** (`calculateLoRAMemory`, memory.ts:3367) | baseModelBytes (frozen), loraStates.parameters/gradients/optimizerStates | activations, 1e9, +`W_allgather` | frozen base keeps zero grad/optimizer (§1.3) |
| **QLoRA** (`calculateQLoRAMemory`, memory.ts:3431) | `calculateQuantizedBaseModelBytes(...)` result, loraStates.* | activations, 1e9, +`W_allgather_qlora` | packed 4-bit shards (§1.4); tooltip: requires `bnb_4bit_quant_storage=bf16` (bnb ≥0.43); published 2×24GB results also use CPU offload (not modeled) |
| **DPO LoRA/QLoRA branch** (`calculateDPOMemory`, memory.ts:3499) | baseModelBytes (quantized if qlora), loraStates.* | activations (2× chosen/rejected), logProbStorage (`communicationBuffers`), 1e9, +`W_allgather`(`_qlora` if qlora) | keep `peakWorkingSet = activations + logProbStorage` shape, add buffer |
| **DPO full branch** (same fn) | policyStates.parameters/gradients/optimizerStates, referenceModelBytes (frozen ⇒ shards, §1.1) | activations, logProbStorage, 1e9, +`W_allgather` | |
| **PPO** (`calculatePPOMemory`, memory.ts:3657) | actor states (full or LoRA), critic states, frozen actor base, referenceModelBytes, rewardModelBytes | rollout/KV/generation working set, ppoBuffers, 5e9, +`W_allgather` | shard **training** states only; generation peak stays per-GPU (each rank serves a full policy during rollout); existing `max(update, generation)` picks the binding phase. Least-calibrated; tooltip notes |
| **GRPO** (`calculateGRPOMemory`, memory.ts:3882) | policy/LoRA states, referenceModelBytes, rewardModelBytes | grouped activations (groupSize), rollout/KV, 1e9, +`W_allgather` | same generation caveat as PPO |

QLoRA W_allgather term sums **into the same buffer line** (packed gather + dequant transient).

---

## 5. Parity constraints (hard rules)

Verified mechanics (scripts/parity/parity-check.mjs, re-read first-hand):

1. Exports serialize **`CalculatorOutput`** (computed output) via `serializeCalculatorOutput`
   (gpu-calculator.tsx:4907-4919) — **config is NOT serialized**, so the new
   `PostTrainingConfig.distributedStrategy` field is invisible to parity.
2. The diff walks the **union of object keys** (parity-check.mjs:250-254) — any new
   always-present key in `PostTrainingOutput`/breakdown WOULD diff every post-training
   scenario. Constraints: (a) reuse existing slots (`communicationBuffers`, `items[]`,
   `peakWorkingSet`); (b) any genuinely new output field must be `undefined` under DDP
   (`JSON.stringify` omits `undefined`); prefer (a).
3. Default config stays DDP ⇒ the 7 post-training baselines (`posttrain-default`, `-dpo`,
   `-ppo`, `-grpo`, `-qlora`, `-full-ft`, `-default-text`) must produce **zero diffs**.
   Run the gate with **no `ALLOWED_DIFF_KEYS`** — it must PASS.
4. **Never regenerate/edit `baseline-snapshots.json`.** No FSDP scenario is added to the
   parity script in this change; FSDP invariants live in a separate script (§9).
5. Scenario selectors are label-based: the new control's option labels must not collide with
   any existing option label — use `"Replicated (DDP)"` / `"Sharded (FSDP / ZeRO-3)"`.
6. New warnings / verdict text / export lines must be **FSDP-gated** so DDP byte-streams are
   untouched (incl. `generatePostTrainingMarkdown` and the GPU-requirement line at
   gpu-calculator.tsx:4888-4905).

---

## 6. Verdict & required-GPU logic (gpu-calculator.tsx:3634-3752, 6946-6966)

Current behavior: `estimatePostTrainingRequiredGPUs` computes
`stateFloorBytes = (params+grads+opt+overhead)×1.04` at numGPUs=1; if that exceeds usable
capacity it returns mode `state-sharded-lower-bound` with
`ceil(persistentStates / (usable/1.04 − overhead))` — the *hypothetical* message at :6961.

Under FSDP (all changes gated on `isPostTrainingFSDP`):
- **Skip the replicated-floor early return** (line 3709) — states shard with N, so the floor
  argument no longer applies. Proceed to the existing binary search, which calls
  `getPostTrainingMemory` per candidate N and therefore uses sharded math automatically.
- New mode label (e.g. `"fsdp-sharded"`) with verdict copy reflecting **actual** sharding;
  the hypothetical "ideal ZeRO-3/FSDP lower bound" message remains DDP-only.
- `maxUsefulGPUs` (batch-split limit) stays as the search bound: FSDP is still data parallelism —
  every rank needs ≥1 sample. Document this in the tooltip ("more GPUs than batch items won't
  be suggested").
- **Invariant (tested):** FSDP required-N ≥ the DDP hypothetical lower bound, always — FSDP adds
  `W_allgather` + activations + overhead per-GPU and uses 0.8 ≤ 0.9 capacity, so it strictly
  dominates. If the test ever fails, the math is wrong — fix code, not test.

---

## 7. UI

- **Control:** `SelectInput` "Distributed strategy" in the precision/optimizer layer of
  post-training-panel.tsx (pattern: the existing precision selector at :1009-1020),
  `termKey="distributedStrategy"`, value `config.distributedStrategy ?? "ddp-replicated"`.
- **Breakdown:** under FSDP, item labels annotate sharding (e.g. "Base model (frozen, sharded ÷48)");
  new items line "FSDP all-gather buffer". DDP labels byte-identical.
- **Tooltip (documented assumptions, all surfaced):** PyTorch FSDP FULL_SHARD (≡ ZeRO-3);
  per-transformer-block wrapping; 2× largest-block all-gather peak (BACKWARD_PRE);
  live-set cap 1e9 params; activations/batch still per-GPU; QLoRA: requires
  `bnb_4bit_quant_storage` float dtype (bnb ≥ 0.43 / PEFT-TRL mainstream), packed-4-bit gathers,
  no CPU offload modeled; comm overhead not reflected in time/cost (soft warning);
  capacity factor 0.8 under FSDP.
- **Glossary:** `distributedStrategy` entry in glossary.ts.

---

## 8. Pathfinding results (what else, beyond the ask)

**(a) Required — included in this change:**
1. Persisted-state back-compat (strict-equality FSDP check; select coalesces undefined) — §3.
2. Post-training largest-block helper (new; pretraining helper has incompatible signature) — §3.1.
3. Validation: `fsdp-full-shard` + `numGPUs ≤ 1` rejected (post-training-validation.ts).
4. Verdict/mode coherence under FSDP — §6.

**(b) Recommended — included (small):**
5. FSDP capacity factor 0.8 (user-approved).
6. FSDP-gated soft warnings: interconnect bandwidth sensitivity; time/cost ignores extra
   all-gather comm (~ZeRO-3 ≈ 1.5× comm volume per spec §5.2); MeZO+FSDP rarely worthwhile.
7. Glossary + tooltips.

**(c) Deferred / out of scope (explicitly):**
8. QLoRA CPU-offload toggle (user-approved deferral; revisit to match the 19.6 GB headline regime).
9. Strategy-aware time/cost model (comm overhead) — warning only.
10. Auto strategy recommendation; ZeRO-1/2 stages; expert-parallel sharding degrees.
11. FSDP parity scenarios (would require editing baseline-snapshots.json — forbidden).

---

## 9. Implementation order (each step gated by its verify)

0. **Pre-code re-verification (standing rule):** independently re-verify via web the single
   non-adversarially-verified term used: QLoRA transient per-unit dequant buffer
   (bnb fsdp_qlora doc). If unverifiable → keep the term but mark "documented assumption"
   in tooltip text and code comment.
1. `types.ts` — `export type PostTrainingDistributedStrategy = "ddp-replicated" | "fsdp-full-shard"`;
   add optional-safe field to `PostTrainingConfig` (after `kvCachePrecision`, :275).
   *Verify:* `npx tsc --noEmit` clean.
2. `constants.ts` — `DEFAULT_POST_TRAINING_CONFIG.distributedStrategy: "ddp-replicated"` (:1707).
   *Verify:* default DDP.
3. `post-training-validation.ts` — invalid when FSDP and `numGPUs ≤ 1` (+ wire into the
   validation call sites used by `estimatePostTrainingRequiredGPUs` / panel).
   *Verify:* FSDP+1 GPU → validation error string; DDP+1 GPU unchanged.
4. `memory.ts` — helpers: `isPostTrainingFSDP`, `getPostTrainingLargestBlockParameterCount`
   (ModelArchitecture+MoE → largest block incl. all expert copies), `calculatePostTrainingAllGatherBufferBytes`
   (2× capped, gatherBytes per §3.1, QLoRA variant), FSDP-aware capacity factor helper.
   Citation comments per §1. *Verify:* helpers pure; spot values (Llama-70B block ≈ 0.87B
   → no cap; DSv3 block ~11B → capped at 1e9).
5. Gate the 7 calculators (§4), smallest first: LoRA → QLoRA → SFT-full → MeZO → DPO → GRPO → PPO.
   DDP path byte-identical (guard every new term behind `isPostTrainingFSDP`).
   *Verify after EACH:* quick DDP spot-check of that approach's numbers vs pre-change.
6. Verdict + required-GPU (§6); text-export GPU-requirement line FSDP mode.
   *Verify:* DDP messages byte-identical; FSDP messages coherent.
7. UI (§7): panel control, breakdown labels, tooltips, glossary.
   *Verify:* lint + tsc; labels non-colliding (grep option labels vs parity selectors).
8. **Parity gate:** `npm run dev` + `node scripts/parity/parity-check.mjs` (no env overrides) →
   must PASS with zero diffs. Never touch baselines.
9. **Invariants script** (new, e.g. `scripts/fsdp-invariants/check.mjs`, same playwright
   harness pattern as parity but separate — parity dir untouched):
   - monotone: FSDP per-GPU(N) non-increasing in N (sample N ∈ {2,4,8,48,256});
   - dominance: FSDP total ≤ DDP total at same N (every approach);
   - N=1: sharded terms equal DDP exactly; delta = `W_allgather` only (and validation rejects
     FSDP at N=1 in UI — test via formula-level harness or temporarily-permitted config);
   - FSDP required-N ≥ DDP hypothetical ZeRO-3 lower bound;
   - no-double-count: FSDP params bytes use profile `parameterBytes` (no `+masterWeightBytes`).
10. **Hand-check (motivating scenario):** DeepSeek V3 671B, SFT+LoRA, 48× B200 NVL72:
    - DDP: **1541.3 GB exactly** (unchanged).
    - FSDP: base 671e9×2/48 ≈ 27.96 GB + LoRA states÷48 + `W_allgather` = 2×1e9×2 = 4 GB
      + activations (per-GPU, unchanged) + 1 GB overhead, ×1.04; show full arithmetic vs UI.
11. **Calibration report:** model the §10 datapoints; report deviations; **flag >15% for review
    rather than shipping silently** (expected: reserved-vs-allocated ≈ +10–12% systematic;
    PEFT 70B QLoRA no-offload 35.6 GB likely the hardest to match — report honestly).

## 10. Calibration datapoints (verified measured numbers)

| Source (claim) | Setup | Measured |
|---|---|---|
| HF PEFT docs (`peft-70b-lora-8xh100-bf16`) | Llama-2-70B LoRA r8 all-linear, 8×H100-80GB, bf16, seq 2048, bs 8, GC on, FA2+packing | 72–80 GB/GPU (reserved-style) |
| HF PEFT docs (`peft-70b-qlora-2x24gb-no-offload`) | Llama-2-70B QLoRA NF4-DQ r8, 2 GPUs, seq 2048, bs 2, GC on, **no offload** | **35.6 GB/GPU** |
| HF PEFT docs (offload variant) | same, `fsdp_offload_params=true` | 19.6 GB/GPU + ~107 GB CPU (not modeled — offload deferred) |
| answer.ai (`answerai-7b-qlora-2x3090`) | Llama-2-7B QLoRA, 2×3090, seq 512/2048, bs 1 | peak reserved 4.98 / 5.21 GiB |
| answer.ai (`answerai-7b-lora-2x3090`) | Llama-2-7B LoRA 16-bit, 2×3090 | reserved 10.24/10.22; allocated 8.28/9.16 GiB |
| answer.ai (`answerai-yi34b-qlora-2x3090`) | Yi-34B QLoRA, 2×3090, bs 2 no-offload | 23.05 GiB/GPU (4-bit shard ≈ 8.0–8.5 GiB) |

Calibration notes: all reserved/nvidia-smi-style ⇒ expect modeled (allocated-style) ≈ −10%;
no measured full-SFT FSDP datapoint exists — full-SFT FSDP cross-checked analytically against
DeepSpeed estimator + ZeRO paper only (disclosed in tooltip).

---

## 11. Verification record (2026-06-05, implemented + verified)

Harness: `scripts/fsdp-invariants/check.ts` (separate from scripts/parity/; usage in header).

1. **DDP equivalence (formula level):** 220/220 old-HEAD-vs-worktree comparisons
   byte-identical across methods × approaches × models × N ∈ {1,2,8,48}, including
   legacy configs with the `distributedStrategy` field absent.
2. **Parity gate (browser):** official invocation
   `ALLOWED_DIFF_KEYS=<7 Phase-4 post-training keys> node scripts/parity/parity-check.mjs`
   → **PARITY: PASS** (every pretraining key byte-frozen PASS). Stronger proof: HEAD-capture
   vs worktree-capture `current-snapshots.json` **byte-identical (`cmp` clean)** — the change
   contributes zero bytes even inside the allow-listed keys. Baselines untouched.
   (Notes: the no-env invocation shows only the pre-existing sanctioned Phase-4 seed diffs;
   run the dev server with webpack — `npx next dev --webpack` — Turbopack panicked.)
3. **Invariants (all pass):** FSDP ≤ DDP at N≥2; per-GPU total monotone non-increasing in N
   (N ∈ 2…256); FSDP(N=1) == DDP(N=1) on all numeric fields (gather=0 and capacity collapses
   to 0.9 at world size 1 by design — FSDP at N=1 IS DDP); FSDP required-N ≥ ideal ZeRO-3
   lower bound (70B LoRA 5≥3, DSv3 LoRA 13≥10, DPO-full 44≥21, PPO-LoRA 8≥5, GRPO-QLoRA 17≥1).
4. **Motivating scenario (hand-check, UI-confirmed):** DeepSeek V3 671B SFT+LoRA 48×B200-NVL72:
   DDP **1541.3 GB** unchanged exactly; FSDP **55.99 GB/GPU** = base 27.96 (=671e9×2/48 ✓)
   + LoRA p/g/o 0.28/0.56/1.68 (÷48) + activations 18.36 + gather 4.00 (=2×1e9×2B, the 1e9
   cap binding on DSv3's ~11B-param MoE blocks ✓) + overhead 1.0, ×1.04; fits in 148.8 GB
   (0.8×186). Browser smoke test matches to the cent; N=1 field error renders; DDP
   round-trip returns 1541.3 GB exactly.
5. **Calibration vs measured (modeled = allocated-style; published = reserved-style ≈ +10–12%):**

   | datapoint | modeled | measured | deviation |
   |---|---|---|---|
   | 70B QLoRA NF4 2 GPU no-offload (PEFT docs) | 34.1 GB | 35.6 GB | −4.1% |
   | Yi-34B QLoRA 2×3090 (Answer.AI) | 22.5 GB | 24.7 GB | −9.2% |
   | 7B QLoRA 2×3090 (Answer.AI) | 6.0 GB | 5.6 GB | +7.3% |
   | 7B LoRA 2×3090 (Answer.AI) | 11.2 GB | 11.0 GB | +1.9% |
   | 70B LoRA 8×H100 (PEFT docs) | 63.8 GB | 72–80 GB | **⚑ −16.0%** |

   The flagged −16.0% is consistent with the reserved-vs-allocated systematic:
   63.8 × 1.12 ≈ 71.5 GB ≈ the low end of the published band (a reserved-style wandb
   panel, run with packing). Kept flagged per §9.11 for human review rather than
   silently absorbed.
