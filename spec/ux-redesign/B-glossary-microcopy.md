# Appendix B — Glossary + microcopy pack

*Generated 2026-06-05 from line-anchored code analysis. Part of spec/ux-redesign-plan.md.*

# Glossary + Microcopy Pack — "Calm Layers"

**Scope note for the implementing agent:** This pack provides (1) ~60 glossary entries for `glossary.ts`, (2) verdict-band microcopy, (3) per-layer summary-line templates, and (4) IntentRow/HeroBar/banner copy. All summary-line templates reference **verified** `CalculatorOutput` fields (checked against `components/gpu-calculator/types.ts` and the output-object construction in `gpu-calculator.tsx`). The `fmt*` formatter names used below are import ALIASES of the shared module `components/gpu-calculator/formatters.ts` (`gpu-calculator.tsx:172-179`: `formatCost`→`fmtCurrency`, `formatMemory`→`fmtBytes`, etc.; only `fmtBatchRelation` is genuinely local, `:4765`) — **import the canonical `format*` functions from `../formatters`; do not reimplement.** Definitions are grounded in `spec/research/*.md` with citations; the math is untouched.

---

## 1. Glossary entries (`glossary.ts`)

TypeScript-ready. Shape: `Record<string, { term: string; def: string }>`. Keys are stable identifiers used by `<Term term="...">`. Definitions are 1-2 sentences, novice-readable, non-circular.

```ts
// components/gpu-calculator/components/glossary.ts
export interface GlossaryEntry {
  term: string // display label
  def: string // 1-2 sentence plain-English definition
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
  parameters: {
    term: "Parameters",
    def: "The learned weights inside the model — the numbers it adjusts during training. Model size is usually quoted as a parameter count (e.g. 7B = 7 billion).",
  },
  tokens: {
    term: "Tokens",
    def: "The chunks of text a model reads and writes — roughly a word or word-piece. Training data and context length are both measured in tokens.",
  },
  uniqueTokens: {
    term: "Unique tokens",
    def: "How much distinct text your dataset actually contains, before any repetition. If total training tokens exceed this, the model sees some data more than once (multiple epochs).",
  },
  sequenceLength: {
    term: "Sequence length",
    def: "How many tokens the model processes at once in a single training example — its context window during training. Longer sequences cost more memory and compute, growing steeply with attention.",
  },
  microBatch: {
    term: "Micro-batch size",
    def: "The number of sequences one GPU processes in a single forward/backward pass. Kept small to fit in memory; the real (global) batch is built up from many micro-batches.",
  },
  gradientAccumulation: {
    term: "Gradient accumulation",
    def: "Running several micro-batches and summing their gradients before updating the weights, so you can reach a large effective batch size without the memory of one giant batch.",
  },
  globalBatchSize: {
    term: "Global batch size",
    def: "The effective batch the optimizer actually updates on: micro-batch × gradient-accumulation steps × data-parallel GPUs. This is the number that affects how the model learns.",
  },
  precision: {
    term: "Precision (number format)",
    def: "How many bits each number uses. Lower precision (fp16, bf16, fp8) is faster and uses less memory than fp32, at some risk to numerical stability.",
  },
  bf16: {
    term: "bf16 (bfloat16)",
    def: "A 16-bit format with the same large exponent range as fp32 but fewer precision bits. The default for modern training because it rarely needs loss scaling to stay stable.",
  },
  fp16: {
    term: "fp16 (half precision)",
    def: "A 16-bit format with more precision bits than bf16 but a much smaller range, so it often needs loss scaling to keep small gradients from underflowing to zero.",
  },
  fp32: {
    term: "fp32 (single precision)",
    def: "The classic 32-bit format. Most accurate and most memory-hungry; modern training keeps an fp32 'master copy' of weights even when computing in 16-bit.",
  },
  fp8: {
    term: "fp8 (8-bit float)",
    def: "An 8-bit format on the newest hardware (H100 and later) that can roughly double matmul throughput. Used for the heavy matrix multiplies while a higher-precision copy preserves accuracy.",
  },
  mixedPrecision: {
    term: "Mixed precision",
    def: "Computing the forward/backward pass in 16-bit for speed and memory, while keeping an fp32 master copy of the weights and accumulating into fp32 for stability. This is why Adam-style training costs ~16 bytes per parameter, not 4.",
  },
  optimizerStates: {
    term: "Optimizer states",
    def: "Extra per-parameter memory the optimizer keeps between steps — for Adam, the running mean and variance plus the fp32 master weights. Often the single largest slice of training memory.",
  },
  activationCheckpointing: {
    term: "Activation checkpointing",
    def: "Saving memory by discarding most intermediate activations during the forward pass and recomputing them in the backward pass. Trades extra compute (~30% more) for large memory savings.",
  },
  selectiveCheckpointing: {
    term: "Selective recomputation",
    def: "A lighter form of checkpointing that only recomputes the cheap-but-memory-heavy attention activations, leaving the expensive matmuls stored. Cuts the worst activation memory with minimal extra compute.",
  },
  flashAttention: {
    term: "Flash attention",
    def: "An attention algorithm that never writes the full N×N attention matrix to GPU memory, computing it in fast on-chip tiles instead. Gives the exact same result while removing attention's quadratic memory cost.",
  },
  mfu: {
    term: "MFU (Model FLOPs Utilization)",
    def: "The fraction of a GPU's peak math throughput your training actually achieves, after communication, memory, and bubble overhead. Real LLM runs typically land around 35-55%.",
  },
  tflops: {
    term: "TFLOPS",
    def: "Trillions of floating-point operations per second — a measure of raw math throughput. GPUs are rated at a peak TFLOPS; MFU tells you how much of it you reach.",
  },
  vram: {
    term: "VRAM",
    def: "The memory on a GPU that must hold the weights, gradients, optimizer states, and activations during training. Running out of it ('OOM') is the most common reason a run won't fit.",
  },
  kvCache: {
    term: "KV cache",
    def: "Stored key/value vectors from earlier tokens so a model doesn't recompute them while generating text. It matters for generation and for methods like PPO/GRPO that produce samples during training.",
  },
  dataParallel: {
    term: "Data parallelism (DP)",
    def: "Putting a full copy of the model on each GPU and feeding each a different slice of the batch, then averaging gradients. The simplest way to scale, but every GPU needs to hold the whole model.",
  },
  tensorParallel: {
    term: "Tensor parallelism (TP)",
    def: "Splitting each weight matrix across GPUs so they jointly compute one layer, syncing with an all-reduce. Shrinks per-GPU memory but needs fast interconnect, so it's usually kept within a node. Must evenly divide the attention heads.",
  },
  pipelineParallel: {
    term: "Pipeline parallelism (PP)",
    def: "Splitting the model's layers into stages on different GPUs, passing activations down the line like an assembly line. Scales across nodes well, but leaves some GPUs idle (the 'bubble').",
  },
  contextParallel: {
    term: "Context parallelism (CP)",
    def: "Splitting a single long sequence across GPUs along the token dimension, so each holds part of the context. Lets you train on very long sequences that wouldn't fit on one GPU.",
  },
  expertParallel: {
    term: "Expert parallelism (EP)",
    def: "Placing different experts of a Mixture-of-Experts model on different GPUs and routing each token to the GPUs holding its chosen experts. The standard way to scale MoE models.",
  },
  virtualPipeline: {
    term: "Virtual pipeline (interleaving)",
    def: "Giving each pipeline GPU several small, non-contiguous chunks of layers instead of one big block, which shrinks the idle pipeline bubble at the cost of a bit more communication.",
  },
  zero: {
    term: "ZeRO",
    def: "A DeepSpeed technique that removes the wasteful duplication in data parallelism by partitioning optimizer states, gradients, and parameters across GPUs instead of replicating them. Comes in three stages of increasing savings.",
  },
  zeroStages: {
    term: "ZeRO stages",
    def: "Stage 1 shards optimizer states (~4× less model-state memory), Stage 2 also shards gradients (~8×), and Stage 3 also shards parameters (memory falls roughly linearly with GPU count). Stages 1-2 add almost no extra communication; Stage 3 adds about 50%.",
  },
  fsdp: {
    term: "FSDP",
    def: "PyTorch's Fully Sharded Data Parallel — its native equivalent of ZeRO that shards parameters, gradients, and optimizer states across data-parallel GPUs. FULL_SHARD ≈ ZeRO-3, SHARD_GRAD_OP ≈ ZeRO-2.",
  },
  pipelineBubble: {
    term: "Pipeline bubble",
    def: "The idle time at the start and end of each step when pipeline stages are waiting to be filled or drained. More micro-batches (or virtual pipeline) shrink it; a common rule of thumb is at least 4× the number of pipeline stages.",
  },
  moe: {
    term: "Mixture of Experts (MoE)",
    def: "A model where each layer has many 'expert' sub-networks but each token only uses a few of them, so total parameters can be huge while compute per token stays modest. DeepSeek-V3 and Mixtral are examples.",
  },
  experts: {
    term: "Experts",
    def: "The parallel sub-networks inside an MoE layer. There can be hundreds; a router sends each token to only a small number of them, so most experts sit idle for any given token.",
  },
  topK: {
    term: "Top-k routing",
    def: "How many experts each token is sent to per MoE layer (often 1 or 2). Higher k means more experts active per token — more quality, more compute and communication.",
  },
  router: {
    term: "Router (gating)",
    def: "The small network in each MoE layer that scores the experts for a token and picks the top-k to run. Its choices determine which experts do the work.",
  },
  loadBalancing: {
    term: "Load balancing (MoE)",
    def: "Keeping tokens spread evenly across experts so none is overloaded while others idle. Imbalance wastes capacity and is usually discouraged with an auxiliary loss.",
  },
  gqa: {
    term: "GQA (Grouped-Query Attention)",
    def: "Letting several query heads share one key/value head, between full multi-head and single-head attention. Shrinks the KV cache and memory with little quality loss; used by Llama 2/3 70B.",
  },
  mqa: {
    term: "MQA (Multi-Query Attention)",
    def: "The extreme of GQA where all query heads share a single key/value head. Smallest KV cache and fastest generation, at some cost to quality.",
  },
  mla: {
    term: "MLA (Multi-head Latent Attention)",
    def: "DeepSeek's attention variant that compresses keys and values into a small shared latent vector, cutting KV-cache memory dramatically while keeping multi-head expressiveness.",
  },
  rope: {
    term: "RoPE (Rotary Position Embedding)",
    def: "A way of encoding token positions by rotating the query/key vectors, rather than adding a learned position vector. The de-facto standard in modern LLMs and friendly to long-context extension.",
  },
  swiglu: {
    term: "SwiGLU",
    def: "A gated feed-forward layer that multiplies two projections, one passed through a SiLU activation. It outperforms a plain MLP, which is why its hidden dimension is usually sized to ~⅔ × the naive width.",
  },
  tiedEmbeddings: {
    term: "Tied embeddings",
    def: "Reusing the same weight matrix for the input token embedding and the output projection (the LM head). Saves parameters and memory; common in smaller models like GPT-2.",
  },
  chinchillaOptimal: {
    term: "Chinchilla-optimal",
    def: "The compute-optimal balance between model size and training data found by the Chinchilla study: for a fixed compute budget, scale parameters and tokens together (about 20 tokens per parameter). Bigger isn't better if it's undertrained.",
  },
  tokensPerParameter: {
    term: "Tokens per parameter",
    def: "Training tokens divided by parameter count. Around 20 is the classic compute-optimal sweet spot; far below it underuses the model, far above it spends compute for diminishing loss gains.",
  },
  predictedLoss: {
    term: "Predicted loss (nats)",
    def: "An estimate of the model's final training loss in nats (natural-log units), from the Chinchilla scaling law for your size and token budget. Lower is better; it's a rough forecast, not a guarantee.",
  },
  criticalBatchSize: {
    term: "Critical batch size",
    def: "The batch size beyond which adding more parallel data stops speeding up training and mostly wastes compute. It grows as the loss falls, so larger batches pay off later in a run.",
  },
  sft: {
    term: "SFT (Supervised Fine-Tuning)",
    def: "Fine-tuning a base model on curated input→output examples so it follows instructions or a target style. The usual first step of post-training, and the cheapest.",
  },
  dpo: {
    term: "DPO (Direct Preference Optimization)",
    def: "Aligning a model from pairs of preferred vs. rejected responses directly, without training a separate reward model. Simpler and lighter than PPO, though it keeps a frozen reference copy in memory.",
  },
  ppo: {
    term: "PPO (Proximal Policy Optimization)",
    def: "A reinforcement-learning method (the classic RLHF algorithm) that improves the model against a reward signal while a critic estimates value. Powerful but memory-heavy: it holds the policy, a reference model, a reward model, and a critic at once.",
  },
  criticModel: {
    term: "Critic (value model)",
    def: "In PPO, a helper network that estimates the expected future reward of a partial response, used to reduce variance in the updates. It's roughly the size of the model being trained, so it adds substantial memory.",
  },
  rewardModel: {
    term: "Reward model",
    def: "A model that scores how good a response is, trained from human preferences. PPO and GRPO query it during training to decide which outputs to reinforce.",
  },
  grpo: {
    term: "GRPO (Group Relative Policy Optimization)",
    def: "An RL method that drops PPO's critic and instead samples a group of responses per prompt, scoring each relative to the group average. Cheaper than PPO because there's no value network to hold.",
  },
  lora: {
    term: "LoRA (Low-Rank Adaptation)",
    def: "Freezing the base model and training only small low-rank 'adapter' matrices added to chosen layers. Slashes trainable parameters and optimizer memory while reaching near-full-fine-tuning quality.",
  },
  loraRank: {
    term: "LoRA rank (r)",
    def: "The size of the LoRA adapter's bottleneck — higher rank means more capacity and more trainable parameters. Common values are 8 to 64.",
  },
  loraAlpha: {
    term: "LoRA alpha",
    def: "A scaling factor that controls how strongly the LoRA adapter is applied to the frozen base weights. It's typically set to a small multiple of the rank.",
  },
  loraTargetModules: {
    term: "LoRA target modules",
    def: "Which weight matrices get adapters — e.g. the attention query/value projections, or all linear layers. Targeting more modules raises quality and trainable-parameter count.",
  },
  qlora: {
    term: "QLoRA",
    def: "LoRA on top of a base model stored in 4-bit (NF4), dequantized on the fly for the math. It let a 65B model fine-tune on a single 48 GB GPU with no quality loss versus 16-bit.",
  },
  mezo: {
    term: "MeZO (zeroth-order)",
    def: "A fine-tuning method that estimates gradients from forward passes alone, so it stores no gradients or optimizer states — only the parameters. Extremely memory-light, but slower to converge.",
  },
  checkpoint: {
    term: "Checkpoint",
    def: "A saved snapshot of the model (and optimizer state) written to storage during training so a crashed run can resume. Frequent checkpoints cost storage but reduce lost work after a failure.",
  },
  failureOverhead: {
    term: "Failure overhead",
    def: "Extra wall-clock time and cost from hardware failures: detecting the crash, restarting, and re-doing the work since the last checkpoint. It grows with GPU count and run length.",
  },
}
```

**Grounding citations (definition → source):**
- tokens-per-parameter / Chinchilla-optimal / predicted loss: `chinchilla-scaling-laws-deep-dive.md:12` ("D_optimal is approximately 20N"), `:113` (E = irreducible loss), `:391` (Chinchilla 70B/1.4T = 20×).
- mixed precision / optimizerStates / fp32 / mixed-precision byte counts: `mixed-precision-training-micikevicius-2018-deep-dive.md:12,14,28,34` (fp32 master copy, "16-18 bytes per parameter rather than 4"); fp16 needing loss scaling `:71`; bf16 implied default.
- ZeRO stages: `zero-paper-rajbhandari-2019-deep-dive.md:118` (Stage 1 "4x reduction"), `:136` (Stage 2 "8x"), `:154/:320-324` (Stage 3 linear, Stage 1-2 zero comm overhead, Stage 3 +50%).
- FSDP↔ZeRO mapping: `types.ts:99-104` (`FSDPStrategy`); plan §2 (`pretraining-panel.tsx:117-131` ZeRO derived under FSDP).
- TP (heads divisibility, all-reduce, within-node): `megatron-lm-tensor-parallelism-deep-dive.md:12,43,55,212`.
- activation checkpointing + selective: `korthikanti-activation-recomputation-deep-dive.md:13,154` (selective recompute targets the dominant `5as/h` attention term).
- flash attention (exact, no N×N in HBM): `flash-attention-paper-deep-dive.md:6,22,67`.
- QLoRA (NF4 4-bit, 65B→<48GB, compute in bf16): `qlora-dettmers-2023-deep-dive.md:12,18`.
- critical batch size (grows as loss falls): `mccandlish-large-batch-training-deep-dive.md:6,198,203`.
- MoE / experts / router / EP: `deepseek-memory-analysis-zhang-su-deep-dive.md:12,22,110-112,196`.
- MeZO (no gradients/optimizer states): `constants.ts:1265-1266` ("Zeroth-order fine-tuning without gradients or optimizer states").
- MFU range / TFLOPS: `modal-gpu-utilization-guide-deep-dive.md` (general); pipeline-bubble rule "4× N_pp": `gpu-calculator.tsx:4714`.

---

## 2. Verdict-band microcopy

The verdict band (`verdict-band.tsx`) reads `memory.fits`, `cost.totalCost`, `trainingTime`, and the GPU count from the existing output object (no recompute — plan §3 component list). It has three base states plus post-training variants. Tone: teaching, amber-not-red, always one actionable next step (plan §7).

### Pretraining

**FITS (green)** — `memory.fits === true`:
```
✓ Fits  ·  {fmtCurrency(cost.totalCost)}  ·  {fmtDuration(trainingTime.theoreticalHours)}  ·  {effectiveNumGPUs}× {gpu.name}
```
- Mini-gauge from `gpu-utilization-gauge.tsx`; secondary line (optional, dense): `Peak per-GPU memory {fmtBytes(memory.total)} of {fmtBytes(memory.usableCapacity)} usable.`

**DOESN'T FIT YET (amber, teaching moment)** — `memory.fits === false`:
```
⚠ Doesn't fit yet — needs ~{minGPUsNeeded}× {gpu.name} (you have {effectiveNumGPUs}).
   This config wants {fmtBytes(memory.total)} per GPU but only {fmtBytes(memory.usableCapacity)} is usable.
   [ Fix for me → {minGPUsNeeded} GPUs ]
```
- The button text is literal: `Fix for me → {minGPUsNeeded} GPUs`. Action: `numGPUs := minGPUsNeeded` (plan §3 — no recompute, just sets the input). Never the word "Error" or "Failed."
- **Derived-GPU mode:** when `gpuCountDerivedFromTarget` is true (#GPUs is locked by Target training days), the action first clears `targetTrainingDays` (returning to explicit-GPU mode), then sets `numGPUs := minGPUsNeeded`; the button gains a sub-line `clears your target-days setting` so the side effect is never silent (plan §3).
- If `minGPUsNeeded` and current already match but still doesn't fit (e.g., single-GPU floor exceeded), fall back to: `⚠ Doesn't fit on a single {gpu.name} yet — try a larger-memory GPU or add tensor parallelism.` (no "Fix for me" button when there's no single-number fix).

**INVALID INPUT (neutral, not alarming)** — any field invalid (results dimmed, last-valid retained per plan Phase 5):
```
Check {fieldLabel} to update the verdict.  [ Fix {fieldLabel} → ]
```
- Button scroll-focuses the offending control (plan Phase 5). The band keeps showing the last-valid verdict, dimmed — never blanks. If multiple fields are invalid: `Check {n} fields to update the verdict.`

### Post-training variants

Post-training output has no `effectiveNumGPUs`/`minGPUsNeeded`; it uses `numGPUsNeeded` (nullable) + `numGPUsNeededMode`, `memory.fits`, `cost.totalCost`, `trainingTime`.

**FITS (green)** — `memory.fits === true`:
```
✓ Fits  ·  {fmtCurrency(cost.totalCost)}  ·  {fmtDuration(trainingTime.theoreticalHours)}  ·  {hardware.numGPUs}× {gpu.name}
```

**DOESN'T FIT YET (amber)** — `memory.fits === false`, `numGPUsNeeded !== null`:
- When `numGPUsNeededMode === "data-parallel"`:
```
⚠ Doesn't fit yet — split the work over ~{numGPUsNeeded} data-parallel {gpu.name}s.
   [ Fix for me → {numGPUsNeeded} GPUs ]
```
- When `numGPUsNeededMode === "state-sharded-lower-bound"`:
```
⚠ Doesn't fit yet — even fully sharded (ZeRO-3/FSDP) the model states need at least ~{numGPUsNeeded} {gpu.name}s,
   before activations and overhead. Try more GPUs, a smaller base model, or QLoRA.
```
  (No one-tap "Fix for me" here — it's a lower bound, not a guaranteed fit; offer the teaching alternatives instead.)

**DOESN'T FIT, no estimate** — `memory.fits === false`, `numGPUsNeeded === null`:
```
⚠ Doesn't fit yet on {hardware.numGPUs}× {gpu.name}. Try fewer trainable params (LoRA/QLoRA), a smaller base model, or more GPUs.
```

**AssumptionsLedger chip** (right edge of band, both phases) — when N silent substitutions exist:
```
{n} auto-adjustment{n === 1 ? "" : "s"} ▸
```
Opens the ledger. If `n === 0`, hide the chip.

**Critical warnings in the band** (both phases) — for each `warnings[]` entry with `severity === "critical"`, surface above the verdict line, prefixed:
```
Critical · {warning.message}
```
(Relabel "Error"→"Critical" per plan Phase 1; keep `warning.message` verbatim — predicate semantics unchanged, §7.)

---

## 3. Summary-line templates per layer (closed state)

One template per layer, shown when the `<Layer>` is collapsed (plan §3 LayerStack). Written as illustrative template strings over **verified** `CalculatorOutput` fields; the implementing agent wires the existing `fmt*` formatters. Each is "a sentence a human would say" (§7). `o` = the pretraining output object; post-training notes inline where the layer differs.

**Layer 1 — Memory & feasibility** (output-only, defaults OPEN — summary only used in dense/collapsed):
```ts
o.memory.fits
  ? `Fits — peak ${fmtBytes(o.memory.total)} of ${fmtBytes(o.memory.usableCapacity)} usable per GPU.`
  : `Over by ${fmtBytes(o.memory.total - o.memory.usableCapacity)} per GPU — needs ~${o.minGPUsNeeded} GPUs.`
// post-training: same, using o.memory.fits / o.memory.total / o.memory.usableCapacity;
//   append numGPUsNeeded clause only when non-null.
```

**Layer 2 — Performance & cost** (output-only, defaults OPEN):
```ts
`${fmtDuration(o.trainingTime.theoreticalHours)} · ${fmtCount(o.tokensPerSecond)} tok/s · ${fmtCount(o.trainingTime.totalSteps)} steps · ${fmtCurrency(o.cost.totalCost)} total.`
// post-training: `${fmtDuration(o.trainingTime.theoreticalHours)} · ${fmtCount(o.trainingTime.tokensPerSecond)} tok/s · ${fmtCount(o.trainingTime.totalSteps)} ${o.stepCountLabel} · ${fmtCurrency(o.cost.totalCost)} total.`
//   (post-training throughput is o.trainingTime.tokensPerSecond; pretraining exposes top-level o.tokensPerSecond — both verified.)
```

**Layer 3 — Parallelism** (closed; auto-opens & persists when `parallelismMode === "manual"`):
```ts
trainingConfig.parallelismMode === "auto"
  ? `Auto — we'll pick the layout: ${o.parallelismRecommendation.strategyLabel} (fits in ${o.minGPUsNeeded} GPUs).`
  : `Manual — ${o.parallelismRecommendation.strategyLabel}, ${fmtFractionPercent(o.pipelineBubbleFraction)} pipeline bubble.`
// strategyLabel is the pre-built human string (e.g. "TP=8, PP=2, DP=4, ZeRO-1, Megatron"), gpu-calculator.tsx:5577.
```

**Layer 4 — Model architecture** (closed):
```ts
`${fmtCount(o.parameterCounts.total)} params${o.moeSparsity ? ` (${fmtCount(o.parameterCounts.active)} active)` : ""} · ${arch.d}d × ${arch.L} layers · ${arch.a} heads · ${formatAttentionVariant(arch.attentionVariant)} · ${formatFFN(arch.ffnType)} · seq ${fmtCount(trainingConfig.sequenceLength)}${trainingConfig.flashAttention ? " · flash attn" : ""}.`
// arch = resolved ModelArchitecture; attentionVariant/ffnType are display-formatted (mha/gqa/mqa/mla, standard/swiglu/geglu/moe).
```

**Layer 5 — Precision & optimizer** (closed):
```ts
`${formatPrecision(trainingConfig.precision)} · ${optimizerProfile.name} · micro-batch ${trainingConfig.microBatchSize} × ${trainingConfig.gradientAccumulationSteps} accum · ${formatCheckpointing(trainingConfig.activationCheckpointing)} recompute.`
// optimizerProfile.name from OptimizerProfile (types.ts:289); checkpointing: none/selective/full/partial.
// If effectiveOptimizerId !== selected, the OverrideBadge handles it — do not duplicate in the summary.
```

**Layer 6 — Data & scaling** (closed):
```ts
`${fmtCount(trainingConfig.totalTokens)} tokens · ${fmtMultiplier(o.chinchilla.ratio)} tokens/param · predicted loss ${Number.isFinite(o.predictedLossNats) ? o.predictedLossNats.toFixed(2) : "—"} nats${o.dataRepetition.hasRepetition ? ` · ${o.dataRepetition.epochs.toFixed(1)} epochs` : ""}.`
// chinchilla.ratio, predictedLossNats, dataRepetition.epochs/hasRepetition all verified (types.ts:327-354).
```

**Layer 7 — Cost detail & failures** (closed):
```ts
`${fmtCurrency(o.cost.computeCost)} compute + ${fmtCurrency(o.cost.storageCost)} storage + ${fmtCurrency(o.cost.failureOverheadCost)} failures · ${fmtCount(o.cost.numCheckpoints)} checkpoints${o.trainingTime.failureMultiplier != null ? ` · ${fmtMultiplier(o.trainingTime.failureMultiplier)} failure factor` : ""}.`
// cost.computeCost/storageCost/failureOverheadCost/numCheckpoints and trainingTime.failureMultiplier all verified (types.ts:419-430, 408-417).
```

**Layer 8 — MoE** (closed + dimmed; auto-opens & un-dims when `moe.enabled`):
```ts
moe.enabled
  ? `${moe.E} experts, top-${moe.topk} · ${fmtCount(o.parameterCounts.active)} of ${fmtCount(o.parameterCounts.total)} params active${o.moeSparsity ? ` · ${fmtMultiplier(o.moeSparsity.efficiencyGain)} sparsity gain` : ""}.`
  : `Not a Mixture-of-Experts model.` // dimmed
// moe.E / moe.topk from MoEConfig (types.ts:32-42); moeSparsity.efficiencyGain (types.ts:432-436).
```

### Post-training variants (per-tab layer matrix, plan §3)

The templates above for Layers 3-8 reference `PretrainingOutput`-only fields (`parallelismRecommendation`, `chinchilla`, `moeSparsity`, `parameterCounts`, `dataRepetition`) — none exist on `PostTrainingOutput` (`types.ts:477-487`). On the Post-Training tab: **Layers 3 (Parallelism) and 8 (MoE) do not render at all**; Layers 1-2 use the post-training variants already noted inline above; Layers 4-7 use these (config echoes + verified `PostTrainingOutput` fields only — `p` = `postTrainingConfig`, `o` = post-training output):

```ts
// Layer 4 — Model architecture (reduced: base model + sequence length)
`${baseModelLabel(p.baseModel)} (frozen base) · seq ${fmtCount(p.sequenceLength)}.`
// baseModelLabel: preset name or `${fmtCount(parameterCount)} params` for by-size mode.

// Layer 5 — Precision & optimizer
`${formatPrecision(p.precision)} · ${optimizerProfile.name}${p.chunkedCrossEntropy ? " · chunked CE" : ""} · KV cache ${formatPrecision(p.kvCachePrecision)}.`

// Layer 6 — Data & scaling (reduced: dataset/epochs/batch; no Chinchilla — not computed post-training)
`${fmtCount(p.datasetSizeExamples)} ${dataUnitLabel(p.method)} · ${p.epochs} epoch${p.epochs === 1 ? "" : "s"} · batch ${p.batchSize}.`
// dataUnitLabel mirrors getPostTrainingDataUnitLabels (gpu-calculator.tsx:3143): examples/pairs/prompts by method.

// Layer 7 — Cost detail & failures (no cloud-instance/checkpoint inputs exist in PostTrainingConfig)
`${fmtCurrency(o.cost.computeCost)} compute${o.cost.storageCost ? ` + ${fmtCurrency(o.cost.storageCost)} storage` : ""}${o.cost.failureOverheadCost ? ` + ${fmtCurrency(o.cost.failureOverheadCost)} failures` : ""}${o.trainingTime.failureMultiplier != null ? ` · ${fmtMultiplier(o.trainingTime.failureMultiplier)} failure factor` : ""}.`
```

---

## 4. IntentRow cards + HeroBar + over-budget banner

### HeroBar (`hero-bar.tsx`)

**Headline (Fraunces H1):**
```
How many GPUs to train an LLM?
```
**Subline (value prop):**
```
Estimate the GPUs, cost, and wall-clock time for any pretraining or fine-tuning run — and see exactly why.
```
- Collapses to a thin bar on scroll (plan §3). Right side: theme toggle + `Dense view` affordance.
- `Dense view` toggle label / aria: `Dense view` (off) ⇄ `Calm view` (on). Tooltip: `Expand every layer and tighten spacing for auditing.`
- Preserve existing page `metadata` title/description on the root page (plan §1 Routing).

### IntentRow (`intent-row.tsx`)

**Collapsed affordance:**
```
New here? ▸
```
**Expanded heading (optional, small):**
```
What are you trying to do?
```

**Card 1 — Plan a pretraining run** (sets `activeTab = "pretraining"`, scroll-focus Essentials):
```
Title:    Plan a pretraining run
Subtitle: Train a model from scratch. We'll size the GPUs, cost, and time.
```

**Card 2 — Fine-tune a model** (sets `activeTab = "post-training"`):
```
Title:    Fine-tune a model
Subtitle: Adapt an existing model with SFT, LoRA, or RLHF.
```

**Card 3 — I know my config** (dismisses IntentRow, scroll-focus Essentials):
```
Title:    I know my config
Subtitle: Skip the intro — take me straight to the controls.
```
- Dismissal persists via `localStorage` (plan §3 `localStorage-dismissed`); invisible weight for returning users.

### Over-budget banner (verdict band's expanded teaching message, amber)

When `!memory.fits` and a one-number fix exists (`minGPUsNeeded` / `numGPUsNeeded`):
```
Doesn't fit yet — but it's close. This setup needs about {minGPUsNeeded} {gpu.name}s
({fmtBytes(memory.total)} per GPU vs. {fmtBytes(memory.usableCapacity)} usable).
Bump the GPU count, or trade memory for compute with activation checkpointing or a smaller micro-batch.
[ Fix for me → {minGPUsNeeded} GPUs ]
```
- Always amber, never red; always one concrete next step (§7). The bracketed button is the literal `Fix for me` action (`numGPUs := minGPUsNeeded`, no recompute).
- Post-training `state-sharded-lower-bound` variant drops the button and ends with: `Try more GPUs, a smaller base model, or switch to QLoRA.`

---

### Files referenced (all absolute):
- `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/components/gpu-calculator/types.ts` — all field names verified (`CalculatorOutput`, `PretrainingOutput:443-470`, `PostTrainingOutput:477-487`, `MemoryBreakdown:356-368`, `CostEstimate:419-430`, `TrainingTimeEstimate:408-417`, `ChinchillaAnalysis:327-338`, `MoEConfig:32-42`, `OptimizerProfile:288-297`).
- `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/components/gpu-calculator/gpu-calculator.tsx` — output construction + `fmt*` helper usage (pretraining markdown `:4896-4949`, post-training `:4959-4986`, post-training output object `:6387-6395`, `strategyLabel` build `:5577`, pipeline-bubble "4× N_pp" rule `:4714`).
- `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/components/gpu-calculator/constants.ts` — MeZO description `:1265-1266`, `DEFAULT_POST_TRAINING_CONFIG:1707`.
- `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/components/gpu-calculator/components/results-summary.tsx` — formatter home (reuse, don't reimplement).
- Research grounding: `spec/research/{chinchilla-scaling-laws,zero-paper-rajbhandari-2019,megatron-lm-tensor-parallelism,qlora-dettmers-2023,korthikanti-activation-recomputation,flash-attention-paper,mixed-precision-training-micikevicius-2018,mccandlish-large-batch-training,deepseek-memory-analysis-zhang-su}-deep-dive.md` (line cites in §1).

**Two notes for the implementing agent:** (1) The `fmt*` formatters are local (non-exported) helpers in `gpu-calculator.tsx`/`results-summary.tsx`; summary lines must be rendered where those helpers are in scope, or the helpers promoted to a shared module — do not duplicate formatting logic. (2) Pretraining throughput is the top-level `o.tokensPerSecond`; post-training throughput is `o.trainingTime.tokensPerSecond` — they live at different paths (both verified), so the two phases need slightly different templates as shown in Layer 2.
