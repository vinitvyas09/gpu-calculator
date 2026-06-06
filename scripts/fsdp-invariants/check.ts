// FSDP post-training sharding — formula-level verification harness.
//
// SEPARATE from scripts/parity/ (the byte-level browser gate, which stays
// untouched). This harness checks, at the formula layer:
//
//   1. DDP EQUIVALENCE (regression gate): the HEAD (pre-change) memory module
//      and the working-tree module produce EXACTLY equal outputs for a matrix
//      of DDP configs — including configs whose persisted state predates the
//      distributedStrategy field (undefined).
//   2. FSDP INVARIANTS: dominance (FSDP ≤ DDP at same N≥2), monotone
//      non-increasing per-GPU totals in N, N=1 equivalence (FSDP == DDP
//      exactly), and FSDP required-N ≥ the ideal ZeRO-3 lower bound.
//   3. The DeepSeek V3 671B / 48× B200-NVL72 LoRA hand-check (the motivating
//      scenario): DDP must remain 1541.3 GB; FSDP arithmetic shown term by
//      term.
//   4. Calibration against published measured datapoints (HF PEFT docs,
//      Answer.AI benchmarks) with deviations reported; >15% flagged.
//
// Usage:
//   git archive HEAD components/gpu-calculator | tar -x -C /tmp/head-calc
//   npx --yes tsx scripts/fsdp-invariants/check.ts
//
// (Old module resolves from /tmp/head-calc; refresh it after rebasing.)

import {
  calculateLoRAMemory as newLoRA,
  calculateQLoRAMemory as newQLoRA,
  calculateDPOMemory as newDPO,
  calculatePPOMemory as newPPO,
  calculateGRPOMemory as newGRPO,
} from "../../components/gpu-calculator/formulas/memory"
import {
  calculateLoRAMemory as oldLoRA,
  calculateQLoRAMemory as oldQLoRA,
  calculateDPOMemory as oldDPO,
  calculatePPOMemory as oldPPO,
  calculateGRPOMemory as oldGRPO,
  // eslint-disable-next-line import/no-relative-packages
} from "/tmp/head-calc/components/gpu-calculator/formulas/memory"
import {
  DEFAULT_POST_TRAINING_CONFIG,
  MODEL_PRESETS,
  GPU_SPECS,
} from "../../components/gpu-calculator/constants"
import type {
  PostTrainingConfig,
  PostTrainingMemoryBreakdown,
} from "../../components/gpu-calculator/types"

type Calc = (config: PostTrainingConfig) => PostTrainingMemoryBreakdown

const NEW: Record<string, Calc> = {
  lora: newLoRA,
  qlora: newQLoRA,
  dpo: newDPO,
  ppo: newPPO,
  grpo: newGRPO,
}
const OLD: Record<string, Calc> = {
  lora: oldLoRA,
  qlora: oldQLoRA,
  dpo: oldDPO,
  ppo: oldPPO,
  grpo: oldGRPO,
}

const preset = (id: string) => {
  const p = MODEL_PRESETS.find((m) => m.id === id)
  if (!p) throw new Error(`missing preset ${id}`)
  return p
}
const gpu = (id: string) => {
  const g = GPU_SPECS.find((s) => s.id === id)
  if (!g) throw new Error(`missing gpu ${id}`)
  return g
}

const DEFAULT_MOE = DEFAULT_POST_TRAINING_CONFIG.baseModel.moe

// Architectures for measured datapoints without presets.
const CUSTOM_ARCH: Record<
  string,
  { parameterCount: number; architecture: PostTrainingConfig["baseModel"]["architecture"] }
> = {
  "llama-2-7b": {
    parameterCount: 6.74e9,
    architecture: {
      d: 4096,
      L: 32,
      a: 32,
      a_kv: 32,
      d_ff: 11008,
      V: 32000,
      ffnType: "swiglu",
      normType: "rmsnorm",
      posEmbedding: "rope",
      attentionVariant: "mha",
      tiedEmbeddings: false,
    },
  },
  "yi-34b": {
    parameterCount: 34.389e9,
    architecture: {
      d: 7168,
      L: 60,
      a: 56,
      a_kv: 8,
      d_ff: 20480,
      V: 64000,
      ffnType: "swiglu",
      normType: "rmsnorm",
      posEmbedding: "rope",
      attentionVariant: "gqa",
      tiedEmbeddings: false,
    },
  },
}

function makeConfig(over: {
  presetId?: string
  customId?: string
  method?: PostTrainingConfig["method"]
  approach?: PostTrainingConfig["approach"]
  numGPUs?: number
  gpuId?: string
  precision?: PostTrainingConfig["precision"]
  quantizationBits?: 4 | 8 | null
  batchSize?: number
  sequenceLength?: number
  loraRank?: number
  targetModules?: PostTrainingConfig["lora"]["targetModules"]
  strategy?: PostTrainingConfig["distributedStrategy"] | undefined
  dropStrategyField?: boolean
}): PostTrainingConfig {
  const base = over.presetId ? preset(over.presetId) : null
  const custom = over.customId ? CUSTOM_ARCH[over.customId] : null
  if (over.customId && !custom) throw new Error(`missing custom ${over.customId}`)
  const hw = over.gpuId ? gpu(over.gpuId) : DEFAULT_POST_TRAINING_CONFIG.hardware.gpu
  const cfg: PostTrainingConfig = {
    ...DEFAULT_POST_TRAINING_CONFIG,
    baseModel: base
      ? {
          inputMode: "preset",
          presetId: base.id,
          parameterCount: base.parameterCount,
          architecture: base.architecture,
          moe: base.moe ?? { ...DEFAULT_MOE, enabled: false },
        }
      : custom
        ? {
            inputMode: "parameter-count",
            presetId: null,
            parameterCount: custom.parameterCount,
            architecture: custom.architecture,
            moe: { ...DEFAULT_MOE, enabled: false },
          }
        : DEFAULT_POST_TRAINING_CONFIG.baseModel,
    method: over.method ?? "sft",
    approach: over.approach ?? "lora",
    lora: {
      ...DEFAULT_POST_TRAINING_CONFIG.lora,
      rank: over.loraRank ?? DEFAULT_POST_TRAINING_CONFIG.lora.rank,
      targetModules:
        over.targetModules ?? DEFAULT_POST_TRAINING_CONFIG.lora.targetModules,
      quantizationBits:
        over.quantizationBits !== undefined
          ? over.quantizationBits
          : (over.approach ?? "lora") === "qlora"
            ? 4
            : null,
    },
    sequenceLength:
      over.sequenceLength ?? DEFAULT_POST_TRAINING_CONFIG.sequenceLength,
    batchSize: over.batchSize ?? DEFAULT_POST_TRAINING_CONFIG.batchSize,
    hardware: {
      inputMode: "preset",
      gpuId: hw.id,
      gpu: hw,
      numGPUs: over.numGPUs ?? 2,
    },
    precision: over.precision ?? "bf16",
    distributedStrategy: over.strategy ?? "ddp-replicated",
  }
  if (over.dropStrategyField) {
    // Simulate a config persisted before the field existed.
    delete (cfg as unknown as Record<string, unknown>).distributedStrategy
  }
  return cfg
}

function calcFor(cfg: PostTrainingConfig, table: Record<string, Calc>) {
  if (cfg.method === "dpo") return table.dpo(cfg)
  if (cfg.method === "ppo") return table.ppo(cfg)
  if (cfg.method === "grpo") return table.grpo(cfg)
  if (cfg.approach === "qlora") return table.qlora(cfg)
  return table.lora(cfg)
}

const stable = (b: PostTrainingMemoryBreakdown) =>
  JSON.stringify(b, (_k, v) =>
    typeof v === "number" && !Number.isFinite(v) ? `#${String(v)}` : v,
  )

let failures = 0
const fail = (msg: string) => {
  failures += 1
  console.error(`  ✗ ${msg}`)
}
const ok = (msg: string) => console.log(`  ✓ ${msg}`)
const gb = (x: number) => `${(x / 1e9).toFixed(2)} GB`

// ───────────────────────── 1. DDP equivalence (old vs new) ─────────────────
console.log("\n[1] DDP equivalence: HEAD module vs working tree (exact)")
const MATRIX: Array<Parameters<typeof makeConfig>[0]> = []
for (const presetId of [undefined, "llama-2-70b", "deepseek-v3-671b"]) {
  for (const numGPUs of [1, 2, 8, 48]) {
    MATRIX.push(
      { presetId, method: "sft", approach: "lora", numGPUs },
      { presetId, method: "sft", approach: "qlora", numGPUs },
      { presetId, method: "sft", approach: "qlora", numGPUs, quantizationBits: 8 },
      { presetId, method: "dpo", approach: "lora", numGPUs },
      { presetId, method: "dpo", approach: "full", numGPUs },
      { presetId, method: "ppo", approach: "lora", numGPUs },
      { presetId, method: "ppo", approach: "full", numGPUs },
      { presetId, method: "grpo", approach: "qlora", numGPUs },
      { presetId, method: "grpo", approach: "full", numGPUs },
    )
  }
}
MATRIX.push(
  { presetId: "llama-2-70b", approach: "lora", precision: "fp32", numGPUs: 4 },
  { presetId: "deepseek-v3-671b", approach: "lora", numGPUs: 48, gpuId: "b200-nvl72" },
)

let eqCount = 0
for (const spec of MATRIX) {
  const cfg = makeConfig(spec)
  const oldOut = stable(calcFor(cfg, OLD))
  const newOut = stable(calcFor(cfg, NEW))
  if (oldOut !== newOut) {
    fail(`old≠new for ${JSON.stringify(spec)}`)
    console.error(`    old: ${oldOut.slice(0, 300)}`)
    console.error(`    new: ${newOut.slice(0, 300)}`)
  } else eqCount++
  // Same config with the strategy field ABSENT (old persisted state) must
  // also match byte-for-byte.
  const legacy = makeConfig({ ...spec, dropStrategyField: true })
  if (stable(calcFor(legacy, NEW)) !== oldOut) {
    fail(`legacy-config (no strategy field) ≠ old for ${JSON.stringify(spec)}`)
  } else eqCount++
}
if (failures === 0) ok(`${eqCount} comparisons byte-identical (incl. legacy configs)`)

// ───────────────────────── 2. FSDP invariants ──────────────────────────────
console.log("\n[2] FSDP invariants")
const INV_CASES: Array<Parameters<typeof makeConfig>[0]> = [
  { presetId: "llama-2-70b", method: "sft", approach: "lora", gpuId: "h100-sxm" },
  { presetId: "llama-2-70b", method: "sft", approach: "qlora", gpuId: "h100-sxm" },
  { presetId: "deepseek-v3-671b", method: "sft", approach: "lora", gpuId: "b200-nvl72" },
  { presetId: "llama-2-70b", method: "dpo", approach: "full", gpuId: "h100-sxm" },
  { presetId: "llama-2-70b", method: "ppo", approach: "lora", gpuId: "h100-sxm" },
  { presetId: "llama-2-70b", method: "grpo", approach: "qlora", gpuId: "h100-sxm" },
]
for (const spec of INV_CASES) {
  const name = `${spec.presetId}/${spec.method}/${spec.approach}`
  // dominance + monotonicity
  let prev = Number.POSITIVE_INFINITY
  let monotone = true
  let dominated = true
  for (const n of [2, 4, 8, 16, 48, 96, 256]) {
    const ddp = calcFor(makeConfig({ ...spec, numGPUs: n }), NEW)
    const fsdp = calcFor(
      makeConfig({ ...spec, numGPUs: n, strategy: "fsdp-full-shard" }),
      NEW,
    )
    if (!(fsdp.total <= ddp.total + 1e-6)) {
      dominated = false
      fail(`${name} N=${n}: FSDP ${gb(fsdp.total)} > DDP ${gb(ddp.total)}`)
    }
    if (!(fsdp.total <= prev + 1e-6)) {
      monotone = false
      fail(`${name}: total increased ${gb(prev)} → ${gb(fsdp.total)} at N=${n}`)
    }
    prev = fsdp.total
  }
  if (dominated && monotone) ok(`${name}: FSDP ≤ DDP and monotone over N`)

  // N=1 equivalence: FSDP at world size 1 == DDP exactly (gather buffer 0,
  // capacity factor collapses to 0.9).
  const ddp1 = stable(calcFor(makeConfig({ ...spec, numGPUs: 1 }), NEW))
  const fsdp1Raw = calcFor(
    makeConfig({ ...spec, numGPUs: 1, strategy: "fsdp-full-shard" }),
    NEW,
  )
  // items labels intentionally annotate under FSDP only when sharding; at
  // shardDegree 1 the gather is 0 but `fsdp` labels still apply — compare
  // numeric fields only.
  const fsdp1 = calcFor(
    makeConfig({ ...spec, numGPUs: 1, strategy: "fsdp-full-shard" }),
    NEW,
  )
  const numericKeys = [
    "parameters",
    "gradients",
    "optimizerStates",
    "activations",
    "communicationBuffers",
    "frameworkOverhead",
    "total",
    "usableCapacity",
    "trainableModels",
    "frozenModels",
    "loraAdapter",
    "ppoBuffers",
  ] as const
  const ddp1Obj = calcFor(makeConfig({ ...spec, numGPUs: 1 }), NEW)
  const n1Mismatch = numericKeys.filter(
    (k) => stable({ [k]: fsdp1[k] } as never) !== stable({ [k]: ddp1Obj[k] } as never),
  )
  if (n1Mismatch.length > 0) {
    fail(`${name}: FSDP(N=1) ≠ DDP(N=1) on ${n1Mismatch.join(", ")}`)
  } else {
    ok(`${name}: FSDP(N=1) == DDP(N=1) on all numeric fields`)
  }
  void ddp1
  void fsdp1Raw

  // FSDP required-N ≥ ideal ZeRO-3 state-sharded lower bound.
  // Lower bound (replicates estimatePostTrainingRequiredGPUs's formula, DDP
  // semantics at N=1): ceil(persistentStates / (usable/1.04 − overhead)).
  const one = calcFor(makeConfig({ ...spec, numGPUs: 1 }), NEW)
  const persistent = one.parameters + one.gradients + one.optimizerStates
  const cap = one.usableCapacity / 1.04 - one.frameworkOverhead
  if (Number.isFinite(persistent) && persistent > 0 && cap > 0) {
    const lowerBound = Math.max(1, Math.ceil(persistent / cap))
    let requiredFSDP: number | null = null
    for (let n = 1; n <= 4096; n = n < 8 ? n + 1 : Math.ceil(n * 1.25)) {
      const m = calcFor(
        makeConfig({ ...spec, numGPUs: n, strategy: "fsdp-full-shard" }),
        NEW,
      )
      if (m.fits) {
        requiredFSDP = n
        break
      }
    }
    if (requiredFSDP !== null && requiredFSDP < lowerBound) {
      fail(`${name}: FSDP fits at N=${requiredFSDP} < ideal lower bound ${lowerBound}`)
    } else {
      ok(
        `${name}: FSDP required-N ${requiredFSDP ?? ">4096/none"} ≥ lower bound ${lowerBound}`,
      )
    }
  }
}

// ───────────────────────── 3. DeepSeek V3 hand-check ───────────────────────
console.log("\n[3] DeepSeek V3 671B · SFT+LoRA · 48× B200-NVL72 (motivating scenario)")
{
  const spec = {
    presetId: "deepseek-v3-671b",
    method: "sft" as const,
    approach: "lora" as const,
    gpuId: "b200-nvl72",
    numGPUs: 48,
  }
  const ddp = calcFor(makeConfig(spec), NEW)
  console.log(`  DDP  total: ${(ddp.total / 1e9).toFixed(1)} GB (must be 1541.3 GB exactly)`)
  if (Math.abs(ddp.total / 1e9 - 1541.3) > 0.05) {
    fail(`DDP total drifted: ${(ddp.total / 1e9).toFixed(2)} GB ≠ 1541.3 GB`)
  } else ok("DDP total unchanged: 1541.3 GB")

  const fsdp = calcFor(makeConfig({ ...spec, strategy: "fsdp-full-shard" }), NEW)
  console.log(`  FSDP per-GPU breakdown @48:`)
  console.log(`    parameters (base+LoRA, sharded): ${gb(fsdp.parameters)}`)
  console.log(`    gradients (sharded):             ${gb(fsdp.gradients)}`)
  console.log(`    optimizer states (sharded):      ${gb(fsdp.optimizerStates)}`)
  console.log(`    activations (per-GPU):           ${gb(fsdp.activations)}`)
  console.log(`    all-gather buffer:               ${gb(fsdp.communicationBuffers)}`)
  console.log(`    framework overhead:              ${gb(fsdp.frameworkOverhead)}`)
  console.log(`    TOTAL (×1.04):                   ${gb(fsdp.total)}`)
  console.log(`    usable capacity (0.8×186GB):     ${gb(fsdp.usableCapacity)}`)
  console.log(`    fits: ${fsdp.fits}`)
  // Hand arithmetic (LoRA): base = 671e9 × 2 B ÷ 48; gather = 2 × min(block,1e9) × 2 B = 4 GB.
  const expectBaseShard = (671e9 * 2) / 48
  const handParams = Math.abs(
    fsdp.frozenModels - expectBaseShard,
  )
  if (handParams > expectBaseShard * 0.001) {
    fail(
      `frozen base shard ${gb(fsdp.frozenModels)} ≠ hand 671e9×2/48 = ${gb(expectBaseShard)}`,
    )
  } else ok(`frozen base shard matches hand arithmetic: ${gb(expectBaseShard)}`)
  const expectGather = 2 * 1e9 * 2 // capped block (1e9) × bf16 × 2 units
  if (Math.abs(fsdp.communicationBuffers - expectGather) > 1e6) {
    fail(
      `gather buffer ${gb(fsdp.communicationBuffers)} ≠ hand 2×1e9×2B = ${gb(expectGather)} (cap should bind for DSv3's ~11B-param MoE blocks)`,
    )
  } else ok(`all-gather buffer matches hand arithmetic: ${gb(expectGather)} (1e9 cap binding)`)
}

// ───────────────────────── 4. Calibration datapoints ───────────────────────
console.log("\n[4] Calibration vs published measured datapoints (reserved-style numbers)")
const ALL_LINEAR: PostTrainingConfig["lora"]["targetModules"] = [
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
]
// NOTE: this calculator's batchSize is the GLOBAL batch (split across GPUs by
// getPostTrainingPerGpuBatch); published runs report PER-DEVICE batch, so
// batchSize below = per-device × world size.
const GIB = 1.073741824
const DATAPOINTS: Array<{
  label: string
  spec: Parameters<typeof makeConfig>[0]
  measuredGB: number
  note: string
}> = [
  {
    label:
      "HF PEFT: Llama-2-70B LoRA r8 all-linear, 8×H100-80GB, bf16, seq2048, per-dev bs8",
    spec: {
      presetId: "llama-2-70b",
      approach: "lora",
      gpuId: "h100-sxm",
      numGPUs: 8,
      sequenceLength: 2048,
      batchSize: 64,
      loraRank: 8,
      targetModules: ALL_LINEAR,
      strategy: "fsdp-full-shard",
    },
    measuredGB: 76, // 72–80 GB reported (90–98% of 80GB, reserved-style)
    note: "GC on, FA2, packing",
  },
  {
    label:
      "HF PEFT: Llama-2-70B QLoRA NF4 r8 all-linear, 2×24GB, seq2048, per-dev bs2, NO offload",
    spec: {
      presetId: "llama-2-70b",
      approach: "qlora",
      gpuId: "rtx-3090",
      numGPUs: 2,
      sequenceLength: 2048,
      batchSize: 4,
      loraRank: 8,
      targetModules: ALL_LINEAR,
      strategy: "fsdp-full-shard",
    },
    measuredGB: 35.6,
    note: "GC on (reentrant); exceeds the 24GB card — figure from PEFT docs",
  },
  {
    label: "Answer.AI: Yi-34B QLoRA, 2×3090, seq2048, per-dev bs2, no offload",
    spec: {
      customId: "yi-34b",
      approach: "qlora",
      gpuId: "rtx-3090",
      numGPUs: 2,
      sequenceLength: 2048,
      batchSize: 4,
      strategy: "fsdp-full-shard",
    },
    measuredGB: 23.05 * GIB,
    note: "peak reserved 23.05 GiB; 4-bit shard ≈ 8.0–8.5 GiB of it",
  },
  {
    label: "Answer.AI: Llama-2-7B QLoRA, 2×3090, seq2048, per-dev bs1, no offload",
    spec: {
      customId: "llama-2-7b",
      approach: "qlora",
      gpuId: "rtx-3090",
      numGPUs: 2,
      sequenceLength: 2048,
      batchSize: 2,
      strategy: "fsdp-full-shard",
    },
    measuredGB: 5.21 * GIB,
    note: "peak reserved",
  },
  {
    label: "Answer.AI: Llama-2-7B LoRA 16-bit, 2×3090, seq2048, per-dev bs1",
    spec: {
      customId: "llama-2-7b",
      approach: "lora",
      gpuId: "rtx-3090",
      numGPUs: 2,
      sequenceLength: 2048,
      batchSize: 2,
      strategy: "fsdp-full-shard",
    },
    measuredGB: 10.22 * GIB, // reserved; allocated was 9.16 GiB
    note: "peak reserved 10.22 GiB / allocated 9.16 GiB",
  },
]
for (const dp of DATAPOINTS) {
  let cfg: PostTrainingConfig
  try {
    cfg = makeConfig(dp.spec)
  } catch (e) {
    fail(`${dp.label}: ${String(e)}`)
    continue
  }
  const out = calcFor(cfg, NEW)
  const modeled = out.total / 1e9
  const dev = ((modeled - dp.measuredGB) / dp.measuredGB) * 100
  const flagged = Math.abs(dev) > 15
  console.log(
    `  ${flagged ? "⚑" : "·"} ${dp.label}\n      modeled ${modeled.toFixed(1)} GB vs measured ${dp.measuredGB.toFixed(1)} GB → ${dev >= 0 ? "+" : ""}${dev.toFixed(1)}%  (${dp.note})`,
  )
}
console.log(
  "  (modeled = allocated-style + 4% scalar; published numbers are reserved/nvidia-smi-style ≈ +10–12%)",
)

console.log(
  failures === 0
    ? "\nALL CHECKS PASSED"
    : `\n${failures} FAILURE(S) — see above`,
)
process.exit(failures === 0 ? 0 : 1)
