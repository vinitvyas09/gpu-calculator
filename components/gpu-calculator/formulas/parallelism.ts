/**
 * Parallelism recommendation engine — Spec Sections 9, 5.2, 5.7
 *
 * Pure TypeScript functions. No React, no DOM.
 */
import type {
  FrameworkType,
  GPUSpec,
  MemoryBreakdown,
  ModelArchitecture,
  MoEConfig,
  ParameterCounts,
  ParallelismConfig,
  ParallelismRecommendation,
  TrainingConfig,
  Warning,
  ZeROStage,
} from "../types"
import {
  calculateTotalMemoryPerGPU,
  calculateMinGPUVRAMFloor,
} from "./memory"

// ─── Types ──────────────────────────────────────────────────────────────────

export interface ValidationResult {
  valid: boolean
  message: string
}

export interface ScoredConfiguration {
  config: ParallelismConfig
  score: number
  memory: MemoryBreakdown
  label: string
}

// ─── Constraint Validators ──────────────────────────────────────────────────

export function validateTPDivisibility(
  N_tp: number,
  a: number,
  a_kv: number | null,
  d_ff: number
): ValidationResult {
  if (N_tp <= 1) return { valid: true, message: "No TP active" }

  if (a % N_tp !== 0) {
    return { valid: false, message: `N_tp=${N_tp} does not evenly divide attention heads a=${a}` }
  }

  if (a_kv !== null && a_kv % N_tp !== 0) {
    return { valid: false, message: `N_tp=${N_tp} does not evenly divide KV heads a_kv=${a_kv}` }
  }

  if (d_ff % N_tp !== 0) {
    return { valid: false, message: `N_tp=${N_tp} does not evenly divide d_ff=${d_ff}` }
  }

  return { valid: true, message: "TP dimensions are compatible" }
}

export function validatePPDivisibility(
  N_pp: number,
  L: number
): ValidationResult {
  if (N_pp <= 1) return { valid: true, message: "No PP active" }

  if (L % N_pp === 0) {
    return { valid: true, message: `L=${L} divides into ${N_pp} stages (${L / N_pp} layers each)` }
  }

  if ((L + 2) % N_pp === 0) {
    return {
      valid: true,
      message: `Embedding-aware partitioning: (L+2)=${L + 2} divides into ${N_pp} stages`,
    }
  }

  return {
    valid: false,
    message: `Neither L=${L} nor (L+2)=${L + 2} is divisible by N_pp=${N_pp}`,
  }
}

export function validateZeroPPCompatibility(
  zeroStage: ZeROStage,
  N_pp: number,
  framework: FrameworkType
): ValidationResult {
  if (N_pp <= 1) return { valid: true, message: "No PP active" }

  if (framework === "deepspeed" && zeroStage >= 2) {
    return {
      valid: false,
      message: `DeepSpeed ZeRO-${zeroStage} is incompatible with pipeline parallelism. Use ZeRO-0 or ZeRO-1.`,
    }
  }

  if ((framework === "fsdp" || framework === "hf_trainer") && zeroStage === 3) {
    return {
      valid: false,
      message: "FSDP FULL_SHARD (ZeRO-3) is incompatible with pipeline parallelism.",
    }
  }

  return { valid: true, message: "ZeRO stage compatible with PP" }
}

export function validateWorldSize(
  config: ParallelismConfig,
  numGPUs: number,
  moeEnabled: boolean
): ValidationResult {
  const world = moeEnabled
    ? config.N_dp * config.N_tp * config.N_pp * config.N_cp * config.N_ep
    : config.N_dp * config.N_tp * config.N_pp * config.N_cp

  if (world !== numGPUs) {
    return { valid: false, message: `World size ${world} ≠ ${numGPUs} GPUs` }
  }

  return { valid: true, message: `World size ${world} = ${numGPUs} GPUs` }
}

export function validateMicrobatches(
  numMicrobatches: number,
  N_pp: number,
  VP: number
): ValidationResult {
  if (N_pp <= 1) return { valid: true, message: "No PP active" }

  if (numMicrobatches < N_pp - 1) {
    return {
      valid: false,
      message: `1F1B schedule requires num_microbatches (${numMicrobatches}) ≥ N_pp-1 (${N_pp - 1})`,
    }
  }

  if (VP > 1 && numMicrobatches % N_pp !== 0) {
    return {
      valid: false,
      message: `Interleaved PP requires num_microbatches (${numMicrobatches}) divisible by N_pp (${N_pp})`,
    }
  }

  return { valid: true, message: "Microbatch count valid for PP schedule" }
}

export function validateHiddenDimAlignment(d: number): ValidationResult {
  if (d % 128 !== 0) {
    return {
      valid: false,
      message: `Hidden dimension d=${d} not aligned to 128 (causes ~38% throughput loss from partial tensor core tiles)`,
    }
  }

  return { valid: true, message: "Hidden dimension aligned to 128" }
}

export function calculateVocabPadding(V: number, N_tp: number): number {
  if (N_tp <= 1) return V
  const alignment = 128 * N_tp
  return Math.ceil(V / alignment) * alignment
}

// ─── Pipeline Bubble ──────────────────────────────────────────────────────────

export function calculatePipelineBubble(
  N_pp: number,
  numMicrobatches: number,
  VP: number = 1
): number {
  if (N_pp <= 1) return 0
  if (numMicrobatches <= 0) return 1

  const effectiveMicrobatches = VP > 1 ? VP * numMicrobatches : numMicrobatches
  return (N_pp - 1) / (effectiveMicrobatches + N_pp - 1)
}

// ─── Internal Helpers ──────────────────────────────────────────────────────────

function resolveFFNIntermediateSize(arch: ModelArchitecture, moe: MoEConfig): number {
  if (moe.enabled && moe.denseIntermediateSize) return moe.denseIntermediateSize
  if (arch.d_ff !== null) return arch.d_ff
  const isSwiGLU = arch.ffnType === "swiglu" || arch.ffnType === "geglu" || arch.ffnType === "moe"
  return isSwiGLU ? Math.round((8 / 3) * arch.d) : 4 * arch.d
}

function isPCIeOnly(gpu: GPUSpec): boolean {
  return gpu.interconnect === "pcie" || gpu.interconnect === "none"
}

function clampToPowerOf2(value: number): number {
  if (value <= 1) return 1
  return Math.pow(2, Math.round(Math.log2(value)))
}

function checkMemoryFit(
  params: ParameterCounts,
  baseConfig: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  parallelism: ParallelismConfig
): MemoryBreakdown {
  return calculateTotalMemoryPerGPU(
    params,
    { ...baseConfig, parallelism },
    arch,
    moe,
    gpu
  )
}

function makeStrategyLabel(p: ParallelismConfig, moeEnabled: boolean): string {
  const parts: string[] = []
  parts.push(`DP=${p.N_dp}`)
  if (p.N_tp > 1) parts.push(`TP=${p.N_tp}`)
  if (p.N_pp > 1) parts.push(`PP=${p.N_pp}`)
  if (p.N_cp > 1) parts.push(`CP=${p.N_cp}`)
  if (moeEnabled && p.N_ep > 1) parts.push(`EP=${p.N_ep}`)
  parts.push(`ZeRO-${p.zeroStage}`)
  return parts.join(", ")
}

function estimateMaxMicroBatch(
  memory: MemoryBreakdown,
  currentMicroBatch: number
): number {
  if (memory.activations <= 0 || currentMicroBatch <= 0) return Math.max(1, currentMicroBatch)

  const activationPerSample = memory.activations / currentMicroBatch
  if (activationPerSample <= 0) return currentMicroBatch

  const nonActivation =
    memory.parameters + memory.gradients + memory.optimizerStates +
    memory.communicationBuffers + memory.frameworkOverhead
  const availableRaw = memory.usableCapacity / 1.04 - nonActivation

  if (availableRaw <= 0) return 0
  return Math.max(1, Math.floor(availableRaw / activationPerSample))
}

function getZeROStagesForPP(framework: FrameworkType): ZeROStage[] {
  if (framework === "deepspeed") return [1, 0]
  if (framework === "fsdp" || framework === "hf_trainer") return [1, 0, 2]
  return [1, 0, 2, 3]
}

function getEPCandidates(E: number, N_tp: number, gpusPerNode: number): number[] {
  const candidates: number[] = [1]
  if (E <= 0) return candidates

  const maxEP = Math.floor(gpusPerNode / N_tp)
  for (let nep = 2; nep <= Math.min(E, maxEP); nep++) {
    if (E % nep === 0) candidates.push(nep)
  }

  return candidates
}

function findMinimumGPUCount(
  params: ParameterCounts,
  baseConfig: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  recommended: ParallelismConfig,
  framework: FrameworkType,
  moeEnabled: boolean
): number {
  const topologyBase =
    recommended.N_tp * recommended.N_pp * recommended.N_cp *
    (moeEnabled ? recommended.N_ep : 1)

  const maxZeRO: ZeROStage = recommended.N_pp > 1
    ? (framework === "deepspeed" ? 1 : (framework === "fsdp" || framework === "hf_trainer") ? 2 : 3)
    : 3

  for (let dp = 1; dp <= 2048; dp++) {
    const p: ParallelismConfig = { ...recommended, N_dp: dp, zeroStage: maxZeRO }
    const mem = checkMemoryFit(params, baseConfig, arch, moe, gpu, p)
    if (mem.fits) return topologyBase * dp
  }

  return topologyBase
}

// ─── Throughput Scoring ──────────────────────────────────────────────────────

export function scoreConfigurations(
  configs: Array<{ config: ParallelismConfig; memory: MemoryBreakdown; label: string }>,
  currentMicroBatch: number,
  gradientAccumulationSteps: number
): ScoredConfiguration[] {
  return configs
    .map(({ config, memory, label }) => {
      const maxBatch = estimateMaxMicroBatch(memory, currentMicroBatch)
      const hasTP = config.N_tp > 1
      const hasPP = config.N_pp > 1
      const isZeRO3 = config.zeroStage === 3
      const isDPOnly = !hasTP && !hasPP && config.N_cp === 1 && config.N_ep === 1

      let score: number
      if (isDPOnly && !isZeRO3) {
        score = maxBatch * config.N_dp * 1.5
      } else if (isDPOnly && isZeRO3) {
        score = maxBatch * config.N_dp
      } else if (hasTP && config.N_dp <= 1 && !hasPP) {
        score = maxBatch
      } else {
        score = maxBatch * config.N_dp
      }

      if (hasPP) {
        const bubble = calculatePipelineBubble(
          config.N_pp,
          gradientAccumulationSteps,
          config.VP
        )
        score *= 1 - bubble
      }

      return { config, score, memory, label }
    })
    .sort((a, b) => b.score - a.score)
}

// ─── Main Recommendation Engine ──────────────────────────────────────────────

export function recommendParallelism(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig
): ParallelismRecommendation {
  const reasoning: string[] = []
  const warnings: Warning[] = []
  const framework = config.parallelism.framework
  const resolvedDFF = resolveFFNIntermediateSize(arch, moe)
  const pcie = isPCIeOnly(gpu)
  const moeEnabled = moe.enabled && moe.E > 0

  if (!validateHiddenDimAlignment(arch.d).valid) {
    warnings.push({
      severity: "warning",
      category: "parallelism",
      message: `Hidden dimension d=${arch.d} is not aligned to 128, causing significant throughput loss.`,
    })
  }

  // ── Helpers ──

  function makeConfig(overrides: Partial<ParallelismConfig>): ParallelismConfig {
    return {
      N_dp: numGPUs,
      N_tp: 1,
      N_pp: 1,
      N_cp: 1,
      N_ep: 1,
      zeroStage: 0 as ZeROStage,
      fsdpStrategy: null,
      framework,
      sequenceParallelism: config.parallelism.sequenceParallelism,
      VP: config.parallelism.VP,
      ...overrides,
    }
  }

  function test(p: ParallelismConfig): MemoryBreakdown {
    return checkMemoryFit(params, config, arch, moe, gpu, p)
  }

  type Candidate = { config: ParallelismConfig; memory: MemoryBreakdown; label: string }

  const allowedPPStages = getZeROStagesForPP(framework)
  const maxTP = Math.min(gpu.gpusPerNode, numGPUs)
  const tpDegrees = [2, 4, 8].filter((t) => t <= maxTP)
  const tpAll = [1, ...tpDegrees]
  const ppDegrees = [2, 4, 8, 16, 32].filter((p) => p <= numGPUs)
  const zeroOrderTP: ZeROStage[] = pcie ? [3, 2, 1, 0] : [1, 0, 2, 3]

  /**
   * Generates all valid parallelism configs for given TP/PP/CP ranges.
   * Tests each against memory and returns those that fit.
   */
  function collectCandidates(
    tpRange: number[],
    ppRange: number[],
    ncp: number,
    zeroStagesNoPP: ZeROStage[]
  ): Candidate[] {
    const result: Candidate[] = []

    for (const ntp of tpRange) {
      if (ntp > 1 && !validateTPDivisibility(ntp, arch.a, arch.a_kv, resolvedDFF).valid) continue

      for (const npp of ppRange) {
        if (npp > 1 && !validatePPDivisibility(npp, arch.L).valid) continue
        const stages = npp > 1 ? allowedPPStages : zeroStagesNoPP

        const epRange = moeEnabled
          ? getEPCandidates(moe.E, ntp, gpu.gpusPerNode)
          : [1]

        for (const nep of epRange) {
          const topology = ntp * npp * ncp * nep
          if (topology > numGPUs || numGPUs % topology !== 0) continue
          const ndp = numGPUs / topology

          for (const stage of stages) {
            const p = makeConfig({
              N_dp: ndp,
              N_tp: ntp,
              N_pp: npp,
              N_cp: ncp,
              N_ep: nep,
              zeroStage: stage,
            })
            const mem = test(p)
            if (mem.fits) {
              result.push({ config: p, memory: mem, label: makeStrategyLabel(p, moeEnabled) })
            }
          }
        }
      }
    }

    return result
  }

  function finalize(best: ScoredConfiguration): ParallelismRecommendation {
    const p = best.config
    const testCfg: TrainingConfig = { ...config, parallelism: p }
    const minVRAMFloor = calculateMinGPUVRAMFloor(params, testCfg)
    const bubble = calculatePipelineBubble(
      p.N_pp,
      config.gradientAccumulationSteps,
      p.VP
    )
    const minGPUs = findMinimumGPUCount(
      params, config, arch, moe, gpu, p, framework, moeEnabled
    )

    if (p.N_pp > 1) {
      const mbVal = validateMicrobatches(config.gradientAccumulationSteps, p.N_pp, p.VP)
      if (!mbVal.valid) {
        warnings.push({ severity: "warning", category: "parallelism", message: mbVal.message })
      }
      if (bubble > 0.2) {
        warnings.push({
          severity: "warning",
          category: "parallelism",
          message: `Pipeline bubble is ${(bubble * 100).toFixed(1)}%. Increase gradient accumulation steps to ≥${4 * p.N_pp} for <20% bubble.`,
        })
      }
    }

    if (p.N_tp > 1 && p.N_tp > gpu.gpusPerNode) {
      warnings.push({
        severity: "warning",
        category: "parallelism",
        message: `TP=${p.N_tp} exceeds GPUs per node (${gpu.gpusPerNode}). TP across nodes requires very high-bandwidth interconnect.`,
      })
    }

    if (p.N_tp > 1) {
      const paddedV = calculateVocabPadding(arch.V, p.N_tp)
      if (paddedV > arch.V) {
        warnings.push({
          severity: "info",
          category: "parallelism",
          message: `Vocabulary padded from ${arch.V.toLocaleString()} to ${paddedV.toLocaleString()} for TP=${p.N_tp} alignment.`,
        })
      }
    }

    return {
      config: p,
      minGPUs,
      minVRAMFloor,
      pipelineBubbleFraction: bubble,
      strategyLabel: makeStrategyLabel(p, moeEnabled),
      reasoning,
      warnings,
    }
  }

  // ── Single device ──

  if (gpu.singleDeviceOnly || numGPUs === 1) {
    const p = makeConfig({ N_dp: 1 })
    const mem = test(p)

    if (!mem.fits) {
      warnings.push({
        severity: "critical",
        category: "memory",
        message: "Model does not fit on a single GPU. Multi-GPU training required.",
      })
    }

    reasoning.push(
      gpu.singleDeviceOnly
        ? `${gpu.name} supports single-device training only`
        : "Single GPU: using data parallelism only"
    )

    return finalize({ config: p, score: 0, memory: mem, label: makeStrategyLabel(p, moeEnabled) })
  }

  // ── Phase 1: DP-only with ZeRO ──

  const dpCandidates = collectCandidates(
    [1], [1], 1,
    [0, 1, 2, 3] as ZeROStage[]
  )

  if (dpCandidates.length > 0) {
    reasoning.push(`Model fits with data parallelism only (${numGPUs} GPUs)`)
    const scored = scoreConfigurations(
      dpCandidates, config.microBatchSize, config.gradientAccumulationSteps
    )
    reasoning.push(`Selected ${scored[0].label} (best throughput among ${dpCandidates.length} DP-only options)`)
    return finalize(scored[0])
  }

  reasoning.push("Model does not fit with data parallelism alone")

  // ── Phase 2: TP + DP (+ EP for MoE) ──

  if (pcie) {
    reasoning.push("PCIe interconnect: preferring ZeRO-3 over tensor parallelism where possible")
  }

  const phase2TP = moeEnabled ? tpAll : tpDegrees
  const tpCandidates = collectCandidates(phase2TP, [1], 1, zeroOrderTP)

  if (tpCandidates.length > 0) {
    reasoning.push("Adding tensor parallelism")
    const scored = scoreConfigurations(
      tpCandidates, config.microBatchSize, config.gradientAccumulationSteps
    )
    reasoning.push(`Selected ${scored[0].label}`)
    return finalize(scored[0])
  }

  reasoning.push("TP alone insufficient; adding pipeline parallelism")

  // ── Phase 3: TP + PP + DP (+ EP for MoE) ──

  const ppCandidates = collectCandidates(tpAll, ppDegrees, 1, zeroOrderTP)

  if (ppCandidates.length > 0) {
    reasoning.push("Adding pipeline parallelism")
    const scored = scoreConfigurations(
      ppCandidates, config.microBatchSize, config.gradientAccumulationSteps
    )
    reasoning.push(`Selected ${scored[0].label}`)
    return finalize(scored[0])
  }

  reasoning.push("TP + PP insufficient")

  // ── Phase 4: Context parallelism for long sequences ──

  if (config.sequenceLength > 32768) {
    const ncp = clampToPowerOf2(config.sequenceLength / 8192)
    reasoning.push(
      `Long sequence length (${config.sequenceLength}): trying context parallelism CP=${ncp}`
    )

    const cpCandidates = collectCandidates(
      tpAll, [1, ...ppDegrees], ncp, zeroOrderTP
    )

    if (cpCandidates.length > 0) {
      reasoning.push("Adding context parallelism")
      const scored = scoreConfigurations(
        cpCandidates, config.microBatchSize, config.gradientAccumulationSteps
      )
      reasoning.push(`Selected ${scored[0].label}`)
      return finalize(scored[0])
    }
  }

  // ── Fallback: nothing fits ──

  reasoning.push("No parallelism configuration fits within GPU memory")
  warnings.push({
    severity: "critical",
    category: "memory",
    message: `Model does not fit on ${numGPUs}× ${gpu.name}. Increase GPU count or reduce model/batch size.`,
  })

  const bestTP = [...tpDegrees].reverse().find((t) =>
    validateTPDivisibility(t, arch.a, arch.a_kv, resolvedDFF).valid
  ) ?? 1
  const bestPP = ppDegrees.find((p) => validatePPDivisibility(p, arch.L).valid) ?? 1
  const fallbackTopology = bestTP * bestPP
  const fallbackDP = numGPUs % fallbackTopology === 0
    ? numGPUs / fallbackTopology
    : Math.max(1, Math.floor(numGPUs / fallbackTopology))
  const fallbackStages = bestPP > 1
    ? [...getZeROStagesForPP(framework)].sort((a, b) => b - a)
    : ([3, 2, 1, 0] as ZeROStage[])

  const fallback = makeConfig({
    N_dp: fallbackDP,
    N_tp: bestTP,
    N_pp: bestPP,
    zeroStage: fallbackStages[0],
  })
  const fallbackMem = test(fallback)

  return finalize({
    config: fallback,
    score: 0,
    memory: fallbackMem,
    label: makeStrategyLabel(fallback, moeEnabled),
  })
}
