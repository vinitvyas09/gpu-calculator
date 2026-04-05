/**
 * Parallelism recommendation engine — Spec Sections 9, 5.2, 5.7
 *
 * Pure TypeScript functions. No React, no DOM.
 */
import type {
  FSDPStrategy,
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
  calculateMinGPUVRAMFloor,
  calculateTotalMemoryPerGPU,
} from "./memory"

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

interface SearchStage {
  zeroStage: ZeROStage
  VP: number
}

interface Candidate {
  config: ParallelismConfig
  memory: MemoryBreakdown
  label: string
}

interface SearchResult {
  fits: Candidate[]
  attempts: Candidate[]
}

function normalizeDegree(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 1
}

// ─── Constraint Validators ──────────────────────────────────────────────────

export function validateTPDivisibility(
  N_tp: number,
  a: number,
  a_kv: number | null,
  d_ff: number
): ValidationResult {
  if (N_tp <= 1) {
    return { valid: true, message: "No TP active" }
  }

  if (a % N_tp !== 0) {
    return {
      valid: false,
      message: `N_tp=${N_tp} does not evenly divide attention heads a=${a}`,
    }
  }

  if (a_kv !== null && a_kv % N_tp !== 0) {
    return {
      valid: false,
      message: `N_tp=${N_tp} does not evenly divide KV heads a_kv=${a_kv}`,
    }
  }

  if (d_ff % N_tp !== 0) {
    return {
      valid: false,
      message: `N_tp=${N_tp} does not evenly divide d_ff=${d_ff}`,
    }
  }

  return { valid: true, message: "TP dimensions are compatible" }
}

export function validatePPDivisibility(
  N_pp: number,
  L: number
): ValidationResult {
  if (N_pp <= 1) {
    return { valid: true, message: "No PP active" }
  }

  if (L % N_pp === 0) {
    return {
      valid: true,
      message: `L=${L} divides into ${N_pp} stages (${L / N_pp} layers each)`,
    }
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
  if (N_pp <= 1) {
    return { valid: true, message: "No PP active" }
  }

  if ((framework === "deepspeed" || framework === "hf_trainer") && zeroStage >= 2) {
    return {
      valid: false,
      message: `${framework === "hf_trainer" ? "HF Trainer" : "DeepSpeed"} ZeRO-${zeroStage} is incompatible with pipeline parallelism. Use ZeRO-0 or ZeRO-1.`,
    }
  }

  if (framework === "fsdp" && zeroStage === 3) {
    return {
      valid: false,
      message: "FSDP FULL_SHARD (ZeRO-3) is incompatible with pipeline parallelism.",
    }
  }

  return { valid: true, message: "ZeRO stage compatible with PP" }
}

export function validateWorldSize(
  config: ParallelismConfig,
  numGPUs: number
): ValidationResult {
  const world =
    normalizeDegree(config.N_dp) *
    normalizeDegree(config.N_tp) *
    normalizeDegree(config.N_pp) *
    normalizeDegree(config.N_cp) *
    normalizeDegree(config.N_ep)

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
  if (N_pp <= 1) {
    return { valid: true, message: "No PP active" }
  }

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
  if (N_tp <= 1) {
    return V
  }

  const alignment = 128 * N_tp
  return Math.ceil(V / alignment) * alignment
}

// ─── Pipeline Bubble ────────────────────────────────────────────────────────

export function calculatePipelineBubble(
  N_pp: number,
  numMicrobatches: number,
  VP: number = 1
): number {
  if (N_pp <= 1) {
    return 0
  }

  if (numMicrobatches <= 0) {
    return 1
  }

  const effectiveMicrobatches = Math.max(1, VP) * numMicrobatches
  return (N_pp - 1) / (effectiveMicrobatches + N_pp - 1)
}

// ─── Internal Helpers ───────────────────────────────────────────────────────

function resolveFFNIntermediateSize(
  arch: ModelArchitecture,
  moe: MoEConfig
): number {
  if (moe.enabled && moe.denseIntermediateSize !== null) {
    return moe.denseIntermediateSize
  }

  if (arch.d_ff !== null) {
    return arch.d_ff
  }

  const isSwiGLU =
    arch.ffnType === "swiglu" ||
    arch.ffnType === "geglu" ||
    arch.ffnType === "moe"

  return isSwiGLU ? Math.round((8 / 3) * arch.d) : 4 * arch.d
}

function isPCIeOnly(gpu: GPUSpec): boolean {
  return gpu.interconnect === "pcie" || gpu.interconnect === "none"
}

function isMoEEnabled(moe: MoEConfig): boolean {
  return moe.enabled && moe.E > 0
}

function usesAFABSchedule(
  framework: FrameworkType,
  N_pp: number,
  zeroStage: ZeROStage,
  numMicrobatches: number
): boolean {
  return framework === "fsdp" &&
    N_pp > 1 &&
    zeroStage === 2 &&
    numMicrobatches < 2 * N_pp
}

function mapZeROStageToFSDPStrategy(zeroStage: ZeROStage): FSDPStrategy | null {
  switch (zeroStage) {
    case 0:
      return "NO_SHARD"
    case 2:
      return "SHARD_GRAD_OP"
    case 3:
      return "FULL_SHARD"
    case 1:
    default:
      return null
  }
}

function applyFrameworkStage(
  parallelism: ParallelismConfig
): ParallelismConfig {
  if (parallelism.framework !== "fsdp") {
    return { ...parallelism, fsdpStrategy: null }
  }

  return {
    ...parallelism,
    fsdpStrategy: mapZeROStageToFSDPStrategy(parallelism.zeroStage),
  }
}

function applyVocabPadding(
  params: ParameterCounts,
  arch: ModelArchitecture,
  N_tp: number
): ParameterCounts {
  const paddedVocab = calculateVocabPadding(arch.V, N_tp)

  if (paddedVocab === arch.V) {
    return params
  }

  const extraEntries = paddedVocab - arch.V
  const embeddingDelta = extraEntries * arch.d
  const outputDelta = arch.tiedEmbeddings ? 0 : extraEntries * arch.d

  return {
    ...params,
    total: params.total + embeddingDelta + outputDelta,
    active: params.active + extraEntries * arch.d,
    embedding: params.embedding + embeddingDelta,
    outputProjection: params.outputProjection + outputDelta,
  }
}

function adjustForAFABIfNeeded(
  memory: MemoryBreakdown,
  config: TrainingConfig,
  parallelism: ParallelismConfig
): MemoryBreakdown {
  const N_pp = normalizeDegree(parallelism.N_pp)
  const numMicrobatches = normalizeDegree(config.gradientAccumulationSteps)

  if (!usesAFABSchedule(parallelism.framework, N_pp, parallelism.zeroStage, numMicrobatches)) {
    return memory
  }

  const currentInflight = Math.max(1, Math.min(N_pp, numMicrobatches))
  const desiredInflight = Math.max(1, numMicrobatches)

  if (desiredInflight <= currentInflight) {
    return memory
  }

  // Conservative correction: AFAB keeps all micro-batch activations resident.
  const adjustedActivations = memory.activations * (desiredInflight / currentInflight)
  const total =
    (memory.parameters +
      memory.gradients +
      memory.optimizerStates +
      adjustedActivations +
      memory.communicationBuffers +
      memory.frameworkOverhead) *
    1.04

  return {
    ...memory,
    activations: adjustedActivations,
    total,
    freeHeadroom: Math.max(0, memory.usableCapacity - total),
    fits: total <= memory.usableCapacity,
  }
}

function checkMemoryFit(
  params: ParameterCounts,
  baseConfig: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  parallelism: ParallelismConfig
): MemoryBreakdown {
  const effectiveParams = applyVocabPadding(params, arch, normalizeDegree(parallelism.N_tp))
  const memory = calculateTotalMemoryPerGPU(
    effectiveParams,
    { ...baseConfig, parallelism },
    arch,
    moe,
    gpu
  )

  return adjustForAFABIfNeeded(memory, baseConfig, parallelism)
}

function calculateMinVRAMFloorForConfig(
  params: ParameterCounts,
  config: TrainingConfig,
  arch: ModelArchitecture
): number {
  const effectiveParams = applyVocabPadding(
    params,
    arch,
    normalizeDegree(config.parallelism.N_tp)
  )

  return calculateMinGPUVRAMFloor(effectiveParams, config)
}

function makeStrategyLabel(p: ParallelismConfig, moeEnabled: boolean): string {
  const parts: string[] = [`DP=${p.N_dp}`]

  if (p.N_tp > 1) {
    parts.push(`TP=${p.N_tp}`)
  }

  if (moeEnabled && p.N_ep > 1) {
    parts.push(`EP=${p.N_ep}`)
  }

  if (p.N_pp > 1) {
    parts.push(`PP=${p.N_pp}`)
  }

  if (p.N_cp > 1) {
    parts.push(`CP=${p.N_cp}`)
  }

  parts.push(`ZeRO-${p.zeroStage}`)
  return parts.join(", ")
}

function estimateMaxMicroBatch(
  memory: MemoryBreakdown,
  currentMicroBatch: number
): number {
  if (memory.activations <= 0 || currentMicroBatch <= 0) {
    return Math.max(1, currentMicroBatch)
  }

  const activationPerSample = memory.activations / currentMicroBatch

  if (activationPerSample <= 0) {
    return currentMicroBatch
  }

  const nonActivation =
    memory.parameters +
    memory.gradients +
    memory.optimizerStates +
    memory.communicationBuffers +
    memory.frameworkOverhead
  const availableRaw = memory.usableCapacity / 1.04 - nonActivation

  if (availableRaw <= 0) {
    return 0
  }

  return Math.max(1, Math.floor(availableRaw / activationPerSample))
}

function buildParallelismConfig(
  baseConfig: TrainingConfig,
  framework: FrameworkType,
  overrides: Partial<ParallelismConfig>
): ParallelismConfig {
  return applyFrameworkStage({
    N_dp: 1,
    N_tp: 1,
    N_pp: 1,
    N_cp: 1,
    N_ep: 1,
    zeroStage: 0,
    fsdpStrategy: null,
    framework,
    sequenceParallelism: baseConfig.parallelism.sequenceParallelism,
    VP: Math.max(1, baseConfig.parallelism.VP),
    ...overrides,
  })
}

function getTPDegrees(
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  numGPUs: number
): number[] {
  const maxTP = Math.min(numGPUs, gpu.gpusPerNode, 8)
  const resolvedDFF = resolveFFNIntermediateSize(arch, moe)
  const degrees: number[] = []

  for (let N_tp = 2; N_tp <= maxTP; N_tp++) {
    if (validateTPDivisibility(N_tp, arch.a, arch.a_kv, resolvedDFF).valid) {
      degrees.push(N_tp)
    }
  }

  return degrees
}

function getPPDegrees(L: number, numGPUs: number): number[] {
  const upperBound = Math.min(numGPUs, L + 2)
  const degrees: number[] = []

  for (let N_pp = 2; N_pp <= upperBound; N_pp++) {
    if (validatePPDivisibility(N_pp, L).valid) {
      degrees.push(N_pp)
    }
  }

  return degrees
}

function getEPCandidates(E: number, N_tp: number, gpusPerNode: number): number[] {
  if (E <= 0) {
    return []
  }

  const maxEP = Math.floor(gpusPerNode / N_tp)
  const values: number[] = []

  for (let N_ep = 2; N_ep <= Math.min(E, maxEP); N_ep++) {
    if (E % N_ep === 0) {
      values.push(N_ep)
    }
  }

  return values.sort((a, b) => b - a)
}

function getCPDegrees(sequenceLength: number, numGPUs: number): number[] {
  if (sequenceLength <= 32768) {
    return []
  }

  const maxByChunk = Math.max(1, Math.floor(sequenceLength / 2048))
  const maxCP = Math.min(numGPUs, maxByChunk)
  const defaultCP = Math.max(1, sequenceLength / 8192)
  const values: number[] = []

  for (let N_cp = 2; N_cp <= maxCP; N_cp *= 2) {
    values.push(N_cp)
  }

  return values.sort((left, right) => {
    const leftDistance = Math.abs(Math.log2(left) - Math.log2(defaultCP))
    const rightDistance = Math.abs(Math.log2(right) - Math.log2(defaultCP))
    return leftDistance - rightDistance || left - right
  })
}

function getStageSearchOrder(
  framework: FrameworkType,
  N_pp: number,
  numMicrobatches: number,
  baseVP: number,
  allowStage0: boolean
): SearchStage[] {
  const safeVP = Math.max(1, baseVP)

  if (N_pp <= 1) {
    return allowStage0
      ? [
          { zeroStage: 0, VP: safeVP },
          { zeroStage: 1, VP: safeVP },
          { zeroStage: 2, VP: safeVP },
          { zeroStage: 3, VP: safeVP },
        ]
      : [
          { zeroStage: 1, VP: safeVP },
          { zeroStage: 2, VP: safeVP },
          { zeroStage: 3, VP: safeVP },
        ]
  }

  if (framework === "deepspeed" || framework === "hf_trainer") {
    return allowStage0
      ? [
          { zeroStage: 0, VP: safeVP },
          { zeroStage: 1, VP: safeVP },
        ]
      : [{ zeroStage: 1, VP: safeVP }]
  }

  if (framework === "fsdp") {
    if (numMicrobatches >= 2 * N_pp) {
      return allowStage0
        ? [
            { zeroStage: 0, VP: safeVP },
            { zeroStage: 1, VP: Math.max(2, safeVP) },
            { zeroStage: 2, VP: 1 },
          ]
        : [
            { zeroStage: 1, VP: Math.max(2, safeVP) },
            { zeroStage: 2, VP: 1 },
          ]
    }

    return allowStage0
      ? [
          { zeroStage: 0, VP: safeVP },
          { zeroStage: 2, VP: 1 },
        ]
      : [{ zeroStage: 2, VP: 1 }]
  }

  return allowStage0
    ? [
        { zeroStage: 0, VP: safeVP },
        { zeroStage: 1, VP: safeVP },
        { zeroStage: 2, VP: safeVP },
        { zeroStage: 3, VP: safeVP },
      ]
    : [
        { zeroStage: 1, VP: safeVP },
        { zeroStage: 2, VP: safeVP },
        { zeroStage: 3, VP: safeVP },
      ]
}

function getMinGPUStageSearchOrder(
  framework: FrameworkType,
  N_pp: number,
  numMicrobatches: number,
  baseVP: number
): SearchStage[] {
  const safeVP = Math.max(1, baseVP)

  if (N_pp <= 1) {
    return [
      { zeroStage: 3, VP: safeVP },
      { zeroStage: 2, VP: safeVP },
      { zeroStage: 1, VP: safeVP },
      { zeroStage: 0, VP: safeVP },
    ]
  }

  if (framework === "deepspeed" || framework === "hf_trainer") {
    return [
      { zeroStage: 1, VP: safeVP },
      { zeroStage: 0, VP: safeVP },
    ]
  }

  if (framework === "fsdp") {
    if (numMicrobatches >= 2 * N_pp) {
      return [
        { zeroStage: 2, VP: 1 },
        { zeroStage: 1, VP: Math.max(2, safeVP) },
        { zeroStage: 0, VP: safeVP },
      ]
    }

    return [
      { zeroStage: 2, VP: 1 },
      { zeroStage: 0, VP: safeVP },
    ]
  }

  return [
    { zeroStage: 3, VP: safeVP },
    { zeroStage: 2, VP: safeVP },
    { zeroStage: 1, VP: safeVP },
    { zeroStage: 0, VP: safeVP },
  ]
}

function validateScheduleForConfig(
  candidate: ParallelismConfig,
  numMicrobatches: number
): ValidationResult {
  if (candidate.N_pp <= 1) {
    return { valid: true, message: "No PP active" }
  }

  if (usesAFABSchedule(candidate.framework, candidate.N_pp, candidate.zeroStage, numMicrobatches)) {
    return {
      valid: true,
      message: "FSDP SHARD_GRAD_OP + AFAB schedule selected",
    }
  }

  return validateMicrobatches(numMicrobatches, candidate.N_pp, candidate.VP)
}

function collectCandidates(
  params: ParameterCounts,
  arch: ModelArchitecture,
  baseConfig: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig,
  options: {
    framework: FrameworkType
    allowStage0: boolean
    tpDegrees: number[]
    ppDegrees: number[]
    cpDegrees: number[]
    epDegreesForTP: (N_tp: number) => number[]
  }
): SearchResult {
  const fits: Candidate[] = []
  const attempts: Candidate[] = []

  for (const N_cp of options.cpDegrees) {
    if (N_cp > 1 && baseConfig.sequenceLength / N_cp < 2048) {
      continue
    }

    for (const N_pp of options.ppDegrees) {
      if (!validatePPDivisibility(N_pp, arch.L).valid) {
        continue
      }

      const searchStages = getStageSearchOrder(
        options.framework,
        N_pp,
        normalizeDegree(baseConfig.gradientAccumulationSteps),
        baseConfig.parallelism.VP,
        options.allowStage0
      )

      for (const N_tp of options.tpDegrees) {
        const topologyWithoutDP = N_tp * N_pp * N_cp

        if (topologyWithoutDP > numGPUs) {
          continue
        }

        const epCandidates = options.epDegreesForTP(N_tp)

        for (const N_ep of epCandidates) {
          const topology = topologyWithoutDP * N_ep

          if (topology > numGPUs || numGPUs % topology !== 0) {
            continue
          }

          const N_dp = numGPUs / topology
          let bestFitForTopology: Candidate | null = null

          for (const stage of searchStages) {
            if (!validateZeroPPCompatibility(stage.zeroStage, N_pp, options.framework).valid) {
              continue
            }

            const candidate = buildParallelismConfig(baseConfig, options.framework, {
              N_dp,
              N_tp,
              N_pp,
              N_cp,
              N_ep,
              zeroStage: stage.zeroStage,
              VP: stage.VP,
            })

            if (!validateWorldSize(candidate, numGPUs).valid) {
              continue
            }

            const scheduleValidation = validateScheduleForConfig(
              candidate,
              normalizeDegree(baseConfig.gradientAccumulationSteps)
            )

            if (!scheduleValidation.valid) {
              continue
            }

            const memory = checkMemoryFit(
              params,
              baseConfig,
              arch,
              moe,
              gpu,
              candidate
            )
            const evaluatedCandidate = {
              config: candidate,
              memory,
              label: makeStrategyLabel(candidate, isMoEEnabled(moe)),
            }

            attempts.push(evaluatedCandidate)

            if (memory.fits) {
              bestFitForTopology = evaluatedCandidate
              break
            }
          }

          if (bestFitForTopology !== null) {
            fits.push(bestFitForTopology)
          }
        }
      }
    }
  }

  return { fits, attempts }
}

function pickBestCandidate(
  candidates: Candidate[],
  currentMicroBatch: number,
  gradientAccumulationSteps: number,
  gpusPerNode: number
): ScoredConfiguration | null {
  if (candidates.length === 0) {
    return null
  }

  const lowestStage = Math.min(...candidates.map((candidate) => candidate.config.zeroStage))
  const stageFiltered = candidates.filter(
    (candidate) => candidate.config.zeroStage === lowestStage
  )
  const scored = scoreConfigurations(
    stageFiltered,
    currentMicroBatch,
    gradientAccumulationSteps,
    gpusPerNode
  )

  return scored[0] ?? null
}

function getLowestZeROStage(candidates: Candidate[]): ZeROStage | null {
  if (candidates.length === 0) {
    return null
  }

  return Math.min(...candidates.map((candidate) => candidate.config.zeroStage)) as ZeROStage
}

function pickClosestAttempt(attempts: Candidate[]): Candidate | null {
  if (attempts.length === 0) {
    return null
  }

  return [...attempts].sort((left, right) => {
    const leftOverage = Math.max(0, left.memory.total - left.memory.usableCapacity)
    const rightOverage = Math.max(0, right.memory.total - right.memory.usableCapacity)

    return leftOverage - rightOverage ||
      left.config.zeroStage - right.config.zeroStage ||
      left.memory.total - right.memory.total
  })[0]
}

function findMinimumGPUCount(
  params: ParameterCounts,
  baseConfig: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  recommended: ParallelismConfig
): number {
  const topologyBase =
    normalizeDegree(recommended.N_tp) *
    normalizeDegree(recommended.N_pp) *
    normalizeDegree(recommended.N_cp) *
    normalizeDegree(recommended.N_ep)
  const stages = getMinGPUStageSearchOrder(
    recommended.framework,
    normalizeDegree(recommended.N_pp),
    normalizeDegree(baseConfig.gradientAccumulationSteps),
    recommended.VP
  )

  for (let N_dp = 1; N_dp <= 4096; N_dp++) {
    for (const stage of stages) {
      const candidate = applyFrameworkStage({
        ...recommended,
        N_dp,
        zeroStage: stage.zeroStage,
        VP: stage.VP,
      })

      const scheduleValidation = validateScheduleForConfig(
        candidate,
        normalizeDegree(baseConfig.gradientAccumulationSteps)
      )

      if (!scheduleValidation.valid) {
        continue
      }

      const memory = checkMemoryFit(params, baseConfig, arch, moe, gpu, candidate)

      if (memory.fits) {
        return topologyBase * N_dp
      }
    }
  }

  return topologyBase * normalizeDegree(recommended.N_dp)
}

// ─── Throughput Scoring ─────────────────────────────────────────────────────

export function scoreConfigurations(
  configs: Array<{ config: ParallelismConfig; memory: MemoryBreakdown; label: string }>,
  currentMicroBatch: number,
  gradientAccumulationSteps: number,
  gpusPerNode: number = Number.POSITIVE_INFINITY
): ScoredConfiguration[] {
  return configs
    .map(({ config, memory, label }) => {
      const maxBatch = estimateMaxMicroBatch(memory, currentMicroBatch)
      const hasTP = config.N_tp > 1
      const hasPP = config.N_pp > 1
      const isZeRO3 = config.zeroStage === 3
      const isDPOnly =
        !hasTP &&
        !hasPP &&
        config.N_cp === 1 &&
        config.N_ep === 1
      const effectiveDP =
        hasTP || hasPP || config.N_cp > 1 || config.N_ep > 1
          ? Math.min(config.N_dp, Math.max(1, gpusPerNode))
          : config.N_dp

      let score: number

      if (isDPOnly && !isZeRO3) {
        score = maxBatch * config.N_dp * 1.5
      } else if (isDPOnly && isZeRO3) {
        score = maxBatch * config.N_dp
      } else if (hasTP && config.N_dp <= 1 && !hasPP && config.N_cp === 1 && config.N_ep === 1) {
        score = maxBatch
      } else {
        score = maxBatch * effectiveDP
      }

      if (hasPP) {
        score *= 1 - calculatePipelineBubble(
          config.N_pp,
          gradientAccumulationSteps,
          config.VP
        )
      }

      return { config, score, memory, label }
    })
    .sort((left, right) => {
      return right.score - left.score ||
        left.config.zeroStage - right.config.zeroStage ||
        right.config.N_ep - left.config.N_ep ||
        right.config.N_dp - left.config.N_dp ||
        left.config.N_tp - right.config.N_tp ||
        left.config.N_pp - right.config.N_pp
    })
}

// ─── Main Recommendation Engine ─────────────────────────────────────────────

export function recommendParallelism(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig
): ParallelismRecommendation {
  const framework = config.parallelism.framework
  const moeEnabled = isMoEEnabled(moe)
  const warnings: Warning[] = []
  const reasoning: string[] = []
  const attemptedCandidates: Candidate[] = []
  const deferredFits: Candidate[] = []
  const tpDegrees = getTPDegrees(arch, moe, gpu, numGPUs)
  const ppDegrees = getPPDegrees(arch.L, numGPUs)
  const cpDegrees = getCPDegrees(config.sequenceLength, numGPUs)
  const pcieOnly = isPCIeOnly(gpu)
  const wholeModelFitsSingleGPU = checkMemoryFit(
    params,
    config,
    arch,
    moe,
    gpu,
    buildParallelismConfig(config, framework, {
      N_dp: 1,
      N_tp: 1,
      N_pp: 1,
      N_cp: 1,
      N_ep: 1,
      zeroStage: 0,
      VP: 1,
    })
  ).fits

  if (!validateHiddenDimAlignment(arch.d).valid) {
    warnings.push({
      severity: "warning",
      category: "parallelism",
      message: `Hidden dimension d=${arch.d} is not aligned to 128, causing significant throughput loss.`,
    })
  }

  function finalize(candidate: ScoredConfiguration | Candidate): ParallelismRecommendation {
    const selected = "score" in candidate ? candidate : {
      ...candidate,
      score: 0,
    }
    const parallelism = selected.config
    const finalConfig: TrainingConfig = {
      ...config,
      parallelism,
    }
    const minVRAMFloor = calculateMinVRAMFloorForConfig(params, finalConfig, arch)
    const pipelineBubbleFraction = usesAFABSchedule(
      parallelism.framework,
      parallelism.N_pp,
      parallelism.zeroStage,
      normalizeDegree(config.gradientAccumulationSteps)
    )
      ? calculatePipelineBubble(parallelism.N_pp, normalizeDegree(config.gradientAccumulationSteps))
      : calculatePipelineBubble(
          parallelism.N_pp,
          normalizeDegree(config.gradientAccumulationSteps),
          parallelism.VP
        )
    const minGPUs = findMinimumGPUCount(
      params,
      config,
      arch,
      moe,
      gpu,
      parallelism
    )
    const paddedVocab = calculateVocabPadding(arch.V, parallelism.N_tp)

    if (parallelism.N_pp > 1) {
      const scheduleValidation = validateScheduleForConfig(
        parallelism,
        normalizeDegree(config.gradientAccumulationSteps)
      )

      if (!scheduleValidation.valid) {
        warnings.push({
          severity: "warning",
          category: "parallelism",
          message: scheduleValidation.message,
        })
      }

      if (pipelineBubbleFraction > 0.5) {
        warnings.push({
          severity: "warning",
          category: "parallelism",
          message: `Pipeline bubble is ${(pipelineBubbleFraction * 100).toFixed(1)}%. Increase gradient accumulation steps to reduce idle time.`,
        })
      } else if (pipelineBubbleFraction > 0.2) {
        warnings.push({
          severity: "info",
          category: "parallelism",
          message: `Pipeline bubble is ${(pipelineBubbleFraction * 100).toFixed(1)}%. A rule of thumb is gradient accumulation ≥ ${4 * parallelism.N_pp}.`,
        })
      }
    }

    if (parallelism.zeroStage === 3) {
      warnings.push({
        severity: "info",
        category: "parallelism",
        message: "ZeRO-3 has the highest communication overhead; prefer lower stages when they fit.",
      })
    }

    if (parallelism.N_tp > 1 && pcieOnly) {
      warnings.push({
        severity: "warning",
        category: "parallelism",
        message: `TP=${parallelism.N_tp} on PCIe-only GPUs can be severely bandwidth-limited versus NVLink.`,
      })
    }

    if (parallelism.N_cp > 1 && pcieOnly) {
      warnings.push({
        severity: "warning",
        category: "parallelism",
        message: `CP=${parallelism.N_cp} adds high-bandwidth sequence communication; PCIe-only clusters may see poor scaling.`,
      })
    }

    if (parallelism.N_tp > 1 && paddedVocab > arch.V) {
      warnings.push({
        severity: "info",
        category: "parallelism",
        message: `Vocabulary padded from ${arch.V.toLocaleString()} to ${paddedVocab.toLocaleString()} for TP=${parallelism.N_tp} alignment.`,
      })
    }

    if (params.total % parallelism.N_dp !== 0) {
      warnings.push({
        severity: "info",
        category: "parallelism",
        message: `Parameter count is not evenly divisible by N_dp=${parallelism.N_dp}; some frameworks will pad shards automatically.`,
      })
    }

    if (minVRAMFloor > gpu.memoryGB * 1e9 * 0.8) {
      warnings.push({
        severity: "warning",
        category: "memory",
        message: "The largest-layer working set is close to the GPU's usable VRAM floor even with full sharding.",
      })
    }

    if (
      usesAFABSchedule(
        parallelism.framework,
        parallelism.N_pp,
        parallelism.zeroStage,
        normalizeDegree(config.gradientAccumulationSteps)
      )
    ) {
      warnings.push({
        severity: "info",
        category: "parallelism",
        message: "FSDP SHARD_GRAD_OP with PP uses the AFAB schedule here; activation memory is conservatively adjusted above the 1F1B estimate.",
      })
    }

    return {
      config: parallelism,
      minGPUs,
      minVRAMFloor,
      pipelineBubbleFraction,
      strategyLabel: makeStrategyLabel(parallelism, moeEnabled),
      reasoning,
      warnings,
    }
  }

  const singleDeviceCandidate = buildParallelismConfig(config, framework, {
    N_dp: 1,
  })

  if (gpu.singleDeviceOnly || numGPUs === 1) {
    const memory = checkMemoryFit(
      params,
      config,
      arch,
      moe,
      gpu,
      singleDeviceCandidate
    )

    if (!memory.fits) {
      warnings.push({
        severity: "critical",
        category: "memory",
        message: "Model does not fit on a single GPU. Multi-GPU training is required.",
      })
    }

    reasoning.push(
      gpu.singleDeviceOnly
        ? `${gpu.name} supports single-device training only`
        : "Single GPU available, so no model parallelism can be introduced."
    )

    return finalize({
      config: singleDeviceCandidate,
      memory,
      label: makeStrategyLabel(singleDeviceCandidate, moeEnabled),
    })
  }

  // Phase 1: DP only, escalating ZeRO stages within pure data parallelism.
  const dpSearch = collectCandidates(
    params,
    arch,
    config,
    gpu,
    numGPUs,
    moe,
    {
      framework,
      allowStage0: wholeModelFitsSingleGPU,
      tpDegrees: [1],
      ppDegrees: [1],
      cpDegrees: [1],
      epDegreesForTP: () => [1],
    }
  )

  attemptedCandidates.push(...dpSearch.attempts)

  if (dpSearch.fits.length > 0) {
      const bestDP = pickBestCandidate(
        dpSearch.fits,
        config.microBatchSize,
        normalizeDegree(config.gradientAccumulationSteps),
        gpu.gpusPerNode
      )
    const lowestDPStage = getLowestZeROStage(dpSearch.fits)

    if (bestDP !== null && (pcieOnly || (lowestDPStage !== null && lowestDPStage <= 1))) {
      reasoning.push("Model fits with pure data parallelism, so TP/PP/CP are unnecessary.")
      reasoning.push(`Selected ${bestDP.label} after preferring the lowest ZeRO stage that fit.`)
      return finalize(bestDP)
    }

    deferredFits.push(...dpSearch.fits)
    reasoning.push(
      "Pure data parallelism only fit with ZeRO-2/3, so the engine keeps searching for a lower-overhead TP/PP strategy."
    )
  }

  if (dpSearch.fits.length === 0) {
    reasoning.push("Pure data parallelism did not fit in GPU memory, so additional model sharding is required.")
  }

  // Phase 2: TP only (or TP + DP), still without EP/PP/CP.
  if (tpDegrees.length > 0) {
    if (pcieOnly) {
      reasoning.push("PCIe-only interconnect: TP is only considered after DP + ZeRO was exhausted.")
    } else {
      reasoning.push("Exploring tensor parallelism within a node before adding pipeline stages.")
    }

    const tpSearch = collectCandidates(
      params,
      arch,
      config,
      gpu,
      numGPUs,
      moe,
      {
        framework,
        allowStage0: wholeModelFitsSingleGPU,
        tpDegrees,
        ppDegrees: [1],
        cpDegrees: [1],
        epDegreesForTP: () => [1],
      }
    )

    attemptedCandidates.push(...tpSearch.attempts)

    if (tpSearch.fits.length > 0) {
      deferredFits.push(...tpSearch.fits)

      if ((getLowestZeROStage(tpSearch.fits) ?? 3) <= 1) {
        reasoning.push(
          "A low-stage TP strategy fits, but PP is still evaluated because it may trade memory for higher throughput."
        )
      } else {
        reasoning.push(
          "TP-only still required ZeRO-2/3, so the search continues to expert or pipeline parallelism."
        )
      }
    }
  }

  // Phase 3: MoE expert parallelism after TP.
  if (moeEnabled && tpDegrees.length > 0) {
    reasoning.push("MoE model still does not fit after TP alone, so expert parallelism is being considered next.")

    const epSearch = collectCandidates(
      params,
      arch,
      config,
      gpu,
      numGPUs,
      moe,
      {
        framework,
        allowStage0: wholeModelFitsSingleGPU,
        tpDegrees,
        ppDegrees: [1],
        cpDegrees: [1],
        epDegreesForTP: (N_tp) => getEPCandidates(moe.E, N_tp, gpu.gpusPerNode),
      }
    )

    attemptedCandidates.push(...epSearch.attempts)

    if (epSearch.fits.length > 0) {
      deferredFits.push(...epSearch.fits)

      if ((getLowestZeROStage(epSearch.fits) ?? 3) <= 1) {
        reasoning.push(
          "EP produces low-stage MoE candidates, but PP is still checked before the final choice."
        )
      } else {
        reasoning.push(
          "EP reduced expert memory, but the fitting options still need ZeRO-2/3, so pipeline parallelism is explored next."
        )
      }
    }
  }

  reasoning.push("Intra-layer sharding was insufficient, so the search moves to pipeline parallelism.")

  // Phase 4: PP, optionally combined with TP and EP.
  if (ppDegrees.length > 0) {
    const ppSearch = collectCandidates(
      params,
      arch,
      config,
      gpu,
      numGPUs,
      moe,
      {
        framework,
        allowStage0: wholeModelFitsSingleGPU,
        tpDegrees: tpDegrees.length > 0 ? tpDegrees : [1],
        ppDegrees,
        cpDegrees: [1],
        epDegreesForTP: (N_tp) =>
          moeEnabled ? [1, ...getEPCandidates(moe.E, N_tp, gpu.gpusPerNode)] : [1],
      }
    )

    attemptedCandidates.push(...ppSearch.attempts)

    if (ppSearch.fits.length > 0) {
      deferredFits.push(...ppSearch.fits)

      const bestPP = pickBestCandidate(
        deferredFits,
        config.microBatchSize,
        normalizeDegree(config.gradientAccumulationSteps),
        gpu.gpusPerNode
      )

      if (bestPP !== null) {
        reasoning.push(`Selected ${bestPP.label}.`)
        return finalize(bestPP)
      }
    }
  }

  reasoning.push("TP/EP/PP still did not fit within VRAM.")

  // Phase 5: CP for long-context runs only.
  if (cpDegrees.length > 0) {
    reasoning.push(
      `Sequence length ${config.sequenceLength.toLocaleString()} is long enough to justify context parallelism.`
    )

    const cpSearch = collectCandidates(
      params,
      arch,
      config,
      gpu,
      numGPUs,
      moe,
      {
        framework,
        allowStage0: wholeModelFitsSingleGPU,
        tpDegrees: tpDegrees.length > 0 ? tpDegrees : [1],
        ppDegrees: [1, ...ppDegrees],
        cpDegrees,
        epDegreesForTP: (N_tp) =>
          moeEnabled ? [1, ...getEPCandidates(moe.E, N_tp, gpu.gpusPerNode)] : [1],
      }
    )

    attemptedCandidates.push(...cpSearch.attempts)

    if (cpSearch.fits.length > 0) {
      deferredFits.push(...cpSearch.fits)

      const bestCP = pickBestCandidate(
        deferredFits,
        config.microBatchSize,
        normalizeDegree(config.gradientAccumulationSteps),
        gpu.gpusPerNode
      )

      if (bestCP !== null) {
        reasoning.push(`Selected ${bestCP.label}.`)
        return finalize(bestCP)
      }
    }
  }

  if (deferredFits.length > 0) {
    const bestDeferred = pickBestCandidate(
      deferredFits,
      config.microBatchSize,
      normalizeDegree(config.gradientAccumulationSteps),
      gpu.gpusPerNode
    )

    if (bestDeferred !== null) {
      reasoning.push("Returning the closest fitting strategy found before the final VRAM failure path.")
      return finalize(bestDeferred)
    }
  }

  reasoning.push("No valid configuration fit within the available VRAM.")
  warnings.push({
    severity: "critical",
    category: "memory",
    message: `Model does not fit on ${numGPUs}× ${gpu.name}. Increase GPU count or reduce the model, batch size, or sequence length.`,
  })

  const closestAttempt = pickClosestAttempt(attemptedCandidates)

  if (closestAttempt !== null) {
    return finalize(closestAttempt)
  }

  return finalize({
    config: singleDeviceCandidate,
    memory: checkMemoryFit(params, config, arch, moe, gpu, singleDeviceCandidate),
    label: makeStrategyLabel(singleDeviceCandidate, moeEnabled),
  })
}
