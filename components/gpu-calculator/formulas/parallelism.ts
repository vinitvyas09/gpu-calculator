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

interface SearchOutcome {
  recommended: Candidate | ScoredConfiguration | null
  closestAttempt: Candidate | null
  reasoning: string[]
}

interface CollectionOptions {
  framework: FrameworkType
  tpDegrees: number[]
  ppDegrees: number[]
  cpDegrees: number[]
  epDegreesForTP: (N_tp: number) => number[]
  stageSearch: (N_pp: number) => SearchStage[]
  requireModelParallel?: boolean
}

function normalizeDegree(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 1
}

function resolveFFNIntermediateSize(
  arch: ModelArchitecture,
  moe: MoEConfig
): number {
  if (moe.enabled && moe.denseIntermediateSize !== null) {
    return moe.denseIntermediateSize
  }

  return arch.d_ff ?? 4 * arch.d
}

function isMoEEnabled(moe: MoEConfig): boolean {
  return moe.enabled && moe.E > 0
}

function isPCIeOnly(gpu: GPUSpec): boolean {
  return gpu.interconnect === "pcie" || gpu.interconnect === "none"
}

function maxTensorParallelDegree(gpu: GPUSpec, numGPUs: number): number {
  if (gpu.id === "rtx-3090") {
    return Math.min(numGPUs, 2)
  }

  return Math.min(numGPUs, gpu.gpusPerNode, 8)
}

function usesAFABSchedule(
  framework: FrameworkType,
  N_pp: number,
  zeroStage: ZeROStage,
  numMicrobatches: number
): boolean {
  return (
    framework === "fsdp" &&
    N_pp > 1 &&
    zeroStage === 2 &&
    numMicrobatches < 2 * N_pp
  )
}

function mapZeROStageToFSDPStrategy(
  zeroStage: ZeROStage
): FSDPStrategy | null {
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
    VP: 1,
    ...overrides,
  })
}

function makeStrategyLabel(
  parallelism: ParallelismConfig,
  moeEnabled: boolean
): string {
  const parts: string[] = [`DP=${parallelism.N_dp}`]

  if (parallelism.N_tp > 1) {
    parts.push(`TP=${parallelism.N_tp}`)
  }

  if (moeEnabled && parallelism.N_ep > 1) {
    parts.push(`EP=${parallelism.N_ep}`)
  }

  if (parallelism.N_pp > 1) {
    parts.push(`PP=${parallelism.N_pp}`)
  }

  if (parallelism.N_cp > 1) {
    parts.push(`CP=${parallelism.N_cp}`)
  }

  if (parallelism.N_pp > 1 && parallelism.VP > 1) {
    parts.push(`VP=${parallelism.VP}`)
  }

  parts.push(`ZeRO-${parallelism.zeroStage}`)
  return parts.join(", ")
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
      message: `Embedding-aware partitioning enabled: (L+2)=${L + 2} divides into ${N_pp} stages`,
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

  if (framework === "fsdp") {
    if (zeroStage === 3) {
      return {
        valid: false,
        message: "FSDP FULL_SHARD (ZeRO-3) is incompatible with PP.",
      }
    }

    return { valid: true, message: "ZeRO stage compatible with PP" }
  }

  // DeepSpeed, HF Trainer, Megatron: only ZeRO-0/1 with PP
  if (zeroStage >= 2) {
    const frameworkLabel =
      framework === "hf_trainer"
        ? "HF Trainer"
        : framework === "megatron"
          ? "Megatron-LM"
          : "DeepSpeed"

    return {
      valid: false,
      message: `${frameworkLabel} ZeRO-${zeroStage} is incompatible with PP. Use ZeRO-0 or ZeRO-1.`,
    }
  }

  return { valid: true, message: "ZeRO stage compatible with PP" }
}

export function validateWorldSize(
  config: ParallelismConfig,
  numGPUs?: number
): ValidationResult {
  const world =
    normalizeDegree(config.N_dp) *
    normalizeDegree(config.N_tp) *
    normalizeDegree(config.N_pp) *
    normalizeDegree(config.N_cp) *
    normalizeDegree(config.N_ep)

  if (numGPUs === undefined) {
    return {
      valid: Number.isInteger(world) && world >= 1,
      message: `World size = ${world}`,
    }
  }

  if (world !== numGPUs) {
    return {
      valid: false,
      message: `World size ${world} ≠ ${numGPUs} GPUs`,
    }
  }

  return {
    valid: true,
    message: `World size ${world} = ${numGPUs} GPUs`,
  }
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
      message: `Hidden dimension d=${d} is not aligned to 128`,
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

  const effectiveVP = Math.max(1, normalizeDegree(VP))
  return (N_pp - 1) / (effectiveVP * numMicrobatches + N_pp - 1)
}

// ─── Memory Helpers ────────────────────────────────────────────────────────

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

function checkMemoryFit(
  params: ParameterCounts,
  baseConfig: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  parallelism: ParallelismConfig
): MemoryBreakdown {
  const effectiveParams = applyVocabPadding(
    params,
    arch,
    normalizeDegree(parallelism.N_tp)
  )

  return calculateTotalMemoryPerGPU(
    effectiveParams,
    { ...baseConfig, parallelism },
    arch,
    moe,
    gpu
  )
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

// ─── Search Helpers ────────────────────────────────────────────────────────

function resolveExpertIntermediateSize(
  arch: ModelArchitecture,
  moe: MoEConfig
): number | null {
  if (!moe.enabled || moe.E <= 0 || moe.L_moe <= 0) {
    return null
  }

  return moe.expertIntermediateSize ?? resolveFFNIntermediateSize(arch, moe)
}

function getTPDegrees(
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  numGPUs: number
): number[] {
  const maxTP = maxTensorParallelDegree(gpu, numGPUs)
  const dFF = resolveFFNIntermediateSize(arch, moe)
  const expertDFF = resolveExpertIntermediateSize(arch, moe)
  const preferredDegrees = [2, 4, 8]

  return preferredDegrees.filter((N_tp) => {
    if (N_tp > maxTP) {
      return false
    }

    if (!validateTPDivisibility(N_tp, arch.a, arch.a_kv, dFF).valid) {
      return false
    }

    if (expertDFF !== null && expertDFF % N_tp !== 0) {
      return false
    }

    return true
  })
}

function getPPDegrees(L: number, numGPUs: number): number[] {
  const upperBound = Math.min(numGPUs, L + 2)
  const values: number[] = []

  for (let N_pp = 2; N_pp <= upperBound; N_pp++) {
    if (validatePPDivisibility(N_pp, L).valid) {
      values.push(N_pp)
    }
  }

  return values
}

function getEPCandidates(
  totalExperts: number,
  N_tp: number,
  gpusPerNode: number
): number[] {
  if (totalExperts <= 0) {
    return []
  }

  const maxEP = Math.floor(gpusPerNode / Math.max(1, N_tp))
  const values: number[] = []

  for (let N_ep = 2; N_ep <= Math.min(totalExperts, maxEP); N_ep++) {
    if (totalExperts % N_ep === 0) {
      values.push(N_ep)
    }
  }

  return values.sort((left, right) => right - left)
}

function nearestPowerOfTwo(value: number): number {
  if (value <= 1) {
    return 1
  }

  const lower = 2 ** Math.floor(Math.log2(value))
  const upper = 2 ** Math.ceil(Math.log2(value))
  return value - lower <= upper - value ? lower : upper
}

function getCPDegrees(sequenceLength: number, numGPUs: number): number[] {
  if (sequenceLength < 32768) {
    return []
  }

  const rawDefault = sequenceLength / 8192
  const preferred = Math.max(2, nearestPowerOfTwo(rawDefault))
  const maxCP = Math.min(
    numGPUs,
    Math.max(2, 2 ** Math.floor(Math.log2(Math.max(2, sequenceLength / 2048))))
  )
  const values: number[] = []

  for (let N_cp = 2; N_cp <= maxCP; N_cp *= 2) {
    values.push(N_cp)
  }

  return values.sort((left, right) => {
    const leftDistance = Math.abs(Math.log2(left) - Math.log2(preferred))
    const rightDistance = Math.abs(Math.log2(right) - Math.log2(preferred))
    return leftDistance - rightDistance || left - right
  })
}

function getNoPPStageSearchOrder(): SearchStage[] {
  return [
    { zeroStage: 1, VP: 1 },
    { zeroStage: 2, VP: 1 },
    { zeroStage: 3, VP: 1 },
  ]
}

function getPPStageSearchOrder(
  framework: FrameworkType,
  N_pp: number,
  numMicrobatches: number,
  baseVP: number
): SearchStage[] {
  const safeVP = Math.max(1, normalizeDegree(baseVP))

  if (framework === "fsdp") {
    if (numMicrobatches < 2 * N_pp) {
      return [{ zeroStage: 2, VP: 1 }]
    }

    return [{ zeroStage: 1, VP: Math.max(2, safeVP) }]
  }

  // DeepSpeed, HF Trainer, and Megatron: PP is only compatible with ZeRO-0/1.
  // The spec's compatibility table (Section 9) marks ZeRO-2/3 + PP as incompatible.
  return [{ zeroStage: 1, VP: safeVP }]
}

function validateScheduleForConfig(
  candidate: ParallelismConfig,
  numMicrobatches: number
): ValidationResult {
  if (candidate.N_pp <= 1) {
    return { valid: true, message: "No PP active" }
  }

  if (
    usesAFABSchedule(
      candidate.framework,
      candidate.N_pp,
      candidate.zeroStage,
      numMicrobatches
    )
  ) {
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
  options: CollectionOptions
): SearchResult {
  const fits: Candidate[] = []
  const attempts: Candidate[] = []
  const numMicrobatches = normalizeDegree(baseConfig.gradientAccumulationSteps)
  const moeEnabled = isMoEEnabled(moe)

  for (const N_cp of options.cpDegrees) {
    for (const N_pp of options.ppDegrees) {
      if (!validatePPDivisibility(N_pp, arch.L).valid) {
        continue
      }

      const searchStages = options.stageSearch(N_pp)

      for (const N_tp of options.tpDegrees) {
        const epCandidates = options.epDegreesForTP(N_tp)

        for (const N_ep of epCandidates) {
          const topology = N_tp * N_pp * N_cp * N_ep

          if (topology > numGPUs || numGPUs % topology !== 0) {
            continue
          }

          if (options.requireModelParallel && topology === 1) {
            continue
          }

          const N_dp = numGPUs / topology
          let bestFitForTopology: Candidate | null = null

          for (const stage of searchStages) {
            if (
              !validateZeroPPCompatibility(
                stage.zeroStage,
                N_pp,
                options.framework
              ).valid
            ) {
              continue
            }

            if (
              moeEnabled &&
              N_ep > 1 &&
              N_tp * N_ep > gpu.gpusPerNode
            ) {
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
              numMicrobatches
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
            const evaluatedCandidate: Candidate = {
              config: candidate,
              memory,
              label: makeStrategyLabel(candidate, moeEnabled),
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

function estimateMaxMicroBatch(
  memory: MemoryBreakdown,
  currentMicroBatch: number
): number {
  if (memory.activations <= 0 || currentMicroBatch <= 0) {
    return Math.max(1, currentMicroBatch)
  }

  const activationPerSample = memory.activations / currentMicroBatch

  if (activationPerSample <= 0) {
    return Math.max(1, currentMicroBatch)
  }

  const nonActivationTotal =
    memory.parameters +
    memory.gradients +
    memory.optimizerStates +
    memory.communicationBuffers +
    memory.frameworkOverhead
  const availableRaw = memory.usableCapacity / 1.04 - nonActivationTotal

  if (availableRaw <= 0) {
    return 0
  }

  return Math.max(1, Math.floor(availableRaw / activationPerSample))
}

function lowestZeROStage(candidates: Candidate[]): ZeROStage | null {
  if (candidates.length === 0) {
    return null
  }

  return Math.min(
    ...candidates.map((candidate) => candidate.config.zeroStage)
  ) as ZeROStage
}

function pickClosestAttempt(attempts: Candidate[]): Candidate | null {
  if (attempts.length === 0) {
    return null
  }

  return [...attempts].sort((left, right) => {
    const leftOverage = Math.max(0, left.memory.total - left.memory.usableCapacity)
    const rightOverage = Math.max(0, right.memory.total - right.memory.usableCapacity)

    return (
      leftOverage - rightOverage ||
      left.config.zeroStage - right.config.zeroStage ||
      left.memory.total - right.memory.total
    )
  })[0]
}

function filterToLowestStage(candidates: Candidate[]): Candidate[] {
  const lowestStage = lowestZeROStage(candidates)

  if (lowestStage === null) {
    return []
  }

  return candidates.filter((candidate) => candidate.config.zeroStage === lowestStage)
}

function pickBestFeasibleCandidate(
  candidates: Candidate[],
  currentMicroBatch: number,
  gradientAccumulationSteps: number
): ScoredConfiguration | null {
  if (candidates.length === 0) {
    return null
  }

  return (
    scoreConfigurations(
      candidates,
      currentMicroBatch,
      gradientAccumulationSteps
    )[0] ?? null
  )
}

function searchRecommendation(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig
): SearchOutcome {
  const framework = config.parallelism.framework
  const reasoning: string[] = []
  const attemptedCandidates: Candidate[] = []
  const fallbackFits: Candidate[] = []
  const moeEnabled = isMoEEnabled(moe)
  const pcieOnly = isPCIeOnly(gpu)
  const tpDegrees = getTPDegrees(arch, moe, gpu, numGPUs)
  const ppDegrees = getPPDegrees(arch.L, numGPUs)
  const cpDegrees = getCPDegrees(config.sequenceLength, numGPUs)
  const singleDeviceCandidate = buildParallelismConfig(config, framework, {
    N_dp: Math.max(1, numGPUs),
    N_tp: 1,
    N_pp: 1,
    N_cp: 1,
    N_ep: 1,
    zeroStage: 0,
    VP: 1,
  })

  const singleDeviceMemory = checkMemoryFit(
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
  )

  if (gpu.singleDeviceOnly || numGPUs === 1) {
    const singleGPUConfig = buildParallelismConfig(config, framework, {
      N_dp: 1,
      N_tp: 1,
      N_pp: 1,
      N_cp: 1,
      N_ep: 1,
      zeroStage: 0,
      VP: 1,
    })
    const singleGPUCandidate = {
      config: singleGPUConfig,
      memory: singleDeviceMemory,
      label: makeStrategyLabel(singleGPUConfig, moeEnabled),
    }

    reasoning.push(
      gpu.singleDeviceOnly
        ? `${gpu.name} supports single-device training only.`
        : "Only one GPU is available, so no model parallelism can be introduced."
    )

    return {
      recommended: singleDeviceMemory.fits ? singleGPUCandidate : null,
      closestAttempt: singleGPUCandidate,
      reasoning,
    }
  }

  if (singleDeviceMemory.fits) {
    reasoning.push("The model fits on one GPU with room for activations, so pure data parallelism is sufficient.")
    return {
      recommended: {
        config: singleDeviceCandidate,
        memory: checkMemoryFit(
          params,
          config,
          arch,
          moe,
          gpu,
          singleDeviceCandidate
        ),
        label: makeStrategyLabel(singleDeviceCandidate, moeEnabled),
      },
      closestAttempt: null,
      reasoning,
    }
  }

  const dpSearch = collectCandidates(
    params,
    arch,
    config,
    gpu,
    numGPUs,
    moe,
    {
      framework,
      tpDegrees: [1],
      ppDegrees: [1],
      cpDegrees: [1],
      epDegreesForTP: () => [1],
      stageSearch: () => getNoPPStageSearchOrder(),
    }
  )

  attemptedCandidates.push(...dpSearch.attempts)

  if (dpSearch.fits.length > 0) {
    const bestDP = pickBestFeasibleCandidate(
      dpSearch.fits,
      config.microBatchSize,
      normalizeDegree(config.gradientAccumulationSteps)
    )
    const bestStage = lowestZeROStage(dpSearch.fits)

    if (bestDP !== null && bestStage !== null && (pcieOnly || bestStage <= 1)) {
      reasoning.push(
        pcieOnly
          ? "Pure data parallelism fits, and PCIe-only GPUs should prefer ZeRO over TP."
          : `Pure data parallelism fits with ZeRO-${bestStage}, so lower-overhead model sharding is unnecessary.`
      )

      return {
        recommended: bestDP,
        closestAttempt: null,
        reasoning,
      }
    }

    fallbackFits.push(...dpSearch.fits)
    reasoning.push(
      `Pure data parallelism only fit with ZeRO-${bestStage}; searching for TP/EP/PP combinations that may recover throughput.`
    )
  } else {
    reasoning.push("Pure data parallelism does not fit in GPU memory, so model sharding is required.")
  }

  const denseModelParallelSearch = collectCandidates(
    params,
    arch,
    config,
    gpu,
    numGPUs,
    moe,
    {
      framework,
      tpDegrees: moeEnabled ? [1, ...tpDegrees] : tpDegrees,
      ppDegrees: [1],
      cpDegrees: [1],
      epDegreesForTP: (N_tp) =>
        moeEnabled ? [1, ...getEPCandidates(moe.E, N_tp, gpu.gpusPerNode)] : [1],
      stageSearch: () => getNoPPStageSearchOrder(),
      requireModelParallel: true,
    }
  )

  attemptedCandidates.push(...denseModelParallelSearch.attempts)

  if (denseModelParallelSearch.fits.length > 0) {
    const lowStageFits = denseModelParallelSearch.fits.filter(
      (candidate) => candidate.config.zeroStage <= 1
    )

    if (lowStageFits.length > 0) {
      const bestLowStage = pickBestFeasibleCandidate(
        lowStageFits,
        config.microBatchSize,
        normalizeDegree(config.gradientAccumulationSteps)
      )

      if (bestLowStage !== null) {
        reasoning.push(
          moeEnabled
            ? "A low-stage TP/EP configuration fits, so PP is unnecessary."
            : "A low-stage TP configuration fits, so PP is unnecessary."
        )

        return {
          recommended: bestLowStage,
          closestAttempt: null,
          reasoning,
        }
      }
    }

    fallbackFits.push(...denseModelParallelSearch.fits)
    reasoning.push(
      moeEnabled
        ? "TP/EP reduce memory pressure, but the fitting options still require ZeRO-2/3, so pipeline parallelism is explored next."
        : "TP reduces memory pressure, but the fitting options still require ZeRO-2/3, so pipeline parallelism is explored next."
    )
  } else if (tpDegrees.length > 0 || (moeEnabled && moe.E > 1)) {
    reasoning.push(
      moeEnabled
        ? "TP/EP without PP still do not fit the model in memory."
        : "TP without PP still does not fit the model in memory."
    )
  }

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
        tpDegrees: [1, ...tpDegrees],
        ppDegrees,
        cpDegrees: [1],
        epDegreesForTP: (N_tp) =>
          moeEnabled ? [1, ...getEPCandidates(moe.E, N_tp, gpu.gpusPerNode)] : [1],
        stageSearch: (N_pp) =>
          getPPStageSearchOrder(
            framework,
            N_pp,
            normalizeDegree(config.gradientAccumulationSteps),
            config.parallelism.VP
          ),
      }
    )

    attemptedCandidates.push(...ppSearch.attempts)

    if (ppSearch.fits.length > 0) {
      const lowestPPStage = lowestZeROStage(ppSearch.fits)
      const lowestFallbackStage = lowestZeROStage(fallbackFits)
      const shouldPreferPPStage =
        lowestPPStage !== null &&
        (lowestFallbackStage === null || lowestPPStage < lowestFallbackStage)
      const scoringPool = shouldPreferPPStage
        ? filterToLowestStage(ppSearch.fits)
        : lowestPPStage !== null &&
            lowestFallbackStage !== null &&
            lowestPPStage === lowestFallbackStage
          ? [
              ...filterToLowestStage(fallbackFits),
              ...filterToLowestStage(ppSearch.fits),
            ]
          : fallbackFits.length > 0
            ? fallbackFits
            : ppSearch.fits
      const bestPP = pickBestFeasibleCandidate(
        scoringPool,
        config.microBatchSize,
        normalizeDegree(config.gradientAccumulationSteps)
      )

      if (bestPP !== null) {
        reasoning.push(
          shouldPreferPPStage
            ? "Adding PP unlocked a lower ZeRO stage, so the best low-stage PP configuration was selected."
            : "Selected the best feasible configuration after evaluating pipeline parallelism."
        )
        return {
          recommended: bestPP,
          closestAttempt: null,
          reasoning,
        }
      }
    }
  }

  if (fallbackFits.length > 0) {
    const bestFallback = pickBestFeasibleCandidate(
      fallbackFits,
      config.microBatchSize,
      normalizeDegree(config.gradientAccumulationSteps)
    )

    if (bestFallback !== null) {
      reasoning.push("No PP strategy improved on the earlier feasible candidates, so the best non-PP fallback is returned.")
      return {
        recommended: bestFallback,
        closestAttempt: null,
        reasoning,
      }
    }
  }

  if (cpDegrees.length > 0) {
    const cpSearch = collectCandidates(
      params,
      arch,
      config,
      gpu,
      numGPUs,
      moe,
      {
        framework,
        tpDegrees: [1, ...tpDegrees],
        ppDegrees: [1, ...ppDegrees],
        cpDegrees,
        epDegreesForTP: (N_tp) =>
          moeEnabled ? [1, ...getEPCandidates(moe.E, N_tp, gpu.gpusPerNode)] : [1],
        stageSearch: (N_pp) =>
          N_pp > 1
            ? getPPStageSearchOrder(
                framework,
                N_pp,
                normalizeDegree(config.gradientAccumulationSteps),
                config.parallelism.VP
              )
            : getNoPPStageSearchOrder(),
      }
    )

    attemptedCandidates.push(...cpSearch.attempts)

    if (cpSearch.fits.length > 0) {
      const bestCP = pickBestFeasibleCandidate(
        cpSearch.fits,
        config.microBatchSize,
        normalizeDegree(config.gradientAccumulationSteps)
      )

      if (bestCP !== null) {
        reasoning.push(
          `Sequence length ${config.sequenceLength.toLocaleString()} is long enough to justify context parallelism.`
        )
        return {
          recommended: bestCP,
          closestAttempt: null,
          reasoning,
        }
      }
    }
  }

  reasoning.push("No feasible configuration fits within the available VRAM.")
  return {
    recommended: null,
    closestAttempt: pickClosestAttempt(attemptedCandidates),
    reasoning,
  }
}

function hasFeasibleRecommendation(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig
): boolean {
  return (
    searchRecommendation(params, arch, config, gpu, numGPUs, moe).recommended !==
    null
  )
}

function findMinimumGPUCount(
  params: ParameterCounts,
  config: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  currentNumGPUs: number
): number {
  if (gpu.singleDeviceOnly) {
    return 1
  }

  const cap = 4096
  let upperBound = Math.max(1, currentNumGPUs)

  while (
    upperBound <= cap &&
    !hasFeasibleRecommendation(params, arch, config, gpu, upperBound, moe)
  ) {
    upperBound *= 2
  }

  upperBound = Math.min(upperBound, cap)

  for (let candidateGPUs = 1; candidateGPUs <= upperBound; candidateGPUs++) {
    if (hasFeasibleRecommendation(params, arch, config, gpu, candidateGPUs, moe)) {
      return candidateGPUs
    }
  }

  return upperBound
}

// ─── Throughput Scoring ─────────────────────────────────────────────────────

export function scoreConfigurations(
  configs: Array<{
    config: ParallelismConfig
    memory: MemoryBreakdown
    label: string
  }>,
  currentMicroBatch: number,
  gradientAccumulationSteps: number
): ScoredConfiguration[] {
  const numMicrobatches = normalizeDegree(gradientAccumulationSteps)

  return configs
    .map(({ config, memory, label }) => {
      const maxBatch = estimateMaxMicroBatch(memory, currentMicroBatch)
      const isPureDP =
        config.N_tp === 1 &&
        config.N_pp === 1 &&
        config.N_cp === 1 &&
        config.N_ep === 1
      const isTPOnly =
        config.N_tp > 1 &&
        config.N_dp === 1 &&
        config.N_pp === 1 &&
        config.N_cp === 1 &&
        config.N_ep === 1

      let score: number

      if (isPureDP && config.zeroStage !== 3) {
        score = maxBatch * config.N_dp * 1.5
      } else if (isPureDP && config.zeroStage === 3) {
        score = maxBatch * config.N_dp
      } else if (isTPOnly) {
        score = maxBatch
      } else {
        score = maxBatch * config.N_dp
      }

      if (config.N_pp > 1) {
        score *= 1 - calculatePipelineBubble(config.N_pp, numMicrobatches, config.VP)
      }

      return { config, score, memory, label }
    })
    .sort((left, right) => {
      return (
        right.score - left.score ||
        left.config.zeroStage - right.config.zeroStage ||
        right.config.N_ep - left.config.N_ep ||
        right.config.N_dp - left.config.N_dp ||
        left.config.N_tp - right.config.N_tp ||
        left.config.N_pp - right.config.N_pp ||
        left.config.N_cp - right.config.N_cp
      )
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
  const warnings: Warning[] = []
  const moeEnabled = isMoEEnabled(moe)
  const pcieOnly = isPCIeOnly(gpu)
  const searchOutcome = searchRecommendation(
    params,
    arch,
    config,
    gpu,
    numGPUs,
    moe
  )

  if (!validateHiddenDimAlignment(arch.d).valid) {
    warnings.push({
      severity: "warning",
      category: "parallelism",
      message: `Hidden dimension d=${arch.d} is not aligned to 128, causing significant tensor-core inefficiency.`,
    })
  }

  const chosen =
    searchOutcome.recommended ??
    searchOutcome.closestAttempt ?? {
      config: buildParallelismConfig(config, config.parallelism.framework, {
        N_dp: Math.max(1, numGPUs),
        N_tp: 1,
        N_pp: 1,
        N_cp: 1,
        N_ep: 1,
        zeroStage: 0,
        VP: 1,
      }),
      memory: checkMemoryFit(
        params,
        config,
        arch,
        moe,
        gpu,
        buildParallelismConfig(config, config.parallelism.framework, {
          N_dp: Math.max(1, numGPUs),
          N_tp: 1,
          N_pp: 1,
          N_cp: 1,
          N_ep: 1,
          zeroStage: 0,
          VP: 1,
        })
      ),
      label: makeStrategyLabel(
        buildParallelismConfig(config, config.parallelism.framework, {
          N_dp: Math.max(1, numGPUs),
          N_tp: 1,
          N_pp: 1,
          N_cp: 1,
          N_ep: 1,
          zeroStage: 0,
          VP: 1,
        }),
        moeEnabled
      ),
    }

  const parallelism = chosen.config
  const finalConfig: TrainingConfig = {
    ...config,
    parallelism,
  }
  const minVRAMFloor = calculateMinVRAMFloorForConfig(params, finalConfig, arch)
  const pipelineBubbleFraction = calculatePipelineBubble(
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
    Math.max(1, numGPUs)
  )
  const paddedVocab = calculateVocabPadding(arch.V, parallelism.N_tp)

  if (searchOutcome.recommended === null) {
    warnings.push({
      severity: "critical",
      category: "memory",
      message: `Model does not fit on ${numGPUs}× ${gpu.name}. Increase GPU count or reduce the model, batch size, or sequence length.`,
    })
  }

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
        message: `Pipeline bubble is ${(pipelineBubbleFraction * 100).toFixed(1)}%. A common rule of thumb is num_microbatches ≥ ${4 * parallelism.N_pp}.`,
      })
    }
  }

  if (parallelism.zeroStage === 3) {
    warnings.push({
      severity: "info",
      category: "parallelism",
      message: "ZeRO-3 maximizes memory efficiency but adds the highest communication overhead.",
    })
  }

  if (parallelism.N_tp > 1 && pcieOnly) {
    warnings.push({
      severity: "warning",
      category: "parallelism",
      message: `TP=${parallelism.N_tp} on PCIe-only GPUs is bandwidth-limited relative to NVLink-equipped systems.`,
    })
  }

  if (parallelism.N_cp > 1 && (pcieOnly || parallelism.N_tp * parallelism.N_cp > gpu.gpusPerNode)) {
    warnings.push({
      severity: "warning",
      category: "parallelism",
      message: `CP=${parallelism.N_cp} introduces additional high-bandwidth traffic; scaling may be poor when CP extends beyond a node or runs on PCIe-only GPUs.`,
    })
  }

  if (parallelism.N_tp > 1 && paddedVocab > arch.V) {
    warnings.push({
      severity: "info",
      category: "parallelism",
      message: `Vocabulary padded from ${arch.V.toLocaleString()} to ${paddedVocab.toLocaleString()} for TP=${parallelism.N_tp}.`,
    })
  }

  if (parallelism.zeroStage > 0 && params.total % parallelism.N_dp !== 0) {
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
      message: "The largest-layer working set is close to the minimum usable VRAM floor even with full sharding.",
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
      message: "FSDP SHARD_GRAD_OP + PP uses the AFAB schedule here; this avoids the 1F1B microbatch minimum but increases activation residency.",
    })
  }

  return {
    config: parallelism,
    minGPUs,
    minVRAMFloor,
    pipelineBubbleFraction,
    strategyLabel: makeStrategyLabel(parallelism, moeEnabled),
    reasoning: searchOutcome.reasoning,
    warnings,
  }
}
