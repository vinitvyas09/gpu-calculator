import type {
  ComputeEstimate,
  CostEstimate,
  FP8Config,
  GPUSpec,
  PostTrainingConfig,
  PostTrainingMethod,
  TrainingConfig,
  TrainingPrecision,
  TrainingTimeEstimate,
} from "../types"
import { MFU_DEFAULTS, OPTIMIZER_PROFILES } from "../constants"
import { calculateParameterCount } from "./compute"

interface FailureAdjustedTime {
  adjustedDays: number
  adjustedHours: number
  multiplier: number
}

interface GenerationTimeEstimate {
  prefillSeconds: number
  decodeSeconds: number
  totalSeconds: number
  isMemoryBound: boolean
}

function matchesParamRange(
  value: number,
  min: number | null,
  max: number | null,
): boolean {
  const meetsMin = min === null || value >= min
  const meetsMax = max === null || value < max
  return meetsMin && meetsMax
}

function matchesGPUCountRange(
  value: number,
  min: number | null,
  max: number | null,
): boolean {
  const meetsMin = min === null || value >= min
  const meetsMax = max === null || value < max
  return meetsMin && meetsMax
}

function normalizeDegree(value: number): number {
  return Number.isFinite(value) && value > 0 ? value : 1
}

function multiplyFactors(...factors: number[]): number {
  if (factors.some((factor) => factor === 0)) {
    return 0
  }

  return factors.reduce((product, factor) => product * factor, 1)
}

function getConfiguredWorldSize(config: TrainingConfig): number {
  const { N_dp, N_tp, N_pp, N_cp, N_ep } = config.parallelism
  return (
    normalizeDegree(N_dp) *
    normalizeDegree(N_tp) *
    normalizeDegree(N_pp) *
    normalizeDegree(N_cp) *
    normalizeDegree(N_ep)
  )
}

function getTrainingNumGPUs(config: TrainingConfig): number {
  const explicitNumGPUs = config.hardware.numGPUs
  if (
    typeof explicitNumGPUs === "number" &&
    Number.isFinite(explicitNumGPUs) &&
    explicitNumGPUs > 0
  ) {
    return explicitNumGPUs
  }

  return Math.max(getConfiguredWorldSize(config), 1)
}

function resolveDataParallelDegree(
  config: TrainingConfig,
  numGPUs: number,
): number {
  const configuredDP = config.parallelism.N_dp
  const nonDPProduct =
    normalizeDegree(config.parallelism.N_tp) *
    normalizeDegree(config.parallelism.N_pp) *
    normalizeDegree(config.parallelism.N_cp) *
    normalizeDegree(config.parallelism.N_ep)

  if (
    Number.isFinite(configuredDP) &&
    configuredDP > 0 &&
    configuredDP * nonDPProduct === numGPUs
  ) {
    return configuredDP
  }

  if (nonDPProduct > 0 && numGPUs > 0 && numGPUs % nonDPProduct === 0) {
    return Math.max(numGPUs / nonDPProduct, 1)
  }

  if (Number.isFinite(configuredDP) && configuredDP > 0) {
    return configuredDP
  }

  return Math.max(numGPUs, 1)
}

function getPostTrainingNumGPUs(config: PostTrainingConfig): number {
  return config.hardware.numGPUs > 0 ? config.hardware.numGPUs : 1
}

function getTrainingParameterCounts(config: TrainingConfig) {
  return calculateParameterCount(
    config.model.architecture,
    config.model.moe,
    config.sequenceLength,
  )
}

function getOptimizerVariant(config: TrainingConfig) {
  const profile = OPTIMIZER_PROFILES.find(
    (candidate) => candidate.id === config.optimizer,
  )

  if (!profile) {
    throw new Error(`Unknown optimizer profile: ${config.optimizer}`)
  }

  return config.gradientPrecision === "bf16"
    ? profile.bf16Grad
    : profile.fp32Grad
}

/**
 * Section 6.1 / 6.2:
 * - bf16/fp16 training uses the dense half-precision matmul peak.
 * - fp32 training uses TF32 peak on Ampere+ GPUs when available.
 * - fp8 training uses BF16 peak scaled by the empirical fp8 speedup factor,
 *   never the raw fp8 spec-sheet peak.
 */
export function getEffectiveTrainingTFLOPS(
  gpu: GPUSpec,
  precision: TrainingPrecision,
  fp8Config: FP8Config,
): number {
  switch (precision) {
    case "bf16":
    case "fp16":
      return gpu.halfPrecisionTFLOPS
    case "fp32":
      if (gpu.supportsTF32 && gpu.tf32TFLOPS !== null) {
        return gpu.tf32TFLOPS
      }
      return gpu.halfPrecisionTFLOPS / 8
    case "fp8":
      return gpu.halfPrecisionTFLOPS * fp8Config.kernelSpeedupFactor
  }
}

/**
 * Section 10.3 uses dense-matmul FLOPS for the compute-bound prefill/decode
 * terms. For fp8, stay conservative and use the dense half-precision peak
 * because post-training config does not expose a separate fp8 inference
 * speedup factor.
 */
function getEffectiveGenerationTFLOPS(
  gpu: GPUSpec,
  precision: TrainingPrecision,
): number {
  switch (precision) {
    case "fp32":
      if (gpu.supportsTF32 && gpu.tf32TFLOPS !== null) {
        return gpu.tf32TFLOPS
      }
      return gpu.halfPrecisionTFLOPS / 8
    case "bf16":
    case "fp16":
    case "fp8":
      return gpu.halfPrecisionTFLOPS
  }
}

/**
 * Weight-storage bytes used in the Section 10.3 decode memory-bound term.
 * TransformerEngine-style fp8 keeps weights in bf16/fp16 storage, so default
 * fp8 generation still behaves like 2 bytes/parameter here.
 */
function getGenerationWeightBytes(precision: TrainingPrecision): number {
  return precision === "fp32" ? 4 : 2
}

/**
 * Persisted checkpoint bytes per parameter.
 *
 * `kOpt` already includes master weights whenever they exist; pure fp32 or
 * no-master optimizers store the live parameter tensor instead.
 */
function getCheckpointBytesPerParam(config: TrainingConfig): number {
  const optimizerVariant = getOptimizerVariant(config)

  return optimizerVariant.masterWeightBytes > 0
    ? optimizerVariant.kOpt
    : optimizerVariant.parameterBytes + optimizerVariant.kOpt
}

// ---------------------------------------------------------------------------
// getDefaultMFU — Section 6.3
// ---------------------------------------------------------------------------

export function getDefaultMFU(params: number, numGPUs: number): number {
  const nonAdvisoryDefaults = MFU_DEFAULTS.filter((entry) => !entry.advisoryOnly)

  const exactMatch = nonAdvisoryDefaults.find(
    (entry) =>
      matchesParamRange(params, entry.minParams, entry.maxParams) &&
      matchesGPUCountRange(numGPUs, entry.minGPUs, entry.maxGPUs),
  )

  if (exactMatch) {
    return exactMatch.defaultMFU
  }

  const paramTier = nonAdvisoryDefaults.find((entry) =>
    matchesParamRange(params, entry.minParams, entry.maxParams),
  )
  const gpuTier = nonAdvisoryDefaults.find((entry) =>
    matchesGPUCountRange(numGPUs, entry.minGPUs, entry.maxGPUs),
  )

  if (paramTier && gpuTier) {
    return Math.min(paramTier.defaultMFU, gpuTier.defaultMFU)
  }

  return paramTier?.defaultMFU ?? gpuTier?.defaultMFU ?? 0.4
}

// ---------------------------------------------------------------------------
// calculateFailureAdjustedTime — Section 6.5
// ---------------------------------------------------------------------------

export function calculateFailureAdjustedTime(
  theoreticalDays: number,
  config: TrainingConfig,
): FailureAdjustedTime {
  const numGPUs = getTrainingNumGPUs(config)
  const gpusPerNode = Math.max(config.hardware.gpu.gpusPerNode, 1)
  const failureRate = Math.max(
    config.failureModel.failureRatePerInstancePerDay,
    0,
  )

  if (failureRate === 0) {
    return {
      adjustedDays: theoreticalDays,
      adjustedHours: theoreticalDays * 24,
      multiplier: 1,
    }
  }

  const nInstances = Math.ceil(numGPUs / gpusPerNode)
  const recoveryDays = Math.max(config.failureModel.recoveryTimeHours, 0) / 24
  const checkpointFrequency = Math.max(
    config.failureModel.checkpointFrequencyPerDay,
    0,
  )
  const averageLostWorkDays =
    checkpointFrequency > 0
      ? 1 / (2 * checkpointFrequency)
      : Number.POSITIVE_INFINITY
  const denominator =
    1 - failureRate * nInstances * (recoveryDays + averageLostWorkDays)

  if (denominator <= 0) {
    return {
      adjustedDays: Number.POSITIVE_INFINITY,
      adjustedHours: Number.POSITIVE_INFINITY,
      multiplier: Number.POSITIVE_INFINITY,
    }
  }

  const adjustedDays = theoreticalDays / denominator

  return {
    adjustedDays,
    adjustedHours: adjustedDays * 24,
    multiplier: 1 / denominator,
  }
}

// ---------------------------------------------------------------------------
// calculateTrainingTime — Section 6.1
// ---------------------------------------------------------------------------

export function calculateTrainingTime(
  compute: ComputeEstimate,
  config: TrainingConfig,
): TrainingTimeEstimate
export function calculateTrainingTime(
  compute: ComputeEstimate,
  config: TrainingConfig,
  activeParams: number,
): TrainingTimeEstimate
export function calculateTrainingTime(
  compute: ComputeEstimate,
  config: TrainingConfig,
  activeParamsOverride?: number,
): TrainingTimeEstimate {
  const numGPUs = getTrainingNumGPUs(config)
  const gpu = config.hardware.gpu
  const activeParams =
    activeParamsOverride ?? getTrainingParameterCounts(config).active
  const fPeakFLOPS =
    getEffectiveTrainingTFLOPS(gpu, config.precision, config.fp8) * 1e12
  const mfu = config.mfuOverride ?? getDefaultMFU(activeParams, numGPUs)
  const denominator = numGPUs * fPeakFLOPS * mfu
  const theoreticalSeconds =
    denominator > 0 ? compute.totalFLOPs / denominator : Number.POSITIVE_INFINITY
  const theoreticalDays = theoreticalSeconds / 86400
  const theoreticalHours = theoreticalSeconds / 3600
  const derivedTotalTokens =
    compute.flopsPerToken > 0
      ? compute.totalFLOPs / compute.flopsPerToken
      : config.totalTokens
  const totalTokens =
    Number.isFinite(derivedTotalTokens) && derivedTotalTokens > 0
      ? derivedTotalTokens
      : config.totalTokens
  const dataParallelDegree = resolveDataParallelDegree(config, numGPUs)
  const globalBatchTokens =
    config.microBatchSize *
    config.sequenceLength *
    config.gradientAccumulationSteps *
    dataParallelDegree
  const totalSteps =
    globalBatchTokens > 0 ? Math.ceil(totalTokens / globalBatchTokens) : 0
  const tokensPerSecond =
    Number.isFinite(theoreticalSeconds) && theoreticalSeconds > 0
      ? totalTokens / theoreticalSeconds
      : 0
  const secondsPerStep =
    totalSteps > 0 && Number.isFinite(theoreticalSeconds)
      ? theoreticalSeconds / totalSteps
      : totalSteps > 0
        ? Number.POSITIVE_INFINITY
        : 0
  const failureAdjusted = calculateFailureAdjustedTime(theoreticalDays, config)

  return {
    theoreticalDays,
    theoreticalHours,
    failureAdjustedDays: failureAdjusted.adjustedDays,
    failureAdjustedHours: failureAdjusted.adjustedHours,
    failureMultiplier: failureAdjusted.multiplier,
    tokensPerSecond,
    totalSteps,
    secondsPerStep,
  }
}

// ---------------------------------------------------------------------------
// calculateCost — Section 8
// ---------------------------------------------------------------------------

export function calculateCost(
  time: TrainingTimeEstimate,
  config: TrainingConfig,
): CostEstimate
export function calculateCost(
  time: TrainingTimeEstimate,
  config: TrainingConfig,
  totalParams: number,
): CostEstimate
export function calculateCost(
  time: TrainingTimeEstimate,
  config: TrainingConfig,
  totalParamsOverride?: number,
): CostEstimate {
  const numGPUs = getTrainingNumGPUs(config)
  const pricing = config.pricing
  const totalParams =
    totalParamsOverride ?? getTrainingParameterCounts(config).total
  const failureAdjusted =
    time.failureAdjustedDays !== null && time.failureAdjustedHours !== null
      ? {
          adjustedDays: time.failureAdjustedDays,
          adjustedHours: time.failureAdjustedHours,
        }
      : calculateFailureAdjustedTime(time.theoreticalDays, config)
  const computeCost = multiplyFactors(
    numGPUs,
    time.theoreticalHours,
    pricing.costPerGPUHour,
  )
  const actualComputeCost = multiplyFactors(
    numGPUs,
    failureAdjusted.adjustedHours,
    pricing.costPerGPUHour,
  )
  const failureOverheadCost =
    actualComputeCost === computeCost
      ? 0
      : !Number.isFinite(actualComputeCost) || !Number.isFinite(computeCost)
        ? Number.POSITIVE_INFINITY
        : Math.max(actualComputeCost - computeCost, 0)
  const checkpointSize = getCheckpointBytesPerParam(config) * totalParams
  const checkpointFrequency = Math.max(
    config.failureModel.checkpointFrequencyPerDay,
    0,
  )
  const retention = Math.max(pricing.checkpointRetentionCount, 0)
  const numCheckpoints =
    checkpointFrequency > 0
      ? Math.ceil(failureAdjusted.adjustedDays * checkpointFrequency)
      : 0
  const peakCheckpointStorage =
    Math.min(numCheckpoints, retention) * checkpointSize

  let avgCheckpointCount = 0
  if (retention > 0 && numCheckpoints > 0) {
    avgCheckpointCount =
      numCheckpoints <= retention
        ? (numCheckpoints + 1) / 2
        : retention - (retention * (retention - 1)) / (2 * numCheckpoints)
  }

  const averageCheckpointStorage = avgCheckpointCount * checkpointSize
  const averageCheckpointStorageGB = averageCheckpointStorage / 1e9
  const runDurationMonths = failureAdjusted.adjustedDays / 30.25
  const storageCost =
    averageCheckpointStorageGB > 0 && runDurationMonths > 0
      ? multiplyFactors(
          pricing.storagePricePerGBMonth,
          averageCheckpointStorageGB,
          runDurationMonths,
        )
      : 0
  const totalCost =
    Number.isFinite(computeCost) &&
    Number.isFinite(storageCost) &&
    Number.isFinite(failureOverheadCost)
      ? computeCost + storageCost + failureOverheadCost
      : Number.POSITIVE_INFINITY

  return {
    computeCost,
    actualComputeCost,
    storageCost,
    failureOverheadCost,
    totalCost,
    checkpointSize,
    numCheckpoints,
    peakCheckpointStorage,
    averageCheckpointStorage,
  }
}

// ---------------------------------------------------------------------------
// calculatePostTrainingCompute — Section 10.5
// ---------------------------------------------------------------------------

export function calculatePostTrainingCompute(
  method: PostTrainingMethod,
  params: number,
  config: PostTrainingConfig,
): { totalFLOPs: number; flopsPerToken: number } {
  const multiplierByMethod: Record<PostTrainingMethod, number> = {
    sft: 6,
    dpo: 8,
    ppo: 20,
    grpo: 10,
  }
  const flopsPerToken = multiplierByMethod[method] * params
  const totalTokens =
    config.datasetSizeExamples * config.epochs * config.sequenceLength

  return {
    totalFLOPs: totalTokens * flopsPerToken,
    flopsPerToken,
  }
}

// ---------------------------------------------------------------------------
// calculateGenerationTime — Section 10.3
// ---------------------------------------------------------------------------

export function calculateGenerationTime(
  params: number,
  config: PostTrainingConfig,
  batchGen: number,
  nTokens: number,
  sPrompt: number,
): GenerationTimeEstimate
export function calculateGenerationTime(
  params: number,
  gpu: GPUSpec,
  numGPUs: number,
  precision: TrainingPrecision,
  batchGen: number,
  nTokens: number,
  sPrompt: number,
): GenerationTimeEstimate
export function calculateGenerationTime(
  params: number,
  configOrGPU: PostTrainingConfig | GPUSpec,
  numGPUsOrBatchGen: number,
  precisionOrNTokens: TrainingPrecision | number,
  batchGenOrPrompt: number,
  nTokensMaybe?: number,
  sPromptMaybe?: number,
): GenerationTimeEstimate {
  const usingConfig = "hardware" in configOrGPU
  const gpu = usingConfig ? configOrGPU.hardware.gpu : configOrGPU
  const numGPUs = usingConfig
    ? getPostTrainingNumGPUs(configOrGPU)
    : Math.max(numGPUsOrBatchGen, 1)
  const precision = usingConfig
    ? configOrGPU.precision
    : (precisionOrNTokens as TrainingPrecision)
  const batchGen = Math.max(usingConfig ? numGPUsOrBatchGen : batchGenOrPrompt, 0)
  const nTokens = Math.max(
    usingConfig ? (precisionOrNTokens as number) : (nTokensMaybe ?? 0),
    0,
  )
  const sPrompt = Math.max(
    usingConfig ? batchGenOrPrompt : (sPromptMaybe ?? 0),
    0,
  )
  const fPeakFLOPS = getEffectiveGenerationTFLOPS(gpu, precision) * 1e12
  const bwMemBps = gpu.memoryBandwidthGBps * 1e9 * 0.9
  const weightBytes = getGenerationWeightBytes(precision)

  // Section 10.3 gives the prefill term for one prompt; for `batchGen`
  // concurrent prompts, total prefill FLOPs scale linearly with the batch.
  const prefillSeconds =
    (2 * params * sPrompt * batchGen) / (fPeakFLOPS * numGPUs)
  const memoryBoundPerToken = (2 * params * weightBytes) / (bwMemBps * numGPUs)
  const computeBoundPerToken =
    (2 * params * batchGen) / (fPeakFLOPS * numGPUs)
  const decodePerToken = Math.max(memoryBoundPerToken, computeBoundPerToken)
  const decodeSeconds = nTokens * decodePerToken

  return {
    prefillSeconds,
    decodeSeconds,
    totalSeconds: prefillSeconds + decodeSeconds,
    isMemoryBound: memoryBoundPerToken >= computeBoundPerToken,
  }
}
