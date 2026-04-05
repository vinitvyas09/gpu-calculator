import type {
  ComputeEstimate,
  CostEstimate,
  FP8Config,
  FailureModelConfig,
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
  const meetsMax = max === null || value <= max
  return meetsMin && meetsMax
}

function getTrainingNumGPUs(config: TrainingConfig): number {
  return config.hardware.numGPUs && config.hardware.numGPUs > 0
    ? config.hardware.numGPUs
    : 1
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
 * Effective peak TFLOPS for the training-time formula.
 *
 * - `bf16` / `fp16`: half-precision tensor-core TFLOPS from the GPU table.
 * - `fp32`: TF32 tensor-core TFLOPS on Ampere+ when available; otherwise use
 *   a conservative `half / 8` fallback because `GPUSpec` does not expose raw
 *   non-tensor-core FP32 throughput.
 * - `fp8`: BF16 TFLOPS × empirical speedup factor. Never use raw FP8 peak.
 */
function getEffectiveTrainingTFLOPS(
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
 * Generation uses the selected storage precision for the memory-bound term and
 * the nearest available dense matmul rate for compute-bound terms.
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
 * Bytes per stored parameter.
 *
 * FP8 defaults to 2 bytes here because the calculator's default FP8 mode is
 * TransformerEngine-style compute acceleration without weight-storage savings.
 */
function getBytesPerParam(precision: TrainingPrecision): number {
  return precision === "fp32" ? 4 : 2
}

// ---------------------------------------------------------------------------
// getDefaultMFU — Section 6.3
// ---------------------------------------------------------------------------

export function getDefaultMFU(params: number, numGPUs: number): number {
  const exactMatch = MFU_DEFAULTS.find(
    (entry) =>
      !entry.advisoryOnly &&
      matchesParamRange(params, entry.minParams, entry.maxParams) &&
      matchesGPUCountRange(numGPUs, entry.minGPUs, entry.maxGPUs),
  )

  if (exactMatch) {
    return exactMatch.defaultMFU
  }

  const paramsOnlyMatch = MFU_DEFAULTS.find(
    (entry) =>
      !entry.advisoryOnly &&
      matchesParamRange(params, entry.minParams, entry.maxParams),
  )

  return paramsOnlyMatch?.defaultMFU ?? 0.4
}

// ---------------------------------------------------------------------------
// calculateFailureAdjustedTime — Section 6.5
// ---------------------------------------------------------------------------

export function calculateFailureAdjustedTime(
  theoreticalDays: number,
  config: TrainingConfig,
): FailureAdjustedTime | null
export function calculateFailureAdjustedTime(
  theoreticalDays: number,
  numGPUs: number,
  gpusPerNode: number,
  failureModel: FailureModelConfig,
): FailureAdjustedTime | null
export function calculateFailureAdjustedTime(
  theoreticalDays: number,
  configOrNumGPUs: TrainingConfig | number,
  gpusPerNode?: number,
  failureModel?: FailureModelConfig,
): FailureAdjustedTime | null {
  const numGPUs =
    typeof configOrNumGPUs === "number"
      ? configOrNumGPUs
      : getTrainingNumGPUs(configOrNumGPUs)
  const resolvedGPUsPerNode =
    typeof configOrNumGPUs === "number"
      ? Math.max(gpusPerNode ?? 1, 1)
      : Math.max(configOrNumGPUs.hardware.gpu.gpusPerNode, 1)
  const resolvedFailureModel =
    typeof configOrNumGPUs === "number"
      ? failureModel
      : configOrNumGPUs.failureModel

  if (!resolvedFailureModel) {
    return null
  }

  const failureRate = resolvedFailureModel.failureRatePerInstancePerDay
  if (failureRate <= 0) {
    return { adjustedDays: theoreticalDays, multiplier: 1 }
  }

  const checkpointFrequency = resolvedFailureModel.checkpointFrequencyPerDay
  if (checkpointFrequency <= 0) {
    return null
  }

  const nInstances = Math.ceil(numGPUs / resolvedGPUsPerNode)
  const recoveryDays = resolvedFailureModel.recoveryTimeHours / 24
  const avgLostWorkDays = 1 / (2 * checkpointFrequency)
  const denominator =
    1 - failureRate * nInstances * (recoveryDays + avgLostWorkDays)

  if (denominator <= 0) {
    return null
  }

  return {
    adjustedDays: theoreticalDays / denominator,
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
  const parameterCounts = getTrainingParameterCounts(config)
  const activeParams = activeParamsOverride ?? parameterCounts.active
  const fPeakFLOPS =
    getEffectiveTrainingTFLOPS(gpu, config.precision, config.fp8) * 1e12
  const mfu = config.mfuOverride ?? getDefaultMFU(activeParams, numGPUs)
  const denominator = numGPUs * fPeakFLOPS * mfu
  const totalTokens =
    compute.flopsPerToken > 0
      ? compute.totalFLOPs / compute.flopsPerToken
      : config.totalTokens
  const theoreticalSeconds =
    denominator > 0 ? compute.totalFLOPs / denominator : Number.POSITIVE_INFINITY
  const theoreticalDays = theoreticalSeconds / 86400
  const theoreticalHours = theoreticalSeconds / 3600
  const dataParallelDegree =
    config.parallelism.N_dp > 0 ? config.parallelism.N_dp : numGPUs
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
  const failureResult = calculateFailureAdjustedTime(theoreticalDays, config)

  return {
    theoreticalDays,
    theoreticalHours,
    failureAdjustedDays: failureResult?.adjustedDays ?? null,
    failureAdjustedHours: failureResult
      ? failureResult.adjustedDays * 24
      : null,
    failureMultiplier: failureResult?.multiplier ?? null,
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
  const recomputedFailure = calculateFailureAdjustedTime(
    time.theoreticalDays,
    config,
  )
  const failureAdjustedDays =
    time.failureAdjustedDays ?? recomputedFailure?.adjustedDays ?? null
  const failureAdjustedHours =
    time.failureAdjustedHours ??
    (failureAdjustedDays !== null ? failureAdjustedDays * 24 : null)
  const computeCost = numGPUs * time.theoreticalHours * pricing.costPerGPUHour
  const actualComputeCost =
    failureAdjustedHours === null
      ? null
      : numGPUs * failureAdjustedHours * pricing.costPerGPUHour
  const failureOverheadCost =
    actualComputeCost === null
      ? Number.POSITIVE_INFINITY
      : Math.max(actualComputeCost - computeCost, 0)
  const optimizerVariant = getOptimizerVariant(config)
  const checkpointBytesPerParam =
    optimizerVariant.masterWeightBytes > 0
      ? optimizerVariant.kOpt
      : optimizerVariant.parameterBytes + optimizerVariant.kOpt
  const checkpointSize = checkpointBytesPerParam * totalParams
  const effectiveDays = failureAdjustedDays ?? time.theoreticalDays
  const checkpointFrequency = config.failureModel.checkpointFrequencyPerDay
  const retention = Math.max(pricing.checkpointRetentionCount, 0)
  const numCheckpoints =
    checkpointFrequency > 0 ? Math.ceil(effectiveDays * checkpointFrequency) : 0
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
  const avgStorageGB = averageCheckpointStorage / 1e9
  const storageCost =
    pricing.storagePricePerGBMonth * avgStorageGB * (effectiveDays / 30.25)
  const totalCost =
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
  const batchGen = usingConfig ? numGPUsOrBatchGen : batchGenOrPrompt
  const nTokens = usingConfig
    ? (precisionOrNTokens as number)
    : (nTokensMaybe ?? 0)
  const sPrompt = usingConfig ? batchGenOrPrompt : (sPromptMaybe ?? 0)
  const fPeakFLOPS = getEffectiveGenerationTFLOPS(gpu, precision) * 1e12
  const bwMemBps = gpu.memoryBandwidthGBps * 1e9 * 0.9
  const beta = getBytesPerParam(precision)

  // Section 10.3 writes the prefill term per sequence; for a batch of
  // concurrent prompts the total prefill FLOPs scale linearly with `batchGen`.
  const prefillSeconds =
    (2 * params * sPrompt * batchGen) / (fPeakFLOPS * numGPUs)
  const memoryBoundPerToken = (2 * params * beta) / (bwMemBps * numGPUs)
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
