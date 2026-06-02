import type {
  ComputeEstimate,
  CostEstimate,
  FP8Config,
  GPUSpec,
  OptimizerMemoryVariant,
  PostTrainingConfig,
  PostTrainingMethod,
  TrainingConfig,
  TrainingPrecision,
  TrainingTimeEstimate,
} from "../types"
import { MFU_DEFAULTS, OPTIMIZER_PROFILES } from "../constants"
import { calculateParameterCount } from "./compute"
import {
  hasInvalidPostTrainingOptimizer,
  hasInvalidPretrainingOptimizer,
} from "./optimizer-validation"
import {
  hasInvalidQLoRAQuantizationBits,
  hasInvalidPostTrainingApproachConfig,
  hasInvalidPostTrainingModelShape,
  hasInvalidPostTrainingMethodApproach,
} from "./post-training-validation"
import {
  calculateLoRAParamCountForArchitecture,
  calculateQuantizedActiveModelBytesPerParam,
  calculateQuantizedBaseModelBytes,
  hasInvalidLoRATargetModules,
} from "./memory"
import {
  hasInvalidCPUOffloadConfig,
  hasInvalidManualContextParallelismTopology,
  hasInvalidManualExpertParallelismTopology,
  hasInvalidManualWorldSize,
  hasInvalidManualPipelineTopology,
  hasInvalidManualTensorExpertSequenceParallelismTopology,
  hasInvalidManualTensorParallelismTopology,
  resolveEffectiveZeroStage,
} from "./parallelism-validation"
import {
  hasInvalidFP8Config,
  hasInvalidFP8StorageMode,
  isValidFP8KernelSpeedupFactor,
} from "./fp8-validation"
import { hasInvalidTrainingHardware } from "./hardware"
import { hasInvalidPostTrainingKVCachePrecision } from "./kv-cache-validation"

export const MAX_MFU_OVERRIDE = 0.7

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

const CHECKPOINT_FILE_OVERHEAD_FACTOR = 1.04
const FAILURE_ADJUSTMENT_DISPLAY_GPU_THRESHOLD = 256

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
  return Number.isFinite(value) && value > 0 ? Math.max(1, Math.floor(value)) : 1
}

function isFinitePositiveInteger(value: number): boolean {
  return Number.isFinite(value) && value > 0 && Number.isInteger(value)
}

function hasInvalidManualParallelismDegrees(config: TrainingConfig): boolean {
  if (config.parallelismMode !== "manual") {
    return false
  }

  const { N_dp, N_tp, N_pp, N_cp, N_ep, VP } = config.parallelism
  return [N_dp, N_tp, N_pp, N_cp, N_ep, VP].some(
    (degree) => !isFinitePositiveInteger(degree),
  )
}

function hasInvalidTrainingGPUCount(config: TrainingConfig): boolean {
  return (
    (config.hardware.numGPUs !== null &&
      !isFinitePositiveInteger(config.hardware.numGPUs)) ||
    hasInvalidManualWorldSize(config)
  )
}

function hasInvalidPartialActivationCheckpointing(config: TrainingConfig): boolean {
  if (config.activationCheckpointing !== "partial") {
    return false
  }

  const depth = config.partialCheckpointDepth
  const layerCount = config.model.architecture.L
  const pipelineDegree = normalizeDegree(config.parallelism.N_pp)
  const maxLayersPerStage =
    Number.isFinite(layerCount) && layerCount > 0
      ? Math.max(1, Math.ceil(Math.floor(layerCount) / pipelineDegree))
      : 0

  return (
    !isFinitePositiveInteger(depth ?? Number.NaN) ||
    maxLayersPerStage <= 0 ||
    (depth ?? 0) > maxLayersPerStage
  )
}

function hasInvalidPostTrainingGPUCount(config: PostTrainingConfig): boolean {
  return (
    !config.hardware.gpu.singleDeviceOnly &&
    !isFinitePositiveInteger(config.hardware.numGPUs)
  )
}

function normalizeNonNegativeCount(value: number): number | null {
  return Number.isFinite(value) && value >= 0 && Number.isInteger(value)
    ? value
    : null
}

function multiplyFactors(...factors: number[]): number {
  if (factors.some((factor) => factor === 0)) {
    return 0
  }

  return factors.reduce((product, factor) => product * factor, 1)
}

export function calculateGPUHourlyCost(
  numGPUs: number,
  hours: number,
  costPerGPUHour: number | null,
): number {
  if (
    costPerGPUHour === null ||
    !Number.isFinite(numGPUs) ||
    !Number.isFinite(hours) ||
    numGPUs < 0 ||
    hours < 0
  ) {
    return Number.POSITIVE_INFINITY
  }

  if (numGPUs === 0 || hours === 0 || costPerGPUHour === 0) {
    return 0
  }

  return numGPUs * hours * costPerGPUHour
}

function divideWork(numerator: number, denominator: number): number {
  if (numerator === 0) {
    return 0
  }

  if (!Number.isFinite(numerator)) {
    return Number.POSITIVE_INFINITY
  }

  if (!Number.isFinite(denominator) || denominator <= 0) {
    return Number.POSITIVE_INFINITY
  }

  return numerator / denominator
}

function getFiniteNonNegativeOrInfinity(value: number): number {
  return Number.isFinite(value) && value >= 0
    ? value
    : Number.POSITIVE_INFINITY
}

function getFinitePositiveOrInfinity(value: number): number {
  return Number.isFinite(value) && value > 0
    ? value
    : Number.POSITIVE_INFINITY
}

function getFinitePositiveIntegerOrInfinity(value: number): number {
  return isFinitePositiveInteger(value) ? value : Number.POSITIVE_INFINITY
}

function getFiniteNonNegativeOrZero(value: number): number {
  return Number.isFinite(value) && value >= 0 ? value : 0
}

function getFiniteNonNegative(value: number): number | null {
  return Number.isFinite(value) && value >= 0 ? value : null
}

function getFinitePositive(value: number): number | null {
  return Number.isFinite(value) && value > 0 ? value : null
}

function invalidCostEstimate(): CostEstimate {
  return {
    computeCost: Number.POSITIVE_INFINITY,
    actualComputeCost: Number.POSITIVE_INFINITY,
    storageCost: Number.POSITIVE_INFINITY,
    failureOverheadCost: Number.POSITIVE_INFINITY,
    totalCost: Number.POSITIVE_INFINITY,
    checkpointSize: Number.POSITIVE_INFINITY,
    numCheckpoints: Number.POSITIVE_INFINITY,
    peakCheckpointStorage: Number.POSITIVE_INFINITY,
    averageCheckpointStorage: Number.POSITIVE_INFINITY,
    datasetStorageBytes: Number.POSITIVE_INFINITY,
  }
}

function invalidGenerationTime(): GenerationTimeEstimate {
  return {
    prefillSeconds: Number.POSITIVE_INFINITY,
    decodeSeconds: Number.POSITIVE_INFINITY,
    totalSeconds: Number.POSITIVE_INFINITY,
    isMemoryBound: false,
  }
}

function getFinitePositiveInteger(value: number): number | null {
  return isFinitePositiveInteger(value) ? value : null
}

function infiniteFailureAdjustedTime(): FailureAdjustedTime {
  return {
    adjustedDays: Number.POSITIVE_INFINITY,
    adjustedHours: Number.POSITIVE_INFINITY,
    multiplier: Number.POSITIVE_INFINITY,
  }
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

function hasValidConfiguredWorldSize(config: TrainingConfig): boolean {
  const { N_dp, N_tp, N_pp, N_cp, N_ep } = config.parallelism

  return [N_dp, N_tp, N_pp, N_cp, N_ep].every(
    (degree) => Number.isFinite(degree) && degree > 0,
  )
}

function getTrainingNumGPUs(config: TrainingConfig): number {
  if (config.hardware.gpu.singleDeviceOnly) {
    return 1
  }

  const configuredWorldSize = getConfiguredWorldSize(config)

  if (
    hasValidConfiguredWorldSize(config) &&
    Number.isFinite(configuredWorldSize) &&
    configuredWorldSize > 0
  ) {
    return configuredWorldSize
  }

  const explicitNumGPUs = config.hardware.numGPUs
  if (
    typeof explicitNumGPUs === "number" &&
    Number.isFinite(explicitNumGPUs) &&
    explicitNumGPUs > 0
  ) {
    return Math.max(1, Math.floor(explicitNumGPUs))
  }

  return Math.max(getConfiguredWorldSize(config), 1)
}

function shouldSurfaceFailureAdjustedTime(numGPUs: number): boolean {
  return (
    Number.isFinite(numGPUs) &&
    numGPUs >= FAILURE_ADJUSTMENT_DISPLAY_GPU_THRESHOLD
  )
}

function canUseInterleavedPipelineSchedule(
  N_pp: number,
  numMicrobatches: number,
  VP: number,
): boolean {
  return N_pp > 1 && VP > 1 && numMicrobatches % N_pp === 0
}

function resolveDataParallelDegree(
  config: TrainingConfig,
  numGPUs: number,
): number {
  if (config.hardware.gpu.singleDeviceOnly) {
    return 1
  }

  const configuredDP = config.parallelism.N_dp

  if (Number.isFinite(configuredDP) && configuredDP > 0) {
    return normalizeDegree(configuredDP)
  }

  const nonDPProduct =
    normalizeDegree(config.parallelism.N_tp) *
    normalizeDegree(config.parallelism.N_pp) *
    normalizeDegree(config.parallelism.N_cp) *
    normalizeDegree(config.parallelism.N_ep)

  if (nonDPProduct > 0 && numGPUs > 0 && numGPUs % nonDPProduct === 0) {
    return Math.max(numGPUs / nonDPProduct, 1)
  }

  return Math.max(numGPUs, 1)
}

function getPostTrainingNumGPUs(config: PostTrainingConfig): number {
  if (config.hardware.gpu.singleDeviceOnly) {
    return 1
  }

  return isFinitePositiveInteger(config.hardware.numGPUs)
    ? config.hardware.numGPUs
    : Number.POSITIVE_INFINITY
}

function getTrainingParameterCounts(config: TrainingConfig) {
  return calculateParameterCount(
    config.model.architecture,
    config.model.moe,
    config.sequenceLength,
  )
}

function applyAMPAutocastOptimizerVariant(
  variant: OptimizerMemoryVariant,
  config: TrainingConfig,
): OptimizerMemoryVariant {
  if (!config.ampAutocast) {
    return variant
  }

  const kOpt = Math.max(0, variant.kOpt - variant.masterWeightBytes)
  const optimizerStateBytes = Math.max(
    0,
    variant.optimizerStateBytes - variant.masterWeightBytes,
  )

  return {
    ...variant,
    parameterBytes: 4,
    masterWeightBytes: 0,
    optimizerStateBytes,
    kOpt,
    phi: 4 + variant.betaGrad + kOpt,
    breakdown: `4 (fp32 params/autocast master) + ${variant.betaGrad} (grads) + ${kOpt} (optimizer states)`,
  }
}

function usesFSDPMixedPrecision(config: TrainingConfig): boolean {
  return (
    config.parallelism.framework === "fsdp" &&
    !config.ampAutocast &&
    config.precision !== "fp32" &&
    !(
      config.precision === "fp8" &&
      config.fp8.storageMode === "ms-amp"
    )
  )
}

function applyFSDPMixedPrecisionOptimizerVariant(
  variant: OptimizerMemoryVariant,
  config: TrainingConfig,
): OptimizerMemoryVariant {
  if (!usesFSDPMixedPrecision(config)) {
    return variant
  }

  const kOpt = Math.max(0, variant.kOpt - variant.masterWeightBytes)
  const optimizerStateBytes = Math.max(
    0,
    variant.optimizerStateBytes - variant.masterWeightBytes,
  )

  return {
    ...variant,
    parameterBytes: 4,
    masterWeightBytes: 0,
    optimizerStateBytes,
    kOpt,
    phi: 4 + variant.betaGrad + kOpt,
    breakdown: `4 (fp32 FSDP param shards) + ${variant.betaGrad} (grads) + ${kOpt} (optimizer states)`,
  }
}

function applyFP32PrecisionOptimizerVariant(
  variant: OptimizerMemoryVariant,
  precision: TrainingPrecision,
): OptimizerMemoryVariant {
  if (precision !== "fp32") {
    return variant
  }

  const kOpt = Math.max(0, variant.kOpt - variant.masterWeightBytes)
  const optimizerStateBytes = Math.max(
    0,
    variant.optimizerStateBytes - variant.masterWeightBytes,
  )
  const betaGrad = variant.betaGrad > 0 ? 4 : 0

  return {
    ...variant,
    parameterBytes: 4,
    betaGrad,
    masterWeightBytes: 0,
    optimizerStateBytes,
    kOpt,
    phi: 4 + betaGrad + kOpt,
    breakdown: `4 (fp32 params) + ${betaGrad} (fp32 grads) + ${kOpt} (optimizer states)`,
  }
}

function invalidOptimizerVariant(): OptimizerMemoryVariant {
  return {
    parameterBytes: Number.POSITIVE_INFINITY,
    betaGrad: Number.POSITIVE_INFINITY,
    masterWeightBytes: Number.POSITIVE_INFINITY,
    optimizerStateBytes: Number.POSITIVE_INFINITY,
    kOpt: Number.POSITIVE_INFINITY,
    phi: Number.POSITIVE_INFINITY,
    breakdown: "Invalid optimizer",
  }
}

function getOptimizerVariant(config: TrainingConfig) {
  if (
    hasInvalidPretrainingOptimizer(config.optimizer) ||
    hasInvalidFP8StorageMode(config)
  ) {
    return invalidOptimizerVariant()
  }

  const optimizer =
    config.optimizer === "adamw-fp8" &&
    (config.precision !== "fp8" ||
      !config.hardware.gpu.supportsFP8 ||
      config.fp8.storageMode === "transformer-engine")
      ? "adamw-mixed"
      : config.optimizer
  const profile = OPTIMIZER_PROFILES.find(
    (candidate) => candidate.id === optimizer,
  )

  if (!profile) {
    throw new Error(`Unknown optimizer profile: ${optimizer}`)
  }

  const variant =
    config.gradientPrecision === "bf16"
      ? profile.bf16Grad
      : profile.fp32Grad

  return applyFP32PrecisionOptimizerVariant(
    applyFSDPMixedPrecisionOptimizerVariant(
      applyAMPAutocastOptimizerVariant(variant, config),
      config,
    ),
    config.precision,
  )
}

function getPostTrainingOptimizerVariant(config: PostTrainingConfig) {
  if (
    hasInvalidPostTrainingOptimizer(config.optimizer) ||
    hasInvalidFP8StorageMode(config)
  ) {
    return invalidOptimizerVariant()
  }

  const optimizer =
    config.optimizer === "adamw-fp8" &&
    (config.precision !== "fp8" ||
      !config.hardware.gpu.supportsFP8 ||
      config.fp8.storageMode === "transformer-engine")
      ? "adamw-mixed"
      : config.optimizer
  const profile = OPTIMIZER_PROFILES.find(
    (candidate) => candidate.id === optimizer,
  )

  if (!profile) {
    throw new Error(`Unknown optimizer profile: ${optimizer}`)
  }

  const variant =
    config.gradientPrecision === "bf16"
      ? profile.bf16Grad
      : profile.fp32Grad

  return applyFP32PrecisionOptimizerVariant(variant, config.precision)
}

function getEffectiveFP32TFLOPS(gpu: GPUSpec): number {
  if (
    gpu.supportsTF32 &&
    gpu.tf32TFLOPS !== null &&
    Number.isFinite(gpu.tf32TFLOPS) &&
    gpu.tf32TFLOPS > 0
  ) {
    return gpu.tf32TFLOPS
  }

  if (
    gpu.fp32TFLOPS !== null &&
    gpu.fp32TFLOPS !== undefined &&
    Number.isFinite(gpu.fp32TFLOPS) &&
    gpu.fp32TFLOPS > 0
  ) {
    return gpu.fp32TFLOPS
  }

  return gpu.halfPrecisionTFLOPS / 8
}

/**
 * Section 6.1 / 6.2:
 * - bf16/fp16 training uses the dense half-precision matmul peak.
 * - fp32 training uses TF32 peak on Ampere+ GPUs when available.
 * - fp8 training uses BF16 peak scaled by the empirical fp8 speedup factor,
 *   never the raw fp8 spec-sheet peak.
 * - unsupported requested precision returns 0 so ungated callers fail closed.
 */
export function getEffectiveTrainingTFLOPS(
  gpu: GPUSpec,
  precision: TrainingPrecision,
  fp8Config: FP8Config,
): number {
  switch (precision) {
    case "bf16":
      if (!gpu.supportsBF16) {
        return 0
      }

      return gpu.halfPrecisionTFLOPS
    case "fp16":
      return gpu.halfPrecisionTFLOPS
    case "fp32":
      return getEffectiveFP32TFLOPS(gpu)
    case "fp8":
      if (!gpu.supportsFP8) {
        return 0
      }

      return isValidFP8KernelSpeedupFactor(fp8Config.kernelSpeedupFactor)
        ? gpu.halfPrecisionTFLOPS * fp8Config.kernelSpeedupFactor
        : 0
  }
}

/**
 * Section 10.3 uses matmul-class throughput for compute-bound prefill/decode
 * terms. For fp8, use the configured effective kernel speedup when a full
 * post-training config is available; legacy callers without FP8 settings fall
 * back to the dense half-precision peak.
 */
function getEffectiveGenerationTFLOPS(
  gpu: GPUSpec,
  precision: TrainingPrecision,
  fp8Config?: FP8Config,
): number {
  switch (precision) {
    case "fp32":
      return getEffectiveFP32TFLOPS(gpu)
    case "bf16":
      if (!gpu.supportsBF16) {
        return 0
      }

      return gpu.halfPrecisionTFLOPS
    case "fp16":
      return gpu.halfPrecisionTFLOPS
    case "fp8":
      if (!gpu.supportsFP8) {
        return 0
      }

      if (!fp8Config) {
        return gpu.halfPrecisionTFLOPS
      }

      return isValidFP8KernelSpeedupFactor(fp8Config.kernelSpeedupFactor)
        ? gpu.halfPrecisionTFLOPS * fp8Config.kernelSpeedupFactor
        : 0
  }
}

/**
 * Weight-storage bytes used in the Section 10.3 decode memory-bound term.
 * TransformerEngine-style fp8 keeps weights in bf16/fp16 storage, so default
 * fp8 generation still behaves like 2 bytes/parameter here. Full-model
 * MS-AMP FP8 is the exception because the policy parameters themselves are
 * stored in fp8. QLoRA uses the quantized base footprint because Section 10.3's
 * memory-bound decode term streams resident weights.
 */
function getGenerationWeightBytes(precision: TrainingPrecision): number {
  return precision === "fp32" ? 4 : 2
}

export function getPostTrainingGenerationWeightBytes(
  config: PostTrainingConfig,
): number {
  if (config.approach === "full") {
    return getPostTrainingOptimizerVariant(config).parameterBytes
  }

  if (config.approach === "qlora") {
    const activeWeightBytes = calculateQuantizedActiveModelBytesPerParam(
      config,
      config.lora.quantizationBits ?? 4,
    )

    if (activeWeightBytes !== null) {
      return activeWeightBytes
    }

    const parameterCount = config.baseModel.parameterCount

    if (Number.isFinite(parameterCount) && parameterCount > 0) {
      return (
        calculateQuantizedBaseModelBytes(
          config,
          config.lora.quantizationBits ?? 4,
        ) / parameterCount
      )
    }
  }

  return getGenerationWeightBytes(config.precision)
}

/**
 * Persisted checkpoint bytes per parameter.
 *
 * Training-restart checkpoints store the model parameter tensor plus optimizer
 * state. Gradients are recomputed on resume and are not counted.
 */
function getCheckpointBytesPerParam(config: TrainingConfig): number {
  const optimizerVariant = getOptimizerVariant(config)

  return optimizerVariant.parameterBytes + optimizerVariant.kOpt
}

function getAverageRetainedCheckpointCount(
  checkpointSpan: number,
  retention: number,
): number {
  const retentionCount = Math.floor(retention)

  if (retentionCount <= 0 || checkpointSpan <= 0) {
    return 0
  }

  if (!Number.isFinite(retentionCount)) {
    return Number.POSITIVE_INFINITY
  }

  if (!Number.isFinite(checkpointSpan)) {
    return retentionCount
  }

  const fullCadenceIntervals = Math.floor(checkpointSpan)
  const partialCadenceInterval = checkpointSpan - fullCadenceIntervals
  const rampIntervals = Math.min(fullCadenceIntervals, retentionCount)
  const rampRetainedCountSum = (rampIntervals * (rampIntervals - 1)) / 2
  const steadyRetainedCountSum =
    retentionCount * Math.max(fullCadenceIntervals - retentionCount, 0)
  const partialRetainedCountSum =
    Math.min(fullCadenceIntervals, retentionCount) * partialCadenceInterval

  return (
    (rampRetainedCountSum +
      steadyRetainedCountSum +
      partialRetainedCountSum) /
    checkpointSpan
  )
}

function getStepLimitedCheckpointFrequencyPerDay(
  checkpointFrequencyPerDay: number,
  theoreticalDays: number,
  totalSteps: number | null | undefined,
): number {
  if (
    !Number.isFinite(checkpointFrequencyPerDay) ||
    checkpointFrequencyPerDay <= 0 ||
    !Number.isFinite(theoreticalDays) ||
    theoreticalDays <= 0 ||
    typeof totalSteps !== "number" ||
    !Number.isFinite(totalSteps) ||
    totalSteps <= 0
  ) {
    return checkpointFrequencyPerDay
  }

  const optimizerStepsPerDay = totalSteps / theoreticalDays

  return optimizerStepsPerDay > 0
    ? Math.min(checkpointFrequencyPerDay, optimizerStepsPerDay)
    : checkpointFrequencyPerDay
}

function getStepLimitedCheckpointSpan(
  checkpointSpan: number,
  totalSteps: number | null | undefined,
): number {
  if (
    !Number.isFinite(checkpointSpan) ||
    typeof totalSteps !== "number" ||
    !Number.isFinite(totalSteps) ||
    totalSteps <= 0
  ) {
    return checkpointSpan
  }

  return Math.min(checkpointSpan, totalSteps)
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

export function getEffectiveDefaultTrainingMFU(
  config: TrainingConfig,
  activeParams: number,
  numGPUs: number,
): number {
  return (
    getDefaultMFU(activeParams, numGPUs) *
    calculatePipelineScheduleEfficiency(config) *
    calculateActivationRecomputeMFUFactor(config)
  )
}

export function resolveTrainingMFU(
  config: TrainingConfig,
  activeParams: number,
  numGPUs: number,
): number {
  if (config.mfuOverride === null) {
    return getEffectiveDefaultTrainingMFU(config, activeParams, numGPUs)
  }

  return Number.isFinite(config.mfuOverride) &&
    config.mfuOverride > 0 &&
    config.mfuOverride <= MAX_MFU_OVERRIDE
    ? config.mfuOverride
    : 0
}

export function calculatePipelineScheduleEfficiency(
  config: TrainingConfig,
): number {
  if (hasInvalidManualPipelineTopology(config)) {
    return 0
  }

  const N_pp = normalizeDegree(config.parallelism.N_pp)

  if (N_pp <= 1) {
    return 1
  }

  const numMicrobatches = normalizeDegree(config.gradientAccumulationSteps)
  const usesAFAB =
    config.parallelism.framework === "fsdp" &&
    resolveEffectiveZeroStage(config.parallelism) === 2 &&
    numMicrobatches < 2 * N_pp
  const requestedVP = usesAFAB
    ? 1
    : Math.max(1, normalizeDegree(config.parallelism.VP))
  const VP = canUseInterleavedPipelineSchedule(
    N_pp,
    numMicrobatches,
    requestedVP,
  )
    ? requestedVP
    : 1
  const denominator = VP * numMicrobatches + N_pp - 1

  if (!Number.isFinite(denominator) || denominator <= 0) {
    return 0
  }

  return Math.max(0, Math.min(1, (VP * numMicrobatches) / denominator))
}

export function calculateActivationRecomputeMFUFactor(
  config: TrainingConfig,
): number {
  const arch = config.model.architecture

  if (config.activationCheckpointing === "none") {
    return 1
  }

  if (config.activationCheckpointing === "selective") {
    if (config.flashAttention || arch.d <= 0) {
      return 1
    }

    return 1 / (1 + config.sequenceLength / (6 * arch.d))
  }

  if (config.activationCheckpointing === "partial") {
    if (hasInvalidPartialActivationCheckpointing(config)) {
      return 0
    }

    const N_pp = normalizeDegree(config.parallelism.N_pp)
    const layersPerStage = Math.max(1, Math.ceil(arch.L / N_pp))
    const depth = config.partialCheckpointDepth ?? 0
    const recomputedFraction = Math.min(depth, layersPerStage) / layersPerStage

    return 1 / (1 + recomputedFraction / 3)
  }

  return 0.75
}

function getCPUOffloadBandwidthBytesPerSecond(gpu: GPUSpec): number {
  switch (gpu.interconnect) {
    case "none":
      return 16e9
    case "pcie":
    case "nvlink":
    case "xgmi":
    default:
      // CPU offload traffic traverses the host link, not the GPU-GPU fabric.
      // Use a conservative PCIe Gen4/5-class per-GPU planning bandwidth.
      return 32e9
  }
}

function calculateOffloadComponentEfficiency(
  arithmeticIntensity: number,
  bandwidthBytesPerSecond: number,
  fPeakFLOPS: number,
): number {
  const offloadThroughput = arithmeticIntensity * bandwidthBytesPerSecond
  const denominator = offloadThroughput + fPeakFLOPS

  if (
    !Number.isFinite(offloadThroughput) ||
    offloadThroughput <= 0 ||
    !Number.isFinite(denominator) ||
    denominator <= 0
  ) {
    return 0
  }

  return Math.max(0, Math.min(1, offloadThroughput / denominator))
}

export function calculateCPUOffloadEfficiency(config: TrainingConfig): number {
  if (config.cpuOffload === "none") {
    return 1
  }

  const sequenceLength =
    Number.isFinite(config.sequenceLength) && config.sequenceLength > 0
      ? config.sequenceLength
      : 0
  const microBatchSize =
    Number.isFinite(config.microBatchSize) && config.microBatchSize > 0
      ? config.microBatchSize
      : 0
  const arithmeticIntensity = sequenceLength * microBatchSize
  const optimizerArithmeticIntensity = arithmeticIntensity / 4
  const bandwidthBytesPerSecond = getCPUOffloadBandwidthBytesPerSecond(
    config.hardware.gpu,
  )
  const fPeakFLOPS =
    getEffectiveTrainingTFLOPS(
      config.hardware.gpu,
      config.precision,
      config.fp8,
    ) * 1e12
  const optimizerEfficiency = calculateOffloadComponentEfficiency(
    optimizerArithmeticIntensity,
    bandwidthBytesPerSecond,
    fPeakFLOPS,
  )

  if (config.cpuOffload === "optimizer-only") {
    return optimizerEfficiency
  }

  const parameterEfficiency = calculateOffloadComponentEfficiency(
    arithmeticIntensity,
    bandwidthBytesPerSecond,
    fPeakFLOPS,
  )

  // Parameter+optimizer offload transfers two differently reused state groups.
  // Multiplying the component efficiencies is intentionally conservative and
  // avoids showing parameter offload as no slower than optimizer-only offload.
  return optimizerEfficiency * parameterEfficiency
}

// ---------------------------------------------------------------------------
// calculateFailureAdjustedTime — Section 6.5
// ---------------------------------------------------------------------------

export function calculateFailureAdjustedTime(
  theoreticalDays: number,
  config: TrainingConfig,
  totalSteps?: number,
): FailureAdjustedTime {
  const numGPUs = getTrainingNumGPUs(config)
  const gpusPerNode = isFinitePositiveInteger(config.hardware.gpu.gpusPerNode)
    ? config.hardware.gpu.gpusPerNode
    : null
  const failureRate = getFiniteNonNegative(
    config.failureModel.failureRatePerInstancePerDay,
  )
  const recoveryHours = getFiniteNonNegative(
    config.failureModel.recoveryTimeHours,
  )
  const checkpointFrequency = getFiniteNonNegative(
    config.failureModel.checkpointFrequencyPerDay,
  )

  if (
    failureRate === null ||
    recoveryHours === null ||
    checkpointFrequency === null ||
    gpusPerNode === null
  ) {
    return infiniteFailureAdjustedTime()
  }

  if (failureRate === 0) {
    const adjustedDays =
      Number.isFinite(theoreticalDays) && theoreticalDays >= 0
        ? theoreticalDays
        : Number.POSITIVE_INFINITY

    return {
      adjustedDays,
      adjustedHours: adjustedDays * 24,
      multiplier: 1,
    }
  }

  const checkpointRetention = normalizeNonNegativeCount(
    config.pricing.checkpointRetentionCount,
  )

  if (checkpointRetention === null) {
    return infiniteFailureAdjustedTime()
  }

  const effectiveCheckpointFrequency = getStepLimitedCheckpointFrequencyPerDay(
    checkpointFrequency,
    theoreticalDays,
    totalSteps,
  )
  const nInstances = Math.ceil(numGPUs / gpusPerNode)
  const recoveryDays = recoveryHours / 24
  const averageLostWorkDays =
    effectiveCheckpointFrequency > 0 && checkpointRetention > 0
      ? 1 / (2 * effectiveCheckpointFrequency)
      : Number.POSITIVE_INFINITY
  const denominator =
    1 - failureRate * nInstances * (recoveryDays + averageLostWorkDays)

  if (!Number.isFinite(denominator) || denominator <= 0) {
    return infiniteFailureAdjustedTime()
  }

  const multiplier = 1 / denominator

  if (!Number.isFinite(theoreticalDays) || theoreticalDays < 0) {
    return {
      adjustedDays: Number.POSITIVE_INFINITY,
      adjustedHours: Number.POSITIVE_INFINITY,
      multiplier,
    }
  }

  const adjustedDays = theoreticalDays * multiplier

  return {
    adjustedDays,
    adjustedHours: adjustedDays * 24,
    multiplier,
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
  const mfu = resolveTrainingMFU(config, activeParams, numGPUs)
  const hasInvalidManualParallelism = hasInvalidManualParallelismDegrees(config)
  const hasInvalidComputeShape =
    !Number.isFinite(activeParams) ||
    activeParams <= 0 ||
    !Number.isFinite(compute.totalFLOPs) ||
    compute.totalFLOPs <= 0 ||
    !Number.isFinite(compute.flopsPerToken) ||
    compute.flopsPerToken <= 0
  const hasInvalidBatchShape =
    hasInvalidManualParallelism ||
    hasInvalidComputeShape ||
    hasInvalidTrainingHardware(
      config.hardware.inputMode,
      gpu,
      config.precision,
    ) ||
    hasInvalidManualTensorParallelismTopology(config) ||
    hasInvalidManualTensorExpertSequenceParallelismTopology(config) ||
    hasInvalidManualContextParallelismTopology(config) ||
    hasInvalidManualExpertParallelismTopology(config) ||
    hasInvalidManualPipelineTopology(config) ||
    hasInvalidCPUOffloadConfig(config) ||
    hasInvalidPretrainingOptimizer(config.optimizer) ||
    hasInvalidTrainingGPUCount(config) ||
    hasInvalidPartialActivationCheckpointing(config) ||
    hasInvalidFP8Config(config) ||
    !isFinitePositiveInteger(config.totalTokens) ||
    !isFinitePositiveInteger(config.microBatchSize) ||
    !isFinitePositiveInteger(config.gradientAccumulationSteps) ||
    !isFinitePositiveInteger(config.sequenceLength)
  // MFU is the single wall-clock efficiency knob. Pipeline bubbles,
  // communication, checkpointing, and offload stalls are displayed separately
  // as guidance, but should not be stacked as extra time multipliers.
  const denominator = numGPUs * fPeakFLOPS * mfu
  const theoreticalSeconds =
    !hasInvalidBatchShape && denominator > 0
      ? compute.totalFLOPs / denominator
      : Number.POSITIVE_INFINITY
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
  const microBatchSize = hasInvalidBatchShape
    ? Number.POSITIVE_INFINITY
    : normalizeDegree(config.microBatchSize)
  const gradientAccumulationSteps = hasInvalidBatchShape
    ? Number.POSITIVE_INFINITY
    : normalizeDegree(config.gradientAccumulationSteps)
  const globalBatchTokens = hasInvalidBatchShape
    ? Number.POSITIVE_INFINITY
    : microBatchSize *
      config.sequenceLength *
      gradientAccumulationSteps *
      dataParallelDegree
  const totalSteps =
    hasInvalidBatchShape
      ? Number.POSITIVE_INFINITY
      : globalBatchTokens > 0
        ? Math.ceil(totalTokens / globalBatchTokens)
        : 0
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
  const failureAdjusted = shouldSurfaceFailureAdjustedTime(numGPUs)
    ? calculateFailureAdjustedTime(theoreticalDays, config, totalSteps)
    : null

  return {
    theoreticalDays,
    theoreticalHours,
    failureAdjustedDays: failureAdjusted?.adjustedDays ?? null,
    failureAdjustedHours: failureAdjusted?.adjustedHours ?? null,
    failureMultiplier: failureAdjusted?.multiplier ?? null,
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
  const costPerGPUHour = getFiniteNonNegative(pricing.costPerGPUHour)
  const storagePricePerGBMonth = getFiniteNonNegative(
    pricing.storagePricePerGBMonth,
  )
  const datasetStorageGB = getFiniteNonNegative(pricing.datasetStorageGB)
  const checkpointFrequency = getFiniteNonNegative(
    config.failureModel.checkpointFrequencyPerDay,
  )
  const hasInvalidCheckpointFrequency = checkpointFrequency === null
  const checkpointsDisabled = checkpointFrequency === 0
  const retention = normalizeNonNegativeCount(pricing.checkpointRetentionCount)
  const totalParams =
    totalParamsOverride ?? getTrainingParameterCounts(config).total
  const hasInvalidTime =
    !Number.isFinite(time.theoreticalDays) ||
    time.theoreticalDays < 0 ||
    !Number.isFinite(time.theoreticalHours) ||
    time.theoreticalHours < 0 ||
    !isFinitePositiveInteger(time.totalSteps) ||
    (time.failureAdjustedDays !== null &&
      (Number.isNaN(time.failureAdjustedDays) ||
        time.failureAdjustedDays < 0)) ||
    (time.failureAdjustedHours !== null &&
      (Number.isNaN(time.failureAdjustedHours) ||
        time.failureAdjustedHours < 0))

  if (
    hasInvalidTime ||
    costPerGPUHour === null ||
    storagePricePerGBMonth === null ||
    datasetStorageGB === null ||
    checkpointFrequency === null ||
    retention === null ||
    !Number.isFinite(totalParams) ||
    totalParams <= 0 ||
    hasInvalidTrainingGPUCount(config) ||
    hasInvalidTrainingHardware(
      config.hardware.inputMode,
      config.hardware.gpu,
      config.precision,
    ) ||
    hasInvalidPretrainingOptimizer(config.optimizer) ||
    hasInvalidFP8StorageMode(config)
  ) {
    return invalidCostEstimate()
  }

  const failureAdjusted =
    time.failureAdjustedDays !== null && time.failureAdjustedHours !== null
      ? {
          adjustedDays: time.failureAdjustedDays,
          adjustedHours: time.failureAdjustedHours,
        }
      : null
  const computeCost = calculateGPUHourlyCost(
    numGPUs,
    time.theoreticalHours,
    costPerGPUHour,
  )
  const actualComputeCost =
    failureAdjusted !== null
      ? calculateGPUHourlyCost(
          numGPUs,
          failureAdjusted.adjustedHours,
          costPerGPUHour,
        )
      : null
  const failureOverheadCost =
    actualComputeCost === null || actualComputeCost === computeCost
      ? 0
      : !Number.isFinite(actualComputeCost) || !Number.isFinite(computeCost)
        ? Number.POSITIVE_INFINITY
        : Math.max(actualComputeCost - computeCost, 0)
  const checkpointSize =
    getCheckpointBytesPerParam(config) *
    totalParams *
    CHECKPOINT_FILE_OVERHEAD_FACTOR
  // Follow calculateTrainingTime's failure-adjustment gate; small runs use the
  // theoretical duration for checkpoint storage and report no failure overhead.
  const storageDurationDays = failureAdjusted?.adjustedDays ?? time.theoreticalDays
  const checkpointSpan =
    hasInvalidCheckpointFrequency
      ? Number.POSITIVE_INFINITY
      : checkpointsDisabled
      ? 0
      : multiplyFactors(storageDurationDays, checkpointFrequency)
  const stepLimitedCheckpointSpan = getStepLimitedCheckpointSpan(
    checkpointSpan,
    time.totalSteps,
  )
  const numCheckpoints =
    hasInvalidCheckpointFrequency
      ? Number.POSITIVE_INFINITY
      : checkpointsDisabled
      ? 0
      : stepLimitedCheckpointSpan > 0
      ? Math.ceil(stepLimitedCheckpointSpan)
      : 0
  const peakCheckpointStorage =
    hasInvalidCheckpointFrequency || retention === null
      ? Number.POSITIVE_INFINITY
      : Math.min(numCheckpoints, retention) * checkpointSize

  const avgCheckpointCount =
    hasInvalidCheckpointFrequency || retention === null
      ? Number.POSITIVE_INFINITY
      : checkpointsDisabled
      ? 0
      : getAverageRetainedCheckpointCount(stepLimitedCheckpointSpan, retention)
  const averageCheckpointStorage = avgCheckpointCount * checkpointSize
  const averageCheckpointStorageGB = averageCheckpointStorage / 1e9
  const datasetStorageBytes =
    datasetStorageGB === null ? Number.POSITIVE_INFINITY : datasetStorageGB * 1e9
  const billableStorageGB =
    datasetStorageGB === null
      ? Number.POSITIVE_INFINITY
      : averageCheckpointStorageGB + datasetStorageGB
  const runDurationMonths = storageDurationDays / 30.25
  const storageCost =
    storagePricePerGBMonth === null ||
    datasetStorageGB === null ||
    hasInvalidCheckpointFrequency
      ? Number.POSITIVE_INFINITY
      : billableStorageGB > 0 && runDurationMonths > 0
      ? multiplyFactors(
          storagePricePerGBMonth,
          billableStorageGB,
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
    datasetStorageBytes,
  }
}

// ---------------------------------------------------------------------------
// calculatePostTrainingCompute — Section 10.5
// ---------------------------------------------------------------------------

function getPostTrainingTrainableFraction(config: PostTrainingConfig): number {
  if (config.approach !== "full") {
    return 1
  }

  const percentage = config.trainableParameterPercentage
  if (percentage === null) {
    return 1
  }

  return Number.isFinite(percentage) && percentage > 0 && percentage <= 100
    ? percentage / 100
    : Number.POSITIVE_INFINITY
}

function hasInvalidPostTrainingTrainablePercentage(
  config: PostTrainingConfig,
): boolean {
  const percentage = config.trainableParameterPercentage

  return (
    (config.approach === "full" || config.approach === "mezo") &&
    percentage !== null &&
    (!Number.isFinite(percentage) || percentage <= 0 || percentage > 100)
  )
}

function hasInvalidPostTrainingLoRATargets(config: PostTrainingConfig): boolean {
  return (
    (config.approach === "lora" || config.approach === "qlora") &&
    hasInvalidLoRATargetModules(config.lora)
  )
}

function hasInvalidPostTrainingMethodConfig(config: PostTrainingConfig): boolean {
  if (config.method === "grpo") {
    return resolveGRPOGroupSize(config) === Number.POSITIVE_INFINITY
  }

  if (config.method === "ppo") {
    return (
      getFinitePositiveInteger(config.ppo.criticModelParameterCount) === null ||
      getFinitePositiveInteger(config.ppo.rewardModelParameterCount) === null ||
      resolvePPOUpdateEpochs(config) === Number.POSITIVE_INFINITY
    )
  }

  return false
}

function getPostTrainingAttentionProjectionWidth(
  config: PostTrainingConfig,
): number {
  const arch = config.baseModel.architecture
  const explicitHeadDim = arch.d_head

  if (explicitHeadDim !== null && explicitHeadDim !== undefined) {
    return Number.isFinite(explicitHeadDim) &&
      explicitHeadDim > 0 &&
      Number.isInteger(explicitHeadDim)
      ? arch.a * explicitHeadDim
      : Number.POSITIVE_INFINITY
  }

  return arch.d
}

function getPostTrainingAttentionForwardFLOPsPerToken(
  config: PostTrainingConfig,
): number {
  return getPostTrainingAttentionForwardFLOPsForContext(
    config,
    config.sequenceLength,
  )
}

function getPostTrainingAttentionForwardFLOPsForContext(
  config: PostTrainingConfig,
  contextLength: number,
): number {
  const arch = config.baseModel.architecture
  const attentionProjectionWidth =
    getPostTrainingAttentionProjectionWidth(config)

  return Number.isFinite(arch.L) &&
    arch.L > 0 &&
    Number.isFinite(attentionProjectionWidth) &&
    attentionProjectionWidth > 0 &&
    Number.isFinite(contextLength) &&
    contextLength >= 0
    ? 4 * arch.L * attentionProjectionWidth * contextLength
    : Number.POSITIVE_INFINITY
}

function scalePostTrainingAttentionFLOPs(
  attentionFLOPsPerToken: number,
  modelParams: number,
  policyParams: number,
): number {
  if (!Number.isFinite(attentionFLOPsPerToken)) {
    return attentionFLOPsPerToken
  }

  if (
    !Number.isFinite(modelParams) ||
    !Number.isFinite(policyParams) ||
    modelParams <= 0 ||
    policyParams <= 0
  ) {
    return attentionFLOPsPerToken
  }

  return attentionFLOPsPerToken * Math.max(0, modelParams / policyParams)
}

function getPolicyTrainingAttentionFLOPsPerToken(
  attentionForwardFLOPsPerToken: number,
  config: PostTrainingConfig,
): number {
  if (config.approach === "mezo") {
    return 2 * attentionForwardFLOPsPerToken
  }

  if (config.approach === "lora" || config.approach === "qlora") {
    // LoRA skips frozen base weight gradients, but still backpropagates through
    // the full attention computation to form adapter gradients.
    return 3 * attentionForwardFLOPsPerToken
  }

  return (
    (1 + 2 * getPostTrainingTrainableFraction(config)) *
    attentionForwardFLOPsPerToken
  )
}

function estimateLoRAAdapterParameterCount(
  params: number,
  config: PostTrainingConfig,
): number {
  if (config.approach !== "lora" && config.approach !== "qlora") {
    return 0
  }

  if (
    hasInvalidQLoRAQuantizationBits(config) ||
    hasInvalidLoRATargetModules(config.lora) ||
    !Number.isFinite(config.lora.rank) ||
    config.lora.rank < 1 ||
    !Number.isInteger(config.lora.rank)
  ) {
    return Number.POSITIVE_INFINITY
  }

  const moe = config.baseModel.moe
  const activeMoe = moe.enabled
    ? {
        ...moe,
        E:
          Math.min(Math.max(0, moe.topk), Math.max(0, moe.E)) *
            (Number.isFinite(moe.loadBalanceFactor)
              ? Math.max(1, moe.loadBalanceFactor)
              : 1) +
          Math.max(0, moe.E_s),
        E_s: 0,
      }
    : moe
  const adapterParams = calculateLoRAParamCountForArchitecture(
    config.baseModel.architecture,
    activeMoe,
    config.lora,
  )

  if (Number.isFinite(adapterParams) && adapterParams > 0) {
    return adapterParams
  }

  return Number.POSITIVE_INFINITY
}

function estimatePostTrainingActiveRoutedExpertParameterCount(
  params: number,
  config: PostTrainingConfig,
): number {
  const moe = config.baseModel.moe

  if (!moe.enabled) {
    return 0
  }

  const counts = calculateParameterCount(
    config.baseModel.architecture,
    moe,
    config.sequenceLength,
  )
  const activeRoutedExpertParameters =
    counts.moe?.activeRoutedExpertParameters ?? 0

  if (
    !Number.isFinite(activeRoutedExpertParameters) ||
    activeRoutedExpertParameters <= 0
  ) {
    return 0
  }

  if (!Number.isFinite(params)) {
    return params > 0 ? Number.POSITIVE_INFINITY : 0
  }

  if (!Number.isFinite(counts.active) || counts.active <= 0 || params <= 0) {
    return 0
  }

  return activeRoutedExpertParameters * (params / counts.active)
}

export function estimatePostTrainingMoELoadBalanceFLOPsPerToken(
  params: number,
  config: PostTrainingConfig,
  passCoefficient: number,
): number {
  if (hasInvalidPostTrainingModelShape(config)) {
    return Number.POSITIVE_INFINITY
  }

  const loadBalanceFactor = Number.isFinite(config.baseModel.moe.loadBalanceFactor)
    ? Math.max(1, config.baseModel.moe.loadBalanceFactor)
    : 1

  if (
    loadBalanceFactor <= 1 ||
    !Number.isFinite(passCoefficient) ||
    passCoefficient <= 0
  ) {
    return 0
  }

  const activeRoutedExpertParameters =
    estimatePostTrainingActiveRoutedExpertParameterCount(params, config)

  return Number.isFinite(activeRoutedExpertParameters)
    ? passCoefficient *
        activeRoutedExpertParameters *
        (loadBalanceFactor - 1)
    : Number.POSITIVE_INFINITY
}

function getPolicyTrainingFLOPsPerToken(
  params: number,
  config: PostTrainingConfig,
): number {
  if (config.approach === "lora" || config.approach === "qlora") {
    const adapterParams = estimateLoRAAdapterParameterCount(params, config)

    return (
      4 * params +
      estimatePostTrainingMoELoadBalanceFLOPsPerToken(params, config, 4) +
      6 * adapterParams
    )
  }

  const trainableFraction = getPostTrainingTrainableFraction(config)
  const passCoefficient = 2 + 4 * trainableFraction

  return (
    2 * params +
    4 * params * trainableFraction +
    estimatePostTrainingMoELoadBalanceFLOPsPerToken(
      params,
      config,
      passCoefficient,
    )
  )
}

function resolveGRPOGroupSize(config: PostTrainingConfig): number {
  return Number.isFinite(config.grpo.groupSize) &&
    config.grpo.groupSize >= 2 &&
    Number.isInteger(config.grpo.groupSize)
    ? config.grpo.groupSize
    : Number.POSITIVE_INFINITY
}

function resolvePPOUpdateEpochs(config: PostTrainingConfig): number {
  return Number.isFinite(config.ppo.updateEpochs) &&
    config.ppo.updateEpochs >= 1 &&
    Number.isInteger(config.ppo.updateEpochs)
    ? config.ppo.updateEpochs
    : Number.POSITIVE_INFINITY
}

export function calculatePostTrainingCompute(
  method: PostTrainingMethod,
  params: number,
  config: PostTrainingConfig,
): { totalFLOPs: number; flopsPerToken: number; totalTokens: number } {
  if (
    hasInvalidPostTrainingMethodApproach(method, config.approach) ||
    hasInvalidPostTrainingModelShape(config) ||
    hasInvalidPostTrainingApproachConfig(config) ||
    hasInvalidPostTrainingMethodConfig(config) ||
    hasInvalidQLoRAQuantizationBits(config) ||
    hasInvalidPostTrainingLoRATargets(config)
  ) {
    return {
      totalFLOPs: Number.POSITIVE_INFINITY,
      flopsPerToken: Number.POSITIVE_INFINITY,
      totalTokens: Number.POSITIVE_INFINITY,
    }
  }

  const policyParams = getFinitePositiveOrInfinity(params)
  const ppoUpdateEpochs = resolvePPOUpdateEpochs(config)
  const ppoRewardParams = getFinitePositiveIntegerOrInfinity(
    config.ppo.rewardModelParameterCount,
  )
  const ppoCriticParams = getFinitePositiveIntegerOrInfinity(
    config.ppo.criticModelParameterCount,
  )
  const policyAttentionForwardFLOPs =
    getPostTrainingAttentionForwardFLOPsPerToken(config)
  const policyTrainingAttentionFLOPs =
    getPolicyTrainingAttentionFLOPsPerToken(policyAttentionForwardFLOPs, config)
  const ppoRewardAttentionForwardFLOPs = scalePostTrainingAttentionFLOPs(
    policyAttentionForwardFLOPs,
    ppoRewardParams,
    policyParams,
  )
  const ppoCriticTrainingAttentionFLOPs =
    3 *
    scalePostTrainingAttentionFLOPs(
      policyAttentionForwardFLOPs,
      ppoCriticParams,
      policyParams,
    )
  const policyForwardMoELoadBalance =
    estimatePostTrainingMoELoadBalanceFLOPsPerToken(policyParams, config, 2)
  const policyGenerationFLOPs =
    2 * policyParams + policyForwardMoELoadBalance
  const flopsPerToken =
    hasInvalidPostTrainingTrainablePercentage(config)
      ? Number.POSITIVE_INFINITY
      : method === "sft" && config.approach === "mezo"
      ? // MeZO estimates updates from forward-pass perturbations, not backward.
        // A symmetric finite-difference step evaluates two forward passes.
        4 * policyParams +
        estimatePostTrainingMoELoadBalanceFLOPsPerToken(
          policyParams,
          config,
          4,
        ) +
        policyTrainingAttentionFLOPs
      : method === "sft"
      ? getPolicyTrainingFLOPsPerToken(policyParams, config) +
        policyTrainingAttentionFLOPs
      : method === "dpo"
      ? // Policy train pass plus frozen reference forward pass.
        getPolicyTrainingFLOPsPerToken(policyParams, config) +
        policyTrainingAttentionFLOPs +
        2 * policyParams +
        policyForwardMoELoadBalance +
        policyAttentionForwardFLOPs
      : method === "ppo"
      ? // Section 10.3 phases: generation + reward scoring once, then K
        // PPO update epochs over policy, critic, and reference/KL minibatches.
        policyGenerationFLOPs +
        2 * ppoRewardParams +
        ppoRewardAttentionForwardFLOPs +
        ppoUpdateEpochs *
          (getPolicyTrainingFLOPsPerToken(policyParams, config) +
            policyTrainingAttentionFLOPs +
            6 * ppoCriticParams +
            ppoCriticTrainingAttentionFLOPs +
            2 * policyParams +
            policyForwardMoELoadBalance +
            policyAttentionForwardFLOPs)
      : // GRPO: generation + policy update + frozen reference scoring.
        policyGenerationFLOPs +
        getPolicyTrainingFLOPsPerToken(policyParams, config) +
        policyTrainingAttentionFLOPs +
        2 * policyParams +
        policyForwardMoELoadBalance +
        policyAttentionForwardFLOPs
  const tokenMultiplier =
    method === "dpo"
      ? 2
      : method === "grpo"
        ? resolveGRPOGroupSize(config)
        : 1
  const datasetSizeExamples = getFinitePositiveInteger(
    config.datasetSizeExamples,
  )
  const epochs = getFinitePositive(config.epochs)
  const sequenceLength = getFinitePositiveInteger(config.sequenceLength)
  const totalTokens =
    datasetSizeExamples !== null &&
    epochs !== null &&
    sequenceLength !== null &&
    Number.isFinite(tokenMultiplier) &&
    tokenMultiplier > 0
      ? datasetSizeExamples * epochs * sequenceLength * tokenMultiplier
      : Number.POSITIVE_INFINITY

  return {
    totalFLOPs:
      Number.isFinite(totalTokens) && Number.isFinite(flopsPerToken)
        ? totalTokens * flopsPerToken
        : Number.POSITIVE_INFINITY,
    flopsPerToken,
    totalTokens,
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
  const precision = usingConfig
    ? configOrGPU.precision
    : (precisionOrNTokens as TrainingPrecision)
  if (
    usingConfig &&
    (hasInvalidPostTrainingGPUCount(configOrGPU) ||
      hasInvalidPostTrainingOptimizer(configOrGPU.optimizer) ||
      hasInvalidPostTrainingModelShape(configOrGPU) ||
      hasInvalidPostTrainingApproachConfig(configOrGPU) ||
      hasInvalidPostTrainingMethodConfig(configOrGPU) ||
      hasInvalidQLoRAQuantizationBits(configOrGPU) ||
      hasInvalidPostTrainingLoRATargets(configOrGPU) ||
      hasInvalidPostTrainingTrainablePercentage(configOrGPU) ||
      hasInvalidPostTrainingKVCachePrecision(configOrGPU) ||
      hasInvalidFP8Config(configOrGPU) ||
      hasInvalidTrainingHardware(
        configOrGPU.hardware.inputMode,
        gpu,
        precision,
      ))
  ) {
    return invalidGenerationTime()
  }

  const rawConfiguredNumGPUs = usingConfig
    ? getPostTrainingNumGPUs(configOrGPU)
    : numGPUsOrBatchGen
  const rawBatchGen = usingConfig
    ? numGPUsOrBatchGen
    : batchGenOrPrompt
  const rawNTokens = usingConfig
    ? (precisionOrNTokens as number)
    : nTokensMaybe
  const rawSPrompt = usingConfig ? batchGenOrPrompt : sPromptMaybe

  if (
    !Number.isFinite(params) ||
    params < 0 ||
    !Number.isFinite(rawConfiguredNumGPUs) ||
    rawConfiguredNumGPUs <= 0 ||
    !Number.isFinite(rawBatchGen) ||
    rawBatchGen < 0 ||
    typeof rawNTokens !== "number" ||
    !Number.isFinite(rawNTokens) ||
    rawNTokens < 0 ||
    typeof rawSPrompt !== "number" ||
    !Number.isFinite(rawSPrompt) ||
    rawSPrompt < 0
  ) {
    return invalidGenerationTime()
  }

  const configuredNumGPUs =
    Number.isFinite(rawConfiguredNumGPUs) && rawConfiguredNumGPUs > 0
      ? rawConfiguredNumGPUs
      : 1
  const batchGen = getFiniteNonNegativeOrZero(rawBatchGen)
  const numGPUs =
    batchGen > 0
      ? Math.min(configuredNumGPUs, Math.max(1, Math.ceil(batchGen)))
      : configuredNumGPUs
  const localBatchGen =
    batchGen > 0 && numGPUs > 0 ? Math.ceil(batchGen / numGPUs) : 0
  const nTokens = getFiniteNonNegativeOrZero(rawNTokens)
  const sPrompt = getFiniteNonNegativeOrZero(rawSPrompt)
  const parameterCount = getFiniteNonNegativeOrInfinity(params)
  const fPeakTFLOPS = getEffectiveGenerationTFLOPS(
    gpu,
    precision,
    usingConfig ? configOrGPU.fp8 : undefined,
  )
  const fPeakFLOPS =
    Number.isFinite(fPeakTFLOPS) && fPeakTFLOPS > 0
      ? fPeakTFLOPS * 1e12
      : 0
  const bwMemBps =
    Number.isFinite(gpu.memoryBandwidthGBps) && gpu.memoryBandwidthGBps > 0
      ? gpu.memoryBandwidthGBps * 1e9 * 0.9
      : 0
  const weightBytes = usingConfig
    ? getPostTrainingGenerationWeightBytes(configOrGPU)
    : getGenerationWeightBytes(precision)
  const routedExpertOverheadFLOPsPerToken = usingConfig
    ? estimatePostTrainingMoELoadBalanceFLOPsPerToken(
        parameterCount,
        configOrGPU,
        2,
      )
    : 0
  const modelFLOPsPerToken =
    2 * parameterCount + routedExpertOverheadFLOPsPerToken

  // Section 10.3 gives the prefill term for one prompt. With data-parallel
  // replicas, wall-clock prefill is set by the fullest local batch, not by a
  // fractional global batch/G average.
  const prefillAttentionFLOPs =
    usingConfig && sPrompt > 0
      ? multiplyFactors(
          getPostTrainingAttentionForwardFLOPsForContext(
            configOrGPU,
            sPrompt,
          ),
          sPrompt,
          localBatchGen,
        )
      : 0
  const prefillSeconds = divideWork(
    multiplyFactors(modelFLOPsPerToken, sPrompt, localBatchGen) +
      prefillAttentionFLOPs,
    fPeakFLOPS,
  )
  // Each data-parallel GPU owns a full model replica and streams its local
  // weights once per decode token. More replicas increase throughput by serving
  // more sequences concurrently, but they do not reduce the per-replica
  // memory-bound latency for a fixed decode step.
  const memoryBoundPerToken =
    localBatchGen > 0
      ? divideWork(multiplyFactors(parameterCount, weightBytes), bwMemBps)
      : 0
  const averageDecodeContextLength =
    nTokens > 0 ? sPrompt + (nTokens + 1) / 2 : 0
  const decodeAttentionFLOPsPerToken =
    usingConfig && localBatchGen > 0 && nTokens > 0
      ? multiplyFactors(
          getPostTrainingAttentionForwardFLOPsForContext(
            configOrGPU,
            averageDecodeContextLength,
          ),
          localBatchGen,
        )
      : 0
  const computeBoundPerToken = divideWork(
    multiplyFactors(modelFLOPsPerToken, localBatchGen) +
      decodeAttentionFLOPsPerToken,
    fPeakFLOPS,
  )
  const decodePerToken = Math.max(memoryBoundPerToken, computeBoundPerToken)
  const decodeSeconds = multiplyFactors(nTokens, decodePerToken)

  return {
    prefillSeconds,
    decodeSeconds,
    totalSeconds: prefillSeconds + decodeSeconds,
    isMemoryBound: memoryBoundPerToken >= computeBoundPerToken,
  }
}
