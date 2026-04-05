import type {
  CheckpointingMode,
  CPUOffloadMode,
  FSDPStrategy,
  GPUSpec,
  GradientPrecision,
  MemoryBreakdown,
  ModelArchitecture,
  MoEConfig,
  OptimizerType,
  ParameterCounts,
  ParallelismConfig,
  PostTrainingConfig,
  PostTrainingMemoryBreakdown,
  PostTrainingModelMemoryLineItem,
  TrainingConfig,
  ZeROStage,
} from "../types"
import { OPTIMIZER_PROFILES } from "../constants"

export interface OptimizerValues {
  phi: number
  kOpt: number
  betaGrad: number
  parameterBytes: number
  masterWeightBytes: number
  optimizerStateBytes: number
}

export function getOptimizerProfile(
  optimizer: OptimizerType,
  gradPrecision: GradientPrecision
): OptimizerValues {
  const profile = OPTIMIZER_PROFILES.find((candidate) => candidate.id === optimizer)

  if (!profile) {
    throw new Error(`Unknown optimizer: ${optimizer}`)
  }

  const variant = gradPrecision === "bf16" ? profile.bf16Grad : profile.fp32Grad

  return {
    phi: variant.phi,
    kOpt: variant.kOpt,
    betaGrad: variant.betaGrad,
    parameterBytes: variant.parameterBytes,
    masterWeightBytes: variant.masterWeightBytes,
    optimizerStateBytes: variant.optimizerStateBytes,
  }
}

export interface ModelStateMemoryResult {
  parameters: number
  gradients: number
  optimizerStates: number
  total: number
}

interface ActivationCoefficients {
  nonFFNLinear: number
  ffnLinear: number
  attentionQuadratic: number
}

interface ParameterPartitioning {
  embeddingTotal: number
  embeddingLocal: number
  nonExpertLocal: number
  routedExpertLocal: number
}

interface PostTrainingFinalizeArgs {
  gpu: GPUSpec
  parameters: number
  gradients: number
  optimizerStates: number
  activations: number
  communicationBuffers: number
  frameworkOverhead: number
  peakWorkingSet: number
  trainableModels: number
  frozenModels: number
  loraAdapter: number
  ppoBuffers: number
  items: PostTrainingModelMemoryLineItem[]
}

const MEGATRON_STYLE_OVERHEAD_BYTES = 5e9
const LIGHTWEIGHT_OVERHEAD_BYTES = 2e9
const DEFAULT_POST_TRAINING_OVERHEAD_BYTES = 1e9

function clampDegree(value: number): number {
  return Math.max(1, value)
}

function getTrainingComputeBytes(config: TrainingConfig): number {
  if (config.precision === "fp32") {
    return 4
  }

  if (config.precision === "fp8") {
    return config.fp8.storageMode === "ms-amp" ? 1 : 2
  }

  return 2
}

function getPostTrainingWeightBytes(config: PostTrainingConfig): number {
  return config.precision === "fp32" ? 4 : 2
}

function getPostTrainingActivationBytes(config: PostTrainingConfig): number {
  return config.precision === "fp32" ? 4 : 2
}

function getKVCacheBytesPerElement(precision: PostTrainingConfig["kvCachePrecision"]): number {
  return precision === "int8" ? 1 : 2
}

function resolveTrainingOptimizerProfile(config: TrainingConfig): OptimizerValues {
  if (
    config.optimizer === "adamw-fp8" &&
    config.fp8.storageMode === "transformer-engine"
  ) {
    return getOptimizerProfile("adamw-mixed", config.gradientPrecision)
  }

  return getOptimizerProfile(config.optimizer, config.gradientPrecision)
}

function resolveZeROStage(config: TrainingConfig): ZeROStage {
  const strategy = config.parallelism.fsdpStrategy

  if (strategy === null) {
    return config.parallelism.zeroStage
  }

  const mapping: Record<FSDPStrategy, ZeROStage> = {
    NO_SHARD: 0,
    SHARD_GRAD_OP: 2,
    FULL_SHARD: 3,
    HYBRID_SHARD: 3,
    HYBRID_SHARD_ZERO2: 2,
  }

  return mapping[strategy]
}

function usesHybridShard(config: TrainingConfig): boolean {
  return (
    config.parallelism.fsdpStrategy === "HYBRID_SHARD" ||
    config.parallelism.fsdpStrategy === "HYBRID_SHARD_ZERO2"
  )
}

function isSequenceParallelEnabled(parallelism: ParallelismConfig): boolean {
  if (parallelism.sequenceParallelism === "enabled") {
    return true
  }

  if (parallelism.sequenceParallelism === "disabled") {
    return false
  }

  return parallelism.N_tp > 1
}

function usesSequenceParallelOptimizerSharding(config: TrainingConfig): boolean {
  if (config.parallelism.sequenceParallelism === "enabled") {
    return true
  }

  if (config.parallelism.sequenceParallelism === "disabled") {
    return false
  }

  // Be conservative in "auto" mode. Megatron-LM commonly shards optimizer
  // state across TP/SP ranks; other frameworks do not always do so.
  return (
    config.parallelism.framework === "megatron" &&
    clampDegree(config.parallelism.N_tp) > 1
  )
}

function getStateShardDegree(config: TrainingConfig): number {
  return usesHybridShard(config)
    ? clampDegree(config.hardware.gpu.gpusPerNode)
    : clampDegree(config.parallelism.N_dp)
}

function getNonExpertOptimizerShardDegree(config: TrainingConfig): number {
  return (
    getStateShardDegree(config) *
    (usesSequenceParallelOptimizerSharding(config)
      ? clampDegree(config.parallelism.N_tp)
      : 1)
  )
}

function applyCPUOffload(
  memory: ModelStateMemoryResult,
  cpuOffload: CPUOffloadMode,
  zeroStage: ZeROStage
): ModelStateMemoryResult {
  if (cpuOffload === "optimizer-only") {
    return {
      parameters: memory.parameters,
      gradients: memory.gradients,
      optimizerStates: 0,
      total: memory.parameters + memory.gradients,
    }
  }

  if (cpuOffload === "optimizer-and-params" && zeroStage === 3) {
    return {
      parameters: 0,
      gradients: 0,
      optimizerStates: 0,
      total: 0,
    }
  }

  return memory
}

function addModelStateMemory(
  left: ModelStateMemoryResult,
  right: ModelStateMemoryResult
): ModelStateMemoryResult {
  return {
    parameters: left.parameters + right.parameters,
    gradients: left.gradients + right.gradients,
    optimizerStates: left.optimizerStates + right.optimizerStates,
    total: left.total + right.total,
  }
}

function calculateStateGroupMemory(
  parameterCount: number,
  optimizer: OptimizerValues,
  zeroStage: ZeROStage,
  stateShardDegree: number,
  optimizerShardDegree: number
): ModelStateMemoryResult {
  const stateDegree = clampDegree(stateShardDegree)
  const optimizerDegree = clampDegree(optimizerShardDegree)

  if (parameterCount <= 0) {
    return {
      parameters: 0,
      gradients: 0,
      optimizerStates: 0,
      total: 0,
    }
  }

  switch (zeroStage) {
    case 1: {
      const parameters = parameterCount * optimizer.parameterBytes
      const gradients = parameterCount * optimizer.betaGrad
      const optimizerStates = (parameterCount * optimizer.kOpt) / optimizerDegree

      return {
        parameters,
        gradients,
        optimizerStates,
        total: parameters + gradients + optimizerStates,
      }
    }
    case 2: {
      const parameters = parameterCount * optimizer.parameterBytes
      const gradients = (parameterCount * optimizer.betaGrad) / stateDegree
      const optimizerStates = (parameterCount * optimizer.kOpt) / optimizerDegree

      return {
        parameters,
        gradients,
        optimizerStates,
        total: parameters + gradients + optimizerStates,
      }
    }
    case 3: {
      const parameters = (parameterCount * optimizer.parameterBytes) / stateDegree
      const gradients = (parameterCount * optimizer.betaGrad) / stateDegree
      const optimizerStates = (parameterCount * optimizer.kOpt) / optimizerDegree

      return {
        parameters,
        gradients,
        optimizerStates,
        total: parameters + gradients + optimizerStates,
      }
    }
    case 0:
    default: {
      const parameters = parameterCount * optimizer.parameterBytes
      const gradients = parameterCount * optimizer.betaGrad
      const optimizerStates = parameterCount * optimizer.kOpt

      return {
        parameters,
        gradients,
        optimizerStates,
        total: parameters + gradients + optimizerStates,
      }
    }
  }
}

function getEmbeddingParameterCount(params: ParameterCounts): number {
  return (
    params.embedding +
    params.outputProjection +
    params.positionalEmbedding +
    params.finalNorm
  )
}

function getParameterPartitioning(
  params: ParameterCounts,
  config: TrainingConfig
): ParameterPartitioning {
  const N_tp = clampDegree(config.parallelism.N_tp)
  const N_pp = clampDegree(config.parallelism.N_pp)
  const N_ep = clampDegree(config.parallelism.N_ep)
  const embeddingTotal = getEmbeddingParameterCount(params)
  const routedExpertTotal = params.moe?.expertParameters ?? 0
  const nonExpertTransformerTotal = params.total - embeddingTotal - routedExpertTotal

  return {
    embeddingTotal,
    embeddingLocal: embeddingTotal / N_tp,
    nonExpertLocal: (nonExpertTransformerTotal / N_pp + embeddingTotal) / N_tp,
    routedExpertLocal:
      routedExpertTotal > 0
        ? routedExpertTotal / (N_pp * N_tp * (config.parallelism.N_ep > 1 ? N_ep : 1))
        : 0,
  }
}

function getLargestLayerParameterCount(
  params: ParameterCounts,
  config: TrainingConfig
): number {
  const N_tp = clampDegree(config.parallelism.N_tp)
  const denseLayer =
    (params.perLayer.attention + params.perLayer.ffn + params.perLayer.norm) / N_tp

  if (
    !config.model.moe.enabled ||
    params.moe === null ||
    config.model.moe.L_moe <= 0
  ) {
    return denseLayer
  }

  const N_ep = clampDegree(config.parallelism.N_ep)
  const moeLayers = config.model.moe.L_moe
  const routerPerLayer = params.moe.routerParameters / moeLayers
  const sharedExpertsPerLayer = params.moe.sharedExpertParameters / moeLayers
  const routedExpertsPerLayer = params.moe.expertParameters / moeLayers

  const moeLayer =
    (params.perLayer.attention +
      params.perLayer.norm +
      routerPerLayer +
      sharedExpertsPerLayer) /
      N_tp +
    routedExpertsPerLayer / (N_tp * (config.parallelism.N_ep > 1 ? N_ep : 1))

  return Math.max(denseLayer, moeLayer)
}

function resolveDenseIntermediateSize(
  arch: ModelArchitecture,
  moe: MoEConfig
): number {
  return moe.enabled
    ? (moe.denseIntermediateSize ?? arch.d_ff ?? 4 * arch.d)
    : (arch.d_ff ?? 4 * arch.d)
}

function resolveExpertIntermediateSize(
  arch: ModelArchitecture,
  moe: MoEConfig
): number {
  return moe.expertIntermediateSize ?? resolveDenseIntermediateSize(arch, moe)
}

function getActivationCoefficients(
  arch: ModelArchitecture,
  config: TrainingConfig,
  checkpointing: CheckpointingMode,
  ffnWidth: number
): ActivationCoefficients {
  const N_tp = clampDegree(config.parallelism.N_tp)
  const N_cp = clampDegree(config.parallelism.N_cp)
  const ampLinearDelta = config.ampAutocast ? 2 : 0
  const attentionCoefficient = config.ampAutocast ? 6 : 5
  const sequenceLengthPerRank = config.sequenceLength / N_cp
  const attentionQuadratic =
    checkpointing === "full" || checkpointing === "selective" || config.flashAttention
      ? 0
      : (attentionCoefficient * arch.a * sequenceLengthPerRank) / (arch.d * N_tp)

  const ffnLinear = (4 * ffnWidth) / (arch.d * N_tp)

  if (N_tp === 1) {
    return {
      nonFFNLinear: 10 + ampLinearDelta + 8,
      ffnLinear,
      attentionQuadratic,
    }
  }

  if (isSequenceParallelEnabled(config.parallelism)) {
    return {
      nonFFNLinear: (18 + ampLinearDelta) / N_tp,
      ffnLinear,
      attentionQuadratic,
    }
  }

  return {
    nonFFNLinear: 10 + ampLinearDelta + 8 / N_tp,
    ffnLinear,
    attentionQuadratic,
  }
}

function calculateStoredActivationPerLayer(
  arch: ModelArchitecture,
  config: TrainingConfig,
  checkpointing: CheckpointingMode,
  ffnWidth: number,
  moeFFNScale: number
): number {
  const N_cp = clampDegree(config.parallelism.N_cp)
  const sequenceLengthPerRank = config.sequenceLength / N_cp
  const baseElements = sequenceLengthPerRank * config.microBatchSize * arch.d

  if (checkpointing === "full") {
    return baseElements * getTrainingComputeBytes(config)
  }

  const coefficients = getActivationCoefficients(
    arch,
    config,
    checkpointing,
    ffnWidth
  )

  return (
    baseElements *
    (coefficients.nonFFNLinear +
      coefficients.ffnLinear * moeFFNScale +
      coefficients.attentionQuadratic)
  )
}

function calculateFullCheckpointWorkingMemory(
  arch: ModelArchitecture,
  config: TrainingConfig,
  moe: MoEConfig
): number {
  const denseWorking = calculateStoredActivationPerLayer(
    arch,
    config,
    "none",
    resolveDenseIntermediateSize(arch, moe),
    1
  )

  if (!moe.enabled || moe.E <= 0 || moe.L_moe <= 0) {
    return denseWorking
  }

  const moeWorking = calculateStoredActivationPerLayer(
    arch,
    config,
    "none",
    resolveExpertIntermediateSize(arch, moe),
    moe.topk / moe.E
  )

  return Math.max(denseWorking, moeWorking)
}

function getFrameworkOverheadBytes(config: TrainingConfig): number {
  return config.parallelism.framework === "megatron" ||
    config.parallelism.framework === "deepspeed"
    ? MEGATRON_STYLE_OVERHEAD_BYTES
    : LIGHTWEIGHT_OVERHEAD_BYTES
}

function getOutputLogitsBytes(
  arch: ModelArchitecture,
  config: TrainingConfig
): number {
  if (config.chunkedCrossEntropy) {
    return 0
  }

  const N_cp = clampDegree(config.parallelism.N_cp)

  return (
    config.microBatchSize *
    (config.sequenceLength / N_cp) *
    arch.V *
    getTrainingComputeBytes(config)
  )
}

function getLogitsGradientPeakExtraBytes(
  arch: ModelArchitecture,
  config: TrainingConfig
): number {
  if (config.chunkedCrossEntropy) {
    return 0
  }

  const N_cp = clampDegree(config.parallelism.N_cp)
  const N_tp = clampDegree(config.parallelism.N_tp)

  return (
    (4 * config.microBatchSize * (config.sequenceLength / N_cp) * arch.V) / N_tp
  )
}

function resolveBucketSizeElements(
  config: TrainingConfig,
  kind: "allgather" | "reduce"
): number {
  if (config.zeroCommunication.mode === "deepspeed-defaults") {
    return 5e8
  }

  if (config.zeroCommunication.mode === "custom") {
    if (kind === "allgather" && config.zeroCommunication.allgatherBucketSizeElements !== null) {
      return config.zeroCommunication.allgatherBucketSizeElements
    }

    if (kind === "reduce" && config.zeroCommunication.reduceBucketSizeElements !== null) {
      return config.zeroCommunication.reduceBucketSizeElements
    }
  }

  return config.model.architecture.d * config.model.architecture.d
}

function calculateTrainableModelStates(
  parameterCount: number,
  optimizer: OptimizerValues
): ModelStateMemoryResult {
  const parameters = parameterCount * optimizer.parameterBytes
  const gradients = parameterCount * optimizer.betaGrad
  const optimizerStates = parameterCount * optimizer.kOpt

  return {
    parameters,
    gradients,
    optimizerStates,
    total: parameters + gradients + optimizerStates,
  }
}

function calculateQuantizedBaseModelBytes(
  parameterCount: number,
  quantizationBits: 4 | 8 | null
): number {
  if (quantizationBits === 8) {
    return parameterCount * 1.01
  }

  return parameterCount * 0.55
}

function calculatePostTrainingActivationMemory(
  arch: ModelArchitecture,
  config: PostTrainingConfig
): number {
  return (
    arch.L *
    config.sequenceLength *
    config.batchSize *
    arch.d *
    getPostTrainingActivationBytes(config)
  )
}

function calculateKVCacheBytes(
  arch: ModelArchitecture,
  batch: number,
  sequenceLength: number,
  precision: PostTrainingConfig["kvCachePrecision"]
): number {
  const kvHeads = arch.a_kv ?? arch.a
  const headDim = arch.d / arch.a

  return (
    batch *
    2 *
    arch.L *
    kvHeads *
    headDim *
    sequenceLength *
    getKVCacheBytesPerElement(precision)
  )
}

function finalizePostTrainingMemoryBreakdown(
  args: PostTrainingFinalizeArgs
): PostTrainingMemoryBreakdown {
  const total =
    (args.parameters +
      args.gradients +
      args.optimizerStates +
      args.peakWorkingSet +
      args.frameworkOverhead) *
    1.04
  const gpuCapacity = args.gpu.memoryGB * 1e9
  const usableCapacity = gpuCapacity * 0.9

  return {
    parameters: args.parameters,
    gradients: args.gradients,
    optimizerStates: args.optimizerStates,
    activations: args.activations,
    communicationBuffers: args.communicationBuffers,
    frameworkOverhead: args.frameworkOverhead,
    freeHeadroom: Math.max(0, usableCapacity - total),
    total,
    gpuCapacity,
    usableCapacity,
    fits: total <= usableCapacity,
    trainableModels: args.trainableModels,
    frozenModels: args.frozenModels,
    loraAdapter: args.loraAdapter,
    ppoBuffers: args.ppoBuffers,
    items: args.items,
  }
}

export function calculateModelStateMemory(
  params: ParameterCounts,
  config: TrainingConfig
): ModelStateMemoryResult {
  const optimizer = resolveTrainingOptimizerProfile(config)
  const zeroStage = resolveZeROStage(config)
  const partitioning = getParameterPartitioning(params, config)
  const stateShardDegree = getStateShardDegree(config)
  const optimizerShardDegree = getNonExpertOptimizerShardDegree(config)
  const routedExpertUsesSeparateSharding =
    config.model.moe.enabled &&
    params.moe !== null &&
    config.parallelism.N_ep > 1 &&
    partitioning.routedExpertLocal > 0

  const nonExpertMemory = calculateStateGroupMemory(
    routedExpertUsesSeparateSharding
      ? partitioning.nonExpertLocal
      : partitioning.nonExpertLocal + partitioning.routedExpertLocal,
    optimizer,
    zeroStage,
    stateShardDegree,
    optimizerShardDegree
  )

  const totalMemory = routedExpertUsesSeparateSharding
    ? addModelStateMemory(
        nonExpertMemory,
        calculateStateGroupMemory(
          partitioning.routedExpertLocal,
          optimizer,
          zeroStage,
          (stateShardDegree * clampDegree(config.parallelism.N_tp)) /
            clampDegree(config.parallelism.N_ep),
          (stateShardDegree * clampDegree(config.parallelism.N_tp)) /
            clampDegree(config.parallelism.N_ep)
        )
      )
    : nonExpertMemory

  return applyCPUOffload(totalMemory, config.cpuOffload, zeroStage)
}

export function calculateActivationMemory(
  arch: ModelArchitecture,
  config: TrainingConfig,
  moe: MoEConfig
): number {
  const N_pp = clampDegree(config.parallelism.N_pp)
  const layersPerStage = arch.L / N_pp
  const moeLayersPerStage =
    moe.enabled && moe.L_moe > 0 ? moe.L_moe / N_pp : 0
  const denseLayersPerStage = layersPerStage - moeLayersPerStage
  const denseFFNWidth = resolveDenseIntermediateSize(arch, moe)
  const expertFFNWidth = resolveExpertIntermediateSize(arch, moe)
  const effectiveCheckpointing =
    config.activationCheckpointing === "partial" ? "none" : config.activationCheckpointing
  const denseLayerStored = calculateStoredActivationPerLayer(
    arch,
    config,
    effectiveCheckpointing,
    denseFFNWidth,
    1
  )
  const moeLayerStored =
    moe.enabled && moe.E > 0 && moe.L_moe > 0
      ? calculateStoredActivationPerLayer(
          arch,
          config,
          effectiveCheckpointing,
          expertFFNWidth,
          moe.topk / moe.E
        )
      : denseLayerStored

  let activationPerStage: number

  if (config.activationCheckpointing === "partial") {
    const checkpointedPerLayer = calculateStoredActivationPerLayer(
      arch,
      config,
      "full",
      denseFFNWidth,
      1
    )
    const averageNonFullPerLayer =
      layersPerStage > 0
        ? (denseLayersPerStage * denseLayerStored + moeLayersPerStage * moeLayerStored) /
          layersPerStage
        : 0
    const checkpointedLayers = Math.min(
      Math.max(0, config.partialCheckpointDepth ?? 0),
      layersPerStage
    )

    activationPerStage =
      checkpointedLayers * checkpointedPerLayer +
      Math.max(0, layersPerStage - checkpointedLayers) * averageNonFullPerLayer
  } else {
    activationPerStage =
      denseLayersPerStage * denseLayerStored + moeLayersPerStage * moeLayerStored
  }

  const VP = Math.max(1, config.parallelism.VP)
  const inFlightMicrobatches = Math.max(
    1,
    Math.min(N_pp, config.gradientAccumulationSteps)
  )
  const interleavedMultiplier =
    VP > 1 ? 1 + (N_pp - 1) / (N_pp * VP) : 1
  let total =
    activationPerStage * inFlightMicrobatches * interleavedMultiplier +
    getOutputLogitsBytes(arch, config)

  if (config.activationCheckpointing === "full") {
    total += calculateFullCheckpointWorkingMemory(arch, config, moe)
  }

  return total
}

export function calculateCommunicationBuffers(
  params: ParameterCounts,
  config: TrainingConfig,
  arch: ModelArchitecture
): number {
  const zeroStage = resolveZeROStage(config)
  const optimizer = resolveTrainingOptimizerProfile(config)
  const partitioning = getParameterPartitioning(params, config)
  const N_tp = clampDegree(config.parallelism.N_tp)
  const N_pp = clampDegree(config.parallelism.N_pp)
  const N_cp = clampDegree(config.parallelism.N_cp)
  const computeBytes = getTrainingComputeBytes(config)
  const sequenceLengthPerRank = config.sequenceLength / N_cp
  const largestLayer = getLargestLayerParameterCount(params, config)
  let buffers = 0

  if (zeroStage === 3) {
    if (config.parallelism.framework === "fsdp") {
      buffers +=
        2 * Math.max(largestLayer, partitioning.embeddingLocal) * optimizer.parameterBytes
    } else {
      buffers +=
        Math.max(partitioning.embeddingLocal, 2 * largestLayer) * optimizer.parameterBytes
    }
  }

  if (zeroStage >= 2 && (config.zeroCommunication.overlapComm || zeroStage === 3)) {
    const allgatherBucketSize = resolveBucketSizeElements(config, "allgather")
    const reduceBucketSize = resolveBucketSizeElements(config, "reduce")

    buffers += 4.5 * (allgatherBucketSize + reduceBucketSize) * computeBytes
  }

  buffers += getLogitsGradientPeakExtraBytes(arch, config)

  if (N_tp > 1) {
    buffers +=
      config.microBatchSize *
      sequenceLengthPerRank *
      arch.d *
      ((N_tp - 1) / N_tp) *
      computeBytes
  }

  if (N_pp > 1) {
    buffers += config.microBatchSize * sequenceLengthPerRank * arch.d * computeBytes
  }

  if (config.torchCompile) {
    buffers += 0.1 * calculateModelStateMemory(params, config).parameters
  }

  return buffers
}

export function calculateTotalMemoryPerGPU(
  params: ParameterCounts,
  config: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec
): MemoryBreakdown {
  const modelState = calculateModelStateMemory(params, config)
  const activations = calculateActivationMemory(arch, config, moe)
  const communicationBuffers = calculateCommunicationBuffers(params, config, arch)
  const frameworkOverhead = getFrameworkOverheadBytes(config)
  const total =
    (modelState.total + activations + communicationBuffers + frameworkOverhead) * 1.04
  const gpuCapacity = gpu.memoryGB * 1e9
  const usableCapacity =
    gpuCapacity *
    (config.parallelism.framework === "megatron" ||
    config.parallelism.framework === "deepspeed"
      ? 0.9
      : 0.8)
  const minGPUFloor = calculateMinGPUVRAMFloor(params, config)

  return {
    parameters: modelState.parameters,
    gradients: modelState.gradients,
    optimizerStates: modelState.optimizerStates,
    activations,
    communicationBuffers,
    frameworkOverhead,
    freeHeadroom: Math.max(0, usableCapacity - total),
    total,
    gpuCapacity,
    usableCapacity,
    fits: total <= usableCapacity && minGPUFloor <= usableCapacity,
  }
}

export function calculateMinGPUVRAMFloor(
  params: ParameterCounts,
  config: TrainingConfig
): number {
  const optimizer = resolveTrainingOptimizerProfile(config)

  return (
    getLargestLayerParameterCount(params, config) *
    (optimizer.parameterBytes + optimizer.betaGrad)
  )
}

export function calculateLoRAParamCount(config: PostTrainingConfig): number {
  const architecture = config.baseModel.architecture
  const d = architecture.d
  const dFF = architecture.d_ff ?? 4 * d
  const kvWidth =
    architecture.a_kv !== null && architecture.a > 0
      ? (d * architecture.a_kv) / architecture.a
      : d

  const moduleShapes = {
    q_proj: [d, d],
    k_proj: [d, kvWidth],
    v_proj: [d, kvWidth],
    o_proj: [d, d],
    gate_proj: [d, dFF],
    up_proj: [d, dFF],
    down_proj: [dFF, d],
  } as const

  const perLayer = config.lora.targetModules.reduce((sum, moduleId) => {
    const [inputDim, outputDim] = moduleShapes[moduleId]
    return sum + config.lora.rank * (inputDim + outputDim)
  }, 0)

  return perLayer * architecture.L
}

export function calculateLoRAMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const optimizer = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const baseModelBytes =
    config.baseModel.parameterCount * getPostTrainingWeightBytes(config)
  const loraParameterCount = calculateLoRAParamCount(config)
  const loraStates = calculateTrainableModelStates(loraParameterCount, optimizer)
  const activations = calculatePostTrainingActivationMemory(
    config.baseModel.architecture,
    config
  )

  return finalizePostTrainingMemoryBreakdown({
    gpu: config.hardware.gpu,
    parameters: baseModelBytes + loraStates.parameters,
    gradients: loraStates.gradients,
    optimizerStates: loraStates.optimizerStates,
    activations,
    communicationBuffers: 0,
    frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
    peakWorkingSet: activations,
    trainableModels: loraStates.total,
    frozenModels: baseModelBytes,
    loraAdapter: loraStates.total,
    ppoBuffers: 0,
    items: [
      {
        label: "Base model (frozen)",
        category: "frozen",
        bytes: baseModelBytes,
      },
      {
        label: "LoRA parameters",
        category: "adapter",
        bytes: loraStates.parameters,
      },
      {
        label: "LoRA gradients",
        category: "adapter",
        bytes: loraStates.gradients,
      },
      {
        label: "LoRA optimizer states",
        category: "adapter",
        bytes: loraStates.optimizerStates,
      },
      {
        label: "Activations",
        category: "buffer",
        bytes: activations,
      },
    ],
  })
}

export function calculateQLoRAMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const optimizer = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const quantizationBits = config.lora.quantizationBits ?? 4
  const baseModelBytes = calculateQuantizedBaseModelBytes(
    config.baseModel.parameterCount,
    quantizationBits
  )
  const loraParameterCount = calculateLoRAParamCount(config)
  const loraStates = calculateTrainableModelStates(loraParameterCount, optimizer)
  const activations = calculatePostTrainingActivationMemory(
    config.baseModel.architecture,
    config
  )

  return finalizePostTrainingMemoryBreakdown({
    gpu: config.hardware.gpu,
    parameters: baseModelBytes + loraStates.parameters,
    gradients: loraStates.gradients,
    optimizerStates: loraStates.optimizerStates,
    activations,
    communicationBuffers: 0,
    frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
    peakWorkingSet: activations,
    trainableModels: loraStates.total,
    frozenModels: baseModelBytes,
    loraAdapter: loraStates.total,
    ppoBuffers: 0,
    items: [
      {
        label: `Base model (${quantizationBits}-bit quantized)`,
        category: "frozen",
        bytes: baseModelBytes,
      },
      {
        label: "LoRA parameters",
        category: "adapter",
        bytes: loraStates.parameters,
      },
      {
        label: "LoRA gradients",
        category: "adapter",
        bytes: loraStates.gradients,
      },
      {
        label: "LoRA optimizer states",
        category: "adapter",
        bytes: loraStates.optimizerStates,
      },
      {
        label: "Activations",
        category: "buffer",
        bytes: activations,
      },
    ],
  })
}

export function calculateDPOMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const optimizer = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const activations =
    2 * calculatePostTrainingActivationMemory(config.baseModel.architecture, config)
  const logProbStorage = 2 * config.batchSize * config.sequenceLength * 4

  if (config.approach === "lora" || config.approach === "qlora") {
    const baseModelBytes =
      config.approach === "qlora"
        ? calculateQuantizedBaseModelBytes(
            config.baseModel.parameterCount,
            config.lora.quantizationBits ?? 4
          )
        : config.baseModel.parameterCount * getPostTrainingWeightBytes(config)
    const loraStates = calculateTrainableModelStates(
      calculateLoRAParamCount(config),
      optimizer
    )

    return finalizePostTrainingMemoryBreakdown({
      gpu: config.hardware.gpu,
      parameters: baseModelBytes + loraStates.parameters,
      gradients: loraStates.gradients,
      optimizerStates: loraStates.optimizerStates,
      activations,
      communicationBuffers: logProbStorage,
      frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
      peakWorkingSet: activations + logProbStorage,
      trainableModels: loraStates.total,
      frozenModels: baseModelBytes,
      loraAdapter: loraStates.total,
      ppoBuffers: 0,
      items: [
        {
          label:
            config.approach === "qlora"
              ? "Shared reference base (quantized)"
              : "Shared reference base (frozen)",
          category: "frozen",
          bytes: baseModelBytes,
        },
        {
          label: "LoRA parameters",
          category: "adapter",
          bytes: loraStates.parameters,
        },
        {
          label: "LoRA gradients",
          category: "adapter",
          bytes: loraStates.gradients,
        },
        {
          label: "LoRA optimizer states",
          category: "adapter",
          bytes: loraStates.optimizerStates,
        },
        {
          label: "Activations (chosen + rejected)",
          category: "buffer",
          bytes: activations,
        },
        {
          label: "DPO log-prob storage",
          category: "buffer",
          bytes: logProbStorage,
        },
      ],
    })
  }

  const policyStates = calculateTrainableModelStates(
    config.baseModel.parameterCount,
    optimizer
  )
  const referenceModelBytes =
    config.baseModel.parameterCount * getPostTrainingWeightBytes(config)

  return finalizePostTrainingMemoryBreakdown({
    gpu: config.hardware.gpu,
    parameters: policyStates.parameters + referenceModelBytes,
    gradients: policyStates.gradients,
    optimizerStates: policyStates.optimizerStates,
    activations,
    communicationBuffers: logProbStorage,
    frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
    peakWorkingSet: activations + logProbStorage,
    trainableModels: policyStates.total,
    frozenModels: referenceModelBytes,
    loraAdapter: 0,
    ppoBuffers: 0,
    items: [
      {
        label: "Policy parameters",
        category: "trainable",
        bytes: policyStates.parameters,
      },
      {
        label: "Policy gradients",
        category: "trainable",
        bytes: policyStates.gradients,
      },
      {
        label: "Policy optimizer states",
        category: "trainable",
        bytes: policyStates.optimizerStates,
      },
      {
        label: "Reference model (frozen)",
        category: "frozen",
        bytes: referenceModelBytes,
      },
      {
        label: "Activations (chosen + rejected)",
        category: "buffer",
        bytes: activations,
      },
      {
        label: "DPO log-prob storage",
        category: "buffer",
        bytes: logProbStorage,
      },
    ],
  })
}

export function calculatePPOMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const optimizer = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const frozenWeightBytes = getPostTrainingWeightBytes(config)
  const actorActivations = calculatePostTrainingActivationMemory(
    config.baseModel.architecture,
    config
  )
  const rolloutBuffers = 16 * config.sequenceLength * config.batchSize
  const kvCacheBytes = calculateKVCacheBytes(
    config.baseModel.architecture,
    config.batchSize,
    config.sequenceLength,
    config.kvCachePrecision
  )
  const criticStates = calculateTrainableModelStates(
    config.ppo.criticModelParameterCount,
    optimizer
  )
  const rewardModelBytes =
    config.ppo.rewardModelParameterCount * frozenWeightBytes
  const items: PostTrainingModelMemoryLineItem[] = [
    {
      label: "Reward model (frozen)",
      category: "frozen",
      bytes: rewardModelBytes,
    },
    {
      label: "Critic parameters",
      category: "trainable",
      bytes: criticStates.parameters,
    },
    {
      label: "Critic gradients",
      category: "trainable",
      bytes: criticStates.gradients,
    },
    {
      label: "Critic optimizer states",
      category: "trainable",
      bytes: criticStates.optimizerStates,
    },
  ]

  let parameters = criticStates.parameters + rewardModelBytes
  let gradients = criticStates.gradients
  let optimizerStates = criticStates.optimizerStates
  let trainableModels = criticStates.total
  let frozenModels = rewardModelBytes
  let loraAdapter = 0

  if (config.approach === "lora" || config.approach === "qlora") {
    const actorBaseBytes =
      config.approach === "qlora"
        ? calculateQuantizedBaseModelBytes(
            config.baseModel.parameterCount,
            config.lora.quantizationBits ?? 4
          )
        : config.baseModel.parameterCount * frozenWeightBytes
    const actorLoRAStates = calculateTrainableModelStates(
      calculateLoRAParamCount(config),
      optimizer
    )

    parameters += actorBaseBytes + actorLoRAStates.parameters
    gradients += actorLoRAStates.gradients
    optimizerStates += actorLoRAStates.optimizerStates
    trainableModels += actorLoRAStates.total
    frozenModels += actorBaseBytes
    loraAdapter = actorLoRAStates.total

    items.unshift({
      label:
        config.approach === "qlora"
          ? "Actor base (quantized, shared reference)"
          : "Actor base (frozen, shared reference)",
      category: "frozen",
      bytes: actorBaseBytes,
    })
    items.splice(1, 0, {
      label: "Actor LoRA parameters",
      category: "adapter",
      bytes: actorLoRAStates.parameters,
    })
    items.splice(2, 0, {
      label: "Actor LoRA gradients",
      category: "adapter",
      bytes: actorLoRAStates.gradients,
    })
    items.splice(3, 0, {
      label: "Actor LoRA optimizer states",
      category: "adapter",
      bytes: actorLoRAStates.optimizerStates,
    })
  } else {
    const actorStates = calculateTrainableModelStates(
      config.baseModel.parameterCount,
      optimizer
    )
    const referenceModelBytes =
      config.baseModel.parameterCount * frozenWeightBytes

    parameters += actorStates.parameters + referenceModelBytes
    gradients += actorStates.gradients
    optimizerStates += actorStates.optimizerStates
    trainableModels += actorStates.total
    frozenModels += referenceModelBytes

    items.unshift({
      label: "Reference model (frozen)",
      category: "frozen",
      bytes: referenceModelBytes,
    })
    items.unshift({
      label: "Actor optimizer states",
      category: "trainable",
      bytes: actorStates.optimizerStates,
    })
    items.unshift({
      label: "Actor gradients",
      category: "trainable",
      bytes: actorStates.gradients,
    })
    items.unshift({
      label: "Actor parameters",
      category: "trainable",
      bytes: actorStates.parameters,
    })
  }

  items.push(
    {
      label: "Actor activations",
      category: "buffer",
      bytes: actorActivations,
    },
    {
      label: "PPO rollout buffers",
      category: "buffer",
      bytes: rolloutBuffers,
    },
    {
      label: "KV cache (generation)",
      category: "buffer",
      bytes: kvCacheBytes,
    }
  )

  return finalizePostTrainingMemoryBreakdown({
    gpu: config.hardware.gpu,
    parameters,
    gradients,
    optimizerStates,
    activations: actorActivations,
    communicationBuffers: Math.max(rolloutBuffers, kvCacheBytes),
    frameworkOverhead: MEGATRON_STYLE_OVERHEAD_BYTES,
    peakWorkingSet: Math.max(actorActivations + rolloutBuffers, kvCacheBytes),
    trainableModels,
    frozenModels,
    loraAdapter,
    ppoBuffers: rolloutBuffers + kvCacheBytes,
    items,
  })
}

export function calculateGRPOMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const optimizer = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const frozenWeightBytes = getPostTrainingWeightBytes(config)
  const activations = calculatePostTrainingActivationMemory(
    config.baseModel.architecture,
    config
  )
  const kvCacheBytes = calculateKVCacheBytes(
    config.baseModel.architecture,
    config.grpo.groupSize * config.batchSize,
    config.sequenceLength,
    config.kvCachePrecision
  )

  if (config.approach === "lora" || config.approach === "qlora") {
    const baseModelBytes =
      config.approach === "qlora"
        ? calculateQuantizedBaseModelBytes(
            config.baseModel.parameterCount,
            config.lora.quantizationBits ?? 4
          )
        : config.baseModel.parameterCount * frozenWeightBytes
    const loraStates = calculateTrainableModelStates(
      calculateLoRAParamCount(config),
      optimizer
    )

    return finalizePostTrainingMemoryBreakdown({
      gpu: config.hardware.gpu,
      parameters: baseModelBytes + loraStates.parameters,
      gradients: loraStates.gradients,
      optimizerStates: loraStates.optimizerStates,
      activations,
      communicationBuffers: kvCacheBytes,
      frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
      peakWorkingSet: Math.max(activations, kvCacheBytes),
      trainableModels: loraStates.total,
      frozenModels: baseModelBytes,
      loraAdapter: loraStates.total,
      ppoBuffers: kvCacheBytes,
      items: [
        {
          label:
            config.approach === "qlora"
              ? "Policy base (quantized, shared reference)"
              : "Policy base (frozen, shared reference)",
          category: "frozen",
          bytes: baseModelBytes,
        },
        {
          label: "LoRA parameters",
          category: "adapter",
          bytes: loraStates.parameters,
        },
        {
          label: "LoRA gradients",
          category: "adapter",
          bytes: loraStates.gradients,
        },
        {
          label: "LoRA optimizer states",
          category: "adapter",
          bytes: loraStates.optimizerStates,
        },
        {
          label: "Activations",
          category: "buffer",
          bytes: activations,
        },
        {
          label: `KV cache (generation, G=${config.grpo.groupSize})`,
          category: "buffer",
          bytes: kvCacheBytes,
        },
      ],
    })
  }

  const policyStates = calculateTrainableModelStates(
    config.baseModel.parameterCount,
    optimizer
  )
  const referenceModelBytes =
    config.baseModel.parameterCount * frozenWeightBytes

  return finalizePostTrainingMemoryBreakdown({
    gpu: config.hardware.gpu,
    parameters: policyStates.parameters + referenceModelBytes,
    gradients: policyStates.gradients,
    optimizerStates: policyStates.optimizerStates,
    activations,
    communicationBuffers: kvCacheBytes,
    frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
    peakWorkingSet: Math.max(activations, kvCacheBytes),
    trainableModels: policyStates.total,
    frozenModels: referenceModelBytes,
    loraAdapter: 0,
    ppoBuffers: kvCacheBytes,
    items: [
      {
        label: "Policy parameters",
        category: "trainable",
        bytes: policyStates.parameters,
      },
      {
        label: "Policy gradients",
        category: "trainable",
        bytes: policyStates.gradients,
      },
      {
        label: "Policy optimizer states",
        category: "trainable",
        bytes: policyStates.optimizerStates,
      },
      {
        label: "Reference model (frozen)",
        category: "frozen",
        bytes: referenceModelBytes,
      },
      {
        label: "Activations",
        category: "buffer",
        bytes: activations,
      },
      {
        label: `KV cache (generation, G=${config.grpo.groupSize})`,
        category: "buffer",
        bytes: kvCacheBytes,
      },
    ],
  })
}
