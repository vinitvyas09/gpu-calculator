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
  TrainingPrecision,
  ZeROStage,
} from "../types"
import { OPTIMIZER_PROFILES } from "../constants"

// ---------------------------------------------------------------------------
// 1. Optimizer profile lookup (Section 5.1)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

export interface ModelStateMemoryResult {
  parameters: number
  gradients: number
  optimizerStates: number
  total: number
}

interface ActivationComponentBreakdown {
  nonFFNLinear: number
  ffnLinear: number
  attentionScore: number
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

function getActivationElementBytes(precision: TrainingPrecision): number {
  return precision === "fp32" ? 4 : 2
}

function resolveZeROStage(config: TrainingConfig): ZeROStage {
  const fsdp = config.parallelism.fsdpStrategy

  if (fsdp === null) {
    return config.parallelism.zeroStage
  }

  const mapping: Record<FSDPStrategy, ZeROStage> = {
    NO_SHARD: 0,
    SHARD_GRAD_OP: 2,
    FULL_SHARD: 3,
    HYBRID_SHARD: 3,
    HYBRID_SHARD_ZERO2: 2,
  }

  return mapping[fsdp]
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

function getStateShardDegree(config: TrainingConfig): number {
  return usesHybridShard(config)
    ? Math.max(1, config.hardware.gpu.gpusPerNode)
    : Math.max(1, config.parallelism.N_dp)
}

function getNonExpertOptimizerShardDegree(config: TrainingConfig): number {
  const sequenceParallelDegree = isSequenceParallelEnabled(config.parallelism)
    ? Math.max(1, config.parallelism.N_tp)
    : 1

  return getStateShardDegree(config) * sequenceParallelDegree
}

function getMoEParameterGroups(
  params: ParameterCounts,
  config: TrainingConfig
): { standardPerGPU: number; routedExpertPerGPU: number } {
  const extras =
    params.embedding +
    params.outputProjection +
    params.positionalEmbedding +
    params.finalNorm
  const standardTotal = params.total - extras - (params.moe?.expertParameters ?? 0)
  const standardPerStage = standardTotal / Math.max(1, config.parallelism.N_pp) + extras
  const routedExpertsPerStage =
    (params.moe?.expertParameters ?? 0) / Math.max(1, config.parallelism.N_pp)

  const standardPerGPU = standardPerStage / Math.max(1, config.parallelism.N_tp)
  const routedExpertPerGPU =
    config.parallelism.N_ep > 1
      ? routedExpertsPerStage /
        (Math.max(1, config.parallelism.N_tp) * Math.max(1, config.parallelism.N_ep))
      : routedExpertsPerStage / Math.max(1, config.parallelism.N_tp)

  return {
    standardPerGPU,
    routedExpertPerGPU,
  }
}

function calculateShardedStateGroup(
  parameterCount: number,
  optimizer: OptimizerValues,
  zeroStage: ZeROStage,
  stateShardDegree: number,
  optimizerShardDegree: number
): ModelStateMemoryResult {
  switch (zeroStage) {
    case 0:
      return {
        parameters: parameterCount * optimizer.parameterBytes,
        gradients: parameterCount * optimizer.betaGrad,
        optimizerStates: parameterCount * optimizer.kOpt,
        total:
          parameterCount *
          (optimizer.parameterBytes + optimizer.betaGrad + optimizer.kOpt),
      }
    case 1: {
      const parameters = parameterCount * optimizer.parameterBytes
      const gradients = parameterCount * optimizer.betaGrad
      const optimizerStates =
        (parameterCount * optimizer.kOpt) / Math.max(1, optimizerShardDegree)

      return {
        parameters,
        gradients,
        optimizerStates,
        total: parameters + gradients + optimizerStates,
      }
    }
    case 2: {
      const parameters = parameterCount * optimizer.parameterBytes
      const gradients =
        (parameterCount * optimizer.betaGrad) / Math.max(1, stateShardDegree)
      const optimizerStates =
        (parameterCount * optimizer.kOpt) / Math.max(1, optimizerShardDegree)

      return {
        parameters,
        gradients,
        optimizerStates,
        total: parameters + gradients + optimizerStates,
      }
    }
    case 3: {
      const parameters =
        (parameterCount * optimizer.parameterBytes) / Math.max(1, stateShardDegree)
      const gradients =
        (parameterCount * optimizer.betaGrad) / Math.max(1, stateShardDegree)
      const optimizerStates =
        (parameterCount * optimizer.kOpt) / Math.max(1, optimizerShardDegree)

      return {
        parameters,
        gradients,
        optimizerStates,
        total: parameters + gradients + optimizerStates,
      }
    }
    default:
      return {
        parameters: parameterCount * optimizer.parameterBytes,
        gradients: parameterCount * optimizer.betaGrad,
        optimizerStates: parameterCount * optimizer.kOpt,
        total:
          parameterCount *
          (optimizer.parameterBytes + optimizer.betaGrad + optimizer.kOpt),
      }
  }
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

function getLinearActivationComponents(
  arch: ModelArchitecture,
  config: TrainingConfig
): Pick<ActivationComponentBreakdown, "nonFFNLinear" | "ffnLinear"> {
  const N_tp = Math.max(1, config.parallelism.N_tp)
  const ampDelta = config.ampAutocast ? 2 : 0
  const dFF = arch.d_ff ?? 4 * arch.d
  const ffnLinear = 4 * dFF / arch.d

  if (N_tp === 1) {
    return {
      nonFFNLinear: 18 + ampDelta,
      ffnLinear,
    }
  }

  if (isSequenceParallelEnabled(config.parallelism)) {
    return {
      nonFFNLinear: (18 + ampDelta) / N_tp,
      ffnLinear: ffnLinear / N_tp,
    }
  }

  return {
    nonFFNLinear: 10 + ampDelta + 8 / N_tp,
    ffnLinear: ffnLinear / N_tp,
  }
}

function getAttentionScoreCoefficient(
  arch: ModelArchitecture,
  config: TrainingConfig,
  checkpointingMode: CheckpointingMode,
  sequenceLengthPerRank: number
): number {
  if (checkpointingMode === "full" || checkpointingMode === "selective" || config.flashAttention) {
    return 0
  }

  const baseAttentionCoefficient = config.ampAutocast ? 6 : 5
  const tpDivisor = config.parallelism.N_tp > 1 ? config.parallelism.N_tp : 1

  return (
    (baseAttentionCoefficient * arch.a * sequenceLengthPerRank) /
    (arch.d * tpDivisor)
  )
}

function getActivationComponents(
  arch: ModelArchitecture,
  config: TrainingConfig,
  checkpointingMode: CheckpointingMode
): ActivationComponentBreakdown {
  const sequenceLengthPerRank =
    config.sequenceLength / Math.max(1, config.parallelism.N_cp)
  const linear = getLinearActivationComponents(arch, config)

  return {
    nonFFNLinear: linear.nonFFNLinear,
    ffnLinear: linear.ffnLinear,
    attentionScore: getAttentionScoreCoefficient(
      arch,
      config,
      checkpointingMode,
      sequenceLengthPerRank
    ),
  }
}

function calculateStoredActivationPerLayer(
  arch: ModelArchitecture,
  config: TrainingConfig,
  checkpointingMode: CheckpointingMode,
  moeFFNScale: number
): number {
  const sequenceLengthPerRank =
    config.sequenceLength / Math.max(1, config.parallelism.N_cp)
  const microBatchSize = config.microBatchSize
  const hiddenSize = arch.d

  if (checkpointingMode === "full") {
    return 2 * sequenceLengthPerRank * microBatchSize * hiddenSize
  }

  const components = getActivationComponents(arch, config, checkpointingMode)

  return (
    sequenceLengthPerRank *
    microBatchSize *
    hiddenSize *
    (components.nonFFNLinear +
      components.ffnLinear * moeFFNScale +
      components.attentionScore)
  )
}

function calculateFullCheckpointWorkingMemory(
  arch: ModelArchitecture,
  config: TrainingConfig,
  moe: MoEConfig
): number {
  const denseLayer = calculateStoredActivationPerLayer(arch, config, "none", 1)

  if (!moe.enabled || moe.E <= 0 || moe.L_moe <= 0) {
    return denseLayer
  }

  const moeLayer = calculateStoredActivationPerLayer(
    arch,
    config,
    "none",
    moe.topk / moe.E
  )

  return Math.max(denseLayer, moeLayer)
}

function getInterleavedActivationMultiplier(config: TrainingConfig): number {
  const N_pp = Math.max(1, config.parallelism.N_pp)
  const VP = Math.max(1, config.parallelism.VP)

  if (N_pp <= 1 || VP <= 1) {
    return 1
  }

  return 1 + (N_pp - 1) / (N_pp * VP)
}

function getLargestLayerParameterCount(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig
): number {
  const denseLayer = params.perLayer.attention + params.perLayer.ffn + params.perLayer.norm

  if (!config.model.moe.enabled || !params.moe || config.model.moe.L_moe <= 0) {
    return denseLayer / Math.max(1, config.parallelism.N_tp)
  }

  const moeLayers = config.model.moe.L_moe
  const routerPerLayer = params.moe.routerParameters / moeLayers
  const routedExpertsPerLayer = params.moe.expertParameters / moeLayers
  const sharedExpertsPerLayer = params.moe.sharedExpertParameters / moeLayers

  const localMoELayer =
    (params.perLayer.attention + params.perLayer.norm + routerPerLayer) /
      Math.max(1, config.parallelism.N_tp) +
    routedExpertsPerLayer /
      (Math.max(1, config.parallelism.N_tp) * Math.max(1, config.parallelism.N_ep)) +
    sharedExpertsPerLayer / Math.max(1, config.parallelism.N_tp)

  return Math.max(denseLayer / Math.max(1, config.parallelism.N_tp), localMoELayer)
}

function getEmbeddingLoadPerGPU(params: ParameterCounts, N_tp: number): number {
  return (
    (params.embedding +
      params.outputProjection +
      params.positionalEmbedding +
      params.finalNorm) /
    Math.max(1, N_tp)
  )
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

function getFrozenModelWeightBytes(config: PostTrainingConfig): number {
  return getOptimizerProfile(config.optimizer, config.gradientPrecision).parameterBytes
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
  return arch.L * 2 * config.sequenceLength * config.batchSize * arch.d
}

function finalizePostTrainingMemoryBreakdown(
  args: PostTrainingFinalizeArgs
): PostTrainingMemoryBreakdown {
  const rawTotal =
    args.parameters +
    args.gradients +
    args.optimizerStates +
    args.peakWorkingSet +
    args.frameworkOverhead
  const total = rawTotal * 1.04
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

// ---------------------------------------------------------------------------
// 2. Model state memory (Sections 5.1, 5.2, 5.7)
// ---------------------------------------------------------------------------

export function calculateModelStateMemory(
  params: ParameterCounts,
  config: TrainingConfig
): ModelStateMemoryResult {
  const optimizer = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const zeroStage = resolveZeROStage(config)
  const stateShardDegree = getStateShardDegree(config)
  const nonExpertOptimizerShardDegree = getNonExpertOptimizerShardDegree(config)
  const { standardPerGPU, routedExpertPerGPU } = getMoEParameterGroups(params, config)

  const specialMoESharding =
    config.model.moe.enabled &&
    params.moe !== null &&
    config.parallelism.N_ep > 1 &&
    routedExpertPerGPU > 0

  const standardGroup = calculateShardedStateGroup(
    specialMoESharding ? standardPerGPU : standardPerGPU + routedExpertPerGPU,
    optimizer,
    zeroStage,
    stateShardDegree,
    nonExpertOptimizerShardDegree
  )

  let combined: ModelStateMemoryResult = standardGroup

  if (specialMoESharding) {
    const expertShardDegree =
      (stateShardDegree * Math.max(1, config.parallelism.N_tp)) /
      Math.max(1, config.parallelism.N_ep)

    const expertGroup = calculateShardedStateGroup(
      routedExpertPerGPU,
      optimizer,
      zeroStage,
      expertShardDegree,
      expertShardDegree
    )

    combined = {
      parameters: standardGroup.parameters + expertGroup.parameters,
      gradients: standardGroup.gradients + expertGroup.gradients,
      optimizerStates: standardGroup.optimizerStates + expertGroup.optimizerStates,
      total: standardGroup.total + expertGroup.total,
    }
  }

  return applyCPUOffload(combined, config.cpuOffload, zeroStage)
}

// ---------------------------------------------------------------------------
// 3. Activation memory (Section 5.3)
// ---------------------------------------------------------------------------

export function calculateActivationMemory(
  arch: ModelArchitecture,
  config: TrainingConfig,
  moe: MoEConfig
): number {
  const layersPerStage = arch.L / Math.max(1, config.parallelism.N_pp)
  const moeLayersPerStage =
    moe.enabled && moe.L_moe > 0
      ? moe.L_moe / Math.max(1, config.parallelism.N_pp)
      : 0
  const denseLayersPerStage = layersPerStage - moeLayersPerStage
  const inflightMicrobatches = Math.max(
    1,
    Math.min(Math.max(1, config.parallelism.N_pp), config.gradientAccumulationSteps)
  )
  const interleavedMultiplier = getInterleavedActivationMultiplier(config)
  const activationBytes = getActivationElementBytes(config.precision)
  const sequenceLengthPerRank =
    config.sequenceLength / Math.max(1, config.parallelism.N_cp)
  const outputLogits = config.chunkedCrossEntropy
    ? 0
    : config.microBatchSize * sequenceLengthPerRank * arch.V * activationBytes

  const denseStoredPerLayer = calculateStoredActivationPerLayer(
    arch,
    config,
    config.activationCheckpointing === "partial"
      ? "none"
      : config.activationCheckpointing,
    1
  )
  const moeStoredPerLayer =
    moe.enabled && moe.E > 0 && moe.L_moe > 0
      ? calculateStoredActivationPerLayer(
          arch,
          config,
          config.activationCheckpointing === "partial"
            ? "none"
            : config.activationCheckpointing,
          moe.topk / moe.E
        )
      : denseStoredPerLayer

  let activationPerStage: number

  if (config.activationCheckpointing === "partial") {
    const checkpointedPerLayer = calculateStoredActivationPerLayer(
      arch,
      config,
      "full",
      1
    )
    const averageFullPerLayer =
      layersPerStage > 0
        ? (denseLayersPerStage * denseStoredPerLayer +
            moeLayersPerStage * moeStoredPerLayer) /
          layersPerStage
        : 0
    const checkpointedLayers = Math.min(
      Math.max(0, config.partialCheckpointDepth ?? 0),
      layersPerStage
    )

    activationPerStage =
      checkpointedLayers * checkpointedPerLayer +
      Math.max(0, layersPerStage - checkpointedLayers) * averageFullPerLayer
  } else {
    activationPerStage =
      denseLayersPerStage * denseStoredPerLayer +
      moeLayersPerStage * moeStoredPerLayer
  }

  const layerActivations =
    activationPerStage * inflightMicrobatches * interleavedMultiplier
  let total = layerActivations + outputLogits

  if (config.activationCheckpointing === "full") {
    total += calculateFullCheckpointWorkingMemory(arch, config, moe)
  }

  return total
}

// ---------------------------------------------------------------------------
// 4. Communication buffers (Sections 5.4, 5.6, 5.7)
// ---------------------------------------------------------------------------

export function calculateCommunicationBuffers(
  params: ParameterCounts,
  config: TrainingConfig,
  arch: ModelArchitecture
): number {
  const zeroStage = resolveZeROStage(config)
  const optimizer = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const activationBytes = getActivationElementBytes(config.precision)
  const weightBytes = optimizer.parameterBytes
  const sequenceLengthPerRank =
    config.sequenceLength / Math.max(1, config.parallelism.N_cp)
  const N_tp = Math.max(1, config.parallelism.N_tp)
  const N_pp = Math.max(1, config.parallelism.N_pp)
  const VP = Math.max(1, config.parallelism.VP)
  const largestLayer = getLargestLayerParameterCount(params, arch, config)
  const embeddingLoad = getEmbeddingLoadPerGPU(params, N_tp)
  const logitGradientPeak =
    config.chunkedCrossEntropy
      ? 0
      : (4 * config.microBatchSize * sequenceLengthPerRank * arch.V) / N_tp

  let buffers = 0

  if (zeroStage === 3) {
    if (config.parallelism.framework === "fsdp") {
      buffers += 2 * Math.max(largestLayer, embeddingLoad) * weightBytes
    } else {
      buffers += Math.max(embeddingLoad, 2 * largestLayer) * weightBytes
    }
  }

  if (zeroStage >= 2) {
    const overlapCommEnabled =
      config.zeroCommunication.overlapComm || zeroStage === 3

    if (overlapCommEnabled) {
      const allgatherBucketSize =
        config.zeroCommunication.mode === "custom" &&
        config.zeroCommunication.allgatherBucketSizeElements !== null
          ? config.zeroCommunication.allgatherBucketSizeElements
          : config.zeroCommunication.mode === "deepspeed-defaults"
            ? 5e8
            : arch.d * arch.d

      const reduceBucketSize =
        config.zeroCommunication.mode === "custom" &&
        config.zeroCommunication.reduceBucketSizeElements !== null
          ? config.zeroCommunication.reduceBucketSizeElements
          : config.zeroCommunication.mode === "deepspeed-defaults"
            ? 5e8
            : arch.d * arch.d

      buffers +=
        4.5 * (allgatherBucketSize + reduceBucketSize) * activationBytes
    }
  }

  buffers += logitGradientPeak

  if (N_tp > 1) {
    buffers +=
      config.microBatchSize *
      sequenceLengthPerRank *
      arch.d *
      ((N_tp - 1) / N_tp) *
      activationBytes
  }

  if (N_pp > 1) {
    buffers +=
      sequenceLengthPerRank *
      config.microBatchSize *
      arch.d *
      activationBytes *
      VP
  }

  if (config.torchCompile) {
    buffers += 0.1 * params.total * weightBytes
  }

  return buffers
}

// ---------------------------------------------------------------------------
// 5. Total memory per GPU (Section 5.5)
// ---------------------------------------------------------------------------

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
  const frameworkOverhead =
    config.parallelism.framework === "megatron" ||
    config.parallelism.framework === "deepspeed"
      ? 5e9
      : 2e9
  const rawTotal =
    modelState.total + activations + communicationBuffers + frameworkOverhead
  const total = rawTotal * 1.04
  const gpuCapacity = gpu.memoryGB * 1e9
  const usableCapacity =
    gpuCapacity *
    (config.parallelism.framework === "fsdp" ||
    config.parallelism.framework === "hf_trainer"
      ? 0.8
      : 0.9)
  const minFloor = calculateMinGPUVRAMFloor(params, config)

  return {
    parameters: modelState.parameters,
    gradients: modelState.gradients,
    optimizerStates: modelState.optimizerStates,
    activations,
    communicationBuffers,
    frameworkOverhead,
    freeHeadroom: Math.max(
      0,
      Math.min(usableCapacity - total, gpuCapacity * 0.8 - minFloor)
    ),
    total,
    gpuCapacity,
    usableCapacity,
    fits: total <= usableCapacity && minFloor <= gpuCapacity * 0.8,
  }
}

export function calculateMinGPUVRAMFloor(
  params: ParameterCounts,
  config: TrainingConfig
): number {
  const optimizer = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const largestLayer =
    params.perLayer.attention + params.perLayer.ffn + params.perLayer.norm

  return largestLayer * (optimizer.parameterBytes + optimizer.betaGrad)
}

// ---------------------------------------------------------------------------
// 6. Post-training memory functions (Section 10)
// ---------------------------------------------------------------------------

export function calculateLoRAParamCount(config: PostTrainingConfig): number {
  return (
    2 *
    config.lora.rank *
    config.baseModel.architecture.d *
    config.lora.targetModules.length *
    config.baseModel.architecture.L
  )
}

export function calculateLoRAMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const optimizer = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const baseModelBytes =
    config.baseModel.parameterCount * getFrozenModelWeightBytes(config)
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
    frameworkOverhead: 1e9,
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
    frameworkOverhead: 1e9,
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
    const loraParameterCount = calculateLoRAParamCount(config)
    const loraStates = calculateTrainableModelStates(loraParameterCount, optimizer)
    const baseModelBytes =
      config.approach === "qlora"
        ? calculateQuantizedBaseModelBytes(
            config.baseModel.parameterCount,
            config.lora.quantizationBits ?? 4
          )
        : config.baseModel.parameterCount * getFrozenModelWeightBytes(config)

    return finalizePostTrainingMemoryBreakdown({
      gpu: config.hardware.gpu,
      parameters: baseModelBytes + loraStates.parameters,
      gradients: loraStates.gradients,
      optimizerStates: loraStates.optimizerStates,
      activations,
      communicationBuffers: logProbStorage,
      frameworkOverhead: 1e9,
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
    config.baseModel.parameterCount * getFrozenModelWeightBytes(config)

  return finalizePostTrainingMemoryBreakdown({
    gpu: config.hardware.gpu,
    parameters: policyStates.parameters + referenceModelBytes,
    gradients: policyStates.gradients,
    optimizerStates: policyStates.optimizerStates,
    activations,
    communicationBuffers: logProbStorage,
    frameworkOverhead: 1e9,
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
  const baseModelWeightBytes = getFrozenModelWeightBytes(config)
  const actorActivationMemory = calculatePostTrainingActivationMemory(
    config.baseModel.architecture,
    config
  )
  const criticStates = calculateTrainableModelStates(
    config.ppo.criticModelParameterCount,
    optimizer
  )
  const rewardModelBytes =
    config.ppo.rewardModelParameterCount * baseModelWeightBytes
  const rolloutBuffers = 16 * config.sequenceLength * config.batchSize
  const kvCacheBytes =
    config.batchSize *
    2 *
    config.baseModel.architecture.L *
    (config.baseModel.architecture.a_kv ?? config.baseModel.architecture.a) *
    (config.baseModel.architecture.d / config.baseModel.architecture.a) *
    config.sequenceLength *
    (config.kvCachePrecision === "int8" ? 1 : 2)

  let parameters = 0
  let gradients = 0
  let optimizerStates = 0
  let trainableModels = 0
  let frozenModels = rewardModelBytes
  let loraAdapter = 0
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

  parameters += criticStates.parameters + rewardModelBytes
  gradients += criticStates.gradients
  optimizerStates += criticStates.optimizerStates
  trainableModels += criticStates.total

  if (config.approach === "lora" || config.approach === "qlora") {
    const actorBaseBytes =
      config.approach === "qlora"
        ? calculateQuantizedBaseModelBytes(
            config.baseModel.parameterCount,
            config.lora.quantizationBits ?? 4
          )
        : config.baseModel.parameterCount * baseModelWeightBytes
    const adapterStates = calculateTrainableModelStates(
      calculateLoRAParamCount(config),
      optimizer
    )

    parameters += actorBaseBytes + adapterStates.parameters
    gradients += adapterStates.gradients
    optimizerStates += adapterStates.optimizerStates
    trainableModels += adapterStates.total
    frozenModels += actorBaseBytes
    loraAdapter = adapterStates.total

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
      bytes: adapterStates.parameters,
    })
    items.splice(2, 0, {
      label: "Actor LoRA gradients",
      category: "adapter",
      bytes: adapterStates.gradients,
    })
    items.splice(3, 0, {
      label: "Actor LoRA optimizer states",
      category: "adapter",
      bytes: adapterStates.optimizerStates,
    })
  } else {
    const actorStates = calculateTrainableModelStates(
      config.baseModel.parameterCount,
      optimizer
    )
    const referenceModelBytes =
      config.baseModel.parameterCount * baseModelWeightBytes

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
      bytes: actorActivationMemory,
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
    activations: actorActivationMemory,
    communicationBuffers: Math.max(rolloutBuffers, kvCacheBytes),
    frameworkOverhead: 5e9,
    peakWorkingSet: Math.max(actorActivationMemory + rolloutBuffers, kvCacheBytes),
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
  const baseModelWeightBytes = getFrozenModelWeightBytes(config)
  const activations = calculatePostTrainingActivationMemory(
    config.baseModel.architecture,
    config
  )
  const kvCacheBytes =
    config.grpo.groupSize *
    config.batchSize *
    2 *
    config.baseModel.architecture.L *
    (config.baseModel.architecture.a_kv ?? config.baseModel.architecture.a) *
    (config.baseModel.architecture.d / config.baseModel.architecture.a) *
    config.sequenceLength *
    (config.kvCachePrecision === "int8" ? 1 : 2)

  if (config.approach === "lora" || config.approach === "qlora") {
    const baseModelBytes =
      config.approach === "qlora"
        ? calculateQuantizedBaseModelBytes(
            config.baseModel.parameterCount,
            config.lora.quantizationBits ?? 4
          )
        : config.baseModel.parameterCount * baseModelWeightBytes
    const adapterStates = calculateTrainableModelStates(
      calculateLoRAParamCount(config),
      optimizer
    )

    return finalizePostTrainingMemoryBreakdown({
      gpu: config.hardware.gpu,
      parameters: baseModelBytes + adapterStates.parameters,
      gradients: adapterStates.gradients,
      optimizerStates: adapterStates.optimizerStates,
      activations,
      communicationBuffers: kvCacheBytes,
      frameworkOverhead: 1e9,
      peakWorkingSet: Math.max(activations, kvCacheBytes),
      trainableModels: adapterStates.total,
      frozenModels: baseModelBytes,
      loraAdapter: adapterStates.total,
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
          bytes: adapterStates.parameters,
        },
        {
          label: "LoRA gradients",
          category: "adapter",
          bytes: adapterStates.gradients,
        },
        {
          label: "LoRA optimizer states",
          category: "adapter",
          bytes: adapterStates.optimizerStates,
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
    config.baseModel.parameterCount * baseModelWeightBytes

  return finalizePostTrainingMemoryBreakdown({
    gpu: config.hardware.gpu,
    parameters: policyStates.parameters + referenceModelBytes,
    gradients: policyStates.gradients,
    optimizerStates: policyStates.optimizerStates,
    activations,
    communicationBuffers: kvCacheBytes,
    frameworkOverhead: 1e9,
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
