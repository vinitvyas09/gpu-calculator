import type {
  CheckpointingMode,
  CPUOffloadMode,
  FSDPStrategy,
  GPUSpec,
  GradientPrecision,
  LoRATargetModule,
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
import { DEFAULT_TRAINING_CONFIG, OPTIMIZER_PROFILES } from "../constants"
import { calculateParameterCount } from "./compute"
import { getParallelismLocalGroupSize } from "./hardware"

export interface OptimizerValues {
  phi: number
  kOpt: number
  betaGrad: number
  parameterBytes: number
  masterWeightBytes: number
  optimizerStateBytes: number
}

type ActivationSchedule = "none" | "1f1b" | "interleaved" | "afab"

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

function resolvePretrainingOptimizer(optimizer: OptimizerType): OptimizerType {
  const profile = OPTIMIZER_PROFILES.find((candidate) => candidate.id === optimizer)

  if (!profile || profile.supportsPretraining) {
    return optimizer
  }

  return DEFAULT_TRAINING_CONFIG.optimizer
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
  moeLocal: number
  routedExpertLocal: number
  sharedExpertLocal: number
  routerLocal: number
  stages: ParameterStagePartitioning[]
}

interface ParameterStagePartitioning {
  nonExpertLocal: number
  moeLocal: number
  routedExpertLocal: number
  sharedExpertLocal: number
  routerLocal: number
}

interface PipelineStageLayout {
  transformerLayers: number
  moeLayers: number
  boundaryLocal: number
}

interface ActivationMemoryDetails {
  activations: number
  logitsGradientPeakExtra: number
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
const POST_TRAINING_ROLLOUT_BYTES_PER_TOKEN = 16

function clampDegree(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.max(1, Math.floor(value)) : 1
}

function getAttentionHeadDim(arch: ModelArchitecture): number {
  const explicitHeadDim = arch.d_head

  if (explicitHeadDim !== null && explicitHeadDim !== undefined) {
    return typeof explicitHeadDim === "number" &&
      Number.isFinite(explicitHeadDim) &&
      explicitHeadDim > 0 &&
      Number.isInteger(explicitHeadDim)
      ? explicitHeadDim
      : Number.POSITIVE_INFINITY
  }

  return arch.d / arch.a
}

function getAttentionProjectionWidth(arch: ModelArchitecture): number {
  return arch.a * getAttentionHeadDim(arch)
}

function getKVProjectionWidth(arch: ModelArchitecture): number {
  const kvHeads = arch.a_kv ?? arch.a
  return kvHeads * getAttentionHeadDim(arch)
}

function getPartialCheckpointDepth(config: TrainingConfig): number {
  const depth = config.partialCheckpointDepth ?? 0
  return Number.isFinite(depth) ? Math.max(0, Math.floor(depth)) : 0
}

function canUseInterleavedPipelineSchedule(
  N_pp: number,
  numMicrobatches: number,
  VP: number
): boolean {
  return N_pp > 1 && VP > 1 && numMicrobatches % N_pp === 0
}

function getTrainingActivationBytes(config: TrainingConfig): number {
  return config.precision === "fp32" ? 4 : 2
}

function usesAMPAutocastActivationCorrections(config: TrainingConfig): boolean {
  return config.ampAutocast && config.precision !== "fp32"
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

function applyAMPAutocastOptimizerProfile(
  profile: OptimizerValues,
  config: TrainingConfig
): OptimizerValues {
  if (!config.ampAutocast) {
    return profile
  }

  const kOpt = Math.max(0, profile.kOpt - profile.masterWeightBytes)
  const optimizerStateBytes = Math.max(
    0,
    profile.optimizerStateBytes - profile.masterWeightBytes
  )

  return {
    ...profile,
    parameterBytes: 4,
    masterWeightBytes: 0,
    optimizerStateBytes,
    kOpt,
    phi: 4 + profile.betaGrad + kOpt,
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

function applyFSDPMixedPrecisionOptimizerProfile(
  profile: OptimizerValues,
  config: TrainingConfig
): OptimizerValues {
  if (!usesFSDPMixedPrecision(config)) {
    return profile
  }

  const kOpt = Math.max(0, profile.kOpt - profile.masterWeightBytes)
  const optimizerStateBytes = Math.max(
    0,
    profile.optimizerStateBytes - profile.masterWeightBytes
  )

  return {
    ...profile,
    parameterBytes: 4,
    masterWeightBytes: 0,
    optimizerStateBytes,
    kOpt,
    phi: 4 + profile.betaGrad + kOpt,
  }
}

function applyFP32PrecisionOptimizerProfile(
  profile: OptimizerValues,
  precision: TrainingConfig["precision"] | PostTrainingConfig["precision"]
): OptimizerValues {
  if (precision !== "fp32") {
    return profile
  }

  const kOpt = Math.max(0, profile.kOpt - profile.masterWeightBytes)
  const optimizerStateBytes = Math.max(
    0,
    profile.optimizerStateBytes - profile.masterWeightBytes
  )
  const betaGrad = profile.betaGrad > 0 ? 4 : 0

  return {
    ...profile,
    parameterBytes: 4,
    betaGrad,
    masterWeightBytes: 0,
    optimizerStateBytes,
    kOpt,
    phi: 4 + betaGrad + kOpt,
  }
}

function getTensorParallelPaddedVocabSize(V: number, N_tp: number): number {
  if (N_tp <= 1) {
    return V
  }

  const alignment = 128 * N_tp
  return Math.ceil(V / alignment) * alignment
}

function getPostTrainingPerGpuBatch(
  config: PostTrainingConfig,
  multiplier = 1,
): number {
  const batch = Number.isFinite(config.batchSize) && config.batchSize > 0
    ? Math.max(1, Math.ceil(config.batchSize))
    : 0
  const totalBatch = batch * Math.max(1, multiplier)
  let numGPUs = 1
  if (
    !config.hardware.gpu.singleDeviceOnly &&
    Number.isFinite(config.hardware.numGPUs) &&
    config.hardware.numGPUs > 0
  ) {
    numGPUs = Math.max(1, Math.floor(config.hardware.numGPUs))
  }

  return totalBatch > 0 ? Math.max(1, Math.ceil(totalBatch / numGPUs)) : 0
}

function applyTrainingOptimizerProfileAdjustments(
  profile: OptimizerValues,
  config: TrainingConfig
): OptimizerValues {
  return applyFP32PrecisionOptimizerProfile(
    applyFSDPMixedPrecisionOptimizerProfile(
      applyAMPAutocastOptimizerProfile(profile, config),
      config
    ),
    config.precision
  )
}

function resolveTrainingOptimizerProfile(config: TrainingConfig): OptimizerValues {
  const optimizer = resolvePretrainingOptimizer(config.optimizer)

  if (
    optimizer === "adamw-fp8" &&
    (config.precision !== "fp8" ||
      !config.hardware.gpu.supportsFP8 ||
      config.fp8.storageMode === "transformer-engine")
  ) {
    return applyTrainingOptimizerProfileAdjustments(
      getOptimizerProfile("adamw-mixed", config.gradientPrecision),
      config
    )
  }

  return applyTrainingOptimizerProfileAdjustments(
    getOptimizerProfile(optimizer, config.gradientPrecision),
    config
  )
}

export function resolvePostTrainingOptimizerProfile(
  config: PostTrainingConfig
): OptimizerValues {
  if (
    config.optimizer === "adamw-fp8" &&
    (config.precision !== "fp8" ||
      !config.hardware.gpu.supportsFP8 ||
      config.fp8.storageMode === "transformer-engine")
  ) {
    return applyFP32PrecisionOptimizerProfile(
      getOptimizerProfile("adamw-mixed", config.gradientPrecision),
      config.precision
    )
  }

  return applyFP32PrecisionOptimizerProfile(
    getOptimizerProfile(config.optimizer, config.gradientPrecision),
    config.precision
  )
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

function getSequenceParallelActivationShardDegree(
  parallelism: ParallelismConfig
): number {
  return isSequenceParallelEnabled(parallelism)
    ? clampDegree(parallelism.N_tp)
    : 1
}

function isSwiGLUStyle(ffnType: ModelArchitecture["ffnType"]): boolean {
  return ffnType === "swiglu" || ffnType === "geglu" || ffnType === "moe"
}

function resolveDefaultIntermediateSize(
  arch: ModelArchitecture,
  swiGLUStyle = isSwiGLUStyle(arch.ffnType)
): number {
  if (arch.d_ff !== null) {
    return arch.d_ff
  }

  return swiGLUStyle ? Math.round((8 / 3) * arch.d) : 4 * arch.d
}

export function calculateDenseStateShardDegree(config: TrainingConfig): number {
  const N_dp = clampDegree(config.parallelism.N_dp)
  const N_cp = clampDegree(config.parallelism.N_cp)
  const replicaShardDegree = N_dp * N_cp

  // Context parallelism splits tokens while leaving dense weights duplicated
  // across CP ranks. Megatron folds those ranks into the DP communication group
  // for model-state sharding; Megatron-style sequence parallelism does not add
  // another shard factor because it is the TP rank group itself.
  if (!usesHybridShard(config)) {
    return replicaShardDegree
  }

  const localNonReplicaRanks =
    clampDegree(config.parallelism.N_tp) * clampDegree(config.parallelism.N_pp)
  const localReplicaCapacity = Math.max(
    1,
    Math.floor(
      getParallelismLocalGroupSize(config.hardware.gpu) / localNonReplicaRanks,
    ),
  )

  return Math.min(replicaShardDegree, localReplicaCapacity)
}

function getStateShardDegree(config: TrainingConfig): number {
  return calculateDenseStateShardDegree(config)
}

function getNonExpertOptimizerShardDegree(config: TrainingConfig): number {
  return getStateShardDegree(config)
}

function getExpertDataParallelDegree(
  config: TrainingConfig,
  stateShardDegree = getStateShardDegree(config)
): number {
  const N_tp = clampDegree(config.parallelism.N_tp)
  const N_ep = clampDegree(config.parallelism.N_ep)

  // Spec Section 5.2: routed expert states use the expert data-parallel group,
  // N_edp = N_dp x N_cp x N_tp / N_ep. Shared experts are replicated on EP ranks
  // and follow the dense replica shard group instead.
  return Math.max(1, (stateShardDegree * N_tp) / N_ep)
}

function applyCPUOffload(
  memory: ModelStateMemoryResult,
  cpuOffload: CPUOffloadMode,
  zeroStage: ZeROStage
): ModelStateMemoryResult {
  if (cpuOffload === "optimizer-only" && zeroStage >= 1) {
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

function getDeepSpeedZeRO2GradientUpcastBytesPerParam(
  optimizer: OptimizerValues,
  config: TrainingConfig,
  zeroStage: ZeROStage
): number {
  if (
    config.parallelism.framework !== "deepspeed" ||
    zeroStage !== 2 ||
    config.cpuOffload !== "none" ||
    config.ampAutocast ||
    config.precision === "fp32" ||
    optimizer.betaGrad <= 0
  ) {
    return 0
  }

  // Spec Section 5.2 models DeepSpeed FusedAdam's ZeRO-2 optimizer-step
  // upcast as an extra sharded 2-byte gradient buffer.
  return 2
}

function calculateStateGroupMemory(
  parameterCount: number,
  optimizer: OptimizerValues,
  zeroStage: ZeROStage,
  stateShardDegree: number,
  optimizerShardDegree: number,
  gradientTransientBytesPerParam = 0
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
      const gradients =
        (parameterCount *
          (optimizer.betaGrad + gradientTransientBytesPerParam)) /
        stateDegree
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

function getOutputHeadParameterCount(params: ParameterCounts): number {
  return params.outputProjection > 0 ? params.outputProjection : params.embedding
}

function getPipelineBoundaryParameterCount(
  params: ParameterCounts,
  N_pp: number
): number {
  if (N_pp <= 1) {
    return getEmbeddingParameterCount(params)
  }

  const firstStage = params.embedding + params.positionalEmbedding
  const lastStage = getOutputHeadParameterCount(params) + params.finalNorm

  return Math.max(firstStage, lastStage)
}

function getLargestPipelineBoundaryParameterCount(params: ParameterCounts): number {
  const firstStage = params.embedding + params.positionalEmbedding
  const lastStage = getOutputHeadParameterCount(params) + params.finalNorm
  return Math.max(firstStage, lastStage)
}

function uniqueStageMoELayerCountsForLayerCount(
  totalLayers: number,
  moeLayers: number,
  transformerLayers: number
): number[] {
  if (transformerLayers <= 0 || totalLayers <= 0 || moeLayers <= 0) {
    return [0]
  }

  const boundedMoELayers = Math.min(Math.max(0, moeLayers), totalLayers)
  const expectedMoELayers =
    (Math.min(transformerLayers, totalLayers) * boundedMoELayers) / totalLayers
  const minMoELayers = Math.min(
    transformerLayers,
    boundedMoELayers,
    Math.floor(expectedMoELayers)
  )
  const maxMoELayers = Math.min(
    transformerLayers,
    boundedMoELayers,
    Math.ceil(expectedMoELayers)
  )

  return Array.from(new Set([minMoELayers, maxMoELayers]))
}

function getPipelineTransformerLayerCandidates(
  totalLayers: number,
  N_pp: number
): Array<{ transformerLayers: number; boundary: "first" | "last" | "none" }> {
  const pipelineDegree = clampDegree(N_pp)

  if (pipelineDegree <= 1) {
    return [{ transformerLayers: totalLayers, boundary: "first" }]
  }

  if (totalLayers % pipelineDegree === 0) {
    const layersPerStage = totalLayers / pipelineDegree
    return [
      { transformerLayers: layersPerStage, boundary: "first" },
      { transformerLayers: layersPerStage, boundary: "last" },
      { transformerLayers: layersPerStage, boundary: "none" },
    ]
  }

  if ((totalLayers + 2) % pipelineDegree === 0) {
    const slotsPerStage = (totalLayers + 2) / pipelineDegree
    const candidates: Array<{
      transformerLayers: number
      boundary: "first" | "last" | "none"
    }> = [
      {
        transformerLayers: Math.max(0, slotsPerStage - 1),
        boundary: "first",
      },
      {
        transformerLayers: Math.max(0, slotsPerStage - 1),
        boundary: "last",
      },
    ]

    if (pipelineDegree > 2) {
      candidates.push({ transformerLayers: slotsPerStage, boundary: "none" })
    }

    return candidates
  }

  const lower = Math.floor(totalLayers / pipelineDegree)
  const upper = Math.ceil(totalLayers / pipelineDegree)

  return [
    { transformerLayers: lower, boundary: "first" },
    { transformerLayers: lower, boundary: "last" },
    { transformerLayers: lower, boundary: "none" },
    { transformerLayers: upper, boundary: "none" },
  ]
}

function getPipelineStageLayouts(
  params: ParameterCounts,
  arch: ModelArchitecture,
  moe: MoEConfig,
  N_pp: number,
  N_tp: number
): PipelineStageLayout[] {
  const firstBoundaryLocal =
    N_pp <= 1
      ? getEmbeddingParameterCount(params) / N_tp
      : (params.embedding + params.positionalEmbedding) / N_tp
  const lastBoundaryLocal =
    N_pp <= 1
      ? 0
      : (getOutputHeadParameterCount(params) + params.finalNorm) / N_tp
  const boundedMoELayers =
    moe.enabled ? Math.min(Math.max(0, moe.L_moe), arch.L) : 0
  const layouts = getPipelineTransformerLayerCandidates(arch.L, N_pp).flatMap(
    ({ transformerLayers, boundary }) => {
      const boundaryLocal =
        boundary === "first"
          ? firstBoundaryLocal
          : boundary === "last"
            ? lastBoundaryLocal
            : 0

      return uniqueStageMoELayerCountsForLayerCount(
        arch.L,
        boundedMoELayers,
        transformerLayers
      ).map((moeLayers) => ({
        transformerLayers,
        moeLayers,
        boundaryLocal,
      }))
    }
  )

  return Array.from(
    new Map(
      layouts.map((layout) => [
        `${layout.transformerLayers}:${layout.moeLayers}:${layout.boundaryLocal}`,
        layout,
      ])
    ).values()
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
  const embeddingLocal = getPipelineBoundaryParameterCount(params, N_pp) / N_tp
  const moeEnabled = config.model.moe.enabled && params.moe !== null
  const routedExpertTotal = moeEnabled ? (params.moe?.expertParameters ?? 0) : 0
  const sharedExpertTotal = moeEnabled
    ? (params.moe?.sharedExpertParameters ?? 0)
    : 0
  const routerTotal = moeEnabled ? (params.moe?.routerParameters ?? 0) : 0
  const boundedMoELayers = Math.min(
    Math.max(0, config.model.moe.L_moe),
    config.model.architecture.L
  )
  const stageLayouts = getPipelineStageLayouts(
    params,
    config.model.architecture,
    config.model.moe,
    N_pp,
    N_tp
  )
  const commonPerLayer = params.perLayer.attention + params.perLayer.norm
  const denseFFNPerLayer = params.perLayer.ffn
  const routedExpertPerMoELayer =
    moeEnabled && boundedMoELayers > 0
      ? routedExpertTotal / boundedMoELayers
      : 0
  const sharedExpertPerMoELayer =
    moeEnabled && boundedMoELayers > 0
      ? sharedExpertTotal / boundedMoELayers
      : 0
  const routerPerMoELayer =
    moeEnabled && boundedMoELayers > 0
      ? routerTotal / boundedMoELayers
      : 0
  const stages = stageLayouts.map((layout): ParameterStagePartitioning => {
    const denseLayers = Math.max(0, layout.transformerLayers - layout.moeLayers)
    const nonExpertLocal =
      (layout.transformerLayers * commonPerLayer + denseLayers * denseFFNPerLayer) /
        N_tp +
      layout.boundaryLocal
    const routedExpertLocal =
      layout.moeLayers > 0
        ? (layout.moeLayers * routedExpertPerMoELayer) / N_ep
        : 0
    const sharedExpertLocal =
      layout.moeLayers > 0 ? layout.moeLayers * sharedExpertPerMoELayer : 0
    const routerLocal =
      layout.moeLayers > 0 ? layout.moeLayers * routerPerMoELayer : 0

    return {
      nonExpertLocal,
      moeLocal: routedExpertLocal + sharedExpertLocal + routerLocal,
      routedExpertLocal,
      sharedExpertLocal,
      routerLocal,
    }
  })
  const peakByLocalCount = stages.reduce((peak, stage) =>
    stage.nonExpertLocal + stage.moeLocal > peak.nonExpertLocal + peak.moeLocal
      ? stage
      : peak
  )

  return {
    embeddingTotal,
    embeddingLocal,
    nonExpertLocal: peakByLocalCount.nonExpertLocal,
    moeLocal: peakByLocalCount.moeLocal,
    routedExpertLocal: peakByLocalCount.routedExpertLocal,
    sharedExpertLocal: peakByLocalCount.sharedExpertLocal,
    routerLocal: peakByLocalCount.routerLocal,
    stages,
  }
}

function getLargestLayerParameterCount(
  params: ParameterCounts,
  config: TrainingConfig
): number {
  const N_tp = clampDegree(config.parallelism.N_tp)
  const moeLayers =
    config.model.moe.enabled && params.moe !== null
      ? Math.min(
          Math.max(0, config.model.moe.L_moe),
          config.model.architecture.L
        )
      : 0
  const denseLayers = Math.max(config.model.architecture.L - moeLayers, 0)
  const denseLayer =
    denseLayers > 0
      ? (params.perLayer.attention + params.perLayer.ffn + params.perLayer.norm) /
        N_tp
      : 0

  if (
    !config.model.moe.enabled ||
    params.moe === null ||
    moeLayers <= 0
  ) {
    return denseLayer
  }

  const N_ep = clampDegree(config.parallelism.N_ep)
  const routerPerLayer = params.moe.routerParameters / moeLayers
  const sharedExpertsPerLayer = params.moe.sharedExpertParameters / moeLayers
  const routedExpertsPerLayer = params.moe.expertParameters / moeLayers

  const moeLayer =
    (params.perLayer.attention + params.perLayer.norm) / N_tp +
    routerPerLayer +
    sharedExpertsPerLayer +
    routedExpertsPerLayer / N_ep

  return Math.max(denseLayer, moeLayer)
}

function resolveDenseIntermediateSize(
  arch: ModelArchitecture,
  moe: MoEConfig
): number {
  return moe.enabled
    ? (moe.denseIntermediateSize ?? resolveDefaultIntermediateSize(arch))
    : resolveDefaultIntermediateSize(arch)
}

function resolveExpertIntermediateSize(
  arch: ModelArchitecture,
  moe: MoEConfig
): number {
  return moe.expertIntermediateSize ?? resolveDefaultIntermediateSize(arch, true)
}

function getMoEFFNActivationScale(
  config: TrainingConfig,
  moe: MoEConfig
): number {
  const N_ep = clampDegree(config.parallelism.N_ep)
  const routedExpertsPerToken =
    Number.isFinite(moe.topk) && Number.isFinite(moe.E)
      ? Math.min(Math.max(0, moe.topk), Math.max(0, moe.E))
      : 0
  const sharedExpertsPerToken = Number.isFinite(moe.E_s)
    ? Math.max(0, moe.E_s)
    : 0
  const loadBalanceFactor = Number.isFinite(moe.loadBalanceFactor)
    ? Math.max(1, moe.loadBalanceFactor)
    : 1

  // Routed expert activations are distributed over EP ranks. Shared experts
  // are present on every EP rank in this calculator's MoE state model.
  return (routedExpertsPerToken / N_ep) * loadBalanceFactor + sharedExpertsPerToken
}

function getMoEExpertActivationShardDegree(config: TrainingConfig): number {
  return isSequenceParallelEnabled(config.parallelism)
    ? clampDegree(config.parallelism.N_tp)
    : 1
}

function getMoERoutedExpertsPerToken(moe: MoEConfig): number {
  return Number.isFinite(moe.topk) && Number.isFinite(moe.E)
    ? Math.min(Math.max(0, moe.topk), Math.max(0, moe.E))
    : 0
}

function calculateMoEDispatchMaskBytes(
  config: TrainingConfig,
  moe: MoEConfig
): number {
  const routedExpertsPerToken = getMoERoutedExpertsPerToken(moe)

  if (!moe.enabled || routedExpertsPerToken <= 0) {
    return 0
  }

  const sequenceShardDegree = getSequenceParallelActivationShardDegree(
    config.parallelism
  )
  const sequenceLengthPerRank =
    config.sequenceLength /
    (clampDegree(config.parallelism.N_cp) * sequenceShardDegree)

  return 2 * config.microBatchSize * sequenceLengthPerRank * routedExpertsPerToken
}

function getStoredActivationCoefficientScale(config: TrainingConfig): number {
  if (usesAMPAutocastActivationCorrections(config)) {
    return 1
  }

  return config.precision === "fp32" ? 2 : 1
}

function getAttentionLinearActivationCoefficient(arch: ModelArchitecture): number {
  const queryRatio = getAttentionProjectionWidth(arch) / arch.d
  const kvRatio = getKVProjectionWidth(arch) / arch.d

  // Q and attention-output activations follow a*d_head; K and V follow
  // a_kv*d_head, which can differ from d_model for PaLM-style heads.
  return 4 * queryRatio + 4 * kvRatio
}

function getActivationCoefficients(
  arch: ModelArchitecture,
  config: TrainingConfig,
  checkpointing: CheckpointingMode,
  ffnWidth: number,
  ffnActivationShardDegree = clampDegree(config.parallelism.N_tp)
): ActivationCoefficients {
  const N_tp = clampDegree(config.parallelism.N_tp)
  const useAMPCorrections = usesAMPAutocastActivationCorrections(config)
  const ampLinearDelta = useAMPCorrections ? 2 : 0
  const attentionCoefficient = useAMPCorrections ? 6 : 5
  const attentionLinear = getAttentionLinearActivationCoefficient(arch)
  const attentionKeyLength = config.sequenceLength
  const attentionQuadratic =
    checkpointing === "full" || checkpointing === "selective" || config.flashAttention
      ? 0
      : (attentionCoefficient * arch.a * attentionKeyLength) / (arch.d * N_tp)

  const ffnLinear = (4 * ffnWidth) / (arch.d * ffnActivationShardDegree)

  if (N_tp === 1) {
    return {
      nonFFNLinear: 10 + ampLinearDelta + attentionLinear,
      ffnLinear,
      attentionQuadratic,
    }
  }

  if (isSequenceParallelEnabled(config.parallelism)) {
    return {
      nonFFNLinear: (10 + ampLinearDelta + attentionLinear) / N_tp,
      ffnLinear,
      attentionQuadratic,
    }
  }

  return {
    nonFFNLinear: 10 + ampLinearDelta + attentionLinear / N_tp,
    ffnLinear,
    attentionQuadratic,
  }
}

function calculateStoredActivationPerLayer(
  arch: ModelArchitecture,
  config: TrainingConfig,
  checkpointing: CheckpointingMode,
  ffnWidth: number,
  moeFFNScale: number,
  ffnActivationShardDegree?: number
): number {
  const N_cp = clampDegree(config.parallelism.N_cp)
  const sequenceLengthPerRank = config.sequenceLength / N_cp
  const baseElements = sequenceLengthPerRank * config.microBatchSize * arch.d

  if (checkpointing === "full") {
    const sequenceShardDegree = getSequenceParallelActivationShardDegree(
      config.parallelism
    )

    return (
      (baseElements / sequenceShardDegree) *
      getTrainingActivationBytes(config)
    )
  }

  const coefficients = getActivationCoefficients(
    arch,
    config,
    checkpointing,
    ffnWidth,
    ffnActivationShardDegree
  )
  const N_tp = clampDegree(config.parallelism.N_tp)
  const flashAttentionStatsBytes = config.flashAttention
    ? (4 * arch.a * sequenceLengthPerRank * config.microBatchSize) / N_tp
    : 0

  return (
    baseElements *
      (coefficients.nonFFNLinear +
        coefficients.ffnLinear * moeFFNScale +
        coefficients.attentionQuadratic) *
      getStoredActivationCoefficientScale(config) +
    flashAttentionStatsBytes
  )
}

function calculateMoEStoredActivationPerLayer(
  arch: ModelArchitecture,
  config: TrainingConfig,
  checkpointing: CheckpointingMode,
  moe: MoEConfig,
  expertFFNWidth: number
): number {
  const stored = calculateStoredActivationPerLayer(
    arch,
    config,
    checkpointing,
    expertFFNWidth,
    getMoEFFNActivationScale(config, moe),
    getMoEExpertActivationShardDegree(config)
  )

  if (checkpointing !== "full") {
    return stored
  }

  // MoE routing decisions must be replayed exactly during backward, so the
  // dispatch mask remains resident even when the expert block is recomputed.
  return stored + calculateMoEDispatchMaskBytes(config, moe)
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
    getMoEFFNActivationScale(config, moe),
    getMoEExpertActivationShardDegree(config)
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
  const N_tp = clampDegree(config.parallelism.N_tp)
  const paddedVocab = getTensorParallelPaddedVocabSize(arch.V, N_tp)

  return (
    config.microBatchSize *
    (config.sequenceLength / N_cp) *
    (paddedVocab / N_tp) *
    getTrainingActivationBytes(config)
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
  const paddedVocab = getTensorParallelPaddedVocabSize(arch.V, N_tp)

  return (
    (4 * config.microBatchSize * (config.sequenceLength / N_cp) * paddedVocab) /
    N_tp
  )
}

function calculateExpertParallelRoutingBufferBytes(
  arch: ModelArchitecture,
  config: TrainingConfig
): number {
  const moe = config.model.moe
  const N_ep = clampDegree(config.parallelism.N_ep)

  if (!moe.enabled || N_ep <= 1 || moe.L_moe <= 0) {
    return 0
  }

  const routedExpertsPerToken =
    Number.isFinite(moe.topk) && Number.isFinite(moe.E)
      ? Math.min(Math.max(0, moe.topk), Math.max(0, moe.E))
      : 0

  if (routedExpertsPerToken <= 0) {
    return 0
  }

  const N_cp = clampDegree(config.parallelism.N_cp)
  const sequenceShardDegree = getSequenceParallelActivationShardDegree(
    config.parallelism
  )
  const sequenceLengthPerRank =
    config.sequenceLength / (N_cp * sequenceShardDegree)
  const activationBytes = getTrainingActivationBytes(config)
  const loadBalanceFactor = Number.isFinite(moe.loadBalanceFactor)
    ? Math.max(1, moe.loadBalanceFactor)
    : 1
  const perDirectionVolume =
    (routedExpertsPerToken / N_ep) *
    config.microBatchSize *
    sequenceLengthPerRank *
    arch.d *
    ((N_ep - 1) / N_ep) *
    activationBytes *
    loadBalanceFactor

  // Peak residency for one routed MoE layer: an all-to-all may hold both send
  // and receive staging buffers. Dispatch and combine do not coexist, so this
  // is not multiplied by the number of MoE layers.
  return 2 * perDirectionVolume
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

function resolvePrefetchBucketSizeElements(config: TrainingConfig): number {
  if (config.zeroCommunication.mode === "deepspeed-defaults") {
    return 5e7
  }

  if (
    config.zeroCommunication.mode === "custom" &&
    config.zeroCommunication.prefetchBucketSizeElements !== null
  ) {
    return config.zeroCommunication.prefetchBucketSizeElements
  }

  return 0.9 * config.model.architecture.d * config.model.architecture.d
}

function calculateTrainableModelStates(
  parameterCount: number,
  optimizer: OptimizerValues
): ModelStateMemoryResult {
  const safeParameterCount = getPositiveParameterCountOrInfinity(parameterCount)
  const parameters = multiplyParameterBytes(
    safeParameterCount,
    optimizer.parameterBytes
  )
  const gradients = multiplyParameterBytes(safeParameterCount, optimizer.betaGrad)
  const optimizerStates = multiplyParameterBytes(safeParameterCount, optimizer.kOpt)

  return {
    parameters,
    gradients,
    optimizerStates,
    total: parameters + gradients + optimizerStates,
  }
}

function getPositiveParameterCountOrInfinity(parameterCount: number): number {
  return Number.isFinite(parameterCount) && parameterCount > 0
    ? parameterCount
    : Number.POSITIVE_INFINITY
}

function multiplyParameterBytes(parameterCount: number, bytesPerParameter: number): number {
  if (!Number.isFinite(bytesPerParameter) || bytesPerParameter <= 0) {
    return 0
  }

  return parameterCount * bytesPerParameter
}

function resolvePostTrainingTrainableParameterCount(
  config: PostTrainingConfig,
  parameterCount = config.baseModel.parameterCount
): number {
  const safeParameterCount = getPositiveParameterCountOrInfinity(parameterCount)
  const percentage = config.trainableParameterPercentage
  const trainableFraction =
    percentage === null || !Number.isFinite(percentage) || percentage <= 0
      ? 1
      : Math.min(percentage, 100) / 100

  return safeParameterCount * trainableFraction
}

function calculatePartiallyTrainableModelStates(
  parameterCount: number,
  trainableParameterCount: number,
  optimizer: OptimizerValues
) {
  const safeParameterCount = getPositiveParameterCountOrInfinity(parameterCount)
  const trainableCount =
    Number.isFinite(safeParameterCount)
      ? Number.isFinite(trainableParameterCount) && trainableParameterCount > 0
        ? Math.min(trainableParameterCount, safeParameterCount)
        : safeParameterCount
      : Number.POSITIVE_INFINITY
  const frozenCount =
    Number.isFinite(safeParameterCount) && Number.isFinite(trainableCount)
      ? Math.max(safeParameterCount - trainableCount, 0)
      : 0
  const trainableParameters = multiplyParameterBytes(
    trainableCount,
    optimizer.parameterBytes
  )
  const frozenParameters = multiplyParameterBytes(
    frozenCount,
    optimizer.parameterBytes
  )
  const gradients = multiplyParameterBytes(trainableCount, optimizer.betaGrad)
  const optimizerStates = multiplyParameterBytes(trainableCount, optimizer.kOpt)

  return {
    parameters: trainableParameters + frozenParameters,
    trainableParameters,
    frozenParameters,
    gradients,
    optimizerStates,
    trainableTotal: trainableParameters + gradients + optimizerStates,
    frozenTotal: frozenParameters,
  }
}

const NF4_DOUBLE_QUANT_BYTES_PER_PARAM = 0.5159
const QLORA_SIMPLE_NF4_BYTES_PER_PARAM = 0.55
const INT8_QUANTIZED_BYTES_PER_PARAM = 1.01

function formatQLoRAQuantizationLabel(quantizationBits: 4 | 8 | null): string {
  return quantizationBits === 8
    ? "8-bit quantized (LLM.int8/GPTQ/AWQ)"
    : "4-bit quantized (NF4/GPTQ/AWQ)"
}

function calculateQLoRANonQuantizedParameterCount(
  config: PostTrainingConfig
): number | null {
  const parameterCount = config.baseModel.parameterCount
  const counts = calculateParameterCount(
    config.baseModel.architecture,
    config.baseModel.moe,
    config.sequenceLength
  )

  if (
    !Number.isFinite(parameterCount) ||
    parameterCount <= 0 ||
    !Number.isFinite(counts.total) ||
    counts.total <= 0
  ) {
    return null
  }

  const nonQuantizedParams =
    counts.embedding +
    counts.outputProjection +
    counts.positionalEmbedding +
    counts.finalNorm +
    counts.perLayer.norm * config.baseModel.architecture.L
  const scaledNonQuantizedParams =
    nonQuantizedParams * (parameterCount / counts.total)

  return Math.max(0, Math.min(parameterCount, scaledNonQuantizedParams))
}

export function calculateQuantizedBaseModelBytes(
  config: PostTrainingConfig,
  quantizationBits: 4 | 8 | null
): number {
  const parameterCount = config.baseModel.parameterCount

  if (!Number.isFinite(parameterCount) || parameterCount <= 0) {
    return Number.POSITIVE_INFINITY
  }

  if (quantizationBits === 8) {
    const nonQuantizedParams = calculateQLoRANonQuantizedParameterCount(config)

    if (nonQuantizedParams === null) {
      return parameterCount * INT8_QUANTIZED_BYTES_PER_PARAM
    }

    const quantizedParams = Math.max(0, parameterCount - nonQuantizedParams)

    return (
      quantizedParams * INT8_QUANTIZED_BYTES_PER_PARAM +
      nonQuantizedParams * getPostTrainingWeightBytes(config)
    )
  }

  const nonQuantizedParams = calculateQLoRANonQuantizedParameterCount(config)

  if (nonQuantizedParams === null) {
    return parameterCount * QLORA_SIMPLE_NF4_BYTES_PER_PARAM
  }

  const quantizedParams = Math.max(0, parameterCount - nonQuantizedParams)

  return (
    quantizedParams * NF4_DOUBLE_QUANT_BYTES_PER_PARAM +
    nonQuantizedParams * getPostTrainingWeightBytes(config)
  )
}

function calculateQLoRAOutputHeadParameterCount(
  config: PostTrainingConfig,
  counts: ParameterCounts
): number {
  const outputProjectionForCompute =
    config.baseModel.architecture.V * config.baseModel.architecture.d

  if (
    Number.isFinite(outputProjectionForCompute) &&
    outputProjectionForCompute > 0
  ) {
    return outputProjectionForCompute
  }

  return counts.outputProjection > 0 ? counts.outputProjection : counts.embedding
}

export function calculateQuantizedActiveModelBytesPerParam(
  config: PostTrainingConfig,
  quantizationBits: 4 | 8 | null
): number | null {
  const parameterCount = config.baseModel.parameterCount
  const counts = calculateParameterCount(
    config.baseModel.architecture,
    config.baseModel.moe,
    config.sequenceLength
  )

  if (
    !Number.isFinite(parameterCount) ||
    parameterCount <= 0 ||
    !Number.isFinite(counts.total) ||
    counts.total <= 0 ||
    !Number.isFinite(counts.active) ||
    counts.active <= 0
  ) {
    return null
  }

  const activeParameterCount =
    config.baseModel.moe.enabled &&
    Number.isFinite(config.baseModel.moe.activeParameterCount) &&
    config.baseModel.moe.activeParameterCount !== null &&
    config.baseModel.moe.activeParameterCount > 0
      ? config.baseModel.moe.activeParameterCount
      : counts.active * (parameterCount / counts.total)

  if (
    !Number.isFinite(activeParameterCount) ||
    activeParameterCount <= 0
  ) {
    return null
  }

  const activeScale = activeParameterCount / counts.active
  // Decode streams output-head and layer-norm weights, but not input-only
  // embedding tables. Tied embeddings still serve as the output head.
  const activeNonQuantizedParams = Math.max(
    0,
    Math.min(
      activeParameterCount,
      (calculateQLoRAOutputHeadParameterCount(config, counts) +
        counts.perLayer.norm * config.baseModel.architecture.L) *
        activeScale
    )
  )
  const activeQuantizedParams = Math.max(
    0,
    activeParameterCount - activeNonQuantizedParams
  )
  const quantizedBytesPerParam =
    quantizationBits === 8
      ? INT8_QUANTIZED_BYTES_PER_PARAM
      : NF4_DOUBLE_QUANT_BYTES_PER_PARAM
  const activeBytes =
    activeQuantizedParams * quantizedBytesPerParam +
    activeNonQuantizedParams * getPostTrainingWeightBytes(config)

  return activeBytes / activeParameterCount
}

function getPostTrainingMoEFFNActivationScale(moe: MoEConfig): number {
  const routedExpertsPerToken =
    Number.isFinite(moe.topk) && Number.isFinite(moe.E)
      ? Math.min(Math.max(0, moe.topk), Math.max(0, moe.E))
      : 0
  const sharedExpertsPerToken = Number.isFinite(moe.E_s)
    ? Math.max(0, moe.E_s)
    : 0
  const loadBalanceFactor = Number.isFinite(moe.loadBalanceFactor)
    ? Math.max(1, moe.loadBalanceFactor)
    : 1

  return routedExpertsPerToken * loadBalanceFactor + sharedExpertsPerToken
}

function getPostTrainingAttentionQuadraticActivationCoefficient(
  arch: ModelArchitecture,
  config: PostTrainingConfig
): number {
  return (5 * arch.a * config.sequenceLength) / arch.d
}

export function calculatePostTrainingActivationMemory(
  arch: ModelArchitecture,
  config: PostTrainingConfig,
  batchMultiplier = 1
): number {
  return (
    calculatePostTrainingTransformerActivationMemory(
      arch,
      config,
      batchMultiplier
    ) +
    calculatePostTrainingOutputLogitsMemory(arch, config, batchMultiplier) +
    calculatePostTrainingLogitsGradientMemory(arch, config, batchMultiplier)
  )
}

function calculatePostTrainingTransformerActivationMemory(
  arch: ModelArchitecture,
  config: PostTrainingConfig,
  batchMultiplier = 1
): number {
  const perGpuBatch = getPostTrainingPerGpuBatch(config, batchMultiplier)
  const moe = config.baseModel.moe
  const boundedMoELayers =
    moe.enabled && moe.L_moe > 0
      ? Math.min(Math.max(0, moe.L_moe), arch.L)
      : 0
  const storedCheckpoints =
    arch.L *
    config.sequenceLength *
    perGpuBatch *
    arch.d *
    getPostTrainingActivationBytes(config)
  const moeDispatchMasks =
    boundedMoELayers *
    2 *
    config.sequenceLength *
    perGpuBatch *
    getMoERoutedExpertsPerToken(moe)

  return (
    storedCheckpoints +
    moeDispatchMasks +
    calculatePostTrainingForwardWorkingMemory(arch, config, batchMultiplier)
  )
}

export function calculatePostTrainingOutputLogitsMemory(
  arch: ModelArchitecture,
  config: PostTrainingConfig,
  batchMultiplier = 1
): number {
  if (config.chunkedCrossEntropy) {
    return 0
  }

  const perGpuBatch = getPostTrainingPerGpuBatch(config, batchMultiplier)

  return (
    perGpuBatch *
    config.sequenceLength *
    arch.V *
    getPostTrainingActivationBytes(config)
  )
}

function calculatePostTrainingLogitsGradientMemory(
  arch: ModelArchitecture,
  config: PostTrainingConfig,
  batchMultiplier = 1
): number {
  if (config.chunkedCrossEntropy) {
    return 0
  }

  const perGpuBatch = getPostTrainingPerGpuBatch(config, batchMultiplier)

  return perGpuBatch * config.sequenceLength * arch.V * 4
}

export function calculatePostTrainingForwardWorkingMemory(
  arch: ModelArchitecture,
  config: PostTrainingConfig,
  batchMultiplier = 1
): number {
  const perGpuBatch = getPostTrainingPerGpuBatch(config, batchMultiplier)
  const activationBytes = getPostTrainingActivationBytes(config)
  const baseElements = config.sequenceLength * perGpuBatch * arch.d
  const bytesScale = activationBytes / 2
  const denseFFNWidth = resolveDenseIntermediateSize(arch, config.baseModel.moe)
  const nonFFNLinear = 10 + getAttentionLinearActivationCoefficient(arch)
  const attentionQuadratic =
    getPostTrainingAttentionQuadraticActivationCoefficient(arch, config)
  const denseWorking =
    baseElements *
    bytesScale *
    (nonFFNLinear + (4 * denseFFNWidth) / arch.d + attentionQuadratic)
  const moe = config.baseModel.moe

  if (!moe.enabled || moe.E <= 0 || moe.L_moe <= 0) {
    return denseWorking
  }

  const expertFFNWidth = resolveExpertIntermediateSize(arch, moe)
  const expertWorking =
    baseElements *
    bytesScale *
    (nonFFNLinear +
      ((4 * expertFFNWidth) / arch.d) *
        getPostTrainingMoEFFNActivationScale(moe) +
      attentionQuadratic)

  return Math.max(denseWorking, expertWorking)
}

function calculateKVCacheBytes(
  arch: ModelArchitecture,
  batch: number,
  sequenceLength: number,
  precision: PostTrainingConfig["kvCachePrecision"]
): number {
  const kvHeads = arch.a_kv ?? arch.a
  const headDim = getAttentionHeadDim(arch)

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

function calculatePostTrainingRolloutBufferBytes(
  sequenceLength: number,
  localBatch: number
): number {
  return POST_TRAINING_ROLLOUT_BYTES_PER_TOKEN * sequenceLength * localBatch
}

interface PostTrainingGenerationWorkingSet {
  total: number
  kvCacheBytes: number
}

function calculatePostTrainingPeakGenerationWorkingSet({
  gpu,
  parameters,
  gradients,
  optimizerStates,
  frameworkOverhead,
  rolloutBuffers,
  kvCacheBytes,
  requestedLocalGenerationBatch,
}: {
  gpu: GPUSpec
  parameters: number
  gradients: number
  optimizerStates: number
  frameworkOverhead: number
  rolloutBuffers: number
  kvCacheBytes: number
  requestedLocalGenerationBatch: number
}): PostTrainingGenerationWorkingSet {
  const fullGenerationWorkingSet = rolloutBuffers + kvCacheBytes
  const kvBytesPerGeneration =
    requestedLocalGenerationBatch > 0
      ? kvCacheBytes / requestedLocalGenerationBatch
      : 0

  if (!Number.isFinite(kvBytesPerGeneration) || kvBytesPerGeneration <= 0) {
    return {
      total: fullGenerationWorkingSet,
      kvCacheBytes,
    }
  }

  const usableCapacity = gpu.memoryGB * 1e9 * 0.9
  const availableForRoundKV =
    usableCapacity / 1.04 -
    parameters -
    gradients -
    optimizerStates -
    frameworkOverhead -
    rolloutBuffers

  if (
    !Number.isFinite(availableForRoundKV) ||
    availableForRoundKV < kvBytesPerGeneration
  ) {
    return {
      total: fullGenerationWorkingSet,
      kvCacheBytes,
    }
  }

  const peakLocalGenerationBatch = Math.min(
    requestedLocalGenerationBatch,
    Math.max(1, Math.floor(availableForRoundKV / kvBytesPerGeneration))
  )
  const peakKVCacheBytes = kvBytesPerGeneration * peakLocalGenerationBatch

  return {
    total: rolloutBuffers + peakKVCacheBytes,
    kvCacheBytes: peakKVCacheBytes,
  }
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
  const expertShardDegree = getExpertDataParallelDegree(config, stateShardDegree)
  const gradientTransientBytesPerParam =
    getDeepSpeedZeRO2GradientUpcastBytesPerParam(optimizer, config, zeroStage)
  const stageMemories = partitioning.stages.map((stage) => {
    const nonExpertMemory = calculateStateGroupMemory(
      stage.nonExpertLocal + stage.routerLocal,
      optimizer,
      zeroStage,
      stateShardDegree,
      optimizerShardDegree,
      gradientTransientBytesPerParam
    )
    let stageMemory = nonExpertMemory

    if (!config.model.moe.enabled || params.moe === null || stage.moeLocal <= 0) {
      return applyCPUOffload(stageMemory, config.cpuOffload, zeroStage)
    }

    if (stage.routedExpertLocal > 0) {
      stageMemory = addModelStateMemory(
        stageMemory,
        calculateStateGroupMemory(
          stage.routedExpertLocal,
          optimizer,
          zeroStage,
          expertShardDegree,
          expertShardDegree,
          gradientTransientBytesPerParam
        )
      )
    }

    if (stage.sharedExpertLocal > 0) {
      stageMemory = addModelStateMemory(
        stageMemory,
        calculateStateGroupMemory(
          stage.sharedExpertLocal,
          optimizer,
          zeroStage,
          stateShardDegree,
          optimizerShardDegree,
          gradientTransientBytesPerParam
        )
      )
    }

    return applyCPUOffload(stageMemory, config.cpuOffload, zeroStage)
  })
  const totalMemory = stageMemories.reduce((peak, stage) =>
    stage.total > peak.total ? stage : peak
  )

  return totalMemory
}

export function calculateActivationMemory(
  arch: ModelArchitecture,
  config: TrainingConfig,
  moe: MoEConfig,
  schedule: ActivationSchedule = "none"
): number {
  return calculateActivationMemoryDetails(arch, config, moe, schedule).activations
}

function calculateActivationMemoryDetails(
  arch: ModelArchitecture,
  config: TrainingConfig,
  moe: MoEConfig,
  schedule: ActivationSchedule = "none"
): ActivationMemoryDetails {
  const N_pp = clampDegree(config.parallelism.N_pp)
  const boundedMoELayers =
    moe.enabled && moe.L_moe > 0
      ? Math.min(Math.max(0, moe.L_moe), arch.L)
      : 0
  const stageLayouts = getPipelineTransformerLayerCandidates(arch.L, N_pp).flatMap(
    ({ transformerLayers, boundary }) =>
      uniqueStageMoELayerCountsForLayerCount(
        arch.L,
        boundedMoELayers,
        transformerLayers
      ).map((moeLayers) => ({
        transformerLayers,
        moeLayers,
        boundary,
      }))
  )
  const denseFFNWidth = resolveDenseIntermediateSize(arch, moe)
  const expertFFNWidth = resolveExpertIntermediateSize(arch, moe)
  const partialCheckpointDepth = getPartialCheckpointDepth(config)
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
      ? calculateMoEStoredActivationPerLayer(
          arch,
          config,
          effectiveCheckpointing,
          moe,
          expertFFNWidth
        )
      : denseLayerStored

  const VP = clampDegree(config.parallelism.VP)
  const numMicrobatches = clampDegree(config.gradientAccumulationSteps)
  const usesAFAB = schedule === "afab" && N_pp > 1
  const inFlightMicrobatches = Math.max(
    1,
    usesAFAB ? numMicrobatches : Math.min(N_pp, numMicrobatches)
  )
  const interleavedMultiplier =
    !usesAFAB && canUseInterleavedPipelineSchedule(N_pp, numMicrobatches, VP)
      ? 1 + (N_pp - 1) / (N_pp * VP)
      : 1
  const outputLogitsMicrobatches = usesAFAB ? inFlightMicrobatches : 1
  const outputLogitsBytes = getOutputLogitsBytes(arch, config)

  let activationPeak = 0
  let finalStageActivationPeak = 0

  for (const { transformerLayers, moeLayers, boundary } of stageLayouts) {
    const denseLayersPerStage = Math.max(0, transformerLayers - moeLayers)
    const hasOutputLogits = N_pp <= 1 || boundary === "last"
    const outputLogits =
      hasOutputLogits ? outputLogitsBytes * outputLogitsMicrobatches : 0
    const applyPipelineResidency = (stageActivation: number) =>
      stageActivation * inFlightMicrobatches * interleavedMultiplier +
      outputLogits

    if (config.activationCheckpointing === "partial") {
      const denseCheckpointedPerLayer = calculateStoredActivationPerLayer(
        arch,
        config,
        "full",
        denseFFNWidth,
        1
      )
      const moeCheckpointedPerLayer =
        moe.enabled && moe.E > 0 && moe.L_moe > 0
          ? calculateMoEStoredActivationPerLayer(
              arch,
              config,
              "full",
              moe,
              expertFFNWidth
            )
          : denseCheckpointedPerLayer
      const checkpointedLayers = Math.min(
        partialCheckpointDepth,
        transformerLayers
      )
      const minCheckpointedMoELayers = Math.max(
        0,
        checkpointedLayers - denseLayersPerStage
      )
      const maxCheckpointedMoELayers = Math.min(
        moeLayers,
        checkpointedLayers
      )
      let peakPartialStage = 0

      for (
        let checkpointedMoELayers = minCheckpointedMoELayers;
        checkpointedMoELayers <= maxCheckpointedMoELayers;
        checkpointedMoELayers += 1
      ) {
        const checkpointedDenseLayers =
          checkpointedLayers - checkpointedMoELayers
        const nonCheckpointedDenseLayers =
          denseLayersPerStage - checkpointedDenseLayers
        const nonCheckpointedMoELayers = moeLayers - checkpointedMoELayers
        const stageActivation =
          checkpointedDenseLayers * denseCheckpointedPerLayer +
          checkpointedMoELayers * moeCheckpointedPerLayer +
          nonCheckpointedDenseLayers * denseLayerStored +
          nonCheckpointedMoELayers * moeLayerStored

        peakPartialStage = Math.max(peakPartialStage, stageActivation)
      }

      const stageActivation = applyPipelineResidency(peakPartialStage)
      activationPeak = Math.max(activationPeak, stageActivation)
      if (hasOutputLogits) {
        finalStageActivationPeak = Math.max(
          finalStageActivationPeak,
          stageActivation
        )
      }
      continue
    }

    const stageActivation = applyPipelineResidency(
      denseLayersPerStage * denseLayerStored + moeLayers * moeLayerStored
    )
    activationPeak = Math.max(activationPeak, stageActivation)
    if (hasOutputLogits) {
      finalStageActivationPeak = Math.max(
        finalStageActivationPeak,
        stageActivation
      )
    }
  }

  let recomputeWorkingMemory = 0

  const hasPartialCheckpointedLayers =
    config.activationCheckpointing === "partial" &&
    stageLayouts.some(
      ({ transformerLayers }) =>
        Math.min(partialCheckpointDepth, transformerLayers) > 0
    )

  if (config.activationCheckpointing === "full" || hasPartialCheckpointedLayers) {
    recomputeWorkingMemory = calculateFullCheckpointWorkingMemory(
      arch,
      config,
      moe
    )
  }

  const total = activationPeak + recomputeWorkingMemory
  // Recompute working activations and the loss/logits gradient are distinct
  // backward peaks, so compare the logits peak against the checkpointed peak
  // instead of adding both transient buffers unconditionally.
  const logitsGradientPeakExtra = Math.max(
    0,
    finalStageActivationPeak +
      getLogitsGradientPeakExtraBytes(arch, config) -
      total
  )

  return {
    activations: total,
    logitsGradientPeakExtra,
  }
}

export function calculateCommunicationBuffers(
  params: ParameterCounts,
  config: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig = config.model.moe,
  schedule: ActivationSchedule = "none"
): number {
  const zeroStage = resolveZeROStage(config)
  const optimizer = resolveTrainingOptimizerProfile(config)
  const N_tp = clampDegree(config.parallelism.N_tp)
  const N_pp = clampDegree(config.parallelism.N_pp)
  const N_cp = clampDegree(config.parallelism.N_cp)
  const activationBytes = getTrainingActivationBytes(config)
  const sequenceLengthPerRank = config.sequenceLength / N_cp
  const largestLayer = getLargestLayerParameterCount(params, config)
  const largestBoundaryUnit =
    getLargestPipelineBoundaryParameterCount(params) / N_tp
  let buffers = 0

  if (zeroStage === 3) {
    if (config.parallelism.framework === "fsdp") {
      const largestWrappingUnit = Math.max(largestLayer, largestBoundaryUnit)
      // PyTorch FSDP's all-gather limiter allows at most two unsharded
      // wrapping units to be resident, so model the peak rather than one unit.
      buffers += usesFSDPMixedPrecision(config)
        ? 2 * largestWrappingUnit * getTrainingActivationBytes(config)
        : 2 * largestWrappingUnit * optimizer.parameterBytes
    } else {
      const prefetchBucketSize = resolvePrefetchBucketSizeElements(config)
      const nextPrefetch = Math.min(largestLayer, prefetchBucketSize)
      const largestParameterUnit = Math.max(largestLayer, largestBoundaryUnit)

      buffers +=
        (largestParameterUnit + nextPrefetch) * optimizer.parameterBytes
    }
  }

  if (
    config.parallelism.framework !== "fsdp" &&
    zeroStage >= 2 &&
    (config.zeroCommunication.overlapComm || zeroStage === 3)
  ) {
    const allgatherBucketSize = resolveBucketSizeElements(config, "allgather")
    const reduceBucketSize = resolveBucketSizeElements(config, "reduce")

    buffers +=
      4.5 *
      (allgatherBucketSize + reduceBucketSize) *
      optimizer.parameterBytes
  }

  buffers += calculateActivationMemoryDetails(
    arch,
    config,
    moe,
    schedule
  ).logitsGradientPeakExtra

  if (N_tp > 1) {
    buffers +=
      config.microBatchSize *
      sequenceLengthPerRank *
      arch.d *
      ((N_tp - 1) / N_tp) *
      activationBytes
  }

  if (N_cp > 1) {
    const remoteContextFraction = (N_cp - 1) / N_cp
    const kvWidthPerTensorParallelRank = getKVProjectionWidth(arch) / N_tp

    buffers +=
      2 *
      config.microBatchSize *
      config.sequenceLength *
      kvWidthPerTensorParallelRank *
      remoteContextFraction *
      activationBytes
  }

  if (N_pp > 1) {
    buffers +=
      config.microBatchSize * sequenceLengthPerRank * arch.d * activationBytes
  }

  buffers += calculateExpertParallelRoutingBufferBytes(arch, config)

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
  gpu: GPUSpec,
  schedule: ActivationSchedule = "none"
): MemoryBreakdown {
  const modelState = calculateModelStateMemory(params, config)
  const activations = calculateActivationMemory(arch, config, moe, schedule)
  const communicationBuffers = calculateCommunicationBuffers(
    params,
    config,
    arch,
    moe,
    schedule
  )
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
  const N_tp = clampDegree(config.parallelism.N_tp)
  const largestTransformerBlock = getLargestLayerParameterCount(params, config)
  const largestBoundaryUnit =
    getLargestPipelineBoundaryParameterCount(params) / N_tp
  const gatheredParameterBytes = usesFSDPMixedPrecision(config)
    ? getTrainingActivationBytes(config)
    : optimizer.parameterBytes

  return (
    Math.max(largestTransformerBlock, largestBoundaryUnit) *
    (gatheredParameterBytes + optimizer.betaGrad)
  )
}

export function calculateLoRAParamCount(config: PostTrainingConfig): number {
  return calculateLoRAParamCountForArchitecture(
    config.baseModel.architecture,
    config.baseModel.moe,
    config.lora,
  )
}

export function calculateLoRAParamCountForArchitecture(
  architecture: ModelArchitecture,
  moe: MoEConfig,
  lora: PostTrainingConfig["lora"],
): number {
  const rank =
    Number.isFinite(lora.rank) && lora.rank >= 1
      ? Math.floor(lora.rank)
      : Number.POSITIVE_INFINITY
  const d = architecture.d
  const queryWidth = getAttentionProjectionWidth(architecture)
  const kvWidth = getKVProjectionWidth(architecture)
  const attentionModuleShapes: Partial<Record<LoRATargetModule, [number, number]>> = {
    q_proj: [d, queryWidth],
    k_proj: [d, kvWidth],
    v_proj: [d, kvWidth],
    o_proj: [queryWidth, d],
  }
  const moeLayerCount =
    moe.enabled && moe.L_moe > 0
      ? Math.min(Math.max(0, moe.L_moe), architecture.L)
      : 0
  const denseLayerCount = architecture.L - moeLayerCount
  const denseFFNWidth =
    moe.enabled && moe.denseIntermediateSize !== null
      ? moe.denseIntermediateSize
      : resolveDefaultIntermediateSize(architecture)
  const expertFFNWidth =
    moe.expertIntermediateSize ?? resolveDefaultIntermediateSize(architecture, true)
  const expertCopies = Math.max(0, moe.E) + Math.max(0, moe.E_s)
  const denseHasGateProjection = isSwiGLUStyle(architecture.ffnType)

  return lora.targetModules.reduce((sum, moduleId) => {
    const attentionShape = attentionModuleShapes[moduleId]

    if (attentionShape) {
      const [inputDim, outputDim] = attentionShape
      return sum + architecture.L * rank * (inputDim + outputDim)
    }

    const denseFFNAdapters =
      !denseHasGateProjection && moduleId === "gate_proj"
        ? 0
        : denseLayerCount * rank * (d + denseFFNWidth)
    const expertFFNAdapters =
      moeLayerCount * expertCopies * rank * (d + expertFFNWidth)

    return sum + denseFFNAdapters + expertFFNAdapters
  }, 0)
}

export function calculateLoRAMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const optimizer = resolvePostTrainingOptimizerProfile(config)
  const baseModelBytes =
    getPositiveParameterCountOrInfinity(config.baseModel.parameterCount) *
    getPostTrainingWeightBytes(config)
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
  const optimizer = resolvePostTrainingOptimizerProfile(config)
  const quantizationBits = config.lora.quantizationBits ?? 4
  const quantizationLabel = formatQLoRAQuantizationLabel(quantizationBits)
  const baseModelBytes = calculateQuantizedBaseModelBytes(
    config,
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
        label: `Base model (${quantizationLabel})`,
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
  const optimizer = resolvePostTrainingOptimizerProfile(config)
  const chosenRejectedMultiplier = 2
  const activations = calculatePostTrainingActivationMemory(
    config.baseModel.architecture,
    config,
    chosenRejectedMultiplier
  )
  const logProbStorage =
    getPostTrainingPerGpuBatch(config, chosenRejectedMultiplier) *
    config.sequenceLength *
    4

  if (config.approach === "lora" || config.approach === "qlora") {
    const qloraQuantizationLabel = formatQLoRAQuantizationLabel(
      config.lora.quantizationBits ?? 4
    )
    const baseModelBytes =
      config.approach === "qlora"
        ? calculateQuantizedBaseModelBytes(
            config,
            config.lora.quantizationBits ?? 4
          )
        : getPositiveParameterCountOrInfinity(config.baseModel.parameterCount) *
          getPostTrainingWeightBytes(config)
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
              ? `Shared reference base (${qloraQuantizationLabel})`
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

  const policyStates = calculatePartiallyTrainableModelStates(
    config.baseModel.parameterCount,
    resolvePostTrainingTrainableParameterCount(config),
    optimizer
  )
  const referenceModelBytes =
    getPositiveParameterCountOrInfinity(config.baseModel.parameterCount) *
    getPostTrainingWeightBytes(config)

  return finalizePostTrainingMemoryBreakdown({
    gpu: config.hardware.gpu,
    parameters: policyStates.parameters + referenceModelBytes,
    gradients: policyStates.gradients,
    optimizerStates: policyStates.optimizerStates,
    activations,
    communicationBuffers: logProbStorage,
    frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
    peakWorkingSet: activations + logProbStorage,
    trainableModels: policyStates.trainableTotal,
    frozenModels: referenceModelBytes + policyStates.frozenTotal,
    loraAdapter: 0,
    ppoBuffers: 0,
    items: [
      {
        label:
          policyStates.frozenParameters > 0
            ? "Policy trainable parameters"
            : "Policy parameters",
        category: "trainable",
        bytes: policyStates.trainableParameters,
      },
      ...(policyStates.frozenParameters > 0
        ? [
            {
              label: "Policy frozen parameters",
              category: "frozen" as const,
              bytes: policyStates.frozenParameters,
            },
          ]
        : []),
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
  const optimizer = resolvePostTrainingOptimizerProfile(config)
  const frozenWeightBytes = getPostTrainingWeightBytes(config)
  const criticParameterCount = getPositiveParameterCountOrInfinity(
    config.ppo.criticModelParameterCount
  )
  const rewardParameterCount = getPositiveParameterCountOrInfinity(
    config.ppo.rewardModelParameterCount
  )
  const actorActivations = calculatePostTrainingActivationMemory(
    config.baseModel.architecture,
    config
  )
  const actorTransformerActivations =
    calculatePostTrainingTransformerActivationMemory(
      config.baseModel.architecture,
      config
    )
  const criticActivationScale =
    config.baseModel.parameterCount > 0
      ? Math.max(0, criticParameterCount / config.baseModel.parameterCount)
      : 1
  const criticActivations = actorTransformerActivations * criticActivationScale
  const trainingActivations = actorActivations + criticActivations
  const perGpuBatch = getPostTrainingPerGpuBatch(config)
  const rolloutBuffers = calculatePostTrainingRolloutBufferBytes(
    config.sequenceLength,
    perGpuBatch
  )
  const kvCacheBytes = calculateKVCacheBytes(
    config.baseModel.architecture,
    perGpuBatch,
    config.sequenceLength,
    config.kvCachePrecision
  )
  const updateWorkingSet = trainingActivations + rolloutBuffers
  const criticStates = calculateTrainableModelStates(
    criticParameterCount,
    optimizer
  )
  const rewardModelBytes = rewardParameterCount * frozenWeightBytes
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
    const qloraQuantizationLabel = formatQLoRAQuantizationLabel(
      config.lora.quantizationBits ?? 4
    )
    const actorBaseBytes =
      config.approach === "qlora"
        ? calculateQuantizedBaseModelBytes(
            config,
            config.lora.quantizationBits ?? 4
          )
        : getPositiveParameterCountOrInfinity(config.baseModel.parameterCount) *
          frozenWeightBytes
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
          ? `Actor base (${qloraQuantizationLabel}, shared reference)`
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
    const actorStates = calculatePartiallyTrainableModelStates(
      config.baseModel.parameterCount,
      resolvePostTrainingTrainableParameterCount(config),
      optimizer
    )
    const referenceModelBytes =
      getPositiveParameterCountOrInfinity(config.baseModel.parameterCount) *
      frozenWeightBytes

    parameters += actorStates.parameters + referenceModelBytes
    gradients += actorStates.gradients
    optimizerStates += actorStates.optimizerStates
    trainableModels += actorStates.trainableTotal
    frozenModels += referenceModelBytes + actorStates.frozenTotal

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
    if (actorStates.frozenParameters > 0) {
      items.unshift({
        label: "Actor frozen parameters",
        category: "frozen",
        bytes: actorStates.frozenParameters,
      })
    }
    items.unshift({
      label:
        actorStates.frozenParameters > 0
          ? "Actor trainable parameters"
          : "Actor parameters",
      category: "trainable",
      bytes: actorStates.trainableParameters,
    })
  }

  const generationWorkingSet = calculatePostTrainingPeakGenerationWorkingSet({
    gpu: config.hardware.gpu,
    parameters,
    gradients,
    optimizerStates,
    frameworkOverhead: MEGATRON_STYLE_OVERHEAD_BYTES,
    rolloutBuffers,
    kvCacheBytes,
    requestedLocalGenerationBatch: perGpuBatch,
  })

  items.push(
    {
      label: "Actor activations",
      category: "buffer",
      bytes: actorActivations,
    },
    {
      label: "Critic activations",
      category: "buffer",
      bytes: criticActivations,
    },
    {
      label: "PPO rollout buffers",
      category: "buffer",
      bytes: rolloutBuffers,
    },
    {
      label: "KV cache (generation peak)",
      category: "buffer",
      bytes: generationWorkingSet.kvCacheBytes,
    }
  )

  return finalizePostTrainingMemoryBreakdown({
    gpu: config.hardware.gpu,
    parameters,
    gradients,
    optimizerStates,
    activations: trainingActivations,
    communicationBuffers: generationWorkingSet.total,
    frameworkOverhead: MEGATRON_STYLE_OVERHEAD_BYTES,
    peakWorkingSet: Math.max(updateWorkingSet, generationWorkingSet.total),
    trainableModels,
    frozenModels,
    loraAdapter,
    ppoBuffers: rolloutBuffers + generationWorkingSet.kvCacheBytes,
    items,
  })
}

export function calculateGRPOMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const optimizer = resolvePostTrainingOptimizerProfile(config)
  const frozenWeightBytes = getPostTrainingWeightBytes(config)
  const groupSize = Number.isFinite(config.grpo.groupSize)
    ? Math.max(2, Math.ceil(config.grpo.groupSize))
    : 2
  const activations = calculatePostTrainingActivationMemory(
    config.baseModel.architecture,
    config,
    groupSize
  )
  const perGpuGenerationBatch = getPostTrainingPerGpuBatch(config, groupSize)
  const kvCacheBytes = calculateKVCacheBytes(
    config.baseModel.architecture,
    perGpuGenerationBatch,
    config.sequenceLength,
    config.kvCachePrecision
  )
  const rolloutBuffers = calculatePostTrainingRolloutBufferBytes(
    config.sequenceLength,
    perGpuGenerationBatch
  )
  const updateWorkingSet = activations + rolloutBuffers

  if (config.approach === "lora" || config.approach === "qlora") {
    const qloraQuantizationLabel = formatQLoRAQuantizationLabel(
      config.lora.quantizationBits ?? 4
    )
    const baseModelBytes =
      config.approach === "qlora"
        ? calculateQuantizedBaseModelBytes(
            config,
            config.lora.quantizationBits ?? 4
          )
        : getPositiveParameterCountOrInfinity(config.baseModel.parameterCount) *
          frozenWeightBytes
    const loraStates = calculateTrainableModelStates(
      calculateLoRAParamCount(config),
      optimizer
    )
    const parameters = baseModelBytes + loraStates.parameters
    const gradients = loraStates.gradients
    const optimizerStates = loraStates.optimizerStates
    const generationWorkingSet = calculatePostTrainingPeakGenerationWorkingSet({
      gpu: config.hardware.gpu,
      parameters,
      gradients,
      optimizerStates,
      frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
      rolloutBuffers,
      kvCacheBytes,
      requestedLocalGenerationBatch: perGpuGenerationBatch,
    })

    return finalizePostTrainingMemoryBreakdown({
      gpu: config.hardware.gpu,
      parameters,
      gradients,
      optimizerStates,
      activations,
      communicationBuffers: generationWorkingSet.total,
      frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
      peakWorkingSet: Math.max(updateWorkingSet, generationWorkingSet.total),
      trainableModels: loraStates.total,
      frozenModels: baseModelBytes,
      loraAdapter: loraStates.total,
      ppoBuffers: rolloutBuffers + generationWorkingSet.kvCacheBytes,
      items: [
        {
          label:
            config.approach === "qlora"
              ? `Policy base (${qloraQuantizationLabel}, shared reference)`
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
          label: "GRPO rollout buffers",
          category: "buffer",
          bytes: rolloutBuffers,
        },
        {
          label: `KV cache (generation peak, G=${groupSize})`,
          category: "buffer",
          bytes: generationWorkingSet.kvCacheBytes,
        },
      ],
    })
  }

  const policyStates = calculatePartiallyTrainableModelStates(
    config.baseModel.parameterCount,
    resolvePostTrainingTrainableParameterCount(config),
    optimizer
  )
  const referenceModelBytes =
    getPositiveParameterCountOrInfinity(config.baseModel.parameterCount) *
    frozenWeightBytes
  const parameters = policyStates.parameters + referenceModelBytes
  const gradients = policyStates.gradients
  const optimizerStates = policyStates.optimizerStates
  const generationWorkingSet = calculatePostTrainingPeakGenerationWorkingSet({
    gpu: config.hardware.gpu,
    parameters,
    gradients,
    optimizerStates,
    frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
    rolloutBuffers,
    kvCacheBytes,
    requestedLocalGenerationBatch: perGpuGenerationBatch,
  })

  return finalizePostTrainingMemoryBreakdown({
    gpu: config.hardware.gpu,
    parameters,
    gradients,
    optimizerStates,
    activations,
    communicationBuffers: generationWorkingSet.total,
    frameworkOverhead: DEFAULT_POST_TRAINING_OVERHEAD_BYTES,
    peakWorkingSet: Math.max(updateWorkingSet, generationWorkingSet.total),
    trainableModels: policyStates.trainableTotal,
    frozenModels: referenceModelBytes + policyStates.frozenTotal,
    loraAdapter: 0,
    ppoBuffers: rolloutBuffers + generationWorkingSet.kvCacheBytes,
    items: [
      {
        label:
          policyStates.frozenParameters > 0
            ? "Policy trainable parameters"
            : "Policy parameters",
        category: "trainable",
        bytes: policyStates.trainableParameters,
      },
      ...(policyStates.frozenParameters > 0
        ? [
            {
              label: "Policy frozen parameters",
              category: "frozen" as const,
              bytes: policyStates.frozenParameters,
            },
          ]
        : []),
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
        label: "GRPO rollout buffers",
        category: "buffer",
        bytes: rolloutBuffers,
      },
      {
        label: `KV cache (generation peak, G=${groupSize})`,
        category: "buffer",
        bytes: generationWorkingSet.kvCacheBytes,
      },
    ],
  })
}
