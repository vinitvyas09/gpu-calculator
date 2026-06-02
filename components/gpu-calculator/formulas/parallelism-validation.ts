import type {
  FSDPStrategy,
  FrameworkType,
  ParallelismConfig,
  TrainingConfig,
  ZeROStage,
} from "../types"
import { hasInvalidMoEConfig } from "./compute"
import { getParallelismLocalGroupSize } from "./hardware"

function isFinitePositiveInteger(value: number): boolean {
  return Number.isFinite(value) && value > 0 && Number.isInteger(value)
}

function isValidZeROStage(value: unknown): value is ZeROStage {
  return value === 0 || value === 1 || value === 2 || value === 3
}

function isValidFrameworkType(value: unknown): value is FrameworkType {
  return (
    value === "megatron" ||
    value === "deepspeed" ||
    value === "fsdp" ||
    value === "hf_trainer"
  )
}

function isValidFSDPStrategy(value: unknown): value is FSDPStrategy {
  return (
    value === "NO_SHARD" ||
    value === "SHARD_GRAD_OP" ||
    value === "FULL_SHARD" ||
    value === "HYBRID_SHARD" ||
    value === "HYBRID_SHARD_ZERO2"
  )
}

export function resolveEffectiveZeroStage(
  parallelism: ParallelismConfig,
): ZeROStage | null {
  if (!isValidFrameworkType(parallelism.framework)) {
    return null
  }

  if (parallelism.framework !== "fsdp") {
    return isValidZeROStage(parallelism.zeroStage)
      ? parallelism.zeroStage
      : null
  }

  const strategy = parallelism.fsdpStrategy ?? "FULL_SHARD"

  if (!isValidFSDPStrategy(strategy)) {
    return null
  }

  switch (strategy) {
    case "NO_SHARD":
      return 0
    case "SHARD_GRAD_OP":
    case "HYBRID_SHARD_ZERO2":
      return 2
    case "FULL_SHARD":
    case "HYBRID_SHARD":
      return 3
  }
}

export function hasInvalidManualShardingMode(config: TrainingConfig): boolean {
  return (
    config.parallelismMode === "manual" &&
    resolveEffectiveZeroStage(config.parallelism) === null
  )
}

function usesHybridShard(parallelism: ParallelismConfig): boolean {
  return (
    parallelism.fsdpStrategy === "HYBRID_SHARD" ||
    parallelism.fsdpStrategy === "HYBRID_SHARD_ZERO2"
  )
}

function hasValidPipelineStagePartition(N_pp: number, layerCount: number): boolean {
  return layerCount % N_pp === 0 || (layerCount + 2) % N_pp === 0
}

function isSwiGLUStyle(ffnType: string): boolean {
  return ffnType === "swiglu" || ffnType === "geglu" || ffnType === "moe"
}

function resolveTPShardedFFNIntermediateSize(
  config: TrainingConfig,
): number | null {
  const { architecture, moe } = config.model

  if (
    moe.enabled &&
    Number.isFinite(moe.L_moe) &&
    moe.L_moe >= architecture.L
  ) {
    return null
  }

  if (moe.enabled && moe.denseIntermediateSize !== null) {
    return moe.denseIntermediateSize
  }

  if (architecture.d_ff !== null) {
    return architecture.d_ff
  }

  return isSwiGLUStyle(architecture.ffnType)
    ? Math.round((8 / 3) * architecture.d)
    : 4 * architecture.d
}

function getManualParallelWorldSize(config: TrainingConfig): number | null {
  const { N_dp, N_tp, N_pp, N_cp, N_ep } = config.parallelism
  const degrees = [N_dp, N_tp, N_pp, N_cp, N_ep]

  if (!degrees.every(isFinitePositiveInteger)) {
    return null
  }

  return degrees.reduce((product, degree) => product * degree, 1)
}

function getRequestedManualNumGPUs(config: TrainingConfig): number | null {
  if (config.hardware.numGPUs === null) {
    return 1
  }

  return isFinitePositiveInteger(config.hardware.numGPUs)
    ? config.hardware.numGPUs
    : null
}

function calculateDenseStateShardDegreeForValidation(
  config: TrainingConfig,
): number | null {
  const { N_dp, N_tp, N_pp, N_cp } = config.parallelism

  if (![N_dp, N_tp, N_pp, N_cp].every(isFinitePositiveInteger)) {
    return null
  }

  const replicaShardDegree = N_dp * N_cp

  if (!usesHybridShard(config.parallelism)) {
    return replicaShardDegree
  }

  const localNonReplicaRanks = N_tp * N_pp
  const localReplicaCapacity = Math.max(
    1,
    Math.floor(
      getParallelismLocalGroupSize(config.hardware.gpu) / localNonReplicaRanks,
    ),
  )

  return Math.min(replicaShardDegree, localReplicaCapacity)
}

export function hasInvalidManualWorldSize(config: TrainingConfig): boolean {
  if (config.parallelismMode !== "manual") {
    return false
  }

  const worldSize = getManualParallelWorldSize(config)
  const requestedNumGPUs = getRequestedManualNumGPUs(config)

  return (
    worldSize !== null &&
    requestedNumGPUs !== null &&
    worldSize !== requestedNumGPUs
  )
}

export function hasInvalidManualTensorParallelismTopology(
  config: TrainingConfig,
): boolean {
  if (config.parallelismMode !== "manual") {
    return false
  }

  const { N_tp } = config.parallelism

  if (!isFinitePositiveInteger(N_tp) || N_tp <= 1) {
    return false
  }

  const { architecture, moe } = config.model
  const { d, a, a_kv } = architecture
  const dFF = resolveTPShardedFFNIntermediateSize(config)

  if (
    !isFinitePositiveInteger(d) ||
    !isFinitePositiveInteger(a) ||
    hasInvalidMoEConfig(moe, architecture.L) ||
    (a_kv !== null && !isFinitePositiveInteger(a_kv)) ||
    (dFF !== null && !isFinitePositiveInteger(dFF))
  ) {
    return false
  }

  return (
    d % N_tp !== 0 ||
    a % N_tp !== 0 ||
    (a_kv !== null && a_kv % N_tp !== 0) ||
    (dFF !== null && dFF % N_tp !== 0)
  )
}

export function hasInvalidManualExpertParallelismTopology(
  config: TrainingConfig,
): boolean {
  if (config.parallelismMode !== "manual") {
    return false
  }

  const { moe, architecture } = config.model

  if (!moe.enabled || hasInvalidMoEConfig(moe, architecture.L)) {
    return false
  }

  const { N_dp, N_tp, N_pp, N_cp, N_ep } = config.parallelism

  if (![N_dp, N_tp, N_pp, N_cp, N_ep].every(isFinitePositiveInteger)) {
    return false
  }

  if (N_ep <= 1) {
    return false
  }

  if (moe.E % N_ep !== 0) {
    return true
  }

  const denseStateShardDegree = calculateDenseStateShardDegreeForValidation(
    config,
  )

  if (denseStateShardDegree === null) {
    return false
  }

  const expertDataParallelNumerator = denseStateShardDegree * N_tp

  return (
    expertDataParallelNumerator % N_ep !== 0 ||
    N_tp * N_ep > getParallelismLocalGroupSize(config.hardware.gpu)
  )
}

export function hasInvalidManualTensorExpertSequenceParallelismTopology(
  config: TrainingConfig,
): boolean {
  if (config.parallelismMode !== "manual") {
    return false
  }

  const { moe, architecture } = config.model

  if (!moe.enabled || hasInvalidMoEConfig(moe, architecture.L)) {
    return false
  }

  const { N_tp, N_ep, sequenceParallelism } = config.parallelism

  return (
    isFinitePositiveInteger(N_tp) &&
    isFinitePositiveInteger(N_ep) &&
    N_tp > 1 &&
    N_ep > 1 &&
    sequenceParallelism === "disabled"
  )
}

export function hasInvalidManualPipelineTopology(config: TrainingConfig): boolean {
  if (config.parallelismMode !== "manual") {
    return false
  }

  const { N_pp, VP } = config.parallelism

  if (!isFinitePositiveInteger(N_pp)) {
    return false
  }

  if (N_pp <= 1) {
    return false
  }

  const layerCount = config.model.architecture.L

  if (
    !isFinitePositiveInteger(layerCount) ||
    !hasValidPipelineStagePartition(N_pp, layerCount)
  ) {
    return true
  }

  const numMicrobatches = config.gradientAccumulationSteps

  if (!isFinitePositiveInteger(numMicrobatches) || !isFinitePositiveInteger(VP)) {
    return true
  }

  const zeroStage = resolveEffectiveZeroStage(config.parallelism)

  if (zeroStage === null) {
    return true
  }

  if (config.parallelism.framework === "fsdp") {
    if (zeroStage === 3) {
      return true
    }

    if (zeroStage === 2) {
      return numMicrobatches >= 2 * N_pp
    }
  } else if (zeroStage >= 2) {
    return true
  }

  if (numMicrobatches < N_pp - 1) {
    return true
  }

  return VP > 1 && numMicrobatches % N_pp !== 0
}

export function hasInvalidManualContextParallelismTopology(
  config: TrainingConfig,
): boolean {
  if (config.parallelismMode !== "manual") {
    return false
  }

  const { N_cp } = config.parallelism

  if (!isFinitePositiveInteger(N_cp) || N_cp <= 1) {
    return false
  }

  return (
    !isFinitePositiveInteger(config.sequenceLength) ||
    config.sequenceLength % N_cp !== 0
  )
}

export function hasInvalidCPUOffloadConfig(config: TrainingConfig): boolean {
  const zeroStage = resolveEffectiveZeroStage(config.parallelism)

  if (zeroStage === null) {
    return config.cpuOffload !== "none"
  }

  return (
    (config.cpuOffload === "optimizer-only" && zeroStage < 1) ||
    (config.cpuOffload === "optimizer-and-params" && zeroStage !== 3)
  )
}
