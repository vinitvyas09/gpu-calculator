import type { ParallelismConfig, TrainingConfig, ZeROStage } from "../types"

function isFinitePositiveInteger(value: number): boolean {
  return Number.isFinite(value) && value > 0 && Number.isInteger(value)
}

function resolveEffectiveZeroStage(parallelism: ParallelismConfig): ZeROStage {
  const strategy = parallelism.fsdpStrategy

  if (strategy === null) {
    return parallelism.zeroStage
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

function hasValidPipelineStagePartition(N_pp: number, layerCount: number): boolean {
  return layerCount % N_pp === 0 || (layerCount + 2) % N_pp === 0
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
