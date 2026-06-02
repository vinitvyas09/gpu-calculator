import { OPTIMIZER_PROFILES } from "../constants"
import type { OptimizerType } from "../types"

export function hasInvalidPretrainingOptimizer(
  optimizer: OptimizerType
): boolean {
  const profile = OPTIMIZER_PROFILES.find((candidate) => candidate.id === optimizer)
  return !profile || !profile.supportsPretraining
}

export function hasInvalidPostTrainingOptimizer(
  optimizer: OptimizerType
): boolean {
  const profile = OPTIMIZER_PROFILES.find((candidate) => candidate.id === optimizer)
  return !profile || !profile.supportsPostTraining
}
