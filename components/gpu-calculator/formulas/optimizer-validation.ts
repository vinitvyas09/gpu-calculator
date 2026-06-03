import { OPTIMIZER_PROFILES } from "../constants"
import type { GradientPrecision, OptimizerType } from "../types"

const VALID_GRADIENT_PRECISIONS: ReadonlySet<GradientPrecision> = new Set([
  "fp32",
  "bf16",
])

export function hasInvalidGradientPrecision(
  gradientPrecision: unknown,
): boolean {
  return !VALID_GRADIENT_PRECISIONS.has(gradientPrecision as GradientPrecision)
}

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
