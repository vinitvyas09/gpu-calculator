import type {
  FineTuningApproach,
  OptimizerType,
  PostTrainingConfig,
  PostTrainingMethod,
} from "../types"

const POST_TRAINING_METHODS: readonly PostTrainingMethod[] = [
  "sft",
  "dpo",
  "ppo",
  "grpo",
]

const FINE_TUNING_APPROACHES: readonly FineTuningApproach[] = [
  "full",
  "lora",
  "qlora",
  "mezo",
]

export function hasInvalidPostTrainingMethod(method: unknown): boolean {
  return !POST_TRAINING_METHODS.some((candidate) => candidate === method)
}

export function hasInvalidPostTrainingApproach(approach: unknown): boolean {
  return !FINE_TUNING_APPROACHES.some((candidate) => candidate === approach)
}

export function hasInvalidPostTrainingMethodApproach(
  method: PostTrainingMethod,
  approach: FineTuningApproach,
): boolean {
  return (
    hasInvalidPostTrainingMethod(method) ||
    hasInvalidPostTrainingApproach(approach) ||
    (approach === "mezo" && method !== "sft")
  )
}

export function hasInvalidPostTrainingOptimizerApproach(
  optimizer: OptimizerType,
  approach: FineTuningApproach,
): boolean {
  return (
    hasInvalidPostTrainingApproach(approach) ||
    (approach === "mezo" ? optimizer !== "mezo" : optimizer === "mezo")
  )
}

export function hasInvalidPostTrainingApproachConfig(
  config: Pick<PostTrainingConfig, "method" | "approach" | "optimizer">,
): boolean {
  return (
    hasInvalidPostTrainingMethodApproach(config.method, config.approach) ||
    hasInvalidPostTrainingOptimizerApproach(config.optimizer, config.approach)
  )
}
