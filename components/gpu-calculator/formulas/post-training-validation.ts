import type {
  FineTuningApproach,
  OptimizerType,
  PostTrainingConfig,
  PostTrainingMethod,
} from "../types"
import { hasInvalidArchitectureConfig, hasInvalidMoEConfig } from "./compute"
import { hasInvalidPostTrainingBaseModelInputMode } from "./model-input-validation"

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

function isFinitePositiveInteger(value: number): boolean {
  return Number.isFinite(value) && value > 0 && Number.isInteger(value)
}

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

export function hasInvalidQLoRAQuantizationBits(
  config: Pick<PostTrainingConfig, "approach" | "lora">,
): boolean {
  const quantizationBits = config.lora.quantizationBits as number | null

  return (
    config.approach === "qlora" &&
    quantizationBits !== null &&
    quantizationBits !== 4 &&
    quantizationBits !== 8
  )
}

export function hasInvalidLoRARankValue(
  lora: Pick<PostTrainingConfig["lora"], "rank">,
): boolean {
  return !isFinitePositiveInteger(lora.rank)
}

export function hasInvalidLoRARank(
  config: Pick<PostTrainingConfig, "approach" | "lora">,
): boolean {
  return (
    (config.approach === "lora" || config.approach === "qlora") &&
    hasInvalidLoRARankValue(config.lora)
  )
}

export function hasInvalidLoRAAlphaValue(
  lora: Pick<PostTrainingConfig["lora"], "alpha">,
): boolean {
  return !isFinitePositiveInteger(lora.alpha)
}

export function hasInvalidLoRAAlpha(
  config: Pick<PostTrainingConfig, "approach" | "lora">,
): boolean {
  return (
    (config.approach === "lora" || config.approach === "qlora") &&
    hasInvalidLoRAAlphaValue(config.lora)
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

export function hasInvalidPostTrainingActiveParameterCount(
  config: Pick<PostTrainingConfig, "baseModel">,
): boolean {
  const { parameterCount, moe } = config.baseModel
  const activeParameterCount = moe.activeParameterCount

  if (!moe.enabled || activeParameterCount === null) {
    return false
  }

  if (!isFinitePositiveInteger(activeParameterCount)) {
    return true
  }

  return (
    isFinitePositiveInteger(parameterCount) &&
    activeParameterCount > parameterCount
  )
}

export function hasInvalidPostTrainingBaseParameterCount(
  config: Pick<PostTrainingConfig, "baseModel">,
): boolean {
  return !isFinitePositiveInteger(config.baseModel.parameterCount)
}

export function hasInvalidPostTrainingModelShape(
  config: Pick<PostTrainingConfig, "baseModel" | "sequenceLength">,
): boolean {
  return (
    hasInvalidPostTrainingBaseModelInputMode(config) ||
    hasInvalidPostTrainingBaseParameterCount(config) ||
    hasInvalidArchitectureConfig(
      config.baseModel.architecture,
      config.sequenceLength,
    ) ||
    hasInvalidMoEConfig(
      config.baseModel.moe,
      config.baseModel.architecture.L,
    ) ||
    hasInvalidPostTrainingActiveParameterCount(config)
  )
}
