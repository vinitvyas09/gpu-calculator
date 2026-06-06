import type {
  FineTuningApproach,
  OptimizerType,
  PostTrainingConfig,
  PostTrainingMethod,
} from "../types"
import {
  calculateParameterCount,
  hasInvalidArchitectureConfig,
  hasInvalidMoEConfig,
} from "./compute"
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

export function hasInvalidPostTrainingDistributedQuantization(
  config: Pick<PostTrainingConfig, "approach" | "lora" | "distributedStrategy">,
): boolean {
  // The verified FSDP sharding path for quantized bases is the bitsandbytes
  // 4-bit quant_storage route (Linear4bit/Params4bit, bnb PR #970); 8-bit
  // quantized bases have no documented FSDP sharding support.
  // https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora
  return (
    config.distributedStrategy === "fsdp-full-shard" &&
    config.approach === "qlora" &&
    (config.lora.quantizationBits ?? 4) === 8
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
  config: Pick<PostTrainingConfig, "baseModel" | "sequenceLength">,
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
    (activeParameterCount > parameterCount ||
      hasImpossibleMoEActiveParameterCount(config, activeParameterCount))
  )
}

function hasImpossibleMoEActiveParameterCount(
  config: Pick<PostTrainingConfig, "baseModel" | "sequenceLength">,
  activeParameterCount: number,
): boolean {
  const { parameterCount, architecture, moe } = config.baseModel

  if (
    !moe.enabled ||
    hasInvalidMoEConfig(moe, architecture.L) ||
    !isFinitePositiveInteger(parameterCount)
  ) {
    return false
  }

  const counts = calculateParameterCount(architecture, moe, config.sequenceLength)

  if (
    counts.moe === null ||
    !Number.isFinite(counts.total) ||
    counts.total <= 0 ||
    !Number.isFinite(counts.active) ||
    counts.active <= 0 ||
    !Number.isFinite(counts.moe.activeRoutedExpertParameters) ||
    counts.moe.activeRoutedExpertParameters <= 0
  ) {
    return false
  }

  const nonRoutedActive =
    counts.active - counts.moe.activeRoutedExpertParameters
  const totalScale = parameterCount / counts.total
  const minimumActiveParameterCount = nonRoutedActive * totalScale
  const tolerance = Math.max(1, minimumActiveParameterCount * 1e-9)

  return activeParameterCount + tolerance < minimumActiveParameterCount
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
