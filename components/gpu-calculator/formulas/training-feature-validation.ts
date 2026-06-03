import type { PostTrainingConfig, TrainingConfig } from "../types"

export function hasInvalidFlashAttentionFlag(
  config: Pick<TrainingConfig, "flashAttention">,
): boolean {
  return typeof config.flashAttention !== "boolean"
}

export function hasInvalidAMPAutocastFlag(
  config: Pick<TrainingConfig, "ampAutocast">,
): boolean {
  return typeof config.ampAutocast !== "boolean"
}

export function hasInvalidChunkedCrossEntropyFlag(
  config: Pick<TrainingConfig | PostTrainingConfig, "chunkedCrossEntropy">,
): boolean {
  return typeof config.chunkedCrossEntropy !== "boolean"
}

export function hasInvalidTorchCompileFlag(
  config: Pick<TrainingConfig, "torchCompile">,
): boolean {
  return typeof config.torchCompile !== "boolean"
}
