import type { TrainingConfig } from "../types"

export function hasInvalidFlashAttentionFlag(
  config: Pick<TrainingConfig, "flashAttention">,
): boolean {
  return typeof config.flashAttention !== "boolean"
}
