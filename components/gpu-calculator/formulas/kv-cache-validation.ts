import type { KVCachePrecision, PostTrainingConfig } from "../types"

export function isValidKVCachePrecision(
  precision: unknown,
): precision is KVCachePrecision {
  return precision === "bf16" || precision === "fp16" || precision === "int8"
}

export function usesPostTrainingKVCache(
  config: Pick<PostTrainingConfig, "method">,
): boolean {
  return config.method === "ppo" || config.method === "grpo"
}

export function hasInvalidPostTrainingKVCachePrecision(
  config: Pick<PostTrainingConfig, "method" | "kvCachePrecision">,
): boolean {
  return (
    usesPostTrainingKVCache(config) &&
    !isValidKVCachePrecision(config.kvCachePrecision)
  )
}
