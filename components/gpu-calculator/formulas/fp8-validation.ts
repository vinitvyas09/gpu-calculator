import type { FP8Config, TrainingPrecision } from "../types"

interface FP8ConfigHost {
  precision: TrainingPrecision
  fp8: FP8Config
}

export function isValidFP8KernelSpeedupFactor(value: number): boolean {
  return Number.isFinite(value) && value >= 1 && value <= 2
}

export function isValidFP8StorageMode(value: unknown): boolean {
  return value === "transformer-engine" || value === "ms-amp"
}

export function hasInvalidFP8KernelSpeedupFactor(
  config: FP8ConfigHost,
): boolean {
  return (
    config.precision === "fp8" &&
    !isValidFP8KernelSpeedupFactor(config.fp8.kernelSpeedupFactor)
  )
}

export function hasInvalidFP8StorageMode(config: FP8ConfigHost): boolean {
  return (
    config.precision === "fp8" && !isValidFP8StorageMode(config.fp8.storageMode)
  )
}

export function hasInvalidFP8Config(config: FP8ConfigHost): boolean {
  return (
    hasInvalidFP8KernelSpeedupFactor(config) ||
    hasInvalidFP8StorageMode(config)
  )
}
