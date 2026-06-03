import { CLOUD_INSTANCES } from "../constants"
import type { CloudInstance, TrainingConfig } from "../types"

export function resolveCloudInstance(
  instanceId: string | null,
): CloudInstance | null {
  if (instanceId === null) return null
  return (
    CLOUD_INSTANCES.find((instance) => instance.id === instanceId) ?? null
  )
}

export function getCloudInstanceGPUHourlyRate(
  instance: CloudInstance,
): number {
  return instance.pricePerHour / instance.gpuCount
}

export function getSelectedHardwareGPUId(config: TrainingConfig): string {
  return config.hardware.gpuId ?? config.hardware.gpu.id
}

export function isCloudInstanceCompatible(
  config: TrainingConfig,
  instance: CloudInstance,
): boolean {
  return (
    config.hardware.inputMode === "preset" &&
    getSelectedHardwareGPUId(config) === instance.gpuId
  )
}

export function hasInvalidCloudInstanceSelection(
  config: TrainingConfig,
): boolean {
  const instance = resolveCloudInstance(config.pricing.cloudInstanceId)

  if (config.pricing.cloudInstanceId === null) return false
  if (instance === null) return true
  return !isCloudInstanceCompatible(config, instance)
}

export function getSelectedCompatibleCloudInstance(
  config: TrainingConfig,
): CloudInstance | null {
  const instance = resolveCloudInstance(config.pricing.cloudInstanceId)

  if (instance === null || !isCloudInstanceCompatible(config, instance)) {
    return null
  }

  return instance
}

export function getBillableCloudInstanceCount(
  numGPUs: number,
  instance: CloudInstance,
): number {
  if (
    !Number.isFinite(numGPUs) ||
    numGPUs < 0 ||
    !Number.isFinite(instance.gpuCount) ||
    instance.gpuCount <= 0
  ) {
    return Number.POSITIVE_INFINITY
  }

  return Math.ceil(numGPUs / instance.gpuCount)
}

export function calculateCloudInstanceHourlyCost(
  numGPUs: number,
  hours: number,
  instance: CloudInstance,
): number {
  if (
    !Number.isFinite(hours) ||
    hours < 0 ||
    !Number.isFinite(instance.pricePerHour) ||
    instance.pricePerHour < 0
  ) {
    return Number.POSITIVE_INFINITY
  }

  const billableInstances = getBillableCloudInstanceCount(numGPUs, instance)

  if (!Number.isFinite(billableInstances)) {
    return Number.POSITIVE_INFINITY
  }

  if (billableInstances === 0 || hours === 0 || instance.pricePerHour === 0) {
    return 0
  }

  return billableInstances * hours * instance.pricePerHour
}
