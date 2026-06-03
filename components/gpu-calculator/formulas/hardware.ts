import { GPU_SPECS } from "../constants"
import type { GPUInputMode, GPUSpec, TrainingPrecision } from "../types"

type ThroughputField = "halfPrecisionTFLOPS" | "tf32TFLOPS" | "fp8TFLOPS"

const VALID_GPU_INPUT_MODES = new Set(["preset", "custom"])
const VALID_GPU_VENDORS = new Set(["nvidia", "amd", "apple"])
const VALID_GPU_CATEGORIES = new Set([
  "nvidia-datacenter",
  "nvidia-consumer",
  "amd-datacenter",
  "apple-silicon",
])
const VALID_TRAINING_PRECISIONS = new Set(["fp32", "bf16", "fp16", "fp8"])
const VALID_GPU_MEMORY_TYPES = new Set(["vram", "unified"])
const VALID_HALF_PRECISION_FORMATS = new Set(["bf16", "fp16"])
const VALID_INTERCONNECTS = new Set(["nvlink", "pcie", "xgmi", "none"])
const SPARSE_TFLOPS_FACTOR = 2
const SPARSE_TFLOPS_RELATIVE_TOLERANCE = 0.035
const SPARSE_TFLOPS_ABSOLUTE_TOLERANCE = 1

const THROUGHPUT_FIELDS: Array<{
  key: ThroughputField
  label: string
}> = [
  { key: "halfPrecisionTFLOPS", label: "BF16/FP16" },
  { key: "tf32TFLOPS", label: "TF32" },
  { key: "fp8TFLOPS", label: "FP8" },
]

function isPositiveFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
}

function isPositiveFiniteInteger(value: number): boolean {
  return Number.isFinite(value) && value > 0 && Number.isInteger(value)
}

function hasInvalidSetPositiveNumber(value: number | null | undefined): boolean {
  return value !== null && value !== undefined && !isPositiveFiniteNumber(value)
}

export function hasInvalidGPUInputMode(inputMode: GPUInputMode): boolean {
  return !VALID_GPU_INPUT_MODES.has(inputMode)
}

export function hasInvalidTrainingPrecision(
  precision: TrainingPrecision,
): boolean {
  return !VALID_TRAINING_PRECISIONS.has(precision)
}

export function getInvalidCustomGPUMetadataMessages(gpu: GPUSpec): string[] {
  const messages: string[] = []

  if (!VALID_GPU_VENDORS.has(gpu.vendor)) {
    messages.push("Custom GPU vendor must be NVIDIA, AMD, or Apple.")
  }

  if (!VALID_GPU_CATEGORIES.has(gpu.category)) {
    messages.push("Custom GPU category must be a supported category.")
  }

  if (!VALID_GPU_MEMORY_TYPES.has(gpu.memoryType)) {
    messages.push("Custom GPU memory type must be VRAM or unified memory.")
  }

  if (!VALID_HALF_PRECISION_FORMATS.has(gpu.halfPrecisionFormat)) {
    messages.push("Custom GPU half-precision format must be BF16 or FP16.")
  }

  if (!VALID_INTERCONNECTS.has(gpu.interconnect)) {
    messages.push("Custom GPU interconnect must be NVLink, PCIe, xGMI, or none.")
  }

  if (typeof gpu.singleDeviceOnly !== "boolean") {
    messages.push("Custom GPU single-device-only flag must be true or false.")
  }

  if (typeof gpu.supportsBF16 !== "boolean") {
    messages.push("Custom GPU BF16 support flag must be true or false.")
  }

  if (typeof gpu.supportsTF32 !== "boolean") {
    messages.push("Custom GPU TF32 support flag must be true or false.")
  }

  if (typeof gpu.supportsFP8 !== "boolean") {
    messages.push("Custom GPU FP8 support flag must be true or false.")
  }

  return messages
}

function hasInvalidCustomGPUMetadata(gpu: GPUSpec): boolean {
  return getInvalidCustomGPUMetadataMessages(gpu).length > 0
}

function hasInvalidCustomGPUFP32Throughput(gpu: GPUSpec): boolean {
  const hasValidTF32 =
    gpu.supportsTF32 && isPositiveFiniteNumber(gpu.tf32TFLOPS)

  if (
    gpu.supportsTF32 &&
    hasInvalidSetPositiveNumber(gpu.tf32TFLOPS)
  ) {
    return true
  }

  return !hasValidTF32 && hasInvalidSetPositiveNumber(gpu.fp32TFLOPS)
}

function hasInvalidGPUTrainingHardwareFields(
  gpu: GPUSpec,
  precision: TrainingPrecision,
): boolean {
  if (
    hasInvalidCustomGPUMetadata(gpu) ||
    !isPositiveFiniteNumber(gpu.memoryGB) ||
    !isPositiveFiniteNumber(gpu.halfPrecisionTFLOPS) ||
    !isPositiveFiniteNumber(gpu.memoryBandwidthGBps) ||
    !isPositiveFiniteInteger(gpu.gpusPerNode)
  ) {
    return true
  }

  return precision === "fp32" && hasInvalidCustomGPUFP32Throughput(gpu)
}

export function hasInvalidCustomGPUTrainingHardware(
  inputMode: GPUInputMode,
  gpu: GPUSpec,
  precision: TrainingPrecision,
): boolean {
  if (
    hasInvalidGPUInputMode(inputMode) ||
    hasInvalidTrainingPrecision(precision)
  ) {
    return true
  }

  if (inputMode !== "custom") {
    return false
  }

  return hasInvalidGPUTrainingHardwareFields(gpu, precision)
}

export function hasUnsupportedTrainingPrecision(
  gpu: GPUSpec,
  precision: TrainingPrecision,
): boolean {
  if (hasInvalidTrainingPrecision(precision)) {
    return true
  }

  return (
    (precision === "bf16" && gpu.supportsBF16 !== true) ||
    (precision === "fp8" && gpu.supportsFP8 !== true)
  )
}

export function hasInvalidTrainingHardware(
  inputMode: GPUInputMode,
  gpu: GPUSpec,
  precision: TrainingPrecision,
): boolean {
  if (
    hasInvalidGPUInputMode(inputMode) ||
    hasInvalidTrainingPrecision(precision) ||
    hasInvalidGPUTrainingHardwareFields(gpu, precision)
  ) {
    return true
  }

  return (
    hasUnsupportedTrainingPrecision(gpu, precision)
  )
}

export function getParallelismLocalGroupSize(gpu: GPUSpec): number {
  const configuredGroupSize =
    Number.isFinite(gpu.gpusPerNode) && gpu.gpusPerNode > 0
      ? Math.floor(gpu.gpusPerNode)
      : 1

  if (gpu.singleDeviceOnly !== false) {
    return 1
  }

  // Bridge-only NVLink SKUs can sit in larger hosts, but TP/EP traffic should
  // stay inside the fast pair rather than assuming all GPUs share NVSwitch.
  if (gpu.id === "h100-nvl" || gpu.id === "rtx-3090") {
    return Math.min(configuredGroupSize, 2)
  }

  return configuredGroupSize
}

function formatTFLOPS(value: number): string {
  if (value >= 100) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 })
  }

  return value.toLocaleString(undefined, { maximumFractionDigits: 1 })
}

function isCloseToSparsePeak(value: number, denseValue: number): boolean {
  const sparseValue = denseValue * SPARSE_TFLOPS_FACTOR
  const tolerance = Math.max(
    SPARSE_TFLOPS_ABSOLUTE_TOLERANCE,
    sparseValue * SPARSE_TFLOPS_RELATIVE_TOLERANCE,
  )

  return Math.abs(value - sparseValue) <= tolerance
}

function findSparsePeakMatch(value: number, field: ThroughputField): GPUSpec | null {
  return (
    GPU_SPECS.find((preset) => {
      const denseValue = preset[field]
      return (
        isPositiveFiniteNumber(denseValue) &&
        isCloseToSparsePeak(value, denseValue)
      )
    }) ?? null
  )
}

export function getSparseThroughputWarningMessages(
  gpu: GPUSpec,
  inputMode: GPUInputMode,
): string[] {
  if (inputMode !== "custom") {
    return []
  }

  return THROUGHPUT_FIELDS.flatMap(({ key, label }) => {
    const value = gpu[key]
    if (!isPositiveFiniteNumber(value)) {
      return []
    }

    const matchingPreset = findSparsePeakMatch(value, key)
    if (matchingPreset === null) {
      return []
    }

    const denseValue = matchingPreset[key]
    if (!isPositiveFiniteNumber(denseValue)) {
      return []
    }

    return [
      `Custom ${label} TFLOPS (${formatTFLOPS(value)}) is close to 2x the dense ${matchingPreset.name} value (${formatTFLOPS(denseValue)}). NVIDIA spec sheets often quote 2:4 structured-sparsity peaks; dense training estimates should use unsparsified TFLOPS or time can be understated by about 2x.`,
    ]
  })
}
