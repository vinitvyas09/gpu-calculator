import { GPU_SPECS } from "../constants"
import type { GPUInputMode, GPUSpec } from "../types"

type ThroughputField = "halfPrecisionTFLOPS" | "tf32TFLOPS" | "fp8TFLOPS"

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
