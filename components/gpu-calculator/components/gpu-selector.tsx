"use client"

import { useMemo } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { AlertTriangle } from "lucide-react"
import type {
  GPUCategory,
  GPUInputMode,
  GPUSpec,
  InterconnectType,
  TrainingPrecision,
} from "../types"
import { GPU_SPECS } from "../constants"
import {
  type CalculatorColors,
  CollapsibleSection,
  NumberInput,
  SearchableSelect,
  SelectInput,
  Stat,
  ToggleInput,
} from "./input-controls"
import { getSparseThroughputWarningMessages } from "../formulas/hardware"

// ---------------------------------------------------------------------------
// Category display order + labels
// ---------------------------------------------------------------------------
const CATEGORY_LABELS: Record<GPUCategory, string> = {
  "nvidia-datacenter": "NVIDIA Datacenter",
  "nvidia-consumer": "NVIDIA Consumer",
  "amd-datacenter": "AMD Datacenter",
  "apple-silicon": "Apple Silicon",
}

const PRE_AMPERE_NVIDIA_IDS = new Set(["v100-32gb", "t4"])

// ---------------------------------------------------------------------------
// GPUSelector
// ---------------------------------------------------------------------------
export function GPUSelector({
  gpuId,
  gpu,
  inputMode,
  onChange,
  colors,
  tpDegree = 1,
  precision = "bf16",
}: {
  gpuId: string | null
  gpu: GPUSpec
  inputMode: GPUInputMode
  onChange: (u: {
    gpuId: string | null
    gpu: GPUSpec
    inputMode: GPUInputMode
  }) => void
  colors: CalculatorColors
  tpDegree?: number
  precision?: TrainingPrecision
}) {
  const gpuOptions = useMemo(
    () =>
      GPU_SPECS.map((g) => ({
        value: g.id,
        label: `${g.name} (${g.memoryGB} GB)`,
        group: CATEGORY_LABELS[g.category],
      })),
    [],
  )

  const setMode = (mode: GPUInputMode) => {
    if (mode === "preset") {
      const g =
        GPU_SPECS.find((x) => x.id === (gpuId ?? gpu.id)) ||
        GPU_SPECS[0]
      onChange({ inputMode: mode, gpu: g, gpuId: g.id })
    } else {
      onChange({ inputMode: mode, gpu, gpuId: null })
    }
  }

  const setGPU = (id: string) => {
    const g = GPU_SPECS.find((x) => x.id === id)
    if (g) onChange({ inputMode: "preset", gpu: g, gpuId: id })
  }

  const updateCustom = (patch: Partial<GPUSpec>) =>
    onChange({ inputMode: "custom", gpu: { ...gpu, ...patch }, gpuId: null })

  // Warnings
  const warnings = useMemo(() => {
    const w: string[] = []
    if (gpu.interconnect === "pcie" && tpDegree > 1)
      w.push(
        "PCIe interconnect with tensor parallelism > 1 will have very low inter-GPU bandwidth. Consider NVLink-equipped GPUs for TP.",
      )
    if (precision === "bf16" && !gpu.supportsBF16) {
      if (
        gpu.vendor === "nvidia" &&
        PRE_AMPERE_NVIDIA_IDS.has(gpu.id)
      ) {
        w.push(
          `${gpu.name} is a pre-Ampere NVIDIA GPU and cannot execute BF16 kernels. Use FP16 instead, or switch to Ampere-or-newer hardware.`,
        )
      } else {
        w.push(
          `${gpu.name} does not support BF16. Select FP16 or different hardware for native BF16 execution.`,
        )
      }
    }
    w.push(...getSparseThroughputWarningMessages(gpu, inputMode))
    return w
  }, [gpu, inputMode, tpDegree, precision])

  return (
    <div className="space-y-3">
      {/* Preset / Custom toggle */}
      <div
        className="flex gap-0.5 rounded-lg p-[3px]"
        style={{ backgroundColor: colors.bg }}
      >
        {(
          [
            { key: "preset" as const, label: "GPU Preset" },
            { key: "custom" as const, label: "Custom GPU" },
          ] as const
        ).map(({ key, label }) => (
          <button
            key={key}
            type="button"
            onClick={() => setMode(key)}
            className="flex flex-1 items-center justify-center rounded-md px-3 py-[7px] text-[11px] font-semibold tracking-wide transition-all"
            style={{
              backgroundColor:
                inputMode === key ? colors.cardBg : "transparent",
              color:
                inputMode === key ? colors.accent : colors.textSecondary,
              boxShadow:
                inputMode === key
                  ? `0 1px 4px ${colors.accentMuted}, 0 0 0 1px ${colors.border}`
                  : "none",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {inputMode === "preset" ? (
        <>
          <SearchableSelect
            label="GPU"
            value={gpuId || GPU_SPECS[0].id}
            onChange={setGPU}
            options={gpuOptions}
            colors={colors}
          />
          <GPUSpecsCard gpu={gpu} colors={colors} />
        </>
      ) : (
        <CollapsibleSection
          title="Custom GPU specs"
          defaultOpen
          colors={colors}
        >
          <CustomGPUForm gpu={gpu} onChange={updateCustom} colors={colors} />
        </CollapsibleSection>
      )}

      {/* Warnings */}
      <AnimatePresence>
        {warnings.map((msg) => (
          <motion.div
            key={msg}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="flex gap-2 overflow-hidden rounded-lg border px-3 py-2 text-[11px] leading-relaxed"
            style={{
              borderColor: colors.warningBorder,
              backgroundColor: colors.warningBg,
              color: colors.warning,
            }}
          >
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{msg}</span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

// ---------------------------------------------------------------------------
// GPU specs readout
// ---------------------------------------------------------------------------
function GPUSpecsCard({
  gpu,
  colors,
}: {
  gpu: GPUSpec
  colors: CalculatorColors
}) {
  return (
    <div
      className="grid grid-cols-2 gap-x-3 gap-y-2 rounded-lg border p-3 sm:grid-cols-3"
      style={{ borderColor: colors.border, backgroundColor: colors.bg }}
    >
      <Stat
        label="VRAM"
        value={`${gpu.memoryGB} GB${gpu.memoryType === "unified" ? " (uni)" : ""}`}
        colors={colors}
      />
      <Stat
        label={`Dense ${gpu.halfPrecisionFormat.toUpperCase()}`}
        value={`${gpu.halfPrecisionTFLOPS} TFLOPS`}
        colors={colors}
      />
      <Stat
        label="Dense TF32"
        value={gpu.tf32TFLOPS ? `${gpu.tf32TFLOPS} TFLOPS` : "N/A"}
        colors={colors}
      />
      <Stat
        label="Bandwidth"
        value={`${gpu.memoryBandwidthGBps} GB/s`}
        colors={colors}
      />
      <Stat
        label="Dense FP8"
        value={gpu.fp8TFLOPS ? `${gpu.fp8TFLOPS} TFLOPS` : "N/A"}
        colors={colors}
      />
      <Stat
        label="Interconnect"
        value={
          gpu.nvlinkBandwidthGBps
            ? `NVLink ${gpu.nvlinkBandwidthGBps} GB/s`
            : gpu.interconnect.toUpperCase()
        }
        colors={colors}
      />
      <Stat
        label="GPUs / node"
        value={String(gpu.gpusPerNode)}
        colors={colors}
      />
      <Stat
        label="TDP"
        value={gpu.tdpWatts ? `${gpu.tdpWatts} W` : "N/A"}
        colors={colors}
      />
      <Stat
        label="Modes"
        value={[
          gpu.supportsBF16 ? "BF16" : null,
          gpu.supportsTF32 ? "TF32" : null,
          gpu.supportsFP8 ? "FP8" : null,
        ]
          .filter(Boolean)
          .join(" / ") || "FP16 only"}
        colors={colors}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Custom GPU form
// ---------------------------------------------------------------------------
function CustomGPUForm({
  gpu,
  onChange,
  colors,
}: {
  gpu: GPUSpec
  onChange: (patch: Partial<GPUSpec>) => void
  colors: CalculatorColors
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <SelectInput
        label="Vendor"
        value={gpu.vendor}
        onChange={(vendor) =>
          onChange({ vendor: vendor as GPUSpec["vendor"] })
        }
        options={[
          { value: "nvidia", label: "NVIDIA" },
          { value: "amd", label: "AMD" },
          { value: "apple", label: "Apple" },
        ]}
        colors={colors}
      />
      <SelectInput
        label="Category"
        value={gpu.category}
        onChange={(category) =>
          onChange({ category: category as GPUCategory })
        }
        options={[
          {
            value: "nvidia-datacenter",
            label: "NVIDIA Datacenter",
          },
          {
            value: "nvidia-consumer",
            label: "NVIDIA Consumer",
          },
          { value: "amd-datacenter", label: "AMD Datacenter" },
          { value: "apple-silicon", label: "Apple Silicon" },
        ]}
        colors={colors}
      />
      <SelectInput
        label="Memory type"
        value={gpu.memoryType}
        onChange={(memoryType) =>
          onChange({
            memoryType: memoryType as GPUSpec["memoryType"],
          })
        }
        options={[
          { value: "vram", label: "Discrete VRAM" },
          { value: "unified", label: "Unified memory" },
        ]}
        colors={colors}
      />
      <SelectInput
        label="Half precision mode"
        value={gpu.halfPrecisionFormat}
        onChange={(halfPrecisionFormat) =>
          onChange({
            halfPrecisionFormat:
              halfPrecisionFormat as GPUSpec["halfPrecisionFormat"],
          })
        }
        options={[
          { value: "bf16", label: "BF16 throughput" },
          { value: "fp16", label: "FP16 throughput" },
        ]}
        colors={colors}
      />
      <NumberInput
        label="VRAM (GB)"
        value={gpu.memoryGB}
        onChange={(v) => onChange({ memoryGB: v })}
        min={1}
        colors={colors}
      />
      <NumberInput
        label="Dense BF16/FP16 TFLOPS"
        value={gpu.halfPrecisionTFLOPS}
        onChange={(v) => onChange({ halfPrecisionTFLOPS: v })}
        min={1}
        tooltip="Use dense, unsparsified tensor-core throughput. Vendor sheets often quote 2:4 sparsity peaks that are about 2x higher."
        colors={colors}
      />
      <NumberInput
        label="Mem bandwidth (GB/s)"
        value={gpu.memoryBandwidthGBps}
        onChange={(v) => onChange({ memoryBandwidthGBps: v })}
        min={1}
        colors={colors}
      />
      <NumberInput
        label="Dense TF32 TFLOPS"
        value={gpu.tf32TFLOPS || 0}
        onChange={(v) => onChange({ tf32TFLOPS: v || null })}
        min={0}
        tooltip="Used for fp32 training on Ampere+ GPUs. Use dense, unsparsified throughput."
        colors={colors}
      />
      <NumberInput
        label="FP32 TFLOPS"
        value={gpu.fp32TFLOPS || 0}
        onChange={(v) => onChange({ fp32TFLOPS: v || null })}
        min={0}
        tooltip="Used for fp32 training when TF32 is unavailable; set 0 to use the BF16/FP16 / 8 fallback."
        colors={colors}
      />
      <NumberInput
        label="Dense FP8 TFLOPS"
        value={gpu.fp8TFLOPS || 0}
        onChange={(v) => onChange({ fp8TFLOPS: v || null })}
        min={0}
        tooltip="Set 0 if the device has no FP8 path. Training time uses the configured FP8 kernel speedup, not raw spec-sheet FP8 peak."
        colors={colors}
      />
      <NumberInput
        label="GPUs per node"
        value={gpu.gpusPerNode}
        onChange={(v) => onChange({ gpusPerNode: v })}
        min={1}
        integer
        colors={colors}
      />
      <SelectInput
        label="Interconnect"
        value={gpu.interconnect}
        onChange={(v) =>
          onChange({ interconnect: v as InterconnectType })
        }
        options={[
          { value: "nvlink", label: "NVLink" },
          { value: "pcie", label: "PCIe" },
          { value: "xgmi", label: "xGMI (AMD)" },
          { value: "none", label: "None" },
        ]}
        colors={colors}
      />
      <ToggleInput
        label="Supports BF16"
        value={gpu.supportsBF16}
        onChange={(v) => onChange({ supportsBF16: v })}
        colors={colors}
      />
      <ToggleInput
        label="Supports TF32"
        value={gpu.supportsTF32}
        onChange={(v) => onChange({ supportsTF32: v })}
        colors={colors}
      />
      <ToggleInput
        label="Supports FP8"
        value={gpu.supportsFP8}
        onChange={(v) => onChange({ supportsFP8: v })}
        colors={colors}
      />
      <ToggleInput
        label="Single-device only"
        value={gpu.singleDeviceOnly}
        onChange={(v) => onChange({ singleDeviceOnly: v })}
        tooltip="Useful for Apple Silicon and other non-multi-GPU setups."
        colors={colors}
      />
    </div>
  )
}
