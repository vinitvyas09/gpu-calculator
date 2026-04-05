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
  NumberInput,
  SelectInput,
  Stat,
  ToggleInput,
} from "./input-controls"

// ---------------------------------------------------------------------------
// Category display order + labels
// ---------------------------------------------------------------------------
const CATEGORY_LABELS: Record<GPUCategory, string> = {
  "nvidia-datacenter": "NVIDIA Datacenter",
  "nvidia-consumer": "NVIDIA Consumer",
  "amd-datacenter": "AMD Datacenter",
  "apple-silicon": "Apple Silicon",
}

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
      const g = GPU_SPECS.find((x) => x.id === gpuId) || GPU_SPECS[0]
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
    if (!gpu.supportsBF16 && precision === "bf16")
      w.push(
        `${gpu.name} does not support BF16. Training will fall back to FP16 with loss scaling, or choose a supported GPU.`,
      )
    return w
  }, [gpu, tpDegree, precision])

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
          <SelectInput
            label="GPU"
            value={gpuId || GPU_SPECS[0].id}
            onChange={setGPU}
            options={gpuOptions}
            colors={colors}
          />
          <GPUSpecsCard gpu={gpu} colors={colors} />
        </>
      ) : (
        <CustomGPUForm gpu={gpu} onChange={updateCustom} colors={colors} />
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
      className="grid grid-cols-3 gap-x-3 gap-y-2 rounded-lg border p-3"
      style={{ borderColor: colors.border, backgroundColor: colors.bg }}
    >
      <Stat
        label="VRAM"
        value={`${gpu.memoryGB} GB${gpu.memoryType === "unified" ? " (uni)" : ""}`}
        colors={colors}
      />
      <Stat
        label={gpu.halfPrecisionFormat.toUpperCase()}
        value={`${gpu.halfPrecisionTFLOPS} TFLOPS`}
        colors={colors}
      />
      <Stat
        label="Bandwidth"
        value={`${gpu.memoryBandwidthGBps} GB/s`}
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
      <NumberInput
        label="VRAM (GB)"
        value={gpu.memoryGB}
        onChange={(v) => onChange({ memoryGB: v })}
        min={1}
        colors={colors}
      />
      <NumberInput
        label="BF16/FP16 TFLOPS"
        value={gpu.halfPrecisionTFLOPS}
        onChange={(v) => onChange({ halfPrecisionTFLOPS: v })}
        min={1}
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
        label="NVLink BW (GB/s)"
        value={gpu.nvlinkBandwidthGBps || 0}
        onChange={(v) =>
          onChange({ nvlinkBandwidthGBps: v || null })
        }
        min={0}
        tooltip="Set 0 for no NVLink"
        colors={colors}
      />
      <NumberInput
        label="GPUs per node"
        value={gpu.gpusPerNode}
        onChange={(v) => onChange({ gpusPerNode: v })}
        min={1}
        colors={colors}
      />
      <NumberInput
        label="TDP (watts)"
        value={gpu.tdpWatts || 0}
        onChange={(v) => onChange({ tdpWatts: v || null })}
        min={0}
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
    </div>
  )
}
