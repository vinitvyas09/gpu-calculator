"use client"

import { useMemo, type ReactNode } from "react"
import {
  Brain,
  Database,
  HardDrive,
  Network,
  SlidersHorizontal,
} from "lucide-react"
import type {
  CheckpointingMode,
  CPUOffloadMode,
  FSDPStrategy,
  FrameworkType,
  GradientPrecision,
  HardwareSelection,
  InterNodeBandwidthPreset,
  ModelSelection,
  OptimizerType,
  ParallelismConfig,
  ParallelismMode,
  PricingConfig,
  ParallelismRecommendation,
  SequenceParallelismMode,
  TrainingConfig,
  TrainingPrecision,
  ZeROCommunicationBucketMode,
  ZeROCommunicationConfig,
  ZeROStage,
} from "../types"
import {
  CLOUD_PRICING_PRESETS,
  OPTIMIZER_PROFILES,
} from "../constants"
import {
  getEffectiveDefaultTrainingMFU,
  MAX_MFU_OVERRIDE,
} from "../formulas/cost"
import {
  type CalculatorColors,
  CollapsibleSection,
  NumberInput,
  SelectInput,
  SliderInput,
  ToggleInput,
  formatPercent,
} from "./input-controls"
import {
  ModelSelector,
  getModelPresetDefaultSequenceLength,
} from "./model-selector"
import { GPUSelector } from "./gpu-selector"

// ---------------------------------------------------------------------------
// Section header — icon + uppercase title + faint rule
// ---------------------------------------------------------------------------
function Section({
  title,
  icon: Icon,
  colors,
  children,
}: {
  title: string
  icon: typeof Brain
  colors: CalculatorColors
  children: ReactNode
}) {
  return (
    <div>
      <div className="mb-4 flex items-center gap-2">
        <Icon className="h-3.5 w-3.5" style={{ color: colors.accent }} />
        <span
          className="text-[11px] font-medium uppercase tracking-[0.12em]"
          style={{ color: colors.accent }}
        >
          {title}
        </span>
        <div
          className="h-px flex-1"
          style={{ backgroundColor: colors.border }}
        />
      </div>
      {children}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Subsection label inside the advanced collapsible
// ---------------------------------------------------------------------------
function SubLabel({
  children,
  colors,
}: {
  children: string
  colors: CalculatorColors
}) {
  return (
    <div
      className="mb-2 mt-1 text-[10px] font-semibold uppercase tracking-[0.1em]"
      style={{ color: colors.textSecondary }}
    >
      {children}
    </div>
  )
}

function resolveFSDPZeroStage(
  strategy: FSDPStrategy | null,
): ZeROStage {
  switch (strategy ?? "FULL_SHARD") {
    case "NO_SHARD":
      return 0
    case "SHARD_GRAD_OP":
    case "HYBRID_SHARD_ZERO2":
      return 2
    case "FULL_SHARD":
    case "HYBRID_SHARD":
    default:
      return 3
  }
}

function getMaxTransformerLayersPerPipelineStage(
  config: TrainingConfig,
  parallelism: ParallelismConfig,
): number {
  const layers =
    Number.isFinite(config.model.architecture.L) &&
    config.model.architecture.L > 0
      ? Math.max(1, Math.floor(config.model.architecture.L))
      : 1
  const N_pp =
    Number.isFinite(parallelism.N_pp) && parallelism.N_pp > 0
      ? Math.max(1, Math.floor(parallelism.N_pp))
      : 1

  return Math.max(1, Math.ceil(layers / N_pp))
}

// ---------------------------------------------------------------------------
// PretrainingPanel
// ---------------------------------------------------------------------------
export function PretrainingPanel({
  config,
  onChange,
  colors,
  activeParameterCount,
  effectiveNumGPUs,
  gpuCountDerivedFromTarget,
  autoParallelismRecommendation,
}: {
  config: TrainingConfig
  onChange: (c: TrainingConfig) => void
  colors: CalculatorColors
  activeParameterCount: number
  effectiveNumGPUs: number
  gpuCountDerivedFromTarget: boolean
  autoParallelismRecommendation: ParallelismRecommendation
}) {
  // Convenience updaters for nested state
  const set = (patch: Partial<TrainingConfig>) =>
    onChange({ ...config, ...patch })

  const setModel = (model: ModelSelection) => {
    const previousDefaultSequenceLength =
      config.model.inputMode === "preset"
        ? getModelPresetDefaultSequenceLength(config.model.presetId)
        : null
    const nextDefaultSequenceLength =
      model.inputMode === "preset"
        ? getModelPresetDefaultSequenceLength(model.presetId)
        : null
    const presetChanged =
      config.model.inputMode !== model.inputMode ||
      config.model.presetId !== model.presetId
    const shouldTrackPresetDefault =
      presetChanged &&
      nextDefaultSequenceLength !== null &&
      (previousDefaultSequenceLength === null ||
        config.sequenceLength === previousDefaultSequenceLength)

    onChange({
      ...config,
      model,
      sequenceLength: shouldTrackPresetDefault
        ? nextDefaultSequenceLength
        : config.sequenceLength,
    })
  }

  const setHw = (patch: Partial<HardwareSelection>) =>
    onChange({ ...config, hardware: { ...config.hardware, ...patch } })

  const setHardwareSelection = (hardware: {
    gpuId: string | null
    gpu: HardwareSelection["gpu"]
    inputMode: HardwareSelection["inputMode"]
  }) => {
    const nextConfig: TrainingConfig = {
      ...config,
      hardware: { ...config.hardware, ...hardware },
    }

    if (config.pricing.cloudPricingPresetId !== null) {
      const matchingPricePreset =
        hardware.inputMode === "preset"
          ? CLOUD_PRICING_PRESETS.find(
              (preset) => preset.gpuId === (hardware.gpuId ?? hardware.gpu.id),
            )
          : undefined

      nextConfig.pricing = matchingPricePreset
        ? {
            ...config.pricing,
            cloudPricingPresetId: matchingPricePreset.id,
            costPerGPUHour: matchingPricePreset.priceDefault,
          }
        : {
            ...config.pricing,
            cloudPricingPresetId: null,
          }
    }

    onChange(nextConfig)
  }

  const setPar = (patch: Partial<ParallelismConfig>) => {
    const nextParallelism = { ...config.parallelism, ...patch }

    if (nextParallelism.framework === "fsdp") {
      const fsdpStrategy = nextParallelism.fsdpStrategy ?? "FULL_SHARD"

      onChange({
        ...config,
        parallelism: {
          ...nextParallelism,
          fsdpStrategy,
          zeroStage: resolveFSDPZeroStage(fsdpStrategy),
        },
      })
      return
    }

    onChange({
      ...config,
      parallelism: {
        ...nextParallelism,
        fsdpStrategy: null,
      },
    })
  }

  const setPrice = (patch: Partial<PricingConfig>) =>
    onChange({ ...config, pricing: { ...config.pricing, ...patch } })

  const setZero = (patch: Partial<ZeROCommunicationConfig>) =>
    onChange({
      ...config,
      zeroCommunication: { ...config.zeroCommunication, ...patch },
    })

  const setTotalTokens = (totalTokens: number) =>
    onChange({
      ...config,
      totalTokens,
      uniqueTokens:
        config.uniqueTokens === config.totalTokens
          ? totalTokens
          : config.uniqueTokens,
    })

  // Derived
  const optimizerOptions = OPTIMIZER_PROFILES.filter(
    (o) => o.supportsPretraining,
  ).map((o) => ({ value: o.id, label: o.name }))

  const cloudPresetOptions = [
    ...CLOUD_PRICING_PRESETS.map((p) => ({
      value: p.id,
      label: `${p.label} ($${p.priceLow}-${p.priceHigh}/hr)`,
    })),
    { value: "custom", label: "Custom price" },
  ]

  const isQuickMode = config.model.inputMode === "quick"
  const moeEnabled =
    config.model.moe.enabled ||
    config.model.architecture.ffnType === "moe"
  const displayParallelism =
    config.parallelismMode === "auto"
      ? autoParallelismRecommendation.config
      : config.parallelism
  const defaultMFU = useMemo(() => {
    return getEffectiveDefaultTrainingMFU(
      {
        ...config,
        parallelism: displayParallelism,
        hardware: {
          ...config.hardware,
          numGPUs: effectiveNumGPUs,
        },
      },
      activeParameterCount,
      effectiveNumGPUs,
    )
  }, [activeParameterCount, config, displayParallelism, effectiveNumGPUs])
  const hasMFUOverride = config.mfuOverride !== null
  const displayMoeEnabled =
    moeEnabled && displayParallelism.N_ep > 1
  const autoLayoutParts: string[] = [`DP=${displayParallelism.N_dp}`]
  if (displayParallelism.N_tp > 1)
    autoLayoutParts.push(`TP=${displayParallelism.N_tp}`)
  if (displayMoeEnabled)
    autoLayoutParts.push(`EP=${displayParallelism.N_ep}`)
  if (displayParallelism.N_pp > 1)
    autoLayoutParts.push(`PP=${displayParallelism.N_pp}`)
  if (displayParallelism.N_cp > 1)
    autoLayoutParts.push(`CP=${displayParallelism.N_cp}`)
  if (displayParallelism.N_pp > 1 && displayParallelism.VP > 1)
    autoLayoutParts.push(`VP=${displayParallelism.VP}`)
  autoLayoutParts.push(
    displayParallelism.framework === "fsdp" && displayParallelism.fsdpStrategy !== null
      ? `FSDP ${displayParallelism.fsdpStrategy}`
      : `ZeRO-${displayParallelism.zeroStage}`,
  )
  const zero3ForcesOverlapComm =
    displayParallelism.framework !== "fsdp" &&
    displayParallelism.zeroStage === 3
  const effectiveOverlapComm =
    config.zeroCommunication.overlapComm || zero3ForcesOverlapComm
  const maxCheckpointedLayersPerStage =
    getMaxTransformerLayersPerPipelineStage(config, displayParallelism)

  return (
    <div className="space-y-8">
      {/* ——— 1. Model specification ——— */}
      <Section title="Model" icon={Brain} colors={colors}>
        <ModelSelector
          selection={config.model}
          onChange={setModel}
          quickTokens={config.totalTokens}
          onQuickTokensChange={setTotalTokens}
          colors={colors}
        />
      </Section>

      {/* ——— 2–2a. Training data ——— */}
      <Section title="Training Data" icon={Database} colors={colors}>
        <div className="grid gap-3 sm:grid-cols-2">
          {!isQuickMode && (
            <NumberInput
              label="Total tokens (D)"
              value={config.totalTokens}
            onChange={setTotalTokens}
            min={1e6}
            max={1e16}
            integer
            compact
            tooltip="Total training tokens including any repetition"
            colors={colors}
            />
          )}
          <NumberInput
            label="Unique tokens (U)"
            value={config.uniqueTokens}
            onChange={(v) => set({ uniqueTokens: v })}
            min={1e6}
            max={1e16}
            integer
            compact
            tooltip="Unique tokens in dataset. Use U > D for less than one epoch over a larger corpus."
            colors={colors}
          />
        </div>
        {isQuickMode && (
          <p
            className="mt-3 text-xs leading-6"
            style={{ color: colors.textSecondary }}
          >
            Total tokens are edited from the Quick model tab so the model-size and
            dataset-size inputs stay grouped in the fast-estimate workflow.
          </p>
        )}
      </Section>

      {/* ——— 3–9. Training configuration ——— */}
      <Section title="Training Setup" icon={SlidersHorizontal} colors={colors}>
        <div className="grid gap-3 sm:grid-cols-2">
          {/* 3 */}
          <SelectInput
            label="Precision"
            value={config.precision}
            onChange={(v) =>
              set({
                precision: v as TrainingPrecision,
                fp8: {
                  ...config.fp8,
                  enabled: v === "fp8",
                },
              })
            }
            options={[
              { value: "bf16", label: "BF16" },
              { value: "fp16", label: "FP16" },
              { value: "fp32", label: "FP32" },
              { value: "fp8", label: "FP8" },
            ]}
            tooltip="Training precision — BF16 is standard for modern GPUs"
            colors={colors}
          />
          {/* 4 */}
          <SelectInput
            label="Optimizer"
            value={config.optimizer}
            onChange={(v) => set({ optimizer: v as OptimizerType })}
            options={optimizerOptions}
            colors={colors}
          />
          {/* 4a */}
          <SelectInput
            label="Gradient precision"
            value={config.gradientPrecision}
            onChange={(v) =>
              set({ gradientPrecision: v as GradientPrecision })
            }
            options={[
              { value: "fp32", label: "FP32 (default)" },
              { value: "bf16", label: "BF16" },
            ]}
            tooltip="Precision for gradient accumulation — affects memory footprint"
            colors={colors}
          />
          {/* 5 */}
          <NumberInput
            label="Micro-batch size (b)"
            value={config.microBatchSize}
            onChange={(v) => set({ microBatchSize: v })}
            min={1}
            integer
            tooltip="Per-GPU micro-batch size in sequences"
            colors={colors}
          />
          {/* 6 */}
          <NumberInput
            label="Sequence length (s)"
            value={config.sequenceLength}
            onChange={(v) => set({ sequenceLength: v })}
            min={128}
            step={128}
            integer
            tooltip="Maximum sequence length in tokens"
            colors={colors}
          />
          {/* 7 */}
          <NumberInput
            label="Grad accum steps (G)"
            value={config.gradientAccumulationSteps}
            onChange={(v) =>
              set({ gradientAccumulationSteps: v })
            }
            min={1}
            integer
            tooltip="Gradient accumulation steps before weight update"
            colors={colors}
          />
          {/* 8 */}
          <SelectInput
            label="Activation checkpointing"
            value={config.activationCheckpointing}
            onChange={(v) =>
              set({
                activationCheckpointing: v as CheckpointingMode,
                partialCheckpointDepth:
                  v === "partial"
                    ? config.partialCheckpointDepth ?? 1
                    : null,
              })
            }
            options={[
              { value: "none", label: "None" },
              { value: "selective", label: "Selective" },
              { value: "full", label: "Full" },
              { value: "partial", label: "Partial" },
            ]}
            tooltip="Trade compute for memory by recomputing activations"
            colors={colors}
          />
        </div>
        {/* 9 — full width toggle */}
        <div className="mt-3">
          <ToggleInput
            label="Flash Attention"
            value={config.flashAttention}
            onChange={(v) => set({ flashAttention: v })}
            tooltip="Use FlashAttention for fused, memory-efficient attention"
            colors={colors}
          />
        </div>
      </Section>

      {/* ——— 10–13, 15. Hardware & cost ——— */}
      <Section title="Hardware" icon={HardDrive} colors={colors}>
        {/* 10 */}
        <GPUSelector
          gpuId={config.hardware.gpuId}
          gpu={config.hardware.gpu}
          inputMode={config.hardware.inputMode}
          onChange={setHardwareSelection}
          colors={colors}
          tpDegree={displayParallelism.N_tp}
          precision={config.precision}
        />

        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {/* 12 */}
          <NumberInput
            label="Number of GPUs"
            value={gpuCountDerivedFromTarget ? effectiveNumGPUs : (config.hardware.numGPUs ?? 8)}
            onChange={(v) => setHw({ numGPUs: v })}
            min={1}
            integer
            tooltip={
              gpuCountDerivedFromTarget
                ? "Resolved from the target training time and current memory/topology constraints. Set target training days to 0 to enter GPU count directly."
                : "Total GPU count across all nodes"
            }
            colors={colors}
            disabled={gpuCountDerivedFromTarget}
          />
          {/* 11 */}
          <NumberInput
            label="Target training days"
            value={config.hardware.targetTrainingDays ?? 0}
            onChange={(v) =>
              setHw({ targetTrainingDays: v > 0 ? v : null })
            }
            min={0}
            tooltip="Optional — set 0 to compute time from GPU count instead"
            colors={colors}
          />
        </div>
        {gpuCountDerivedFromTarget && (
          <p
            className="mt-3 text-xs leading-6"
            style={{ color: colors.textSecondary }}
          >
            GPU count is resolved from the target training-time constraint plus
            current memory, topology, MFU, and schedule assumptions.
          </p>
        )}

        {/* 13 — MFU slider */}
        <div className="mt-3">
          <ToggleInput
            label="Override MFU default"
            value={hasMFUOverride}
            onChange={(enabled) =>
              set({
                mfuOverride: enabled ? defaultMFU : null,
              })
            }
            tooltip={`Use a manual MFU instead of the smart default (${formatPercent(defaultMFU)} for the current model, GPU count, checkpointing, and pipeline schedule).`}
            colors={colors}
          />
          <SliderInput
            label="MFU Override"
            value={config.mfuOverride ?? defaultMFU}
            onChange={(v) => set({ mfuOverride: v })}
            min={0.01}
            max={MAX_MFU_OVERRIDE}
            step={0.01}
            formatDisplay={(n) => formatPercent(n)}
            tooltip="End-to-end Model FLOPS Utilization after schedule and system overhead"
            colors={colors}
            disabled={!hasMFUOverride}
          />
        </div>

        {/* 15 — Cost */}
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <SelectInput
            label="Cloud pricing preset"
            value={config.pricing.cloudPricingPresetId || "custom"}
            onChange={(v) => {
              const preset = CLOUD_PRICING_PRESETS.find(
                (p) => p.id === v,
              )
              if (preset) {
                setPrice({
                  cloudPricingPresetId: v,
                  costPerGPUHour: preset.priceDefault,
                })
              } else {
                setPrice({ cloudPricingPresetId: null })
              }
            }}
            options={cloudPresetOptions}
            tooltip="Representative on-demand defaults; cloud prices change often, so override with your actual quote or committed-use rate."
            colors={colors}
          />
          <NumberInput
            label="Cost per GPU-hour"
            value={config.pricing.costPerGPUHour}
            onChange={(v) =>
              setPrice({ costPerGPUHour: v, cloudPricingPresetId: null })
            }
            min={0}
            step={0.1}
            unit="$/hr"
            colors={colors}
          />
        </div>
      </Section>

      {/* ——— 14. Parallelism ——— */}
      <Section title="Parallelism" icon={Network} colors={colors}>
        <SelectInput
          label="Mode"
          value={config.parallelismMode}
          onChange={(v) => set({ parallelismMode: v as ParallelismMode })}
          options={[
            {
              value: "auto",
              label: "Auto-recommend",
            },
            { value: "manual", label: "Manual configuration" },
          ]}
          tooltip="Auto uses the parallelism recommendation engine; manual lets you set each dimension"
          colors={colors}
        />

        {config.parallelismMode === "auto" && (
          <div
            className="mt-3 space-y-3 rounded-xl border p-4"
            style={{
              borderColor: colors.border,
              backgroundColor: colors.bg,
            }}
          >
            <div>
              <div
                className="text-[10px] font-semibold uppercase tracking-[0.08em]"
                style={{ color: colors.textSecondary }}
              >
                Live Recommendation
              </div>
              <p
                className="mt-2 font-mono text-sm leading-6"
                style={{ color: colors.text }}
              >
                {autoLayoutParts.join(", ")}
              </p>
              <p
                className="mt-2 text-xs leading-6"
                style={{ color: colors.textSecondary }}
              >
                {autoParallelismRecommendation.strategyLabel}
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <div
                  className="text-[10px] font-semibold uppercase tracking-[0.08em]"
                  style={{ color: colors.textSecondary }}
                >
                  Minimum GPUs
                </div>
                <div className="mt-1 text-sm font-semibold" style={{ color: colors.text }}>
                  {autoParallelismRecommendation.minGPUs.toLocaleString()}
                </div>
              </div>
              <div>
                <div
                  className="text-[10px] font-semibold uppercase tracking-[0.08em]"
                  style={{ color: colors.textSecondary }}
                >
                  Pipeline Bubble
                </div>
                <div className="mt-1 text-sm font-semibold" style={{ color: colors.text }}>
                  {formatPercent(autoParallelismRecommendation.pipelineBubbleFraction, 1)}
                </div>
              </div>
            </div>
          </div>
        )}

        {config.parallelismMode === "manual" && (
          <div className="mt-3 space-y-3">
            {/* Framework first — determines ZeRO vs FSDP */}
            <SelectInput
              label="Framework"
              value={config.parallelism.framework}
              onChange={(v) =>
                setPar({ framework: v as FrameworkType })
              }
              options={[
                { value: "deepspeed", label: "DeepSpeed" },
                { value: "megatron", label: "Megatron-LM" },
                { value: "fsdp", label: "PyTorch FSDP" },
                { value: "hf_trainer", label: "HF Trainer" },
              ]}
              tooltip="Training framework — affects parallelism options and memory accounting"
              colors={colors}
            />

            <div className="grid gap-3 sm:grid-cols-2">
              <NumberInput
                label="Tensor parallel (N_tp)"
                value={config.parallelism.N_tp}
                onChange={(v) => setPar({ N_tp: v })}
                min={1}
                integer
                tooltip="Tensor parallelism degree — splits each layer across GPUs"
                colors={colors}
              />
              <NumberInput
                label="Pipeline parallel (N_pp)"
                value={config.parallelism.N_pp}
                onChange={(v) => setPar({ N_pp: v })}
                min={1}
                integer
                tooltip="Pipeline parallelism stages"
                colors={colors}
              />
              <NumberInput
                label="Data parallel (N_dp)"
                value={config.parallelism.N_dp}
                onChange={(v) => setPar({ N_dp: v })}
                min={1}
                integer
                tooltip="Data parallelism degree"
                colors={colors}
              />

              {/* ZeRO stage for DeepSpeed / HF Trainer */}
              {(config.parallelism.framework === "deepspeed" ||
                config.parallelism.framework === "hf_trainer") && (
                <SelectInput
                  label="ZeRO stage"
                  value={String(config.parallelism.zeroStage)}
                  onChange={(v) =>
                    setPar({ zeroStage: Number(v) as ZeROStage })
                  }
                  options={[
                    { value: "0", label: "Stage 0 (none)" },
                    { value: "1", label: "Stage 1 (optimizer)" },
                    { value: "2", label: "Stage 2 (+ gradients)" },
                    { value: "3", label: "Stage 3 (+ params)" },
                  ]}
                  tooltip="ZeRO Redundancy Optimizer stage"
                  colors={colors}
                />
              )}

              {/* FSDP strategy */}
              {config.parallelism.framework === "fsdp" && (
                <SelectInput
                  label="FSDP strategy"
                  value={config.parallelism.fsdpStrategy || "FULL_SHARD"}
                  onChange={(v) =>
                    setPar({ fsdpStrategy: v as FSDPStrategy })
                  }
                  options={[
                    { value: "NO_SHARD", label: "No Shard" },
                    { value: "SHARD_GRAD_OP", label: "Shard Grad+Op" },
                    { value: "FULL_SHARD", label: "Full Shard" },
                    { value: "HYBRID_SHARD", label: "Hybrid Shard" },
                    {
                      value: "HYBRID_SHARD_ZERO2",
                      label: "Hybrid Shard (ZeRO-2)",
                    },
                  ]}
                  colors={colors}
                />
              )}
            </div>
          </div>
        )}
      </Section>

      {/* ——— 16–32. Advanced ——— */}
      <CollapsibleSection
        title="Advanced Settings"
        badge="17 options"
        colors={colors}
      >
        <div className="space-y-5">
          {/* Parallelism fine-tuning (16, 17, 20, 22) */}
          <div>
            <SubLabel colors={colors}>Parallelism details</SubLabel>
            {config.parallelismMode === "auto" ? (
              <div
                className="rounded-xl border p-4"
                style={{
                  borderColor: colors.border,
                  backgroundColor: colors.bg,
                }}
              >
                <div className="grid gap-3 sm:grid-cols-2">
                  <div>
                    <div
                      className="text-[10px] font-semibold uppercase tracking-[0.08em]"
                      style={{ color: colors.textSecondary }}
                    >
                      Context Parallel
                    </div>
                    <div className="mt-1 text-sm font-semibold" style={{ color: colors.text }}>
                      {displayParallelism.N_cp}
                    </div>
                  </div>
                  <div>
                    <div
                      className="text-[10px] font-semibold uppercase tracking-[0.08em]"
                      style={{ color: colors.textSecondary }}
                    >
                      Expert Parallel
                    </div>
                    <div className="mt-1 text-sm font-semibold" style={{ color: colors.text }}>
                      {displayParallelism.N_ep}
                    </div>
                  </div>
                  <div>
                    <div
                      className="text-[10px] font-semibold uppercase tracking-[0.08em]"
                      style={{ color: colors.textSecondary }}
                    >
                      Virtual Pipeline
                    </div>
                    <div className="mt-1 text-sm font-semibold" style={{ color: colors.text }}>
                      {displayParallelism.VP}
                    </div>
                  </div>
                  <div>
                    <div
                      className="text-[10px] font-semibold uppercase tracking-[0.08em]"
                      style={{ color: colors.textSecondary }}
                    >
                      Sequence Parallel
                    </div>
                    <div className="mt-1 text-sm font-semibold" style={{ color: colors.text }}>
                      {displayParallelism.sequenceParallelism}
                    </div>
                  </div>
                </div>
                <p
                  className="mt-3 text-[11px] leading-6"
                  style={{ color: colors.textSecondary }}
                >
                  These values are computed live from the current model, sequence length, GPU type, and GPU count.
                </p>
              </div>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2">
                {/* 16 */}
                <NumberInput
                  label="Context parallel (N_cp)"
                  value={config.parallelism.N_cp}
                  onChange={(v) => setPar({ N_cp: v })}
                  min={1}
                  integer
                  tooltip="Context parallelism — splits long sequences across GPUs"
                  colors={colors}
                />
                {/* 17 */}
                <NumberInput
                  label="Expert parallel (N_ep)"
                  value={config.parallelism.N_ep}
                  onChange={(v) => setPar({ N_ep: v })}
                  min={1}
                  disabled={!moeEnabled}
                  integer
                  tooltip="Expert parallelism for MoE models"
                  colors={colors}
                />
                {/* 20 */}
                <NumberInput
                  label="Virtual pipeline chunks (VP)"
                  value={config.parallelism.VP}
                  onChange={(v) => setPar({ VP: v })}
                  min={1}
                  integer
                  tooltip="Interleaved pipeline schedule chunks — reduces pipeline bubble"
                  colors={colors}
                />
                {/* 22 */}
                <SelectInput
                  label="Sequence parallelism"
                  value={config.parallelism.sequenceParallelism}
                  onChange={(v) =>
                    setPar({
                      sequenceParallelism:
                        v as SequenceParallelismMode,
                    })
                  }
                  options={[
                    { value: "auto", label: "Auto (on when TP > 1)" },
                    { value: "enabled", label: "Enabled" },
                    { value: "disabled", label: "Disabled" },
                  ]}
                  tooltip="Sequence parallelism — reduces activation memory when TP > 1"
                  colors={colors}
                />
              </div>
            )}
          </div>

          {moeEnabled && (
            <div>
              <SubLabel colors={colors}>MoE routing</SubLabel>
              <div className="grid gap-3 sm:grid-cols-2">
                <NumberInput
                  label="Total experts (E)"
                  value={config.model.moe.E}
                  onChange={(v) =>
                    setModel({
                      ...config.model,
                      moe: { ...config.model.moe, E: v },
                    })
                  }
                  min={1}
                  integer
                  colors={colors}
                />
                <NumberInput
                  label="Active experts (topk)"
                  value={config.model.moe.topk}
                  onChange={(v) =>
                    setModel({
                      ...config.model,
                      moe: { ...config.model.moe, topk: v },
                    })
                  }
                  min={1}
                  max={Math.max(config.model.moe.E, 1)}
                  integer
                  colors={colors}
                />
                <NumberInput
                  label="MoE layers (L_moe)"
                  value={config.model.moe.L_moe}
                  onChange={(v) =>
                    setModel({
                      ...config.model,
                      moe: { ...config.model.moe, L_moe: v },
                    })
                  }
                  min={1}
                  max={config.model.architecture.L}
                  integer
                  colors={colors}
                />
                <NumberInput
                  label="Shared experts (E_s)"
                  value={config.model.moe.E_s}
                  onChange={(v) =>
                    setModel({
                      ...config.model,
                      moe: { ...config.model.moe, E_s: v },
                    })
                  }
                  min={0}
                  integer
                  colors={colors}
                />
                <NumberInput
                  label="Load-balance factor"
                  value={config.model.moe.loadBalanceFactor}
                  onChange={(v) =>
                    setModel({
                      ...config.model,
                      moe: {
                        ...config.model.moe,
                        loadBalanceFactor: v,
                      },
                    })
                  }
                  min={1}
                  max={2}
                  step={0.05}
                  tooltip="Multiplier for routing imbalance overhead."
                  colors={colors}
                />
                <NumberInput
                  label="Dense FFN size"
                  value={
                    config.model.moe.denseIntermediateSize ??
                    config.model.architecture.d_ff ??
                    4 * config.model.architecture.d
                  }
                  onChange={(v) =>
                    setModel({
                      ...config.model,
                      moe: {
                      ...config.model.moe,
                      denseIntermediateSize: v,
                    },
                  })
                }
                min={1}
                integer
                tooltip="Intermediate size for dense FFN layers."
                colors={colors}
              />
                <NumberInput
                  label="Expert FFN size"
                  value={
                    config.model.moe.expertIntermediateSize ??
                    config.model.architecture.d_ff ??
                    4 * config.model.architecture.d
                  }
                  onChange={(v) =>
                    setModel({
                      ...config.model,
                      moe: {
                      ...config.model.moe,
                      expertIntermediateSize: v,
                    },
                  })
                }
                min={1}
                integer
                tooltip="Intermediate size used by each expert block."
                colors={colors}
              />
              </div>
            </div>
          )}

          {/* Framework & communication (21, 23, 24, 25, 26) */}
          <div>
            <SubLabel colors={colors}>
              Framework &amp; Communication
            </SubLabel>
            <div className="space-y-3">
              <div className="grid gap-3 sm:grid-cols-2">
                {/* 21 — also shown in parallelism section when manual */}
                {config.parallelismMode === "auto" && (
                  <SelectInput
                    label="Framework"
                    value={config.parallelism.framework}
                    onChange={(v) =>
                      setPar({ framework: v as FrameworkType })
                    }
                    options={[
                      { value: "deepspeed", label: "DeepSpeed" },
                      { value: "megatron", label: "Megatron-LM" },
                      { value: "fsdp", label: "PyTorch FSDP" },
                      {
                        value: "hf_trainer",
                        label: "HF Trainer",
                      },
                    ]}
                    colors={colors}
                  />
                )}
                {/* 24 */}
                <SelectInput
                  label="CPU offloading"
                  value={config.cpuOffload}
                  onChange={(v) =>
                    set({ cpuOffload: v as CPUOffloadMode })
                  }
                  options={[
                    { value: "none", label: "None" },
                    {
                      value: "optimizer-only",
                      label: "Optimizer only",
                    },
                    {
                      value: "optimizer-and-params",
                      label: "Optimizer + params",
                    },
                  ]}
                  tooltip="Offload model state to CPU RAM. Optimizer offload is broadly supported; parameter offload requires ZeRO-3 / FSDP FULL_SHARD."
                  colors={colors}
                />
                {/* 26 */}
                <SelectInput
                  label="Inter-node bandwidth assumption"
                  value={config.hardware.interNodeBandwidthPreset}
                  onChange={(v) => {
                    const preset =
                      v as InterNodeBandwidthPreset
                    const bw =
                      preset === "hdr-200"
                        ? 25
                        : preset === "ndr-400"
                          ? 50
                          : config.hardware
                              .interNodeBandwidthGBps
                    setHw({
                      interNodeBandwidthPreset: preset,
                      interNodeBandwidthGBps: bw,
                    })
                  }}
                  options={[
                    {
                      value: "hdr-200",
                      label: "HDR InfiniBand (200 Gb/s, ~25 GB/s)",
                    },
                    {
                      value: "ndr-400",
                      label: "NDR InfiniBand (400 Gb/s, ~50 GB/s)",
                    },
                    { value: "custom", label: "Custom" },
                  ]}
                  tooltip="Stored as GB/s after converting InfiniBand link rates from Gb/s. The default training-time estimate uses MFU as the all-in efficiency factor, so this is not stacked as a separate runtime multiplier."
                  colors={colors}
                />
                {config.hardware.interNodeBandwidthPreset ===
                  "custom" && (
                  <NumberInput
                    label="Custom bandwidth (GB/s)"
                    value={
                      config.hardware.interNodeBandwidthGBps
                    }
                    onChange={(v) =>
                      setHw({ interNodeBandwidthGBps: v })
                    }
                    min={1}
                    tooltip="Enter one-way effective inter-node bandwidth in GB/s for communication diagnostics and assumptions."
                    colors={colors}
                  />
                )}
              </div>

              {/* 23 */}
              <ToggleInput
                label="AMP autocast"
                value={config.ampAutocast}
                onChange={(v) => set({ ampAutocast: v })}
                tooltip="PyTorch AMP autocast — off by default, using explicit bf16 mode"
                colors={colors}
              />

              {/* 25 — ZeRO communication bucket config */}
              <SelectInput
                label="ZeRO communication buckets"
                value={config.zeroCommunication.mode}
                onChange={(v) =>
                  setZero({
                    mode: v as ZeROCommunicationBucketMode,
                  })
                }
                options={[
                  { value: "hf-auto", label: "HF auto" },
                  {
                    value: "deepspeed-defaults",
                    label: "DeepSpeed defaults",
                  },
                  {
                    value: "custom",
                    label: "Custom bucket sizes",
                  },
                ]}
                tooltip="Controls allgather, reduce, and ZeRO-3 prefetch bucket sizing"
                colors={colors}
              />
              <ToggleInput
                label="Overlap communication"
                value={effectiveOverlapComm}
                onChange={(v) => setZero({ overlapComm: v })}
                tooltip={
                  zero3ForcesOverlapComm
                    ? "DeepSpeed-style ZeRO-3 defaults overlap communication on; estimates include the overlap buffer cost."
                    : "Overlap gradient communication with backward pass"
                }
                colors={colors}
                disabled={zero3ForcesOverlapComm}
              />
              {config.zeroCommunication.mode === "custom" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <NumberInput
                    label="Allgather bucket (elements)"
                    value={
                      config.zeroCommunication
                        .allgatherBucketSizeElements ?? 0
                    }
                    onChange={(v) =>
                      setZero({
                        allgatherBucketSizeElements: v,
                      })
                    }
                    min={0}
                    integer
                    colors={colors}
                  />
                  <NumberInput
                    label="Reduce bucket (elements)"
                    value={
                      config.zeroCommunication
                        .reduceBucketSizeElements ?? 0
                    }
                    onChange={(v) =>
                      setZero({
                        reduceBucketSizeElements: v,
                      })
                    }
                    min={0}
                    integer
                    colors={colors}
                  />
                  <NumberInput
                    label="Prefetch bucket (elements)"
                    value={
                      config.zeroCommunication
                        .prefetchBucketSizeElements ?? 0
                    }
                    onChange={(v) =>
                      setZero({
                        prefetchBucketSizeElements: v,
                      })
                    }
                    min={0}
                    integer
                    colors={colors}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Memory optimizations (19, 27, 28, 29, 30) */}
          <div>
            <SubLabel colors={colors}>Memory optimizations</SubLabel>
            <div className="space-y-3">
              <div className="grid gap-3 sm:grid-cols-2">
                {/* 19 — only when checkpointing = partial */}
                {config.activationCheckpointing === "partial" && (
                  <NumberInput
                    label="Checkpointed layers/stage"
                    value={config.partialCheckpointDepth ?? 1}
                    onChange={(v) =>
                      set({ partialCheckpointDepth: v })
                    }
                    min={1}
                    max={maxCheckpointedLayersPerStage}
                    integer
                    tooltip="Number of layers per pipeline stage to fully checkpoint and recompute (N_recomp)"
                    colors={colors}
                  />
                )}
                {/* 30 */}
                <SelectInput
                  label="KV cache precision"
                  value={config.kvCachePrecision}
                  onChange={(v) =>
                    set({
                      kvCachePrecision:
                        v as TrainingConfig["kvCachePrecision"],
                    })
                  }
                  options={[
                    { value: "bf16", label: "BF16" },
                    { value: "fp16", label: "FP16" },
                    { value: "int8", label: "INT8" },
                  ]}
                  tooltip="Precision for KV cache in post-training generation phases"
                  colors={colors}
                />
              </div>

              {/* 27 */}
              <ToggleInput
                label="torch.compile"
                value={config.torchCompile}
                onChange={(v) => set({ torchCompile: v })}
                tooltip="Enables torch.compile — adds ~10% model-weights overhead"
                colors={colors}
              />
              {/* 28 */}
              <ToggleInput
                label="Chunked cross-entropy"
                value={config.chunkedCrossEntropy}
                onChange={(v) => set({ chunkedCrossEntropy: v })}
                tooltip="Eliminates output logits tensor from activation memory"
                colors={colors}
              />

              {/* 29 — FP8 options, shown when precision = fp8 */}
              {config.precision === "fp8" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <NumberInput
                    label="FP8 kernel speedup"
                    value={config.fp8.kernelSpeedupFactor}
                    onChange={(v) =>
                      set({
                        fp8: {
                          ...config.fp8,
                          kernelSpeedupFactor: v,
                        },
                      })
                    }
                    min={1.0}
                    max={2.0}
                    step={0.05}
                    tooltip="Effective compute speedup from FP8 kernels (default 1.3x)"
                    colors={colors}
                  />
                  <SelectInput
                    label="FP8 storage mode"
                    value={config.fp8.storageMode}
                    onChange={(v) =>
                      set({
                        fp8: {
                          ...config.fp8,
                          storageMode:
                            v as TrainingConfig["fp8"]["storageMode"],
                        },
                      })
                    }
                    options={[
                      {
                        value: "transformer-engine",
                        label: "TransformerEngine",
                      },
                      { value: "ms-amp", label: "MS-AMP" },
                    ]}
                    colors={colors}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Checkpoint & failure (31, 32) */}
          <div>
            <SubLabel colors={colors}>Checkpoint &amp; Failure</SubLabel>
            <div className="grid gap-3 sm:grid-cols-2">
              {/* 31 */}
              <NumberInput
                label="Checkpoint retention count"
                value={config.pricing.checkpointRetentionCount}
                onChange={(v) =>
                  setPrice({ checkpointRetentionCount: v })
                }
                min={1}
                integer
                tooltip="Number of checkpoints kept — caps peak storage"
                colors={colors}
              />
              <NumberInput
                label="Storage price"
                value={config.pricing.storagePricePerGBMonth}
                onChange={(v) =>
                  setPrice({ storagePricePerGBMonth: v })
                }
                min={0}
                step={0.001}
                unit="$/GB/mo"
                colors={colors}
              />
              <NumberInput
                label="Dataset storage"
                value={config.pricing.datasetStorageGB}
                onChange={(v) => setPrice({ datasetStorageGB: v })}
                min={0}
                step={100}
                unit="GB"
                tooltip="Static dataset or object-store footprint included in storage cost"
                colors={colors}
              />
            </div>

            {/* 32 — Failure model, shown when GPUs >= 256 */}
            {effectiveNumGPUs >= 256 && (
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <NumberInput
                  label="Failure rate"
                  value={
                    config.failureModel
                      .failureRatePerInstancePerDay
                  }
                  onChange={(v) =>
                    set({
                      failureModel: {
                        ...config.failureModel,
                        failureRatePerInstancePerDay: v,
                      },
                    })
                  }
                  min={0}
                  step={0.001}
                  unit="/inst/day"
                  tooltip="Expected failure rate per instance per day"
                  colors={colors}
                />
                <NumberInput
                  label="Recovery time"
                  value={config.failureModel.recoveryTimeHours}
                  onChange={(v) =>
                    set({
                      failureModel: {
                        ...config.failureModel,
                        recoveryTimeHours: v,
                      },
                    })
                  }
                  min={0}
                  step={0.25}
                  unit="hours"
                  colors={colors}
                />
                <NumberInput
                  label="Checkpoint freq"
                  value={
                    config.failureModel
                      .checkpointFrequencyPerDay
                  }
                  onChange={(v) =>
                    set({
                      failureModel: {
                        ...config.failureModel,
                        checkpointFrequencyPerDay: v,
                      },
                    })
                  }
                  min={1}
                  unit="/day"
                  colors={colors}
                />
              </div>
            )}
          </div>
        </div>
      </CollapsibleSection>
    </div>
  )
}
