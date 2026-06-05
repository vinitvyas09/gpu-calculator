"use client"

import { useMemo, type ReactNode } from "react"
import type {
  CheckpointingMode,
  CPUOffloadMode,
  FSDPStrategy,
  FrameworkType,
  GradientPrecision,
  HardwareSelection,
  InterNodeBandwidthMode,
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
  CLOUD_INSTANCES,
  CLOUD_PRICING_PRESETS,
  GPU_SPECS,
  INTER_NODE_BANDWIDTH_PRESETS,
  OPTIMIZER_PROFILES,
} from "../constants"
import { getCloudInstanceGPUHourlyRate } from "../formulas/pricing"
import {
  getEffectiveDefaultTrainingMFU,
  MAX_MFU_OVERRIDE,
} from "../formulas/cost"
import {
  resolveDefaultFFNIntermediateSize,
  resolveDefaultMoEExpertIntermediateSize,
} from "../formulas/compute"
import {
  type CalculatorColors,
  NumberInput,
  SelectInput,
  SliderInput,
  ToggleInput,
  formatPercent,
} from "./input-controls"
import {
  ModelArchitectureFields,
  ModelSelector,
  MoEOverviewNote,
  getModelPresetDefaultSequenceLength,
} from "./model-selector"
import { GPUSelector } from "./gpu-selector"
import { Layer, type LayerHostProps } from "./layer"
import { LayerStack } from "./layer-stack"

// ---------------------------------------------------------------------------
// EssentialsGroup — a calm, labelled cluster for the always-visible strip.
//
// Replaces the old Section icon-rule chrome (Phase 3) with a quieter group:
// a small uppercase eyebrow over the controls, no leading icon or full-width
// rule. The pickers' own internal cards still carry the visual identity.
// ---------------------------------------------------------------------------
function EssentialsGroup({
  label,
  colors,
  className,
  children,
}: {
  label: string
  colors: CalculatorColors
  className?: string
  children: ReactNode
}) {
  return (
    <div className={className}>
      <div
        className="mb-2.5 text-[10px] font-semibold uppercase tracking-[0.12em]"
        style={{ color: colors.textSecondary }}
      >
        {label}
      </div>
      {children}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Subsection label inside a layer body
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
// Shared props for the pretraining panel surface
// ---------------------------------------------------------------------------
interface PretrainingCommonProps {
  config: TrainingConfig
  onChange: (c: TrainingConfig) => void
  colors: CalculatorColors
  activeParameterCount: number
  effectiveNumGPUs: number
  gpuCountDerivedFromTarget: boolean
}

// ---------------------------------------------------------------------------
// usePretrainingWiring — all set*/derivation closures, moved VERBATIM.
//
// These closures encode coupling semantics (price-preset sync,
// FSDP-derives-ZeRO, fp8 fallback) and must not change behavior. Both
// PretrainingEssentials and PretrainingLayers consume this hook.
// ---------------------------------------------------------------------------
function usePretrainingWiring({
  config,
  onChange,
  activeParameterCount,
  effectiveNumGPUs,
  autoParallelismRecommendation,
}: {
  config: TrainingConfig
  onChange: (c: TrainingConfig) => void
  activeParameterCount: number
  effectiveNumGPUs: number
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
    let nextPricing = config.pricing
    const nextConfig: TrainingConfig = {
      ...config,
      hardware: { ...config.hardware, ...hardware },
    }

    if (config.pricing.cloudInstanceId !== null) {
      nextPricing = {
        ...nextPricing,
        cloudInstanceId: null,
      }
    }

    if (config.pricing.cloudPricingPresetId !== null) {
      const matchingPricePreset =
        hardware.inputMode === "preset"
          ? CLOUD_PRICING_PRESETS.find(
              (preset) => preset.gpuId === (hardware.gpuId ?? hardware.gpu.id),
            )
          : undefined

      nextPricing = matchingPricePreset
        ? {
            ...nextPricing,
            cloudPricingPresetId: matchingPricePreset.id,
            costPerGPUHour: matchingPricePreset.priceDefault,
          }
        : {
            ...nextPricing,
            cloudPricingPresetId: null,
          }
    }

    if (nextPricing !== config.pricing) {
      nextConfig.pricing = nextPricing
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

  const setCloudInstance = (instanceId: string) => {
    if (instanceId === "none") {
      setPrice({ cloudInstanceId: null })
      return
    }

    const instance = CLOUD_INSTANCES.find((item) => item.id === instanceId)
    const gpu = instance
      ? GPU_SPECS.find((item) => item.id === instance.gpuId)
      : undefined

    if (!instance || !gpu) {
      setPrice({ cloudInstanceId: null })
      return
    }

    onChange({
      ...config,
      hardware: {
        ...config.hardware,
        inputMode: "preset",
        gpuId: gpu.id,
        gpu,
      },
      pricing: {
        ...config.pricing,
        cloudPricingPresetId: null,
        cloudInstanceId: instance.id,
        costPerGPUHour: getCloudInstanceGPUHourlyRate(instance),
      },
    })
  }

  const setZero = (patch: Partial<ZeROCommunicationConfig>) =>
    onChange({
      ...config,
      zeroCommunication: { ...config.zeroCommunication, ...patch },
    })

  const setInterNodeBandwidthMode = (mode: InterNodeBandwidthMode) => {
    const preset = INTER_NODE_BANDWIDTH_PRESETS.find((item) => item.id === mode)

    onChange({
      ...config,
      interNodeBandwidth: {
        mode,
        customGBps:
          preset?.bandwidthGBps ??
          config.interNodeBandwidth.customGBps ??
          50,
      },
    })
  }

  const setInterNodeCustomBandwidth = (customGBps: number) =>
    onChange({
      ...config,
      interNodeBandwidth: {
        mode: "custom",
        customGBps,
      },
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
  const effectiveOptimizerId =
    config.optimizer === "adamw-fp8" &&
    (config.precision !== "fp8" ||
      !config.hardware.gpu.supportsFP8 ||
      config.fp8.storageMode === "transformer-engine")
      ? "adamw-mixed"
      : config.optimizer
  const selectedOptimizerProfile = OPTIMIZER_PROFILES.find(
    (optimizer) => optimizer.id === effectiveOptimizerId,
  )
  const optimizerFixesGradientStorage =
    selectedOptimizerProfile?.fixedGradientStorage ?? false
  const gradientPrecisionValue = optimizerFixesGradientStorage
    ? "fixed"
    : config.gradientPrecision
  const gradientPrecisionOptions = optimizerFixesGradientStorage
    ? [
        {
          value: "fixed",
          label:
            selectedOptimizerProfile?.id === "adamw-fp8"
              ? "FP8 gradients (fixed)"
              : "FP32 gradients (fixed)",
        },
      ]
    : [
        { value: "fp32", label: "FP32 (default)" },
        { value: "bf16", label: "BF16" },
      ]
  const gradientPrecisionTooltip = optimizerFixesGradientStorage
    ? `${selectedOptimizerProfile?.name ?? "Selected optimizer"} fixes gradient storage internally, so this setting does not change memory.`
    : "Precision for gradient accumulation — affects memory footprint"

  const cloudPresetOptions = [
    ...CLOUD_PRICING_PRESETS.map((p) => ({
      value: p.id,
      label: `${p.label} ($${p.priceLow}-${p.priceHigh}/hr)`,
    })),
    { value: "custom", label: "Custom price" },
  ]
  const cloudInstanceOptions = [
    { value: "none", label: "None" },
    ...CLOUD_INSTANCES.map((instance) => {
      const gpu = GPU_SPECS.find((item) => item.id === instance.gpuId)
      const perGpuHour = getCloudInstanceGPUHourlyRate(instance)

      return {
        value: instance.id,
        label: `${instance.instanceType} (${instance.gpuCount}x ${
          gpu?.name ?? instance.gpuId
        }, $${instance.pricePerHour.toFixed(2)}/hr, $${perGpuHour.toFixed(
          2,
        )}/GPU-hr)`,
        group: instance.provider,
      }
    }),
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

  return {
    set,
    setModel,
    setHw,
    setHardwareSelection,
    setPar,
    setPrice,
    setCloudInstance,
    setZero,
    setInterNodeBandwidthMode,
    setInterNodeCustomBandwidth,
    setTotalTokens,
    optimizerOptions,
    effectiveOptimizerId,
    optimizerFixesGradientStorage,
    gradientPrecisionValue,
    gradientPrecisionOptions,
    gradientPrecisionTooltip,
    cloudPresetOptions,
    cloudInstanceOptions,
    isQuickMode,
    moeEnabled,
    displayParallelism,
    defaultMFU,
    hasMFUOverride,
    autoLayoutParts,
    zero3ForcesOverlapComm,
    effectiveOverlapComm,
    maxCheckpointedLayersPerStage,
  }
}

// ---------------------------------------------------------------------------
// PretrainingEssentials — the always-visible block.
//
// Keeps today's Section chrome (Phase 4 replaces it): Model (whole
// ModelSelector), Training Data (Total tokens + quick-mode hint only), and
// Hardware (whole GPUSelector + #GPUs + target days + $/GPU-hr).
// ---------------------------------------------------------------------------
export function PretrainingEssentials({
  config,
  onChange,
  colors,
  activeParameterCount,
  effectiveNumGPUs,
  gpuCountDerivedFromTarget,
  autoParallelismRecommendation,
}: PretrainingCommonProps & {
  autoParallelismRecommendation: ParallelismRecommendation
}) {
  const {
    setModel,
    setHw,
    setHardwareSelection,
    setPrice,
    setTotalTokens,
    isQuickMode,
    displayParallelism,
    defaultMFU,
  } = usePretrainingWiring({
    config,
    onChange,
    activeParameterCount,
    effectiveNumGPUs,
    autoParallelismRecommendation,
  })
  // defaultMFU is consumed in the layers; reference here keeps the hook's
  // memo dependency identical regardless of which surface renders it.
  void defaultMFU

  return (
    <div className="space-y-6">
      {/* ——— Model (full-width; segmented Quick/Preset/Detailed + spec card) ——— */}
      <EssentialsGroup label="Model" colors={colors}>
        <ModelSelector
          selection={config.model}
          onChange={setModel}
          quickTokens={config.totalTokens}
          onQuickTokensChange={setTotalTokens}
          colors={colors}
        />
      </EssentialsGroup>

      {/* ——— Data + hardware strip ——— */}
      <div className="grid gap-x-6 gap-y-6 lg:grid-cols-2">
        {/* Left column: training tokens + cost */}
        <div className="space-y-4">
          <EssentialsGroup label="Training data" colors={colors}>
            {/* P21 — hidden in Quick mode (edited via the Quick model tab) */}
            {!isQuickMode ? (
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
            ) : (
              /* R9 */
              <p className="text-xs leading-6" style={{ color: colors.textSecondary }}>
                Total tokens are edited from the Quick model tab so the model-size
                and dataset-size inputs stay grouped in the fast-estimate workflow.
              </p>
            )}
          </EssentialsGroup>

          {/* P39 — Cost per GPU-hour */}
          <EssentialsGroup label="Cost" colors={colors}>
            <NumberInput
              label="Cost per GPU-hour"
              value={config.pricing.costPerGPUHour}
              onChange={(v) =>
                setPrice({
                  costPerGPUHour: v,
                  cloudPricingPresetId: null,
                  cloudInstanceId: null,
                })
              }
              min={0}
              step={0.1}
              unit="$/hr"
              colors={colors}
            />
          </EssentialsGroup>
        </div>

        {/* Right column: GPU picker + count/target (coupled) */}
        <EssentialsGroup label="Hardware" colors={colors}>
          {/* P31 / P32 */}
          <GPUSelector
            gpuId={config.hardware.gpuId}
            gpu={config.hardware.gpu}
            inputMode={config.hardware.inputMode}
            onChange={setHardwareSelection}
            colors={colors}
            tpDegree={displayParallelism.N_tp}
            precision={config.precision}
          />

          {/* P33 + P34 — coupled: target days derives/locks #GPUs */}
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
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
          {/* R10 */}
          {gpuCountDerivedFromTarget && (
            <p
              className="mt-3 text-xs leading-6"
              style={{ color: colors.textSecondary }}
            >
              GPU count is resolved from the target training-time constraint plus
              current memory, topology, MFU, and schedule assumptions.
            </p>
          )}
        </EssentialsGroup>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// PretrainingLayers — the six summary-line layers.
// ---------------------------------------------------------------------------
export function PretrainingLayers({
  config,
  onChange,
  colors,
  activeParameterCount,
  effectiveNumGPUs,
  autoParallelismRecommendation,
  host,
}: PretrainingCommonProps & {
  autoParallelismRecommendation: ParallelismRecommendation
  host: LayerHostProps
}) {
  const {
    set,
    setModel,
    setPar,
    setPrice,
    setCloudInstance,
    setZero,
    setInterNodeBandwidthMode,
    setInterNodeCustomBandwidth,
    optimizerOptions,
    optimizerFixesGradientStorage,
    gradientPrecisionValue,
    gradientPrecisionOptions,
    gradientPrecisionTooltip,
    cloudPresetOptions,
    cloudInstanceOptions,
    moeEnabled,
    displayParallelism,
    defaultMFU,
    hasMFUOverride,
    autoLayoutParts,
    zero3ForcesOverlapComm,
    effectiveOverlapComm,
    maxCheckpointedLayersPerStage,
  } = usePretrainingWiring({
    config,
    onChange,
    activeParameterCount,
    effectiveNumGPUs,
    autoParallelismRecommendation,
  })

  const isManual = config.parallelismMode === "manual"

  // One Framework control, rendered once, visible in both modes (P41 / P58).
  const frameworkControl = (
    <SelectInput
      label="Framework"
      value={config.parallelism.framework}
      onChange={(v) => setPar({ framework: v as FrameworkType })}
      options={[
        { value: "deepspeed", label: "DeepSpeed" },
        { value: "megatron", label: "Megatron-LM" },
        { value: "fsdp", label: "PyTorch FSDP" },
        { value: "hf_trainer", label: "HF Trainer" },
      ]}
      tooltip="Training framework — affects parallelism options and memory accounting"
      colors={colors}
    />
  )

  return (
    <LayerStack
      colors={colors}
      expandAll={host.expandAll}
      density={host.density}
    >
      {/* ——— Layer 3 · Parallelism ——— */}
      <Layer
        id="parallelism"
        title="Parallelism"
        colors={colors}
        open={host.isLayerOpen("parallelism")}
        onOpenChange={(o) => host.onLayerOpenChange("parallelism", o)}
        summary={host.summaries.parallelism}
        warningCount={host.warningChips.parallelism?.count}
        warningSeverity={host.warningChips.parallelism?.severity}
      >
        <div className="space-y-3">
          {/* P40 */}
          <SelectInput
            label="Mode"
            value={config.parallelismMode}
            onChange={(v) => set({ parallelismMode: v as ParallelismMode })}
            options={[
              { value: "auto", label: "Auto-recommend" },
              { value: "manual", label: "Manual configuration" },
            ]}
            tooltip="Auto uses the parallelism recommendation engine; manual lets you set each dimension"
            colors={colors}
          />

          {/* R5 — auto-mode live recommendation card */}
          {!isManual && (
            <div
              className="space-y-3 rounded-xl border p-4"
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

          {/* Framework — single control, both modes (P41 / P58) */}
          {frameworkControl}

          {/* Manual mesh (P42-P46) or auto read-only mesh (R6) */}
          {isManual ? (
            <div className="grid gap-3 sm:grid-cols-2">
              {/* P42 */}
              <NumberInput
                label="Tensor parallel (N_tp)"
                value={config.parallelism.N_tp}
                onChange={(v) => setPar({ N_tp: v })}
                min={1}
                integer
                tooltip="Tensor parallelism degree — splits each layer across GPUs"
                colors={colors}
              />
              {/* P43 */}
              <NumberInput
                label="Pipeline parallel (N_pp)"
                value={config.parallelism.N_pp}
                onChange={(v) => setPar({ N_pp: v })}
                min={1}
                integer
                tooltip="Pipeline parallelism stages"
                colors={colors}
              />
              {/* P44 */}
              <NumberInput
                label="Data parallel (N_dp)"
                value={config.parallelism.N_dp}
                onChange={(v) => setPar({ N_dp: v })}
                min={1}
                integer
                tooltip="Data parallelism degree"
                colors={colors}
              />

              {/* P45 — ZeRO stage for DeepSpeed / HF Trainer */}
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

              {/* P46 — FSDP strategy */}
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

              {/* P47 */}
              <NumberInput
                label="Context parallel (N_cp)"
                value={config.parallelism.N_cp}
                onChange={(v) => setPar({ N_cp: v })}
                min={1}
                integer
                tooltip="Context parallelism — splits long sequences across GPUs"
                colors={colors}
              />
              {/* P49 */}
              <NumberInput
                label="Virtual pipeline chunks (VP)"
                value={config.parallelism.VP}
                onChange={(v) => setPar({ VP: v })}
                min={1}
                integer
                tooltip="Interleaved pipeline schedule chunks — reduces pipeline bubble"
                colors={colors}
              />
              {/* P50 */}
              <SelectInput
                label="Sequence parallelism"
                value={config.parallelism.sequenceParallelism}
                onChange={(v) =>
                  setPar({
                    sequenceParallelism: v as SequenceParallelismMode,
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
          ) : (
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
          )}

          {/* P62 — Overlap communication */}
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

          {/* P66 / P67 — Inter-node bandwidth */}
          <div className="grid gap-3 sm:grid-cols-2">
            <SelectInput
              label="Inter-node bandwidth"
              value={config.interNodeBandwidth.mode}
              onChange={(v) =>
                setInterNodeBandwidthMode(v as InterNodeBandwidthMode)
              }
              options={[
                ...INTER_NODE_BANDWIDTH_PRESETS.map((preset) => ({
                  value: preset.id,
                  label: preset.label,
                })),
                { value: "custom", label: "Custom GB/s" },
              ]}
              tooltip="Per-GPU cross-node bandwidth assumption for communication diagnostics. This is not stacked on top of MFU by default."
              colors={colors}
            />
            {config.interNodeBandwidth.mode === "custom" && (
              <NumberInput
                label="Custom bandwidth"
                value={config.interNodeBandwidth.customGBps ?? 50}
                onChange={setInterNodeCustomBandwidth}
                min={0.1}
                step={1}
                unit="GB/s"
                tooltip="Sustained per-GPU bandwidth after protocol overhead"
                colors={colors}
              />
            )}
          </div>
        </div>

        {host.outputSlots.parallelism}
        {host.warningSlots.parallelism}
      </Layer>

      {/* ——— Layer 4 · Model architecture ——— */}
      <Layer
        id="architecture"
        title="Model architecture"
        colors={colors}
        open={host.isLayerOpen("architecture")}
        onOpenChange={(o) => host.onLayerOpenChange("architecture", o)}
        summary={host.summaries.architecture}
        warningCount={host.warningChips.architecture?.count}
        warningSeverity={host.warningChips.architecture?.severity}
      >
        <div className="space-y-3">
          {/* P5–P17 — detailed-mode architecture grid (relocated from the model
              picker; only editable in Detailed mode). MoE FFN-size fields live
              in the MoE layer below; this grid carries the MoE-enable toggle. */}
          {config.model.inputMode === "detailed" && (
            <ModelArchitectureFields
              selection={config.model}
              onChange={setModel}
              colors={colors}
            />
          )}
          <div className="grid gap-3 sm:grid-cols-2">
            {/* P27 */}
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
          </div>
          {/* P30 */}
          <ToggleInput
            label="Flash Attention"
            value={config.flashAttention}
            onChange={(v) => set({ flashAttention: v })}
            tooltip="Use FlashAttention for fused, memory-efficient attention"
            colors={colors}
          />
        </div>

        {host.outputSlots.architecture}
        {host.warningSlots.architecture}
      </Layer>

      {/* ——— Layer 5 · Precision & optimizer ——— */}
      <Layer
        id="precision"
        title="Precision & optimizer"
        colors={colors}
        open={host.isLayerOpen("precision")}
        onOpenChange={(o) => host.onLayerOpenChange("precision", o)}
        summary={host.summaries.precision}
        warningCount={host.warningChips.precision?.count}
        warningSeverity={host.warningChips.precision?.severity}
      >
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            {/* P23 */}
            <SelectInput
              label="Precision"
              value={config.precision}
              onChange={(v) => set({ precision: v as TrainingPrecision })}
              options={[
                { value: "bf16", label: "BF16" },
                { value: "fp16", label: "FP16" },
                { value: "fp32", label: "FP32" },
                { value: "fp8", label: "FP8" },
              ]}
              tooltip="Training precision — BF16 is standard for modern GPUs"
              colors={colors}
            />
            {/* P24 */}
            <SelectInput
              label="Optimizer"
              value={config.optimizer}
              onChange={(v) => set({ optimizer: v as OptimizerType })}
              options={optimizerOptions}
              colors={colors}
            />
            {/* P25 */}
            <SelectInput
              label="Gradient precision"
              value={gradientPrecisionValue}
              onChange={(v) =>
                set({ gradientPrecision: v as GradientPrecision })
              }
              options={gradientPrecisionOptions}
              tooltip={gradientPrecisionTooltip}
              disabled={optimizerFixesGradientStorage}
              colors={colors}
            />
            {/* P26 */}
            <NumberInput
              label="Micro-batch size (b)"
              value={config.microBatchSize}
              onChange={(v) => set({ microBatchSize: v })}
              min={1}
              integer
              tooltip="Per-GPU micro-batch size in sequences"
              colors={colors}
            />
            {/* P28 */}
            <NumberInput
              label="Grad accum steps (G)"
              value={config.gradientAccumulationSteps}
              onChange={(v) => set({ gradientAccumulationSteps: v })}
              min={1}
              integer
              tooltip="Gradient accumulation steps before weight update"
              colors={colors}
            />
            {/* P29 */}
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
            {/* P68 — only when checkpointing = partial */}
            {config.activationCheckpointing === "partial" && (
              <NumberInput
                label="Checkpointed layers/stage"
                value={config.partialCheckpointDepth ?? 1}
                onChange={(v) => set({ partialCheckpointDepth: v })}
                min={1}
                max={maxCheckpointedLayersPerStage}
                integer
                tooltip="Number of layers per pipeline stage to fully checkpoint and recompute (N_recomp)"
                colors={colors}
              />
            )}
            {/* P59 */}
            <SelectInput
              label="CPU offloading"
              value={config.cpuOffload}
              onChange={(v) => set({ cpuOffload: v as CPUOffloadMode })}
              options={[
                { value: "none", label: "None" },
                { value: "optimizer-only", label: "Optimizer only" },
                {
                  value: "optimizer-and-params",
                  label: "Optimizer + params",
                },
              ]}
              tooltip="Offload model state to CPU RAM. Optimizer offload is broadly supported; parameter offload requires ZeRO-3 / FSDP FULL_SHARD or HYBRID_SHARD."
              colors={colors}
            />
          </div>

          {/* P60 */}
          <ToggleInput
            label="AMP autocast"
            value={config.ampAutocast}
            onChange={(v) => set({ ampAutocast: v })}
            tooltip="PyTorch AMP autocast — off by default, using explicit bf16 mode"
            colors={colors}
          />
          {/* P69 */}
          <ToggleInput
            label="torch.compile"
            value={config.torchCompile}
            onChange={(v) => set({ torchCompile: v })}
            tooltip="Enables torch.compile — adds ~10% model-weights overhead"
            colors={colors}
          />
          {/* P70 */}
          <ToggleInput
            label="Chunked cross-entropy"
            value={config.chunkedCrossEntropy}
            onChange={(v) => set({ chunkedCrossEntropy: v })}
            tooltip="Eliminates materialized output logits and the fp32 logits-gradient peak from loss memory"
            colors={colors}
          />

          {/* P71 / P72 — FP8 options, shown when precision = fp8 */}
          {config.precision === "fp8" && (
            <div className="grid gap-3 sm:grid-cols-2">
              <NumberInput
                label="FP8 kernel speedup"
                value={config.fp8.kernelSpeedupFactor}
                onChange={(v) =>
                  set({
                    fp8: { ...config.fp8, kernelSpeedupFactor: v },
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
                      storageMode: v as TrainingConfig["fp8"]["storageMode"],
                    },
                  })
                }
                options={[
                  { value: "transformer-engine", label: "TransformerEngine" },
                  { value: "ms-amp", label: "MS-AMP" },
                ]}
                colors={colors}
              />
            </div>
          )}

          {/* P35 / P36 — MFU override */}
          <div>
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
        </div>

        {host.outputSlots.precision}
        {host.warningSlots.precision}
      </Layer>

      {/* ——— Layer 6 · Data & scaling ——— */}
      <Layer
        id="data"
        title="Data & scaling"
        colors={colors}
        open={host.isLayerOpen("data")}
        onOpenChange={(o) => host.onLayerOpenChange("data", o)}
        summary={host.summaries.data}
        warningCount={host.warningChips.data?.count}
        warningSeverity={host.warningChips.data?.severity}
      >
        <div className="grid gap-3 sm:grid-cols-2">
          {/* P22 */}
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

        {host.outputSlots.data}
        {host.warningSlots.data}
      </Layer>

      {/* ——— Layer 7 · Cost detail & failures ——— */}
      <Layer
        id="cost"
        title="Cost detail & failures"
        colors={colors}
        open={host.isLayerOpen("cost")}
        onOpenChange={(o) => host.onLayerOpenChange("cost", o)}
        summary={host.summaries.cost}
        warningCount={host.warningChips.cost?.count}
        warningSeverity={host.warningChips.cost?.severity}
      >
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            {/* P37 */}
            <SelectInput
              label="Cloud instance"
              value={config.pricing.cloudInstanceId || "none"}
              onChange={setCloudInstance}
              options={cloudInstanceOptions}
              tooltip="Optional instance-hour preset. Selecting one also switches the GPU preset, bills whole instances, and uses its GPU count for failure-rate estimates."
              colors={colors}
            />
            {/* P38 */}
            <SelectInput
              label="Cloud pricing preset"
              value={config.pricing.cloudPricingPresetId || "custom"}
              onChange={(v) => {
                const preset = CLOUD_PRICING_PRESETS.find((p) => p.id === v)
                if (preset) {
                  setPrice({
                    cloudPricingPresetId: v,
                    cloudInstanceId: null,
                    costPerGPUHour: preset.priceDefault,
                  })
                } else {
                  setPrice({
                    cloudPricingPresetId: null,
                    cloudInstanceId: null,
                  })
                }
              }}
              options={cloudPresetOptions}
              tooltip="Representative on-demand defaults; cloud prices change often, so override with your actual quote or committed-use rate."
              colors={colors}
            />
          </div>

          {/* P61 — ZeRO communication buckets */}
          <SelectInput
            label="ZeRO communication buckets"
            value={config.zeroCommunication.mode}
            onChange={(v) =>
              setZero({ mode: v as ZeROCommunicationBucketMode })
            }
            options={[
              { value: "hf-auto", label: "HF auto" },
              { value: "deepspeed-defaults", label: "DeepSpeed defaults" },
              { value: "custom", label: "Custom bucket sizes" },
            ]}
            tooltip="Controls allgather, reduce, and ZeRO-3 prefetch bucket sizing"
            colors={colors}
          />
          {/* P63 / P64 / P65 */}
          {config.zeroCommunication.mode === "custom" && (
            <div className="grid gap-3 sm:grid-cols-2">
              <NumberInput
                label="Allgather bucket (elements)"
                value={
                  config.zeroCommunication.allgatherBucketSizeElements ?? 0
                }
                onChange={(v) =>
                  setZero({ allgatherBucketSizeElements: v })
                }
                min={0}
                integer
                colors={colors}
              />
              <NumberInput
                label="Reduce bucket (elements)"
                value={config.zeroCommunication.reduceBucketSizeElements ?? 0}
                onChange={(v) => setZero({ reduceBucketSizeElements: v })}
                min={0}
                integer
                colors={colors}
              />
              <NumberInput
                label="Prefetch bucket (elements)"
                value={
                  config.zeroCommunication.prefetchBucketSizeElements ?? 0
                }
                onChange={(v) =>
                  setZero({ prefetchBucketSizeElements: v })
                }
                min={0}
                integer
                colors={colors}
              />
            </div>
          )}

          <div className="grid gap-3 sm:grid-cols-2">
            {/* P73 */}
            <NumberInput
              label="Checkpoint retention count"
              value={config.pricing.checkpointRetentionCount}
              onChange={(v) => setPrice({ checkpointRetentionCount: v })}
              min={0}
              integer
              tooltip="Number of checkpoints kept — caps peak storage; set 0 to disable checkpoint storage accounting"
              colors={colors}
            />
            {/* P74 */}
            <NumberInput
              label="Checkpoint freq"
              value={config.failureModel.checkpointFrequencyPerDay}
              onChange={(v) =>
                set({
                  failureModel: {
                    ...config.failureModel,
                    checkpointFrequencyPerDay: v,
                  },
                })
              }
              min={0}
              unit="/day"
              tooltip="Set 0 to disable checkpoint creation/storage. Failure recovery requires a positive frequency when failures are enabled."
              colors={colors}
            />
            {/* P75 */}
            <NumberInput
              label="Storage price"
              value={config.pricing.storagePricePerGBMonth}
              onChange={(v) => setPrice({ storagePricePerGBMonth: v })}
              min={0}
              step={0.001}
              unit="$/GB/mo"
              colors={colors}
            />
            {/* P76 */}
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

          {/* P77 / P78 — Failure model, shown when GPUs >= 256 */}
          {effectiveNumGPUs >= 256 && (
            <div className="grid gap-3 sm:grid-cols-2">
              <NumberInput
                label="Failure rate"
                value={config.failureModel.failureRatePerInstancePerDay}
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
            </div>
          )}
        </div>

        {host.outputSlots.cost}
        {host.warningSlots.cost}
      </Layer>

      {/* ——— Layer 8 · MoE ——— */}
      <Layer
        id="moe"
        title="Mixture of Experts (MoE)"
        colors={colors}
        dimmed={!moeEnabled}
        badge={moeEnabled ? "MoE" : undefined}
        open={host.isLayerOpen("moe")}
        onOpenChange={(o) => host.onLayerOpenChange("moe", o)}
        summary={host.summaries.moe}
        warningCount={host.warningChips.moe?.count}
        warningSeverity={host.warningChips.moe?.severity}
      >
        <div className="space-y-3">
          {moeEnabled && <MoEOverviewNote colors={colors} />}
          <SubLabel colors={colors}>MoE routing</SubLabel>
          <div className="grid gap-3 sm:grid-cols-2">
            {/* P51 */}
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
            {/* P52 */}
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
            {/* P53 */}
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
            {/* P54 */}
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
            {/* P55 */}
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
            {/* P56 */}
            <NumberInput
              label="Dense FFN size"
              value={
                config.model.moe.denseIntermediateSize ??
                config.model.architecture.d_ff ??
                resolveDefaultFFNIntermediateSize(
                  config.model.architecture.d,
                  config.model.architecture.ffnType,
                )
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
              tooltip="Intermediate size for dense FFN layers. Defaults to d_ff or the dense FFN type default."
              colors={colors}
            />
            {/* P57 */}
            <NumberInput
              label="Expert FFN size"
              value={
                config.model.moe.expertIntermediateSize ??
                resolveDefaultMoEExpertIntermediateSize(
                  config.model.architecture.d,
                )
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
              tooltip="Intermediate size used by each expert block. Defaults to round(8d/3), independent of dense d_ff."
              colors={colors}
            />
            {/* P48 */}
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
          </div>
        </div>

        {host.outputSlots.moe}
        {host.warningSlots.moe}
      </Layer>
    </LayerStack>
  )
}
