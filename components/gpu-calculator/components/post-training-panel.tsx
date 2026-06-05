"use client"

import type { ReactNode } from "react"
import {
  Brain,
  Database,
  HardDrive,
  Settings2,
  Wrench,
} from "lucide-react"
import type {
  BaseModelSelection,
  FineTuningApproach,
  GradientPrecision,
  KVCachePrecision,
  LoRAConfig,
  LoRATargetModule,
  OptimizerType,
  PostTrainingConfig,
  PostTrainingHardwareSelection,
  PostTrainingMethod,
  TrainingPrecision,
} from "../types"
import { CLOUD_PRICING_PRESETS, OPTIMIZER_PROFILES } from "../constants"
import { estimateParametersQuick } from "../formulas/compute"
import { calculateLoRAParamCountForArchitecture } from "../formulas/memory"
import {
  type CalculatorColors,
  CheckboxGroupInput,
  NumberInput,
  SelectInput,
  ToggleInput,
  formatCompact,
} from "./input-controls"
import { BaseModelSelector } from "./model-selector"
import { GPUSelector } from "./gpu-selector"
import { Layer, type LayerHostProps } from "./layer"

// ---------------------------------------------------------------------------
// Section header (same pattern as pretraining)
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
// All LoRA target module options
// ---------------------------------------------------------------------------
const LORA_MODULE_OPTIONS: { value: LoRATargetModule; label: string }[] = [
  { value: "q_proj", label: "q_proj" },
  { value: "k_proj", label: "k_proj" },
  { value: "v_proj", label: "v_proj" },
  { value: "o_proj", label: "o_proj" },
  { value: "gate_proj", label: "gate_proj" },
  { value: "up_proj", label: "up_proj" },
  { value: "down_proj", label: "down_proj" },
]

const COST_SYNC_EPSILON = 1e-9
const PARAMETER_SYNC_RELATIVE_EPSILON = 1e-9

function getHardwarePricePreset(hardware: PostTrainingHardwareSelection) {
  if (hardware.inputMode !== "preset") {
    return null
  }

  const gpuId = hardware.gpuId ?? hardware.gpu.id
  return CLOUD_PRICING_PRESETS.find((preset) => preset.gpuId === gpuId) ?? null
}

function shouldSyncCostToHardware(config: PostTrainingConfig): boolean {
  const currentPreset = getHardwarePricePreset(config.hardware)

  return (
    currentPreset !== null &&
    Math.abs(config.costPerGPUHour - currentPreset.priceDefault) <=
      COST_SYNC_EPSILON
  )
}

function shouldSyncParameterCount(
  currentValue: number,
  previousBaseParameterCount: number,
): boolean {
  if (
    !Number.isFinite(currentValue) ||
    !Number.isFinite(previousBaseParameterCount) ||
    previousBaseParameterCount <= 0
  ) {
    return false
  }

  const tolerance = Math.max(
    1,
    Math.abs(previousBaseParameterCount) * PARAMETER_SYNC_RELATIVE_EPSILON,
  )

  return Math.abs(currentValue - previousBaseParameterCount) <= tolerance
}

function resolveLoRABaseArchitecture(config: PostTrainingConfig) {
  return config.baseModel.inputMode === "preset"
    ? config.baseModel.architecture
    : estimateParametersQuick(config.baseModel.parameterCount)
}

function resolveLoRABaseMoE(config: PostTrainingConfig) {
  return config.baseModel.inputMode === "preset"
    ? config.baseModel.moe
    : {
        ...config.baseModel.moe,
        enabled: false,
        activeParameterCount: null,
      }
}

function estimateLoRAParameterCount(
  config: PostTrainingConfig,
): number | null {
  if (
    config.approach !== "lora" &&
    config.approach !== "qlora"
  ) {
    return null
  }
  if (config.baseModel.parameterCount <= 0) {
    return null
  }

  const architecture = resolveLoRABaseArchitecture(config)
  const moe = resolveLoRABaseMoE(config)

  const parameterCount = calculateLoRAParamCountForArchitecture(
    architecture,
    moe,
    config.lora,
  )

  return Number.isFinite(parameterCount) ? parameterCount : null
}

function normalizePostTrainingConfig(
  config: PostTrainingConfig,
): PostTrainingConfig {
  const normalizedConfig: PostTrainingConfig = {
    ...config,
    method: config.approach === "mezo" ? "sft" : config.method,
    grpo: {
      ...config.grpo,
      rewardModelParameterCount:
        config.grpo.rewardModelParameterCount ?? 0,
    },
    optimizer:
      config.approach === "mezo"
        ? "mezo"
        : config.optimizer === "mezo"
          ? "adamw-mixed"
          : config.optimizer,
  }

  if (
    normalizedConfig.approach !== "lora" &&
    normalizedConfig.approach !== "qlora"
  ) {
    return normalizedConfig
  }

  const estimatedLoRAParams = estimateLoRAParameterCount(normalizedConfig)
  const trainableParameterPercentage =
    estimatedLoRAParams !== null && normalizedConfig.baseModel.parameterCount > 0
      ? (estimatedLoRAParams / normalizedConfig.baseModel.parameterCount) * 100
      : null

  return {
    ...normalizedConfig,
    trainableParameterPercentage,
  }
}

function getTrainingDataLabels(method: PostTrainingMethod): {
  datasetUnit: string
  datasetTooltip: string
  sequenceTooltip: string
  batchTooltip: string
} {
  switch (method) {
    case "dpo":
      return {
        datasetUnit: "pairs",
        datasetTooltip:
          "Number of preference pairs. Each pair contains chosen and rejected responses, so scored tokens are doubled.",
        sequenceTooltip:
          "Token length per chosen or rejected response. Prompt prefill is not modeled separately.",
        batchTooltip: "Preference pairs per training batch.",
      }
    case "ppo":
      return {
        datasetUnit: "prompts",
        datasetTooltip:
          "Number of rollout prompts. Generation, reward, value, and reference/KL scoring are counted once per prompt.",
        sequenceTooltip:
          "Generated/scored token horizon per rollout response. Prompt prefill is not modeled separately.",
        batchTooltip: "Rollout prompts per batch.",
      }
    case "grpo":
      return {
        datasetUnit: "prompts",
        datasetTooltip:
          "Number of prompts. Generated and scored responses multiply by the GRPO group size.",
        sequenceTooltip:
          "Generated/scored token horizon per response in the group. Prompt prefill is not modeled separately.",
        batchTooltip: "Prompts per batch before multiplying by GRPO group size.",
      }
    case "sft":
    default:
      return {
        datasetUnit: "examples",
        datasetTooltip:
          "Number of supervised training examples. Total tokens scale with examples, epochs, and sequence length.",
        sequenceTooltip: "Token length per supervised training example.",
        batchTooltip: "Training examples per batch.",
      }
  }
}

// ---------------------------------------------------------------------------
// usePostTrainingWiring — all the set* helpers + derivations for the panel.
//
// Moved verbatim out of the legacy PostTrainingPanel so both Essentials and
// Layers can share one wiring source. No numeric behavior changes: the same
// commitConfig/normalize path, the same sync helpers, the same derived ids.
// ---------------------------------------------------------------------------
function usePostTrainingWiring(
  config: PostTrainingConfig,
  onChange: (c: PostTrainingConfig) => void,
) {
  const commitConfig = (nextConfig: PostTrainingConfig) =>
    onChange(normalizePostTrainingConfig(nextConfig))

  const set = (patch: Partial<PostTrainingConfig>) =>
    commitConfig({ ...config, ...patch })

  const setBaseModel = (baseModel: BaseModelSelection) => {
    const previousBaseParameterCount = config.baseModel.parameterCount
    const nextBaseParameterCount = baseModel.parameterCount
    const shouldSyncCritic = shouldSyncParameterCount(
      config.ppo.criticModelParameterCount,
      previousBaseParameterCount,
    )
    const shouldSyncReward = shouldSyncParameterCount(
      config.ppo.rewardModelParameterCount,
      previousBaseParameterCount,
    )
    const shouldSyncGRPOReward = shouldSyncParameterCount(
      config.grpo.rewardModelParameterCount ?? 0,
      previousBaseParameterCount,
    )
    const hasNextBaseParameterCount =
      Number.isFinite(nextBaseParameterCount) && nextBaseParameterCount > 0

    commitConfig({
      ...config,
      baseModel,
      ppo:
        hasNextBaseParameterCount && (shouldSyncCritic || shouldSyncReward)
          ? {
              ...config.ppo,
              criticModelParameterCount: shouldSyncCritic
                ? nextBaseParameterCount
                : config.ppo.criticModelParameterCount,
              rewardModelParameterCount: shouldSyncReward
                ? nextBaseParameterCount
                : config.ppo.rewardModelParameterCount,
            }
          : config.ppo,
      grpo:
        hasNextBaseParameterCount && shouldSyncGRPOReward
          ? {
              ...config.grpo,
              rewardModelParameterCount: nextBaseParameterCount,
            }
          : config.grpo,
    })
  }

  const setLora = (patch: Partial<LoRAConfig>) =>
    commitConfig({ ...config, lora: { ...config.lora, ...patch } })

  const setHw = (patch: Partial<PostTrainingHardwareSelection>) =>
    commitConfig({
      ...config,
      hardware: { ...config.hardware, ...patch },
    })

  const setHardwareSelection = (hardware: {
    gpuId: string | null
    gpu: PostTrainingHardwareSelection["gpu"]
    inputMode: PostTrainingHardwareSelection["inputMode"]
  }) => {
    const nextHardware = { ...config.hardware, ...hardware }
    const nextConfig: PostTrainingConfig = {
      ...config,
      hardware: nextHardware,
    }

    if (shouldSyncCostToHardware(config)) {
      const nextPreset = getHardwarePricePreset(nextHardware)
      if (nextPreset !== null) {
        nextConfig.costPerGPUHour = nextPreset.priceDefault
      }
    }

    commitConfig(nextConfig)
  }

  const setApproach = (approach: FineTuningApproach) => {
    const leavingAdapterMode =
      config.approach === "lora" || config.approach === "qlora"

    commitConfig({
      ...config,
      approach,
      method: approach === "mezo" ? "sft" : config.method,
      optimizer:
        approach === "mezo"
          ? "mezo"
          : config.optimizer === "mezo"
            ? "adamw-mixed"
            : config.optimizer,
      lora: {
        ...config.lora,
        quantizationBits:
          approach === "qlora"
            ? config.lora.quantizationBits ?? 4
            : null,
      },
      trainableParameterPercentage:
        leavingAdapterMode && (approach === "full" || approach === "mezo")
          ? null
          : config.trainableParameterPercentage,
    })
  }

  const optimizerOptions = OPTIMIZER_PROFILES.filter(
    (o) => o.supportsPostTraining && o.id !== "mezo",
  ).map((o) => ({ value: o.id, label: o.name }))

  const isLoRA = config.approach === "lora" || config.approach === "qlora"
  const isMeZO = config.approach === "mezo"
  const effectiveOptimizerId =
    config.optimizer === "adamw-fp8" &&
    (config.precision !== "fp8" ||
      !config.hardware.gpu.supportsFP8 ||
      config.fp8.storageMode === "transformer-engine")
      ? "adamw-mixed"
      : config.optimizer
  const selectedOptimizerProfile = OPTIMIZER_PROFILES.find(
    (optimizer) => optimizer.id === (isMeZO ? "mezo" : effectiveOptimizerId),
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
            selectedOptimizerProfile?.id === "mezo"
              ? "No gradients"
              : selectedOptimizerProfile?.id === "adamw-fp8"
                ? "FP8 gradients (fixed)"
                : "FP32 gradients (fixed)",
        },
      ]
    : [
        { value: "fp32", label: "FP32" },
        { value: "bf16", label: "BF16" },
      ]
  const gradientPrecisionTooltip = optimizerFixesGradientStorage
    ? `${selectedOptimizerProfile?.name ?? "Selected optimizer"} fixes gradient storage internally, so this setting does not change memory.`
    : "Precision for gradient accumulation — affects memory footprint."
  const estimatedLoRAParams = estimateLoRAParameterCount(config)
  const estimatedLoRAPercentage =
    estimatedLoRAParams && config.baseModel.parameterCount > 0
      ? (estimatedLoRAParams / config.baseModel.parameterCount) * 100
      : null
  const trainingDataLabels = getTrainingDataLabels(config.method)

  return {
    set,
    setBaseModel,
    setLora,
    setHw,
    setHardwareSelection,
    setApproach,
    optimizerOptions,
    isLoRA,
    isMeZO,
    optimizerFixesGradientStorage,
    gradientPrecisionValue,
    gradientPrecisionOptions,
    gradientPrecisionTooltip,
    estimatedLoRAParams,
    estimatedLoRAPercentage,
    trainingDataLabels,
  }
}

// ---------------------------------------------------------------------------
// PostTrainingEssentials — always-visible block (base model, method/approach,
// adapter + RL extras, dataset size + epochs, hardware & cost).
//
// Sequence length (T18) and batch size (T19) live in the layer stack, not here.
// ---------------------------------------------------------------------------
export function PostTrainingEssentials({
  config,
  onChange,
  colors,
}: {
  config: PostTrainingConfig
  onChange: (c: PostTrainingConfig) => void
  colors: CalculatorColors
}) {
  const {
    set,
    setBaseModel,
    setLora,
    setHardwareSelection,
    setHw,
    setApproach,
    isLoRA,
    isMeZO,
    estimatedLoRAParams,
    estimatedLoRAPercentage,
    trainingDataLabels,
  } = usePostTrainingWiring(config, onChange)

  return (
    <div className="space-y-8">
      {/* ——— Base model ——— */}
      <Section title="Base Model" icon={Brain} colors={colors}>
        <BaseModelSelector
          selection={config.baseModel}
          onChange={setBaseModel}
          colors={colors}
        />
      </Section>

      {/* ——— Method & approach ——— */}
      <Section title="Method &amp; Approach" icon={Wrench} colors={colors}>
        <div className="grid gap-3 sm:grid-cols-2">
          {/* T4 */}
          <SelectInput
            label="Method"
            value={isMeZO ? "sft" : config.method}
            onChange={(v) => set({ method: v as PostTrainingMethod })}
            options={[
              { value: "sft", label: "SFT (Supervised Fine-Tuning)" },
              { value: "dpo", label: "DPO (Direct Preference)" },
              { value: "ppo", label: "PPO (Proximal Policy)" },
              { value: "grpo", label: "GRPO (Group Relative)" },
            ]}
            disabled={isMeZO}
            colors={colors}
          />
          {/* T5 */}
          <SelectInput
            label="Approach"
            value={config.approach}
            onChange={(v) => setApproach(v as FineTuningApproach)}
            options={[
              { value: "full", label: "Full fine-tuning" },
              { value: "lora", label: "LoRA" },
              { value: "qlora", label: "QLoRA" },
              { value: "mezo", label: "MeZO (zeroth-order)" },
            ]}
            colors={colors}
          />
        </div>

        {/* T6 — Trainable param % (for full fine-tuning or partial layer freezing) */}
        {(config.approach === "full" || config.approach === "mezo") && (
          <div className="mt-3">
            <NumberInput
              label="Trainable parameter %"
              value={config.trainableParameterPercentage ?? 100}
              onChange={(v) =>
                set({
                  trainableParameterPercentage:
                    v >= 100 ? null : v,
                })
              }
              min={1}
              max={100}
              unit="%"
              tooltip="Percentage of parameters to train. Partial full fine-tuning assumes frozen layers can skip backward; MeZO still runs full-model forwards."
              colors={colors}
            />
          </div>
        )}
        {isLoRA && estimatedLoRAParams !== null && estimatedLoRAPercentage !== null && (
          <div
            className="mt-3 rounded-lg border p-3"
            style={{
              borderColor: colors.border,
              backgroundColor: colors.bg,
            }}
          >
            <div
              className="text-[11px] font-semibold uppercase tracking-[0.08em]"
              style={{ color: colors.textSecondary }}
            >
              Computed trainable footprint
            </div>
            <p className="mt-2 text-sm" style={{ color: colors.text }}>
              {formatCompact(estimatedLoRAParams)} trainable adapter parameters
            </p>
            <p className="mt-1 text-xs leading-6" style={{ color: colors.textSecondary }}>
              Approx. {estimatedLoRAPercentage < 1 ? estimatedLoRAPercentage.toFixed(3) : estimatedLoRAPercentage.toFixed(2)}% of the base model, derived from the selected LoRA targets and rank.
            </p>
          </div>
        )}
      </Section>

      {/* ——— LoRA configuration (conditional) ——— */}
      {isLoRA && (
        <Section title="LoRA Configuration" icon={Settings2} colors={colors}>
          <div className="grid gap-3 sm:grid-cols-2">
            {/* T7 */}
            <NumberInput
              label="Rank (r)"
              value={config.lora.rank}
              onChange={(v) => setLora({ rank: v })}
              min={1}
              max={256}
              integer
              tooltip="LoRA rank — controls adapter capacity"
              colors={colors}
            />
            {/* T8 */}
            <NumberInput
              label="Alpha"
              value={config.lora.alpha}
              onChange={(v) => setLora({ alpha: v })}
              min={1}
              integer
              tooltip="LoRA scaling factor — typically 2x rank"
              colors={colors}
            />
            {/* T9 */}
            {config.approach === "qlora" && (
              <SelectInput
                label="Quantization bits"
                value={String(config.lora.quantizationBits ?? 4)}
                onChange={(v) =>
                  setLora({
                    quantizationBits: Number(v) as 4 | 8,
                  })
                }
                options={[
                  { value: "4", label: "4-bit (NF4/GPTQ/AWQ)" },
                  { value: "8", label: "8-bit (LLM.int8/GPTQ/AWQ)" },
                ]}
                tooltip="Base model quantization for QLoRA"
                colors={colors}
              />
            )}
          </div>
          {/* T10 */}
          <div className="mt-3">
            <CheckboxGroupInput
              label="Target modules"
              values={config.lora.targetModules}
              allOptions={LORA_MODULE_OPTIONS}
              onChange={(v) =>
                setLora({ targetModules: v as LoRATargetModule[] })
              }
              tooltip="Which linear layers to apply LoRA adapters to"
              colors={colors}
            />
          </div>
        </Section>
      )}

      {/* ——— PPO configuration (conditional) ——— */}
      {config.method === "ppo" && (
        <Section title="PPO Configuration" icon={Settings2} colors={colors}>
          <div className="grid gap-3 sm:grid-cols-2">
            {/* T11 */}
            <NumberInput
              label="Critic model params"
              value={config.ppo.criticModelParameterCount}
              onChange={(v) =>
                set({
                  ppo: {
                    ...config.ppo,
                    criticModelParameterCount: v,
                  },
                })
              }
              min={1e6}
              integer
              compact
              tooltip="Critic (value) model parameter count"
              colors={colors}
            />
            {/* T12 */}
            <NumberInput
              label="Reward model params"
              value={config.ppo.rewardModelParameterCount}
              onChange={(v) =>
                set({
                  ppo: {
                    ...config.ppo,
                    rewardModelParameterCount: v,
                  },
                })
              }
              min={1e6}
              integer
              compact
              tooltip="Reward model parameter count"
              colors={colors}
            />
            {/* T13 */}
            <NumberInput
              label="Update epochs"
              value={config.ppo.updateEpochs}
              onChange={(v) =>
                set({
                  ppo: {
                    ...config.ppo,
                    updateEpochs: v,
                  },
                })
              }
              min={1}
              max={32}
              integer
              tooltip="PPO optimization epochs per rollout batch; generation, reward, value, and reference/KL scoring are counted once, while policy and critic optimizer work scale with epochs."
              colors={colors}
            />
          </div>
        </Section>
      )}

      {/* ——— GRPO configuration (conditional) ——— */}
      {config.method === "grpo" && (
        <Section title="GRPO Configuration" icon={Settings2} colors={colors}>
          <div className="grid gap-3 sm:grid-cols-2">
            {/* T14 */}
            <NumberInput
              label="Group size (G)"
              value={config.grpo.groupSize}
              onChange={(v) =>
                set({ grpo: { ...config.grpo, groupSize: v } })
              }
              min={2}
              integer
              tooltip="Number of responses per prompt in group relative scoring"
              colors={colors}
            />
            {/* T15 */}
            <NumberInput
              label="Reward model params"
              value={config.grpo.rewardModelParameterCount ?? 0}
              onChange={(v) =>
                set({
                  grpo: {
                    ...config.grpo,
                    rewardModelParameterCount: v,
                  },
                })
              }
              min={0}
              integer
              compact
              tooltip="Set 0 for rule-based, verifier, or precomputed rewards. Positive values add a frozen reward-model scoring pass over all GRPO completions."
              colors={colors}
            />
          </div>
        </Section>
      )}

      {/* ——— Training data (dataset size + epochs) ——— */}
      <Section title="Training Data" icon={Database} colors={colors}>
        <div className="grid gap-3 sm:grid-cols-2">
          {/* T16 */}
          <NumberInput
            label="Dataset size"
            value={config.datasetSizeExamples}
            onChange={(v) => set({ datasetSizeExamples: v })}
            min={1}
            integer
            compact
            unit={trainingDataLabels.datasetUnit}
            tooltip={trainingDataLabels.datasetTooltip}
            colors={colors}
          />
          {/* T17 */}
          <NumberInput
            label="Epochs"
            value={config.epochs}
            onChange={(v) => set({ epochs: v })}
            min={1}
            colors={colors}
          />
        </div>
      </Section>

      {/* ——— Hardware & cost ——— */}
      <Section title="Hardware &amp; Cost" icon={HardDrive} colors={colors}>
        {/* T27 / T28 */}
        <GPUSelector
          gpuId={config.hardware.gpuId}
          gpu={config.hardware.gpu}
          inputMode={config.hardware.inputMode}
          onChange={setHardwareSelection}
          colors={colors}
          precision={config.precision}
        />

        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {/* T29 */}
          <NumberInput
            label="Number of GPUs"
            value={config.hardware.numGPUs}
            onChange={(v) => setHw({ numGPUs: v })}
            min={1}
            integer
            colors={colors}
          />
          {/* T30 */}
          <NumberInput
            label="Cost per GPU-hour"
            value={config.costPerGPUHour}
            onChange={(v) => set({ costPerGPUHour: v })}
            min={0}
            step={0.1}
            unit="$/hr"
            colors={colors}
          />
        </div>
      </Section>
    </div>
  )
}

// ---------------------------------------------------------------------------
// PostTrainingLayers — the four summary-line layers for the post-training tab.
//
// No "parallelism" and no "moe" layer (PostTrainingConfig/PostTrainingOutput
// carry neither). Each body is its owned inputs followed by the host's
// pre-rendered output + warning slots for that layer id.
// ---------------------------------------------------------------------------
export function PostTrainingLayers({
  config,
  onChange,
  colors,
  host,
}: {
  config: PostTrainingConfig
  onChange: (c: PostTrainingConfig) => void
  colors: CalculatorColors
  host: LayerHostProps
}) {
  const {
    set,
    isMeZO,
    optimizerOptions,
    optimizerFixesGradientStorage,
    gradientPrecisionValue,
    gradientPrecisionOptions,
    gradientPrecisionTooltip,
    trainingDataLabels,
  } = usePostTrainingWiring(config, onChange)

  const layerProps = (id: string) => ({
    id,
    colors,
    summary: host.summaries[id],
    warningCount: host.warningChips[id]?.count,
    warningSeverity: host.warningChips[id]?.severity,
    open: host.isLayerOpen(id),
    onOpenChange: (next: boolean) => host.onLayerOpenChange(id, next),
  })

  return (
    <>
      {/* ——— 4 Model architecture ——— */}
      <Layer {...layerProps("architecture")} title="Model architecture">
        {/* T18 */}
        <NumberInput
          label="Sequence length"
          value={config.sequenceLength}
          onChange={(v) => set({ sequenceLength: v })}
          min={128}
          step={128}
          integer
          tooltip={trainingDataLabels.sequenceTooltip}
          colors={colors}
        />
        {host.outputSlots.architecture}
        {host.warningSlots.architecture}
      </Layer>

      {/* ——— 5 Precision & optimizer ——— */}
      <Layer {...layerProps("precision")} title="Precision & optimizer">
        <div className="grid gap-3 sm:grid-cols-2">
          {/* T20 */}
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
            colors={colors}
          />
          {/* T21 */}
          <SelectInput
            label="Optimizer"
            value={isMeZO ? "mezo" : config.optimizer}
            onChange={(v) => set({ optimizer: v as OptimizerType })}
            options={
              isMeZO ? [{ value: "mezo", label: "MeZO" }] : optimizerOptions
            }
            disabled={isMeZO}
            colors={colors}
          />
          {/* T22 */}
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
          {/* T23 */}
          <ToggleInput
            label="Chunked cross-entropy"
            value={config.chunkedCrossEntropy}
            onChange={(v) => set({ chunkedCrossEntropy: v })}
            tooltip="Eliminates materialized output logits and fp32 logits-gradient peak from language-model loss memory"
            colors={colors}
          />
          {/* T24 */}
          <SelectInput
            label="KV cache precision"
            value={config.kvCachePrecision}
            onChange={(v) => set({ kvCachePrecision: v as KVCachePrecision })}
            options={[
              { value: "bf16", label: "BF16" },
              { value: "fp16", label: "FP16" },
              { value: "int8", label: "INT8" },
            ]}
            colors={colors}
          />
          {config.precision === "fp8" && (
            <>
              {/* T25 */}
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
              {/* T26 */}
              <SelectInput
                label="FP8 storage mode"
                value={config.fp8.storageMode}
                onChange={(v) =>
                  set({
                    fp8: {
                      ...config.fp8,
                      storageMode:
                        v as PostTrainingConfig["fp8"]["storageMode"],
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
                tooltip="TransformerEngine uses FP8 kernels without model-state memory savings; MS-AMP stores parameters and gradients in FP8."
                colors={colors}
              />
            </>
          )}
        </div>
        {host.outputSlots.precision}
        {host.warningSlots.precision}
      </Layer>

      {/* ——— 6 Data & scaling ——— */}
      <Layer {...layerProps("data")} title="Data & scaling">
        {/* T19 */}
        <NumberInput
          label="Batch size"
          value={config.batchSize}
          onChange={(v) => set({ batchSize: v })}
          min={1}
          integer
          tooltip={trainingDataLabels.batchTooltip}
          colors={colors}
        />
        {host.outputSlots.data}
        {host.warningSlots.data}
      </Layer>

      {/* ——— 7 Cost detail & failures ——— */}
      <Layer {...layerProps("cost")} title="Cost detail & failures">
        {host.outputSlots.cost}
        {host.warningSlots.cost}
      </Layer>
    </>
  )
}

