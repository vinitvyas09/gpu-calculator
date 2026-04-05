"use client"

import type { ReactNode } from "react"
import {
  Brain,
  Database,
  HardDrive,
  Settings2,
  SlidersHorizontal,
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
import { OPTIMIZER_PROFILES } from "../constants"
import { estimateParametersQuick } from "../formulas/compute"
import {
  type CalculatorColors,
  CheckboxGroupInput,
  NumberInput,
  SelectInput,
  formatCompact,
} from "./input-controls"
import { BaseModelSelector } from "./model-selector"
import { GPUSelector } from "./gpu-selector"

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

function resolveLoRABaseArchitecture(config: PostTrainingConfig) {
  return config.baseModel.inputMode === "preset"
    ? config.baseModel.architecture
    : estimateParametersQuick(config.baseModel.parameterCount)
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
  const d = architecture.d
  const dff = architecture.d_ff ?? 4 * d
  const kvWidth =
    architecture.a_kv && architecture.a > 0
      ? Math.round((d * architecture.a_kv) / architecture.a)
      : d

  const moduleShapes: Record<LoRATargetModule, [number, number]> = {
    q_proj: [d, d],
    k_proj: [d, kvWidth],
    v_proj: [d, kvWidth],
    o_proj: [d, d],
    gate_proj: [d, dff],
    up_proj: [d, dff],
    down_proj: [dff, d],
  }

  const perLayer = config.lora.targetModules.reduce((sum, moduleId) => {
    const [inputDim, outputDim] = moduleShapes[moduleId]
    return sum + config.lora.rank * (inputDim + outputDim)
  }, 0)

  return perLayer * architecture.L
}

function normalizePostTrainingConfig(
  config: PostTrainingConfig,
): PostTrainingConfig {
  if (config.approach !== "lora" && config.approach !== "qlora") {
    return config
  }

  const estimatedLoRAParams = estimateLoRAParameterCount(config)
  const trainableParameterPercentage =
    estimatedLoRAParams !== null && config.baseModel.parameterCount > 0
      ? (estimatedLoRAParams / config.baseModel.parameterCount) * 100
      : null

  return {
    ...config,
    trainableParameterPercentage,
  }
}

// ---------------------------------------------------------------------------
// PostTrainingPanel
// ---------------------------------------------------------------------------
export function PostTrainingPanel({
  config,
  onChange,
  colors,
}: {
  config: PostTrainingConfig
  onChange: (c: PostTrainingConfig) => void
  colors: CalculatorColors
}) {
  const commitConfig = (nextConfig: PostTrainingConfig) =>
    onChange(normalizePostTrainingConfig(nextConfig))

  const set = (patch: Partial<PostTrainingConfig>) =>
    commitConfig({ ...config, ...patch })

  const setBaseModel = (baseModel: BaseModelSelection) =>
    commitConfig({ ...config, baseModel })

  const setLora = (patch: Partial<LoRAConfig>) =>
    commitConfig({ ...config, lora: { ...config.lora, ...patch } })

  const setHw = (patch: Partial<PostTrainingHardwareSelection>) =>
    commitConfig({
      ...config,
      hardware: { ...config.hardware, ...patch },
    })

  const setApproach = (approach: FineTuningApproach) => {
    const leavingAdapterMode =
      config.approach === "lora" || config.approach === "qlora"

    commitConfig({
      ...config,
      approach,
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
    (o) => o.supportsPostTraining,
  ).map((o) => ({ value: o.id, label: o.name }))

  const isLoRA = config.approach === "lora" || config.approach === "qlora"
  const isMeZO = config.approach === "mezo"
  const estimatedLoRAParams = estimateLoRAParameterCount(config)
  const estimatedLoRAPercentage =
    estimatedLoRAParams && config.baseModel.parameterCount > 0
      ? (estimatedLoRAParams / config.baseModel.parameterCount) * 100
      : null

  return (
    <div className="space-y-6">
      {/* ——— 1. Base model ——— */}
      <Section title="Base Model" icon={Brain} colors={colors}>
        <BaseModelSelector
          selection={config.baseModel}
          onChange={setBaseModel}
          colors={colors}
        />
      </Section>

      {/* ——— 2–3. Method & approach ——— */}
      <Section title="Method &amp; Approach" icon={Wrench} colors={colors}>
        <div className="grid gap-3 sm:grid-cols-2">
          {/* 2 */}
          <SelectInput
            label="Method"
            value={config.method}
            onChange={(v) => set({ method: v as PostTrainingMethod })}
            options={[
              { value: "sft", label: "SFT (Supervised Fine-Tuning)" },
              { value: "dpo", label: "DPO (Direct Preference)" },
              { value: "ppo", label: "PPO (Proximal Policy)" },
              { value: "grpo", label: "GRPO (Group Relative)" },
            ]}
            colors={colors}
          />
          {/* 3 */}
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

        {/* 4a — Trainable param % (for full fine-tuning or partial layer freezing) */}
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
              tooltip="Percentage of parameters to train — for partial layer freezing or limited-scope MeZO updates."
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

      {/* ——— 4. LoRA configuration (conditional) ——— */}
      {isLoRA && (
        <Section title="LoRA Configuration" icon={Settings2} colors={colors}>
          <div className="grid gap-3 sm:grid-cols-2">
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
            <NumberInput
              label="Alpha"
              value={config.lora.alpha}
              onChange={(v) => setLora({ alpha: v })}
              min={1}
              integer
              tooltip="LoRA scaling factor — typically 2x rank"
              colors={colors}
            />
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
                  { value: "4", label: "4-bit (NF4)" },
                  { value: "8", label: "8-bit" },
                ]}
                tooltip="Base model quantization for QLoRA"
                colors={colors}
              />
            )}
          </div>
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

      {/* ——— 5. PPO configuration (conditional) ——— */}
      {config.method === "ppo" && (
        <Section title="PPO Configuration" icon={Settings2} colors={colors}>
          <div className="grid gap-3 sm:grid-cols-2">
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
          </div>
        </Section>
      )}

      {/* ——— 6. GRPO configuration (conditional) ——— */}
      {config.method === "grpo" && (
        <Section title="GRPO Configuration" icon={Settings2} colors={colors}>
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
        </Section>
      )}

      {/* ——— 7–9. Training data ——— */}
      <Section title="Training Data" icon={Database} colors={colors}>
        <div className="grid gap-3 sm:grid-cols-2">
          {/* 7 */}
          <NumberInput
            label="Dataset size"
            value={config.datasetSizeExamples}
            onChange={(v) => set({ datasetSizeExamples: v })}
            min={1}
            integer
            compact
            unit="examples"
            tooltip="Number of training examples"
            colors={colors}
          />
          {/* 8 */}
          <NumberInput
            label="Epochs"
            value={config.epochs}
            onChange={(v) => set({ epochs: v })}
            min={1}
            colors={colors}
          />
          {/* 9 */}
          <NumberInput
            label="Sequence length"
            value={config.sequenceLength}
            onChange={(v) => set({ sequenceLength: v })}
            min={128}
            step={128}
            integer
            colors={colors}
          />
          <NumberInput
            label="Batch size"
            value={config.batchSize}
            onChange={(v) => set({ batchSize: v })}
            min={1}
            integer
            colors={colors}
          />
        </div>
      </Section>

      {/* ——— 11–15. Training setup ——— */}
      <Section
        title="Training Setup"
        icon={SlidersHorizontal}
        colors={colors}
      >
        <div className="grid gap-3 sm:grid-cols-2">
          {/* 11 */}
          <SelectInput
            label="Precision"
            value={config.precision}
            onChange={(v) =>
              set({ precision: v as TrainingPrecision })
            }
            options={[
              { value: "bf16", label: "BF16" },
              { value: "fp16", label: "FP16" },
              { value: "fp32", label: "FP32" },
              { value: "fp8", label: "FP8" },
            ]}
            colors={colors}
          />
          {/* 12 */}
          <SelectInput
            label="Optimizer"
            value={isMeZO ? "mezo" : config.optimizer}
            onChange={(v) => set({ optimizer: v as OptimizerType })}
            options={
              isMeZO
                ? [{ value: "mezo", label: "MeZO" }]
                : optimizerOptions
            }
            disabled={isMeZO}
            colors={colors}
          />
          {/* 13 */}
          <SelectInput
            label="Gradient precision"
            value={config.gradientPrecision}
            onChange={(v) =>
              set({
                gradientPrecision: v as GradientPrecision,
              })
            }
            options={[
              { value: "fp32", label: "FP32" },
              { value: "bf16", label: "BF16" },
            ]}
            disabled={isMeZO}
            colors={colors}
          />
          {/* 15 */}
          <SelectInput
            label="KV cache precision"
            value={config.kvCachePrecision}
            onChange={(v) =>
              set({ kvCachePrecision: v as KVCachePrecision })
            }
            options={[
              { value: "bf16", label: "BF16" },
              { value: "fp16", label: "FP16" },
              { value: "int8", label: "INT8" },
            ]}
            colors={colors}
          />
        </div>
      </Section>

      {/* ——— 10, 14. Hardware & cost ——— */}
      <Section title="Hardware &amp; Cost" icon={HardDrive} colors={colors}>
        {/* 10 */}
        <GPUSelector
          gpuId={config.hardware.gpuId}
          gpu={config.hardware.gpu}
          inputMode={config.hardware.inputMode}
          onChange={({ gpuId, gpu, inputMode }) =>
            setHw({ gpuId, gpu, inputMode })
          }
          colors={colors}
          precision={config.precision}
        />

        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <NumberInput
            label="Number of GPUs"
            value={config.hardware.numGPUs}
            onChange={(v) => setHw({ numGPUs: v })}
            min={1}
            integer
            colors={colors}
          />
          {/* 14 */}
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
