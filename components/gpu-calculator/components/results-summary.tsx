"use client"

import type { ReactNode } from "react"
import { motion } from "framer-motion"
import {
  AlertCircle,
  AlertTriangle,
  BarChart3,
  Clock,
  Cpu,
  DollarSign,
  Grid3X3,
  Info,
  Layers,
  Zap,
} from "lucide-react"
import type {
  CalculatorOutput,
  CostEstimate,
  ParallelismConfig,
  PostTrainingOutput,
  PretrainingOutput,
  Warning,
} from "../types"
import MemoryBreakdownBar from "./memory-breakdown-bar"
import GpuUtilizationGauge from "./gpu-utilization-gauge"
import ParallelismLayout from "./parallelism-layout"

function formatMemory(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "--"
  }

  const tb = bytes / 1e12
  const gb = bytes / 1e9
  const mb = bytes / 1e6
  const kb = bytes / 1e3

  if (tb >= 1) return `${tb.toFixed(tb >= 10 ? 1 : 2)} TB`
  if (gb >= 999.5) return `${(gb / 1000).toFixed(2)} TB`
  if (gb >= 100) return `${Math.round(gb)} GB`
  if (gb >= 10) return `${gb.toFixed(1)} GB`
  if (gb >= 1) return `${gb.toFixed(2)} GB`
  if (mb >= 999.5) return `${(mb / 1000).toFixed(2)} GB`
  if (mb >= 1) return `${mb.toFixed(0)} MB`
  if (kb >= 999.5) return `${(kb / 1000).toFixed(0)} MB`
  if (kb >= 1) return `${kb.toFixed(0)} KB`
  return "< 1 KB"
}

function formatParams(value: number): string {
  if (!Number.isFinite(value)) {
    return "--"
  }

  const absolute = Math.abs(value)

  if (absolute >= 1e12) return `${(value / 1e12).toFixed(2)}T`
  if (absolute >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (absolute >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  if (absolute >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return value.toLocaleString()
}

const formatCount = formatParams

function formatFLOPs(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "--"
  }

  if (value >= 1e21) return `${(value / 1e21).toFixed(2)} ZFLOPs`
  if (value >= 1e18) return `${(value / 1e18).toFixed(2)} EFLOPs`
  if (value >= 1e15) return `${(value / 1e15).toFixed(2)} PFLOPs`
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} TFLOPs`
  return `${(value / 1e9).toFixed(2)} GFLOPs`
}

function formatCost(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "--"
  }

  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`
  if (value >= 1e3) return `$${Math.round(value).toLocaleString()}`
  return `$${value.toFixed(2)}`
}

function formatStorageFootprint(cost: CostEstimate): string {
  const checkpointFootprint = `Peak retained ${formatMemory(cost.peakCheckpointStorage)}`

  return Number.isFinite(cost.datasetStorageBytes) && cost.datasetStorageBytes > 0
    ? `${checkpointFootprint} + dataset ${formatMemory(cost.datasetStorageBytes)}`
    : checkpointFootprint
}

function formatDuration(hours: number): string {
  if (!Number.isFinite(hours) || hours < 0) {
    return "--"
  }

  if (hours >= 24 * 365) return `${(hours / (24 * 365)).toFixed(1)} years`
  if (hours >= 24) return `${(hours / 24).toFixed(1)} days`
  if (hours >= 1) return `${hours.toFixed(1)} hr`
  return `${Math.round(hours * 60)} min`
}

function formatFractionPercent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) {
    return "--"
  }

  return `${(value * 100).toFixed(digits)}%`
}

function formatPercent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) {
    return "--"
  }

  return `${value.toFixed(digits)}%`
}

function formatMultiplier(value: number): string {
  if (!Number.isFinite(value)) {
    return "--"
  }

  return `${value.toFixed(2)}x`
}

function formatBatchRelation(relation: PretrainingOutput["batchEfficiency"]["relation"]): string {
  if (relation === "below") {
    return "below B_crit, time-inefficient"
  }

  if (relation === "above") {
    return "above B_crit, compute-inefficient"
  }

  return "at B_crit"
}

function formatParallelism(config: ParallelismConfig): string {
  const parts = [`DP ${config.N_dp}`, `TP ${config.N_tp}`]

  if (config.N_cp > 1) {
    parts.push(`CP ${config.N_cp}`)
  }

  parts.push(`PP ${config.N_pp}`)

  if (config.N_ep > 1) {
    parts.push(`EP ${config.N_ep}`)
  }

  const sharding = config.fsdpStrategy
    ? `FSDP ${config.fsdpStrategy}`
    : `ZeRO-${config.zeroStage}`

  return `${parts.join(" x ")} | ${sharding}`
}

function isPretraining(output: CalculatorOutput): output is PretrainingOutput {
  return "parameterCounts" in output
}

function ResultCard({
  title,
  icon: Icon,
  children,
  className,
}: {
  title: string
  icon?: typeof Cpu
  children: ReactNode
  className?: string
}) {
  return (
    <motion.section
      className={`rounded-xl border border-border bg-surface-elevated/50 p-5 sm:p-6 backdrop-blur-sm ${className ?? ""}`}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="mb-5 flex items-center gap-2">
        {Icon && <Icon className="h-4 w-4 text-accent" />}
        <h3 className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted" style={{ fontFamily: "var(--font-sans)" }}>
          {title}
        </h3>
      </div>
      {children}
    </motion.section>
  )
}

function Stat({
  label,
  value,
  sub,
  highlight,
}: {
  label: string
  value: string
  sub?: string
  highlight?: boolean
}) {
  return (
    <div
      className={`min-w-0 overflow-hidden rounded-lg border p-4 ${
        highlight ? "border-accent/30 bg-accent-soft/30" : "border-border bg-background/25"
      }`}
    >
      <div className="text-[10px] font-medium uppercase tracking-[0.16em] text-muted">{label}</div>
      <div
        className={`mt-2.5 min-w-0 overflow-hidden font-mono text-lg font-semibold leading-tight tabular-nums [overflow-wrap:anywhere] ${
          highlight ? "text-accent" : "text-foreground"
        }`}
      >
        {value}
      </div>
      {sub && <div className="mt-1.5 text-xs leading-relaxed text-muted">{sub}</div>}
    </div>
  )
}

const SEVERITY_META = {
  info: {
    label: "Info",
    icon: Info,
    light: {
      bg: "oklch(0.965 0.022 220)",
      border: "oklch(0.89 0.05 220)",
      text: "oklch(0.43 0.11 220)",
    },
    dark: {
      bg: "oklch(0.22 0.03 220)",
      border: "oklch(0.35 0.05 220)",
      text: "oklch(0.77 0.09 220)",
    },
  },
  warning: {
    label: "Warning",
    icon: AlertTriangle,
    light: {
      bg: "oklch(0.975 0.03 80)",
      border: "oklch(0.90 0.07 80)",
      text: "oklch(0.52 0.13 80)",
    },
    dark: {
      bg: "oklch(0.24 0.05 80)",
      border: "oklch(0.38 0.07 80)",
      text: "oklch(0.82 0.12 80)",
    },
  },
  critical: {
    label: "Error",
    icon: AlertCircle,
    light: {
      bg: "oklch(0.97 0.04 25)",
      border: "oklch(0.9 0.09 25)",
      text: "oklch(0.49 0.18 25)",
    },
    dark: {
      bg: "oklch(0.23 0.07 25)",
      border: "oklch(0.39 0.1 25)",
      text: "oklch(0.8 0.15 25)",
    },
  },
} as const

function WarningsPanel({ warnings, isDark }: { warnings: Warning[]; isDark: boolean }) {
  const mode = isDark ? "dark" : "light"
  const sortedWarnings = [...warnings].sort((left, right) => {
    const priority = { critical: 0, warning: 1, info: 2 }
    return priority[left.severity] - priority[right.severity]
  })

  return (
    <ResultCard title="Warnings" icon={AlertTriangle}>
      <div className="space-y-2">
        {sortedWarnings.map((warning, index) => {
          const meta = SEVERITY_META[warning.severity]
          const Icon = meta.icon

          return (
            <motion.div
              key={`${warning.severity}-${warning.category}-${index}`}
              className="rounded-lg border px-4 py-3"
              style={{
                backgroundColor: meta[mode].bg,
                borderColor: meta[mode].border,
                color: meta[mode].text,
              }}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.03, duration: 0.22 }}
            >
              <div className="flex items-start gap-3">
                <Icon className="mt-0.5 h-4 w-4 shrink-0" />
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[10px] font-medium uppercase tracking-[0.18em] opacity-75">
                      {meta.label}
                    </span>
                    <span className="text-[10px] uppercase tracking-[0.18em] opacity-55">
                      {warning.category}
                    </span>
                  </div>
                  <p className="mt-1 text-sm leading-6">{warning.message}</p>
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>
    </ResultCard>
  )
}

const POST_TRAINING_ITEM_META = {
  trainable: {
    label: "Trainable",
    light: "oklch(0.55 0.145 180)",
    dark: "oklch(0.72 0.12 180)",
  },
  frozen: {
    label: "Frozen",
    light: "oklch(0.60 0.03 260)",
    dark: "oklch(0.58 0.03 260)",
  },
  adapter: {
    label: "Adapter",
    light: "oklch(0.60 0.16 320)",
    dark: "oklch(0.72 0.14 320)",
  },
  buffer: {
    label: "Buffer",
    light: "oklch(0.60 0.13 150)",
    dark: "oklch(0.74 0.10 150)",
  },
} as const

function PostTrainingMemoryItems({
  output,
  isDark,
}: {
  output: PostTrainingOutput
  isDark: boolean
}) {
  const items = [...output.memory.items]
    .filter((item) => item.bytes > 0)
    .sort((left, right) => right.bytes - left.bytes)
  const total = Math.max(output.memory.total, 1)
  const mode = isDark ? "dark" : "light"

  return (
    <div className="space-y-3">
      {items.map((item, index) => {
        const meta = POST_TRAINING_ITEM_META[item.category]
        const share = (item.bytes / total) * 100

        return (
          <motion.div
            key={`${item.label}-${index}`}
            className="rounded-lg border border-border bg-background/25 p-4"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.03, duration: 0.22 }}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: meta[mode] }}
                  />
                  <span className="truncate text-sm text-foreground">{item.label}</span>
                </div>
                <div className="mt-1 text-xs text-muted">{meta.label}</div>
              </div>

              <div className="text-right">
                <div className="font-mono text-sm font-semibold tabular-nums text-foreground">
                  {formatMemory(item.bytes)}
                </div>
                <div className="text-xs text-muted">{formatPercent(share)}</div>
              </div>
            </div>

            <div className="mt-3 h-2 overflow-hidden rounded-full bg-surface">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: meta[mode] }}
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(share, 100)}%` }}
                transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}

interface Props {
  output: CalculatorOutput
  isDark: boolean
}

export default function ResultsSummary({ output, isDark }: Props) {
  if (isPretraining(output)) {
    return <PretrainingResults output={output} isDark={isDark} />
  }

  return <PostTrainingResults output={output} isDark={isDark} />
}

function PretrainingResults({
  output,
  isDark,
}: {
  output: PretrainingOutput
  isDark: boolean
}) {
  const dataSeverity =
    output.dataRepetition.severity === "none" ? "info" : output.dataRepetition.severity
  const dataTone = SEVERITY_META[dataSeverity][isDark ? "dark" : "light"]

  return (
    <div className="space-y-5">
      <ResultCard title="Memory Breakdown" icon={BarChart3}>
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.65fr)]">
          <MemoryBreakdownBar breakdown={output.memory} isDark={isDark} />

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
            <div className="rounded-lg border border-border bg-background/25 p-4">
              <GpuUtilizationGauge breakdown={output.memory} isDark={isDark} />
            </div>

            <div className="grid gap-3">
              <Stat label="Minimum GPUs Needed" value={formatCount(output.minGPUsNeeded)} />
              <Stat
                label="Minimum VRAM Floor"
                value={formatMemory(output.minVRAMFloor)}
                sub="Largest transformer block"
              />
              <Stat
                label="Maximum Micro-Batch"
                value={formatCount(output.maxMicroBatchSize)}
                sub="Sequences per GPU after model-state allocation"
              />
            </div>
          </div>
        </div>
      </ResultCard>

      <ResultCard title="Model and Compute" icon={Zap}>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          <Stat
            label="Parameters"
            value={formatParams(output.parameterCounts.total)}
            sub={
              output.parameterCounts.active !== output.parameterCounts.total
                ? `${formatParams(output.parameterCounts.active)} active`
                : undefined
            }
          />
          <Stat
            label="Total FLOPs"
            value={formatFLOPs(output.computeEstimate.totalFLOPs)}
            sub={`${formatFLOPs(output.computeEstimate.flopsPerToken)} per token`}
          />
          <Stat
            label="Chinchilla Ratio"
            value={formatMultiplier(output.chinchilla.ratio)}
            sub={`20x basis ${formatParams(output.chinchilla.parameterCount)} params; power-law target ${formatCount(output.chinchilla.powerLawOptimalTokens)} tok`}
          />
          <Stat
            label="Attention Overhead"
            value={formatFractionPercent(output.attentionOverheadFraction)}
            sub="Quadratic attention FLOPs relative to model FLOPs"
          />
          <Stat
            label="Predicted Loss"
            value={
              Number.isFinite(output.predictedLossNats)
                ? `${output.predictedLossNats.toFixed(3)} nats`
                : "--"
            }
            sub={`${output.chinchilla.coefficientRowLabel}; ${formatCount(output.chinchilla.effectiveLossTokens)} effective tok`}
          />
          {output.moeSparsity && (
            <Stat
              label="MoE Sparsity"
              value={formatFractionPercent(output.moeSparsity.sparsityRatio)}
              sub={`${formatMultiplier(output.moeSparsity.efficiencyGain)} memory efficiency`}
            />
          )}
        </div>

        <div className="mt-4 rounded-lg border border-border bg-background/25 p-4">
          <div className="text-[10px] uppercase tracking-[0.18em] text-muted">
            Chinchilla Recommendation
          </div>
          <p className="mt-2 text-sm leading-6 text-foreground">
            {output.chinchilla.recommendation}
          </p>
        </div>
      </ResultCard>

      <ResultCard title="Parallelism Strategy" icon={Grid3X3}>
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(18rem,0.85fr)]">
          <ParallelismLayout config={output.parallelismRecommendation.config} isDark={isDark} />

          <div className="space-y-3">
            <div className="rounded-lg border border-border bg-background/25 p-4">
              <div className="text-[10px] uppercase tracking-[0.18em] text-muted">
                Layout
              </div>
              <p className="mt-2 font-mono text-sm leading-6 text-foreground">
                {formatParallelism(output.parallelismRecommendation.config)}
              </p>
            </div>

            <div className="rounded-lg border border-border bg-background/25 p-4">
              <div className="text-[10px] uppercase tracking-[0.18em] text-muted">
                Recommendation
              </div>
              <p className="mt-2 text-sm leading-6 text-foreground">
                {output.parallelismRecommendation.strategyLabel}
              </p>
            </div>

            <Stat
              label="Pipeline Bubble"
              value={formatFractionPercent(output.pipelineBubbleFraction)}
              sub="Idle fraction from pipeline flush/fill"
            />

            {output.parallelismRecommendation.reasoning.length > 0 && (
              <div className="rounded-lg border border-border bg-background/25 p-4">
                <div className="text-[10px] uppercase tracking-[0.18em] text-muted">
                  Reasoning
                </div>
                <ul className="mt-3 space-y-2 text-sm leading-6 text-foreground">
                  {output.parallelismRecommendation.reasoning.map((reason, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className="mt-2 inline-block h-1.5 w-1.5 rounded-full bg-accent" />
                      <span>{reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </ResultCard>

      <ResultCard title="Training Performance" icon={Clock}>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Stat
            label="Training Time"
            value={formatDuration(output.trainingTime.theoreticalHours)}
            sub={
              output.trainingTime.failureAdjustedHours != null
                ? `${formatDuration(output.trainingTime.failureAdjustedHours)} failure-adjusted`
                : undefined
            }
          />
          <Stat
            label="Throughput"
            value={`${formatCount(output.tokensPerSecond)} tok/s`}
            sub={`${formatCount(output.trainingTime.totalSteps)} total steps`}
          />
          <Stat
            label="Global Batch Size"
            value={`${formatCount(output.globalBatchSize.sequences)} seq`}
            sub={`${formatCount(output.globalBatchSize.tokens)} tokens`}
          />
          <Stat
            label="Batch Compute Overhead"
            value={formatMultiplier(output.batchEfficiency.computeMultiplier)}
            sub={`${formatCount(output.batchEfficiency.actualBatchTokens)} tok vs ${formatCount(output.batchEfficiency.criticalBatchTokens)} tok, ${formatBatchRelation(output.batchEfficiency.relation)}, ${formatFractionPercent(output.batchEfficiency.wastedComputeFraction)} of actual compute above optimum`}
          />
        </div>

        {output.dataRepetition.hasRepetition && (
          <div
            className="mt-4 rounded-2xl border px-4 py-3"
            style={{
              backgroundColor: dataTone.bg,
              borderColor: dataTone.border,
              color: dataTone.text,
            }}
          >
            <div className="text-[10px] uppercase tracking-[0.18em] opacity-75">
              Data Repetition
            </div>
            <div className="mt-2 font-mono text-lg font-semibold">
              {output.dataRepetition.epochs.toFixed(1)} epochs
            </div>
            <p className="mt-1 text-sm leading-6">{output.dataRepetition.recommendation}</p>
            <div className="mt-2 text-xs opacity-80">
              Effective ceiling: {formatCount(output.dataRepetition.effectiveDataCeiling)} tokens
            </div>
          </div>
        )}
      </ResultCard>

      <ResultCard title="Cost Estimate" icon={DollarSign}>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Stat
            label="Compute Cost"
            value={formatCost(output.cost.computeCost)}
            sub={
              output.cost.actualComputeCost != null &&
              output.cost.actualComputeCost !== output.cost.computeCost
                ? `Actual compute ${formatCost(output.cost.actualComputeCost)}`
                : undefined
            }
          />
          <Stat
            label="Storage Cost"
            value={formatCost(output.cost.storageCost)}
            sub={formatStorageFootprint(output.cost)}
          />
          <Stat
            label="Failure Overhead"
            value={formatCost(output.cost.failureOverheadCost)}
            sub={`${output.cost.numCheckpoints.toLocaleString()} checkpoints`}
          />
          <Stat label="Total Cost" value={formatCost(output.cost.totalCost)} highlight />
        </div>

        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <Stat
            label="Checkpoint Size"
            value={formatMemory(output.checkpointSize)}
            sub={`${formatCount(output.cost.numCheckpoints)} projected saves`}
          />
          <Stat
            label="Checkpoint Storage"
            value={formatMemory(output.cost.averageCheckpointStorage)}
            sub={`Average retained footprint, peak ${formatMemory(output.cost.peakCheckpointStorage)}`}
          />
        </div>
      </ResultCard>

      {output.moeSparsity && (
        <ResultCard title="MoE Metrics" icon={Layers}>
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat
              label="Sparsity Ratio"
              value={formatFractionPercent(output.moeSparsity.sparsityRatio)}
            />
            <Stat
              label="Efficiency Gain"
              value={formatMultiplier(output.moeSparsity.efficiencyGain)}
            />
            <Stat
              label="Load Balance Factor"
              value={
                Number.isFinite(output.moeSparsity.loadBalanceFactor)
                  ? output.moeSparsity.loadBalanceFactor.toFixed(2)
                  : "--"
              }
            />
          </div>
        </ResultCard>
      )}

      {output.warnings.length > 0 && <WarningsPanel warnings={output.warnings} isDark={isDark} />}
    </div>
  )
}

function PostTrainingResults({
  output,
  isDark,
}: {
  output: PostTrainingOutput
  isDark: boolean
}) {
  return (
    <div className="space-y-5">
      <ResultCard title="Memory Breakdown" icon={BarChart3}>
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.65fr)]">
          <MemoryBreakdownBar breakdown={output.memory} isDark={isDark} />

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
            <div className="rounded-lg border border-border bg-background/25 p-4">
              <GpuUtilizationGauge breakdown={output.memory} isDark={isDark} />
            </div>

            <div className="grid gap-3">
              <Stat
                label="GPUs Needed"
                value={
                  output.numGPUsNeeded === null
                    ? "--"
                    : formatCount(output.numGPUsNeeded)
                }
              />
              <Stat
                label="Free Headroom"
                value={formatMemory(output.memory.freeHeadroom)}
                sub="Remaining usable VRAM per GPU"
              />
              <Stat
                label="Working Set"
                value={formatMemory(output.memory.total)}
                sub="Allocator-adjusted per-GPU estimate"
              />
            </div>
          </div>
        </div>
      </ResultCard>

      {output.memory.items.length > 0 && (
        <ResultCard title="Post-Training Results" icon={Layers}>
          <PostTrainingMemoryItems output={output} isDark={isDark} />
        </ResultCard>
      )}

      <ResultCard title="Training Time" icon={Clock}>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Stat
            label="Estimated Time"
            value={formatDuration(output.trainingTime.theoreticalHours)}
            sub={
              output.trainingTime.failureAdjustedHours != null
                ? `${formatDuration(output.trainingTime.failureAdjustedHours)} failure-adjusted`
                : undefined
            }
          />
          <Stat
            label="Throughput"
            value={`${formatCount(output.trainingTime.tokensPerSecond)} tok/s`}
            sub={`${formatCount(output.trainingTime.totalSteps)} total steps`}
          />
          <Stat
            label="Seconds per Step"
            value={
              Number.isFinite(output.trainingTime.secondsPerStep)
                ? `${output.trainingTime.secondsPerStep.toFixed(2)} s`
                : "--"
            }
          />
          <Stat
            label="Failure Multiplier"
            value={
              output.trainingTime.failureMultiplier != null
                ? formatMultiplier(output.trainingTime.failureMultiplier)
                : "--"
            }
          />
        </div>
      </ResultCard>

      <ResultCard title="Cost Estimate" icon={DollarSign}>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Stat
            label="Compute Cost"
            value={formatCost(output.cost.computeCost)}
            sub={
              output.cost.actualComputeCost != null &&
              output.cost.actualComputeCost !== output.cost.computeCost
                ? `Actual compute ${formatCost(output.cost.actualComputeCost)}`
                : undefined
            }
          />
          <Stat
            label="Storage Cost"
            value={formatCost(output.cost.storageCost)}
            sub={formatStorageFootprint(output.cost)}
          />
          <Stat
            label="Failure Overhead"
            value={formatCost(output.cost.failureOverheadCost)}
            sub={`${output.cost.numCheckpoints.toLocaleString()} checkpoints`}
          />
          <Stat label="Total Cost" value={formatCost(output.cost.totalCost)} highlight />
        </div>
      </ResultCard>

      {output.warnings.length > 0 && <WarningsPanel warnings={output.warnings} isDark={isDark} />}
    </div>
  )
}
