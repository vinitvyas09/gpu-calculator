"use client"

import type { ReactNode } from "react"
import { motion, useReducedMotion } from "framer-motion"
import {
  AlertCircle,
  AlertTriangle,
  Cpu,
  Info,
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
import {
  formatCost,
  formatCount,
  formatDuration,
  formatFLOPs,
  formatFractionPercent,
  formatMemory,
  formatMultiplier,
  formatPercent,
} from "../formatters"

const formatParams = formatCount

function formatStorageFootprint(cost: CostEstimate): string {
  const checkpointFootprint = `Peak retained ${formatMemory(cost.peakCheckpointStorage)}`

  return Number.isFinite(cost.datasetStorageBytes) && cost.datasetStorageBytes > 0
    ? `${checkpointFootprint} + dataset ${formatMemory(cost.datasetStorageBytes)}`
    : checkpointFootprint
}

function formatBatchRelation(relation: PretrainingOutput["batchEfficiency"]["relation"]): string {
  if (relation === "below") {
    return "below B_crit, time-inefficient"
  }

  if (relation === "above") {
    return "above B_crit, compute-inefficient"
  }

  if (relation === "near") {
    return "near B_crit"
  }

  return "B_crit unavailable"
}

function formatPretrainingParameterSub(output: PretrainingOutput): string | undefined {
  const parts: string[] = []
  const rawCounts = output.parameterCounts
  const implementationCounts = output.implementationParameterCounts
  const hasActive = rawCounts.active !== rawCounts.total
  const hasImplementationPadding =
    implementationCounts.total !== rawCounts.total ||
    implementationCounts.active !== rawCounts.active

  if (hasActive) {
    parts.push(`${formatParams(rawCounts.active)} active`)
  }

  if (hasImplementationPadding) {
    const implementationSummary =
      implementationCounts.active !== implementationCounts.total
        ? `${formatParams(implementationCounts.total)} total, ${formatParams(implementationCounts.active)} active`
        : formatParams(implementationCounts.total)

    parts.push(`${implementationSummary} TP-padded implementation`)
  }

  return parts.length > 0 ? parts.join("; ") : undefined
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

export function isPretraining(output: CalculatorOutput): output is PretrainingOutput {
  return "parameterCounts" in output
}

export function ResultCard({
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
  const reduceMotion = useReducedMotion()
  return (
    <motion.section
      className={`rounded-xl border border-border bg-surface-elevated/50 p-5 sm:p-6 backdrop-blur-sm ${className ?? ""}`}
      initial={reduceMotion ? false : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={reduceMotion ? { duration: 0 } : { duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
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

export function Stat({
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
        className={`mt-2.5 min-w-0 overflow-hidden whitespace-nowrap font-mono text-lg font-semibold leading-tight tabular-nums ${
          highlight ? "text-accent" : "text-foreground"
        }`}
      >
        {value}
      </div>
      {sub && <div className="mt-1.5 text-xs leading-relaxed text-muted">{sub}</div>}
    </div>
  )
}

export const SEVERITY_META = {
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
    label: "Critical",
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

// ---------------------------------------------------------------------------
// WarningList — renders a GIVEN warnings array (no filtering; the host slices).
//
//   variant="inline"    → the warning-callout row styling (SEVERITY_META tints,
//                         icon + label + category + message). Severity sort is
//                         preserved so a mixed slice still reads critical-first.
//   variant="footnote"  → compact, de-emphasized small-print list for info items
//                         rendered at the bottom of a layer.
//
// No <ResultCard> wrapper: the host mounts this inside a Layer body / footnote.
// ---------------------------------------------------------------------------
export function WarningList({
  warnings,
  isDark,
  variant,
}: {
  warnings: Warning[]
  isDark: boolean
  variant: "inline" | "footnote"
}) {
  const reduceMotion = useReducedMotion()

  if (warnings.length === 0) {
    return null
  }

  const mode = isDark ? "dark" : "light"
  const sortedWarnings = [...warnings].sort((left, right) => {
    const priority = { critical: 0, warning: 1, info: 2 }
    return priority[left.severity] - priority[right.severity]
  })

  if (variant === "footnote") {
    return (
      <ul className="space-y-1.5">
        {sortedWarnings.map((warning, index) => (
          <li
            key={`${warning.severity}-${warning.category}-${index}`}
            className="flex items-start gap-2 text-xs leading-5 text-muted"
          >
            <Info className="mt-0.5 h-3 w-3 shrink-0 opacity-60" />
            <span className="min-w-0">
              <span className="uppercase tracking-[0.16em] opacity-60">{warning.category}</span>
              <span className="opacity-50"> &middot; </span>
              <span>{warning.message}</span>
            </span>
          </li>
        ))}
      </ul>
    )
  }

  return (
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
            initial={reduceMotion ? false : { opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={reduceMotion ? { duration: 0 } : { delay: index * 0.03, duration: 0.22 }}
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
  const reduceMotion = useReducedMotion()
  const items = [...output.memory.items]
    .filter((item) => Number.isFinite(item.bytes) && item.bytes > 0)
    .sort((left, right) => right.bytes - left.bytes)
  const itemTotal = Math.max(
    items.reduce((sum, item) => sum + item.bytes, 0),
    1,
  )
  const mode = isDark ? "dark" : "light"

  return (
    <div className="space-y-3">
      {items.map((item, index) => {
        const meta = POST_TRAINING_ITEM_META[item.category]
        const share = (item.bytes / itemTotal) * 100

        return (
          <motion.div
            key={`${item.label}-${index}`}
            className="rounded-lg border border-border bg-background/25 p-4"
            initial={reduceMotion ? false : { opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={reduceMotion ? { duration: 0 } : { delay: index * 0.03, duration: 0.22 }}
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
                initial={reduceMotion ? false : { width: 0 }}
                animate={{ width: `${Math.min(share, 100)}%` }}
                transition={reduceMotion ? { duration: 0 } : { duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}

// ===========================================================================
// Per-layer result fragments (Phase 3, Stage B1).
//
// Each fragment renders the metrics/viz/prose for ONE Layer per the D.6
// relocation table, preserving every Stat/visualization/prose EXACTLY as the
// monolith renders today (same formatters, same labels, same conditional
// logic — relabels are Phase 6). Fragments return their grid/blocks directly;
// the Layer header replaces the old ResultCard title, so they are NOT wrapped
// in a ResultCard. The host mounts each fragment inside its Layer body.
// ===========================================================================

// Layer 1 — Memory & feasibility (pretraining): D.6 rows MemoryBreakdownBar,
// GpuUtilizationGauge, Effective GPUs, Minimum GPUs Needed, Minimum VRAM Floor,
// Maximum Micro-Batch.
export function PretrainMemoryBody({
  output,
  isDark,
}: {
  output: PretrainingOutput
  isDark: boolean
}) {
  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.65fr)]">
      <MemoryBreakdownBar breakdown={output.memory} isDark={isDark} />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
        <div className="rounded-lg border border-border bg-background/25 p-4">
          <GpuUtilizationGauge breakdown={output.memory} isDark={isDark} />
        </div>

        <div className="grid gap-3">
          <Stat
            label="Effective GPUs"
            value={formatCount(output.effectiveNumGPUs)}
            sub="Used for time and cost estimates"
          />
          <Stat label="Minimum GPUs Needed" value={formatCount(output.minGPUsNeeded)} />
          <Stat
            label="Minimum VRAM Floor"
            value={formatMemory(output.minVRAMFloor)}
            sub="Largest block or embedding/head unit"
          />
          <Stat
            label="Maximum Micro-Batch"
            value={formatCount(output.maxMicroBatchSize)}
            sub="Sequences per GPU after model-state allocation"
          />
        </div>
      </div>
    </div>
  )
}

// Layer 2 — Performance & cost (pretraining): D.6 rows Training Time,
// Throughput, Global Batch Size, Batch Compute Multiplier, Total FLOPs stat,
// Compute Cost, Total Cost.
export function PretrainPerformanceBody({
  output,
}: {
  output: PretrainingOutput
  isDark: boolean
}) {
  return (
    <div className="space-y-4">
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
          label="Batch Compute Multiplier"
          value={formatMultiplier(output.batchEfficiency.computeMultiplier)}
          sub={`${formatCount(output.batchEfficiency.actualBatchTokens)} tok vs ${formatCount(output.batchEfficiency.criticalBatchTokens)} tok, ${formatBatchRelation(output.batchEfficiency.relation)}, ${formatFractionPercent(output.batchEfficiency.wastedComputeFraction)} wasted-compute fraction`}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat
          label="Total FLOPs"
          value={formatFLOPs(output.computeEstimate.totalFLOPs)}
          sub={`${formatFLOPs(output.computeEstimate.flopsPerToken)} per token`}
        />
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
        <Stat label="Total Cost" value={formatCost(output.cost.totalCost)} highlight />
      </div>
    </div>
  )
}

// Layer 3 — Parallelism (pretraining): D.6 rows ParallelismLayout mesh, Layout
// string, Recommendation strategyLabel, Pipeline Bubble, Inter-node Bandwidth,
// reasoning[] bullets.
export function ParallelismResultsBody({
  output,
  isDark,
}: {
  output: PretrainingOutput
  isDark: boolean
}) {
  return (
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

        <Stat
          label="Inter-node Bandwidth"
          value={
            Number.isFinite(output.interNodeBandwidthGBps)
              ? `${output.interNodeBandwidthGBps.toFixed(1)} GB/s`
              : "--"
          }
          sub={output.interNodeBandwidthLabel}
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
  )
}

// Layer 4 — Model architecture (pretraining): D.6 rows Model Parameters (+sub),
// Attention Overhead.
export function ArchitectureStatsBody({
  output,
}: {
  output: PretrainingOutput
  isDark: boolean
}) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      <Stat
        label="Model Parameters"
        value={formatParams(output.parameterCounts.total)}
        sub={formatPretrainingParameterSub(output)}
      />
      <Stat
        label="Attention Overhead"
        value={formatFractionPercent(output.attentionOverheadFraction)}
        sub="Quadratic attention FLOPs relative to model FLOPs"
      />
    </div>
  )
}

// Layer 6 — Data & scaling (pretraining): D.6 rows Chinchilla Ratio, Predicted
// Loss, Chinchilla Recommendation prose, Data Repetition callout.
export function DataScalingBody({
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
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <Stat
          label="Chinchilla Ratio"
          value={formatMultiplier(output.chinchilla.ratio)}
          sub={`20x basis ${formatParams(output.chinchilla.parameterCount)} params; power-law target ${formatCount(output.chinchilla.powerLawOptimalTokens)} tok`}
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
      </div>

      <div className="rounded-lg border border-border bg-background/25 p-4">
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted">
          Chinchilla Recommendation
        </div>
        <p className="mt-2 text-sm leading-6 text-foreground">
          {output.chinchilla.recommendation}
        </p>
      </div>

      {output.dataRepetition.hasRepetition && (
        <div
          className="rounded-2xl border px-4 py-3"
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
    </div>
  )
}

// Layer 7 — Cost detail & failures (pretraining): D.6 rows Storage Cost,
// Failure Overhead, Checkpoint Size, Checkpoint Storage.
export function CostDetailBody({
  output,
}: {
  output: PretrainingOutput
  isDark: boolean
}) {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
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
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
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
    </div>
  )
}

// Layer 8 — MoE (pretraining): D.6 rows MoE Sparsity stat, Sparsity Ratio,
// Efficiency Gain, Load Balance Factor. Renders null when MoE is not enabled.
export function MoEMetricsBody({
  output,
}: {
  output: PretrainingOutput
  isDark: boolean
}) {
  if (!output.moeSparsity) {
    return null
  }

  const moeSparsity = output.moeSparsity

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <Stat
          label="MoE Sparsity"
          value={formatFractionPercent(moeSparsity.sparsityRatio)}
          sub={`${formatMultiplier(moeSparsity.efficiencyGain)} memory efficiency`}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Stat
          label="Sparsity Ratio"
          value={formatFractionPercent(moeSparsity.sparsityRatio)}
        />
        <Stat
          label="Efficiency Gain"
          value={formatMultiplier(moeSparsity.efficiencyGain)}
        />
        <Stat
          label="Load Balance Factor"
          value={
            Number.isFinite(moeSparsity.loadBalanceFactor)
              ? moeSparsity.loadBalanceFactor.toFixed(2)
              : "--"
          }
        />
      </div>
    </div>
  )
}

// Layer 1 — Memory & feasibility (post-training): D.6 rows MemoryBreakdownBar,
// GpuUtilizationGauge, GPUs Needed (+gpuRequirementSub), Free Headroom, Working
// Set, plus the Memory Line Items block (PostTrainingMemoryItems promoted here).
export function PostMemoryBody({
  output,
  isDark,
}: {
  output: PostTrainingOutput
  isDark: boolean
}) {
  const hasMemoryItems = output.memory.items.some(
    (item) => Number.isFinite(item.bytes) && item.bytes > 0,
  )
  const gpuRequirementSub = output.memory.fits
    ? "Current GPU count fits memory"
    : output.numGPUsNeeded === null
      ? "No data-parallel fit found"
      : output.numGPUsNeededMode === "state-sharded-lower-bound"
        ? "Ideal state-sharded lower bound; full fit needs more headroom"
        : "Estimated data-parallel count to fit memory"

  return (
    <div className="space-y-5">
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
              sub={gpuRequirementSub}
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

      {hasMemoryItems && <PostTrainingMemoryItems output={output} isDark={isDark} />}
    </div>
  )
}

// Layer 2 — Performance & cost (post-training): D.6 rows Estimated Time,
// Throughput, stepTimeLabel stat, Failure Multiplier, Compute Cost, Total Cost.
export function PostPerformanceBody({
  output,
}: {
  output: PostTrainingOutput
  isDark: boolean
}) {
  const hasStorageOrFailureCost =
    output.cost.storageCost > 0 ||
    output.cost.failureOverheadCost > 0 ||
    output.cost.numCheckpoints > 0 ||
    output.cost.peakCheckpointStorage > 0 ||
    output.cost.datasetStorageBytes > 0

  return (
    <div className="space-y-4">
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
          sub={`${formatCount(output.trainingTime.totalSteps)} ${output.stepCountLabel}`}
        />
        <Stat
          label={output.stepTimeLabel}
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

      <div className="grid gap-4 sm:grid-cols-2">
        <Stat
          label="Compute Cost"
          value={formatCost(output.cost.computeCost)}
          sub={
            output.cost.actualComputeCost != null &&
            output.cost.actualComputeCost !== output.cost.computeCost
              ? `Actual compute ${formatCost(output.cost.actualComputeCost)}`
              : !hasStorageOrFailureCost
                ? "Checkpoint storage and failure recovery not modeled"
              : undefined
          }
        />
        <Stat label="Total Cost" value={formatCost(output.cost.totalCost)} highlight />
      </div>
    </div>
  )
}

// Layer 7 — Cost detail & failures (post-training): the conditional Storage
// Cost / Failure Overhead detail. Renders null when there is nothing to show.
export function PostCostDetailBody({
  output,
}: {
  output: PostTrainingOutput
  isDark: boolean
}) {
  const hasStorageOrFailureCost =
    output.cost.storageCost > 0 ||
    output.cost.failureOverheadCost > 0 ||
    output.cost.numCheckpoints > 0 ||
    output.cost.peakCheckpointStorage > 0 ||
    output.cost.datasetStorageBytes > 0

  if (!hasStorageOrFailureCost) {
    return null
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2">
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
    </div>
  )
}
