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
  Gauge,
  Grid3X3,
  Info,
  Layers,
  Zap,
} from "lucide-react"
import type {
  CalculatorTab,
  PostTrainingOutput,
  PretrainingOutput,
  Warning,
} from "../types"
import MemoryBreakdownBar from "./memory-breakdown-bar"
import GpuUtilizationGauge from "./gpu-utilization-gauge"
import ParallelismLayout from "./parallelism-layout"

// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------

function fmtBytes(bytes: number): string {
  const gb = bytes / 1024 ** 3
  if (gb >= 1000) return `${(gb / 1024).toFixed(1)} TB`
  if (gb >= 100) return `${Math.round(gb)} GB`
  if (gb >= 10) return `${gb.toFixed(1)} GB`
  if (gb >= 0.1) return `${gb.toFixed(2)} GB`
  const mb = bytes / 1024 ** 2
  if (mb >= 1) return `${mb.toFixed(0)} MB`
  return "< 1 MB"
}

function fmtParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return n.toLocaleString()
}

function fmtFLOPs(flops: number): string {
  if (flops >= 1e21) return `${(flops / 1e21).toFixed(1)} ZFLOPs`
  if (flops >= 1e18) return `${(flops / 1e18).toFixed(1)} EFLOPs`
  if (flops >= 1e15) return `${(flops / 1e15).toFixed(1)} PFLOPs`
  if (flops >= 1e12) return `${(flops / 1e12).toFixed(1)} TFLOPs`
  return `${(flops / 1e9).toFixed(1)} GFLOPs`
}

function fmtCost(dollars: number): string {
  if (dollars >= 1e6) return `$${(dollars / 1e6).toFixed(2)}M`
  if (dollars >= 1e3) return `$${Math.round(dollars).toLocaleString()}`
  return `$${dollars.toFixed(2)}`
}

function fmtTime(hours: number): string {
  if (hours >= 24 * 365) return `${(hours / (24 * 365)).toFixed(1)} years`
  if (hours >= 48) return `${(hours / 24).toFixed(1)} days`
  if (hours >= 1) return `${hours.toFixed(1)} hours`
  return `${Math.round(hours * 60)} min`
}

function fmtNum(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return n.toLocaleString()
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`
}

// ---------------------------------------------------------------------------
// Type guard
// ---------------------------------------------------------------------------

function isPretraining(
  output: PretrainingOutput | PostTrainingOutput,
): output is PretrainingOutput {
  return "parameterCounts" in output
}

// ---------------------------------------------------------------------------
// Shared sub-components
// ---------------------------------------------------------------------------

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
    <motion.div
      className={`rounded-2xl border border-border bg-surface-elevated/50 p-5 backdrop-blur-sm ${className ?? ""}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="mb-4 flex items-center gap-2">
        {Icon && <Icon className="h-3.5 w-3.5 text-accent" />}
        <h4 className="text-xs font-medium uppercase tracking-[0.2em] text-muted">
          {title}
        </h4>
      </div>
      {children}
    </motion.div>
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
    <div className={highlight ? "rounded-xl bg-accent-soft/40 p-3" : ""}>
      <div className="text-[10px] uppercase tracking-[0.18em] text-muted">{label}</div>
      <div
        className={`mt-1 font-mono text-xl font-semibold tabular-nums ${
          highlight ? "text-accent" : "text-foreground"
        }`}
      >
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-muted">{sub}</div>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Severity palette
// ---------------------------------------------------------------------------

const SEVERITY_ICON = {
  info: Info,
  warning: AlertTriangle,
  critical: AlertCircle,
} as const

const SEVERITY_COLORS = {
  info: {
    light: {
      bg: "oklch(0.96 0.03 224)",
      border: "oklch(0.86 0.06 224)",
      text: "oklch(0.42 0.12 224)",
    },
    dark: {
      bg: "oklch(0.20 0.03 224)",
      border: "oklch(0.32 0.06 224)",
      text: "oklch(0.76 0.10 224)",
    },
  },
  warning: {
    light: {
      bg: "oklch(0.97 0.04 85)",
      border: "oklch(0.90 0.08 85)",
      text: "oklch(0.46 0.12 85)",
    },
    dark: {
      bg: "oklch(0.22 0.04 85)",
      border: "oklch(0.35 0.06 85)",
      text: "oklch(0.80 0.10 85)",
    },
  },
  critical: {
    light: {
      bg: "oklch(0.96 0.04 25)",
      border: "oklch(0.86 0.10 25)",
      text: "oklch(0.44 0.18 25)",
    },
    dark: {
      bg: "oklch(0.20 0.06 25)",
      border: "oklch(0.35 0.10 25)",
      text: "oklch(0.76 0.16 25)",
    },
  },
} as const

function WarningsPanel({
  warnings,
  isDark,
}: {
  warnings: Warning[]
  isDark: boolean
}) {
  const sorted = [...warnings].sort((a, b) => {
    const order = { critical: 0, warning: 1, info: 2 }
    return order[a.severity] - order[b.severity]
  })
  const mode = isDark ? "dark" : "light"

  return (
    <ResultCard title="Warnings" icon={AlertTriangle}>
      <div className="space-y-2">
        {sorted.map((w, i) => {
          const Icon = SEVERITY_ICON[w.severity]
          const c = SEVERITY_COLORS[w.severity][mode]
          return (
            <motion.div
              key={`${w.severity}-${w.category}-${i}`}
              className="flex items-start gap-2.5 rounded-xl border px-3.5 py-2.5 text-sm"
              style={{
                backgroundColor: c.bg,
                borderColor: c.border,
                color: c.text,
              }}
              initial={{ opacity: 0, x: -4 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04, duration: 0.25 }}
            >
              <Icon className="mt-0.5 h-4 w-4 shrink-0" />
              <div>
                <span className="text-[10px] font-medium uppercase tracking-wider opacity-70">
                  {w.category}
                </span>
                <p className="mt-0.5">{w.message}</p>
              </div>
            </motion.div>
          )
        })}
      </div>
    </ResultCard>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface Props {
  output: PretrainingOutput | PostTrainingOutput
  activeTab: CalculatorTab
  isDark: boolean
}

export default function ResultsSummary({ output, activeTab, isDark }: Props) {
  if (isPretraining(output)) {
    return <PretrainingResults output={output} isDark={isDark} />
  }
  return <PostTrainingResults output={output} isDark={isDark} />
}

// ---------------------------------------------------------------------------
// Pretraining — all 19 output items from spec Section 11.2
// ---------------------------------------------------------------------------

function PretrainingResults({
  output: o,
  isDark,
}: {
  output: PretrainingOutput
  isDark: boolean
}) {
  const repSeverity =
    o.dataRepetition.severity === "none" ? "info" : o.dataRepetition.severity
  const repMode = isDark ? "dark" : "light"

  return (
    <div className="space-y-4">
      {/* ── 4. Hero: Memory Breakdown ────────────────────────────── */}
      <ResultCard title="Memory Breakdown" icon={BarChart3}>
        <MemoryBreakdownBar breakdown={o.memory} isDark={isDark} />
      </ResultCard>

      {/* ── Utilisation + Requirements ───────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-2">
        <ResultCard title="GPU Utilization" icon={Gauge}>
          <GpuUtilizationGauge breakdown={o.memory} isDark={isDark} />
        </ResultCard>

        <ResultCard title="GPU Requirements" icon={Cpu}>
          <div className="space-y-4">
            {/* 5 */}
            <Stat
              label="Min GPUs Needed"
              value={o.minGPUsNeeded.toLocaleString()}
            />
            {/* 6 */}
            <Stat
              label="Min VRAM Floor"
              value={fmtBytes(o.minVRAMFloor)}
              sub="Largest transformer block"
            />
            {/* 16 */}
            <Stat
              label="Max Micro-Batch"
              value={o.maxMicroBatchSize.toLocaleString()}
              sub="Sequences per GPU"
            />
          </div>
        </ResultCard>
      </div>

      {/* ── 1-3, 14-15, 18. Model & Compute ─────────────────────── */}
      <ResultCard title="Model & Compute" icon={Zap}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {/* 1 */}
          <Stat
            label="Parameters"
            value={fmtParams(o.parameterCounts.total)}
            sub={
              o.parameterCounts.active !== o.parameterCounts.total
                ? `${fmtParams(o.parameterCounts.active)} active`
                : undefined
            }
          />
          {/* 2 */}
          <Stat label="Total FLOPs" value={fmtFLOPs(o.computeEstimate.totalFLOPs)} />
          {/* 3 */}
          <Stat
            label="Chinchilla Ratio"
            value={`${o.chinchilla.ratio.toFixed(1)}\u00d7`}
            sub={o.chinchilla.recommendation}
          />
          {/* 14 */}
          <Stat label="Attention Overhead" value={pct(o.attentionOverheadFraction)} />
          {/* 15 */}
          <Stat
            label="Predicted Loss"
            value={`${o.predictedLossNats.toFixed(3)} nats`}
          />
          {/* 18 (conditional) */}
          {o.moeSparsity && (
            <Stat
              label="MoE Sparsity"
              value={pct(o.moeSparsity.sparsityRatio)}
              sub={`${o.moeSparsity.efficiencyGain.toFixed(1)}\u00d7 efficiency`}
            />
          )}
        </div>
      </ResultCard>

      {/* ── 7-8. Parallelism ─────────────────────────────────────── */}
      <ResultCard title="Parallelism Strategy" icon={Grid3X3}>
        <p className="mb-4 text-sm text-foreground">
          {o.parallelismRecommendation.strategyLabel}
        </p>
        <ParallelismLayout
          config={o.parallelismRecommendation.config}
          totalGPUs={o.minGPUsNeeded}
          isDark={isDark}
        />
        {/* 8 */}
        {o.pipelineBubbleFraction > 0 && (
          <div className="mt-4">
            <Stat
              label="Pipeline Bubble"
              value={pct(o.pipelineBubbleFraction)}
            />
          </div>
        )}
        {o.parallelismRecommendation.reasoning.length > 0 && (
          <ul className="mt-3 space-y-1 text-xs text-muted">
            {o.parallelismRecommendation.reasoning.map((r, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="mt-1.5 inline-block h-1 w-1 shrink-0 rounded-full bg-accent" />
                {r}
              </li>
            ))}
          </ul>
        )}
      </ResultCard>

      {/* ── 9-10, 12, 17, 19. Training Performance ──────────────── */}
      <ResultCard title="Training Performance" icon={Clock}>
        <div className="grid gap-4 sm:grid-cols-2">
          {/* 9 */}
          <Stat
            label="Training Time"
            value={fmtTime(o.trainingTime.theoreticalHours)}
            sub={
              o.trainingTime.failureAdjustedHours != null
                ? `${fmtTime(o.trainingTime.failureAdjustedHours)} failure-adjusted`
                : undefined
            }
          />
          {/* 10 */}
          <Stat
            label="Throughput"
            value={`${fmtNum(o.tokensPerSecond)} tok/s`}
          />
          {/* 12 */}
          <Stat
            label="Global Batch Size"
            value={`${o.globalBatchSize.sequences.toLocaleString()} seq`}
            sub={`${fmtNum(o.globalBatchSize.tokens)} tokens`}
          />
          {/* 19 */}
          <Stat
            label="Batch Efficiency"
            value={`${o.batchEfficiency.computeMultiplier.toFixed(1)}\u00d7`}
            sub={`vs B_crit \u00b7 ${pct(o.batchEfficiency.wastedComputeFraction)} overhead`}
          />
        </div>

        {/* 17. Data repetition */}
        {o.dataRepetition.hasRepetition && (
          <div
            className="mt-4 rounded-xl border px-4 py-3 text-sm"
            style={{
              backgroundColor: SEVERITY_COLORS[repSeverity][repMode].bg,
              borderColor: SEVERITY_COLORS[repSeverity][repMode].border,
              color: SEVERITY_COLORS[repSeverity][repMode].text,
            }}
          >
            <div className="font-medium">
              Data Repetition: {o.dataRepetition.epochs.toFixed(1)} epochs
            </div>
            <p className="mt-1 text-xs opacity-80">
              {o.dataRepetition.recommendation}
            </p>
          </div>
        )}
      </ResultCard>

      {/* ── 11, 13. Cost ─────────────────────────────────────────── */}
      <ResultCard title="Cost Estimate" icon={DollarSign}>
        <div className="grid gap-4 sm:grid-cols-2">
          <Stat label="Compute Cost" value={fmtCost(o.cost.computeCost)} />
          <Stat label="Storage Cost" value={fmtCost(o.cost.storageCost)} />
          <Stat
            label="Failure Overhead"
            value={fmtCost(o.cost.failureOverheadCost)}
          />
          <Stat label="Total Cost" value={fmtCost(o.cost.totalCost)} highlight />
        </div>
        {/* 13 */}
        <div className="mt-4">
          <Stat label="Checkpoint Size" value={fmtBytes(o.checkpointSize)} />
        </div>
      </ResultCard>

      {/* ── 18. MoE detail (conditional) ─────────────────────────── */}
      {o.moeSparsity && (
        <ResultCard title="MoE Metrics" icon={Layers}>
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat
              label="Sparsity Ratio"
              value={pct(o.moeSparsity.sparsityRatio)}
            />
            <Stat
              label="Efficiency Gain"
              value={`${o.moeSparsity.efficiencyGain.toFixed(1)}\u00d7`}
            />
            <Stat
              label="Load Balance Factor"
              value={o.moeSparsity.loadBalanceFactor.toFixed(2)}
            />
          </div>
        </ResultCard>
      )}

      {/* ── Warnings ─────────────────────────────────────────────── */}
      {o.warnings.length > 0 && (
        <WarningsPanel warnings={o.warnings} isDark={isDark} />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Post-Training
// ---------------------------------------------------------------------------

function PostTrainingResults({
  output: o,
  isDark,
}: {
  output: PostTrainingOutput
  isDark: boolean
}) {
  return (
    <div className="space-y-4">
      {/* Memory */}
      <ResultCard title="Memory Breakdown" icon={BarChart3}>
        <MemoryBreakdownBar breakdown={o.memory} isDark={isDark} />
      </ResultCard>

      <div className="grid gap-4 sm:grid-cols-2">
        <ResultCard title="GPU Utilization" icon={Gauge}>
          <GpuUtilizationGauge breakdown={o.memory} isDark={isDark} />
        </ResultCard>

        <ResultCard title="Requirements" icon={Cpu}>
          <Stat
            label="GPUs Needed"
            value={o.numGPUsNeeded.toLocaleString()}
          />
        </ResultCard>
      </div>

      {/* Training Time */}
      <ResultCard title="Training Time" icon={Clock}>
        <div className="grid gap-4 sm:grid-cols-2">
          <Stat
            label="Estimated Time"
            value={fmtTime(o.trainingTime.theoreticalHours)}
            sub={
              o.trainingTime.failureAdjustedHours != null
                ? `${fmtTime(o.trainingTime.failureAdjustedHours)} failure-adjusted`
                : undefined
            }
          />
          <Stat
            label="Throughput"
            value={`${fmtNum(o.trainingTime.tokensPerSecond)} tok/s`}
          />
        </div>
      </ResultCard>

      {/* Cost */}
      <ResultCard title="Cost Estimate" icon={DollarSign}>
        <div className="grid gap-4 sm:grid-cols-2">
          <Stat label="Compute Cost" value={fmtCost(o.cost.computeCost)} />
          <Stat label="Storage Cost" value={fmtCost(o.cost.storageCost)} />
          <Stat
            label="Failure Overhead"
            value={fmtCost(o.cost.failureOverheadCost)}
          />
          <Stat label="Total Cost" value={fmtCost(o.cost.totalCost)} highlight />
        </div>
      </ResultCard>

      {/* Warnings */}
      {o.warnings.length > 0 && (
        <WarningsPanel warnings={o.warnings} isDark={isDark} />
      )}
    </div>
  )
}
