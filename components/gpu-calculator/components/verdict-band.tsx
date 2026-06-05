"use client"

import type { ReactNode } from "react"
import { AlertCircle, AlertTriangle, Check, Wrench } from "lucide-react"
import type { CalculatorOutput, Warning } from "../types"
import { formatCost, formatCount, formatDuration, formatMemory } from "../formatters"
import GpuUtilizationGauge from "./gpu-utilization-gauge"
import { SEVERITY_META, isPretraining } from "./results-summary"

export type VerdictTone = "ok" | "warning" | "critical"

export interface VerdictBandProps {
  /** The active tab's output object (pretraining or post-training union). */
  output: CalculatorOutput
  /** Drives Tailwind semantic classes + indexes into tone maps, like the result cards. */
  isDark: boolean
  /**
   * One-tap remedy when memory does not fit. Wired by the host to set
   * numGPUs := minGPUsNeeded (pretraining) or numGPUsNeeded (post-training).
   * Omitted ⇒ the "Fix for me" affordance is not rendered (e.g. fit unknown).
   */
  onFixForMe?: () => void
  /** Critical warnings hoisted from output.warnings by the host (severity === "critical"). */
  criticalWarnings: Warning[]
  /** Total count of silent substitutions, drives the "N auto-adjustments ▸" chip. */
  adjustmentCount: number
  /** Opens / scrolls to the AssumptionsLedger. */
  onShowLedger: () => void
  /**
   * Display-only GPU label for the active tab's config. The output union does
   * not carry the GPU name, so the host feeds it from hardware.gpu.name.
   */
  gpuName: string
  /**
   * Display-only configured GPU count for the active tab. PostTrainingOutput
   * does not carry it, so the host feeds it from hardware.numGPUs.
   */
  configuredNumGPUs: number
  /**
   * Display-only: true when the pretraining #GPUs is derived from target-days
   * (the field is locked). Adds a sub-line to the Fix-for-me button so the
   * side effect of clearing target-days is never silent. Post-training: false.
   */
  gpuCountDerivedFromTarget?: boolean
}

function toneStyles(tone: VerdictTone, isDark: boolean) {
  const mode = isDark ? "dark" : "light"
  if (tone === "ok") {
    // Accent/teal lane — uses the result-card semantic tokens.
    return {
      bg: "var(--accent-soft)",
      border: "var(--border)",
      text: "var(--accent)",
    }
  }
  const meta = tone === "critical" ? SEVERITY_META.critical : SEVERITY_META.warning
  return {
    bg: meta[mode].bg,
    border: meta[mode].border,
    text: meta[mode].text,
  }
}

/** A single cost/time/count figure that must never wrap mid-digit. */
function Figure({ children }: { children: ReactNode }) {
  return (
    <span className="whitespace-nowrap font-mono font-semibold tabular-nums">{children}</span>
  )
}

function Dot() {
  return (
    <span aria-hidden className="select-none px-1 text-muted">
      ·
    </span>
  )
}

export default function VerdictBand({
  output,
  isDark,
  onFixForMe,
  criticalWarnings,
  adjustmentCount,
  onShowLedger,
  gpuName,
  configuredNumGPUs,
  gpuCountDerivedFromTarget = false,
}: VerdictBandProps) {
  const fits = output.memory.fits
  const hasNonMemoryCritical = criticalWarnings.some((w) => w.category !== "memory")
  const tone: VerdictTone = !fits ? (hasNonMemoryCritical ? "critical" : "warning") : "ok"
  const tint = toneStyles(tone, isDark)

  const cost = formatCost(output.cost.totalCost)
  const time = formatDuration(output.trainingTime.theoreticalHours)

  return (
    <div
      className="sticky top-0 z-30 border-b backdrop-blur-xl"
      style={{ backgroundColor: tint.bg, borderColor: tint.border }}
    >
      <div className="px-4 py-3 sm:px-6">
        {/* Critical warnings — assertive, above the verdict line. */}
        {criticalWarnings.length > 0 && (
          <div role="alert" className="mb-3 space-y-1.5">
            {criticalWarnings.map((warning, index) => (
              <div
                key={`${warning.category}-${index}`}
                className="flex items-start gap-2 text-sm leading-6"
                style={{ color: SEVERITY_META.critical[isDark ? "dark" : "light"].text }}
              >
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>
                  <span className="font-semibold">Critical · </span>
                  {warning.message}
                </span>
              </div>
            ))}
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
          {/* Verdict line + mini gauge. */}
          <div className="flex min-w-0 items-center gap-4">
            <div className="shrink-0">
              <GpuUtilizationGauge breakdown={output.memory} isDark={isDark} size="sm" />
            </div>

            <div className="min-w-0">
              {fits ? (
                <FitsVerdict
                  cost={cost}
                  time={time}
                  gpuCount={fitsGpuCount(output, configuredNumGPUs)}
                  gpuName={gpuName}
                  tint={tint}
                />
              ) : (
                <OverBudgetVerdict
                  output={output}
                  gpuName={gpuName}
                  tint={tint}
                  onFixForMe={onFixForMe}
                  gpuCountDerivedFromTarget={gpuCountDerivedFromTarget}
                />
              )}
            </div>
          </div>

          {/* Auto-adjustments ledger chip. */}
          {adjustmentCount > 0 && (
            <button
              type="button"
              onClick={onShowLedger}
              className="shrink-0 rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-muted transition-colors hover:text-foreground"
            >
              {adjustmentCount} auto-adjustment{adjustmentCount === 1 ? "" : "s"} ▸
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/** The configured-count display for the "fits" line, per the per-tab field. */
function fitsGpuCount(output: CalculatorOutput, configuredNumGPUs: number): string {
  // Pretraining quotes effectiveNumGPUs (an output field); post-training has no
  // such field, so the host-fed configured count is used.
  if (isPretraining(output)) {
    return formatCount(output.effectiveNumGPUs)
  }
  return formatCount(configuredNumGPUs)
}

function FitsVerdict({
  cost,
  time,
  gpuCount,
  gpuName,
  tint,
}: {
  cost: string
  time: string
  gpuCount: string
  gpuName: string
  tint: { text: string }
}) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="true"
      className="flex flex-wrap items-center gap-x-1 gap-y-1 text-sm sm:text-base"
      style={{ color: tint.text }}
    >
      <span className="flex items-center gap-1.5 font-semibold">
        <Check className="h-4 w-4 shrink-0" />
        Fits
      </span>
      <Dot />
      <Figure>{cost}</Figure>
      <Dot />
      <Figure>{time}</Figure>
      <Dot />
      <span className="whitespace-nowrap font-medium text-foreground">
        <Figure>{gpuCount}×</Figure> {gpuName}
      </span>
    </div>
  )
}

function OverBudgetVerdict({
  output,
  gpuName,
  tint,
  onFixForMe,
  gpuCountDerivedFromTarget,
}: {
  output: CalculatorOutput
  gpuName: string
  tint: { text: string }
  onFixForMe?: () => void
  gpuCountDerivedFromTarget: boolean
}) {
  const perGPU = formatMemory(output.memory.total)
  const usable = formatMemory(output.memory.usableCapacity)

  // ── Pretraining ──
  if (isPretraining(output)) {
    const needed = output.minGPUsNeeded
    const have = output.effectiveNumGPUs
    // A one-number fix exists only when more GPUs would help.
    const hasFix = onFixForMe != null && Number.isFinite(needed) && needed > have

    return (
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="space-y-1.5"
        style={{ color: tint.text }}
      >
        {hasFix ? (
          <>
            <p className="flex items-start gap-1.5 text-sm leading-6 sm:text-base">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>
                Doesn&apos;t fit yet — needs ~<Figure>{formatCount(needed)}×</Figure> {gpuName}{" "}
                (you have <Figure>{formatCount(have)}</Figure>).
              </span>
            </p>
            <p className="text-xs leading-5 text-muted">
              This config wants <Figure>{perGPU}</Figure> per GPU but only <Figure>{usable}</Figure>{" "}
              is usable.
            </p>
            <FixButton
              label={`Fix for me → ${formatCount(needed)} GPUs`}
              subline={gpuCountDerivedFromTarget ? "clears your target-days setting" : undefined}
              onClick={onFixForMe}
              tint={tint}
            />
          </>
        ) : (
          // No single-number fix (already at/above the floor but still over).
          <p className="flex items-start gap-1.5 text-sm leading-6 sm:text-base">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>
              Doesn&apos;t fit on a single {gpuName} yet — try a larger-memory GPU or add tensor
              parallelism.
            </span>
          </p>
        )}
      </div>
    )
  }

  // ── Post-training ──
  const needed = output.numGPUsNeeded
  const mode = output.numGPUsNeededMode

  let body: ReactNode
  if (needed !== null && mode === "data-parallel") {
    body = (
      <>
        <p className="flex items-start gap-1.5 text-sm leading-6 sm:text-base">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            Doesn&apos;t fit yet — split the work over ~<Figure>{formatCount(needed)}</Figure>{" "}
            data-parallel {gpuName}s.
          </span>
        </p>
        {onFixForMe != null && (
          <FixButton
            label={`Fix for me → ${formatCount(needed)} GPUs`}
            onClick={onFixForMe}
            tint={tint}
          />
        )}
      </>
    )
  } else if (needed !== null && mode === "state-sharded-lower-bound") {
    body = (
      <p className="flex items-start gap-1.5 text-sm leading-6 sm:text-base">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
        <span>
          Doesn&apos;t fit yet — even fully sharded (ZeRO-3/FSDP) the model states need at least ~
          <Figure>{formatCount(needed)}</Figure> {gpuName}s, before activations and overhead. Try
          more GPUs, a smaller base model, or QLoRA.
        </span>
      </p>
    )
  } else {
    body = (
      <p className="flex items-start gap-1.5 text-sm leading-6 sm:text-base">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
        <span>
          Doesn&apos;t fit yet on {gpuName}. Try fewer trainable params (LoRA/QLoRA), a smaller base
          model, or more GPUs.
        </span>
      </p>
    )
  }

  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="true"
      className="space-y-1.5"
      style={{ color: tint.text }}
    >
      {body}
    </div>
  )
}

function FixButton({
  label,
  subline,
  onClick,
  tint,
}: {
  label: string
  subline?: string
  onClick?: () => void
  tint: { text: string; border?: string }
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mt-1 inline-flex flex-col items-start rounded-md border border-current/40 bg-surface/60 px-3 py-1.5 text-left transition-colors hover:bg-surface"
      style={{ color: tint.text }}
    >
      <span className="flex items-center gap-1.5 whitespace-nowrap text-sm font-semibold tabular-nums">
        <Wrench className="h-3.5 w-3.5 shrink-0" />
        {label}
      </span>
      {subline && <span className="text-[11px] font-normal opacity-80">{subline}</span>}
    </button>
  )
}
