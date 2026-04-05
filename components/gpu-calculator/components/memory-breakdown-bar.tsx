"use client"

import { useMemo, useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { AlertTriangle } from "lucide-react"
import type { MemoryBreakdown, PostTrainingMemoryBreakdown } from "../types"

// ---------------------------------------------------------------------------
// Visualization constants
// ---------------------------------------------------------------------------

const VIEW_W = 1000
const VIEW_H = 100

const SEGMENT_META: Record<string, { label: string; light: string; dark: string }> = {
  parameters:           { label: "Parameters",      light: "oklch(0.56 0.19 260)", dark: "oklch(0.67 0.17 260)" },
  gradients:            { label: "Gradients",        light: "oklch(0.55 0.21 310)", dark: "oklch(0.68 0.18 310)" },
  optimizerStates:      { label: "Optimizer States", light: "oklch(0.70 0.17 55)",  dark: "oklch(0.76 0.14 55)"  },
  activations:          { label: "Activations",      light: "oklch(0.60 0.14 170)", dark: "oklch(0.72 0.12 170)" },
  communicationBuffers: { label: "Comm Buffers",     light: "oklch(0.62 0.17 15)",  dark: "oklch(0.72 0.15 15)"  },
  frameworkOverhead:    { label: "Overhead",          light: "oklch(0.54 0.02 260)", dark: "oklch(0.48 0.02 260)" },
  freeHeadroom:         { label: "Free",             light: "oklch(0.94 0.005 260)",dark: "oklch(0.25 0.01 260)" },
  trainableModels:      { label: "Trainable Models", light: "oklch(0.56 0.19 260)", dark: "oklch(0.67 0.17 260)" },
  frozenModels:         { label: "Frozen Models",    light: "oklch(0.62 0.06 260)", dark: "oklch(0.50 0.06 260)" },
  loraAdapter:          { label: "LoRA Adapter",     light: "oklch(0.55 0.21 310)", dark: "oklch(0.68 0.18 310)" },
  ppoBuffers:           { label: "PPO Buffers",      light: "oklch(0.62 0.17 15)",  dark: "oklch(0.72 0.15 15)"  },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatGB(bytes: number): string {
  const gb = bytes / 1024 ** 3
  if (gb >= 1000) return `${(gb / 1024).toFixed(1)} TB`
  if (gb >= 100) return `${Math.round(gb)} GB`
  if (gb >= 10) return `${gb.toFixed(1)} GB`
  if (gb >= 0.1) return `${gb.toFixed(2)} GB`
  const mb = bytes / 1024 ** 2
  if (mb >= 1) return `${mb.toFixed(0)} MB`
  return "< 1 MB"
}

function isPostTraining(b: MemoryBreakdown): b is PostTrainingMemoryBreakdown {
  return "trainableModels" in b
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Segment {
  key: string
  label: string
  bytes: number
  color: string
  x: number
  w: number
  pct: number
}

interface Props {
  breakdown: MemoryBreakdown
  isDark: boolean
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MemoryBreakdownBar({ breakdown, isDark }: Props) {
  const [hovered, setHovered] = useState<string | null>(null)

  const { segments, capacityX } = useMemo(() => {
    if (breakdown.gpuCapacity === 0) return { segments: [] as Segment[], capacityX: VIEW_W }

    const mode = isDark ? "dark" : "light"
    const raw: { key: string; bytes: number }[] = []

    if (isPostTraining(breakdown)) {
      if (breakdown.trainableModels > 0) raw.push({ key: "trainableModels", bytes: breakdown.trainableModels })
      if (breakdown.frozenModels > 0) raw.push({ key: "frozenModels", bytes: breakdown.frozenModels })
      if (breakdown.loraAdapter > 0) raw.push({ key: "loraAdapter", bytes: breakdown.loraAdapter })
      if (breakdown.activations > 0) raw.push({ key: "activations", bytes: breakdown.activations })
      if (breakdown.ppoBuffers > 0) raw.push({ key: "ppoBuffers", bytes: breakdown.ppoBuffers })
      if (breakdown.communicationBuffers > 0) raw.push({ key: "communicationBuffers", bytes: breakdown.communicationBuffers })
      if (breakdown.frameworkOverhead > 0) raw.push({ key: "frameworkOverhead", bytes: breakdown.frameworkOverhead })
    } else {
      if (breakdown.parameters > 0) raw.push({ key: "parameters", bytes: breakdown.parameters })
      if (breakdown.gradients > 0) raw.push({ key: "gradients", bytes: breakdown.gradients })
      if (breakdown.optimizerStates > 0) raw.push({ key: "optimizerStates", bytes: breakdown.optimizerStates })
      if (breakdown.activations > 0) raw.push({ key: "activations", bytes: breakdown.activations })
      if (breakdown.communicationBuffers > 0) raw.push({ key: "communicationBuffers", bytes: breakdown.communicationBuffers })
      if (breakdown.frameworkOverhead > 0) raw.push({ key: "frameworkOverhead", bytes: breakdown.frameworkOverhead })
    }

    if (breakdown.freeHeadroom > 0) {
      raw.push({ key: "freeHeadroom", bytes: breakdown.freeHeadroom })
    }

    const scale = breakdown.fits ? breakdown.gpuCapacity : breakdown.total
    let cx = 0
    const segs: Segment[] = raw.map(({ key, bytes }) => {
      const meta = SEGMENT_META[key]
      const w = (bytes / scale) * VIEW_W
      const seg: Segment = {
        key,
        label: meta?.label ?? key,
        bytes,
        color: meta?.[mode] ?? "#888",
        x: cx,
        w,
        pct: (bytes / breakdown.gpuCapacity) * 100,
      }
      cx += w
      return seg
    })

    const capX = breakdown.fits ? VIEW_W : (breakdown.gpuCapacity / scale) * VIEW_W
    return { segments: segs, capacityX: capX }
  }, [breakdown, isDark])

  const hoveredSeg = segments.find((s) => s.key === hovered)
  const usagePct =
    breakdown.gpuCapacity > 0 ? (breakdown.total / breakdown.gpuCapacity) * 100 : 0
  const tooltipLeft = hoveredSeg
    ? Math.max(8, Math.min(92, ((hoveredSeg.x + hoveredSeg.w / 2) / VIEW_W) * 100))
    : 0

  if (breakdown.gpuCapacity === 0) {
    return <p className="text-sm text-muted">No memory data available.</p>
  }

  return (
    <div className="space-y-3">
      {/* ---- Header ---- */}
      <div className="flex items-center justify-between gap-3">
        <h4 className="text-xs font-medium uppercase tracking-[0.2em] text-muted">
          GPU Memory Breakdown
        </h4>
        {!breakdown.fits && (
          <motion.div
            className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
            style={{
              backgroundColor: isDark ? "oklch(0.22 0.08 25)" : "oklch(0.96 0.03 25)",
              color: isDark ? "oklch(0.78 0.18 25)" : "oklch(0.48 0.20 25)",
            }}
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <AlertTriangle className="h-3 w-3" />
            Exceeds Capacity
          </motion.div>
        )}
      </div>

      {/* ---- Bar ---- */}
      <div className="relative">
        <motion.div
          className="overflow-hidden rounded-xl"
          style={{
            boxShadow: !breakdown.fits
              ? isDark
                ? "inset 0 0 24px oklch(0.30 0.12 25 / 0.25)"
                : "inset 0 0 24px oklch(0.70 0.08 25 / 0.15)"
              : "none",
          }}
          initial={{ clipPath: "inset(0 100% 0 0)" }}
          animate={{ clipPath: "inset(0 0% 0 0)" }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        >
          <svg
            viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
            preserveAspectRatio="none"
            className="block w-full"
            style={{ height: "2.75rem" }}
            role="img"
            aria-label={`GPU memory: ${formatGB(breakdown.total)} of ${formatGB(breakdown.gpuCapacity)} used`}
          >
            <title>GPU Memory Breakdown</title>

            {/* Track */}
            <rect
              x={0}
              y={0}
              width={VIEW_W}
              height={VIEW_H}
              fill={isDark ? "oklch(0.18 0.01 260)" : "oklch(0.96 0.004 260)"}
            />

            {/* Segments */}
            {segments.map((seg) => (
              <motion.rect
                key={seg.key}
                y={0}
                height={VIEW_H}
                fill={seg.color}
                animate={{
                  x: seg.x,
                  width: Math.max(seg.w, 0),
                  opacity: hovered !== null && hovered !== seg.key ? 0.35 : 1,
                }}
                transition={{
                  x: { duration: 0.5, ease: "easeInOut" },
                  width: { duration: 0.5, ease: "easeInOut" },
                  opacity: { duration: 0.15 },
                }}
                onMouseEnter={() => setHovered(seg.key)}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: seg.key === "freeHeadroom" ? "default" : "pointer" }}
              />
            ))}

            {/* Capacity marker on overflow */}
            {!breakdown.fits && (
              <motion.line
                x1={capacityX}
                y1={0}
                x2={capacityX}
                y2={VIEW_H}
                stroke={isDark ? "oklch(0.65 0.22 25)" : "oklch(0.52 0.22 25)"}
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
                strokeDasharray="6 3"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              />
            )}
          </svg>
        </motion.div>

        {/* Tooltip */}
        <AnimatePresence>
          {hoveredSeg && hoveredSeg.key !== "freeHeadroom" && (
            <motion.div
              key={hoveredSeg.key}
              className="pointer-events-none absolute z-10"
              style={{
                left: `${tooltipLeft}%`,
                top: "-0.25rem",
                transform: "translateX(-50%) translateY(-100%)",
              }}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 4 }}
              transition={{ duration: 0.12 }}
            >
              <div className="whitespace-nowrap rounded-lg border border-border bg-surface-elevated px-3 py-2 text-xs shadow-lg backdrop-blur-md">
                <div className="flex items-center gap-2">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-sm"
                    style={{ backgroundColor: hoveredSeg.color }}
                  />
                  <span className="font-medium text-foreground">{hoveredSeg.label}</span>
                </div>
                <div className="mt-1 flex items-center gap-1.5 pl-[18px]">
                  <span className="font-mono tabular-nums text-foreground">
                    {formatGB(hoveredSeg.bytes)}
                  </span>
                  <span className="text-muted">·</span>
                  <span className="tabular-nums text-muted">{hoveredSeg.pct.toFixed(1)}%</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ---- Usage summary ---- */}
      <div className="flex items-baseline justify-between text-sm">
        <span className="text-muted">
          {formatGB(breakdown.total)} of {formatGB(breakdown.gpuCapacity)}
        </span>
        <span
          className="font-mono font-medium tabular-nums"
          style={{
            color:
              usagePct > 100
                ? isDark
                  ? "oklch(0.70 0.22 25)"
                  : "oklch(0.50 0.22 25)"
                : usagePct > 90
                  ? isDark
                    ? "oklch(0.72 0.18 25)"
                    : "oklch(0.52 0.18 25)"
                  : usagePct > 70
                    ? isDark
                      ? "oklch(0.78 0.14 85)"
                      : "oklch(0.60 0.14 85)"
                    : isDark
                      ? "oklch(0.72 0.14 155)"
                      : "oklch(0.48 0.14 155)",
          }}
        >
          {usagePct.toFixed(1)}%
        </span>
      </div>

      {/* ---- Legend ---- */}
      <div className="flex flex-wrap gap-x-3 gap-y-1.5">
        {segments
          .filter((s) => s.bytes > 0 && s.key !== "freeHeadroom")
          .map((seg) => (
            <button
              key={seg.key}
              type="button"
              className="flex items-center gap-1.5 rounded px-1.5 py-0.5 text-xs transition-colors"
              style={{
                backgroundColor:
                  hovered === seg.key
                    ? isDark
                      ? "oklch(0.25 0.02 260)"
                      : "oklch(0.94 0.005 260)"
                    : "transparent",
              }}
              onMouseEnter={() => setHovered(seg.key)}
              onMouseLeave={() => setHovered(null)}
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ backgroundColor: seg.color }}
              />
              <span className="text-muted">{seg.label}</span>
              <span className="font-mono tabular-nums text-muted/70">{formatGB(seg.bytes)}</span>
            </button>
          ))}
      </div>
    </div>
  )
}
