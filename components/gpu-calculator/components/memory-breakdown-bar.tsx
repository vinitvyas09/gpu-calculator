"use client"

import { useId, useMemo, useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { AlertTriangle } from "lucide-react"
import type { MemoryBreakdown } from "../types"

const VIEW_WIDTH = 1000
const VIEW_HEIGHT = 96

type SegmentKey =
  | "parameters"
  | "gradients"
  | "optimizerStates"
  | "activations"
  | "communicationBuffers"
  | "frameworkOverhead"
  | "freeHeadroom"

const SEGMENT_ORDER: SegmentKey[] = [
  "parameters",
  "gradients",
  "optimizerStates",
  "activations",
  "communicationBuffers",
  "frameworkOverhead",
  "freeHeadroom",
]

const SEGMENT_META: Record<
  SegmentKey,
  { label: string; light: string; dark: string }
> = {
  parameters: {
    label: "Parameters",
    light: "oklch(0.59 0.16 250)",
    dark: "oklch(0.72 0.13 245)",
  },
  gradients: {
    label: "Gradients",
    light: "oklch(0.64 0.18 330)",
    dark: "oklch(0.75 0.16 328)",
  },
  optimizerStates: {
    label: "Optimizer States",
    light: "oklch(0.78 0.14 85)",
    dark: "oklch(0.84 0.12 84)",
  },
  activations: {
    label: "Activations",
    light: "oklch(0.70 0.12 165)",
    dark: "oklch(0.79 0.1 166)",
  },
  communicationBuffers: {
    label: "Buffers",
    light: "oklch(0.69 0.15 32)",
    dark: "oklch(0.79 0.13 32)",
  },
  frameworkOverhead: {
    label: "Overhead",
    light: "oklch(0.72 0.02 255)",
    dark: "oklch(0.54 0.02 255)",
  },
  freeHeadroom: {
    label: "Free Headroom",
    light: "oklch(0.95 0.005 255)",
    dark: "oklch(0.31 0.01 255)",
  },
}

interface Segment {
  key: SegmentKey
  label: string
  bytes: number
  color: string
  x: number
  width: number
  pctOfBudget: number
}

interface Props {
  breakdown: MemoryBreakdown
  isDark: boolean
}

function sanitizePositive(value: number): number {
  return Number.isFinite(value) && value > 0 ? value : 0
}

function formatMemory(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "--"
  }

  const tb = bytes / 1e12
  const gb = bytes / 1e9
  const mb = bytes / 1e6

  if (tb >= 1) return `${tb.toFixed(tb >= 10 ? 1 : 2)} TB`
  if (gb >= 999.5) return `${(gb / 1000).toFixed(2)} TB`
  if (gb >= 100) return `${Math.round(gb)} GB`
  if (gb >= 10) return `${gb.toFixed(1)} GB`
  if (gb >= 1) return `${gb.toFixed(2)} GB`
  if (mb >= 999.5) return `${(mb / 1000).toFixed(2)} GB`
  if (mb >= 1) return `${mb.toFixed(0)} MB`
  return "< 1 MB"
}

function formatPercent(value: number): string {
  if (!Number.isFinite(value)) {
    return "--"
  }

  return `${value.toFixed(1)}%`
}

function getUtilizationColor(utilizationPct: number, isDark: boolean): string {
  if (utilizationPct >= 100) {
    return isDark ? "oklch(0.78 0.16 25)" : "oklch(0.54 0.19 25)"
  }
  if (utilizationPct >= 90) {
    return isDark ? "oklch(0.84 0.12 85)" : "oklch(0.62 0.14 85)"
  }
  if (utilizationPct >= 70) {
    return isDark ? "oklch(0.8 0.11 155)" : "oklch(0.57 0.12 155)"
  }
  return isDark ? "oklch(0.73 0.13 210)" : "oklch(0.53 0.13 220)"
}

export default function MemoryBreakdownBar({ breakdown, isDark }: Props) {
  const [hovered, setHovered] = useState<SegmentKey | null>(null)
  const patternId = useId().replace(/:/g, "")

  const {
    physicalCapacity,
    usableCapacity,
    usedMemory,
    allocatorAlignmentOverhead,
    reservedBuffer,
    exceedsUsableBudget,
    exceedsPhysicalCapacity,
    segments,
    reservedStart,
    reservedWidth,
    usableMarkerX,
    physicalMarkerX,
    utilizationPct,
    utilizationColor,
  } = useMemo(() => {
    const physCap = sanitizePositive(breakdown.gpuCapacity)
    const usableCap = sanitizePositive(breakdown.usableCapacity) || physCap
    const rawSegmentSum =
      sanitizePositive(breakdown.parameters) +
      sanitizePositive(breakdown.gradients) +
      sanitizePositive(breakdown.optimizerStates) +
      sanitizePositive(breakdown.activations) +
      sanitizePositive(breakdown.communicationBuffers) +
      sanitizePositive(breakdown.frameworkOverhead)
    // Use breakdown.total as authoritative used memory — it accounts for CUDA
    // alignment (1.04×) and peak-based semantics in PPO/GRPO methods where
    // activations and buffers don't coexist simultaneously.
    const used = sanitizePositive(breakdown.total)
    const alignOverhead = Math.max(0, used - rawSegmentSum)
    const displayBytes: Record<SegmentKey, number> = {
      parameters: sanitizePositive(breakdown.parameters),
      gradients: sanitizePositive(breakdown.gradients),
      optimizerStates: sanitizePositive(breakdown.optimizerStates),
      activations: sanitizePositive(breakdown.activations),
      communicationBuffers: sanitizePositive(breakdown.communicationBuffers),
      frameworkOverhead:
        sanitizePositive(breakdown.frameworkOverhead) + alignOverhead,
      freeHeadroom: sanitizePositive(breakdown.freeHeadroom),
    }
    const resBuf = Math.max(physCap - usableCap, 0)
    const exceedsUsable = used > usableCap + 1
    const exceedsPhysical = used > physCap + 1
    const scale = Math.max(physCap, usableCap, used, 1)

    const mode = isDark ? "dark" : "light"
    const builtSegments = SEGMENT_ORDER.map((key) => ({
      key,
      bytes: displayBytes[key],
    }))
      .filter((seg) => seg.bytes > 0)
      .reduce<Segment[]>((acc, seg) => {
        const x =
          acc.length > 0 ? acc[acc.length - 1].x + acc[acc.length - 1].width : 0
        const width = (seg.bytes / scale) * VIEW_WIDTH

        acc.push({
          key: seg.key,
          label: SEGMENT_META[seg.key].label,
          bytes: seg.bytes,
          color: SEGMENT_META[seg.key][mode],
          x,
          width,
          pctOfBudget: usableCap > 0 ? (seg.bytes / usableCap) * 100 : 0,
        })

        return acc
      }, [])

    const utilPct = usableCap > 0 ? (used / usableCap) * 100 : 0

    return {
      physicalCapacity: physCap,
      usableCapacity: usableCap,
      usedMemory: used,
      allocatorAlignmentOverhead: alignOverhead,
      reservedBuffer: resBuf,
      exceedsUsableBudget: exceedsUsable,
      exceedsPhysicalCapacity: exceedsPhysical,
      segments: builtSegments,
      reservedStart: (usableCap / scale) * VIEW_WIDTH,
      reservedWidth: (resBuf / scale) * VIEW_WIDTH,
      usableMarkerX:
        usableCap > 0 && usableCap < scale
          ? (usableCap / scale) * VIEW_WIDTH
          : null,
      physicalMarkerX:
        physCap > 0 && physCap < scale
          ? (physCap / scale) * VIEW_WIDTH
          : null,
      utilizationPct: utilPct,
      utilizationColor: getUtilizationColor(utilPct, isDark),
    }
  }, [
    breakdown.parameters,
    breakdown.gradients,
    breakdown.optimizerStates,
    breakdown.activations,
    breakdown.communicationBuffers,
    breakdown.frameworkOverhead,
    breakdown.freeHeadroom,
    breakdown.total,
    breakdown.gpuCapacity,
    breakdown.usableCapacity,
    isDark,
  ])

  const hoveredSegment = segments.find((segment) => segment.key === hovered) ?? null
  const tooltipLeft = hoveredSegment
    ? Math.max(
        8,
        Math.min(
          92,
          ((hoveredSegment.x + hoveredSegment.width / 2) / VIEW_WIDTH) * 100,
        ),
      )
    : 50

  if (usableCapacity === 0 && physicalCapacity === 0) {
    return <p className="text-sm text-muted">No memory data available.</p>
  }

  return (
    <div className="space-y-3">
      {(exceedsPhysicalCapacity || exceedsUsableBudget) && (
        <div className="flex justify-end">
          <div
            className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
            style={{
              backgroundColor: exceedsPhysicalCapacity
                ? isDark
                  ? "oklch(0.24 0.08 25)"
                  : "oklch(0.96 0.04 25)"
                : isDark
                  ? "oklch(0.25 0.05 85)"
                  : "oklch(0.98 0.03 85)",
              color: exceedsPhysicalCapacity
                ? isDark
                  ? "oklch(0.8 0.16 25)"
                  : "oklch(0.49 0.18 25)"
                : isDark
                  ? "oklch(0.84 0.12 85)"
                  : "oklch(0.58 0.14 85)",
            }}
          >
            <AlertTriangle className="h-3 w-3" />
            {exceedsPhysicalCapacity ? "Exceeds GPU capacity" : "Into reserve buffer"}
          </div>
        </div>
      )}

      <div className="relative">
        <div className="overflow-hidden rounded-2xl border border-border bg-surface-elevated/60">
          <motion.div
            initial={{ clipPath: "inset(0 100% 0 0)" }}
            animate={{ clipPath: "inset(0 0% 0 0)" }}
            transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
          >
            <svg
              viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
              preserveAspectRatio="none"
              className="block h-14 w-full"
              role="img"
              aria-label={`GPU memory usage ${formatMemory(usedMemory)} of ${formatMemory(usableCapacity)} usable memory${physicalCapacity > usableCapacity ? `, ${formatMemory(physicalCapacity)} physical VRAM` : ""}`}
            >
              <defs>
                <pattern
                  id={patternId}
                  patternUnits="userSpaceOnUse"
                  width="20"
                  height="20"
                  patternTransform="rotate(45)"
                >
                  <line
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="20"
                    stroke={isDark ? "oklch(0.4 0.02 255)" : "oklch(0.82 0.01 255)"}
                    strokeWidth="6"
                  />
                </pattern>
              </defs>

              <rect
                x={0}
                y={0}
                width={VIEW_WIDTH}
                height={VIEW_HEIGHT}
                fill={isDark ? "oklch(0.2 0.015 255)" : "oklch(0.975 0.003 255)"}
              />

              {segments.map((segment) => (
                <motion.rect
                  key={segment.key}
                  x={segment.x}
                  y={0}
                  height={VIEW_HEIGHT}
                  fill={segment.color}
                  initial={false}
                  animate={{
                    width: Math.max(segment.width, 0),
                    opacity: hovered && hovered !== segment.key ? 0.34 : 1,
                  }}
                  transition={{
                    width: { duration: 0.42, ease: [0.22, 1, 0.36, 1] },
                    opacity: { duration: 0.15 },
                  }}
                  onMouseEnter={() => setHovered(segment.key)}
                  onMouseLeave={() => setHovered(null)}
                  style={{ cursor: "pointer" }}
                />
              ))}

              {reservedWidth > 0 && (
                <rect
                  x={reservedStart}
                  y={0}
                  width={reservedWidth}
                  height={VIEW_HEIGHT}
                  fill={`url(#${patternId})`}
                  opacity={0.9}
                  pointerEvents="none"
                />
              )}

              {usableMarkerX !== null && usableMarkerX > 0 && usableMarkerX < VIEW_WIDTH && (
                <line
                  x1={usableMarkerX}
                  y1={0}
                  x2={usableMarkerX}
                  y2={VIEW_HEIGHT}
                  stroke={
                    exceedsUsableBudget
                      ? isDark
                        ? "oklch(0.84 0.12 85)"
                        : "oklch(0.62 0.14 85)"
                      : isDark
                        ? "oklch(0.64 0.02 255)"
                        : "oklch(0.62 0.02 255)"
                  }
                  strokeWidth={2}
                  vectorEffect="non-scaling-stroke"
                  strokeDasharray="6 4"
                />
              )}

              {physicalMarkerX !== null && physicalMarkerX > 0 && physicalMarkerX < VIEW_WIDTH && (
                <motion.line
                  x1={physicalMarkerX}
                  y1={0}
                  x2={physicalMarkerX}
                  y2={VIEW_HEIGHT}
                  stroke={isDark ? "oklch(0.8 0.16 25)" : "oklch(0.54 0.19 25)"}
                  strokeWidth={2}
                  vectorEffect="non-scaling-stroke"
                  strokeDasharray="7 4"
                  animate={{ opacity: [0.55, 1, 0.55] }}
                  transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
                />
              )}
            </svg>
          </motion.div>
        </div>

        <AnimatePresence>
          {hoveredSegment && (
            <motion.div
              key={hoveredSegment.key}
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
              <div className="whitespace-nowrap rounded-xl border border-border bg-surface-elevated px-3 py-2 text-xs shadow-lg backdrop-blur-md">
                <div className="flex items-center gap-2">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-sm"
                    style={{ backgroundColor: hoveredSegment.color }}
                  />
                  <span className="font-medium text-foreground">{hoveredSegment.label}</span>
                </div>
                <div className="mt-1 flex items-center gap-1.5 pl-[18px]">
                  <span className="font-mono tabular-nums text-foreground">
                    {formatMemory(hoveredSegment.bytes)}
                  </span>
                  <span className="text-muted">|</span>
                  <span className="tabular-nums text-muted">
                    {formatPercent(hoveredSegment.pctOfBudget)} of usable
                  </span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="space-y-1">
        <div className="flex items-baseline justify-between gap-3 text-sm">
          <span className="text-muted">
            {formatMemory(usedMemory)} of {formatMemory(usableCapacity)} usable
          </span>
          <span
            className="font-mono text-sm font-semibold tabular-nums"
            style={{ color: utilizationColor }}
          >
            {formatPercent(utilizationPct)}
          </span>
        </div>
        <div className="text-xs text-muted">
          {formatMemory(physicalCapacity)} physical VRAM
          {reservedBuffer > 0
            ? ` | ${formatMemory(reservedBuffer)} reserved for fragmentation/system overhead`
            : ""}
        </div>
        {allocatorAlignmentOverhead > 0 && (
          <div className="text-[11px] text-muted">
            Includes {formatMemory(allocatorAlignmentOverhead)} allocator alignment
            folded into the overhead segment.
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-x-3 gap-y-1.5">
        {segments.map((segment) => (
          <button
            key={segment.key}
            type="button"
            className="flex items-center gap-1.5 rounded-full px-2 py-1 text-xs transition-colors"
            style={{
              backgroundColor:
                hovered === segment.key
                  ? isDark
                    ? "oklch(0.28 0.02 255)"
                    : "oklch(0.95 0.01 255)"
                  : "transparent",
            }}
            onMouseEnter={() => setHovered(segment.key)}
            onMouseLeave={() => setHovered(null)}
            onFocus={() => setHovered(segment.key)}
            onBlur={() => setHovered(null)}
          >
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: segment.color }}
            />
            <span className="text-muted">{segment.label}</span>
            <span className="font-mono tabular-nums text-muted/80">
              {formatMemory(segment.bytes)}
            </span>
          </button>
        ))}

        {reservedBuffer > 0 && (
          <div className="flex items-center gap-1.5 rounded-full px-2 py-1 text-xs">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm border"
              style={{
                borderColor: isDark ? "oklch(0.48 0.02 255)" : "oklch(0.82 0.01 255)",
                backgroundImage: `repeating-linear-gradient(135deg, ${
                  isDark ? "oklch(0.32 0.01 255)" : "oklch(0.94 0.003 255)"
                } 0 4px, transparent 4px 8px)`,
              }}
            />
            <span className="text-muted">Reserved Buffer</span>
            <span className="font-mono tabular-nums text-muted/80">
              {formatMemory(reservedBuffer)}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
