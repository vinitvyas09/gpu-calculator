"use client"

import { motion } from "framer-motion"
import type { MemoryBreakdown } from "../types"

const RADIUS = 80
const CENTER_X = 100
const CENTER_Y = 100
const STROKE = 12
const ARC_LENGTH = 75
const TICKS = [0, 0.25, 0.5, 0.75, 1].map((fraction) => {
  const angle = 135 + fraction * 270
  const radians = (angle * Math.PI) / 180
  const innerRadius = RADIUS - STROKE / 2 - 5
  const outerRadius = RADIUS - STROKE / 2 - 1

  return {
    key: fraction,
    x1: CENTER_X + Math.cos(radians) * innerRadius,
    y1: CENTER_Y + Math.sin(radians) * innerRadius,
    x2: CENTER_X + Math.cos(radians) * outerRadius,
    y2: CENTER_Y + Math.sin(radians) * outerRadius,
  }
})

interface Props {
  breakdown: MemoryBreakdown
  isDark: boolean
  size?: "sm" | "md"
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

function getGaugeStatus(utilizationPct: number, isDark: boolean) {
  if (utilizationPct > 90) {
    return {
      label: utilizationPct >= 100 ? "Over budget" : "High pressure",
      color: isDark ? "oklch(0.8 0.16 25)" : "oklch(0.54 0.19 25)",
      glow: isDark ? "oklch(0.72 0.14 25 / 0.45)" : "oklch(0.6 0.14 25 / 0.28)",
      bg: isDark ? "oklch(0.24 0.08 25)" : "oklch(0.96 0.04 25)",
      text: isDark ? "oklch(0.8 0.16 25)" : "oklch(0.5 0.18 25)",
    }
  }

  if (utilizationPct >= 70) {
    return {
      label: "Moderate pressure",
      color: isDark ? "oklch(0.84 0.12 85)" : "oklch(0.62 0.14 85)",
      glow: isDark ? "oklch(0.78 0.12 85 / 0.38)" : "oklch(0.68 0.11 85 / 0.24)",
      bg: isDark ? "oklch(0.25 0.05 85)" : "oklch(0.98 0.03 85)",
      text: isDark ? "oklch(0.84 0.12 85)" : "oklch(0.56 0.14 85)",
    }
  }

  return {
    label: "Healthy headroom",
    color: isDark ? "oklch(0.74 0.13 165)" : "oklch(0.55 0.13 165)",
    glow: isDark ? "oklch(0.7 0.11 165 / 0.35)" : "oklch(0.58 0.09 165 / 0.22)",
    bg: isDark ? "oklch(0.23 0.04 165)" : "oklch(0.97 0.02 165)",
    text: isDark ? "oklch(0.78 0.11 165)" : "oklch(0.47 0.12 165)",
  }
}

export default function GpuUtilizationGauge({
  breakdown,
  isDark,
  size = "md",
}: Props) {
  const physicalCapacity = sanitizePositive(breakdown.gpuCapacity)
  const usableCapacity = sanitizePositive(breakdown.usableCapacity) || physicalCapacity
  const usedMemory = sanitizePositive(breakdown.total)
  const utilizationPct =
    usableCapacity > 0 ? (usedMemory / usableCapacity) * 100 : 0
  const visualPct = Math.max(0, Math.min(utilizationPct, 100))
  const fillLength = (visualPct / 100) * ARC_LENGTH
  const overflowPct = Math.max(utilizationPct - 100, 0)
  const status = getGaugeStatus(utilizationPct, isDark)
  const dimensionClass = size === "sm" ? "h-32 w-32" : "h-40 w-40"
  const valueFontSize = size === "sm" ? 24 : 34
  const trackColor = isDark ? "oklch(0.26 0.01 255)" : "oklch(0.91 0.004 255)"

  return (
    <div className="flex flex-col items-center gap-3">
      <svg
        viewBox="0 0 200 200"
        className={dimensionClass}
        role="img"
        aria-label={`GPU memory budget utilization ${Math.round(utilizationPct)} percent, ${formatMemory(usedMemory)} of ${formatMemory(usableCapacity)} usable`}
      >
        <circle
          cx={CENTER_X}
          cy={CENTER_Y}
          r={RADIUS + 3}
          fill="none"
          stroke={isDark ? "oklch(0.21 0.006 255)" : "oklch(0.95 0.003 255)"}
          strokeWidth={0.75}
          pathLength={100}
          strokeDasharray="75 100"
          transform={`rotate(135 ${CENTER_X} ${CENTER_Y})`}
        />

        <circle
          cx={CENTER_X}
          cy={CENTER_Y}
          r={RADIUS}
          fill="none"
          stroke={trackColor}
          strokeWidth={STROKE}
          pathLength={100}
          strokeDasharray="75 100"
          strokeLinecap="round"
          transform={`rotate(135 ${CENTER_X} ${CENTER_Y})`}
        />

        {TICKS.map((tick) => (
          <line
            key={tick.key}
            x1={tick.x1}
            y1={tick.y1}
            x2={tick.x2}
            y2={tick.y2}
            stroke={isDark ? "oklch(0.43 0.01 255)" : "oklch(0.75 0.006 255)"}
            strokeWidth={1.5}
            strokeLinecap="round"
          />
        ))}

        <motion.circle
          cx={CENTER_X}
          cy={CENTER_Y}
          r={RADIUS}
          fill="none"
          strokeWidth={STROKE}
          pathLength={100}
          strokeDasharray="75 100"
          strokeLinecap="round"
          transform={`rotate(135 ${CENTER_X} ${CENTER_Y})`}
          style={{ filter: `drop-shadow(0 0 8px ${status.glow})` }}
          initial={{ strokeDashoffset: 75, stroke: status.color }}
          animate={{
            strokeDashoffset: 75 - fillLength,
            stroke: status.color,
          }}
          transition={{
            strokeDashoffset: { duration: 0.9, ease: [0.22, 1, 0.36, 1] },
            stroke: { duration: 0.3 },
          }}
        />

        <text
          x={CENTER_X}
          y={CENTER_Y - 8}
          textAnchor="middle"
          dominantBaseline="central"
          fill={isDark ? "oklch(0.96 0.01 95)" : "oklch(0.22 0.03 248)"}
          fontSize={valueFontSize}
          fontWeight={700}
          fontFamily="var(--font-mono)"
        >
          {Math.round(utilizationPct)}%
        </text>

        <text
          x={CENTER_X}
          y={CENTER_Y + 18}
          textAnchor="middle"
          fill={isDark ? "oklch(0.72 0.02 255)" : "oklch(0.56 0.02 255)"}
          fontSize={10}
          fontFamily="var(--font-sans)"
          style={{ letterSpacing: "0.14em" }}
        >
          USABLE VRAM
        </text>
      </svg>

      <div
        className="rounded-full px-3 py-1 text-xs font-medium"
        style={{
          backgroundColor: status.bg,
          color: status.text,
        }}
      >
        {status.label}
        {overflowPct > 0 ? ` | +${overflowPct.toFixed(1)}%` : ""}
      </div>

      <div className="space-y-0.5 text-center">
        <div className="font-mono text-xs tabular-nums text-muted">
          {formatMemory(usedMemory)} / {formatMemory(usableCapacity)} usable
        </div>
        {physicalCapacity > usableCapacity && (
          <div className="text-[11px] text-muted">
            {formatMemory(physicalCapacity)} physical VRAM
          </div>
        )}
      </div>
    </div>
  )
}
