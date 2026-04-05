"use client"

import { useMemo } from "react"
import { motion } from "framer-motion"
import type { MemoryBreakdown } from "../types"

// ---------------------------------------------------------------------------
// Gauge geometry — 270 ° arc with 90 ° gap centred at the bottom
// ---------------------------------------------------------------------------

const RADIUS = 80
const CX = 100
const CY = 100
const STROKE = 12
const ARC_FRAC = 0.75 // 270 / 360

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatGB(bytes: number): string {
  const gb = bytes / 1024 ** 3
  if (gb >= 1000) return `${(gb / 1024).toFixed(1)} TB`
  if (gb >= 100) return `${Math.round(gb)} GB`
  if (gb >= 10) return `${gb.toFixed(1)} GB`
  return `${gb.toFixed(2)} GB`
}

function gaugeColor(pct: number, isDark: boolean): string {
  if (pct >= 90) return isDark ? "oklch(0.65 0.22 25)" : "oklch(0.55 0.22 25)"
  if (pct >= 70) return isDark ? "oklch(0.78 0.16 85)" : "oklch(0.65 0.16 85)"
  return isDark ? "oklch(0.70 0.17 155)" : "oklch(0.55 0.17 155)"
}

function gaugeGlow(pct: number, isDark: boolean): string {
  if (pct >= 90) return isDark ? "oklch(0.55 0.20 25 / 0.25)" : "oklch(0.55 0.18 25 / 0.18)"
  if (pct >= 70) return isDark ? "oklch(0.70 0.14 85 / 0.20)" : "oklch(0.65 0.12 85 / 0.14)"
  return isDark ? "oklch(0.60 0.14 155 / 0.18)" : "oklch(0.55 0.12 155 / 0.12)"
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Props {
  breakdown: MemoryBreakdown
  isDark: boolean
  size?: "sm" | "md"
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function GpuUtilizationGauge({ breakdown, isDark, size = "md" }: Props) {
  const rawPct =
    breakdown.gpuCapacity > 0 ? (breakdown.total / breakdown.gpuCapacity) * 100 : 0
  const visualPct = Math.min(rawPct, 100)
  const fillUnits = (visualPct / 100) * ARC_FRAC * 100 // 0 – 75

  const color = gaugeColor(rawPct, isDark)
  const glow = gaugeGlow(rawPct, isDark)
  const trackColor = isDark ? "oklch(0.25 0.01 260)" : "oklch(0.90 0.005 260)"

  // Tick marks at 0 %, 25 %, 50 %, 75 %, 100 % of the arc
  const ticks = useMemo(() => {
    const startDeg = 135
    const arcDeg = 270
    return [0, 0.25, 0.5, 0.75, 1].map((frac) => {
      const angle = startDeg + frac * arcDeg
      const rad = (angle * Math.PI) / 180
      const r1 = RADIUS - STROKE / 2 - 5
      const r2 = RADIUS - STROKE / 2 - 1
      return {
        x1: CX + Math.cos(rad) * r1,
        y1: CY + Math.sin(rad) * r1,
        x2: CX + Math.cos(rad) * r2,
        y2: CY + Math.sin(rad) * r2,
        frac,
      }
    })
  }, [])

  const dim = size === "sm" ? "w-32 h-32" : "w-40 h-40"
  const fontSize = size === "sm" ? 26 : 34

  return (
    <div className="flex flex-col items-center gap-2">
      <svg viewBox="0 0 200 200" className={dim}>
        {/* Outer guide ring */}
        <circle
          cx={CX}
          cy={CY}
          r={RADIUS + 3}
          fill="none"
          stroke={isDark ? "oklch(0.22 0.005 260)" : "oklch(0.93 0.003 260)"}
          strokeWidth={0.75}
          pathLength={100}
          strokeDasharray="75 100"
          transform={`rotate(135 ${CX} ${CY})`}
        />

        {/* Background arc */}
        <circle
          cx={CX}
          cy={CY}
          r={RADIUS}
          fill="none"
          stroke={trackColor}
          strokeWidth={STROKE}
          pathLength={100}
          strokeDasharray="75 100"
          strokeLinecap="round"
          transform={`rotate(135 ${CX} ${CY})`}
        />

        {/* Tick marks */}
        {ticks.map((t) => (
          <line
            key={t.frac}
            x1={t.x1}
            y1={t.y1}
            x2={t.x2}
            y2={t.y2}
            stroke={isDark ? "oklch(0.40 0.01 260)" : "oklch(0.75 0.005 260)"}
            strokeWidth={1.5}
            strokeLinecap="round"
          />
        ))}

        {/* Glow filter for fill arc */}
        <defs>
          <filter id="gauge-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Filled arc */}
        <motion.circle
          cx={CX}
          cy={CY}
          r={RADIUS}
          fill="none"
          strokeWidth={STROKE}
          pathLength={100}
          strokeDasharray="75 100"
          strokeLinecap="round"
          transform={`rotate(135 ${CX} ${CY})`}
          filter="url(#gauge-glow)"
          initial={{ strokeDashoffset: 75, stroke: color }}
          animate={{ strokeDashoffset: 75 - fillUnits, stroke: color }}
          transition={{
            strokeDashoffset: { duration: 1, ease: [0.22, 1, 0.36, 1] },
            stroke: { duration: 0.4 },
          }}
        />

        {/* Centre — percentage */}
        <text
          x={CX}
          y={CY - 6}
          textAnchor="middle"
          dominantBaseline="central"
          fill={isDark ? "oklch(0.94 0.01 260)" : "oklch(0.18 0.02 260)"}
          fontSize={fontSize}
          fontWeight={700}
          fontFamily="var(--font-mono)"
        >
          {Math.round(rawPct)}%
        </text>

        {/* Centre — label */}
        <text
          x={CX}
          y={CY + 18}
          textAnchor="middle"
          fill={isDark ? "oklch(0.50 0.01 260)" : "oklch(0.58 0.01 260)"}
          fontSize={10}
          fontFamily="var(--font-sans)"
          style={{ letterSpacing: "0.14em" }}
        >
          VRAM USED
        </text>
      </svg>

      {/* Capacity text */}
      <span className="font-mono text-xs tabular-nums text-muted">
        {formatGB(breakdown.total)} / {formatGB(breakdown.gpuCapacity)}
      </span>
    </div>
  )
}
