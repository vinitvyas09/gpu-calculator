"use client"

import { motion } from "framer-motion"
import type { ParallelismConfig } from "../types"

// ---------------------------------------------------------------------------
// Dimension colours
// ---------------------------------------------------------------------------

const DIM_COLORS: Record<string, { light: string; dark: string }> = {
  dp: { light: "oklch(0.56 0.19 260)", dark: "oklch(0.67 0.17 260)" },
  tp: { light: "oklch(0.70 0.17 55)",  dark: "oklch(0.76 0.14 55)"  },
  pp: { light: "oklch(0.60 0.14 170)", dark: "oklch(0.72 0.12 170)" },
  ep: { light: "oklch(0.55 0.21 310)", dark: "oklch(0.68 0.18 310)" },
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Props {
  config: ParallelismConfig
  totalGPUs: number
  isDark: boolean
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i)
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function LegendChip({
  label,
  color,
  value,
}: {
  label: string
  color: string
  value: number
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="inline-block h-2.5 w-2.5 rounded-sm"
        style={{ backgroundColor: color }}
      />
      <span className="text-muted">{label}</span>
      <span className="font-mono tabular-nums text-foreground">{value}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ParallelismLayout({ config, totalGPUs, isDark }: Props) {
  const { N_dp, N_tp, N_pp, N_ep } = config
  const mode = isDark ? "dark" : "light"

  // Cap visual grid to prevent overflow
  const maxDP = Math.min(N_dp, 8)
  const maxPP = Math.min(N_pp, 8)
  const maxTP = Math.min(N_tp, 16)
  const truncatedDP = N_dp > maxDP
  const truncatedPP = N_pp > maxPP
  const truncatedTP = N_tp > maxTP

  // Adaptive GPU cell size
  const cellSize = maxTP <= 4 ? "h-4 w-4" : maxTP <= 8 ? "h-3 w-3" : "h-2.5 w-2.5"

  const cellBorder = isDark ? "oklch(0.30 0.02 260)" : "oklch(0.88 0.01 260)"
  const cellBg = isDark ? "oklch(0.20 0.015 260)" : "oklch(0.97 0.003 260)"

  // Strategy label
  const label = [
    `DP${N_dp}`,
    `TP${N_tp}`,
    N_pp > 1 ? `PP${N_pp}` : null,
    N_ep > 1 ? `EP${N_ep}` : null,
  ]
    .filter(Boolean)
    .join(" \u00d7 ")

  return (
    <div className="space-y-4">
      {/* Strategy string */}
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-sm font-semibold text-foreground">{label}</span>
        <span className="text-xs tabular-nums text-muted">
          {totalGPUs.toLocaleString()} GPU{totalGPUs !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Grid */}
      {totalGPUs <= 1 ? (
        /* ---- Single GPU ---- */
        <div className="flex justify-center py-4">
          <motion.div
            className="flex h-10 w-10 items-center justify-center rounded-lg"
            style={{
              backgroundColor: DIM_COLORS.tp[mode],
              boxShadow: `0 0 0 3px ${cellBg}, 0 0 0 4px ${DIM_COLORS.tp[mode]}`,
            }}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 20 }}
          >
            <span
              className="text-xs font-bold"
              style={{ color: isDark ? "oklch(0.15 0 0)" : "oklch(1 0 0)" }}
            >
              0
            </span>
          </motion.div>
        </div>
      ) : (
        /* ---- Full grid ---- */
        <div className="space-y-1">
          {/* PP stage column headers */}
          {maxPP > 1 && (
            <div
              className="flex gap-1"
              style={{ paddingLeft: N_dp > 1 ? "2.75rem" : 0 }}
            >
              {range(maxPP).map((pp) => (
                <div
                  key={pp}
                  className="flex-1 text-center font-mono text-[10px]"
                  style={{ color: DIM_COLORS.pp[mode] }}
                >
                  {pp === maxPP - 1 && truncatedPP
                    ? `S${pp}\u2026`
                    : `S${pp}`}
                </div>
              ))}
            </div>
          )}

          {/* DP rows */}
          {range(maxDP).map((dp) => (
            <div key={dp} className="flex items-center gap-1">
              {/* Row label */}
              {N_dp > 1 && (
                <div
                  className="w-10 shrink-0 text-right font-mono text-[10px]"
                  style={{ color: DIM_COLORS.dp[mode] }}
                >
                  {dp === maxDP - 1 && truncatedDP
                    ? `D${dp}\u2026`
                    : `D${dp}`}
                </div>
              )}

              {/* PP stage cells */}
              {range(maxPP).map((pp) => (
                <motion.div
                  key={pp}
                  className="flex flex-1 flex-wrap gap-0.5 rounded-lg border p-1.5"
                  style={{ borderColor: cellBorder, backgroundColor: cellBg }}
                  initial={{ opacity: 0, scale: 0.92 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{
                    delay: (dp * maxPP + pp) * 0.025,
                    duration: 0.3,
                    ease: [0.22, 1, 0.36, 1],
                  }}
                >
                  {range(maxTP).map((tp) => (
                    <div
                      key={tp}
                      className={`rounded-sm ${cellSize}`}
                      style={{ backgroundColor: DIM_COLORS.tp[mode] }}
                      title={`GPU ${dp * N_pp * N_tp + pp * N_tp + tp}`}
                    />
                  ))}
                  {truncatedTP && (
                    <div
                      className={`flex items-center justify-center rounded-sm font-mono text-[8px] font-medium ${cellSize}`}
                      style={{
                        backgroundColor: isDark
                          ? "oklch(0.30 0.02 260)"
                          : "oklch(0.90 0.005 260)",
                        color: isDark
                          ? "oklch(0.58 0.01 260)"
                          : "oklch(0.50 0.01 260)",
                      }}
                    >
                      +
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          ))}

          {/* Truncation notice */}
          {(truncatedDP || truncatedPP) && (
            <p className="pt-1 text-center text-[10px] text-muted">
              Showing {maxDP}×{maxPP} of {N_dp}×{N_pp} grid
            </p>
          )}
        </div>
      )}

      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs">
        {N_dp > 1 && (
          <LegendChip label="Data Parallel" color={DIM_COLORS.dp[mode]} value={N_dp} />
        )}
        <LegendChip label="Tensor Parallel" color={DIM_COLORS.tp[mode]} value={N_tp} />
        {N_pp > 1 && (
          <LegendChip label="Pipeline" color={DIM_COLORS.pp[mode]} value={N_pp} />
        )}
        {N_ep > 1 && (
          <LegendChip label="Expert" color={DIM_COLORS.ep[mode]} value={N_ep} />
        )}
      </div>
    </div>
  )
}
