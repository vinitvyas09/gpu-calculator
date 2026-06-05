"use client"

import { Info } from "lucide-react"
import type { CalculatorColors } from "./input-controls"

// ---------------------------------------------------------------------------
// AssumptionsLedger — the central "N auto-adjustments" panel listing every
// silent substitution as one info-tone row. Pure presentation: `entries` are
// assembled by the host from the SAME already-computed booleans the
// OverrideBadges use; this component recomputes nothing.
//
// Row look mirrors the WarningList inline callout (results-summary.tsx:269-295)
// — a tinted bordered card with an icon, a summary line, and the reason — but
// in info tone and clickable: a row jumps to its owning layer via
// onJumpToLayer(targetLayerId).
// ---------------------------------------------------------------------------
export interface LedgerEntry {
  /** Stable id, e.g. "optimizer-fp8-fallback". */
  id: string
  /** What changed, e.g. "Optimizer: AdamW-FP8 → AdamW (mixed)". */
  summary: string
  /** Why. */
  reason: string
  /** Layer id to scroll/open when the row is clicked (ties into perLayerOpen). */
  targetLayerId?: string
}

export interface AssumptionsLedgerProps {
  colors: CalculatorColors
  entries: LedgerEntry[]
  /** Click a row → open + scroll its owning layer. */
  onJumpToLayer?: (layerId: string) => void
}

export function AssumptionsLedger({
  colors,
  entries,
  onJumpToLayer,
}: AssumptionsLedgerProps) {
  if (entries.length === 0) {
    return null
  }

  return (
    <ul className="space-y-2">
      {entries.map((entry) => {
        const jump = entry.targetLayerId
          ? () => onJumpToLayer?.(entry.targetLayerId as string)
          : undefined
        const interactive = jump !== undefined

        const rowStyle = {
          backgroundColor: colors.accentMuted,
          borderColor: colors.border,
          color: colors.text,
        }
        const rowClassName =
          "flex w-full items-start gap-3 rounded-lg border px-4 py-3 text-left"

        const body = (
          <>
            <Info
              className="mt-0.5 h-4 w-4 shrink-0"
              style={{ color: colors.accent }}
            />
            <span className="min-w-0">
              <span
                className="block text-sm font-medium"
                style={{ color: colors.text }}
              >
                {entry.summary}
              </span>
              <span
                className="mt-1 block text-xs leading-5"
                style={{ color: colors.textSecondary }}
              >
                {entry.reason}
              </span>
            </span>
          </>
        )

        return (
          <li key={entry.id}>
            {interactive ? (
              <button
                type="button"
                onClick={jump}
                className={`${rowClassName} no-theme-transition`}
                style={rowStyle}
              >
                {body}
              </button>
            ) : (
              <div className={rowClassName} style={rowStyle}>
                {body}
              </div>
            )}
          </li>
        )
      })}
    </ul>
  )
}
