"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import type { CalculatorColors } from "./input-controls"

// ---------------------------------------------------------------------------
// OverrideBadge — an inline marker shown at a control when the *effective*
// value differs from the *selected* value (a silent substitution the math
// already applied). Pure presentation: `label`/`reason` are derived strings
// supplied by the host — this component computes nothing.
//
// Visual: a small accent pill (matching the CollapsibleSection badge pill,
// input-controls.tsx:1115-1125) whose hover/focus reveals an info-style
// popover with the reason (the TooltipIcon AnimatePresence markup,
// input-controls.tsx:182-202). The pill itself is the keyboard-focusable
// trigger.
// ---------------------------------------------------------------------------
export interface OverrideBadgeProps {
  colors: CalculatorColors
  /** Short label e.g. "Using AdamW (mixed)". */
  label: string
  /** Why the substitution happened — plain sentence shown on hover/focus. */
  reason: string
}

export function OverrideBadge({ colors, label, reason }: OverrideBadgeProps) {
  const [show, setShow] = useState(false)

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onFocus={() => setShow(true)}
        onBlur={() => setShow(false)}
        aria-label={`${label}. ${reason}`}
        className="inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] transition-opacity focus:outline-none"
        style={{
          backgroundColor: colors.accentMuted,
          color: colors.accent,
        }}
      >
        {label}
      </button>
      <AnimatePresence>
        {show && (
          <motion.div
            initial={{ opacity: 0, y: 4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.97 }}
            transition={{ duration: 0.12 }}
            role="tooltip"
            className="absolute bottom-full left-1/2 z-50 mb-2 -translate-x-1/2 whitespace-normal rounded-lg border px-3 py-2 text-[11px] leading-relaxed shadow-lg backdrop-blur-sm"
            style={{
              backgroundColor: colors.cardBg,
              borderColor: colors.border,
              color: colors.text,
              minWidth: 180,
              maxWidth: 260,
            }}
          >
            {reason}
          </motion.div>
        )}
      </AnimatePresence>
    </span>
  )
}
