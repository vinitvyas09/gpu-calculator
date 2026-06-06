"use client"

import { useState, type ReactNode } from "react"
import { motion, AnimatePresence, useReducedMotion } from "framer-motion"
import type { CalculatorColors } from "./input-controls"
import { GLOSSARY } from "../glossary"

// ---------------------------------------------------------------------------
// Term — dotted-underline inline glossary trigger.
//
// Renders visible text (children ?? GLOSSARY[termKey].term) with a dotted
// underline; on hover OR keyboard focus it opens a definition popover that
// reuses TooltipIcon's AnimatePresence markup/motion (input-controls.tsx). The
// trigger is a focusable <button> so the popover opens on focus and dismisses
// on blur or Escape. Display-only: never reads/writes calculator state.
//
// `interactive={false}` renders the trigger as a non-interactive <span>
// (hover-only popover, no tabindex/role/keyboard focus). Use it wherever a Term
// is woven into copy that already lives inside another interactive control —
// e.g. a closed <CollapsibleSection> summary, which renders inside the header
// toggle <button>. A focusable <button> (or any tabbable/role="button" element)
// there is invalid HTML (`<button>` cannot descend from `<button>`), triggers a
// React hydration error, and nests interactive controls; the static <span>
// keeps the dotted underline + on-hover definition without any of that. Mouse
// users still get the popover; keyboard users reach the same definition via the
// matching interactive Term in the layer body when expanded.
//
// Single source of truth: the definition text always comes from GLOSSARY
// (glossary.ts). In development a missing key surfaces via console.error so a
// typo can't ship a silent empty popover.
// ---------------------------------------------------------------------------
export interface TermProps {
  /** Glossary key; must exist in GLOSSARY (dev-asserts in non-prod). */
  termKey: string
  /** Visible text (defaults to GLOSSARY[termKey].term). */
  children?: ReactNode
  colors: CalculatorColors
  /**
   * Default `true`: focusable <button> trigger (hover + keyboard focus open).
   * `false`: non-interactive <span> trigger (hover-only) — required when the
   * Term is nested inside another interactive control (e.g. a closed-layer
   * summary inside the header toggle button) to avoid nested-interactive
   * invalid HTML / hydration errors.
   */
  interactive?: boolean
}

export function Term({ termKey, children, colors, interactive = true }: TermProps) {
  const [show, setShow] = useState(false)
  const reduce = useReducedMotion()

  const entry = GLOSSARY[termKey]

  if (!entry) {
    if (process.env.NODE_ENV !== "production") {
      console.error(
        `<Term>: unknown termKey "${termKey}" — not found in GLOSSARY (glossary.ts).`,
      )
    }
    // Degrade to plain text: render whatever was passed, no popover/underline.
    return <>{children ?? termKey}</>
  }

  const label = children ?? entry.term

  // Shared trigger presentation — identical dotted-underline styling whether the
  // trigger is the focusable <button> or the static <span>.
  const triggerClassName =
    "cursor-help bg-transparent p-0 text-left font-[inherit] text-[inherit] focus:outline-none"
  const triggerStyle = {
    color: "inherit",
    textDecorationLine: "underline" as const,
    textDecorationStyle: "dotted" as const,
    textUnderlineOffset: "0.2em",
    textDecorationColor: colors.textSecondary,
  }

  return (
    <span className="relative inline-flex">
      {interactive ? (
        <button
          type="button"
          onMouseEnter={() => setShow(true)}
          onMouseLeave={() => setShow(false)}
          onFocus={() => setShow(true)}
          onBlur={() => setShow(false)}
          onKeyDown={(e) => {
            if (e.key === "Escape" && show) {
              e.stopPropagation()
              setShow(false)
            }
          }}
          className={triggerClassName}
          style={triggerStyle}
          aria-label={entry.term}
        >
          {label}
        </button>
      ) : (
        // Non-interactive: a bare <span> (no tabindex/role) is not a control, so
        // it is legal inside another interactive element and never tab-focuses.
        // Hover still opens the popover; keyboard users get the definition from
        // the interactive Term in the expanded layer body.
        <span
          onMouseEnter={() => setShow(true)}
          onMouseLeave={() => setShow(false)}
          className={triggerClassName}
          style={triggerStyle}
        >
          {label}
        </span>
      )}
      <AnimatePresence>
        {show && (
          <motion.span
            initial={{ opacity: 0, y: 4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.97 }}
            transition={reduce ? { duration: 0 } : { duration: 0.12 }}
            role="tooltip"
            className="absolute bottom-full left-1/2 z-50 mb-2 block -translate-x-1/2 whitespace-normal rounded-lg border px-3 py-2 text-left text-[11px] font-normal normal-case leading-relaxed tracking-normal shadow-lg backdrop-blur-sm"
            style={{
              backgroundColor: colors.cardBg,
              borderColor: colors.border,
              color: colors.text,
              minWidth: 200,
              maxWidth: 280,
            }}
          >
            <span
              className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.08em]"
              style={{ color: colors.textSecondary }}
            >
              {entry.term}
            </span>
            {entry.def}
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  )
}
