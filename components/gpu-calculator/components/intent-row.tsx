"use client"

import { useId, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ChevronRight } from "lucide-react"
import type { CalculatorTab } from "../types"
import type { CalculatorColors } from "./input-controls"

// ---------------------------------------------------------------------------
// IntentRow — quiet "New here? ▸" on-ramp (Appendix C §1.6, copy Appendix B §4).
//
// Collapsed by default to a small muted affordance. Expanding reveals three
// plain-verb cards. Every card focuses the Essentials block, dismisses the
// on-ramp (persisted by the host), and collapses the row.
//
// Returning users (dismissed === true) still get the quiet affordance, but it
// stays collapsed until they click it again — "invisible weight."
//
// Takes `colors` and threads it like CollapsibleSection (inline style values),
// reusing that component's collapse motion (height/opacity in AnimatePresence)
// and the results-summary card entrance stagger (delay: index * 0.03).
// ---------------------------------------------------------------------------

export interface IntentRowProps {
  colors: CalculatorColors
  /** Set the phase tab when a verb card is chosen. */
  onChooseTab: (tab: CalculatorTab) => void
  /** Scroll-focuses the Essentials block after a choice. */
  onFocusEssentials: () => void
  /** Persisted dismissal (returning users never see it expanded). */
  dismissed: boolean
  onDismiss: () => void
}

interface IntentCard {
  title: string
  subtitle: string
  /** A chosen tab to set, or null for "I know my config" (dismiss only). */
  tab: CalculatorTab | null
}

const INTENT_CARDS: IntentCard[] = [
  {
    title: "Plan a pretraining run",
    subtitle: "Train a model from scratch. We'll size the GPUs, cost, and time.",
    tab: "pretraining",
  },
  {
    title: "Fine-tune a model",
    subtitle: "Adapt an existing model with SFT, LoRA, or RLHF.",
    tab: "post-training",
  },
  {
    title: "I know my config",
    subtitle: "Skip the intro — take me straight to the controls.",
    tab: null,
  },
]

export function IntentRow({
  colors,
  onChooseTab,
  onFocusEssentials,
  dismissed,
  onDismiss,
}: IntentRowProps) {
  // Start collapsed for everyone; first-time users can open it, returning
  // (dismissed) users see only the quiet affordance until they re-open it.
  const [open, setOpen] = useState(false)
  const contentId = useId()

  // Returning users (dismissed) get an even quieter affordance — the on-ramp
  // has already served its purpose, so it carries less visual weight, but the
  // local toggle below still lets them re-open it on demand.
  const affordanceOpacity = dismissed ? 0.6 : 0.75

  const choose = (card: IntentCard) => {
    if (card.tab) onChooseTab(card.tab)
    onFocusEssentials()
    onDismiss()
    setOpen(false)
  }

  return (
    <div className="min-w-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={contentId}
        className="inline-flex items-center gap-1 rounded-md py-1 text-xs font-medium transition-opacity hover:opacity-100"
        style={{ color: colors.textSecondary, opacity: affordanceOpacity }}
      >
        New here?
        <motion.span
          aria-hidden
          className="inline-flex"
          animate={{ rotate: open ? 90 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </motion.span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={contentId}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="pt-3">
              <div
                className="mb-2.5 text-[11px] font-medium uppercase tracking-[0.08em]"
                style={{ color: colors.textSecondary }}
              >
                What are you trying to do?
              </div>
              <div className="grid gap-2.5 sm:grid-cols-3">
                {INTENT_CARDS.map((card, index) => (
                  <motion.button
                    key={card.title}
                    type="button"
                    onClick={() => choose(card)}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.03, duration: 0.22 }}
                    className="group rounded-lg border p-3.5 text-left transition-colors"
                    style={{
                      borderColor: colors.border,
                      backgroundColor: colors.cardBg,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = colors.accent
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = colors.border
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.borderColor = colors.accent
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = colors.border
                    }}
                  >
                    <div
                      className="text-sm font-semibold"
                      style={{ color: colors.text }}
                    >
                      {card.title}
                    </div>
                    <div
                      className="mt-1 text-xs leading-relaxed"
                      style={{ color: colors.textSecondary }}
                    >
                      {card.subtitle}
                    </div>
                  </motion.button>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
