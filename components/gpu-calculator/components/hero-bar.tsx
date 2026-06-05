"use client"

import { useEffect, useState, type ReactNode } from "react"
import { motion, useReducedMotion } from "framer-motion"

// ---------------------------------------------------------------------------
// HeroBar — root page chrome (Appendix C §1.5).
//
// Page-level chrome, NOT an input/result component: no `colors`/`isDark` prop.
// Styling uses the global CSS tokens directly via Tailwind semantic classes
// (text-foreground / text-muted / text-accent) and the display serif from
// globals.css (the <h1> rule supplies font-family + clamp size for free).
//
// Collapses to a thin bar past a scroll-Y threshold with framer-motion, using
// the easings already in the codebase. The scroll listener is deliberately
// cheap: a single passive listener flipping one boolean.
// ---------------------------------------------------------------------------

export interface HeroBarProps {
  /** Toggles density="compact" + expandAll globally (the expert "Dense view"). */
  denseView: boolean
  onDenseViewChange: (next: boolean) => void
  /** Rendered slot for the existing theme toggle component. */
  themeToggle: ReactNode
}

const COLLAPSE_AT = 120
const EASE = [0.22, 1, 0.36, 1] as const

export default function HeroBar({
  denseView,
  onDenseViewChange,
  themeToggle,
}: HeroBarProps) {
  const reduceMotion = useReducedMotion()
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    const onScroll = () => {
      setCollapsed(window.scrollY > COLLAPSE_AT)
    }
    onScroll()
    window.addEventListener("scroll", onScroll, { passive: true })
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-start justify-between gap-6 px-4 sm:px-6 lg:px-8">
        <motion.div
          className="min-w-0 overflow-hidden"
          animate={{
            paddingTop: collapsed ? 14 : 32,
            paddingBottom: collapsed ? 14 : 28,
          }}
          transition={reduceMotion ? { duration: 0 } : { duration: 0.32, ease: EASE }}
        >
          {/* font-size animates via CSS, not framer-motion: the expanded size
              comes from the global h1 clamp() in px while the collapsed target
              is in rem, and framer interpolates mismatched units numerically
              (56px → 1.25rem starts at 56rem ≈ 900px — a giant flash). CSS
              transitions interpolate computed values, so px↔rem is safe. No
              reduceMotion gate needed here: the global reduced-motion media
              query zeroes transition-duration with !important, which CSS
              transitions honor (framer's JS-driven animations do not). */}
          <h1
            className="min-w-0 truncate text-foreground"
            style={{
              fontFamily: "var(--font-display)",
              fontWeight: 280,
              fontSize: collapsed ? "1.25rem" : undefined,
              transition: `font-size 0.32s cubic-bezier(${EASE.join(", ")}), color 200ms ease`,
            }}
          >
            How many GPUs to train an LLM?
          </h1>
          <motion.p
            className="mt-3 max-w-2xl text-sm leading-relaxed text-muted"
            animate={{
              opacity: collapsed ? 0 : 1,
              height: collapsed ? 0 : "auto",
              marginTop: collapsed ? 0 : 12,
            }}
            transition={reduceMotion ? { duration: 0 } : { duration: 0.28, ease: EASE }}
          >
            Estimate the GPUs, cost, and wall-clock time for any pretraining or
            fine-tuning run — and see exactly why.
          </motion.p>
        </motion.div>

        <div className="flex shrink-0 items-center gap-2.5 pt-7">
          <DenseViewToggle value={denseView} onChange={onDenseViewChange} />
          {themeToggle}
        </div>
      </div>
    </header>
  )
}

// ---------------------------------------------------------------------------
// DenseViewToggle — "Dense view" (off) ⇄ "Calm view" (on) switch with a
// hover/focus tooltip. Page chrome, so it uses Tailwind semantic tokens.
// ---------------------------------------------------------------------------
function DenseViewToggle({
  value,
  onChange,
}: {
  value: boolean
  onChange: (next: boolean) => void
}) {
  const [showTip, setShowTip] = useState(false)
  const label = value ? "Calm view" : "Dense view"

  return (
    <div
      className="relative inline-flex"
      onMouseEnter={() => setShowTip(true)}
      onMouseLeave={() => setShowTip(false)}
    >
      <button
        type="button"
        onClick={() => onChange(!value)}
        onFocus={() => setShowTip(true)}
        onBlur={() => setShowTip(false)}
        aria-pressed={value}
        aria-label={label}
        className="no-theme-transition flex h-9 items-center gap-2 rounded-xl border border-border bg-surface-elevated/70 px-3 text-xs font-medium text-muted backdrop-blur-sm hover:-translate-y-px hover:border-accent/30 hover:text-accent hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
        style={{
          transition:
            "transform 200ms cubic-bezier(0.22, 1, 0.36, 1), box-shadow 200ms ease, color 200ms ease, border-color 200ms ease",
        }}
      >
        <span
          aria-hidden
          className="h-2 w-2 rounded-full"
          style={{
            backgroundColor: value ? "var(--accent)" : "var(--border)",
          }}
        />
        {label}
      </button>
      {showTip && (
        <span
          role="tooltip"
          className="pointer-events-none absolute right-0 top-full z-50 mt-2 w-56 rounded-lg border border-border bg-surface-elevated px-3 py-2 text-[11px] leading-relaxed text-foreground shadow-lg backdrop-blur-sm"
        >
          Expand every layer and tighten spacing for auditing.
        </span>
      )}
    </div>
  )
}
