"use client"

import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ChevronDown, Info, Check, type LucideIcon } from "lucide-react"

// ---------------------------------------------------------------------------
// Shared color palette — produced once in gpu-calculator.tsx, threaded down
// ---------------------------------------------------------------------------
export interface CalculatorColors {
  bg: string
  cardBg: string
  text: string
  textSecondary: string
  border: string
  accent: string
  accentMuted: string
  panel: string
  warning: string
  warningBg: string
  warningBorder: string
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------
function trimTrailingZeros(s: string): string {
  return s.replace(/\.?0+$/, "")
}

export function formatCompact(n: number): string {
  if (n === 0) return "0"
  const abs = Math.abs(n)
  if (abs >= 1e12) return `${trimTrailingZeros((n / 1e12).toFixed(2))}T`
  if (abs >= 1e9) return `${trimTrailingZeros((n / 1e9).toFixed(2))}B`
  if (abs >= 1e6) return `${trimTrailingZeros((n / 1e6).toFixed(2))}M`
  if (abs >= 1e3) return `${trimTrailingZeros((n / 1e3).toFixed(1))}K`
  return String(n)
}

const DECIMAL_NUMBER_PATTERN_SOURCE =
  String.raw`[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?`
const DECIMAL_NUMBER_PATTERN = new RegExp(
  `^${DECIMAL_NUMBER_PATTERN_SOURCE}$`,
)
const COMPACT_NUMBER_PATTERN = new RegExp(
  `^(${DECIMAL_NUMBER_PATTERN_SOURCE})\\s*([KMBT])?$`,
  "i",
)

function parseDecimalNumber(str: string): number | null {
  const s = str.trim()
  if (!s || !DECIMAL_NUMBER_PATTERN.test(s)) return null
  const n = Number(s)
  return Number.isFinite(n) ? n : null
}

export function parseCompactNumber(str: string): number | null {
  const s = str.trim().replace(/,/g, "")
  if (!s) return null
  const match = s.match(COMPACT_NUMBER_PATTERN)
  if (match) {
    const base = parseDecimalNumber(match[1])
    if (base === null) return null
    const suffix = (match[2] || "").toUpperCase()
    const mult: Record<string, number> = { K: 1e3, M: 1e6, B: 1e9, T: 1e12 }
    const value = base * (mult[suffix] || 1)
    return Number.isFinite(value) ? value : null
  }
  return null
}

export function formatPercent(n: number, decimals = 0): string {
  return `${(n * 100).toFixed(decimals)}%`
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
function InputLabel({
  label,
  tooltip,
  htmlFor,
  colors,
}: {
  label: string
  tooltip?: string
  htmlFor?: string
  colors: CalculatorColors
}) {
  return (
    <div className="flex items-center gap-1.5">
      <label
        htmlFor={htmlFor}
        className="text-[11px] font-medium uppercase tracking-[0.08em]"
        style={{ color: colors.textSecondary }}
      >
        {label}
      </label>
      {tooltip && <TooltipIcon content={tooltip} colors={colors} />}
    </div>
  )
}

function focusRing(
  e: React.FocusEvent<HTMLElement>,
  colors: CalculatorColors,
) {
  e.currentTarget.style.borderColor = colors.accent
  e.currentTarget.style.boxShadow = `0 0 0 2px ${colors.accentMuted}`
}

function blurRing(
  e: React.FocusEvent<HTMLElement>,
  colors: CalculatorColors,
) {
  e.currentTarget.style.borderColor = colors.border
  e.currentTarget.style.boxShadow = "none"
}

// ---------------------------------------------------------------------------
// Stat — tiny label/value pair used in spec cards
// ---------------------------------------------------------------------------
export function Stat({
  label,
  value,
  colors,
}: {
  label: string
  value: string
  colors: CalculatorColors
}) {
  return (
    <div>
      <div
        className="text-[10px] font-medium uppercase tracking-[0.08em]"
        style={{ color: colors.textSecondary }}
      >
        {label}
      </div>
      <div className="mt-0.5 text-xs font-semibold" style={{ color: colors.text }}>
        {value}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// TooltipIcon
// ---------------------------------------------------------------------------
export function TooltipIcon({
  content,
  colors,
}: {
  content: string
  colors: CalculatorColors
}) {
  const [show, setShow] = useState(false)

  return (
    <div className="relative inline-flex">
      <button
        type="button"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onFocus={() => setShow(true)}
        onBlur={() => setShow(false)}
        className="inline-flex items-center justify-center rounded-full p-0.5 opacity-60 transition-opacity hover:opacity-100"
        style={{ color: colors.textSecondary }}
        aria-label="More info"
      >
        <Info className="h-3 w-3" />
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
            {content}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ---------------------------------------------------------------------------
// NumberInput — 300 ms debounce, optional compact (B/M/T) formatting
// ---------------------------------------------------------------------------
export function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step,
  tooltip,
  unit,
  colors,
  disabled = false,
  compact = false,
  integer = false,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  tooltip?: string
  unit?: string
  colors: CalculatorColors
  disabled?: boolean
  compact?: boolean
  integer?: boolean
}) {
  const inputId = useId()
  const formatValue = useCallback(
    (n: number) => (compact ? formatCompact(n) : String(n)),
    [compact],
  )
  const parse = compact
    ? parseCompactNumber
    : parseDecimalNumber

  const [local, setLocal] = useState(() => formatValue(value))
  const [isEditing, setIsEditing] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(
    () => () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    },
    [],
  )

  const clamp = (n: number) => {
    let v = n
    if (integer) v = Math.round(v)
    if (min !== undefined) v = Math.max(min, v)
    if (max !== undefined) v = Math.min(max, v)
    return v
  }

  const commitValue = (raw: string) => {
    const parsed = parse(raw)
    if (parsed === null || !Number.isFinite(parsed)) {
      setLocal(formatValue(value))
      return null
    }

    const clamped = clamp(parsed)
    if (!Object.is(clamped, value)) {
      onChange(clamped)
    }
    return clamped
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value
    setIsEditing(true)
    setLocal(raw)
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      timerRef.current = null
      commitValue(raw)
    }, 300)
  }

  const handleBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    blurRing(e, colors)
    setIsEditing(false)
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    const committed = commitValue(local)
    // Always sync display to the authoritative value on blur
    setLocal(formatValue(committed !== null ? committed : value))
  }

  const displayValue = isEditing ? local : formatValue(value)

  return (
    <div className="space-y-1.5">
      <InputLabel
        label={label}
        tooltip={tooltip}
        htmlFor={inputId}
        colors={colors}
      />
      <div className="relative">
        <input
          id={inputId}
          type="text"
          inputMode={compact ? "text" : integer ? "numeric" : "decimal"}
          value={displayValue}
          onChange={handleChange}
          disabled={disabled}
          step={step}
          className="no-theme-transition w-full rounded-lg border px-3 py-2.5 text-sm tabular-nums focus:outline-none"
          style={{
            backgroundColor: colors.bg,
            borderColor: colors.border,
            color: disabled ? colors.textSecondary : colors.text,
            paddingRight: unit ? 44 : 12,
            opacity: disabled ? 0.5 : 1,
            fontFamily: "var(--font-mono)",
            transition: "border-color 200ms ease, box-shadow 200ms ease",
          }}
          onFocus={(e) => {
            setLocal(formatValue(value))
            setIsEditing(true)
            focusRing(e, colors)
          }}
          onBlur={handleBlur}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.currentTarget.blur()
            }
          }}
        />
        {unit && (
          <span
            className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[11px] font-medium"
            style={{ color: colors.textSecondary }}
          >
            {unit}
          </span>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// SliderInput
// ---------------------------------------------------------------------------
export function SliderInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  tooltip,
  unit,
  colors,
  disabled = false,
  formatDisplay,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step?: number
  tooltip?: string
  unit?: string
  colors: CalculatorColors
  disabled?: boolean
  formatDisplay?: (n: number) => string
}) {
  const inputId = useId()
  const pct =
    max > min
      ? Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100))
      : 0

  return (
    <div className="space-y-2" style={{ opacity: disabled ? 0.5 : 1 }}>
      <div className="flex items-center justify-between">
        <InputLabel
          label={label}
          tooltip={tooltip}
          htmlFor={inputId}
          colors={colors}
        />
        <span
          className="text-sm font-semibold tabular-nums"
          style={{ color: colors.text }}
        >
          {formatDisplay ? formatDisplay(value) : value}
          {unit ? ` ${unit}` : ""}
        </span>
      </div>
      <div className="relative flex h-6 items-center">
        {/* Track background */}
        <div
          className="absolute inset-x-0 h-[5px] rounded-full"
          style={{ backgroundColor: colors.border }}
        >
          {/* Filled portion */}
          <div
            className="h-full rounded-full transition-[width] duration-75"
            style={{ width: `${pct}%`, backgroundColor: colors.accent }}
          />
        </div>
        {/* Visual thumb */}
        <div
          className="pointer-events-none absolute h-[14px] w-[14px] -translate-x-1/2 rounded-full border-[2.5px] shadow-sm transition-[left] duration-75"
          style={{
            left: `${pct}%`,
            borderColor: colors.accent,
            backgroundColor: colors.cardBg,
          }}
        />
        {/* Native range input — invisible but handles interaction */}
        <input
          id={inputId}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          disabled={disabled}
          className="absolute inset-0 cursor-pointer opacity-0"
        />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// SelectInput — with optgroup support
// ---------------------------------------------------------------------------
export function SelectInput({
  label,
  value,
  onChange,
  options,
  tooltip,
  colors,
  disabled = false,
  placeholder,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string; group?: string }[]
  tooltip?: string
  colors: CalculatorColors
  disabled?: boolean
  placeholder?: string
}) {
  const inputId = useId()
  const groups = new Map<string, typeof options>()
  for (const opt of options) {
    const key = opt.group || ""
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(opt)
  }

  return (
    <div className="space-y-1.5">
      <InputLabel
        label={label}
        tooltip={tooltip}
        htmlFor={inputId}
        colors={colors}
      />
      <div className="relative">
        <select
          id={inputId}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="no-theme-transition w-full appearance-none rounded-lg border px-3 py-2.5 pr-8 text-sm focus:outline-none"
          style={{
            backgroundColor: colors.bg,
            borderColor: colors.border,
            color: disabled ? colors.textSecondary : colors.text,
            opacity: disabled ? 0.5 : 1,
            transition: "border-color 200ms ease, box-shadow 200ms ease",
          }}
          onFocus={(e) => focusRing(e, colors)}
          onBlur={(e) => blurRing(e, colors)}
        >
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}
          {[...groups.entries()].map(([group, opts]) =>
            group ? (
              <optgroup key={group} label={group}>
                {opts.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </optgroup>
            ) : (
              opts.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))
            ),
          )}
        </select>
        <ChevronDown
          className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2"
          style={{ color: colors.textSecondary }}
        />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// ToggleInput
// ---------------------------------------------------------------------------
export function ToggleInput({
  label,
  value,
  onChange,
  tooltip,
  colors,
  disabled = false,
}: {
  label: string
  value: boolean
  onChange: (v: boolean) => void
  tooltip?: string
  colors: CalculatorColors
  disabled?: boolean
}) {
  return (
    <div className="flex items-center justify-between gap-3 py-0.5">
      <InputLabel label={label} tooltip={tooltip} colors={colors} />
      <button
        type="button"
        role="switch"
        aria-checked={value}
        aria-label={label}
        onClick={() => !disabled && onChange(!value)}
        className="relative h-[22px] w-[38px] shrink-0 rounded-full transition-colors"
        style={{
          backgroundColor: value ? colors.accent : colors.border,
          opacity: disabled ? 0.4 : 1,
          cursor: disabled ? "not-allowed" : "pointer",
        }}
      >
        <motion.span
          className="absolute left-[3px] top-[3px] block h-4 w-4 rounded-full shadow-sm"
          style={{ backgroundColor: colors.cardBg }}
          animate={{ x: value ? 16 : 0 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
        />
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// CheckboxGroupInput — chip-style multi-select
// ---------------------------------------------------------------------------
export function CheckboxGroupInput<T extends string>({
  label,
  values,
  allOptions,
  onChange,
  tooltip,
  colors,
}: {
  label: string
  values: T[]
  allOptions: { value: T; label: string }[]
  onChange: (values: T[]) => void
  tooltip?: string
  colors: CalculatorColors
}) {
  const toggle = (v: T) => {
    onChange(
      values.includes(v) ? values.filter((x) => x !== v) : [...values, v],
    )
  }

  return (
    <div className="space-y-1.5">
      <InputLabel label={label} tooltip={tooltip} colors={colors} />
      <div className="flex flex-wrap gap-1.5">
        {allOptions.map((opt) => {
          const checked = values.includes(opt.value)
          return (
            <button
              key={opt.value}
              type="button"
              role="checkbox"
              aria-checked={checked}
              onClick={() => toggle(opt.value)}
              className="flex items-center gap-1.5 rounded-md border px-2 py-1.5 text-[11px] font-medium transition-all"
              style={{
                borderColor: checked ? colors.accent : colors.border,
                backgroundColor: checked ? colors.accentMuted : "transparent",
                color: checked ? colors.accent : colors.textSecondary,
              }}
            >
              <div
                className="flex h-3 w-3 items-center justify-center rounded-sm border"
                style={{
                  borderColor: checked ? colors.accent : colors.border,
                  backgroundColor: checked ? colors.accent : "transparent",
                }}
              >
                {checked && (
                  <Check className="h-2 w-2" style={{ color: colors.cardBg }} />
                )}
              </div>
              {opt.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// CollapsibleSection — animated expand/collapse
//
// Back-compat: the original 5 props (title, defaultOpen, children, colors,
// badge) behave exactly as before. The added props are all optional and enable
// the <Layer> wrapper (see layer.tsx) to drive controlled open-state, a
// closed-state summary, a warning chip, dimming, and density — without changing
// the aria / chevron-motion / AnimatePresence-collapse markup, which stays
// byte-identical.
// ---------------------------------------------------------------------------
export type CollapsibleDensity = "comfortable" | "compact"

export type CollapsibleWarningSeverity = "info" | "warning" | "critical"

// Inline error tone fallback for the critical warning chip (CalculatorColors
// carries warning tones but no error triplet; these mirror
// SEVERITY_META.critical in results-summary.tsx so chips match the warnings
// panel).
const CRITICAL_CHIP_TONE = {
  light: {
    bg: "oklch(0.97 0.04 25)",
    border: "oklch(0.9 0.09 25)",
    text: "oklch(0.49 0.18 25)",
  },
  dark: {
    bg: "oklch(0.23 0.07 25)",
    border: "oklch(0.39 0.1 25)",
    text: "oklch(0.8 0.15 25)",
  },
} as const

export function CollapsibleSection({
  title,
  defaultOpen = false,
  children,
  colors,
  badge,
  open: controlledOpen,
  onOpenChange,
  summary,
  warningCount = 0,
  warningSeverity = "warning",
  dimmed = false,
  density = "comfortable",
  icon: Icon,
}: {
  title: string
  defaultOpen?: boolean
  children: ReactNode
  colors: CalculatorColors
  badge?: string
  /** Controlled open-state. When provided, the parent owns open/close (and
   *  persistence); when omitted, falls back to internal useState(defaultOpen). */
  open?: boolean
  onOpenChange?: (next: boolean) => void
  /** Rendered in the header ONLY when closed — muted, truncating, right of the title. */
  summary?: ReactNode
  /** Count of owned warnings; renders a severity-tinted chip when > 0. */
  warningCount?: number
  warningSeverity?: CollapsibleWarningSeverity
  /** Visually de-emphasizes the whole section (~0.55 opacity); header stays clickable. */
  dimmed?: boolean
  /** comfortable (default) | compact — tightens header + body padding. */
  density?: CollapsibleDensity
  /** Optional leading lucide icon component. */
  icon?: LucideIcon
}) {
  const [internalOpen, setInternalOpen] = useState(defaultOpen)
  const isControlled = controlledOpen !== undefined
  const open = isControlled ? controlledOpen : internalOpen
  const contentId = useId()

  const toggle = () => {
    const next = !open
    if (!isControlled) setInternalOpen(next)
    onOpenChange?.(next)
  }

  const compact = density === "compact"
  const headerPad = compact ? "px-3 py-2.5" : "px-4 py-3.5"
  const bodyPad = compact ? "px-4 pb-3 pt-2" : "px-4 pb-4 pt-3"

  // Warning chip tone: warning severity reuses CalculatorColors tones; info uses
  // accent tones; critical uses the inline error fallback above (dark/light is
  // not threaded here, so the critical chip leans on the dark-leaning literal
  // that reads on both backgrounds — bg carries its own contrast).
  let chipTone: { bg: string; border: string; text: string }
  if (warningSeverity === "info") {
    chipTone = {
      bg: colors.accentMuted,
      border: colors.accentMuted,
      text: colors.accent,
    }
  } else if (warningSeverity === "critical") {
    chipTone = CRITICAL_CHIP_TONE.light
  } else {
    chipTone = {
      bg: colors.warningBg,
      border: colors.warningBorder,
      text: colors.warning,
    }
  }

  return (
    <div
      className="rounded-lg border"
      style={{
        borderColor: colors.border,
        backgroundColor: open ? colors.bg : "transparent",
        opacity: dimmed ? 0.55 : 1,
        transition: "opacity 150ms ease",
      }}
    >
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        aria-controls={contentId}
        className={`no-theme-transition flex w-full items-center justify-between gap-3 ${headerPad} text-left`}
        style={{ transition: "opacity 150ms ease" }}
      >
        <div className="flex min-w-0 items-center gap-2">
          {Icon && (
            <Icon
              className="h-4 w-4 shrink-0"
              style={{ color: colors.textSecondary }}
            />
          )}
          <span
            className="shrink-0 text-sm font-medium"
            style={{ color: colors.text }}
          >
            {title}
          </span>
          {badge && (
            <span
              className="shrink-0 rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em]"
              style={{
                backgroundColor: colors.accentMuted,
                color: colors.accent,
              }}
            >
              {badge}
            </span>
          )}
          {warningCount > 0 && (
            <span
              className="shrink-0 rounded-full border px-1.5 py-0.5 text-[9px] font-semibold tabular-nums"
              style={{
                backgroundColor: chipTone.bg,
                borderColor: chipTone.border,
                color: chipTone.text,
              }}
            >
              ⚠ {warningCount}
            </span>
          )}
          {!open && summary && (
            <span
              className="min-w-0 truncate text-[11px] font-normal"
              style={{ color: colors.textSecondary }}
            >
              {summary}
            </span>
          )}
        </div>
        <motion.div
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown
            className="h-4 w-4"
            style={{ color: colors.textSecondary }}
          />
        </motion.div>
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
            <div
              className={`border-t ${bodyPad}`}
              style={{ borderColor: colors.border }}
            >
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
