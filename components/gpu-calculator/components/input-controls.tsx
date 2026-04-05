"use client"

import {
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ChevronDown, Info, Check } from "lucide-react"

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

export function parseCompactNumber(str: string): number | null {
  const s = str.trim().replace(/,/g, "")
  if (!s) return null
  const match = s.match(/^([+-]?\d+\.?\d*)\s*([KMBT])?$/i)
  if (match) {
    const base = parseFloat(match[1])
    const suffix = (match[2] || "").toUpperCase()
    const mult: Record<string, number> = { K: 1e3, M: 1e6, B: 1e9, T: 1e12 }
    return base * (mult[suffix] || 1)
  }
  const n = Number(s)
  return isNaN(n) ? null : n
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
        className="text-[11px] font-semibold uppercase tracking-[0.06em]"
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
}) {
  const inputId = useId()
  const formatValue = (n: number) => (compact ? formatCompact(n) : String(n))
  const parse = compact
    ? parseCompactNumber
    : (s: string) => {
        const n = Number(s)
        return isNaN(n) ? null : n
      }

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
    if (min !== undefined) v = Math.max(min, v)
    if (max !== undefined) v = Math.min(max, v)
    return v
  }

  const commitValue = (raw: string) => {
    const parsed = parse(raw)
    if (parsed === null) {
      setLocal(formatValue(value))
      return null
    }

    const clamped = clamp(parsed)
    onChange(clamped)
    setLocal(formatValue(clamped))
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
    if (commitValue(local) === null) {
      setLocal(formatValue(value))
    }
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
          inputMode={compact ? "text" : "decimal"}
          value={displayValue}
          onChange={handleChange}
          disabled={disabled}
          step={step}
          className="w-full rounded-lg border px-3 py-2 text-sm tabular-nums transition-colors focus:outline-none"
          style={{
            backgroundColor: colors.bg,
            borderColor: colors.border,
            color: disabled ? colors.textSecondary : colors.text,
            paddingRight: unit ? 44 : 12,
            opacity: disabled ? 0.5 : 1,
          }}
          onFocus={(e) => {
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
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0

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
          className="w-full appearance-none rounded-lg border px-3 py-2 pr-8 text-sm transition-colors focus:outline-none"
          style={{
            backgroundColor: colors.bg,
            borderColor: colors.border,
            color: disabled ? colors.textSecondary : colors.text,
            opacity: disabled ? 0.5 : 1,
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
// ---------------------------------------------------------------------------
export function CollapsibleSection({
  title,
  defaultOpen = false,
  children,
  colors,
  badge,
}: {
  title: string
  defaultOpen?: boolean
  children: ReactNode
  colors: CalculatorColors
  badge?: string
}) {
  const [open, setOpen] = useState(defaultOpen)
  const contentId = useId()

  return (
    <div
      className="rounded-xl border"
      style={{
        borderColor: colors.border,
        backgroundColor: open ? colors.bg : "transparent",
      }}
    >
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls={contentId}
        className="flex w-full items-center justify-between px-4 py-3 text-left transition-opacity hover:opacity-80"
      >
        <div className="flex items-center gap-2">
          <span
            className="text-sm font-medium"
            style={{ color: colors.text }}
          >
            {title}
          </span>
          {badge && (
            <span
              className="rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em]"
              style={{
                backgroundColor: colors.accentMuted,
                color: colors.accent,
              }}
            >
              {badge}
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
              className="border-t px-4 pb-4 pt-3"
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
