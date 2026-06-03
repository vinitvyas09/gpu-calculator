const KILO = 1e3
const MEGA = 1e6
const GIGA = 1e9
const TERA = 1e12
const PETA = 1e15
const EXA = 1e18
const ZETTA = 1e21

function roundedBoundary(scale: number, digits: number): number {
  // Switch units before toFixed would round the lower unit to 1000.
  return (1000 - 0.5 / 10 ** digits) * scale
}

export function formatMemory(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "--"
  }

  const tb = bytes / TERA
  const gb = bytes / GIGA
  const mb = bytes / MEGA
  const kb = bytes / KILO

  if (bytes === 0) return "0 B"
  if (tb >= 1) return `${tb.toFixed(tb >= 10 ? 1 : 2)} TB`
  if (gb >= 999.5) return `${(gb / 1000).toFixed(2)} TB`
  if (gb >= 100) return `${Math.round(gb)} GB`
  if (gb >= 10) return `${gb.toFixed(1)} GB`
  if (gb >= 1) return `${gb.toFixed(2)} GB`
  if (mb >= 999.5) return `${(mb / 1000).toFixed(2)} GB`
  if (mb >= 1) return `${mb.toFixed(0)} MB`
  if (kb >= 999.5) return `${(kb / 1000).toFixed(0)} MB`
  if (kb >= 1) return `${kb.toFixed(0)} KB`
  return "< 1 KB"
}

export function formatCount(value: number): string {
  if (!Number.isFinite(value)) {
    return "--"
  }

  const absolute = Math.abs(value)

  if (absolute >= roundedBoundary(GIGA, 2)) {
    return `${(value / TERA).toFixed(2)}T`
  }
  if (absolute >= roundedBoundary(MEGA, 2)) {
    return `${(value / GIGA).toFixed(2)}B`
  }
  if (absolute >= roundedBoundary(KILO, 1)) {
    return `${(value / MEGA).toFixed(2)}M`
  }
  if (absolute >= KILO) return `${(value / KILO).toFixed(1)}K`
  return value.toLocaleString()
}

export function formatFLOPs(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "--"
  }

  if (value >= roundedBoundary(EXA, 2)) {
    return `${(value / ZETTA).toFixed(2)} ZFLOPs`
  }
  if (value >= roundedBoundary(PETA, 2)) {
    return `${(value / EXA).toFixed(2)} EFLOPs`
  }
  if (value >= roundedBoundary(TERA, 2)) {
    return `${(value / PETA).toFixed(2)} PFLOPs`
  }
  if (value >= roundedBoundary(GIGA, 2)) {
    return `${(value / TERA).toFixed(2)} TFLOPs`
  }
  return `${(value / GIGA).toFixed(2)} GFLOPs`
}

export function formatCost(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "--"
  }

  if (value >= roundedBoundary(GIGA, 2)) {
    return `$${(value / TERA).toFixed(2)}T`
  }
  if (value >= roundedBoundary(MEGA, 2)) {
    return `$${(value / GIGA).toFixed(2)}B`
  }
  if (value >= MEGA - 0.5) return `$${(value / MEGA).toFixed(2)}M`
  if (value >= KILO - 0.005) return `$${Math.round(value).toLocaleString()}`
  return `$${value.toFixed(2)}`
}

export function formatDuration(hours: number): string {
  if (!Number.isFinite(hours) || hours < 0) {
    return "--"
  }

  if (hours === 0) return "0 min"
  if (hours >= (365 - 0.05) * 24) {
    return `${(hours / (24 * 365)).toFixed(1)} years`
  }
  if (hours >= 24 - 0.05) return `${(hours / 24).toFixed(1)} days`
  if (hours >= 1 - 0.5 / 60) return `${hours.toFixed(1)} hr`

  const minutes = hours * 60
  return minutes >= 1 ? `${Math.round(minutes)} min` : "< 1 min"
}

export function formatFractionPercent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) {
    return "--"
  }

  return `${(value * 100).toFixed(digits)}%`
}

export function formatPercent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) {
    return "--"
  }

  return `${value.toFixed(digits)}%`
}

export function formatMultiplier(value: number): string {
  if (!Number.isFinite(value)) {
    return "--"
  }

  return `${value.toFixed(2)}x`
}
