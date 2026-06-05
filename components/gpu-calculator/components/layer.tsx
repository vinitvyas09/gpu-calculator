"use client"

import { useContext, type ReactNode } from "react"
import type { LucideIcon } from "lucide-react"
import {
  CollapsibleSection,
  type CalculatorColors,
  type CollapsibleWarningSeverity,
} from "./input-controls"
import { LayerStackContext } from "./layer-stack"

export type Density = "comfortable" | "compact"

// ---------------------------------------------------------------------------
// Layer — a thin wrapper over the extended CollapsibleSection.
//
// Adds a stable `id` (for persistence + ⌘K targeting) and reads
// expandAll + density off the LayerStack context:
//   - effective open = expandAll ? true : (controlled `open`)
//   - density flows from context unless explicitly overridden on the Layer
//
// When the user toggles while expandAll forces the layer open, onOpenChange
// still fires — the HOST decides what that means (e.g. it may turn expandAll
// off, or persist the intent). Layer never resolves that policy itself.
// ---------------------------------------------------------------------------

export interface LayerProps {
  /** Stable key for persistence + control-registry/⌘K targeting. Unique per layer. */
  id: string
  title: string
  colors: CalculatorColors
  children: ReactNode

  /** Rendered in the header when collapsed; a sentence derived from CalculatorOutput. */
  summary?: ReactNode
  /** Count of owned warnings; renders a severity-tinted chip. 0 ⇒ no chip. */
  warningCount?: number
  /** Highest severity among owned warnings; tints the chip. */
  warningSeverity?: CollapsibleWarningSeverity

  /** Visually de-emphasized until relevant (MoE-style). Header stays clickable. */
  dimmed?: boolean
  /** Overrides the density inherited from LayerStack context. */
  density?: Density
  /** Back-compat badge pill (e.g. "MoE"). */
  badge?: string
  /** Optional leading lucide icon. */
  icon?: LucideIcon

  /**
   * Controlled open-state. When provided, the parent owns persistence
   * (perLayerOpen map). When omitted, falls back to defaultOpen + internal
   * state (CollapsibleSection's behavior).
   */
  open?: boolean
  onOpenChange?: (next: boolean) => void
  /** Used only in uncontrolled mode. Mirrors CollapsibleSection.defaultOpen. */
  defaultOpen?: boolean
}

export function Layer({
  id,
  title,
  colors,
  children,
  summary,
  warningCount,
  warningSeverity,
  dimmed,
  density,
  badge,
  icon,
  open,
  onOpenChange,
  defaultOpen,
}: LayerProps) {
  const { expandAll, density: contextDensity } = useContext(LayerStackContext)

  // expandAll forces the layer open (controlled). A toggle still calls
  // onOpenChange (next = !true = false) so the host can react; the host owns
  // the policy of what toggling-while-forced means.
  const effectiveOpen = expandAll ? true : open

  return (
    <div data-layer-id={id}>
      <CollapsibleSection
        title={title}
        colors={colors}
        summary={summary}
        warningCount={warningCount}
        warningSeverity={warningSeverity}
        dimmed={dimmed}
        density={density ?? contextDensity}
        badge={badge}
        icon={icon}
        open={effectiveOpen}
        onOpenChange={onOpenChange}
        defaultOpen={defaultOpen}
      >
        {children}
      </CollapsibleSection>
    </div>
  )
}

// ---------------------------------------------------------------------------
// LayerHostProps — the single `host` prop the panels receive.
//
// The host (GpuCalculator) owns all per-layer state and the pre-rendered
// fragments; panels read these to wire each <Layer/> without touching the math
// or owning persistence.
// ---------------------------------------------------------------------------
export interface LayerHostProps {
  isLayerOpen(id: string): boolean
  onLayerOpenChange(id: string, open: boolean): void
  expandAll: boolean
  density: Density
  /** Closed-state summary per layer id. */
  summaries: Record<string, ReactNode>
  /** Warning chip descriptor per layer id (undefined ⇒ no chip). */
  warningChips: Record<
    string,
    { count: number; severity: "info" | "warning" | "critical" } | undefined
  >
  /** Pre-rendered warning / info callouts per layer id (host builds them). */
  warningSlots: Record<string, ReactNode>
  /** Pre-rendered result fragments per layer id (host builds them). */
  outputSlots: Record<string, ReactNode>
}
