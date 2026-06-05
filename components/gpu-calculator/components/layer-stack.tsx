"use client"

import { createContext, type ReactNode } from "react"
import type { CalculatorColors } from "./input-controls"
import type { Density } from "./layer"

// ---------------------------------------------------------------------------
// LayerStack — ordering + global expand-all + density broadcaster.
//
// It does NOT own layer content: it lays out its <Layer> children in order
// inside a simple vertical stack and broadcasts { expandAll, density } via
// context so each Layer can derive its effective open-state + padding without
// prop-drilling. No reordering logic — children render in the order given.
// ---------------------------------------------------------------------------

export interface LayerStackContextValue {
  /** When true, every child Layer is forced open (overrides controlled open). */
  expandAll: boolean
  density: Density
}

export const LayerStackContext = createContext<LayerStackContextValue>({
  expandAll: false,
  density: "comfortable",
})

export interface LayerStackProps {
  colors: CalculatorColors
  /** Ordered <Layer/> elements. */
  children: ReactNode
  /** Global "expand all" pulse; when true, all child Layers honor it. */
  expandAll: boolean
  density: Density
}

export function LayerStack({
  colors,
  children,
  expandAll,
  density,
}: LayerStackProps) {
  return (
    <LayerStackContext.Provider value={{ expandAll, density }}>
      <div
        data-density={density}
        className={density === "compact" ? "space-y-2" : "space-y-3"}
        style={{ color: colors.text }}
      >
        {children}
      </div>
    </LayerStackContext.Provider>
  )
}
