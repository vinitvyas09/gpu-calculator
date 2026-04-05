"use client"

import dynamic from "next/dynamic"

const GpuCalculator = dynamic(
  () => import("@/components/gpu-calculator/gpu-calculator"),
  {
    ssr: false,
    loading: () => (
      <div className="rounded-3xl border border-border bg-surface/80 p-8 shadow-sm backdrop-blur">
        <div className="animate-pulse space-y-4">
          <div className="h-6 w-40 rounded-full bg-accent-soft" />
          <div className="h-14 rounded-2xl bg-background/70" />
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="h-72 rounded-2xl bg-background/70" />
            <div className="h-72 rounded-2xl bg-background/70" />
          </div>
        </div>
      </div>
    ),
  }
)

export default function GpuCalculatorEmbed() {
  return <GpuCalculator />
}
