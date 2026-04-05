"use client"

import dynamic from "next/dynamic"

const GpuCalculator = dynamic(
  () => import("@/components/gpu-calculator/gpu-calculator"),
  {
    ssr: false,
    loading: () => (
      <div className="rounded-2xl border border-border bg-surface/70 p-8 backdrop-blur-sm">
        <div className="animate-pulse space-y-5">
          <div className="h-6 w-40 rounded-full bg-accent-soft" />
          <div className="h-14 rounded-xl bg-surface" />
          <div className="grid gap-5 lg:grid-cols-2">
            <div className="h-72 rounded-xl bg-surface" />
            <div className="h-72 rounded-xl bg-surface" />
          </div>
        </div>
      </div>
    ),
  }
)

export default function GpuCalculatorEmbed() {
  return <GpuCalculator />
}
