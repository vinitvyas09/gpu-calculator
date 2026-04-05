"use client"

import dynamic from "next/dynamic"

const GpuCalculator = dynamic(
  () => import("../../../components/gpu-calculator/gpu-calculator"),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-pulse text-gray-400 dark:text-gray-500">
          Loading calculator...
        </div>
      </div>
    ),
  }
)

export default function GpuCalculatorEmbed() {
  return <GpuCalculator />
}
