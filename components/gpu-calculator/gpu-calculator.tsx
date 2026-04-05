"use client"

import { useState, useEffect, useMemo } from "react"
import { useTheme } from "next-themes"
import type { CalculatorTab } from "./types"

export default function GpuCalculator() {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  const { resolvedTheme } = useTheme()
  const isDark = mounted && resolvedTheme === "dark"

  const [activeTab, setActiveTab] = useState<CalculatorTab>("pretraining")

  const colors = useMemo(
    () => ({
      bg: isDark ? "#1a1a2e" : "#f8f9fa",
      cardBg: isDark ? "#16213e" : "#ffffff",
      text: isDark ? "#e0e0e0" : "#1a1a2e",
      textSecondary: isDark ? "#a0a0b0" : "#6b7280",
      border: isDark ? "#2a2a4a" : "#e5e7eb",
      accent: isDark ? "#818cf8" : "#4f46e5",
      accentMuted: isDark ? "#818cf820" : "#4f46e510",
    }),
    [isDark]
  )

  const tabs: { key: CalculatorTab; label: string }[] = [
    { key: "pretraining", label: "Pretraining" },
    { key: "post-training", label: "Post-Training" },
  ]

  if (!mounted) {
    return (
      <div className="min-h-[400px] flex items-center justify-center">
        <div className="animate-pulse text-gray-400">Loading...</div>
      </div>
    )
  }

  return (
    <div
      className="rounded-xl border overflow-hidden"
      style={{
        backgroundColor: colors.cardBg,
        borderColor: colors.border,
        color: colors.text,
      }}
    >
      {/* Tab Navigation */}
      <div
        className="flex border-b"
        style={{ borderColor: colors.border }}
      >
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className="px-6 py-3 text-sm font-medium transition-colors relative"
            style={{
              color:
                activeTab === tab.key ? colors.accent : colors.textSecondary,
              backgroundColor:
                activeTab === tab.key ? colors.accentMuted : "transparent",
            }}
          >
            {tab.label}
            {activeTab === tab.key && (
              <span
                className="absolute bottom-0 left-0 right-0 h-0.5"
                style={{ backgroundColor: colors.accent }}
              />
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {activeTab === "pretraining" ? (
          <div className="text-center py-12" style={{ color: colors.textSecondary }}>
            <p className="text-lg font-medium" style={{ color: colors.text }}>
              Pretraining Calculator
            </p>
            <p className="mt-2 text-sm">
              Configure your model architecture, training parameters, and hardware to estimate GPU
              requirements.
            </p>
          </div>
        ) : (
          <div className="text-center py-12" style={{ color: colors.textSecondary }}>
            <p className="text-lg font-medium" style={{ color: colors.text }}>
              Post-Training Calculator
            </p>
            <p className="mt-2 text-sm">
              Estimate resources for SFT, DPO, PPO, and GRPO fine-tuning.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
