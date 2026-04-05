"use client"

import { useEffect, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { useTheme } from "next-themes"
import {
  Binary,
  Boxes,
  Gauge,
  Layers3,
  Server,
  SlidersHorizontal,
} from "lucide-react"
import {
  DEFAULT_POST_TRAINING_CONFIG,
  DEFAULT_TRAINING_CONFIG,
  GPU_SPECS,
  MODEL_PRESETS,
  OPTIMIZER_PROFILES,
} from "./constants"
import type { CalculatorTab } from "./types"

const tabs: { key: CalculatorTab; label: string; description: string }[] = [
  {
    key: "pretraining",
    label: "Pretraining",
    description: "Foundation shell for the main cluster planning workflow.",
  },
  {
    key: "post-training",
    label: "Post-Training",
    description: "SFT, DPO, PPO, and GRPO inputs land here next.",
  },
]

export default function GpuCalculator() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    // Required by the app's dark/light hydration pattern from the spec.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true)
  }, [])

  const { resolvedTheme } = useTheme()
  const isDark = mounted && resolvedTheme === "dark"

  const [activeTab, setActiveTab] = useState<CalculatorTab>("pretraining")

  const colors = useMemo(
    () => ({
      bg: isDark ? "#1a1a2e" : "#f8f9fa",
      cardBg: isDark ? "#16213e" : "#ffffff",
      text: isDark ? "#e0e0e0" : "#1a1a2e",
      textSecondary: isDark ? "#a0a0b0" : "#5d6676",
      border: isDark ? "#2a2a4a" : "#dde3ec",
      accent: isDark ? "#83b6ff" : "#1d5fe4",
      accentMuted: isDark ? "rgba(131, 182, 255, 0.14)" : "rgba(29, 95, 228, 0.08)",
      panel: isDark ? "rgba(13, 18, 37, 0.72)" : "rgba(245, 247, 250, 0.92)",
    }),
    [isDark]
  )

  const stats = [
    {
      label: "GPU Presets",
      value: GPU_SPECS.length,
      icon: Server,
    },
    {
      label: "Model Presets",
      value: MODEL_PRESETS.length,
      icon: Layers3,
    },
    {
      label: "Optimizer Profiles",
      value: OPTIMIZER_PROFILES.length,
      icon: Boxes,
    },
  ]

  if (!mounted) {
    return (
      <div className="rounded-3xl border border-border bg-surface/80 p-8 shadow-sm backdrop-blur">
        <div className="animate-pulse space-y-4">
          <div className="h-8 w-48 rounded-full bg-accent-soft" />
          <div className="h-16 rounded-2xl bg-background/70" />
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="h-72 rounded-2xl bg-background/70" />
            <div className="h-72 rounded-2xl bg-background/70" />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className="overflow-hidden rounded-[2rem] border shadow-sm"
      style={{
        backgroundColor: colors.cardBg,
        borderColor: colors.border,
        color: colors.text,
      }}
    >
      <div
        className="border-b px-6 py-6 sm:px-8"
        style={{
          borderColor: colors.border,
          background: `linear-gradient(135deg, ${colors.accentMuted}, transparent 65%)`,
        }}
      >
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-2xl">
            <p className="text-sm font-medium uppercase tracking-[0.24em]" style={{ color: colors.accent }}>
              Phase 1 Scaffold
            </p>
            <h2 className="mt-3 text-3xl" style={{ fontFamily: "var(--font-display)" }}>
              Tool shell, data tables, and type surface are wired in.
            </h2>
            <p className="mt-3 text-sm leading-6" style={{ color: colors.textSecondary }}>
              The calculator logic and full input panels land in later phases. This view
              intentionally exercises the theme pattern, tab shell, and spec-backed
              constants without inventing unfinished math.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            {stats.map(({ label, value, icon: Icon }) => (
              <div
                key={label}
                className="rounded-2xl border px-4 py-3"
                style={{
                  borderColor: colors.border,
                  backgroundColor: colors.panel,
                }}
              >
                <div className="flex items-center gap-2 text-xs uppercase tracking-[0.22em]" style={{ color: colors.textSecondary }}>
                  <Icon className="h-4 w-4" />
                  {label}
                </div>
                <div className="mt-3 text-2xl font-semibold">{value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="border-b px-4 py-4 sm:px-6" style={{ borderColor: colors.border }}>
        <div className="flex flex-wrap gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className="rounded-full px-4 py-2 text-sm font-medium transition-colors"
              style={{
                color: activeTab === tab.key ? colors.accent : colors.textSecondary,
                backgroundColor: activeTab === tab.key ? colors.accentMuted : "transparent",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <p className="mt-3 text-sm" style={{ color: colors.textSecondary }}>
          {tabs.find((tab) => tab.key === activeTab)?.description}
        </p>
      </div>

      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.24, ease: "easeOut" }}
        className="grid gap-4 p-4 sm:p-6 lg:grid-cols-[1.1fr_0.9fr]"
      >
        <section
          className="rounded-[1.5rem] border p-5"
          style={{ borderColor: colors.border, backgroundColor: colors.panel }}
        >
          <div className="flex items-center gap-2 text-sm font-medium" style={{ color: colors.accent }}>
            <SlidersHorizontal className="h-4 w-4" />
            Input Surface
          </div>
          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <PreviewCard
              title="Default model"
              value={
                activeTab === "pretraining"
                  ? DEFAULT_TRAINING_CONFIG.model.presetId ?? "custom"
                  : DEFAULT_POST_TRAINING_CONFIG.baseModel.presetId ?? "custom"
              }
              subtitle={
                activeTab === "pretraining"
                  ? `${DEFAULT_TRAINING_CONFIG.sequenceLength.toLocaleString()} seq, ${DEFAULT_TRAINING_CONFIG.precision}`
                  : `${DEFAULT_POST_TRAINING_CONFIG.sequenceLength.toLocaleString()} seq, ${DEFAULT_POST_TRAINING_CONFIG.approach}`
              }
              colors={colors}
              icon={Binary}
            />
            <PreviewCard
              title="Hardware baseline"
              value={
                activeTab === "pretraining"
                  ? DEFAULT_TRAINING_CONFIG.hardware.gpu.name
                  : DEFAULT_POST_TRAINING_CONFIG.hardware.gpu.name
              }
              subtitle={`${
                activeTab === "pretraining"
                  ? DEFAULT_TRAINING_CONFIG.hardware.numGPUs ?? 1
                  : DEFAULT_POST_TRAINING_CONFIG.hardware.numGPUs
              } GPU${
                (activeTab === "pretraining"
                  ? DEFAULT_TRAINING_CONFIG.hardware.numGPUs ?? 1
                  : DEFAULT_POST_TRAINING_CONFIG.hardware.numGPUs) > 1
                  ? "s"
                  : ""
              }`}
              colors={colors}
              icon={Server}
            />
            <PreviewCard
              title="Optimization path"
              value={
                activeTab === "pretraining"
                  ? DEFAULT_TRAINING_CONFIG.optimizer
                  : DEFAULT_POST_TRAINING_CONFIG.optimizer
              }
              subtitle={
                activeTab === "pretraining"
                  ? DEFAULT_TRAINING_CONFIG.activationCheckpointing
                  : DEFAULT_POST_TRAINING_CONFIG.method
              }
              colors={colors}
              icon={Boxes}
            />
            <PreviewCard
              title="Config mode"
              value={
                activeTab === "pretraining"
                  ? DEFAULT_TRAINING_CONFIG.model.inputMode
                  : DEFAULT_POST_TRAINING_CONFIG.baseModel.inputMode
              }
              subtitle={
                activeTab === "pretraining"
                  ? `${DEFAULT_TRAINING_CONFIG.parallelismMode} parallelism`
                  : `${DEFAULT_POST_TRAINING_CONFIG.batchSize} batch size`
              }
              colors={colors}
              icon={Gauge}
            />
          </div>
        </section>

        <section
          className="rounded-[1.5rem] border p-5"
          style={{ borderColor: colors.border, backgroundColor: colors.panel }}
        >
          <div className="flex items-center gap-2 text-sm font-medium" style={{ color: colors.accent }}>
            <Gauge className="h-4 w-4" />
            Result Shell
          </div>

          <div className="mt-5 space-y-4">
            <PlaceholderRow
              title="Memory breakdown"
              body="Stacked GPU memory bar will render here once formulas and panel wiring are implemented."
              colors={colors}
            />
            <PlaceholderRow
              title="Parallelism recommendation"
              body="Auto strategy selection hooks are typed and ready for the Phase 3 engine."
              colors={colors}
            />
            <PlaceholderRow
              title="Time and cost"
              body="Training time, storage, and failure-adjusted cost outputs will bind to the same shell."
              colors={colors}
            />
          </div>
        </section>
      </motion.div>
    </div>
  )
}

function PreviewCard({
  title,
  value,
  subtitle,
  colors,
  icon: Icon,
}: {
  title: string
  value: string
  subtitle: string
  colors: {
    cardBg: string
    text: string
    textSecondary: string
    border: string
    accent: string
    accentMuted: string
    bg: string
    panel: string
  }
  icon: typeof Binary
}) {
  return (
    <div
      className="rounded-2xl border p-4"
      style={{ borderColor: colors.border, backgroundColor: colors.cardBg }}
    >
      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.22em]" style={{ color: colors.textSecondary }}>
        <Icon className="h-4 w-4" />
        {title}
      </div>
      <div className="mt-3 text-lg font-semibold capitalize" style={{ color: colors.text }}>
        {value.replaceAll("-", " ")}
      </div>
      <p className="mt-2 text-sm" style={{ color: colors.textSecondary }}>
        {subtitle}
      </p>
    </div>
  )
}

function PlaceholderRow({
  title,
  body,
  colors,
}: {
  title: string
  body: string
  colors: {
    cardBg: string
    text: string
    textSecondary: string
    border: string
    accent: string
    accentMuted: string
    bg: string
    panel: string
  }
}) {
  return (
    <div
      className="rounded-2xl border p-4"
      style={{ borderColor: colors.border, backgroundColor: colors.cardBg }}
    >
      <h3 className="text-lg" style={{ color: colors.text }}>
        {title}
      </h3>
      <p className="mt-2 text-sm leading-6" style={{ color: colors.textSecondary }}>
        {body}
      </p>
    </div>
  )
}
