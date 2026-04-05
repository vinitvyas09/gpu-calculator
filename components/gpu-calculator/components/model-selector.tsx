"use client"

import { useEffect, useMemo, useRef } from "react"
import { motion } from "framer-motion"
import { Brain, Cpu, Settings2, Zap } from "lucide-react"
import type {
  BaseModelInputMode,
  BaseModelSelection,
  ModelArchitecture,
  ModelInputMode,
  ModelSelection,
  MoEConfig,
  QuickModeConfig,
} from "../types"
import { MODEL_PRESETS, QUICK_MODE_LOOKUP } from "../constants"
import {
  type CalculatorColors,
  CollapsibleSection,
  NumberInput,
  SelectInput,
  Stat,
  ToggleInput,
  formatCompact,
} from "./input-controls"

type DenseFFNType = Exclude<ModelArchitecture["ffnType"], "moe">

function inferDefaultDenseFFNType(
  arch: Pick<ModelArchitecture, "ffnType" | "normType">,
): DenseFFNType {
  if (arch.ffnType !== "moe") {
    return arch.ffnType
  }

  return arch.normType === "rmsnorm" ? "swiglu" : "standard"
}

// ---------------------------------------------------------------------------
// Quick-mode architecture inference (spec Section 11.1)
// ---------------------------------------------------------------------------
function resolveQuickMode(paramCount: number): {
  architecture: ModelArchitecture
  quickMode: QuickModeConfig
} {
  const lookup =
    QUICK_MODE_LOOKUP.find(
      (r) => paramCount >= r.minParams && paramCount < r.maxParams,
    ) || QUICK_MODE_LOOKUP[QUICK_MODE_LOOKUP.length - 1]

  const L = lookup.layers
  const a = lookup.heads
  const roundTo = 128
  const d = Math.max(
    roundTo,
    Math.round(Math.sqrt(paramCount / (12 * L)) / roundTo) * roundTo,
  )

  const isModern = lookup.family === "modern-open-weights"
  const d_ff = isModern
    ? Math.round(((8 / 3) * d) / roundTo) * roundTo
    : 4 * d
  const a_kv = isModern ? Math.min(8, a) : a
  const V = isModern ? 128000 : 50000

  return {
    architecture: {
      d,
      L,
      a,
      a_kv,
      d_ff,
      V,
      ffnType: isModern ? "swiglu" : "standard",
      normType: isModern ? "rmsnorm" : "layernorm",
      posEmbedding: isModern ? "rope" : "learned",
      attentionVariant: isModern && a_kv < a ? "gqa" : "mha",
      tiedEmbeddings: !isModern,
    },
    quickMode: {
      totalParameters: paramCount,
      family: lookup.family,
      inferredHeads: a,
      inferredLayers: L,
      hiddenSizeRoundingMultiple: roundTo,
    },
  }
}

// ---------------------------------------------------------------------------
// Segmented-control tab button (shared by both selectors)
// ---------------------------------------------------------------------------
function TabButton({
  active,
  label,
  icon: Icon,
  onClick,
  colors,
}: {
  active: boolean
  label: string
  icon: typeof Zap
  onClick: () => void
  colors: CalculatorColors
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-[7px] text-[11px] font-semibold tracking-wide transition-all"
      style={{
        backgroundColor: active ? colors.cardBg : "transparent",
        color: active ? colors.accent : colors.textSecondary,
        boxShadow: active
          ? `0 1px 4px ${colors.accentMuted}, 0 0 0 1px ${colors.border}`
          : "none",
      }}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </button>
  )
}

// ---------------------------------------------------------------------------
// ModelSelector — Quick / Preset / Detailed
// ---------------------------------------------------------------------------
const MODE_TABS: { key: ModelInputMode; label: string; icon: typeof Zap }[] = [
  { key: "quick", label: "Quick", icon: Zap },
  { key: "preset", label: "Preset", icon: Brain },
  { key: "detailed", label: "Detailed", icon: Settings2 },
]

export function ModelSelector({
  selection,
  onChange,
  colors,
  quickTokens,
  onQuickTokensChange,
}: {
  selection: ModelSelection
  onChange: (s: ModelSelection) => void
  colors: CalculatorColors
  quickTokens?: number
  onQuickTokensChange?: (tokens: number) => void
}) {
  const presetOptions = useMemo(
    () =>
      MODEL_PRESETS.map((p) => ({
        value: p.id,
        label: `${p.name} (${formatCompact(p.parameterCount)})`,
      })),
    [],
  )
  const lastDenseFFNTypeRef = useRef<DenseFFNType>(
    inferDefaultDenseFFNType(selection.architecture),
  )

  useEffect(() => {
    if (selection.architecture.ffnType !== "moe") {
      lastDenseFFNTypeRef.current = selection.architecture.ffnType
    }
  }, [selection.architecture.ffnType])

  const setMode = (mode: ModelInputMode) => {
    if (mode === "quick") {
      const resolved = resolveQuickMode(selection.quickMode.totalParameters)
      onChange({ ...selection, inputMode: mode, ...resolved })
    } else if (mode === "preset") {
      const preset =
        MODEL_PRESETS.find((p) => p.id === selection.presetId) ||
        MODEL_PRESETS[0]
      onChange({
        ...selection,
        inputMode: mode,
        presetId: preset.id,
        architecture: { ...preset.architecture },
        moe: preset.moe
          ? { ...preset.moe }
          : { ...selection.moe, enabled: false },
      })
    } else {
      onChange({ ...selection, inputMode: mode })
    }
  }

  const setPreset = (presetId: string) => {
    const preset = MODEL_PRESETS.find((p) => p.id === presetId)
    if (!preset) return
    onChange({
      ...selection,
      presetId,
      architecture: { ...preset.architecture },
      moe: preset.moe
        ? { ...preset.moe }
        : { ...selection.moe, enabled: false },
    })
  }

  const setQuickParams = (totalParameters: number) => {
    const resolved = resolveQuickMode(totalParameters)
    onChange({ ...selection, ...resolved })
  }

  const updateArch = (patch: Partial<ModelArchitecture>) =>
    onChange({
      ...selection,
      architecture: { ...selection.architecture, ...patch },
    })

  const updateMoe = (patch: Partial<MoEConfig>) =>
    onChange({ ...selection, moe: { ...selection.moe, ...patch } })

  const setMoeEnabled = (enabled: boolean) => {
    const defaultIntermediateSize =
      selection.architecture.d_ff ?? 4 * selection.architecture.d

    if (enabled && selection.architecture.ffnType !== "moe") {
      lastDenseFFNTypeRef.current = selection.architecture.ffnType
    }

    onChange({
      ...selection,
      architecture: {
        ...selection.architecture,
        ffnType: enabled ? "moe" : lastDenseFFNTypeRef.current,
      },
      moe: {
        ...selection.moe,
        enabled,
        E:
          enabled && selection.moe.E <= 0
            ? 8
            : selection.moe.E,
        topk:
          enabled && selection.moe.topk <= 0
            ? 2
            : selection.moe.topk,
        L_moe:
          enabled && selection.moe.L_moe <= 0
            ? selection.architecture.L
            : selection.moe.L_moe,
        loadBalanceFactor:
          enabled && selection.moe.loadBalanceFactor < 1
            ? 1.1
            : selection.moe.loadBalanceFactor,
        expertIntermediateSize:
          enabled &&
          selection.moe.expertIntermediateSize === null
            ? defaultIntermediateSize
            : selection.moe.expertIntermediateSize,
        denseIntermediateSize:
          enabled &&
          selection.moe.denseIntermediateSize === null
            ? defaultIntermediateSize
            : selection.moe.denseIntermediateSize,
      },
    })
  }

  return (
    <div className="space-y-3">
      {/* Segmented control */}
      <div
        className="flex gap-0.5 rounded-lg p-[3px]"
        style={{ backgroundColor: colors.bg }}
      >
        {MODE_TABS.map(({ key, label, icon }) => (
          <TabButton
            key={key}
            active={selection.inputMode === key}
            label={label}
            icon={icon}
            onClick={() => setMode(key)}
            colors={colors}
          />
        ))}
      </div>

      {/* Tab content */}
      <motion.div
        key={selection.inputMode}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.15, ease: "easeOut" }}
      >
        {selection.inputMode === "quick" && (
          <QuickTab
            selection={selection}
            onParamChange={setQuickParams}
            quickTokens={quickTokens}
            onQuickTokensChange={onQuickTokensChange}
            colors={colors}
          />
        )}
        {selection.inputMode === "preset" && (
          <PresetTab
            selection={selection}
            presetOptions={presetOptions}
            onPresetChange={setPreset}
            colors={colors}
          />
        )}
        {selection.inputMode === "detailed" && (
          <DetailedTab
            selection={selection}
            onArchChange={updateArch}
            onMoeChange={updateMoe}
            onMoeEnabledChange={setMoeEnabled}
            onDenseFFNTypeChange={(ffnType) => {
              lastDenseFFNTypeRef.current = ffnType
            }}
            colors={colors}
          />
        )}
      </motion.div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Quick tab
// ---------------------------------------------------------------------------
function QuickTab({
  selection,
  onParamChange,
  quickTokens,
  onQuickTokensChange,
  colors,
}: {
  selection: ModelSelection
  onParamChange: (n: number) => void
  quickTokens?: number
  onQuickTokensChange?: (tokens: number) => void
  colors: CalculatorColors
}) {
  const { architecture: a, quickMode: q } = selection

  return (
    <div className="space-y-3">
      <NumberInput
        label="Total Parameters"
        value={q.totalParameters}
        onChange={onParamChange}
        min={1e6}
        max={2e12}
        integer
        compact
        tooltip="Enter parameter count with suffix: M (million), B (billion), T (trillion)."
        colors={colors}
      />
      {quickTokens !== undefined && onQuickTokensChange && (
        <NumberInput
          label="Training Tokens (D)"
          value={quickTokens}
          onChange={onQuickTokensChange}
          min={1e6}
          max={1e16}
          integer
          compact
          tooltip="Quick mode exposes dataset size up front for a fast coarse estimate."
          colors={colors}
        />
      )}

      {/* Inferred architecture summary */}
      <div
        className="rounded-lg border px-3 py-2.5"
        style={{
          borderColor: colors.border,
          backgroundColor: colors.bg,
        }}
      >
        <div className="grid grid-cols-3 gap-x-3 gap-y-1.5">
          <Stat label="d_model" value={String(a.d)} colors={colors} />
          <Stat label="Layers" value={String(a.L)} colors={colors} />
          <Stat label="Heads" value={String(a.a)} colors={colors} />
          <Stat label="d_ff" value={String(a.d_ff ?? 4 * a.d)} colors={colors} />
          <Stat label="Vocab" value={formatCompact(a.V)} colors={colors} />
          <Stat
            label="Style"
            value={
              q.family === "modern-open-weights"
                ? "GQA+SwiGLU"
                : "MHA+FFN"
            }
            colors={colors}
          />
        </div>
        <p
          className="mt-2 text-[10px] leading-relaxed opacity-60"
          style={{ color: colors.textSecondary }}
        >
          Approximate inference — use Preset or Detailed for purchase
          decisions.
        </p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Preset tab
// ---------------------------------------------------------------------------
function PresetTab({
  selection,
  presetOptions,
  onPresetChange,
  colors,
}: {
  selection: ModelSelection
  presetOptions: { value: string; label: string }[]
  onPresetChange: (id: string) => void
  colors: CalculatorColors
}) {
  const preset = MODEL_PRESETS.find((p) => p.id === selection.presetId)

  return (
    <div className="space-y-3">
      <SelectInput
        label="Model Preset"
        value={selection.presetId || MODEL_PRESETS[0].id}
        onChange={onPresetChange}
        options={presetOptions}
        colors={colors}
      />
      {preset && (
        <div
          className="grid grid-cols-3 gap-x-3 gap-y-2 rounded-lg border p-3"
          style={{ borderColor: colors.border, backgroundColor: colors.bg }}
        >
          <Stat
            label="Parameters"
            value={formatCompact(preset.parameterCount)}
            colors={colors}
          />
          <Stat
            label="d_model"
            value={String(preset.architecture.d)}
            colors={colors}
          />
          <Stat
            label="Layers"
            value={String(preset.architecture.L)}
            colors={colors}
          />
          <Stat
            label="Heads"
            value={String(preset.architecture.a)}
            colors={colors}
          />
          <Stat
            label="FFN"
            value={preset.architecture.ffnType}
            colors={colors}
          />
          <Stat
            label="Attention"
            value={preset.architecture.attentionVariant.toUpperCase()}
            colors={colors}
          />
          {preset.moe && (
            <>
              <Stat
                label="Experts"
                value={`${preset.moe.E} (top-${preset.moe.topk})`}
                colors={colors}
              />
              <Stat
                label="Active"
                value={formatCompact(
                  preset.moe.activeParameterCount ??
                    preset.parameterCount,
                )}
                colors={colors}
              />
            </>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Detailed tab
// ---------------------------------------------------------------------------
function DetailedTab({
  selection,
  onArchChange,
  onMoeChange,
  onMoeEnabledChange,
  onDenseFFNTypeChange,
  colors,
}: {
  selection: ModelSelection
  onArchChange: (p: Partial<ModelArchitecture>) => void
  onMoeChange: (p: Partial<MoEConfig>) => void
  onMoeEnabledChange: (enabled: boolean) => void
  onDenseFFNTypeChange: (ffnType: DenseFFNType) => void
  colors: CalculatorColors
}) {
  const { architecture: arch, moe } = selection

  return (
    <div className="space-y-4">
      {/* Core dimensions */}
      <div className="grid gap-3 sm:grid-cols-2">
        <NumberInput
          label="Hidden dim (d)"
          value={arch.d}
          onChange={(d) => onArchChange({ d })}
          min={64}
          step={64}
          integer
          tooltip="Model hidden dimension (d_model)"
          colors={colors}
        />
        <NumberInput
          label="Layers (L)"
          value={arch.L}
          onChange={(L) => onArchChange({ L })}
          min={1}
          integer
          tooltip="Number of transformer layers"
          colors={colors}
        />
        <NumberInput
          label="Attention heads"
          value={arch.a}
          onChange={(a) => onArchChange({ a })}
          min={1}
          integer
          tooltip="Number of query attention heads"
          colors={colors}
        />
        <NumberInput
          label="KV heads"
          value={arch.a_kv ?? arch.a}
          onChange={(a_kv) => onArchChange({ a_kv })}
          min={1}
          integer
          tooltip="KV heads — equals query heads for MHA, fewer for GQA/MQA"
          colors={colors}
        />
        <NumberInput
          label="FFN dim (d_ff)"
          value={arch.d_ff ?? 4 * arch.d}
          onChange={(d_ff) => onArchChange({ d_ff })}
          min={1}
          integer
          tooltip="Feed-forward intermediate dimension"
          colors={colors}
        />
        <NumberInput
          label="Vocab size (V)"
          value={arch.V}
          onChange={(V) => onArchChange({ V })}
          min={1000}
          integer
          tooltip="Vocabulary size"
          colors={colors}
        />
      </div>

      {/* Architecture choices */}
      <div className="grid gap-3 sm:grid-cols-2">
        <SelectInput
          label="FFN type"
          value={arch.ffnType}
          onChange={(v) => {
            const ffnType = v as ModelArchitecture["ffnType"]
            if (ffnType !== "moe") {
              onDenseFFNTypeChange(ffnType)
            }
            onArchChange({ ffnType })
            if (ffnType === "moe" && !moe.enabled) {
              onMoeEnabledChange(true)
            } else if (ffnType !== "moe" && moe.enabled) {
              onMoeEnabledChange(false)
            }
          }}
          options={[
            { value: "standard", label: "Standard (ReLU/GELU)" },
            { value: "swiglu", label: "SwiGLU" },
            { value: "geglu", label: "GeGLU" },
            { value: "moe", label: "Mixture of Experts" },
          ]}
          tooltip="Feed-forward network variant"
          colors={colors}
        />
        <SelectInput
          label="Norm type"
          value={arch.normType}
          onChange={(v) =>
            onArchChange({
              normType: v as ModelArchitecture["normType"],
            })
          }
          options={[
            { value: "layernorm", label: "LayerNorm" },
            { value: "rmsnorm", label: "RMSNorm" },
          ]}
          colors={colors}
        />
        <SelectInput
          label="Positional encoding"
          value={arch.posEmbedding}
          onChange={(v) =>
            onArchChange({
              posEmbedding: v as ModelArchitecture["posEmbedding"],
            })
          }
          options={[
            { value: "learned", label: "Learned" },
            { value: "rope", label: "RoPE" },
            { value: "alibi", label: "ALiBi" },
            { value: "none", label: "None" },
          ]}
          colors={colors}
        />
        <SelectInput
          label="Attention variant"
          value={arch.attentionVariant}
          onChange={(v) =>
            onArchChange({
              attentionVariant:
                v as ModelArchitecture["attentionVariant"],
            })
          }
          options={[
            { value: "mha", label: "Multi-Head (MHA)" },
            { value: "gqa", label: "Grouped-Query (GQA)" },
            { value: "mqa", label: "Multi-Query (MQA)" },
            { value: "mla", label: "Multi-Latent (MLA)" },
          ]}
          colors={colors}
        />
      </div>

      <ToggleInput
        label="Mixture of Experts"
        value={moe.enabled}
        onChange={onMoeEnabledChange}
        tooltip="Enable sparse expert FFN blocks. Expert counts and routing settings live in Advanced Settings."
        colors={colors}
      />

      <ToggleInput
        label="Tied embeddings"
        value={arch.tiedEmbeddings}
        onChange={(v) => onArchChange({ tiedEmbeddings: v })}
        tooltip="Share weights between input embeddings and output projection"
        colors={colors}
      />

      {moe.enabled && (
        <CollapsibleSection
          title="MoE Overview"
          defaultOpen
          colors={colors}
          badge={moe.E > 0 ? `${moe.E} experts` : "Enabled"}
        >
          <div className="space-y-3">
            <p
              className="text-xs leading-6"
              style={{ color: colors.textSecondary }}
            >
              Sparse routing is enabled. Configure expert counts, active experts,
              MoE layers, and optional shared experts in the pretraining panel’s
              Advanced Settings section.
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              <NumberInput
                label="Dense FFN size"
                value={moe.denseIntermediateSize ?? arch.d_ff ?? 4 * arch.d}
                onChange={(denseIntermediateSize) =>
                  onMoeChange({ denseIntermediateSize })
                }
                min={1}
                integer
                tooltip="Intermediate size for dense FFN layers in mixed dense+MoE architectures."
                colors={colors}
              />
              <NumberInput
                label="Expert FFN size"
                value={moe.expertIntermediateSize ?? arch.d_ff ?? 4 * arch.d}
                onChange={(expertIntermediateSize) =>
                  onMoeChange({ expertIntermediateSize })
                }
                min={1}
                integer
                tooltip="Intermediate size used by each expert FFN block."
                colors={colors}
              />
            </div>
          </div>
        </CollapsibleSection>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// BaseModelSelector — simpler variant for post-training (Preset / By Size)
// ---------------------------------------------------------------------------
export function BaseModelSelector({
  selection,
  onChange,
  colors,
}: {
  selection: BaseModelSelection
  onChange: (s: BaseModelSelection) => void
  colors: CalculatorColors
}) {
  const presetOptions = useMemo(
    () =>
      MODEL_PRESETS.map((p) => ({
        value: p.id,
        label: `${p.name} (${formatCompact(p.parameterCount)})`,
      })),
    [],
  )

  const setMode = (mode: BaseModelInputMode) => {
    if (mode === "preset") {
      const preset =
        MODEL_PRESETS.find((p) => p.id === selection.presetId) ||
        MODEL_PRESETS[0]
      onChange({
        ...selection,
        inputMode: mode,
        presetId: preset.id,
        parameterCount: preset.parameterCount,
        architecture: { ...preset.architecture },
        moe: preset.moe
          ? { ...preset.moe }
          : { ...selection.moe, enabled: false },
      })
    } else {
      onChange({ ...selection, inputMode: mode })
    }
  }

  const setPreset = (presetId: string) => {
    const preset = MODEL_PRESETS.find((p) => p.id === presetId)
    if (!preset) return
    onChange({
      ...selection,
      presetId,
      parameterCount: preset.parameterCount,
      architecture: { ...preset.architecture },
      moe: preset.moe
        ? { ...preset.moe }
        : { ...selection.moe, enabled: false },
    })
  }

  return (
    <div className="space-y-3">
      <div
        className="flex gap-0.5 rounded-lg p-[3px]"
        style={{ backgroundColor: colors.bg }}
      >
        <TabButton
          active={selection.inputMode === "preset"}
          label="Preset"
          icon={Brain}
          onClick={() => setMode("preset")}
          colors={colors}
        />
        <TabButton
          active={selection.inputMode === "parameter-count"}
          label="By Size"
          icon={Cpu}
          onClick={() => setMode("parameter-count")}
          colors={colors}
        />
      </div>

      {selection.inputMode === "preset" ? (
        <SelectInput
          label="Base Model"
          value={selection.presetId || MODEL_PRESETS[0].id}
          onChange={setPreset}
          options={presetOptions}
          colors={colors}
        />
      ) : (
        <NumberInput
          label="Parameter Count"
          value={selection.parameterCount}
          onChange={(n) => onChange({ ...selection, parameterCount: n })}
          min={1e6}
          max={2e12}
          integer
          compact
          tooltip="Total parameter count of the base model"
          colors={colors}
        />
      )}
    </div>
  )
}
