"use client"

import { useEffect, useMemo, useRef } from "react"
import { motion, useReducedMotion } from "framer-motion"
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
  estimateParametersQuick,
  getQuickModeHiddenSizeAlignment,
  normalizeAttentionVariantHeads,
  resolveDefaultFFNIntermediateSize,
  resolveDefaultMoEExpertIntermediateSize,
} from "../formulas/compute"
import {
  type CalculatorColors,
  type SearchableSelectOption,
  NumberInput,
  SearchableSelect,
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

function resolveDefaultGQAKVHeads(heads: number): number {
  const safeHeads =
    Number.isFinite(heads) && heads > 0 ? Math.max(1, Math.floor(heads)) : 1
  let kvHeads = Math.min(8, safeHeads)

  while (kvHeads > 1 && safeHeads % kvHeads !== 0) {
    kvHeads--
  }

  return Math.max(1, kvHeads)
}

function resolveKVHeadsForAttentionVariant(
  variant: ModelArchitecture["attentionVariant"],
  arch: ModelArchitecture,
): number | null {
  if (variant === "mha") {
    return arch.a
  }

  if (variant === "mqa") {
    return 1
  }

  if (variant === "mla") {
    return null
  }

  if (arch.a_kv !== null && arch.a_kv > 0 && arch.a_kv < arch.a) {
    return arch.a_kv
  }

  return resolveDefaultGQAKVHeads(arch.a)
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

  const architecture = estimateParametersQuick(paramCount)
  const roundTo = getQuickModeHiddenSizeAlignment(architecture.a)

  return {
    architecture,
    quickMode: {
      totalParameters: paramCount,
      family: lookup.family,
      inferredHeads: architecture.a,
      inferredLayers: architecture.L,
      hiddenSizeRoundingMultiple: roundTo,
    },
  }
}

// ---------------------------------------------------------------------------
// Shared preset-picker options (searchable). The label string is the parity /
// automation contract — keep it EXACTLY "<name> (<compact params>)".
// ---------------------------------------------------------------------------
function useModelPresetOptions(): SearchableSelectOption[] {
  return useMemo(
    () =>
      MODEL_PRESETS.map((p) => ({
        value: p.id,
        label: `${p.name} (${formatCompact(p.parameterCount)})`,
      })),
    [],
  )
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

export function getModelPresetDefaultSequenceLength(
  presetId: string | null,
): number | null {
  return (
    MODEL_PRESETS.find((preset) => preset.id === presetId)
      ?.defaultSequenceLength ?? null
  )
}

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
  const reduceMotion = useReducedMotion()
  const presetOptions = useModelPresetOptions()

  // ── detailedDraft fix (Phase 4 — plan §2/§3, Appendix A P1) ──
  // The detailed tab is the only mode whose edits are not re-seeded on entry.
  // When the user LEAVES detailed mode we snapshot { architecture, moe } here;
  // when they RETURN to detailed mode we restore that snapshot instead of
  // whatever the quick/preset detour wrote. An explicit preset change while in
  // preset mode does NOT clear the draft (setPreset never touches this ref).
  const detailedDraftRef = useRef<Pick<
    ModelSelection,
    "architecture" | "moe"
  > | null>(null)

  const setMode = (mode: ModelInputMode) => {
    // Capture the current detailed edits before any quick/preset detour
    // overwrites architecture+moe.
    if (selection.inputMode === "detailed" && mode !== "detailed") {
      detailedDraftRef.current = {
        architecture: selection.architecture,
        moe: selection.moe,
      }
    }

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
      // Entering detailed: restore the captured draft (if any) so round-trips
      // Quick→Detailed / Preset→Detailed preserve the user's architecture.
      const draft = detailedDraftRef.current
      onChange(
        draft
          ? { ...selection, inputMode: mode, ...draft }
          : { ...selection, inputMode: mode },
      )
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
        initial={reduceMotion ? false : { opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={reduceMotion ? { duration: 0 } : { duration: 0.15, ease: "easeOut" }}
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
          <DetailedPointer colors={colors} />
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
        label="Parameters"
        value={q.totalParameters}
        onChange={onParamChange}
        min={1e6}
        max={2e12}
        integer
        compact
        tooltip="Enter parameter count with suffix: M (million), B (billion), T (trillion)."
        fieldId="quickModeTotalParameters"
        termKey="parameters"
        colors={colors}
      />
      {quickTokens !== undefined && onQuickTokensChange && (
        <NumberInput
          label="Total training tokens (D)"
          value={quickTokens}
          onChange={onQuickTokensChange}
          min={1e6}
          max={1e16}
          integer
          compact
          tooltip="Quick mode exposes dataset size up front for a fast coarse estimate."
          fieldId="totalTokens"
          termKey="tokens"
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
          <Stat
            label="d_ff"
            value={String(
              a.d_ff ?? resolveDefaultFFNIntermediateSize(a.d, a.ffnType)
            )}
            colors={colors}
          />
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
  presetOptions: SearchableSelectOption[]
  onPresetChange: (id: string) => void
  colors: CalculatorColors
}) {
  const preset = MODEL_PRESETS.find((p) => p.id === selection.presetId)

  return (
    <div className="space-y-3">
      <SearchableSelect
        label="Model"
        value={selection.presetId || MODEL_PRESETS[0].id}
        onChange={onPresetChange}
        options={presetOptions}
        fieldId="presetId"
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
            label="Context"
            value={formatCompact(preset.defaultSequenceLength)}
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
      {preset?.notes && (
        <p
          className="text-[11px] leading-relaxed"
          style={{ color: colors.textSecondary }}
        >
          {preset.notes}
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Detailed-mode pointer — the architecture grid now lives in the Model
// architecture layer (ModelArchitectureFields). This keeps ModelSelector's
// public props unchanged while Stage B mounts the fields below.
// ---------------------------------------------------------------------------
function DetailedPointer({ colors }: { colors: CalculatorColors }) {
  return (
    <p
      className="rounded-lg border px-3 py-2.5 text-[11px] leading-relaxed"
      style={{
        borderColor: colors.border,
        backgroundColor: colors.bg,
        color: colors.textSecondary,
      }}
    >
      Architecture fields live in the Model architecture layer below.
    </p>
  )
}

// ---------------------------------------------------------------------------
// useModelArchWiring — shared onChange closures for the detailed-mode
// architecture controls. Extracted verbatim from the old DetailedTab + the
// MoE/FFN coupling that previously lived inside ModelSelector so behavior is
// byte-identical wherever the fields are mounted.
// ---------------------------------------------------------------------------
export function useModelArchWiring(
  selection: ModelSelection,
  onChange: (s: ModelSelection) => void,
) {
  const lastDenseFFNTypeRef = useRef<DenseFFNType>(
    inferDefaultDenseFFNType(selection.architecture),
  )

  useEffect(() => {
    if (selection.architecture.ffnType !== "moe") {
      lastDenseFFNTypeRef.current = selection.architecture.ffnType
    }
  }, [selection.architecture.ffnType])

  const updateArch = (patch: Partial<ModelArchitecture>) =>
    onChange({
      ...selection,
      architecture: normalizeAttentionVariantHeads({
        ...selection.architecture,
        ...patch,
      }),
    })

  const updateMoe = (patch: Partial<MoEConfig>) =>
    onChange({ ...selection, moe: { ...selection.moe, ...patch } })

  const setMoeEnabled = (enabled: boolean) => {
    const denseFFNType =
      selection.architecture.ffnType === "moe"
        ? lastDenseFFNTypeRef.current
        : inferDefaultDenseFFNType(selection.architecture)
    const defaultDenseIntermediateSize =
      selection.architecture.d_ff ??
      resolveDefaultFFNIntermediateSize(selection.architecture.d, denseFFNType)
    const defaultExpertIntermediateSize =
      resolveDefaultMoEExpertIntermediateSize(selection.architecture.d)

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
            ? defaultExpertIntermediateSize
            : selection.moe.expertIntermediateSize,
        denseIntermediateSize:
          enabled &&
          selection.moe.denseIntermediateSize === null
            ? defaultDenseIntermediateSize
            : selection.moe.denseIntermediateSize,
      },
    })
  }

  const setFFNType = (ffnType: ModelArchitecture["ffnType"]) => {
    if (ffnType === "moe") {
      if (!selection.moe.enabled) {
        setMoeEnabled(true)
        return
      }

      updateArch({ ffnType })
      return
    }

    lastDenseFFNTypeRef.current = ffnType

    if (selection.moe.enabled) {
      onChange({
        ...selection,
        architecture: { ...selection.architecture, ffnType },
        moe: { ...selection.moe, enabled: false },
      })
      return
    }

    updateArch({ ffnType })
  }

  const setAttentionHeads = (a: number) => {
    let a_kv = selection.architecture.a_kv

    if (selection.architecture.attentionVariant === "mha") {
      a_kv = a
    } else if (selection.architecture.attentionVariant === "mqa") {
      a_kv = 1
    } else if (selection.architecture.attentionVariant === "mla") {
      a_kv = null
    } else if (
      a_kv !== null &&
      (!Number.isFinite(a_kv) || a_kv <= 0 || a_kv > a || a % a_kv !== 0)
    ) {
      a_kv = resolveDefaultGQAKVHeads(a)
    }

    updateArch({ a, a_kv })
  }

  return { updateArch, updateMoe, setMoeEnabled, setFFNType, setAttentionHeads }
}

// ---------------------------------------------------------------------------
// MoEOverviewNote — the sparse-routing prose that used to sit inside the
// model selector's MoE Overview block. Reworded to point at the MoE layer
// (the canonical home for expert / routing knobs after the Phase-4 merge).
// ---------------------------------------------------------------------------
export function MoEOverviewNote({ colors }: { colors: CalculatorColors }) {
  return (
    <p className="text-xs leading-6" style={{ color: colors.textSecondary }}>
      Sparse routing is enabled. Configure expert counts, active experts, MoE
      layers, and optional shared experts in the MoE layer below.
    </p>
  )
}

// ---------------------------------------------------------------------------
// ModelArchitectureFields — the detailed-mode architecture grid, relocated out
// of the model selector so it can mount inside the Model architecture layer
// (Stage B). The MoE dense/expert FFN-size fields (P18/P19) are intentionally
// NOT rendered here — the MoE layer owns those (canonical copies). The MoE
// enable toggle (P16) and tied-embeddings (P17) stay with the architecture.
// ---------------------------------------------------------------------------
export function ModelArchitectureFields({
  selection,
  onChange,
  colors,
}: {
  selection: ModelSelection
  onChange: (s: ModelSelection) => void
  colors: CalculatorColors
}) {
  const { architecture: arch, moe } = selection
  const { updateArch, setMoeEnabled, setFFNType, setAttentionHeads } =
    useModelArchWiring(selection, onChange)

  return (
    <div className="space-y-4">
      {/* Core dimensions */}
      <div className="grid gap-3 sm:grid-cols-2">
        <NumberInput
          label="Hidden size (d)"
          value={arch.d}
          onChange={(d) => updateArch({ d })}
          min={64}
          step={64}
          integer
          tooltip="Model hidden dimension (d_model)"
          fieldId="architecture-d"
          colors={colors}
        />
        <NumberInput
          label="Layers (L)"
          value={arch.L}
          onChange={(L) => updateArch({ L })}
          min={1}
          integer
          tooltip="Number of transformer layers"
          fieldId="architecture-L"
          colors={colors}
        />
        <NumberInput
          label="Attention heads (a)"
          value={arch.a}
          onChange={setAttentionHeads}
          min={1}
          integer
          tooltip="Number of query attention heads"
          fieldId="architecture-a"
          colors={colors}
        />
        <NumberInput
          label="Head size (d_head)"
          value={
            arch.d_head ??
            (Number.isFinite(arch.a) && arch.a > 0 ? arch.d / arch.a : arch.d)
          }
          onChange={(d_head) => updateArch({ d_head })}
          min={1}
          integer
          tooltip="Per-head projection width. Defaults to d / heads; set explicitly for PaLM-style models where heads x d_head differs from d_model."
          fieldId="architecture-d_head"
          colors={colors}
        />
        <NumberInput
          label="Key/value heads (a_kv)"
          value={
            arch.attentionVariant === "mha"
              ? arch.a
              : arch.attentionVariant === "mqa"
                ? 1
                : arch.attentionVariant === "mla"
                  ? arch.a
                  : (arch.a_kv ?? resolveDefaultGQAKVHeads(arch.a))
          }
          onChange={(a_kv) => updateArch({ a_kv })}
          disabled={arch.attentionVariant !== "gqa"}
          min={1}
          integer
          tooltip="Editable for GQA. MHA uses all query heads, MQA uses one KV head, and MLA uses the full-width fallback."
          fieldId="architecture-a_kv"
          colors={colors}
        />
        <NumberInput
          label="Feed-forward size (d_ff)"
          value={
            arch.d_ff ?? resolveDefaultFFNIntermediateSize(arch.d, arch.ffnType)
          }
          onChange={(d_ff) => updateArch({ d_ff })}
          min={1}
          integer
          tooltip="Feed-forward intermediate dimension. Defaults depend on the FFN type."
          fieldId="architecture-d_ff"
          colors={colors}
        />
        <NumberInput
          label="Vocabulary size (V)"
          value={arch.V}
          onChange={(V) => updateArch({ V })}
          min={1000}
          integer
          tooltip="Vocabulary size"
          fieldId="architecture-V"
          colors={colors}
        />
      </div>

      {/* Architecture choices */}
      <div className="grid gap-3 sm:grid-cols-2">
        <SelectInput
          label="Feed-forward type"
          value={arch.ffnType}
          onChange={(v) => setFFNType(v as ModelArchitecture["ffnType"])}
          options={[
            { value: "standard", label: "Standard (ReLU/GELU)" },
            { value: "swiglu", label: "SwiGLU" },
            { value: "geglu", label: "GeGLU" },
            { value: "moe", label: "Mixture of Experts" },
          ]}
          tooltip="Feed-forward network variant"
          fieldId="architecture-ffnType"
          colors={colors}
        />
        <SelectInput
          label="Normalization"
          value={arch.normType}
          onChange={(v) =>
            updateArch({
              normType: v as ModelArchitecture["normType"],
            })
          }
          options={[
            { value: "layernorm", label: "LayerNorm" },
            { value: "rmsnorm", label: "RMSNorm" },
          ]}
          fieldId="architecture-normType"
          colors={colors}
        />
        <SelectInput
          label="Positional encoding"
          value={arch.posEmbedding}
          onChange={(v) =>
            updateArch({
              posEmbedding: v as ModelArchitecture["posEmbedding"],
            })
          }
          options={[
            { value: "learned", label: "Learned" },
            { value: "rope", label: "RoPE" },
            { value: "alibi", label: "ALiBi" },
            { value: "none", label: "None" },
          ]}
          fieldId="architecture-posEmbedding"
          colors={colors}
        />
        <SelectInput
          label="Attention variant"
          value={arch.attentionVariant}
          onChange={(v) => {
            const attentionVariant = v as ModelArchitecture["attentionVariant"]
            updateArch({
              attentionVariant,
              a_kv: resolveKVHeadsForAttentionVariant(attentionVariant, arch),
            })
          }}
          options={[
            { value: "mha", label: "Multi-Head (MHA)" },
            { value: "gqa", label: "Grouped-Query (GQA)" },
            { value: "mqa", label: "Multi-Query (MQA)" },
            { value: "mla", label: "Multi-Latent (MLA)" },
          ]}
          fieldId="architecture-attentionVariant"
          colors={colors}
        />
      </div>

      <ToggleInput
        label="Mixture of Experts (MoE)"
        value={moe.enabled}
        onChange={setMoeEnabled}
        tooltip="Enable sparse expert FFN blocks. Expert counts and routing settings live in the MoE layer."
        fieldId="moe-enabled"
        termKey="moe"
        colors={colors}
      />

      <ToggleInput
        label="Tied embeddings"
        value={arch.tiedEmbeddings}
        onChange={(v) => updateArch({ tiedEmbeddings: v })}
        tooltip="Share weights between input embeddings and output projection"
        fieldId="architecture-tiedEmbeddings"
        termKey="tiedEmbeddings"
        colors={colors}
      />

      {moe.enabled && <MoEOverviewNote colors={colors} />}
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
  const presetOptions = useModelPresetOptions()
  const preset = MODEL_PRESETS.find((p) => p.id === selection.presetId)

  const setMode = (mode: BaseModelInputMode) => {
    if (mode === "preset") {
      const nextPreset =
        MODEL_PRESETS.find((p) => p.id === selection.presetId) ||
        MODEL_PRESETS[0]
      onChange({
        ...selection,
        inputMode: mode,
        presetId: nextPreset.id,
        parameterCount: nextPreset.parameterCount,
        architecture: { ...nextPreset.architecture },
        moe: nextPreset.moe
          ? { ...nextPreset.moe }
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
        <>
          <SearchableSelect
            label="Base model"
            value={selection.presetId || MODEL_PRESETS[0].id}
            onChange={setPreset}
            options={presetOptions}
            fieldId="baseModel-presetId"
            colors={colors}
          />
          {preset?.notes && (
            <p
              className="text-[11px] leading-relaxed"
              style={{ color: colors.textSecondary }}
            >
              {preset.notes}
            </p>
          )}
        </>
      ) : (
        <NumberInput
          label="Parameters"
          value={selection.parameterCount}
          onChange={(n) => onChange({ ...selection, parameterCount: n })}
          min={1e6}
          max={2e12}
          integer
          compact
          tooltip="Total parameter count of the base model"
          fieldId="baseModel-parameterCount"
          termKey="parameters"
          colors={colors}
        />
      )}
    </div>
  )
}
