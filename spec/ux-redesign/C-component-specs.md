# Appendix C — Component + state specifications

*Generated 2026-06-05 from line-anchored code analysis. Part of spec/ux-redesign-plan.md.*

# Component & State Specifications — "Calm Layers"

> **For the implementing agent.** This is the contract for every new component in plan §3, plus the UI-state model and the color-token consolidation. All line:file citations were verified against the working tree on 2026-06-05. Conventions are derived from existing code; follow them exactly. **No math changes** — every component below is display-only and reads already-computed fields off `CalculatorOutput` / `TrainingConfig` / `PostTrainingConfig`.

## 0. Codebase conventions you MUST follow (verified)

These are non-negotiable patterns the existing code uses; every new component matches them.

1. **Colors are threaded, not imported.** A single `colors` object of type `CalculatorColors` (defined `input-controls.tsx:17-29`) is built once in `GpuCalculator` (`gpu-calculator.tsx:5034-5071`) and passed as a `colors` prop down through every **input/panel** component. Styling is applied via inline `style={{ color: colors.text, ... }}`, **not** Tailwind color classes. (See `NumberInput` `input-controls.tsx:303-352`, `CollapsibleSection` `input-controls.tsx:659-723`.)
2. **`isDark` (boolean) is threaded to the *result/viz* components**, which use it to index `[isDark ? "dark" : "light"]` literal maps and Tailwind semantic classes (`text-foreground`, `bg-surface`, `border-border`). See `results-summary.tsx:218`, `memory-breakdown-bar.tsx:173`, `gpu-utilization-gauge.tsx:86-90`, `parallelism-layout.tsx:113`. **The two systems coexist today.** New components that wrap *inputs* take `colors`; new components that wrap *results/viz* take `isDark` and Tailwind semantic classes. This split is intentional and the §3 color spec below resolves it long-term — but for Phase 1-5, **match the host file's existing convention** (a verdict band sitting above the results grid uses `isDark` + Tailwind classes like the result cards; a Layer wrapping panel inputs uses `colors` like `CollapsibleSection`).
3. **Formatters are imported from `../formatters`** (`formatters.ts`): `formatMemory`, `formatCount`, `formatFLOPs`, `formatCost`, `formatDuration`, `formatFractionPercent`, `formatPercent`, `formatMultiplier`. Never re-implement. (Note: `input-controls.tsx` has its *own* `formatCompact`/`formatPercent`/`parseCompactNumber` for input parsing — those are separate and stay; result-display formatting always uses `../formatters`.) `results-summary.tsx` aliases `const formatParams = formatCount` (`:39`) — reuse that alias name for param counts.
4. **Mount/SSR guard.** The component is gated on `mounted` from `useSyncExternalStore` (`gpu-calculator.tsx:5018-5024`); `isDark = mounted && resolvedTheme === "dark"`. Until `mounted`, a skeleton renders (`:6431-6444`). Any new localStorage-backed atom must use this same pattern (§ UI State below) so SSR markup matches first client paint.
5. **framer-motion patterns in use:** entrance `initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}` (`results-summary.tsx:128-130`); collapse `initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.2, ease: "easeInOut" }}` wrapped in `<AnimatePresence initial={false}>` (`input-controls.tsx:704-722`); chevron `animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}` (`input-controls.tsx:694-697`); toggle thumb spring `{ type: "spring", stiffness: 500, damping: 30 }` (`input-controls.tsx:569`). Reduced-motion is handled globally in CSS (`globals.css:177-185`); do not re-implement per component.
6. **`useId()` for `aria-controls`/`htmlFor`** (`input-controls.tsx:657`, `:236`). Buttons that expand content carry `aria-expanded` + `aria-controls` (`input-controls.tsx:670-671`).
7. **`Warning` shape** (`types.ts:384-396`): `{ severity: "info" | "warning" | "critical"; category: "memory"|"precision"|"parallelism"|"compute"|"data"|"hardware"|"cost"|"generation"; message: string }`. `SEVERITY_META` (`results-summary.tsx:173-216`) is the canonical severity→{label,icon,light/dark tones} map. Per plan §5/§7 the `critical.label` flips `"Error"→"Critical"` (`results-summary.tsx:203`).

---

## 1. New component prop interfaces

All new files live in `components/gpu-calculator/components/` unless noted. Types import from `../types`; formatters from `../formatters`; primitives from `./input-controls`.

### 1.1 `verdict-band.tsx` — `VerdictBand`

Sticky strip; reads fits/cost/time/GPU-count straight off the output union. Wraps `GpuUtilizationGauge` (mini). Holds the `AssumptionsLedger` chip and surfaces critical warnings. **No recompute** — every number is a field read.

```ts
export type VerdictTone = "ok" | "warning" | "critical"

export interface VerdictBandProps {
  /** The active tab's output object (pretraining or post-training union). */
  output: CalculatorOutput
  /** Drives Tailwind semantic classes + indexes into tone maps, like the result cards. */
  isDark: boolean
  /**
   * One-tap remedy when memory does not fit. Wired by the host to set
   * numGPUs := minGPUsNeeded (pretraining) or numGPUsNeeded (post-training).
   * Omitted ⇒ the "Fix for me" affordance is not rendered (e.g. fit unknown).
   */
  onFixForMe?: () => void
  /** Critical warnings hoisted from output.warnings by the host (severity === "critical"). */
  criticalWarnings: Warning[]
  /** Total count of silent substitutions, drives the "N auto-adjustments ▸" chip. */
  adjustmentCount: number
  /** Opens / scrolls to the AssumptionsLedger. */
  onShowLedger: () => void
}
```

- **Reads (pretraining, `PretrainingOutput`):** `memory.fits` (`types.ts:367`/`:450`), `cost.totalCost` (`:419`/`:460`), `trainingTime.theoreticalHours` (`:409`/`:458` → `formatDuration`), `effectiveNumGPUs` (`:451`), `minGPUsNeeded` (`:452`, the "Fix for me" target), `memory` (passed to the mini gauge).
- **Reads (post-training, `PostTrainingOutput`):** `memory.fits`, `cost.totalCost`, `trainingTime.theoreticalHours`, `numGPUsNeeded` (`:479`, the "Fix for me" target; may be `null` ⇒ `onFixForMe` undefined), `memory`.
- **Discriminate** with the existing `isPretraining` pattern (`results-summary.tsx:110-112`: `"parameterCounts" in output`). Re-export or duplicate that guard.
- **Reuse:** `formatCost`, `formatDuration`, `formatCount` from `../formatters`; `<GpuUtilizationGauge breakdown={output.memory} isDark={isDark} size="sm" />` (`gpu-utilization-gauge.tsx:86-90`, `size?: "sm" | "md"` already exists at `:29`).
- **Tone source of truth:** `tone = output.memory.fits ? "ok" : "warning"`; if any `criticalWarnings.length > 0` that are *non-memory*, escalate to `"critical"`. Per plan §1/§7 over-budget is **amber (`"warning"`), never red** — map `"ok"→accent/teal`, `"warning"→warning/amber`, `"critical"→error/red`. Reuse `SEVERITY_META[...].{light,dark}` tones (`results-summary.tsx:173-216`) for the warning/critical tints so it matches the warnings panel exactly.

### 1.2 `essentials.tsx` — `Essentials`

Phase-aware, always-visible control strip (~8 controls). A thin layout shell that **re-renders the existing primitives** with `config`/`onChange` wiring lifted out of the panels.

```ts
export interface EssentialsProps {
  tab: CalculatorTab
  colors: CalculatorColors
  /** Pretraining branch */
  trainingConfig: TrainingConfig
  onTrainingChange: (c: TrainingConfig) => void
  /** Post-training branch */
  postTrainingConfig: PostTrainingConfig
  onPostTrainingChange: (c: PostTrainingConfig) => void
  /** Forwarded to the searchable selectors (notes rendering, derivations). */
  isDark: boolean
}
```

- **Pretraining controls (plan §3):** model preset (searchable `ModelSelector`, `preset.notes` rendered — `ModelPreset.notes` at `types.ts:497`), params, total tokens, GPU (searchable `GpuSelector`), #GPUs, $/GPU-hr. These reuse `ModelSelector`/`GpuSelector` + `NumberInput`. Wiring (`set`, `setModel`) is **lifted verbatim** from `PretrainingPanel` (`pretraining-panel.tsx:170-…`); do not change onChange semantics.
- **Post-training controls:** base model · method · approach · dataset size · epochs · GPU · #GPUs · $/GPU-hr (from `PostTrainingConfig`: `method` `types.ts:259`, `approach` `:259`, `datasetSizeExamples` `:264`, `epochs` `:265`, `hardware.numGPUs` `:254`, `costPerGPUHour` `:274`).
- **Method extras stay hidden:** LoRA rank/alpha/targets/quantization (`LoRAConfig` `types.ts:231-236`), PPO critic/reward (`PPOConfig` `:238-242`), GRPO group size (`GRPOConfig` `:244-247`) live behind a `CollapsibleSection`/`Layer` titled "Customize adapter ▸" *inside* Essentials, revealed by `approach`/`method` — never on first paint (plan §3 bottom).
- **Reuse:** every `input-controls` primitive; `ModelSelector` (`model-selector.tsx`), `GpuSelector` (`gpu-selector.tsx`).

### 1.3 `layer.tsx` — `Layer` (extends `CollapsibleSection`)

`Layer` is `CollapsibleSection` (`input-controls.tsx:643-725`) **plus**: a derived `summary` node shown when closed, a `⚠`-count chip, `dimmed`, density, and **persisted open-state via a stable `id`** (instead of `CollapsibleSection`'s internal `useState(defaultOpen)` at `:656`). The implementer should either (a) add these props to `CollapsibleSection` directly per plan §3 component table ("extend CollapsibleSection → summary node, `dimmed`, persisted-open, `density`"), or (b) build `Layer` as a wrapper that copies `CollapsibleSection`'s markup. **Plan §3 explicitly says extend** — prefer (a), keeping the existing 5 props back-compatible (all new props optional).

```ts
export type Density = "comfortable" | "compact"

export interface LayerProps {
  /** Stable key for persistence + control-registry/⌘K targeting. Unique per layer. */
  id: string
  title: string
  colors: CalculatorColors
  children: ReactNode

  /** Rendered in the header when collapsed; a sentence derived from CalculatorOutput. */
  summary?: ReactNode
  /** Count of owned warnings; renders a severity-tinted "⚠ N" chip. 0 ⇒ no chip. */
  warningCount?: number
  /** Highest severity among owned warnings; tints the chip. */
  warningSeverity?: Warning["severity"]

  /** MoE-style: visually de-emphasized + non-interactive-looking until relevant. */
  dimmed?: boolean
  /** comfortable (default) | compact — tightens padding/type to match Stat density. */
  density?: Density

  /**
   * Controlled open-state. When provided, Layer is controlled and the parent owns
   * persistence (perLayerOpen map, § UI State). When omitted, falls back to
   * defaultOpen + internal useState (CollapsibleSection's current behavior).
   */
  open?: boolean
  onOpenChange?: (next: boolean) => void
  /** Used only in uncontrolled mode. Mirrors CollapsibleSection.defaultOpen. */
  defaultOpen?: boolean
  /** Optional leading icon, matching the panel Section header style. */
  icon?: typeof Cpu
}
```

- **Reuse / extend:** `CollapsibleSection` (`input-controls.tsx:643`) is the base — keep its `aria-expanded`/`aria-controls`/`useId` (`:657,:670-671`), chevron motion (`:694-702`), and `AnimatePresence` collapse (`:704-722`) **byte-identical**. Replace the bare `badge` string slot (`:682-692`) with: optional `badge`-compatible chip **and** the new `warningCount` chip. The existing `badge?: string` prop should remain for back-compat (MoE layer can use it, e.g. `badge="MoE"`).
- **Existing `CollapsibleSection` open-bg behavior** (`:664`: `backgroundColor: open ? colors.bg : "transparent"`) is preserved. `dimmed` adds `opacity` (~0.55) + a "click to enable" affordance; do not actually disable, just de-emphasize (the underlying controls remain reachable once expanded, matching plan §3 "auto-opens/un-dims when MoE enabled").
- **Density:** `compact` reduces header padding `px-4 py-3.5 → px-3 py-2.5` and body `pt-3 pb-4 → pt-2 pb-3` (mirror the `Stat` compact intent). Comfortable = current values.
- **⚠ chip tone:** index `SEVERITY_META[warningSeverity][isDark?"dark":"light"]`. (Layer takes `colors`, not `isDark`; if the chip needs the severity tones, thread `isDark` too OR derive tone from `colors.warning`/`colors.accent`. Simplest, convention-consistent: warning chip uses `colors.warning`+`colors.warningBg`+`colors.warningBorder` which already exist in `CalculatorColors` `:26-28`; critical uses an error tone — propose adding `error`/`errorBg`/`errorBorder` to `CalculatorColors`, see §3.)

### 1.4 `layer-stack.tsx` — `LayerStack`

Ordering + global expand/collapse-all + density + the localStorage "profile". It does **not** own layer content; it lays out `Layer` children and broadcasts `expandAll`/`density`.

```ts
export interface LayerStackProps {
  colors: CalculatorColors
  children: ReactNode // ordered <Layer/> elements
  /** Global "expand all" pulse; when toggled, all layers honor it (override per-layer). */
  expandAll: boolean
  density: Density
}
```

- **Behavior:** `expandAll === true` forces every child `Layer` `open`. The hard rule (plan §3): only Layers 1-2 (output-only) default open; everything input-bearing defaults closed. `LayerStack` enforces ordering 1→8 by child order; it does not reorder. Density is forwarded to each `Layer` via context or cloned props (prefer a tiny React context `LayerDensityContext` to avoid prop drilling through arbitrary children — single-use, but cleaner than cloneElement). Keep it minimal.

### 1.5 `hero-bar.tsx` — `HeroBar`

Root hero (Fraunces H1 + value prop), theme toggle, "Dense view" affordance. Collapses to a thin bar on scroll.

```ts
export interface HeroBarProps {
  /** Toggles density="compact" + expandAll globally (the expert "Dense view"). */
  denseView: boolean
  onDenseViewChange: (next: boolean) => void
  /** Rendered slot for the existing theme toggle component (moved to components/). */
  themeToggle: ReactNode
}
```

- **No `colors`/`isDark` prop:** the hero is page-level chrome; use the global CSS tokens directly via Tailwind semantic classes (`text-foreground`, `text-muted`, `text-accent`) and `style={{ fontFamily: "var(--font-display)", fontWeight: 280 }}` exactly like the current `<h2>` (`gpu-calculator.tsx:6468-6473`). H1 styling comes free from `globals.css:117-130` (`h1` uses `--font-display`, `clamp(2.5rem,5vw,3.5rem)`).
- **Copy:** H1 "How many GPUs to train an LLM?" (plan §3). Preserve page `metadata` separately on `app/page.tsx` (plan §1 routing / §3 table) — HeroBar renders visible copy only.
- **Scroll collapse:** framer-motion height/opacity on scroll-Y threshold; keep it to the existing easing `[0.16, 1, 0.3, 1]` (`gpu-calculator.tsx:6559`) or `[0.22, 1, 0.36, 1]`.
- **`themeToggle` slot:** the existing `theme-toggle.tsx` is *moved* from `app/tools/` to `components/` (plan §1/§3). HeroBar just renders it; do not reimplement.

### 1.6 `intent-row.tsx` — `IntentRow`

Quiet "New here? ▸" expander → 3 verb cards. Sets `activeTab` + scroll-focus; localStorage-dismissed.

```ts
export interface IntentRowProps {
  colors: CalculatorColors
  /** Set the phase tab when a verb card is chosen. */
  onChooseTab: (tab: CalculatorTab) => void
  /** Scroll-focuses the Essentials block after a choice. */
  onFocusEssentials: () => void
  /** Persisted dismissal (returning users never see it expanded). */
  dismissed: boolean
  onDismiss: () => void
}
```

- **3 cards (plan §1/§3/§4):** "Plan a pretraining run" → `onChooseTab("pretraining")`; "Fine-tune a model" → `onChooseTab("post-training")`; "I know my config" → just `onDismiss()` + collapse. Each card calls `onFocusEssentials()`.
- **Expander motion:** reuse the `CollapsibleSection` collapse pattern (`input-controls.tsx:704-722`). Cards use the entrance stagger from `results-summary.tsx` (`delay: index * 0.03`).
- **`CalculatorTab`** is `"pretraining" | "post-training"` (`types.ts:1`).

### 1.7 `override-badge.tsx` — `OverrideBadge`

Inline marker shown at a control when the *effective* value ≠ the *selected* value (silent substitutions). Reason text comes from existing derivation strings — **no new logic**.

```ts
export interface OverrideBadgeProps {
  colors: CalculatorColors
  /** Short label e.g. "Using AdamW (mixed)". */
  label: string
  /** Why the substitution happened — plain sentence shown on hover/focus. */
  reason: string
}
```

- **Data sources (all already computed; plan §2/§3/§5):**
  - AdamW-FP8 → AdamW-mixed: `effectiveOptimizerId` (`pretraining-panel.tsx:355-361`). Badge shows when `config.optimizer !== effectiveOptimizerId`.
  - overlap-comm forced under ZeRO-3: `zero3ForcesOverlapComm` (`pretraining-panel.tsx:454-456`) / `effectiveOverlapComm` (`:457-458`).
  - ZeRO stage derived under FSDP: `resolveFSDPZeroStage(strategy)` (`pretraining-panel.tsx:119-131`), surfaced in the layout string (`:450-452`).
- **Reuse:** built on `TooltipIcon` (`input-controls.tsx:158-204`) for the hover/focus reason popover, or a small inline pill using the same `AnimatePresence` tooltip markup. Tint with `colors.accentMuted`/`colors.accent` (info-style), matching the `CollapsibleSection` badge pill (`input-controls.tsx:682-692`).
- **Lives in Layer 5 (Precision & optimizer)** per plan §3; same badge instance also rendered inline at the relevant control.

### 1.8 `assumptions-ledger.tsx` — `AssumptionsLedger`

Central "N auto-adjustments" panel listing every override row. One row per substitution, same reason strings as `OverrideBadge`.

```ts
export interface LedgerEntry {
  /** Stable id, e.g. "optimizer-fp8-fallback". */
  id: string
  /** What changed, e.g. "Optimizer: AdamW-FP8 → AdamW (mixed)". */
  summary: string
  /** Why. */
  reason: string
  /** Layer id to scroll/open when the row is clicked (ties into perLayerOpen). */
  targetLayerId?: string
}

export interface AssumptionsLedgerProps {
  colors: CalculatorColors
  entries: LedgerEntry[]
  /** Click a row → open + scroll its owning layer. */
  onJumpToLayer?: (layerId: string) => void
}
```

- **Mount:** rendered under the `VerdictBand` (plan §3: "N auto-adjustments ▸" AssumptionsLedger chip). The chip in the band (`VerdictBandProps.adjustmentCount`/`onShowLedger`) toggles this.
- **`entries` are assembled by the host** from the same booleans the badges use (`gpu-calculator.tsx` already computes `effectiveConfig` etc.). The component is pure presentation — it must not recompute anything.
- **Reuse:** row list mirrors `WarningsPanel` list markup (`results-summary.tsx:227-261`) but with info tone.

### 1.9 `term.tsx` — `Term` (+ `glossary.ts`)

Dotted-underline inline glossary over one shared ~30-term map (plan §3 lists the terms).

```ts
// glossary.ts
export interface GlossaryEntry {
  term: string          // canonical key, e.g. "MFU"
  short: string         // 1-2 plain-English sentences
}
export const GLOSSARY: Record<string, GlossaryEntry>

// term.tsx
export interface TermProps {
  /** Glossary key; must exist in GLOSSARY (dev-assert in non-prod). */
  termKey: string
  /** Visible text (defaults to GLOSSARY[termKey].term). */
  children?: ReactNode
  colors: CalculatorColors
}
```

- **Reuse:** the `TooltipIcon` popover markup/motion (`input-controls.tsx:181-201`) for the definition card; the trigger is the underlined word (`text-decoration: underline dotted`) instead of the `Info` icon. Keyboard: focusable `<button>`/`<span tabIndex={0}>` so the tooltip opens on focus, mirroring `TooltipIcon`'s `onFocus/onBlur` (`:173-174`).
- **Single source:** every label/summary that teaches a concept imports `GLOSSARY` from `glossary.ts` (plan §6 gate: "glossary from one source"). No inline definitions elsewhere.

### 1.10 `settings-search.tsx` — `SettingsSearch` (Phase 6)

⌘K palette over a static control registry; focuses the match and auto-opens its owning Layer.

```ts
export interface ControlRegistryEntry {
  id: string            // control id (matches NumberInput useId target or a stable data-attr)
  label: string         // searchable, plain-word-first
  layerId: string       // owning Layer.id → open + scroll
  keywords?: string[]
}

export interface SettingsSearchProps {
  colors: CalculatorColors
  registry: ControlRegistryEntry[]
  /** Open + scroll the owning layer, then focus the control. */
  onOpenLayer: (layerId: string) => void
  onFocusControl: (controlId: string) => void
  /** Controlled palette visibility (host owns ⌘K keybinding). */
  open: boolean
  onOpenChange: (next: boolean) => void
}
```

- **Keyboard:** ⌘K / Ctrl+K opens (host-level listener), `Esc` closes, ↑/↓ move, `Enter` selects. The palette is a `role="dialog"` with `aria-modal`; the input is `role="combobox"` / list `role="listbox"`. Reuse the tooltip `AnimatePresence` entrance motion for the panel.
- **Dev assert (plan §3):** in non-production, assert `registry.length` equals the count of registered controls so a moved control can't silently fall out of search.
- **Static registry** lives beside the component (plan §3 "static control registry"). It maps every control to its `layerId` (matching `Layer.id`).

---

## 2. New shell tree — where each component mounts & what it reads

Replaces `gpu-calculator.tsx:6446-6657` (the JSX only; **all memos `:5034-6399` stay**). The dual-scroll panes (`:6564`, `:6591`, both `lg:max-h-[82vh] lg:overflow-y-auto`), the 3 vanity stat cards (`:6484-6509`), and `<ResultsSummary>` as a monolith (`:6652`) are removed/decomposed. New shell, single scroll column:

```
app/page.tsx                      ← page metadata (preserved) + renders GpuCalculator
└─ GpuCalculator (one column)
   ├─ <HeroBar denseView onDenseViewChange themeToggle={<ThemeToggle/>}/>          // moved theme-toggle
   ├─ <IntentRow colors onChooseTab={setActiveTab} onFocusEssentials dismissed={intentDismissed} onDismiss/>
   ├─ PHASE TABS  (kept verbatim from :6513-6552 — setActiveTab, tabs[] :4998-5011)
   ├─ <VerdictBand
   │     output={currentOutput}            // :6405  (activeTab==="pretraining"?pretrainingOutput:postTrainingOutput)
   │     isDark={isDark}                    // :5024
   │     onFixForMe                         // sets numGPUs := minGPUsNeeded / numGPUsNeeded
   │     criticalWarnings={…}               // currentOutput.warnings.filter(w=>w.severity==="critical")
   │     adjustmentCount onShowLedger/>
   │  └─ <AssumptionsLedger entries={…} onJumpToLayer/>           // built from effectiveConfig derivations
   ├─ <Essentials tab={activeTab} colors trainingConfig onTrainingChange={setTrainingConfig}
   │              postTrainingConfig onPostTrainingChange={setPostTrainingConfig} isDark/>
   ├─ <LayerStack colors expandAll={expandAll} density={density}>
   │   ├─ <Layer id="memory"      open=default>   Layer 1  (output-only)
   │   ├─ <Layer id="performance" open=default>   Layer 2  (output-only)
   │   ├─ <Layer id="parallelism">                Layer 3  (auto-open on manual)
   │   ├─ <Layer id="architecture">               Layer 4
   │   ├─ <Layer id="precision">                  Layer 5  (hosts OverrideBadge)
   │   ├─ <Layer id="data">                       Layer 6
   │   ├─ <Layer id="cost">                       Layer 7
   │   └─ <Layer id="moe" dimmed={!moeEnabled}>   Layer 8  (auto-open + un-dim when MoE on)
   └─ FOOTER  [Expand all]→setExpandAll  [Compact ⇄]→setDensity   export:[Text]→handleCopyText [JSON]→handleCopyJSON
```

### What each Layer reads (existing fields — display only)

| Layer | Body content (existing component or output field) | Closed-summary derived from |
|---|---|---|
| **1 Memory** (open) | `MemoryBreakdownBar` (`memory-breakdown-bar.tsx:173`, props `{breakdown: output.memory, isDark}`) + full `GpuUtilizationGauge` + (pretrain) `minGPUsNeeded`/`minVRAMFloor`/`maxMicroBatchSize`; (post) `PostTrainingMemoryItems` (`results-summary.tsx:290`) | `memory.fits` (`types.ts:367`) + `memory.total`/`usableCapacity` → "Fits — 62 GB / 80 GB" |
| **2 Performance & cost** (open) | `trainingTime.*`, `tokensPerSecond`, `trainingTime.totalSteps`, `globalBatchSize`, `cost.{computeCost,storageCost,failureOverheadCost,totalCost,numCheckpoints}` — the existing `Stat` grids (`results-summary.tsx:521-609`) | `formatDuration(trainingTime.theoreticalHours)` + `formatCost(cost.totalCost)` |
| **3 Parallelism** | `ParallelismLayout` (`parallelism-layout.tsx:113`, `{config: output.parallelismRecommendation.config, isDark}`) + TP/PP/DP/CP/EP/VP, ZeRO/FSDP, framework, seq-par, inter-node BW (`output.interNodeBandwidthGBps`/`Label`); `parallelismRecommendation.reasoning[]` (`types.ts:404`); diff-vs-auto chip | `formatParallelism(config)` (`results-summary.tsx:90-108`) → "DP 8 × TP 8 | ZeRO-1" + `strategyLabel` |
| **4 Architecture** | `ModelArchitecture` fields (`types.ts:11-30`): d, L, a, d_head, a_kv, d_ff, V, ffnType, normType, posEmbedding, attentionVariant, tiedEmbeddings, seqLen, flashAttention | `${d}d × ${L}L`, attention variant |
| **5 Precision & optimizer** | precision, optimizer, gradientPrecision, micro-batch, gradAccum, activationCheckpointing+depth, ampAutocast, torchCompile, chunkedCrossEntropy, FP8 knobs, `mfuOverride`; **`OverrideBadge` here** | precision + optimizer name; `OverrideBadge` if `effectiveOptimizerId !== config.optimizer` |
| **6 Data & scaling** | `uniqueTokens`, `chinchilla.ratio` + `chinchilla.recommendation` (`types.ts:337`, the prose), `predictedLossNats`, `dataRepetition.*` (`:348-354`), `batchEfficiency.*` (`:340-346`) | `formatMultiplier(chinchilla.ratio)` → "20.1× tokens-per-param" |
| **7 Cost detail & failures** | `PricingConfig` (`types.ts:175-182`): cloudPricingPresetId, cloudInstanceId, costPerGPUHour (coupled, with source-of-truth sub-labels), checkpointRetentionCount, checkpoint freq, storagePricePerGBMonth, datasetStorageGB; `FailureModelConfig` (`:169-173`); `ZeROCommunicationConfig` buckets (`:151-157`) | `formatCost(cost.totalCost)` split, `$/GPU-hr` |
| **8 MoE** (dimmed/closed) | merges ALL MoE knobs incl. EP: `MoEConfig` (`types.ts:32-42`) + `moeSparsity.*` (`:432-436`); EP from the parallelism + data panels | `moeSparsity.efficiencyGain` → "MoE off" / "8 experts, top-2" |

- **`currentOutput`** is the existing `activeTab === "pretraining" ? pretrainingOutput : postTrainingOutput` (`gpu-calculator.tsx:6405`). The verdict band and all layers read off it; **none of the ~25 memos change.**
- **Footer export buttons** reuse `handleCopyText` (`:6407-6416`) and `handleCopyJSON` (`:6418-6425`) and the `copied` state (`:5032`) **unchanged** — the JSON/Text exports are load-bearing (plan §0.3).
- **Warnings routing (plan §5):** host partitions `currentOutput.warnings` (`types.ts:469`/`:486`) by severity: `critical` → `VerdictBand.criticalWarnings`; `warning`/`info` → the owning Layer (by `Warning.category` → layer id, e.g. `"memory"→"memory"`, `"parallelism"→"parallelism"`, `"cost"→"cost"`, `"data"→"data"`, `"compute"→"performance"`, `"hardware"→"performance"`, `"precision"→"precision"`, `"generation"→"performance"`) feeding `Layer.warningCount`/`warningSeverity` + a footnote in the body. **Predicate semantics unchanged** (plan §7) — only placement/severity-label.

---

## 3. UI state model (every new atom)

`activeTab` already exists (`gpu-calculator.tsx:5026`, `useState<CalculatorTab>("pretraining")`). All atoms below are added in `GpuCalculator` (the owner), threaded down as props (matching how `setTrainingConfig` is threaded today). **Persisted atoms must use the SSR-safe `useSyncExternalStore` pattern already in the file (`:5018-5022`)** — never read `localStorage` during render or in `useState`'s initializer (that desyncs SSR markup and trips the `mounted` skeleton at `:6431`).

### 3.1 Atom table

| Atom | Type | Owner | Persisted? | localStorage key | Default | Notes |
|---|---|---|---|---|---|---|
| `activeTab` | `CalculatorTab` | `GpuCalculator` | no (exists) | — | `"pretraining"` | unchanged (`:5026`); `IntentRow`/tabs set it |
| `expandAll` | `boolean` | `GpuCalculator` | **yes** | `gpucalc:expandAll` | `false` | footer "Expand all"; forces all `Layer`s open via `LayerStack` |
| `density` | `Density` (`"comfortable" \| "compact"`) | `GpuCalculator` | **yes** | `gpucalc:density` | `"comfortable"` | footer "Compact ⇄" + HeroBar "Dense view"; threaded to `LayerStack`/`Layer` |
| `intentDismissed` | `boolean` | `GpuCalculator` | **yes** | `gpucalc:intentDismissed` | `false` | `IntentRow`; once true the expander stays collapsed for returning users |
| `perLayerOpen` | `Record<string, boolean>` | `GpuCalculator` | **yes** | `gpucalc:layersOpen` | `{ memory: true, performance: true }` (Layers 1-2 only) | controlled-open map keyed by `Layer.id`; the disclosure profile |
| `parallelismManualOpened` | `boolean` | `GpuCalculator` | **yes** | `gpucalc:parallelismManualOpened` | `false` | set `true` (and `perLayerOpen.parallelism = true`) the moment `trainingConfig.parallelismMode` flips to `"manual"`; once opened it **stays** open/persisted (plan §1/§3 "auto-opens & persists when parallelismMode=manual") |
| `denseView` (derived/coupled) | `boolean` | `GpuCalculator` | — | — | — | HeroBar's "Dense view" is sugar that sets `density="compact"` + `expandAll=true` together (plan §3/§4); can be a computed `density==="compact" && expandAll` or its own persisted atom — implementer's call, but keep it one switch |

Additional non-persisted host-derived values (not atoms, just memos/props): `criticalWarnings`, `adjustmentCount`, `ledgerEntries`, `moeEnabled` (already derivable as `pretraining-panel.tsx:414-416`: `config.model.moe.enabled || config.model.architecture.ffnType === "moe"`).

### 3.2 SSR-safe persisted-atom hook (follow `useSyncExternalStore` at `:5018`)

The file already uses `useSyncExternalStore` for the mount flag. Reuse the same primitive for persisted atoms so server snapshot = `default`, client snapshot = stored value, and React reconciles on hydration without a flash. Pattern the implementer writes (one small helper, reused for each key):

```ts
// usePersistedState.ts — SSR-safe, mirrors the existing useSyncExternalStore usage.
function usePersistedState<T>(key: string, initial: T): [T, (next: T) => void] {
  const subscribe = useCallback((cb: () => void) => {
    const handler = (e: StorageEvent) => { if (e.key === key) cb() }
    window.addEventListener("storage", handler)        // cross-tab sync
    return () => window.removeEventListener("storage", handler)
  }, [key])

  const getSnapshot = useCallback((): string | null => {
    try { return window.localStorage.getItem(key) } catch { return null }
  }, [key])

  // CRITICAL: server snapshot is constant ⇒ markup matches first client paint.
  const getServerSnapshot = useCallback(() => null, [])

  const raw = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
  const value = useMemo<T>(() => (raw === null ? initial : safeParse(raw, initial)), [raw, initial])

  const set = useCallback((next: T) => {
    try { window.localStorage.setItem(key, JSON.stringify(next)) } catch {}
    window.dispatchEvent(new StorageEvent("storage", { key }))   // same-tab notify
  }, [key])

  return [value, set]
}
```

- **Why this and not `useEffect`+`useState`:** the file already trusts `useSyncExternalStore` for SSR correctness (`:5018-5022`); using the same primitive keeps one mental model and avoids a hydration mismatch on the `Layer` open-state (a mismatch there would visibly pop layers open/closed on load). The `getServerSnapshot` returning `null` guarantees server render uses `initial`.
- **Gate reads on `mounted` where layout depends on it:** the `Layer` open-state visually differs server vs. client only after the first commit; because the whole calculator is already behind the `mounted` skeleton (`:6431-6444`), persisted layer states are only ever painted client-side — no flash. Keep that skeleton.
- **`parallelismManualOpened` wiring:** in the `onTrainingChange` path (the lifted `set`/`setTrainingConfig`), when `next.parallelismMode === "manual" && prev.parallelismMode !== "manual"`, set `parallelismManualOpened = true` and `perLayerOpen = { ...perLayerOpen, parallelism: true }`. Do this in the host handler, not inside `Layer`, so it's one source of truth and survives reloads.

---

## 4. Color / token consolidation spec (kill the triple system)

Plan §2/§3 mandate consolidating on the **CSS variables** (`globals.css:7-94`). The JS `colors` memo (`gpu-calculator.tsx:5034-5071`) and ~25 inline oklch literals are the duplicates. Below maps **every key** of the `colors` memo to the CSS custom property that should replace it. The literals in the memo are **byte-identical** to the CSS-var definitions (verified: e.g. memo `bg` light `oklch(0.993 0.003 80)` == `--background` `globals.css:9`; memo `accent` light `oklch(0.52 0.135 180)` == `--accent` `:21`; dark `accent` `oklch(0.72 0.12 180)` == `--accent` `.dark` `:60`), so this is a safe, no-visual-change swap.

| `colors` key (`:5034-5071`) | Light literal | Dark literal | Existing CSS var (`globals.css`) | Match? | Action |
|---|---|---|---|---|---|
| `bg` | `0.993 0.003 80` | `0.14 0.005 260` | `--background` (`:9` / `:50`) | ✅ exact | replace with `var(--background)` |
| `cardBg` | `0.99 0.002 80` | `0.185 0.008 260` | `--surface-elevated` (`:12`) / `--surface` (`:52`) | ⚠ light=`surface-elevated`, dark=`surface` | use `var(--surface-elevated)`; note dark `surface-elevated` is `0.22…` not `0.185…`. **Propose:** standardize cardBg → `--surface-elevated` and accept the dark card going from `0.185`→`0.22` (a 1-step lift, within tolerance) OR add a dedicated `--card` var = current values. **Recommend** adding `--card`/`--card-elevated` to be byte-exact and pass parity-screenshot review. |
| `text` | `0.155 0.004 260` | `0.93 0.004 80` | `--foreground` (`:10` / `:51`) | ✅ exact | replace with `var(--foreground)` |
| `textSecondary` | `0.50 0.010 260` | `0.60 0.010 260` | `--muted` (`:17` / `:57`) | ✅ exact | replace with `var(--muted)` |
| `border` | `0.915 0.006 80` | `0.28 0.010 260` | `--border` (`:13` / `:54`) | ✅ exact | replace with `var(--border)` |
| `accent` | `0.52 0.135 180` | `0.72 0.12 180` | `--accent` (`:21` / `:60`) | ✅ exact | replace with `var(--accent)` |
| `accentMuted` | `0.96 0.022 180` | `0.22 0.035 180` | `--accent-soft` (`:22` / `:61`) | ✅ exact | replace with `var(--accent-soft)` |
| `panel` | `0.985 0.003 80 / 0.92` | `0.16 0.006 260 / 0.85` | — | ❌ none (has alpha) | **Propose** add `--panel: oklch(0.985 0.003 80 / 0.92)` (root) / `oklch(0.16 0.006 260 / 0.85)` (.dark). Or express as `color-mix(in oklch, var(--surface) 92%, transparent)` — but the base hues differ slightly (`0.985` vs `--surface` `0.975`), so a literal `--panel` var is the faithful swap. |
| `warning` | `0.56 0.14 80` | `0.80 0.12 80` | `--warning` (`:26` / `:64`) | ✅ exact | replace with `var(--warning)` |
| `warningBg` | `0.97 0.025 80` | `0.22 0.04 80` | `--warning-soft` (`:27` / `:65`) | ✅ exact | replace with `var(--warning-soft)` |
| `warningBorder` | `0.90 0.06 80` | `0.35 0.06 80` | — | ❌ none | **Propose** add `--warning-border: oklch(0.90 0.06 80)` / `oklch(0.35 0.06 80)`. (Currently only `--warning` + `--warning-soft` exist.) |

### 4.1 Missing vars to add to `globals.css` (both `:root` and `.dark`)

To fully retire the JS memo, add these (values copied byte-for-byte from `:5034-5071`):

```css
:root {
  --panel:          oklch(0.985 0.003 80 / 0.92);
  --warning-border: oklch(0.90 0.06 80);
  --card:           oklch(0.99 0.002 80);   /* only if not folding cardBg into --surface-elevated */
  /* error tones already exist (--error :28, --error-soft :29); ADD --error-border for the Layer ⚠ critical chip: */
  --error-border:   oklch(0.9 0.09 25);     /* from SEVERITY_META.critical.light.border, results-summary.tsx:207 */
}
.dark {
  --panel:          oklch(0.16 0.006 260 / 0.85);
  --warning-border: oklch(0.35 0.06 80);
  --card:           oklch(0.185 0.008 260);
  --error-border:   oklch(0.39 0.1 25);     /* SEVERITY_META.critical.dark.border, results-summary.tsx:212 */
}
```

Then add the Tailwind bridge entries in the `@theme inline` block (`globals.css:73-94`), mirroring the existing pattern (`:85-88`): `--color-panel: var(--panel); --color-warning-border: var(--warning-border); --color-error-border: var(--error-border);` (+ `--color-card` if used).

### 4.2 Extend `CalculatorColors` for the error tone

`CalculatorColors` (`input-controls.tsx:17-29`) has `warning`/`warningBg`/`warningBorder` but **no error triplet** — yet the verdict band's critical tone and `Layer`'s critical ⚠ chip need one. Two options:

- **(A, minimal, recommended for Phase 1-3):** the `colors` memo keeps its current 11 keys; the verdict band + critical chips pull error tones from `SEVERITY_META.critical[isDark?"dark":"light"]` (`results-summary.tsx:202-215`) which already exist and are the canonical critical colors. No interface change.
- **(B, for full consolidation in a later phase):** add `error`, `errorBg`, `errorBorder` to `CalculatorColors` and the memo, sourced from `--error`/`--error-soft`/`--error-border`. Do this only alongside the §4.3 migration so you don't grow the JS system you're trying to kill.

### 4.3 Migration sequencing (so the swap is a no-visual-change diff)

1. Add the missing vars (§4.1) — pure additions, zero render change.
2. Replace the **JS memo body** (`gpu-calculator.tsx:5034-5071`) so each key returns the CSS var via `getComputedStyle`-free reference, i.e. set the values to the literal CSS `var(...)` strings: e.g. `bg: "var(--background)"`. Inline `style={{ color: colors.text }}` then resolves through CSS — **but** this only works because the consumers apply `colors.*` as inline `style` values, and `var(--x)` is valid in inline styles. This collapses the triple system to the CSS layer while keeping the *threading convention* (no component signatures change). The `isDark` branching in the memo (`:5036-5069`) is then **deleted** — the `.dark` class on the ancestor drives the var values. This is the cleanest kill of the JS color system without touching ~40 `colors=` call sites.
   - **Caveat to verify per parity screenshot:** a handful of inline literals exist *outside* the memo (e.g. tab-strip bg `gpu-calculator.tsx:6520`, export-bar bg `:6602-6604`, model-selector tones). Plan §2 counts "~25 inline oklch literals" — grep `oklch(` under `components/gpu-calculator/**` and replace each with the matching `var(--*)`; where no var matches, add one (don't invent new colors).
3. Once the memo returns only `var(...)` strings, components can optionally migrate from `style={{color: colors.text}}` to Tailwind `text-foreground` — **but that is cosmetic churn**; per the user's "surgical changes" rule, leave the threading as-is unless a phase explicitly calls for it. The functional consolidation (one set of values, in CSS) is achieved at step 2.

---

### Files this artifact touches (for the implementing agent's map)

- **New:** `verdict-band.tsx`, `essentials.tsx`, `layer.tsx`, `layer-stack.tsx`, `hero-bar.tsx`, `intent-row.tsx`, `override-badge.tsx`, `assumptions-ledger.tsx`, `term.tsx`, `glossary.ts`, `settings-search.tsx`, `usePersistedState.ts` (all in `components/gpu-calculator/components/` except the hook, which may sit at `components/gpu-calculator/`).
- **Edit (existing):** `input-controls.tsx` (extend `CollapsibleSection`→`Layer` props), `gpu-calculator.tsx` (shell JSX `:6446-6657` + new state atoms; memos `:5034-6399` untouched except deleting the `isDark` branching inside the `colors` memo per §4.3), `results-summary.tsx` (decompose into layer bodies; `SEVERITY_META.critical.label` `:203` "Error"→"Critical"), `parallelism-layout.tsx` (uncap `MAX_VISIBLE` `:7`, remove "Showing N of M" `:366`), `globals.css` (add `--panel`/`--warning-border`/`--error-border`/`--card` + bridge), `constants.ts` (Phase 4: `DEFAULT_POST_TRAINING_CONFIG.hardware.numGPUs` at **`:1741`** seed `1`→2 — the ONE approved math-input exception (plan Phase 4)).

**Verified anchor citations:** `CalculatorColors` `input-controls.tsx:17-29`; `colors` memo `gpu-calculator.tsx:5034-5071`; `useSyncExternalStore` mount `:5018-5022`; `currentOutput` `:6405`; `handleCopyText/JSON` `:6407-6425`; shell JSX to replace `:6446-6657`; dual-scroll panes `:6564`/`:6591`; vanity stats `:6484-6509`; `CollapsibleSection` `input-controls.tsx:643-725`; `TooltipIcon` `:158-204`; `Stat` `:131-153`; `SEVERITY_META` `results-summary.tsx:173-216` (`critical.label` `:203`); `WarningsPanel` `:218-265`; `isPretraining` `:110-112`; `formatParallelism` `:90-108`; formatters `formatters.ts:14-131`; `effectiveOptimizerId` `pretraining-panel.tsx:355-361`; `zero3ForcesOverlapComm` `:454-456`; `resolveFSDPZeroStage` `:117-131`; `moeEnabled` derive `:414-416`; `GpuUtilizationGauge` props `gpu-utilization-gauge.tsx:26-30,86-90`; `MemoryBreakdownBar` props `memory-breakdown-bar.tsx:82-84,173`; `ParallelismLayout` cap `parallelism-layout.tsx:7,113,366`; output unions `types.ts:443-487`; `Warning` `:384-396`; `ModelPreset.notes` `:497`; post-training default numGPUs `constants.ts:1741` (config decl `:1707`); CSS tokens `globals.css:7-94`.
