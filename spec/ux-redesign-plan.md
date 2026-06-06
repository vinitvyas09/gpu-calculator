# UX Redesign Plan — "Calm Layers"

**Status:** APPROVED, ready for implementation in a fresh session. No implementation has been started; the working tree contains only this plan bundle + the parity harness.
**Owner decisions locked:** 2026-06-05 (see §1).
**Provenance:** produced by a multi-agent pipeline — 9 parallel code-recon readers (114 issues found) → 3 competing design proposals → 3-judge panel (novice advocate / expert practitioner / implementing tech lead; unanimous winner) → synthesis → 4 extraction agents producing the line-anchored appendices. Every file:line claim was verified against the code as of commit `f993952`.

---

## How to use this plan (read me first, implementing agent)

1. Read this file fully, then skim the four appendices in `spec/ux-redesign/`:
   - **`A-control-relocation.md`** — every one of the ~122 input controls + 24 readouts: exact label, primitive, file:line, config field, visibility condition, and its NEW home (Essentials / Layer 1-8). The handful of judgment calls are marked `DECIDE:` with recommendations — follow the recommendations unless you find a hard conflict.
   - **`B-glossary-microcopy.md`** — ready-to-ship copy: ~60 glossary definitions (grounded in `spec/research/*.md`), verdict-band microcopy per state, closed-layer summary-line templates (referencing exact `CalculatorOutput` fields), IntentRow/HeroBar copy.
   - **`C-component-specs.md`** — TypeScript props interfaces for all 10 new components, mount points, behavior/aria/motion specs, the UI state model (localStorage keys, SSR-safe hydration), and the color-token consolidation map.
   - **`D-warnings-results-exports-a11y.md`** — the warning routing rule + full inventory (~200 warning sites grouped by category), results relocation table (every metric → new layer), the export-surface contract (what you must NOT touch), and the a11y checklist.
2. Verify the baseline before changing anything: `npm run build && npm run lint`, then with `npm run dev` running: `npm i -D playwright-core --no-save && node scripts/parity/parity-check.mjs` → must print `PARITY: PASS` (25/25). If it doesn't pass BEFORE your changes, stop and investigate; do not proceed on a broken baseline.
3. Implement **one phase at a time** (§5), in order. Run the full gate (§6) after each phase. Phases are independently shippable; do not start phase N+1 with phase N's gate red.
4. To see the current UI: `npm run dev` → `http://localhost:3000/tools/gpu-calculator`. Screenshots of the pre-redesign UI can be regenerated headlessly (the parity script shows the playwright-core pattern; baseline problems are described in §2.1 so you don't strictly need them).
5. Committing is allowed (owner's rule, updated 2026-06-05): commit as and when you judge it necessary — a phase boundary with the §6 gate green is the natural checkpoint. Do not push; the owner reviews and pushes.

---

## 0. Hard invariants (violations = rejected work)

1. **Never change the math.** `components/gpu-calculator/formulas/*` is untouchable. All pure helpers and the ~25 chained `useMemo` pipeline blocks inside `GpuCalculator()` (`gpu-calculator.tsx:5034-6091`) keep their numeric behavior byte-for-byte. Relabel/regroup/re-tone/relocate *display* freely; never recompute, reorder math, or change constants/presets.
   - **One approved exception** (owner-flagged, intentional): `DEFAULT_POST_TRAINING_CONFIG.numGPUs` (currently `1`, `constants.ts:1741`) may be seeded so the post-training tab opens in a fitting state instead of today's 111%-VRAM red gauge (Phase 4). This changes a default *input*, never a formula. It is the ONLY allowed `constants.ts` edit, and the only allowed parity diffs are the post-training keys it affects (the harness enforces the exact allowed set; see §6 and Phase 4).
2. **Numbers parity gate after every phase** (§6): `node scripts/parity/parity-check.mjs` → `PARITY: PASS`. Never regenerate or edit `scripts/parity/baseline-snapshots.json`. Selector fixes in the check script are allowed (controls move); numeric diffs are not.
3. **Export values are load-bearing.** The JSON export is `serializeCalculatorOutput` (`gpu-calculator.tsx:4864-4876`); the text export is `generatePretrainingMarkdown`/`generatePostTrainingMarkdown` (`:4878-4992`). These read only `CalculatorOutput` and are NOT touched by display relocation. Note: `WARNING_LABEL.critical === "Error"` (`:4780`) feeds ONLY the text/markdown export. The UI relabel "Error"→"Critical" (Phase 1) edits `SEVERITY_META` in `results-summary.tsx:203`, never `WARNING_LABEL` — and the text-export contract is now gate-pinned by the `*-text` snapshot keys (incl. `pretrain-invalid-gpus-0-text`, which contains the literal "Error"). See Appendix D.7.
   - **Warning push sites are frozen.** Every `Warning`'s `severity`, `category`, and `message` is serialized into the JSON export and is part of the contract. Routing (Phase 1, Appendix D) is READ-ONLY over `output.warnings[]` — filter/group/relabel-for-display freely, but never edit a push site, even when a warning lands in an awkward layer.
4. **Keep the Pretraining / Post-Training tab split.** Two real pipelines. The redesign's new axis is persona (novice↔expert) via disclosure, orthogonal to phase.
5. **Keep the visual identity.** Warm-cream/ink OKLch palette, Fraunces display serif, Inter, JetBrains Mono figures, existing radius scale, framer-motion, dark mode via next-themes. The aesthetic stays; the information design changes. Consolidate to the CSS-variable color system per Appendix C (kill the JS `colors` memo + inline oklch literals as you touch each component).
6. **No new runtime dependencies.** (`playwright-core` is dev-only, `--no-save`, for the parity gate.)
7. **Never push.** Committing is allowed — commit as and when necessary (a green phase gate is the natural checkpoint); the owner reviews and pushes.

## 1. Owner decisions (locked 2026-06-05)

| Decision | Resolution |
|---|---|
| Direction | **Calm Layers**: one adaptive surface, no persona toggle; sticky verdict band; ~8 always-visible Essentials; everything else in summary-line Layers. (Rationale: §7) |
| Parallelism layer default | **Auto-open on manual**: Layer 3 closed by default with a derived summary line; the moment `parallelismMode` switches to `manual`, the layer opens and stays open (persisted). |
| Intent on-ramp | **Quiet expander**: a collapsed "New here? ▸" affordance near the top; expands to 3 plain-verb cards (copy in Appendix B§4); invisible weight for returning users; dismissed-state persisted. |
| Routing | **Root-only**: calculator becomes `app/page.tsx` (THE product). **Delete** `app/tools/` (both routes) and `lib/utils/tools.ts`. Move `gpu-calculator-embed.tsx` + `theme-toggle.tsx` from `app/tools/gpu-calculator/` into `components/`. Preserve page `metadata` (title/description from the old `tools.ts` entry) on the root page. Owner accepts breaking `/tools/*` deep links; grep for any `/tools` references (`href`, imports, `Link`) and clean them all — recon found them only in `app/tools/**` and `lib/utils/tools.ts` themselves, but re-verify at implementation time. |

## 2. Verified architecture facts (line-anchored; corrected by Appendix D where the recon was stale)

- `gpu-calculator.tsx` (6,657 lines): pure compute glue + warning generators span `:327-4762` (warning sites are spread `:327-1594`, `:2489-2562`, `:3713-4762`, plus in-pipeline pushes `:5561-6381`); export builders `:4763-4992`; the React component is `:4990-6657` with the pipeline as ~25 chained `useMemo`s **inside** the component (`:5034-6091`). **Keep the memos and helpers exactly where they live; rewrite only the shell JSX (`:6446-6657`).** Do NOT attempt to extract the pipeline into a separate module.
- Existing primitives to extend, not replace (`input-controls.tsx`): `NumberInput`, `SelectInput`, `ToggleInput`, `SliderInput`, `CheckboxGroupInput`, `CollapsibleSection` (`:643`, already has `defaultOpen`/`badge`/`aria-expanded`/motion — the `<Layer>` base), `TooltipIcon` (`:158`).
- Output objects already carry the novice-grade explanation payloads, mostly unrendered: `memory.fits` (NEVER rendered as a verdict today — `results-summary.tsx:655` only), `reasoning[]` (rendered at `:502`), `chinchilla.recommendation`, `dataRepetition.recommendation`, `strategyLabel`, `preset.notes` (never rendered anywhere).
- UI-layer derivations to surface as overrides (display-only, safe): `effectiveOptimizerId` adamw-fp8→adamw-mixed (`pretraining-panel.tsx:355-361`), overlap-comm forced under ZeRO-3 (`:454-458`), ZeRO stage derived under FSDP (`:117-131`).
- Confirmed bug to fix in Phase 4: model-selector mode switches clobber `architecture`/`moe` (`model-selector.tsx:185-230`); the `detailed` else-branch restores nothing → `detailedDraft` shadow copy (spec in Appendix C).
- Triple color system to consolidate: CSS vars (`globals.css:7-94`) + JS `colors` memo (`gpu-calculator.tsx:5034-5071`) + ~25 inline oklch literals. Target: CSS variables only (map in Appendix C).
- A11y gaps (Appendix D.8): zero `aria-live` anywhere; framer-motion ignores `prefers-reduced-motion` everywhere (the CSS guard at `globals.css:177` does not stop JS animations); no keyboard shortcuts exist; tabs lack `role="tablist"` semantics.

### 2.1 What is wrong with today's UI (the problems this plan fixes, ranked)

1. No persona axis at all — ~30 jargon-grade controls interleaved with the ~10 novice controls in always-open sections ("weird nebulous middle ground" — owner).
2. THE answer (total cost `$144,296`, training time `187.8 days` for defaults) renders at the very BOTTOM of a ~4,000px scroll; `memory.fits` never renders as a verdict; cost figures wrap mid-digit in cramped 4-up card grids; labels truncate ("THROUGHPU").
3. Warnings render LAST (`results-summary.tsx:634/778`) with inconsistent severity labels ("Error" vs "critical").
4. Dual independently-scrolling panes (`lg:max-h-[82vh]` at `:6564/:6591`) break the input→output connection; giant void below the collapsed Advanced section; parallelism mesh clipped ("Showing 6 of 8 TP lanes", `parallelism-layout.tsx:366-370`).
5. Post-training tab opens in an error state (111% VRAM, red gauge) from `numGPUs:1` + LoRA defaults.
6. Help system = one hover tooltip icon; ~30 unexplained acronyms; single-letter labels ("(D)", "(U)", "(G)").
7. Silent overrides: UI shows a value the math ignores (adamw-fp8→adamw-mixed etc.).
8. One invalid field blanks ALL results with no pointer back to the culprit.
9. Model selector destroys detailed edits on mode switch; MoE config split across two distant panels.
10. 3 vanity stat cards ("27 GPU presets") occupy the hero; root URL `/` is dead create-next-app boilerplate.

## 3. Target information architecture

One continuous scroll column at `/`. The verdict band is the only sticky element.

```
<HeroBar>      Fraunces H1 "What does it take to train an LLM?" + value prop; theme toggle; "Dense view" affordance
<IntentRow>    collapsed "New here? ▸" → 3 verb cards; sets activeTab + scroll-focus; localStorage-dismissed
PHASE TABS     [ Pretraining ] [ Post-Training ]   (kept; add proper tablist roles per D.8)
<VerdictBand>  sticky · ✓ FITS | total cost | wall-clock days | N × GPU  · amber + one-tap "Fix for me" when !memory.fits
               · critical warnings render here (role="alert") · "N auto-adjustments ▸" AssumptionsLedger chip
<Essentials>   always visible, plain labels (exact membership; Appendix A's NEW HOME column agrees):
               pretraining (8): model picker (Quick/Preset/Detailed modes; searchable; notes rendered) · params (quick mode) · total tokens · GPU (searchable) · #GPUs · target training days (beside #GPUs; locks it when set) · $/GPU-hr
               post-training (8): base model · method · approach (+ "Customize adapter ▸" gate for LoRA knobs) · dataset size · epochs · GPU · #GPUs · $/GPU-hr
<LayerStack>   each <Layer>: title · summary-line-when-closed (templates in Appendix B§3) · ⚠-count chip
  ▾ 1 Memory & feasibility      OPEN   (output-only)
  ▾ 2 Performance & cost        OPEN   (output-only)
  ▸ 3 Parallelism               closed (auto-opens & persists on parallelismMode=manual)
  ▸ 4 Model architecture        closed
  ▸ 5 Precision & optimizer     closed (OverrideBadge lives here)
  ▸ 6 Data & scaling            closed ("Show your work" narrative)
  ▸ 7 Cost detail & failures    closed (coupled-cost source-of-truth sub-labels)
  ▸ 8 MoE                       closed+dimmed (auto-opens/un-dims when MoE enabled; merges ALL MoE knobs)
FOOTER         [ Expand all ] [ Compact ⇄ ]  export: [ Text ] [ JSON ]  (export handlers move verbatim, D.7)
```

**The hard disclosure rule: no input-bearing layer ever defaults open.** Only Layers 1-2 (output-only) start open. Novice first screen ≈ 1.5 viewports, answer-shaped.

### Per-tab layer matrix (which layers render on each tab)

`PostTrainingOutput` (`types.ts:477-487`) has no parallelism/Chinchilla/MoE fields, and the post-training config has no parallelism/MoE/pricing-detail inputs — mirroring how today's `PostTrainingResults` renders none of those blocks. Layers render per tab:

| Layer | Pretraining | Post-Training |
|---|---|---|
| 1 Memory & feasibility | ✓ | ✓ (MemoryLineItems promoted here) |
| 2 Performance & cost | ✓ | ✓ |
| 3 Parallelism | ✓ | **hidden** (no controls or output exist) |
| 4 Model architecture | ✓ | ✓ reduced: base-model readout + sequence length |
| 5 Precision & optimizer | ✓ | ✓ (precision, optimizer, grad precision, chunked-CE, KV-cache precision, FP8 knobs) |
| 6 Data & scaling | ✓ | ✓ reduced: dataset size/epochs/batch context (no Chinchilla — not computed for post-training) |
| 7 Cost detail & failures | ✓ | ✓ reduced: compute/storage/failure cost detail + failure multiplier (no cloud-instance/checkpoint inputs — not in PostTrainingConfig) |
| 8 MoE | ✓ (dimmed until enabled) | **hidden** |

Post-training closed-layer summary lines MUST use only `PostTrainingOutput`/`PostTrainingConfig` fields (templates in Appendix B§3, "Post-training variants"); the pretraining templates reference fields that do not exist on `PostTrainingOutput` and will render `undefined` if reused.

- Verdict-band "Fix for me": sets `numGPUs := minGPUsNeeded`. When `gpuCountDerivedFromTarget` is true (#GPUs is derived from target-days and the field is disabled), the action first clears `targetTrainingDays` (returning to explicit-GPU mode), then applies the seed — never write `numGPUs` while derive-from-target is active.
- "Dense view" is NOT a separate state atom: the button (and `d` key) is a convenience action that sets `expandAll = true` + `density = "compact"`. `e`/`c` toggle those same underlying atoms directly, so no divergence is possible.
- Exact control→layer assignment: **Appendix A** (follow its `DECIDE:` recommendations).
- Exact metric→layer assignment: **Appendix D.6**.
- Warning→surface routing: **Appendix D.2-D.5** (critical → verdict band; warning → owning layer chip + inline; info → owning layer footnote; `compute` category sub-routes by topic per D.3).
- Component contracts + state model + localStorage keys: **Appendix C**.
- All user-facing copy: **Appendix B**.

## 4. Journeys (acceptance narratives — test against these)

**Novice** ("what does it take to fine-tune a 7B model?"): lands on `/` → verdict band already populated and green → optionally expands "New here? ▸" → picks Llama-3 8B in Essentials (sees `preset.notes` under the picker) → bumps #GPUs → watches verdict + open Layers 1-2 update → never sees ZeRO/TP/rank. If over budget: amber band, teaching message (Appendix B§2), one-tap "Fix for me" (`numGPUs := minGPUsNeeded`). Concepts taught by summary lines + `<Term>` popovers, never by walls of controls.

**Expert** (hand-tunes TP/PP/EP, fp8): arrives, scans the 8 layer summary sentences to audit the whole config without opening anything → clicks "Dense view" once (persisted) → expand-all + compact density → single scroll column, verdict pinned → opens Parallelism, flips to manual (layer now stays open) → uncapped mesh + caption + diff-vs-auto chip → every silent substitution shows an OverrideBadge at the control AND a row in the AssumptionsLedger. `⌘K` jumps to any control (Phase 6). Nothing an expert can reach today becomes unreachable; "Expand all" must expose every control in Appendix A.

## 5. Phases (sequential; each independently shippable)

> Per-phase protocol: read the listed appendix sections → implement → `npm run build && npm run lint` → parity gate (§6) → visual self-check against the phase's acceptance line. Touch only the listed files.

**Phase 1 — Verdict-first results.**
Build `verdict-band.tsx` (spec C§1.1; copy B§2) rendering `memory.fits`/total cost/time/GPUs above the existing grid, both tabs. Route critical warnings into the band (D.2; routing is read-only — push sites frozen per §0.3); move the remaining WarningsPanel ABOVE the result cards; relabel UI "Error"→"Critical" (`SEVERITY_META.critical.label`, `results-summary.tsx:203` — NOT `WARNING_LABEL`, D.7). Export the private symbols the band needs from `results-summary.tsx` (`SEVERITY_META`, `Stat`, `isPretraining`) — today only the default component is exported; numeric formatters come from `../formatters` (already a shared module). Fix number wrapping (own row, `whitespace-nowrap`, `tabular-nums`). While here, give the export buttons stable accessible names (`aria-label="Copy JSON"` / `aria-label="Copy text"`) so their visible "Copied" flash can't confuse tooling. Files: new `verdict-band.tsx`, `gpu-calculator.tsx` (insert above grid), `results-summary.tsx`. *Accept: verdict visible without scrolling on a 1440×900 viewport in both tabs; cost/time never wrap mid-digit; parity PASS.*

**Phase 2 — Single-column spine.**
Replace the dual-scroll grid (`gpu-calculator.tsx:6560-6591`) with one document-flow column; verdict sticky; delete the 3 vanity stat cards (`:6484-6509`) AND the now-orphaned `stats` array (`:5073-5081`) plus any imports only it used; full-width rows for `MemoryBreakdownBar` + `ParallelismLayout`; remove the mesh lane cap + add caption (`parallelism-layout.tsx:366-370`). Files: `gpu-calculator.tsx`, `parallelism-layout.tsx`. *Accept: zero nested scrollbars; no empty void; mesh shows all lanes with caption; parity PASS.*

**Phase 3 — Layer system (the persona axis).**
Extend `CollapsibleSection` → `layer.tsx`/`layer-stack.tsx` (spec C§1.3-1.4); wrap panel sections + result blocks into the layers per Appendices A + D.6 — respecting the per-tab layer matrix (§3) — with summary lines per B§3 (post-training uses the post-training variants); default disclosure per §3; parallelism auto-open-on-manual; MoE auto-open+un-dim on enable; expand/collapse-all + density toggle + localStorage profile (state model C§2); "Dense view" affordance (a convenience action over `expandAll`+`density`, not a separate atom). Export/move the remaining shared `results-summary.tsx` internals the layer bodies need (`ResultCard`, `PostTrainingMemoryItems`, `formatParallelism`, local format helpers). Files: `input-controls.tsx`, both panels, `gpu-calculator.tsx`, `results-summary.tsx`. *Accept: default view ≤ ~1.5 viewports; "Expand all" exposes every Appendix A control (spot-check the count); closed layers show real derived summaries on BOTH tabs (no `undefined` interpolations on Post-Training); keyboard e/c/d work (guarded against typing context, D.8); parity PASS.*

**Phase 4 — Essentials + selectors + root page.**
Build `essentials.tsx` + `intent-row.tsx` + `hero-bar.tsx` (specs C§1.2/1.5/1.6; copy B§4); searchable model/GPU pickers; render `preset.notes`; **fix model-selector data-loss with `detailedDraft`** (C spec; bug at `model-selector.tsx:185-230`); **seed `DEFAULT_POST_TRAINING_CONFIG.numGPUs: 1 → 2`** (`constants.ts:1741` — THE constants.ts exception; 2 = today's `numGPUsNeeded` for the default SFT+LoRA config; other methods may still open amber, which the teaching verdict handles); root-page move + `/tools` deletion per §1 Routing (move embed + theme-toggle to `components/`; hardcode root metadata: title `LLM Training GPU Calculator`, description `Estimate GPU requirements for LLM training — compute memory breakdown, parallelism strategy, training time, and cost across pretraining and post-training phases.` — from the deleted `lib/utils/tools.ts:13-15`). Files: selectors, `app/page.tsx`, delete `app/tools/**` + `lib/utils/tools.ts`, `components/` moves, `constants.ts` (numGPUs only), `gpu-calculator.tsx`. *Accept: `/` serves the calculator; build green with `/tools` gone; no dangling `/tools` references (grep); Quick→Detailed→Preset→Detailed round-trip preserves detailed edits; post-training opens green; parity: `ALLOWED_DIFF_KEYS=posttrain-default,posttrain-dpo,posttrain-ppo,posttrain-grpo,posttrain-qlora,posttrain-full-ft,posttrain-default-text node scripts/parity/parity-check.mjs` → PASS where ONLY seed-affected post-training keys diff (the script rejects any other key) and every pretraining key stays byte-identical; document the new post-training values in the phase notes.*

**Phase 5 — Overrides + validation.**
`override-badge.tsx` inline at controls + `assumptions-ledger.tsx` chip under verdict (specs C§1.7-1.8; the 3 known override sources in §2); coupled-cost source-of-truth sub-labels (Layer 7); `error?` prop on primitives → inline field errors; stop blanking all results on one invalid field — keep last-valid output dimmed + "fix <field> ▸" banner that opens the culprit's layer and focuses it. **Hard constraint: last-valid retention is a purely presentational freeze of already-rendered display.** Do NOT recompute, re-derive, or cache `CalculatorOutput` for export purposes — `handleCopyText`/`handleCopyJSON` MUST keep reading the live `currentOutput` (`gpu-calculator.tsx:6405-6425`), so an invalid state exports exactly what it does today (pinned by the `pretrain-invalid-gpus-0` snapshot keys). Files: `input-controls.tsx`, both panels, `results-summary.tsx`, `verdict-band.tsx`. *Accept: each §2 override renders badge + ledger row; an invalid field shows an inline error WITHOUT blanking the verdict; `pretrain-invalid-gpus-0` + `-text` snapshots still byte-identical; parity PASS (badges are display-only).*

**Phase 6 — Help + ⌘K + polish.**
`glossary.ts` + `term.tsx` (content B§1, ~60 entries) woven through labels and summaries; plain-word-first relabel pass per Appendix A's "New label" column (UI labels only — exported markdown labels are hardcoded separately in the export builders and stay frozen); `settings-search.tsx` ⌘K palette (spec C§1.10; registry-staleness dev assert); a11y completion per D.8 (aria-live verdict, tablist roles, `useReducedMotion` starting with `input-controls.tsx`); mobile pass (verdict 2×2, comfortable density, no horizontal scroll). *Accept: ⌘K finds+focuses+opens a named control; every Appendix A "New label" applied; reduced-motion kills JS animations; parity PASS.*

## 6. Parity protocol (the numbers firewall — in-repo)

- **Harness:** `scripts/parity/parity-check.mjs` (self-documenting header; scenario-based — every capture starts from a fresh page load, so captures are order-independent). Baseline: `scripts/parity/baseline-snapshots.json` — **25 byte-deterministic captures** (verified reproducible across runs), all pinned from the pre-redesign commit `f993952`:
  - *JSON exports (22):* pretraining default; GPT-2 Small/Medium/Large/XL; LLaMA 2 70B; LLaMA 3 70B; LLaMA 3.1 405B; DeepSeek V3 671B (MoE); Mistral 7B on default H100; Mistral 7B on A100 80GB; FP8 precision; manual parallelism; target-training-days=30 (GPU-solver mode); 64 GPUs; invalid #GPUs=0 (blocked-state contract); post-training default + DPO + PPO + GRPO + QLoRA + full fine-tuning.
  - *Text/markdown exports (3):* pretraining default; post-training default; invalid #GPUs=0 (pins `WARNING_LABEL` "Error" + all markdown section labels/figures).
- **Run:** `npm i -D playwright-core --no-save` once; `npm run dev` in another terminal; `node scripts/parity/parity-check.mjs`. It probes `/` then `/tools/gpu-calculator`, clicks "Expand all" if present (post-Phase-3 reachability), drives presets by visible label (native select + combobox fallback), captures the JSON/Text exports via a clipboard shim, and diffs byte-for-byte, printing the first 5 JSON-path diffs per failure.
- **Self-enforcing rules** (the script asserts these; do not weaken them):
  - Never edit the baseline. Selector fixes allowed (controls move) and must keep all 25 captures — a scenario that captures but has no baseline key FAILS loudly.
  - `ALLOWED_DIFF_KEYS` only accepts the Phase-4 seed-affected post-training keys; naming any pretraining key exits with an error. Pretraining keys are byte-frozen forever.
  - `WRITE_BASELINE=1` refuses to run unless every existing key already matches (i.e., it is unusable once the redesign starts).
- **COVERAGE LIMITS (read honestly):** the gate compares sampled end-to-end outputs, not all math. Uncovered axes include: custom GPU specs, cloud-instance pricing presets, activation-checkpointing variants, MFU override, CPU offload, MoE configured manually (beyond the DeepSeek preset), MeZO, LoRA rank/alpha/target variations, ZeRO/FSDP stage variations beyond the auto recommendation, and most warning-bearing edge configs. Passing parity proves the 25 pinned configs are byte-identical — NOT that all math is unchanged. For any change near shared pipeline helpers, the PRIMARY firewall is the git-diff rule: **zero changes under `formulas/` in every phase, zero `constants.ts` changes except the Phase 4 seed, and no edits to the numeric logic of the in-component helpers/memos.** The snapshot gate is a sampled tripwire on top of that rule, not a substitute for it.
- Transient output `scripts/parity/current-snapshots.json` is gitignored.

## 7. Design rationale (why Calm Layers — context for judgment calls)

Three designs competed: **Two Front Doors** (explicit Simple/Expert mode switch, two renderings), **Calm Layers** (one surface, progressive disclosure with derived summary lines), **Runway** (goal-first wizard → workbench). A 3-judge panel (novice advocate, expert practitioner, implementing tech lead) scored them on novice-first-answer, expert efficiency, middle-ground elimination, feasibility, coherence, durability. **Calm Layers won unanimously (~8.7/10 avg)** because: one DOM / no second renderer to keep in parity; persona emerges from disclosure state instead of a self-identification toggle; closed-layer summary lines derived from the same `CalculatorOutput` fields cannot rot independently; and its phasing was the only one matching the verified codebase (the pipeline cannot be extracted "verbatim" — it is in-component memos). Grafted from the losers: the `detailedDraft` selector fix, "over-budget is a teaching moment" + "Fix for me", fitting-default seed, the quiet intent on-ramp, the first-visit "Dense view" escape hatch, ⌘K search, OverrideBadge + AssumptionsLedger (both), coupled-cost source labels. Explicitly rejected: any Simple/Expert panel fork (permanent two-UI maintenance); the full Runway wizard (second layout mode, duplicated input surface); any "extract lines 1-4998 to calculator-core.ts" plan (factually impossible as stated).

**Implication for the implementer:** if you face an unforeseen trade-off, resolve it toward (a) one surface over two renderings, (b) derived display over hand-maintained parallel state, (c) disclosure over removal — experts must reach everything "Expand all" implies, (d) answer-first ordering.

## 8. Risk register

| Risk | Mitigation |
|---|---|
| A relocation accidentally drops a control (expert regression) | Appendix A is the checklist; Phase 3 gate counts controls under "Expand all" against it |
| Summary lines drift from actual values | Summaries MUST interpolate `CalculatorOutput`/config fields (templates B§3), never hand-written constants |
| Parity breaks subtly via JSX-order side effects | The pipeline memos are order-independent of JSX; only reorder JSX, never memo declarations; parity gate catches the rest |
| `WARNING_LABEL` UI/export conflation | UI relabel edits `SEVERITY_META` (`results-summary.tsx:203`) only; `WARNING_LABEL` (`:4780`) is frozen **by rule** (it feeds only the text export, which the JSON gate can't see) and additionally pinned by the `*-text` snapshot keys |
| Warning push-site edits sneak in during routing work | `severity`/`category`/`message` are serialized contract (§0.3); routing is read-only; non-default warning configs pinned (fp8, manual parallelism, invalid-input scenarios) |
| localStorage state poisons SSR/hydration | Follow the `useSyncExternalStore` pattern (`gpu-calculator.tsx:5018`) per C§2 — never read localStorage during render |
| Layer-open thrash on tab switch | Per-phase layer-open maps are keyed by tab (C§2 state model) |
| ⌘K registry rots | Dev-only assert: registry count === rendered control count (Phase 6) |
| Framer-motion height animation jank on huge layer bodies | Use the existing `CollapsibleSection` AnimatePresence pattern; `useReducedMotion` guard (D.8) |

## 9. Out of scope (do NOT do these)

- No formula/preset/constant changes beyond the single numGPUs seed; no "improving" the math, units (decimal GB stays), or Chinchilla coefficients.
- No state-management rewrite (no context/reducer/zustand/URL-state) — local state + localStorage only.
- No new heavy UI libraries (no shadcn/radix/cmdk; build ⌘K on the existing primitives).
- No SSR of the calculator (it stays a client component behind the existing dynamic import).
- No redesign of the export formats' VALUES (cosmetic additions like a TL;DR header in the text export are allowed ONLY if parity keys are unaffected — the text export is not snapshot-keyed, but err conservative).
- No deletion of "dead" code beyond `app/tools/**` + `lib/utils/tools.ts` (e.g., leave `estimateGenerationCrossoverBatch` at `:3038` alone).

## 10. File map after completion (expected)

```
app/page.tsx                          ← THE product (hero + calculator), carries metadata
app/layout.tsx                        ← unchanged (fonts/theme)
components/gpu-calculator-embed.tsx   ← moved from app/tools/gpu-calculator/
components/theme-toggle.tsx           ← moved from app/tools/gpu-calculator/
components/gpu-calculator/gpu-calculator.tsx        ← memos/helpers intact; new shell JSX
components/gpu-calculator/glossary.ts               ← new (content from Appendix B§1)
components/gpu-calculator/components/
  verdict-band.tsx essentials.tsx layer.tsx layer-stack.tsx hero-bar.tsx intent-row.tsx
  override-badge.tsx assumptions-ledger.tsx term.tsx settings-search.tsx   ← new (specs Appendix C)
  pretraining-panel.tsx post-training-panel.tsx     ← rewritten as layer bodies
  model-selector.tsx gpu-selector.tsx               ← restyled + detailedDraft fix
  input-controls.tsx                                ← extended primitives
  results-summary.tsx                               ← dismembered into layer bodies
  memory-breakdown-bar.tsx gpu-utilization-gauge.tsx parallelism-layout.tsx  ← reused/restyled
scripts/parity/                       ← gate (already present)
spec/ux-redesign-plan.md + spec/ux-redesign/A-D     ← this bundle
DELETED: app/tools/** , lib/utils/tools.ts
```
