"use client"

import type {
  CalculatorTab,
  ParallelismRecommendation,
  PostTrainingConfig,
  TrainingConfig,
} from "../types"
import type { CalculatorColors } from "./input-controls"
import { PretrainingEssentials } from "./pretraining-panel"
import { PostTrainingEssentials } from "./post-training-panel"

// ---------------------------------------------------------------------------
// Essentials — the always-visible, plain-label control strip (plan §3).
//
// A dumb tab-switching shell: it renders the active tab's essentials and does
// no wiring of its own. The set*/derivation closures live in the panels' hooks
// (usePretrainingWiring / usePostTrainingWiring); GpuCalculator owns the config
// atoms and the small display props (active params, effective GPU count, the
// derived-from-target flag) and threads them straight through.
//
// `isDark` is part of the contract (forwarded to the searchable selectors for
// note-rendering / derivations); the current panel essentials style via the
// threaded `colors` object, so it is accepted but not consumed here yet.
// ---------------------------------------------------------------------------
export interface EssentialsProps {
  tab: CalculatorTab
  colors: CalculatorColors
  isDark: boolean

  /** Pretraining branch */
  trainingConfig: TrainingConfig
  onTrainingChange: (c: TrainingConfig) => void
  /** Live auto recommendation — feeds the pretraining wiring hook. */
  autoParallelismRecommendation: ParallelismRecommendation
  /** Active (non-embedding/MoE) param count for MFU + readouts. */
  activeParameterCount: number
  /** GPU count after target-days / topology resolution. */
  effectiveNumGPUs: number
  /** True when #GPUs is derived from target training days (locks the field). */
  gpuCountDerivedFromTarget: boolean

  /** Post-training branch */
  postTrainingConfig: PostTrainingConfig
  onPostTrainingChange: (c: PostTrainingConfig) => void
}

export function Essentials({
  tab,
  colors,
  trainingConfig,
  onTrainingChange,
  autoParallelismRecommendation,
  activeParameterCount,
  effectiveNumGPUs,
  gpuCountDerivedFromTarget,
  postTrainingConfig,
  onPostTrainingChange,
}: EssentialsProps) {
  if (tab === "pretraining") {
    return (
      <PretrainingEssentials
        config={trainingConfig}
        onChange={onTrainingChange}
        colors={colors}
        activeParameterCount={activeParameterCount}
        effectiveNumGPUs={effectiveNumGPUs}
        gpuCountDerivedFromTarget={gpuCountDerivedFromTarget}
        autoParallelismRecommendation={autoParallelismRecommendation}
      />
    )
  }

  return (
    <PostTrainingEssentials
      config={postTrainingConfig}
      onChange={onPostTrainingChange}
      colors={colors}
    />
  )
}
