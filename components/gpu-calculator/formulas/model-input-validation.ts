import type {
  BaseModelInputMode,
  ModelInputMode,
  PostTrainingConfig,
  TrainingConfig,
} from "../types"

const PRETRAINING_MODEL_INPUT_MODES: ReadonlySet<ModelInputMode> = new Set([
  "quick",
  "preset",
  "detailed",
])

const POST_TRAINING_BASE_MODEL_INPUT_MODES: ReadonlySet<BaseModelInputMode> =
  new Set(["preset", "parameter-count"])

export function hasInvalidPretrainingModelInputMode(
  config: Pick<TrainingConfig, "model">,
): boolean {
  return !PRETRAINING_MODEL_INPUT_MODES.has(config.model.inputMode)
}

export function hasInvalidPostTrainingBaseModelInputMode(
  config: Pick<PostTrainingConfig, "baseModel">,
): boolean {
  return !POST_TRAINING_BASE_MODEL_INPUT_MODES.has(config.baseModel.inputMode)
}
