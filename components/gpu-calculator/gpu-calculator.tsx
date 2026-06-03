"use client"

import { useCallback, useMemo, useState, useSyncExternalStore } from "react"
import { motion } from "framer-motion"
import { useTheme } from "next-themes"
import {
  Boxes,
  Check,
  ClipboardCopy,
  FileText,
  Gauge,
  Layers3,
  Server,
} from "lucide-react"
import {
  DEFAULT_POST_TRAINING_CONFIG,
  DEFAULT_TRAINING_CONFIG,
  GPU_SPECS,
  MODEL_PRESETS,
  OPTIMIZER_PROFILES,
} from "./constants"
import type {
  CalculatorOutput,
  CalculatorTab,
  CostEstimate,
  MoESparsityMetrics,
  ModelArchitecture,
  MoEConfig,
  ParameterCounts,
  ParallelismConfig,
  ParallelismRecommendation,
  PostTrainingConfig,
  PostTrainingGPURequirementMode,
  PostTrainingModelMemoryLineItem,
  PostTrainingMemoryBreakdown,
  PostTrainingOutput,
  PretrainingOutput,
  TrainingConfig,
  TrainingTimeEstimate,
  Warning,
} from "./types"
import { PretrainingPanel } from "./components/pretraining-panel"
import { PostTrainingPanel } from "./components/post-training-panel"
import ResultsSummary from "./components/results-summary"
import {
  calculateParameterCount,
  calculateFLOPs,
  calculateChinchillaAnalysis,
  calculateCriticalBatchSize,
  analyzeDataRepetition,
  estimateParametersQuick,
  getInvalidArchitectureEnumMessages,
  hasInvalidMoEConfig,
  normalizeAttentionVariantHeads,
} from "./formulas/compute"
import {
  calculateTotalMemoryPerGPU,
  calculateMinGPUVRAMFloor,
  resolvePostTrainingOptimizerProfile,
  calculatePostTrainingActivationMemory,
  calculatePostTrainingForwardWorkingMemory,
  calculatePostTrainingOutputLogitsMemory,
  calculateLoRAMemory,
  calculateLoRAParamCount,
  calculateQLoRAMemory,
  calculateDPOMemory,
  calculatePPOMemory,
  calculateGRPOMemory,
  calculateDenseStateShardDegree,
  hasInvalidLoRATargetModules,
  hasInvalidZeROCommunicationConfig,
} from "./formulas/memory"
import {
  calculateTrainingTime,
  calculateCost,
  calculateGPUHourlyCost,
  getDefaultMFU,
  calculateGenerationTime,
  calculatePostTrainingCompute,
  estimatePostTrainingMoELoadBalanceFLOPsPerToken,
  estimateLoRAAdapterParameterCount,
  getEffectiveTrainingTFLOPS,
  getPostTrainingGenerationWeightBytes,
  resolveTrainingMFU,
  calculateCPUOffloadEfficiency,
  shouldSurfaceFailureAdjustedTime,
  MAX_MFU_OVERRIDE,
} from "./formulas/cost"
import {
  getParallelismLocalGroupSize,
  getInvalidCustomGPUMetadataMessages,
  hasInvalidGPUInputMode,
  hasInvalidTrainingHardware,
  getSparseThroughputWarningMessages,
} from "./formulas/hardware"
import {
  calculateVocabPadding,
  recommendParallelism,
  validateTPDivisibility,
  validatePPDivisibility,
  validateWorldSize,
  validateZeroPPCompatibility,
  validateTensorExpertSequenceParallelism,
  validateContextParallelDivisibility,
  validateMicrobatches,
  calculatePipelineBubble,
  validateHiddenDimAlignment,
  usesEmbeddingAwarePipelinePartition,
  getParallelWorldSize,
  type PipelineSchedule,
} from "./formulas/parallelism"
import {
  hasInvalidCPUOffloadConfig,
  hasInvalidManualContextParallelismTopology,
  hasInvalidManualExpertParallelismTopology,
  hasInvalidManualShardingMode,
  hasInvalidManualWorldSize,
  hasInvalidManualPipelineTopology,
  hasInvalidManualTensorExpertSequenceParallelismTopology,
  hasInvalidManualTensorParallelismTopology,
  hasInvalidParallelismFramework,
  hasInvalidParallelismMode,
  hasInvalidSequenceParallelismMode,
  resolveEffectiveZeroStage,
} from "./formulas/parallelism-validation"
import {
  hasInvalidGradientPrecision,
  hasInvalidPostTrainingOptimizer,
  hasInvalidPretrainingOptimizer,
} from "./formulas/optimizer-validation"
import {
  hasInvalidLoRAAlpha,
  hasInvalidLoRARank,
  hasInvalidPostTrainingApproach,
  hasInvalidPostTrainingApproachConfig,
  hasInvalidPostTrainingActiveParameterCount,
  hasInvalidPostTrainingModelShape,
  hasInvalidPostTrainingMethod,
  hasInvalidPostTrainingMethodApproach,
  hasInvalidPostTrainingOptimizerApproach,
  hasInvalidQLoRAQuantizationBits,
} from "./formulas/post-training-validation"
import {
  hasInvalidFP8Config,
  hasInvalidFP8KernelSpeedupFactor,
  hasInvalidFP8StorageMode,
} from "./formulas/fp8-validation"
import {
  hasInvalidPostTrainingKVCachePrecision,
  isValidKVCachePrecision,
} from "./formulas/kv-cache-validation"
import {
  hasInvalidPostTrainingBaseModelInputMode,
  hasInvalidPretrainingModelInputMode,
} from "./formulas/model-input-validation"
import {
  hasInvalidAMPAutocastFlag,
  hasInvalidChunkedCrossEntropyFlag,
  hasInvalidFlashAttentionFlag,
  hasInvalidTorchCompileFlag,
} from "./formulas/training-feature-validation"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const QLORA_THROUGHPUT_PENALTY = 1.75
const POST_TRAINING_ROLLOUT_BYTES_PER_TOKEN = 16
const TYPICAL_NODE_CPU_MEMORY_WARNING_BYTES = 0.8e12
const VALID_ZERO_COMMUNICATION_BUCKET_MODES: ReadonlySet<string> = new Set([
  "hf-auto",
  "deepspeed-defaults",
  "custom",
])

function resolveExplicitNumGPUs(numGPUs: number | null | undefined): number {
  return typeof numGPUs === "number" && Number.isFinite(numGPUs) && numGPUs > 0
    ? Math.max(1, Math.floor(numGPUs))
    : 1
}

function hasInvalidExplicitTrainingGPUCount(config: TrainingConfig): boolean {
  return (
    config.hardware.numGPUs !== null &&
    getFinitePositiveIntegerOrNull(config.hardware.numGPUs) === null
  )
}

function hasInvalidTrainingGPUCount(config: TrainingConfig): boolean {
  return (
    hasInvalidExplicitTrainingGPUCount(config) ||
    hasInvalidManualWorldSize(config)
  )
}

function hasInvalidTargetTrainingDays(config: TrainingConfig): boolean {
  const targetDays = config.hardware.targetTrainingDays

  return targetDays !== null && getFinitePositiveOrNull(targetDays) === null
}

function hasInvalidPostTrainingGPUCount(config: PostTrainingConfig): boolean {
  return (
    !config.hardware.gpu.singleDeviceOnly &&
    getFinitePositiveIntegerOrNull(config.hardware.numGPUs) === null
  )
}

function hasInvalidPostTrainingMethodConfig(config: PostTrainingConfig): boolean {
  if (config.method === "grpo") {
    return (
      getFinitePositiveIntegerOrNull(config.grpo.groupSize) === null ||
      config.grpo.groupSize < 2
    )
  }

  if (config.method === "ppo") {
    return (
      getFinitePositiveIntegerOrNull(
        config.ppo.criticModelParameterCount,
      ) === null ||
      getFinitePositiveIntegerOrNull(
        config.ppo.rewardModelParameterCount,
      ) === null ||
      getFinitePositiveIntegerOrNull(config.ppo.updateEpochs) === null
    )
  }

  return false
}

function hasInvalidPostTrainingTrainablePercentage(
  config: PostTrainingConfig,
): boolean {
  const percentage = config.trainableParameterPercentage

  return (
    (config.approach === "full" || config.approach === "mezo") &&
    percentage !== null &&
    (getFinitePositiveOrNull(percentage) === null || percentage > 100)
  )
}

function resolveFSDPZeroStage(
  fsdpStrategy: ParallelismConfig["fsdpStrategy"],
): ParallelismConfig["zeroStage"] {
  switch (fsdpStrategy ?? "FULL_SHARD") {
    case "NO_SHARD":
      return 0
    case "SHARD_GRAD_OP":
    case "HYBRID_SHARD_ZERO2":
      return 2
    case "FULL_SHARD":
    case "HYBRID_SHARD":
    default:
      return 3
  }
}

function addPrecisionSupportWarnings(
  warnings: Warning[],
  precision: TrainingConfig["precision"],
  gpu: TrainingConfig["hardware"]["gpu"],
): void {
  if (precision === "bf16" && !gpu.supportsBF16) {
    warnings.push({
      severity: "critical",
      category: "precision",
      message: `${gpu.name} does not support BF16. Select FP16/FP32 or hardware with BF16 support; estimates are disabled for this precision/hardware combination.`,
    })
  }

  if (precision === "fp16") {
    warnings.push({
      severity: "info",
      category: "precision",
      message: gpu.supportsBF16
        ? "FP16 training requires dynamic loss scaling. BF16 is usually preferred on this GPU because it keeps FP32-like exponent range."
        : "FP16 training requires dynamic loss scaling to avoid gradient underflow.",
    })
  }

  if (precision === "fp32") {
    const hasTF32Throughput =
      gpu.supportsTF32 &&
      gpu.tf32TFLOPS !== null &&
      Number.isFinite(gpu.tf32TFLOPS) &&
      gpu.tf32TFLOPS > 0
    const hasFP32Throughput =
      gpu.fp32TFLOPS !== null &&
      gpu.fp32TFLOPS !== undefined &&
      Number.isFinite(gpu.fp32TFLOPS) &&
      gpu.fp32TFLOPS > 0

    warnings.push({
      severity: "info",
      category: "precision",
      message: hasTF32Throughput
        ? "FP32 mode uses TF32 matrix/tensor-core throughput where available, but tensors still occupy FP32 memory. Model states and activations are estimated at 4 bytes per element."
        : "FP32 mode stores tensors in full precision, so model states and activations are estimated at 4 bytes per element.",
    })

    if (!hasTF32Throughput && !hasFP32Throughput) {
      warnings.push({
        severity: "info",
        category: "precision",
        message:
          "No explicit FP32 throughput is set for this GPU, so FP32 training time falls back to BF16/FP16 TFLOPS divided by 8. Enter the device's vector FP32 TFLOPS for non-TF32 hardware if that heuristic is not appropriate.",
      })
    }
  }

  if (precision === "fp8" && !gpu.supportsFP8) {
    warnings.push({
      severity: "critical",
      category: "precision",
      message: `${gpu.name} does not support FP8 kernels. Select BF16/FP16 or FP8-capable hardware; estimates are disabled for this precision/hardware combination.`,
    })
  }
}

function getAdamWFP8FallbackMessage(
  config: TrainingConfig | PostTrainingConfig,
): string | null {
  if (config.optimizer !== "adamw-fp8") {
    return null
  }

  if (config.precision !== "fp8") {
    return "AdamW FP8 storage requires FP8 precision on FP8-capable hardware. Estimates fall back to AdamW mixed-precision optimizer storage."
  }

  if (!config.hardware.gpu.supportsFP8) {
    return "AdamW FP8 storage requires FP8-capable hardware. Estimates are disabled until the selected precision is supported by the selected GPU."
  }

  if (config.fp8.storageMode === "transformer-engine") {
    return "AdamW FP8 storage requires MS-AMP storage mode. TransformerEngine-style FP8 uses FP8 kernels only, so optimizer memory estimates fall back to AdamW mixed precision."
  }

  return null
}

function getFP8StorageInfoMessage(
  config: TrainingConfig | PostTrainingConfig,
): string | null {
  if (config.precision !== "fp8" || !config.hardware.gpu.supportsFP8) {
    return null
  }

  if (hasInvalidFP8StorageMode(config)) {
    return null
  }

  if (config.fp8.storageMode === "ms-amp") {
    return config.optimizer === "adamw-fp8"
      ? "MS-AMP FP8 storage reduces parameter and gradient memory only; activation tensors and any modeled output logits remain estimated at bf16/fp16 size."
      : "MS-AMP storage mode only reduces model-state memory when AdamW FP8 storage is selected; the current optimizer uses its normal parameter, gradient, and optimizer-state bytes."
  }

  return "TransformerEngine-style FP8 is modeled as kernel throughput only; model states, activation tensors, and any modeled output logits remain estimated at bf16/fp16 size."
}

function addFP8KernelSpeedupWarnings(
  warnings: Warning[],
  config: TrainingConfig | PostTrainingConfig,
): void {
  if (config.precision !== "fp8") {
    return
  }

  if (hasInvalidFP8KernelSpeedupFactor(config)) {
    warnings.push({
      severity: "critical",
      category: "precision",
      message: "FP8 kernel speedup factor must be between 1.0x and 2.0x.",
    })
  }

  if (hasInvalidFP8StorageMode(config)) {
    warnings.push({
      severity: "critical",
      category: "precision",
      message: "FP8 storage mode must be TransformerEngine or MS-AMP.",
    })
  }
}

function addCustomGPUThroughputWarnings(
  warnings: Warning[],
  inputMode: TrainingConfig["hardware"]["inputMode"],
  gpu: TrainingConfig["hardware"]["gpu"],
  precision: TrainingConfig["precision"],
): void {
  if (hasInvalidGPUInputMode(inputMode)) {
    warnings.push({
      severity: "critical",
      category: "hardware",
      message: "GPU input mode must be preset or custom.",
    })
    return
  }

  const hardwareLabel =
    inputMode === "custom" ? "Custom GPU" : "GPU preset"

  const addPositiveWarning = (value: number | null | undefined, label: string) => {
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      warnings.push({
        severity: "critical",
        category: "hardware",
        message: `${hardwareLabel} ${label} must be positive.`,
      })
    }
  }
  const addOptionalPositiveWarning = (
    value: number | null | undefined,
    label: string,
    severity: Warning["severity"],
  ) => {
    if (
      value !== null &&
      value !== undefined &&
      (!Number.isFinite(value) || value <= 0)
    ) {
      warnings.push({
        severity,
        category: "hardware",
        message: `${hardwareLabel} ${label} must be positive when set.`,
      })
    }
  }

  addPositiveWarning(gpu.memoryGB, "memory")
  addPositiveWarning(gpu.halfPrecisionTFLOPS, "BF16/FP16 TFLOPS")
  addPositiveWarning(gpu.memoryBandwidthGBps, "memory bandwidth")
  getInvalidCustomGPUMetadataMessages(gpu).forEach((message) => {
    warnings.push({
      severity: "critical",
      category: "hardware",
      message:
        inputMode === "custom"
          ? message
          : message.replace("Custom GPU", "GPU preset"),
    })
  })

  const hasValidFP32TF32 =
    gpu.supportsTF32 &&
    gpu.tf32TFLOPS !== null &&
    Number.isFinite(gpu.tf32TFLOPS) &&
    gpu.tf32TFLOPS > 0
  addOptionalPositiveWarning(
    gpu.tf32TFLOPS,
    "TF32 TFLOPS",
    precision === "fp32" && gpu.supportsTF32 ? "critical" : "warning",
  )
  addOptionalPositiveWarning(
    gpu.fp32TFLOPS,
    "FP32 TFLOPS",
    precision === "fp32" && !hasValidFP32TF32 ? "critical" : "warning",
  )
  addOptionalPositiveWarning(gpu.fp8TFLOPS, "FP8 TFLOPS", "warning")

  if (
    !Number.isFinite(gpu.gpusPerNode) ||
    gpu.gpusPerNode <= 0 ||
    !Number.isInteger(gpu.gpusPerNode)
  ) {
    warnings.push({
      severity: "critical",
      category: "hardware",
      message: `${hardwareLabel} GPUs per node must be a positive integer.`,
    })
  }

  getSparseThroughputWarningMessages(gpu, inputMode).forEach((message) => {
    warnings.push({
      severity: "warning",
      category: "hardware",
      message,
    })
  })
}

function addIntegerCountWarning(
  warnings: Warning[],
  value: number | null | undefined,
  category: Warning["category"],
  label: string,
): void {
  if (Number.isFinite(value) && !Number.isInteger(value)) {
    warnings.push({
      severity: "critical",
      category,
      message: `${label} must be an integer.`,
    })
  }
}

function addPositiveIntegerWarning(
  warnings: Warning[],
  value: number | null | undefined,
  category: Warning["category"],
  label: string,
): void {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(value) ||
    value <= 0 ||
    !Number.isInteger(value)
  ) {
    warnings.push({
      severity: "critical",
      category,
      message: `${label} must be a positive integer.`,
    })
  }
}

function addNonNegativeIntegerWarning(
  warnings: Warning[],
  value: number | null | undefined,
  category: Warning["category"],
  label: string,
): void {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(value) ||
    value < 0 ||
    !Number.isInteger(value)
  ) {
    warnings.push({
      severity: "critical",
      category,
      message: `${label} must be a non-negative finite integer.`,
    })
  }
}

function addParameterScaleWarnings(
  warnings: Warning[],
  value: number,
  label: string,
): void {
  if (!Number.isFinite(value) || value <= 0) {
    return
  }

  if (value < 1e6) {
    warnings.push({
      severity: "warning",
      category: "compute",
      message: `${label} has fewer than 1M parameters.`,
    })
  }

  if (value > 10e12) {
    warnings.push({
      severity: "warning",
      category: "compute",
      message: `${label} exceeds 10T parameters; estimates may be unreliable at this scale.`,
    })
  }
}

function estimateQLoRALoadingGpuBufferBytes(
  config: PostTrainingConfig,
): number | null {
  const parameterCount = config.baseModel.parameterCount
  const counts = calculateParameterCount(
    config.baseModel.architecture,
    config.baseModel.moe,
    config.sequenceLength,
  )

  if (
    !Number.isFinite(parameterCount) ||
    parameterCount <= 0 ||
    !Number.isInteger(parameterCount) ||
    !Number.isFinite(counts.total) ||
    counts.total <= 0
  ) {
    return null
  }

  const outputHeadParams =
    counts.outputProjection > 0 ? counts.outputProjection : counts.embedding
  const largestBoundaryParams = Math.max(
    counts.embedding + counts.positionalEmbedding,
    outputHeadParams + counts.finalNorm,
  )
  const moeLayerCount =
    config.baseModel.moe.enabled &&
    Number.isFinite(config.baseModel.moe.L_moe)
      ? Math.min(
          Math.max(0, config.baseModel.moe.L_moe),
          config.baseModel.architecture.L,
        )
      : 0
  const denseLayerCount = Math.max(
    0,
    config.baseModel.architecture.L - moeLayerCount,
  )
  const denseLayerParams =
    counts.perLayer.attention + counts.perLayer.ffn + counts.perLayer.norm
  let largestParameterUnitParams =
    denseLayerCount > 0
      ? Math.max(denseLayerParams, largestBoundaryParams)
      : largestBoundaryParams

  if (
    config.baseModel.moe.enabled &&
    counts.moe !== null &&
    moeLayerCount > 0
  ) {
    const moeLayerParams =
      counts.perLayer.attention +
      counts.perLayer.norm +
      counts.moe.routerParameters / moeLayerCount +
      counts.moe.sharedExpertParameters / moeLayerCount +
      counts.moe.expertParameters / moeLayerCount

    largestParameterUnitParams = Math.max(
      largestParameterUnitParams,
      moeLayerParams,
    )
  }

  return largestParameterUnitParams * (parameterCount / counts.total) * 2
}

function addPostTrainingInputWarnings(
  warnings: Warning[],
  config: PostTrainingConfig,
  requestedConfig = config,
): void {
  if (
    !Number.isFinite(config.baseModel.parameterCount) ||
    config.baseModel.parameterCount <= 0
  ) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "Base model parameter count must be positive.",
    })
  }
  addIntegerCountWarning(
    warnings,
    config.baseModel.parameterCount,
    "compute",
    "Base model parameter count",
  )
  addParameterScaleWarnings(
    warnings,
    config.baseModel.parameterCount,
    "Base model",
  )

  if (hasInvalidPostTrainingBaseModelInputMode(requestedConfig)) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "Base model input mode must be preset or parameter-count.",
    })
  }

  if (
    requestedConfig.baseModel.inputMode === "preset" &&
    !MODEL_PRESETS.some(
      (preset) => preset.id === requestedConfig.baseModel.presetId,
    )
  ) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "Selected base-model preset could not be resolved.",
    })
  }

  const requestedHasInvalidMethod = hasInvalidPostTrainingMethod(
    requestedConfig.method,
  )
  const requestedHasInvalidApproach = hasInvalidPostTrainingApproach(
    requestedConfig.approach,
  )
  if (requestedHasInvalidMethod) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message:
        "Selected post-training method is not supported. Post-training estimates are disabled until SFT, DPO, PPO, or GRPO is selected.",
    })
  }
  if (requestedHasInvalidApproach) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message:
        "Selected fine-tuning approach is not supported. Post-training estimates are disabled until full fine-tuning, LoRA, QLoRA, or MeZO is selected.",
    })
  }
  if (
    !requestedHasInvalidMethod &&
    !requestedHasInvalidApproach &&
    hasInvalidPostTrainingMethodApproach(
      requestedConfig.method,
      requestedConfig.approach,
    )
  ) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message:
        "MeZO estimates currently support SFT only. Post-training estimates are disabled until a supported method/approach pair is selected.",
    })
  }

  addArchitectureDimensionWarnings(warnings, config.baseModel.architecture)
  addKVHeadValidationWarnings(warnings, config.baseModel.architecture)
  if (typeof config.baseModel.moe.enabled !== "boolean") {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "Base-model MoE enabled must be true or false.",
    })
  }
  if (
    hasInvalidMoEConfig(
      config.baseModel.moe,
      config.baseModel.architecture.L,
    )
  ) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message:
        "Base-model MoE configuration is invalid. Post-training estimates are disabled until the MoE expert, routing, and layer counts are valid.",
    })
  }
  if (hasInvalidPostTrainingActiveParameterCount(config)) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message:
        "Base-model MoE active parameter count must be positive and no greater than total parameter count.",
    })
  }
  const invalidChunkedCrossEntropy =
    hasInvalidChunkedCrossEntropyFlag(requestedConfig)
  if (invalidChunkedCrossEntropy) {
    warnings.push({
      severity: "critical",
      category: "memory",
      message: "Chunked cross-entropy must be true or false.",
    })
  }

  if (requestedConfig.baseModel.inputMode === "parameter-count") {
    warnings.push({
      severity: "info",
      category: "compute",
      message:
        "Post-training parameter-count mode infers a dense architecture from parameter count and disables MoE-specific structure. Memory and time estimates are rough when the real model has unusual vocabulary size, GQA/MLA dimensions, MoE layers, or long-context settings; use a preset when available.",
    })
  }

  warnings.push({
    severity: "info",
    category: "memory",
    message: !invalidChunkedCrossEntropy && config.chunkedCrossEntropy
      ? "Post-training activation memory assumes full activation checkpointing with a one-layer non-Flash attention recompute workspace. Chunked cross-entropy is enabled, so materialized output logits and the transient fp32 logits-gradient peak are excluded; systems without checkpointing or with additional retained logits can require substantially more VRAM."
      : "Post-training activation memory assumes full activation checkpointing with a one-layer non-Flash attention recompute workspace. Trainable language-model passes include the mixed-precision output logits and transient fp32 logits-gradient peak; enabling chunked cross-entropy or fused loss kernels can eliminate this logits peak, while runs without checkpointing or with additional retained logits can require substantially more VRAM.",
  })
  if (config.approach !== "mezo") {
    warnings.push({
      severity: "info",
      category: "compute",
      message:
        "Post-training non-generation time uses ideal model FLOPs with a coarse MFU discount for small batches, optimizer overhead, and checkpoint recompute. Because memory assumes full activation checkpointing, calibrate MFU from measured tokens/sec for exact wall-clock estimates; disabling checkpointing can be faster but requires more VRAM.",
    })
  }

  if (
    (config.approach === "full" || config.approach === "mezo") &&
    config.trainableParameterPercentage !== null &&
    (!Number.isFinite(config.trainableParameterPercentage) ||
      config.trainableParameterPercentage <= 0 ||
      config.trainableParameterPercentage > 100)
  ) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "Trainable parameter percentage must be greater than 0 and at most 100.",
    })
  }

  if (
    config.approach === "full" &&
    config.trainableParameterPercentage !== null &&
    Number.isFinite(config.trainableParameterPercentage) &&
    config.trainableParameterPercentage > 0 &&
    config.trainableParameterPercentage < 100
  ) {
    warnings.push({
      severity: "warning",
      category: "compute",
      message:
        "Partial full fine-tuning scales gradient, optimizer, and policy backward compute by the trainable fraction, while activation memory stays modeled as full-model checkpointed activations. Compute savings assume the frozen portion is a contiguous set of layers whose backward pass can be skipped; if trainable weights are spread through the model, compute can be close to full fine-tuning.",
    })
  }

  if (
    config.approach === "mezo" &&
    config.trainableParameterPercentage !== null &&
    Number.isFinite(config.trainableParameterPercentage) &&
    config.trainableParameterPercentage > 0 &&
    config.trainableParameterPercentage < 100
  ) {
    warnings.push({
      severity: "info",
      category: "compute",
      message:
        "Partial MeZO changes the reported trainable/frozen parameter split but does not reduce modeled wall-clock compute: the estimate still runs full-model perturbation forwards with no gradient or optimizer-state storage.",
    })
  }

  if (
    !Number.isFinite(config.datasetSizeExamples) ||
    config.datasetSizeExamples < 1
  ) {
    warnings.push({
      severity: "critical",
      category: "data",
      message: "Dataset size must be at least 1 example.",
    })
  }
  addIntegerCountWarning(
    warnings,
    config.datasetSizeExamples,
    "data",
    "Dataset size",
  )

  if (!Number.isFinite(config.epochs) || config.epochs <= 0) {
    warnings.push({
      severity: "critical",
      category: "data",
      message: "Epoch count must be positive.",
    })
  }

  if (!Number.isFinite(config.sequenceLength) || config.sequenceLength <= 0) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "Sequence length must be positive.",
    })
  } else if (config.sequenceLength < 128 || config.sequenceLength > 131072) {
    warnings.push({
      severity: "info",
      category: "compute",
      message: "Sequence length is outside the typical 128-131,072 token post-training range.",
    })
  }
  addIntegerCountWarning(
    warnings,
    config.sequenceLength,
    "compute",
    "Sequence length",
  )

  if (config.baseModel.architecture.attentionVariant === "mla") {
    warnings.push({
      severity: "info",
      category: "compute",
      message:
        "MLA models use architecture-specific latent query/KV dimensions that are not exposed in this calculator. Attention and generation KV-cache estimates fall back to standard hidden-width stand-ins and can be high or low depending on the implementation.",
    })

    if (config.approach === "lora" || config.approach === "qlora") {
      warnings.push({
        severity: "info",
        category: "compute",
        message:
          "LoRA/QLoRA adapter counts for MLA models treat q/k/v/o targets as full hidden-width projection stand-ins. Actual MLA implementations often use architecture-specific target names and latent projection shapes, so override the estimate externally if those dimensions are known.",
      })
    }
  }

  if (!Number.isFinite(config.batchSize) || config.batchSize < 1) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "Batch size must be at least 1.",
    })
  }
  addIntegerCountWarning(warnings, config.batchSize, "compute", "Batch size")

  if (
    !Number.isFinite(requestedConfig.hardware.numGPUs) ||
    requestedConfig.hardware.numGPUs < 1
  ) {
    warnings.push({
      severity: "critical",
      category: "hardware",
      message: "GPU count must be at least 1.",
    })
  }
  addIntegerCountWarning(
    warnings,
    requestedConfig.hardware.numGPUs,
    "hardware",
    "GPU count",
  )

  if (
    requestedConfig.hardware.gpu.singleDeviceOnly &&
    resolveExplicitNumGPUs(requestedConfig.hardware.numGPUs) > 1
  ) {
    warnings.push({
      severity: "critical",
      category: "hardware",
      message: `${config.hardware.gpu.name} only supports single-device execution.`,
    })
  }
  if (
    !config.hardware.gpu.singleDeviceOnly &&
    resolveExplicitNumGPUs(config.hardware.numGPUs) > 1
  ) {
    warnings.push({
      severity: "info",
      category: "memory",
      message:
        "Post-training memory assumes data-parallel replicas: GPU count splits batches, activations, and generation cache, but each active GPU still holds the modeled resident parameters, gradients, optimizer states, and frozen models. ZeRO/FSDP/TP/offload placement for post-training model states is not modeled.",
    })
  }
  addCustomGPUThroughputWarnings(
    warnings,
    requestedConfig.hardware.inputMode,
    requestedConfig.hardware.gpu,
    requestedConfig.precision,
  )

  if (!Number.isFinite(config.costPerGPUHour) || config.costPerGPUHour < 0) {
    warnings.push({
      severity: "critical",
      category: "cost",
      message: "Cost per GPU-hour must be a non-negative finite value.",
    })
  }

  const adamWFP8FallbackMessage = getAdamWFP8FallbackMessage(config)
  if (adamWFP8FallbackMessage !== null) {
    warnings.push({
      severity: "warning",
      category: "precision",
      message: adamWFP8FallbackMessage,
    })
  }

  const fp8StorageInfoMessage = getFP8StorageInfoMessage(config)
  if (fp8StorageInfoMessage !== null) {
    warnings.push({
      severity: "info",
      category: "precision",
      message: fp8StorageInfoMessage,
    })
  }
  addFP8KernelSpeedupWarnings(warnings, config)

  const requestedOptimizerProfile = getOptimizerProfileDefinition(
    requestedConfig.optimizer,
  )
  const requestedHasInvalidOptimizerApproach =
    !requestedHasInvalidApproach &&
    hasInvalidPostTrainingOptimizerApproach(
      requestedConfig.optimizer,
      requestedConfig.approach,
    )
  if (
    !requestedOptimizerProfile ||
    !requestedOptimizerProfile.supportsPostTraining ||
    requestedHasInvalidOptimizerApproach
  ) {
    const message =
      requestedHasInvalidOptimizerApproach && requestedConfig.approach === "mezo"
        ? "The MeZO approach requires the MeZO optimizer. Post-training estimates are disabled until the approach and optimizer agree."
        : requestedHasInvalidOptimizerApproach
          ? "The MeZO optimizer is only valid with the MeZO approach. Post-training estimates are disabled until the approach and optimizer agree."
          : "Selected optimizer is not valid for post-training. Post-training estimates are disabled until a supported optimizer is selected."

    warnings.push({
      severity: "critical",
      category: "compute",
      message,
    })
  }

  if (config.approach === "lora" || config.approach === "qlora") {
    const targetModules = config.lora.targetModules as unknown

    if (!Array.isArray(targetModules)) {
      warnings.push({
        severity: "critical",
        category: "compute",
        message: "LoRA target modules must be an array of supported module IDs.",
      })
    } else if (targetModules.length === 0) {
      warnings.push({
        severity: "critical",
        category: "compute",
        message: "At least one LoRA target module must be selected.",
      })
    } else if (hasInvalidLoRATargetModules(config.lora)) {
      warnings.push({
        severity: "critical",
        category: "compute",
        message:
          "LoRA target modules must be unique supported module IDs.",
      })
    }
  }

  if (
    (config.approach === "lora" || config.approach === "qlora") &&
    !hasInvalidLoRATargetModules(config.lora) &&
    Number.isFinite(config.lora.rank) &&
    config.lora.rank >= 1
  ) {
    const loraParameterCount = calculateLoRAParamCount(config)

    if (!Number.isFinite(loraParameterCount)) {
      warnings.push({
        severity: "critical",
        category: "compute",
        message:
          "LoRA adapter parameter count could not be computed from the current architecture. Check model dimensions, attention heads, and target modules.",
      })
    } else if (loraParameterCount <= 0) {
      warnings.push({
        severity: "critical",
        category: "compute",
        message:
          "Selected LoRA target modules do not map to any adapted matrices for this architecture. Choose attention projections or FFN projections that exist on the base model.",
      })
    }
  }

  if (
    (config.approach === "lora" || config.approach === "qlora") &&
    (config.method === "dpo" || config.method === "ppo" || config.method === "grpo")
  ) {
    warnings.push({
      severity: "info",
      category: "memory",
      message:
        "LoRA/QLoRA RL memory assumes the frozen reference policy shares the actor base weights and disables adapters for reference scoring. Keeping a separate reference replica adds another base-model copy.",
    })
  }

  if (config.method === "ppo" || config.method === "grpo") {
    warnings.push({
      severity: "info",
      category: "generation",
      message:
        "PPO/GRPO generation time treats sequence length as the generated decode horizon and does not model a separate prompt-prefill length. Long prompts add prefill time beyond this estimate.",
    })
  }

  if (hasInvalidPostTrainingKVCachePrecision(config)) {
    warnings.push({
      severity: "critical",
      category: "generation",
      message: "KV cache precision must be BF16, FP16, or INT8.",
    })
  }

  if (config.method === "dpo") {
    const usesSharedReference =
      config.approach === "lora" || config.approach === "qlora"

    warnings.push({
      severity: "info",
      category: "memory",
      message:
        usesSharedReference
          ? "DPO LoRA/QLoRA memory already shares the actor base with the reference path. Pipelines that precompute fixed reference log-probs can still avoid repeated-epoch reference scoring compute; that optimization is not modeled here."
          : "DPO estimates keep the reference-policy path in the training loop. Pipelines that precompute fixed reference log-probs can discard the reference model after the cache pass and avoid repeated-epoch reference scoring; that optimization is not modeled here.",
    })
  }

  if (config.method === "ppo") {
    const usesSharedReference =
      config.approach === "lora" || config.approach === "qlora"

    warnings.push({
      severity: "info",
      category: "memory",
      message: usesSharedReference
        ? "PPO LoRA/QLoRA memory keeps the actor base and adapters, critic, and reward models resident on GPU, with reference scoring sharing the actor base by disabling adapters. Phase-specific model offload or hybrid-engine placement can reduce peak memory, but that optimization is not modeled here."
        : "PPO memory keeps actor, critic, reference, and reward models resident on GPU. Phase-specific model offload or hybrid-engine placement can reduce peak memory, but that optimization is not modeled here.",
    })
  }

  if (config.approach === "qlora") {
    const quantizationBits = config.lora.quantizationBits as number | null
    const cpuLoadingBytes =
      Number.isFinite(config.baseModel.parameterCount) &&
      config.baseModel.parameterCount > 0
        ? config.baseModel.parameterCount * 2
        : null
    const gpuLoadingBufferBytes = estimateQLoRALoadingGpuBufferBytes(config)
    const gpuLoadingBufferClause =
      gpuLoadingBufferBytes !== null
        ? ` and about ${fmtBytes(gpuLoadingBufferBytes)} of short-lived GPU room for the largest dequantized parameter unit`
        : " plus short-lived GPU room for one dequantized parameter unit"
    const cpuLoadingMessage =
      cpuLoadingBytes === null
        ? `QLoRA GPU memory excludes transient loading/dequantization and CPU RAM requirements. Loading a quantized checkpoint still needs host RAM for the fp16 base${gpuLoadingBufferClause}.`
        : `QLoRA GPU memory excludes transient loading/dequantization and CPU RAM requirements. Loading usually needs roughly ${fmtBytes(cpuLoadingBytes)} of host RAM for the fp16 base${gpuLoadingBufferClause}.`

    warnings.push({
      severity:
        cpuLoadingBytes !== null &&
        cpuLoadingBytes > TYPICAL_NODE_CPU_MEMORY_WARNING_BYTES
          ? "warning"
          : "info",
      category: "memory",
      message:
        cpuLoadingBytes !== null &&
        cpuLoadingBytes > TYPICAL_NODE_CPU_MEMORY_WARNING_BYTES
          ? `${cpuLoadingMessage} This can exceed typical node RAM; use streaming, sharded, or offloaded loading if the host cannot hold the fp16 base.`
          : cpuLoadingMessage,
    })
    warnings.push({
      severity: "info",
      category: "compute",
      message:
        "QLoRA time applies a 1.75x slowdown to non-generation quantized policy/reference base passes to approximate dequantize-compute overhead. The factor is based on NF4-style QLoRA measurements; 8-bit quantized LoRA implementations can differ.",
    })
    if (
      quantizationBits !== null &&
      quantizationBits !== 4 &&
      quantizationBits !== 8
    )
      warnings.push({
        severity: "critical",
        category: "memory",
        message: "QLoRA quantization bits must be 4 or 8.",
      })
  }

  if (hasInvalidLoRARank(config)) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "LoRA rank must be at least 1.",
    })
  }
  if (config.approach === "lora" || config.approach === "qlora")
    addIntegerCountWarning(warnings, config.lora.rank, "compute", "LoRA rank")
  if (hasInvalidLoRAAlpha(config)) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "LoRA alpha must be a positive integer.",
    })
  }
  if (config.approach === "lora" || config.approach === "qlora") {
    addIntegerCountWarning(
      warnings,
      config.lora.alpha,
      "compute",
      "LoRA alpha",
    )
  }

  if (
    config.method === "grpo" &&
    (!Number.isFinite(config.grpo.groupSize) ||
      config.grpo.groupSize < 2 ||
      !Number.isInteger(config.grpo.groupSize))
  ) {
    warnings.push({
      severity: "critical",
      category: "generation",
      message: "GRPO group size must be an integer of at least 2.",
    })
  }

  if (
    config.method === "ppo" &&
    (!Number.isFinite(config.ppo.criticModelParameterCount) ||
      !Number.isFinite(config.ppo.rewardModelParameterCount) ||
      config.ppo.criticModelParameterCount <= 0 ||
      config.ppo.rewardModelParameterCount <= 0)
  ) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message:
        "PPO critic and reward model parameter counts must be positive integers.",
    })
  }
  if (config.method === "ppo") {
    addIntegerCountWarning(
      warnings,
      config.ppo.criticModelParameterCount,
      "compute",
      "PPO critic model parameter count",
    )
    addIntegerCountWarning(
      warnings,
      config.ppo.rewardModelParameterCount,
      "compute",
      "PPO reward model parameter count",
    )
    addParameterScaleWarnings(
      warnings,
      config.ppo.criticModelParameterCount,
      "PPO critic model",
    )
    addParameterScaleWarnings(
      warnings,
      config.ppo.rewardModelParameterCount,
      "PPO reward model",
    )
  }

  if (
    config.method === "ppo" &&
    (!Number.isFinite(config.ppo.updateEpochs) ||
      config.ppo.updateEpochs < 1)
  ) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "PPO update epochs must be at least 1.",
    })
  }
  if (config.method === "ppo")
    addIntegerCountWarning(
      warnings,
      config.ppo.updateEpochs,
      "compute",
      "PPO update epochs",
    )
}

function isFinitePositive(value: number): boolean {
  return Number.isFinite(value) && value > 0
}

function getFinitePositiveOrNull(value: number): number | null {
  return Number.isFinite(value) && value > 0 ? value : null
}

function getFinitePositiveIntegerOrNull(value: number): number | null {
  return Number.isFinite(value) && value > 0 && Number.isInteger(value)
    ? value
    : null
}

function isValidMoEEnabled(moe: MoEConfig, layerCount: number): boolean {
  return moe.enabled && !hasInvalidMoEConfig(moe, layerCount)
}

function getAttentionHeadDim(architecture: ModelArchitecture): number {
  const explicitHeadDim = architecture.d_head

  if (explicitHeadDim !== null && explicitHeadDim !== undefined) {
    return typeof explicitHeadDim === "number" &&
      Number.isFinite(explicitHeadDim) &&
      explicitHeadDim > 0 &&
      Number.isInteger(explicitHeadDim)
      ? explicitHeadDim
      : Number.POSITIVE_INFINITY
  }

  return architecture.d / architecture.a
}

function addArchitectureDimensionWarnings(
  warnings: Warning[],
  architecture: ModelArchitecture,
): void {
  getInvalidArchitectureEnumMessages(architecture).forEach((message) => {
    warnings.push({
      severity: "critical",
      category: "compute",
      message,
    })
  })
  if (typeof architecture.tiedEmbeddings !== "boolean")
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "Tied embeddings must be true or false.",
    })

  addPositiveIntegerWarning(
    warnings,
    architecture.d,
    "compute",
    "Hidden dimension d",
  )
  addPositiveIntegerWarning(
    warnings,
    architecture.L,
    "compute",
    "Transformer layer count L",
  )
  addPositiveIntegerWarning(
    warnings,
    architecture.a,
    "compute",
    "Attention head count",
  )
  if (architecture.d_ff !== null)
    addPositiveIntegerWarning(
      warnings,
      architecture.d_ff,
      "compute",
      "FFN dimension d_ff",
    )
  addPositiveIntegerWarning(
    warnings,
    architecture.V,
    "compute",
    "Vocabulary size V",
  )

  if (
    (architecture.d_head === null || architecture.d_head === undefined) &&
    Number.isFinite(architecture.d) &&
    Number.isFinite(architecture.a) &&
    architecture.a > 0 &&
    architecture.d % architecture.a !== 0
  )
    warnings.push({
      severity: "critical",
      category: "compute",
      message:
        "Hidden dimension d must be divisible by attention heads when d_head is not set.",
    })
}

function addKVHeadValidationWarnings(
  warnings: Warning[],
  architecture: ModelArchitecture,
): void {
  const { a, a_kv, d_head } = architecture

  if (
    d_head !== null &&
    d_head !== undefined &&
    (!Number.isFinite(d_head) || d_head <= 0 || !Number.isInteger(d_head))
  ) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "Attention head dimension d_head must be a positive integer when set.",
    })
  }

  if (a_kv === null) {
    return
  }

  if (!Number.isFinite(a_kv) || a_kv <= 0 || !Number.isInteger(a_kv)) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "KV head count must be a positive integer.",
    })
    return
  }

  if (Number.isFinite(a) && a > 0 && a_kv > a) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: `KV head count a_kv=${a_kv} cannot exceed attention heads a=${a}.`,
    })
  } else if (Number.isFinite(a) && a > 0 && a % a_kv !== 0) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: `Attention heads a=${a} must be evenly divisible by KV heads a_kv=${a_kv} for a valid GQA grouping.`,
    })
  }
}

function getOptimizerProfileDefinition(optimizer: TrainingConfig["optimizer"]) {
  return OPTIMIZER_PROFILES.find((candidate) => candidate.id === optimizer)
}

function normalizeParallelismConfig(
  parallelism: ParallelismConfig,
  moeEnabled: boolean,
): ParallelismConfig {
  const base = {
    ...parallelism,
    N_ep: moeEnabled ? parallelism.N_ep : 1,
  }

  if (parallelism.framework !== "fsdp") {
    return {
      ...base,
      fsdpStrategy: null,
    }
  }

  const fsdpStrategy = parallelism.fsdpStrategy ?? "FULL_SHARD"

  return {
    ...base,
    fsdpStrategy,
    zeroStage: resolveFSDPZeroStage(fsdpStrategy),
  }
}

function forceSingleDeviceParallelism(
  parallelism: ParallelismConfig,
): ParallelismConfig {
  return {
    ...parallelism,
    N_dp: 1,
    N_tp: 1,
    N_pp: 1,
    N_cp: 1,
    N_ep: 1,
    VP: 1,
  }
}

function resolveTPShardedFFNWidth(
  arch: ModelArchitecture,
  moe: MoEConfig,
): number | null {
  if (
    moe.enabled &&
    Number.isFinite(moe.L_moe) &&
    moe.L_moe >= arch.L
  ) {
    return null
  }

  if (moe.enabled && moe.denseIntermediateSize != null)
    return moe.denseIntermediateSize
  if (arch.d_ff != null) return arch.d_ff
  const sw =
    arch.ffnType === "swiglu" ||
    arch.ffnType === "geglu" ||
    arch.ffnType === "moe"
  return sw ? Math.round((8 / 3) * arch.d) : 4 * arch.d
}

function usesAFABSchedule(
  parallelism: ParallelismConfig,
  numMicrobatches: number,
): boolean {
  const N_pp = normalizeParallelismDegree(parallelism.N_pp)
  const microbatches = normalizeParallelismDegree(numMicrobatches)

  return (
    parallelism.framework === "fsdp" &&
    N_pp > 1 &&
    resolveEffectiveZeroStage(parallelism) === 2 &&
    microbatches < 2 * N_pp
  )
}

function normalizeParallelismDegree(value: number): number {
  return Number.isFinite(value) && value > 0
    ? Math.max(1, Math.floor(value))
    : 1
}

function getMaxTransformerLayersPerPipelineStage(
  architecture: ModelArchitecture,
  parallelism: ParallelismConfig,
): number {
  const layers =
    Number.isFinite(architecture.L) && architecture.L > 0
      ? Math.max(1, Math.floor(architecture.L))
      : 1
  const N_pp = normalizeParallelismDegree(parallelism.N_pp)

  return Math.max(1, Math.ceil(layers / N_pp))
}

function canUseInterleavedPipelineSchedule(
  parallelism: ParallelismConfig,
  numMicrobatches: number,
): boolean {
  const N_pp = normalizeParallelismDegree(parallelism.N_pp)
  const VP = normalizeParallelismDegree(parallelism.VP)
  const microbatches = normalizeParallelismDegree(numMicrobatches)

  return N_pp > 1 && VP > 1 && microbatches % N_pp === 0
}

function resolveActivationSchedule(
  parallelism: ParallelismConfig,
  numMicrobatches: number,
): PipelineSchedule {
  const N_pp = normalizeParallelismDegree(parallelism.N_pp)

  if (N_pp <= 1) {
    return "none"
  }

  if (usesAFABSchedule(parallelism, numMicrobatches)) {
    return "afab"
  }

  return canUseInterleavedPipelineSchedule(parallelism, numMicrobatches)
    ? "interleaved"
    : "1f1b"
}

function getEffectivePipelineBubbleVP(
  parallelism: ParallelismConfig,
  numMicrobatches: number,
): number {
  return usesAFABSchedule(parallelism, numMicrobatches) ||
    !canUseInterleavedPipelineSchedule(parallelism, numMicrobatches)
    ? 1
    : parallelism.VP
}

function estimateMaxMicroBatch(
  params: ParameterCounts,
  config: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: TrainingConfig["hardware"]["gpu"],
  schedule: PipelineSchedule,
  minVRAMFloor = 0,
): number {
  if (
    hasInvalidTrainingGPUCount(config) ||
    hasInvalidTrainingHardware(
      config.hardware.inputMode,
      gpu,
      config.precision,
    ) ||
    hasInvalidAMPAutocastFlag(config) ||
    hasInvalidChunkedCrossEntropyFlag(config) ||
    hasInvalidFlashAttentionFlag(config) ||
    hasInvalidTorchCompileFlag(config) ||
    hasInvalidPretrainingModelInputMode(config) ||
    hasInvalidParallelismFramework(config) ||
    hasInvalidParallelismMode(config) ||
    hasInvalidSequenceParallelismMode(config) ||
    hasInvalidManualParallelismDegrees(config) ||
    hasInvalidManualShardingMode(config) ||
    hasInvalidManualTensorParallelismTopology(config) ||
    hasInvalidManualTensorExpertSequenceParallelismTopology(config) ||
    hasInvalidManualContextParallelismTopology(config) ||
    hasInvalidManualExpertParallelismTopology(config) ||
    hasInvalidManualPipelineTopology(config) ||
    hasInvalidCPUOffloadConfig(config) ||
    hasInvalidGradientPrecision(config.gradientPrecision) ||
    hasInvalidPretrainingOptimizer(config.optimizer)
  ) {
    return Number.POSITIVE_INFINITY
  }

  const withMicroBatch = (microBatchSize: number): TrainingConfig => ({
    ...config,
    microBatchSize,
  })
  const zeroBatchMemory = calculateTotalMemoryPerGPU(
    params,
    withMicroBatch(0),
    arch,
    moe,
    gpu,
    schedule,
    true,
  )

  if (
    Number.isFinite(minVRAMFloor) &&
    minVRAMFloor > zeroBatchMemory.usableCapacity
  ) {
    return 0
  }

  const oneBatchMemory = calculateTotalMemoryPerGPU(
    params,
    withMicroBatch(1),
    arch,
    moe,
    gpu,
    schedule,
  )
  const perSample = oneBatchMemory.total - zeroBatchMemory.total
  if (!Number.isFinite(perSample) || perSample <= 0) {
    return Math.max(1, Math.floor(config.microBatchSize))
  }

  const available = zeroBatchMemory.usableCapacity - zeroBatchMemory.total
  return available <= 0 ? 0 : Math.max(0, Math.floor(available / perSample))
}

function scaleParameterCounts(
  counts: ParameterCounts,
  targetTotal: number | null,
  targetActive: number | null,
): ParameterCounts {
  if (
    !Number.isFinite(counts.total) ||
    counts.total <= 0 ||
    !Number.isFinite(counts.active) ||
    counts.active <= 0
  ) {
    return markParameterCountsInvalid(counts)
  }

  const totalScale =
    targetTotal !== null &&
    isFinitePositive(targetTotal) &&
    isFinitePositive(counts.total)
      ? targetTotal / counts.total
      : 1
  const activeScale =
    targetActive !== null &&
    isFinitePositive(targetActive) &&
    isFinitePositive(counts.active)
      ? targetActive / counts.active
      : totalScale

  return {
    ...counts,
    total: counts.total * totalScale,
    active: counts.active * activeScale,
    embedding: counts.embedding * totalScale,
    outputProjection: counts.outputProjection * totalScale,
    positionalEmbedding: counts.positionalEmbedding * totalScale,
    finalNorm: counts.finalNorm * totalScale,
    perLayer: {
      attention: counts.perLayer.attention * totalScale,
      ffn: counts.perLayer.ffn * totalScale,
      norm: counts.perLayer.norm * totalScale,
    },
    moe: counts.moe
      ? {
          expertParameters: counts.moe.expertParameters * totalScale,
          routerParameters: counts.moe.routerParameters * totalScale,
          sharedExpertParameters: counts.moe.sharedExpertParameters * totalScale,
          activeRoutedExpertParameters:
            counts.moe.activeRoutedExpertParameters * activeScale,
        }
      : null,
  }
}

function scalePresetParameterCounts(
  counts: ParameterCounts,
  defaultCounts: ParameterCounts,
  targetTotal: number | null,
  targetActive: number | null,
): ParameterCounts {
  if (
    !Number.isFinite(counts.total) ||
    counts.total <= 0 ||
    !Number.isFinite(counts.active) ||
    counts.active <= 0
  ) {
    return markParameterCountsInvalid(counts)
  }

  const defaultNonPositionalTotal =
    defaultCounts.total - defaultCounts.positionalEmbedding
  const currentNonPositionalTotal =
    counts.total - counts.positionalEmbedding
  const targetNonPositionalTotal =
    targetTotal !== null &&
    isFinitePositive(targetTotal) &&
    Number.isFinite(defaultCounts.positionalEmbedding)
      ? targetTotal - defaultCounts.positionalEmbedding
      : null
  const totalScale =
    targetNonPositionalTotal !== null &&
    targetNonPositionalTotal > 0 &&
    isFinitePositive(defaultNonPositionalTotal)
      ? targetNonPositionalTotal / defaultNonPositionalTotal
      : targetTotal !== null &&
          isFinitePositive(targetTotal) &&
          isFinitePositive(defaultCounts.total)
        ? targetTotal / defaultCounts.total
        : 1
  const activeScale =
    targetActive !== null &&
    isFinitePositive(targetActive) &&
    isFinitePositive(defaultCounts.active)
      ? targetActive / defaultCounts.active
      : totalScale

  return {
    ...counts,
    total:
      currentNonPositionalTotal * totalScale + counts.positionalEmbedding,
    active: counts.active * activeScale,
    embedding: counts.embedding * totalScale,
    outputProjection: counts.outputProjection * totalScale,
    positionalEmbedding: counts.positionalEmbedding,
    finalNorm: counts.finalNorm * totalScale,
    perLayer: {
      attention: counts.perLayer.attention * totalScale,
      ffn: counts.perLayer.ffn * totalScale,
      norm: counts.perLayer.norm * totalScale,
    },
    moe: counts.moe
      ? {
          expertParameters: counts.moe.expertParameters * totalScale,
          routerParameters: counts.moe.routerParameters * totalScale,
          sharedExpertParameters: counts.moe.sharedExpertParameters * totalScale,
          activeRoutedExpertParameters:
            counts.moe.activeRoutedExpertParameters * activeScale,
        }
      : null,
  }
}

function markParameterCountsInvalid(counts: ParameterCounts): ParameterCounts {
  const invalid = Number.POSITIVE_INFINITY

  return {
    ...counts,
    total: invalid,
    active: invalid,
    embedding: invalid,
    outputProjection: invalid,
    positionalEmbedding: invalid,
    finalNorm: invalid,
    perLayer: {
      attention: invalid,
      ffn: invalid,
      norm: invalid,
    },
    moe: counts.moe
      ? {
          expertParameters: invalid,
          routerParameters: invalid,
          sharedExpertParameters: invalid,
          activeRoutedExpertParameters: invalid,
        }
      : null,
  }
}

function disableMoEConfig(moe: MoEConfig): MoEConfig {
  return {
    ...moe,
    enabled: false,
    activeParameterCount: null,
  }
}

function applyVocabPaddingToCounts(
  counts: ParameterCounts,
  architecture: ModelArchitecture,
  tpDegree: number,
): ParameterCounts {
  if (
    !Number.isFinite(counts.total) ||
    !Number.isFinite(counts.active) ||
    !Number.isFinite(architecture.V) ||
    architecture.V <= 0 ||
    !Number.isInteger(architecture.V) ||
    !Number.isFinite(architecture.d) ||
    architecture.d <= 0 ||
    !Number.isInteger(architecture.d)
  ) {
    return counts
  }

  const paddedVocab = calculateVocabPadding(architecture.V, tpDegree)

  if (!Number.isFinite(paddedVocab) || paddedVocab === architecture.V) {
    return counts
  }

  const extraEntries = paddedVocab - architecture.V
  const embeddingDelta = extraEntries * architecture.d
  const outputDelta = architecture.tiedEmbeddings ? 0 : extraEntries * architecture.d

  return {
    ...counts,
    total: counts.total + embeddingDelta + outputDelta,
    active: counts.active + extraEntries * architecture.d,
    embedding: counts.embedding + embeddingDelta,
    outputProjection: counts.outputProjection + outputDelta,
  }
}

function infiniteParameterCounts(counts: ParameterCounts): ParameterCounts {
  return {
    ...counts,
    total: Number.POSITIVE_INFINITY,
    active: Number.POSITIVE_INFINITY,
    embedding: Number.POSITIVE_INFINITY,
    outputProjection: Number.POSITIVE_INFINITY,
    positionalEmbedding: Number.POSITIVE_INFINITY,
    finalNorm: Number.POSITIVE_INFINITY,
    perLayer: {
      attention: Number.POSITIVE_INFINITY,
      ffn: Number.POSITIVE_INFINITY,
      norm: Number.POSITIVE_INFINITY,
    },
    moe:
      counts.moe === null
        ? null
        : {
            expertParameters: Number.POSITIVE_INFINITY,
            routerParameters: Number.POSITIVE_INFINITY,
            sharedExpertParameters: Number.POSITIVE_INFINITY,
            activeRoutedExpertParameters: Number.POSITIVE_INFINITY,
          },
  }
}

function resolvePretrainingModel(config: TrainingConfig): {
  architecture: ModelArchitecture
  moe: MoEConfig
  parameterCounts: ParameterCounts
} {
  const preset =
    config.model.inputMode === "preset"
      ? MODEL_PRESETS.find((candidate) => candidate.id === config.model.presetId) ??
        null
      : null

  const architecture = normalizeAttentionVariantHeads(
    config.model.inputMode === "quick"
      ? estimateParametersQuick(config.model.quickMode.totalParameters)
      : preset?.architecture ?? config.model.architecture,
  )
  const moe =
    config.model.inputMode === "preset"
      ? preset?.moe ?? disableMoEConfig(config.model.moe)
      : config.model.inputMode === "quick"
        ? disableMoEConfig(config.model.moe)
        : config.model.moe
  const rawCounts = calculateParameterCount(architecture, moe, config.sequenceLength)

  if (hasInvalidPretrainingModelInputMode(config)) {
    return {
      architecture,
      moe,
      parameterCounts: markParameterCountsInvalid(rawCounts),
    }
  }

  if (config.model.inputMode === "preset" && !preset) {
    return {
      architecture,
      moe,
      parameterCounts: markParameterCountsInvalid(rawCounts),
    }
  }

  if (config.model.inputMode === "quick") {
    if (
      !isFinitePositive(config.model.quickMode.totalParameters) ||
      !Number.isInteger(config.model.quickMode.totalParameters)
    ) {
      return {
        architecture,
        moe,
        parameterCounts: markParameterCountsInvalid(rawCounts),
      }
    }

    return {
      architecture,
      moe,
      parameterCounts: scaleParameterCounts(
        rawCounts,
        config.model.quickMode.totalParameters,
        null,
      ),
    }
  }

  if (preset) {
    const defaultRawCounts = calculateParameterCount(
      architecture,
      moe,
      preset.defaultSequenceLength,
    )

    return {
      architecture,
      moe,
      parameterCounts: scalePresetParameterCounts(
        rawCounts,
        defaultRawCounts,
        preset.parameterCount,
        preset.moe ? preset.activeParameterCount : null,
      ),
    }
  }

  return {
    architecture,
    moe,
    parameterCounts: rawCounts,
  }
}

function resolvePostTrainingConfig(config: PostTrainingConfig): PostTrainingConfig {
  const hardware = config.hardware.gpu.singleDeviceOnly
    ? {
        ...config.hardware,
        numGPUs: 1,
      }
    : config.hardware

  if (hasInvalidPostTrainingBaseModelInputMode(config)) {
    return {
      ...config,
      hardware,
      baseModel: {
        ...config.baseModel,
        parameterCount: Number.POSITIVE_INFINITY,
        architecture: normalizeAttentionVariantHeads(
          config.baseModel.architecture,
        ),
      },
    }
  }

  if (config.baseModel.inputMode === "preset") {
    const preset =
      MODEL_PRESETS.find((candidate) => candidate.id === config.baseModel.presetId) ??
      null

    if (!preset) {
      return {
        ...config,
        hardware,
        baseModel: {
          ...config.baseModel,
          parameterCount: Number.POSITIVE_INFINITY,
          architecture: normalizeAttentionVariantHeads(
            config.baseModel.architecture,
          ),
        },
      }
    }

    return {
      ...config,
      hardware,
      baseModel: {
        ...config.baseModel,
        parameterCount: preset.parameterCount,
        architecture: normalizeAttentionVariantHeads(preset.architecture),
        moe: preset.moe ?? disableMoEConfig(config.baseModel.moe),
      },
    }
  }

  return {
    ...config,
    hardware,
    baseModel: {
      ...config.baseModel,
      architecture: normalizeAttentionVariantHeads(
        estimateParametersQuick(config.baseModel.parameterCount),
      ),
      moe: disableMoEConfig(config.baseModel.moe),
    },
  }
}

function resolvePostTrainingComputeParameterCount(
  config: PostTrainingConfig,
): number {
  if (hasInvalidPostTrainingModelShape(config)) {
    return Number.POSITIVE_INFINITY
  }

  const baseParameterCount = getFinitePositiveIntegerOrNull(
    config.baseModel.parameterCount,
  )

  if (baseParameterCount === null) {
    return Number.POSITIVE_INFINITY
  }

  const activeParameterCount =
    config.baseModel.moe.activeParameterCount === null
      ? null
      : getFinitePositiveIntegerOrNull(
          config.baseModel.moe.activeParameterCount,
        )

  if (
    config.baseModel.moe.enabled &&
    activeParameterCount !== null
  ) {
    return activeParameterCount
  }

  const counts = calculateParameterCount(
    config.baseModel.architecture,
    config.baseModel.moe,
    config.sequenceLength,
  )

  if (!Number.isFinite(counts.total) || counts.total <= 0) {
    return baseParameterCount
  }

  return counts.active * (baseParameterCount / counts.total)
}

function resolveRequestedNumGPUs(
  config: TrainingConfig,
  totalFLOPs: number,
  parameterCounts: ParameterCounts,
  architecture: ModelArchitecture,
  moe: MoEConfig,
): number {
  if (config.hardware.gpu.singleDeviceOnly) {
    return 1
  }

  if (hasInvalidExplicitTrainingGPUCount(config)) {
    return Number.POSITIVE_INFINITY
  }

  if (hasInvalidTargetTrainingDays(config)) {
    return Number.POSITIVE_INFINITY
  }

  const explicitNumGPUs = resolveExplicitNumGPUs(config.hardware.numGPUs)
  const targetDays = config.hardware.targetTrainingDays

  if (
    hasInvalidTrainingHardware(
      config.hardware.inputMode,
      config.hardware.gpu,
      config.precision,
    )
  ) {
    return explicitNumGPUs
  }

  if (
    config.parallelismMode !== "auto" ||
    targetDays === null ||
    !Number.isFinite(targetDays) ||
    targetDays <= 0 ||
    !Number.isFinite(totalFLOPs) ||
    totalFLOPs <= 0
  ) {
    return explicitNumGPUs
  }

  const secondsBudget = targetDays * 86400
  const fPeakFLOPS =
    getEffectiveTrainingTFLOPS(
      config.hardware.gpu,
      config.precision,
      config.fp8,
    ) * 1e12

  if (!Number.isFinite(fPeakFLOPS) || fPeakFLOPS <= 0) {
    return explicitNumGPUs
  }

  // In target-days mode, the GPU input is disabled UI state, not a lower bound.
  let guess = 1

  for (let iteration = 0; iteration < 8; iteration += 1) {
    const recommendation = recommendParallelism(
      parameterCounts,
      architecture,
      config,
      config.hardware.gpu,
      guess,
      moe,
    )
    const recommendedWorldSize = resolveParallelWorldSize(recommendation.config)
    const paddedCounts = applyVocabPaddingToCounts(
      parameterCounts,
      architecture,
      recommendation.config.N_tp,
    )
    const recommendedFLOPs = calculateFLOPs(
      paddedCounts,
      {
        totalTokens: config.totalTokens,
        sequenceLength: config.sequenceLength,
      },
      architecture,
      moe,
    ).totalFLOPs
    const mfuConfig: TrainingConfig = {
      ...config,
      parallelism: recommendation.config,
      hardware: {
        ...config.hardware,
        numGPUs: recommendedWorldSize,
      },
    }
    const mfu = resolveTrainingMFU(
      mfuConfig,
      paddedCounts.active,
      recommendedWorldSize,
    )

    if (!Number.isFinite(mfu) || mfu <= 0) {
      return explicitNumGPUs
    }

    const flopsForTarget =
      Number.isFinite(recommendedFLOPs) && recommendedFLOPs > 0
        ? recommendedFLOPs
        : totalFLOPs
    const timeBasedGPUs = Math.ceil(
      flopsForTarget / Math.max(secondsBudget * fPeakFLOPS * mfu, 1),
    )
    const next = Math.max(1, recommendedWorldSize, timeBasedGPUs)

    if (next === guess) {
      return next
    }

    guess = next
  }

  return guess
}

function resolveParallelWorldSize(parallelism: ParallelismConfig): number {
  return resolveExplicitNumGPUs(getParallelWorldSize(parallelism))
}

function positiveIntegerDegree(value: number): number | null {
  return Number.isFinite(value) && value > 0 && Number.isInteger(value)
    ? value
    : null
}

function hasInvalidParallelismDegrees(parallelism: ParallelismConfig): boolean {
  return [
    parallelism.N_dp,
    parallelism.N_tp,
    parallelism.N_pp,
    parallelism.N_cp,
    parallelism.N_ep,
    parallelism.VP,
  ].some((degree) => positiveIntegerDegree(degree) === null)
}

function hasInvalidManualParallelismDegrees(config: TrainingConfig): boolean {
  return (
    config.parallelismMode === "manual" &&
    hasInvalidParallelismDegrees(config.parallelism)
  )
}

function isParameterGroupEvenlySharded(
  parameterCount: number,
  shardDegree: number,
): boolean {
  if (
    !Number.isFinite(parameterCount) ||
    parameterCount <= 0 ||
    !Number.isFinite(shardDegree) ||
    shardDegree <= 1 ||
    !Number.isInteger(shardDegree)
  ) {
    return true
  }

  return parameterCount % shardDegree === 0
}

function addManualStateShardDivisibilityWarnings(
  warnings: Warning[],
  parameterCounts: ParameterCounts,
  architecture: ModelArchitecture,
  moe: MoEConfig,
  config: TrainingConfig,
  parallelism: ParallelismConfig,
  effectiveZeroStage: ParallelismConfig["zeroStage"],
): void {
  if (effectiveZeroStage <= 0) {
    return
  }

  const N_dp = positiveIntegerDegree(parallelism.N_dp)
  const N_cp = positiveIntegerDegree(parallelism.N_cp)
  const N_tp = positiveIntegerDegree(parallelism.N_tp)
  const N_ep = positiveIntegerDegree(parallelism.N_ep)

  if (N_dp === null || N_cp === null || N_tp === null || N_ep === null) {
    return
  }

  const effectiveCounts = applyVocabPaddingToCounts(
    parameterCounts,
    architecture,
    N_tp,
  )
  const routedExpertParameterCount =
    moe.enabled && effectiveCounts.moe !== null
      ? effectiveCounts.moe.expertParameters
      : 0
  const denseReplicaParameterCount = Math.max(
    0,
    effectiveCounts.total - routedExpertParameterCount,
  )
  const denseStateShardDegree = calculateDenseStateShardDegree({
    ...config,
    parallelism,
  })
  const denseShardLabel = usesFSDPHybridShard(parallelism)
    ? "effective hybrid dense state shard degree"
    : "dense state shard degree N_dp × N_cp"

  if (
    !isParameterGroupEvenlySharded(
      denseReplicaParameterCount,
      denseStateShardDegree,
    )
  ) {
    warnings.push({
      severity: "info",
      category: "parallelism",
      message: `Dense/shared parameter count is not evenly divisible by ${denseShardLabel} = ${denseStateShardDegree}; some frameworks will pad shards automatically.`,
    })
  }

  const expertStateShardDegree = (denseStateShardDegree * N_tp) / N_ep

  if (
    moe.enabled &&
    routedExpertParameterCount > 0 &&
    !isParameterGroupEvenlySharded(
      routedExpertParameterCount,
      expertStateShardDegree,
    )
  ) {
    warnings.push({
      severity: "info",
      category: "parallelism",
      message: `Routed expert parameter count is not evenly divisible by expert state shard degree N_edp = ${expertStateShardDegree}; some frameworks will pad shards automatically.`,
    })
  }
}

function hasIntegerExpertDataParallelDegree(
  config: TrainingConfig,
  parallelism: ParallelismConfig,
): boolean {
  const numerator = calculateExpertDataParallelNumerator(config, parallelism)

  return (
    Number.isFinite(numerator) &&
    Number.isInteger(numerator) &&
    Number.isFinite(parallelism.N_ep) &&
    Number.isInteger(parallelism.N_ep) &&
    parallelism.N_ep > 0 &&
    numerator % parallelism.N_ep === 0
  )
}

function calculateExpertDataParallelNumerator(
  config: TrainingConfig,
  parallelism: ParallelismConfig,
): number {
  return (
    calculateDenseStateShardDegree({
      ...config,
      parallelism,
    }) * parallelism.N_tp
  )
}

function usesFSDPHybridShard(parallelism: ParallelismConfig): boolean {
  return (
    parallelism.framework === "fsdp" &&
    (parallelism.fsdpStrategy === "HYBRID_SHARD" ||
      parallelism.fsdpStrategy === "HYBRID_SHARD_ZERO2")
  )
}

function resolveTrainableParameterCount(config: PostTrainingConfig): number {
  const parameterCount = getPostTrainingParameterCountOrInfinity(
    config.baseModel.parameterCount,
  )
  const percentage = config.trainableParameterPercentage
  const ratio =
    percentage === null
      ? 1
      : Number.isFinite(percentage) && percentage > 0 && percentage <= 100
        ? percentage / 100
        : Number.POSITIVE_INFINITY

  return parameterCount * ratio
}

function getPostTrainingParameterCountOrInfinity(parameterCount: number): number {
  return Number.isFinite(parameterCount) &&
    parameterCount > 0 &&
    Number.isInteger(parameterCount)
    ? parameterCount
    : Number.POSITIVE_INFINITY
}

function multiplyPostTrainingParameterBytes(
  parameterCount: number,
  bytesPerParameter: number,
): number {
  if (!Number.isFinite(bytesPerParameter) || bytesPerParameter <= 0) {
    return 0
  }

  return parameterCount * bytesPerParameter
}

function resolvePostTrainingBatchSize(
  config: PostTrainingConfig,
): number | null {
  return getFinitePositiveIntegerOrNull(config.batchSize)
}

function resolvePostTrainingGRPOGroupSize(config: PostTrainingConfig): number {
  return Number.isFinite(config.grpo.groupSize) &&
    config.grpo.groupSize >= 2 &&
    Number.isInteger(config.grpo.groupSize)
    ? config.grpo.groupSize
    : Number.POSITIVE_INFINITY
}

function getPostTrainingParallelWorkItems(config: PostTrainingConfig): number {
  const batch = resolvePostTrainingBatchSize(config) ?? 0
  return getPostTrainingParallelWorkItemsForPromptBatch(config, batch)
}

function getPostTrainingParallelWorkItemsForPromptBatch(
  config: PostTrainingConfig,
  promptBatch: number,
): number {
  const batch = Number.isFinite(promptBatch) && promptBatch > 0 ? promptBatch : 0

  if (config.method === "grpo") {
    return batch * resolvePostTrainingGRPOGroupSize(config)
  }

  if (config.method === "dpo") {
    return 2 * batch
  }

  return batch
}

function getPostTrainingPromptBatchPlan(config: PostTrainingConfig): {
  promptExamples: number
  batchSize: number
  fullPromptBatches: number
  partialPromptBatch: number
} | null {
  const datasetSizeExamples = getFinitePositiveIntegerOrNull(
    config.datasetSizeExamples,
  )
  const epochs = getFinitePositiveOrNull(config.epochs)
  const batchSize = resolvePostTrainingBatchSize(config)

  if (datasetSizeExamples === null || epochs === null || batchSize === null) {
    return null
  }

  const promptExamples = datasetSizeExamples * epochs
  const fullPromptBatches = Math.floor(promptExamples / batchSize)
  const partialPromptBatch = Math.max(
    0,
    promptExamples - fullPromptBatches * batchSize,
  )

  return {
    promptExamples,
    batchSize,
    fullPromptBatches,
    partialPromptBatch,
  }
}

function estimatePostTrainingMaxEffectiveComputeGPUs(
  config: PostTrainingConfig,
  configuredGPUs: number,
): number {
  const plan = getPostTrainingPromptBatchPlan(config)
  const largestPromptBatch =
    plan === null || plan.fullPromptBatches > 0
      ? (resolvePostTrainingBatchSize(config) ?? 0)
      : plan.partialPromptBatch
  const workItems =
    plan === null
      ? getPostTrainingParallelWorkItems(config)
      : getPostTrainingParallelWorkItemsForPromptBatch(
          config,
          largestPromptBatch,
        )

  return Math.min(configuredGPUs, Math.max(1, Math.ceil(workItems)))
}

function estimatePostTrainingBatchedFLOPSeconds(
  config: PostTrainingConfig,
  totalFLOPs: number,
  policyParams: number,
  fPeakFLOPS: number,
  configuredGPUs: number,
): number {
  if (totalFLOPs <= 0) {
    return 0
  }

  const plan = getPostTrainingPromptBatchPlan(config)
  if (plan === null || plan.promptExamples <= 0) {
    return Number.POSITIVE_INFINITY
  }

  const flopsPerPromptExample = totalFLOPs / plan.promptExamples
  const estimatePromptBatchSeconds = (promptBatch: number): number => {
    if (!Number.isFinite(promptBatch) || promptBatch <= 0) {
      return 0
    }

    const activeGPUs = Math.min(
      configuredGPUs,
      Math.max(
        1,
        Math.ceil(
          getPostTrainingParallelWorkItemsForPromptBatch(config, promptBatch),
        ),
      ),
    )
    const mfu = getDefaultMFU(policyParams, activeGPUs) * 0.85
    const denominator = activeGPUs * fPeakFLOPS * mfu

    return denominator > 0
      ? (promptBatch * flopsPerPromptExample) / denominator
      : Number.POSITIVE_INFINITY
  }

  const fullBatchSeconds =
    plan.fullPromptBatches > 0
      ? estimatePromptBatchSeconds(plan.batchSize) * plan.fullPromptBatches
      : 0

  return fullBatchSeconds + estimatePromptBatchSeconds(plan.partialPromptBatch)
}

function estimatePostTrainingBatchedQLoRAPenaltySeconds(
  config: PostTrainingConfig,
  affectedFLOPs: number,
  totalNonGenerationFLOPs: number,
  policyParams: number,
  fPeakFLOPS: number,
  configuredGPUs: number,
): number {
  if (affectedFLOPs <= 0) {
    return 0
  }

  const plan = getPostTrainingPromptBatchPlan(config)
  if (plan === null || plan.promptExamples <= 0) {
    return Number.POSITIVE_INFINITY
  }

  const cappedAffectedFLOPs =
    Number.isFinite(affectedFLOPs) && Number.isFinite(totalNonGenerationFLOPs)
      ? Math.min(affectedFLOPs, totalNonGenerationFLOPs)
      : affectedFLOPs
  const affectedFLOPsPerPromptExample =
    cappedAffectedFLOPs / plan.promptExamples
  const nonGenerationFLOPsPerPromptExample =
    totalNonGenerationFLOPs / plan.promptExamples
  const estimatePromptBatchSeconds = (promptBatch: number): number => {
    if (!Number.isFinite(promptBatch) || promptBatch <= 0) {
      return 0
    }

    const activeGPUs = Math.min(
      configuredGPUs,
      Math.max(
        1,
        Math.ceil(
          getPostTrainingParallelWorkItemsForPromptBatch(config, promptBatch),
        ),
      ),
    )
    const mfu = getDefaultMFU(policyParams, activeGPUs) * 0.85
    const denominator = activeGPUs * fPeakFLOPS * mfu

    return calculateQLoRAPenaltySeconds(
      affectedFLOPsPerPromptExample * promptBatch,
      nonGenerationFLOPsPerPromptExample * promptBatch,
      denominator,
    )
  }

  const fullBatchSeconds =
    plan.fullPromptBatches > 0
      ? estimatePromptBatchSeconds(plan.batchSize) * plan.fullPromptBatches
      : 0

  return fullBatchSeconds + estimatePromptBatchSeconds(plan.partialPromptBatch)
}

function getPostTrainingMemorySplitLimit(config: PostTrainingConfig): number {
  const batch = resolvePostTrainingBatchSize(config) ?? 0
  return Math.max(
    1,
    getPostTrainingParallelWorkItemsForPromptBatch(config, batch),
  )
}

function getKVCacheBytesPerElement(config: PostTrainingConfig): number {
  if (!isValidKVCachePrecision(config.kvCachePrecision)) {
    return Number.POSITIVE_INFINITY
  }

  return config.kvCachePrecision === "int8" ? 1 : 2
}

interface GenerationFeasibilityEstimate {
  requestedBatch: number
  maxBatch: number
  rounds: number
}

function estimateMaxConcurrentGenerations(
  config: PostTrainingConfig,
  memory: PostTrainingMemoryBreakdown,
): GenerationFeasibilityEstimate | null {
  if (config.method !== "ppo" && config.method !== "grpo") {
    return null
  }

  const arch = normalizeAttentionVariantHeads(config.baseModel.architecture)
  const kvHeads = arch.a_kv ?? arch.a
  const sequenceLength = getFinitePositiveIntegerOrNull(config.sequenceLength)
  const headDim =
    Number.isFinite(arch.d) &&
    arch.d > 0 &&
    Number.isFinite(arch.a) &&
    arch.a > 0
      ? getAttentionHeadDim(arch)
      : 0
  const kvPerSequence =
    sequenceLength !== null &&
    Number.isFinite(arch.L) &&
    arch.L > 0 &&
    Number.isFinite(kvHeads) &&
    kvHeads > 0 &&
    headDim > 0
      ? 2 *
        arch.L *
        kvHeads *
        headDim *
        sequenceLength *
        getKVCacheBytesPerElement(config)
      : 0
  const rolloutBytesPerSequence =
    sequenceLength !== null
      ? POST_TRAINING_ROLLOUT_BYTES_PER_TOKEN * sequenceLength
      : 0
  const configuredGPUs = resolveExplicitNumGPUs(config.hardware.numGPUs)
  const batchSize = resolvePostTrainingBatchSize(config) ?? 0
  const requestedBatch =
    config.method === "grpo"
      ? resolvePostTrainingGRPOGroupSize(config) * batchSize
      : batchSize
  const requestedLocalBatch =
    requestedBatch > 0
      ? Math.max(1, Math.ceil(requestedBatch / configuredGPUs))
      : 0
  const rolloutBytesPerGPU = rolloutBytesPerSequence * requestedLocalBatch
  const generationAvailableBytes =
    memory.usableCapacity / 1.04 -
    memory.parameters -
    memory.gradients -
    memory.optimizerStates -
    memory.frameworkOverhead -
    rolloutBytesPerGPU
  const maxBatchPerGPU =
    kvPerSequence > 0
      ? Math.max(
          0,
          Math.floor(generationAvailableBytes / kvPerSequence),
        )
      : generationAvailableBytes >= 0
        ? Number.POSITIVE_INFINITY
        : 0
  const maxBatch = Number.isFinite(maxBatchPerGPU)
    ? maxBatchPerGPU * configuredGPUs
    : maxBatchPerGPU

  return {
    requestedBatch,
    maxBatch,
    rounds:
      maxBatch > 0 && Number.isFinite(maxBatch)
        ? Math.max(1, Math.ceil(requestedBatch / maxBatch))
        : Number.POSITIVE_INFINITY,
  }
}

function formatGenerationCapacityWarning(
  config: PostTrainingConfig,
  feasibility: GenerationFeasibilityEstimate,
): string {
  const capacity = Math.max(feasibility.maxBatch, 0)
  const effectiveBatch = resolvePostTrainingBatchSize(config) ?? 0
  const effectiveGroup = resolvePostTrainingGRPOGroupSize(config)
  const request =
    config.method === "grpo"
      ? `GRPO requests ${feasibility.requestedBatch.toLocaleString()} concurrent generations (effective batch ${effectiveBatch} x group ${effectiveGroup})`
      : `PPO requests ${feasibility.requestedBatch.toLocaleString()} concurrent generations`

  if (!Number.isFinite(feasibility.maxBatch) || feasibility.maxBatch <= 0) {
    return `${request}, but estimated generation working-set capacity is 0. Splitting generation into rounds cannot help until at least one generation fits; reduce batch/group size, sequence length, KV precision, or resident model memory.`
  }

  const reduction = config.method === "grpo" ? "batch/group size" : "batch size"

  return `${request}, but estimated generation working-set capacity is about ${capacity.toLocaleString()}. Split generation into roughly ${feasibility.rounds.toLocaleString()} rounds or reduce ${reduction}.`
}

function estimateGenerationCrossoverBatch(
  config: PostTrainingConfig,
): number | null {
  if (config.method !== "ppo" && config.method !== "grpo") {
    return null
  }

  const gpu = config.hardware.gpu
  const fPeakTFLOPS = getEffectiveTrainingTFLOPS(
    gpu,
    config.precision,
    config.fp8,
  )
  const fPeakFLOPS = fPeakTFLOPS * 1e12
  const bandwidthBytesPerSecond = gpu.memoryBandwidthGBps * 1e9 * 0.9
  const weightBytes = getPostTrainingGenerationWeightBytes(config)
  const policyParams = resolvePostTrainingComputeParameterCount(config)
  const adapterParams = estimateLoRAAdapterParameterCount(policyParams, config)
  const adapterWeightBytes =
    adapterParams > 0
      ? resolvePostTrainingOptimizerProfile(config).parameterBytes
      : 0
  const streamedWeightBytes =
    policyParams * weightBytes + adapterParams * adapterWeightBytes
  const forwardParams = policyParams + adapterParams

  if (
    !Number.isFinite(fPeakFLOPS) ||
    fPeakFLOPS <= 0 ||
    !Number.isFinite(bandwidthBytesPerSecond) ||
    bandwidthBytesPerSecond <= 0 ||
    !Number.isFinite(streamedWeightBytes) ||
    streamedWeightBytes <= 0 ||
    !Number.isFinite(forwardParams) ||
    forwardParams <= 0
  ) {
    return null
  }

  return (
    (streamedWeightBytes * fPeakFLOPS) /
    (2 * forwardParams * bandwidthBytesPerSecond)
  )
}

function estimateLocalGenerationBatch(
  config: PostTrainingConfig,
  globalBatch: number,
): number {
  if (!Number.isFinite(globalBatch) || globalBatch <= 0) {
    return 0
  }

  const configuredGPUs = resolveExplicitNumGPUs(config.hardware.numGPUs)
  const generationGPUs = Math.min(configuredGPUs, Math.max(1, Math.ceil(globalBatch)))

  return Math.ceil(globalBatch / generationGPUs)
}

function getPostTrainingStepLabels(
  method: PostTrainingConfig["method"],
): {
  countLabel: string
  timeLabel: string
  markdownLabel: string
  singular: string
} {
  if (method === "dpo") {
    return {
      countLabel: "total preference batches",
      timeLabel: "Seconds per Preference Batch",
      markdownLabel: "Preference Batches",
      singular: "preference batch",
    }
  }

  if (method === "ppo") {
    return {
      countLabel: "total rollout batches",
      timeLabel: "Seconds per Rollout Batch",
      markdownLabel: "Rollout Batches",
      singular: "rollout batch",
    }
  }

  if (method === "grpo") {
    return {
      countLabel: "total prompt batches",
      timeLabel: "Seconds per Prompt Batch",
      markdownLabel: "Prompt Batches",
      singular: "prompt batch",
    }
  }

  return {
    countLabel: "total batches",
    timeLabel: "Seconds per Batch",
    markdownLabel: "Batches",
    singular: "batch",
  }
}

function estimatePostTrainingGenerationSeconds(
  config: PostTrainingConfig,
  policyParams: number,
  feasibility: GenerationFeasibilityEstimate | null,
): number {
  if (config.method !== "ppo" && config.method !== "grpo") {
    return 0
  }

  if (feasibility === null || feasibility.requestedBatch === 0) {
    return 0
  }

  if (
    !Number.isFinite(feasibility.requestedBatch) ||
    feasibility.requestedBatch < 0
  ) {
    return Number.POSITIVE_INFINITY
  }

  if (!Number.isFinite(feasibility.maxBatch) || feasibility.maxBatch <= 0) {
    return Number.POSITIVE_INFINITY
  }

  const datasetSizeExamples = getFinitePositiveIntegerOrNull(
    config.datasetSizeExamples,
  )
  const epochs = getFinitePositiveOrNull(config.epochs)
  const batchSize = resolvePostTrainingBatchSize(config)
  const sequenceLength = getFinitePositiveIntegerOrNull(config.sequenceLength)

  if (
    datasetSizeExamples === null ||
    epochs === null ||
    batchSize === null ||
    sequenceLength === null
  ) {
    return Number.POSITIVE_INFINITY
  }

  // The UI exposes a single sequence length for post-training. Treat it as the
  // generated/scored token horizon and avoid inventing a separate prompt split.
  const estimateBatchSeconds = (requestedBatch: number): number => {
    if (!Number.isFinite(requestedBatch) || requestedBatch < 0) {
      return Number.POSITIVE_INFINITY
    }

    if (requestedBatch === 0) {
      return 0
    }

    const generationBatch = Math.ceil(requestedBatch)
    const maxBatchPerRound = Math.floor(feasibility.maxBatch)
    if (maxBatchPerRound <= 0) {
      return Number.POSITIVE_INFINITY
    }

    const batchPerRound = Math.min(generationBatch, maxBatchPerRound)
    const fullRounds = Math.floor(generationBatch / batchPerRound)
    const remainderBatch = Math.max(
      0,
      generationBatch - fullRounds * batchPerRound,
    )
    const fullRound = calculateGenerationTime(
      policyParams,
      config,
      batchPerRound,
      sequenceLength,
      0,
    )
    const remainderRound =
      remainderBatch > 0
        ? calculateGenerationTime(
            policyParams,
            config,
            remainderBatch,
            sequenceLength,
            0,
          )
        : null

    return (
      fullRound.totalSeconds * fullRounds +
      (remainderRound?.totalSeconds ?? 0)
    )
  }
  const requestedGenerationsForPromptBatch = (promptBatch: number): number =>
    config.method === "grpo"
      ? promptBatch * resolvePostTrainingGRPOGroupSize(config)
      : promptBatch
  const promptExamples = datasetSizeExamples * epochs
  const fullPromptBatches = Math.floor(promptExamples / batchSize)
  const partialPromptBatch = Math.max(
    0,
    promptExamples - fullPromptBatches * batchSize,
  )

  const fullBatchSeconds =
    fullPromptBatches > 0
      ? estimateBatchSeconds(feasibility.requestedBatch) * fullPromptBatches
      : 0

  return (
    fullBatchSeconds +
    estimateBatchSeconds(requestedGenerationsForPromptBatch(partialPromptBatch))
  )
}

function estimateQLoRAAffectedNonGenerationFLOPs(
  config: PostTrainingConfig,
  policyParams: number,
  totalTokens: number,
): number {
  if (config.approach !== "qlora") {
    return 0
  }

  if (
    !Number.isFinite(policyParams) ||
    policyParams <= 0 ||
    !Number.isFinite(totalTokens) ||
    totalTokens < 0
  ) {
    return totalTokens === 0 ? 0 : Number.POSITIVE_INFINITY
  }

  const ppoUpdateEpochs =
    Number.isFinite(config.ppo.updateEpochs) &&
    config.ppo.updateEpochs >= 1 &&
    Number.isInteger(config.ppo.updateEpochs)
      ? config.ppo.updateEpochs
      : Number.POSITIVE_INFINITY
  const actorBaseFLOPsPerToken =
    config.method === "sft"
      ? 4 * policyParams
      : config.method === "dpo"
        ? 6 * policyParams
        : config.method === "ppo"
          ? ppoUpdateEpochs * 6 * policyParams
          : 6 * policyParams

  const actorBaseMoELoadBalanceFLOPsPerToken =
    config.method === "sft"
      ? estimatePostTrainingMoELoadBalanceFLOPsPerToken(
          policyParams,
          config,
          4,
        )
      : config.method === "dpo"
        ? estimatePostTrainingMoELoadBalanceFLOPsPerToken(
            policyParams,
            config,
            6,
          )
        : config.method === "ppo"
          ? ppoUpdateEpochs *
            estimatePostTrainingMoELoadBalanceFLOPsPerToken(
              policyParams,
              config,
              6,
            )
          : estimatePostTrainingMoELoadBalanceFLOPsPerToken(
              policyParams,
              config,
              6,
            )

  return (
    actorBaseFLOPsPerToken + actorBaseMoELoadBalanceFLOPsPerToken
  ) * totalTokens
}

function calculateQLoRAPenaltySeconds(
  affectedFLOPs: number,
  totalNonGenerationFLOPs: number,
  denominatorFLOPsPerSecond: number,
): number {
  if (
    affectedFLOPs <= 0 ||
    denominatorFLOPsPerSecond <= 0 ||
    !Number.isFinite(denominatorFLOPsPerSecond)
  ) {
    return 0
  }

  const cappedAffectedFLOPs =
    Number.isFinite(affectedFLOPs) && Number.isFinite(totalNonGenerationFLOPs)
      ? Math.min(affectedFLOPs, totalNonGenerationFLOPs)
      : affectedFLOPs

  return Number.isFinite(cappedAffectedFLOPs)
    ? ((QLORA_THROUGHPUT_PENALTY - 1) * cappedAffectedFLOPs) /
        denominatorFLOPsPerSecond
    : Number.POSITIVE_INFINITY
}

// ── Post-training memory dispatchers ──

function calculateSFTFullMemory(
  config: PostTrainingConfig,
): PostTrainingMemoryBreakdown {
  const optimizer = resolvePostTrainingOptimizerProfile(config)
  const totalParamCount = getPostTrainingParameterCountOrInfinity(
    config.baseModel.parameterCount,
  )
  const trainableParamCount = resolveTrainableParameterCount(config)
  const frozenParamCount =
    Number.isFinite(totalParamCount) && Number.isFinite(trainableParamCount)
      ? Math.max(totalParamCount - trainableParamCount, 0)
      : 0
  const parameters = multiplyPostTrainingParameterBytes(
    totalParamCount,
    optimizer.parameterBytes,
  )
  const gradients = multiplyPostTrainingParameterBytes(
    trainableParamCount,
    optimizer.betaGrad,
  )
  const optimizerStates = multiplyPostTrainingParameterBytes(
    trainableParamCount,
    optimizer.kOpt,
  )
  const activations = calculatePostTrainingActivationMemory(
    config.baseModel.architecture,
    config,
  )
  const frameworkOverhead = 2e9
  const total =
    (parameters + gradients + optimizerStates + activations + frameworkOverhead) *
    1.04
  const gpuCapacity = config.hardware.gpu.memoryGB * 1e9
  const usableCapacity = gpuCapacity * 0.9
  return {
    parameters,
    gradients,
    optimizerStates,
    activations,
    communicationBuffers: 0,
    frameworkOverhead,
    freeHeadroom: Math.max(0, usableCapacity - total),
    total,
    gpuCapacity,
    usableCapacity,
    fits: total <= usableCapacity,
    trainableModels:
      multiplyPostTrainingParameterBytes(
        trainableParamCount,
        optimizer.parameterBytes,
      ) +
      gradients +
      optimizerStates,
    frozenModels: multiplyPostTrainingParameterBytes(
      frozenParamCount,
      optimizer.parameterBytes,
    ),
    loraAdapter: 0,
    ppoBuffers: 0,
    items: [
      {
        label:
          frozenParamCount > 0
            ? "Trainable model parameters"
            : "Model parameters",
        category: "trainable",
        bytes: multiplyPostTrainingParameterBytes(
          trainableParamCount,
          optimizer.parameterBytes,
        ),
      },
      ...(frozenParamCount > 0
        ? [
            {
              label: "Frozen model parameters",
              category: "frozen" as const,
              bytes: multiplyPostTrainingParameterBytes(
                frozenParamCount,
                optimizer.parameterBytes,
              ),
            },
          ]
        : []),
      { label: "Gradients", category: "trainable", bytes: gradients },
      {
        label: "Optimizer states",
        category: "trainable",
        bytes: optimizerStates,
      },
      { label: "Activations", category: "buffer", bytes: activations },
    ],
  }
}

function calculateMeZOMemory(
  config: PostTrainingConfig,
): PostTrainingMemoryBreakdown {
  const wb = config.precision === "fp32" ? 4 : 2
  const totalParamCount = getPostTrainingParameterCountOrInfinity(
    config.baseModel.parameterCount,
  )
  const trainableParamCount = resolveTrainableParameterCount(config)
  const frozenParamCount =
    Number.isFinite(totalParamCount) && Number.isFinite(trainableParamCount)
      ? Math.max(totalParamCount - trainableParamCount, 0)
      : 0
  const parameters = multiplyPostTrainingParameterBytes(totalParamCount, wb)
  const activations =
    calculatePostTrainingForwardWorkingMemory(
      config.baseModel.architecture,
      config,
    ) +
    calculatePostTrainingOutputLogitsMemory(
      config.baseModel.architecture,
      config,
    )
  const frameworkOverhead = 1e9
  const total = Number.isFinite(trainableParamCount)
    ? (parameters + activations + frameworkOverhead) * 1.04
    : Number.POSITIVE_INFINITY
  const gpuCapacity = config.hardware.gpu.memoryGB * 1e9
  const usableCapacity = gpuCapacity * 0.9
  return {
    parameters,
    gradients: 0,
    optimizerStates: 0,
    activations,
    communicationBuffers: 0,
    frameworkOverhead,
    freeHeadroom: Math.max(0, usableCapacity - total),
    total,
    gpuCapacity,
    usableCapacity,
    fits: total <= usableCapacity,
    trainableModels: multiplyPostTrainingParameterBytes(trainableParamCount, wb),
    frozenModels: multiplyPostTrainingParameterBytes(frozenParamCount, wb),
    loraAdapter: 0,
    ppoBuffers: 0,
    items: [
      {
        label:
          frozenParamCount > 0
            ? "Trainable model parameters"
            : "Model parameters",
        category: "trainable",
        bytes: multiplyPostTrainingParameterBytes(trainableParamCount, wb),
      },
      ...(frozenParamCount > 0
        ? [
            {
              label: "Frozen model parameters",
              category: "frozen" as const,
              bytes: multiplyPostTrainingParameterBytes(frozenParamCount, wb),
            },
          ]
        : []),
      {
        label: "Forward activations",
        category: "buffer",
        bytes: activations,
      },
    ],
  }
}

function getPostTrainingMemory(
  config: PostTrainingConfig,
): PostTrainingMemoryBreakdown {
  if (
    hasInvalidPostTrainingGPUCount(config) ||
    hasInvalidTrainingHardware(
      config.hardware.inputMode,
      config.hardware.gpu,
      config.precision,
    ) ||
    hasInvalidPostTrainingModelShape(config) ||
    hasInvalidPostTrainingOptimizer(config.optimizer) ||
    hasInvalidGradientPrecision(config.gradientPrecision) ||
    hasInvalidFP8Config(config) ||
    hasInvalidPostTrainingKVCachePrecision(config) ||
    hasInvalidChunkedCrossEntropyFlag(config) ||
    hasInvalidPostTrainingApproachConfig(config) ||
    hasInvalidPostTrainingMethodConfig(config) ||
    hasInvalidQLoRAQuantizationBits(config) ||
    hasInvalidLoRARank(config) ||
    hasInvalidLoRAAlpha(config) ||
    hasInvalidPostTrainingTrainablePercentage(config) ||
    ((config.approach === "lora" || config.approach === "qlora") &&
      hasInvalidLoRATargetModules(config.lora))
  ) {
    const gpuCapacity =
      Number.isFinite(config.hardware.gpu.memoryGB) &&
      config.hardware.gpu.memoryGB > 0
        ? config.hardware.gpu.memoryGB * 1e9
        : 0
    const usableCapacity = gpuCapacity * 0.9

    return {
      parameters: Number.POSITIVE_INFINITY,
      gradients: Number.POSITIVE_INFINITY,
      optimizerStates: Number.POSITIVE_INFINITY,
      activations: Number.POSITIVE_INFINITY,
      communicationBuffers: Number.POSITIVE_INFINITY,
      frameworkOverhead: Number.POSITIVE_INFINITY,
      freeHeadroom: 0,
      total: Number.POSITIVE_INFINITY,
      gpuCapacity,
      usableCapacity,
      fits: false,
      trainableModels: Number.POSITIVE_INFINITY,
      frozenModels: Number.POSITIVE_INFINITY,
      loraAdapter: Number.POSITIVE_INFINITY,
      ppoBuffers: Number.POSITIVE_INFINITY,
      items: [],
    }
  }

  if (config.method === "dpo") return calculateDPOMemory(config)
  if (config.method === "ppo") return calculatePPOMemory(config)
  if (config.method === "grpo") return calculateGRPOMemory(config)
  // SFT
  if (config.approach === "mezo") return calculateMeZOMemory(config)
  if (config.approach === "lora") return calculateLoRAMemory(config)
  if (config.approach === "qlora") return calculateQLoRAMemory(config)
  return calculateSFTFullMemory(config)
}

function withPostTrainingGPUCount(
  config: PostTrainingConfig,
  numGPUs: number,
): PostTrainingConfig {
  return {
    ...config,
    hardware: {
      ...config.hardware,
      numGPUs,
    },
  }
}

function getPostTrainingStateFloorBytes(
  memory: PostTrainingMemoryBreakdown,
): number {
  return (
    memory.parameters +
    memory.gradients +
    memory.optimizerStates +
    memory.frameworkOverhead
  ) * 1.04
}

function estimatePostTrainingRequiredGPUs(config: PostTrainingConfig): {
  numGPUsNeeded: number | null
  stateFloorBytes: number
  maxUsefulGPUs: number
  mode: PostTrainingGPURequirementMode | null
} {
  if (
    hasInvalidPostTrainingGPUCount(config) ||
    hasInvalidTrainingHardware(
      config.hardware.inputMode,
      config.hardware.gpu,
      config.precision,
    ) ||
    hasInvalidPostTrainingModelShape(config) ||
    hasInvalidPostTrainingOptimizer(config.optimizer) ||
    hasInvalidGradientPrecision(config.gradientPrecision) ||
    hasInvalidFP8Config(config) ||
    hasInvalidPostTrainingKVCachePrecision(config) ||
    hasInvalidChunkedCrossEntropyFlag(config) ||
    hasInvalidPostTrainingApproachConfig(config) ||
    hasInvalidPostTrainingMethodConfig(config) ||
    hasInvalidQLoRAQuantizationBits(config) ||
    hasInvalidLoRARank(config) ||
    hasInvalidLoRAAlpha(config) ||
    hasInvalidPostTrainingTrainablePercentage(config) ||
    ((config.approach === "lora" || config.approach === "qlora") &&
      hasInvalidLoRATargetModules(config.lora))
  ) {
    return {
      numGPUsNeeded: null,
      stateFloorBytes: Number.POSITIVE_INFINITY,
      maxUsefulGPUs: 1,
      mode: null,
    }
  }

  const maxUsefulGPUs = config.hardware.gpu.singleDeviceOnly
    ? 1
    : getPostTrainingMemorySplitLimit(config)
  const oneGpuMemory = getPostTrainingMemory(withPostTrainingGPUCount(config, 1))
  const stateFloorBytes = getPostTrainingStateFloorBytes(oneGpuMemory)
  const persistentStateBytes =
    oneGpuMemory.parameters +
    oneGpuMemory.gradients +
    oneGpuMemory.optimizerStates
  const shardedStateCapacity =
    oneGpuMemory.usableCapacity / 1.04 - oneGpuMemory.frameworkOverhead
  const stateShardedLowerBound =
    Number.isFinite(persistentStateBytes) &&
    persistentStateBytes > 0 &&
    Number.isFinite(shardedStateCapacity) &&
    shardedStateCapacity > 0
      ? Math.max(1, Math.ceil(persistentStateBytes / shardedStateCapacity))
      : null

  if (config.hardware.gpu.singleDeviceOnly) {
    return {
      numGPUsNeeded: oneGpuMemory.fits ? 1 : null,
      stateFloorBytes,
      maxUsefulGPUs,
      mode: oneGpuMemory.fits ? "data-parallel" : null,
    }
  }

  if (stateFloorBytes > oneGpuMemory.usableCapacity) {
    return {
      numGPUsNeeded: stateShardedLowerBound,
      stateFloorBytes,
      maxUsefulGPUs,
      mode:
        stateShardedLowerBound !== null
          ? "state-sharded-lower-bound"
          : null,
    }
  }

  const maxUsefulMemory = getPostTrainingMemory(
    withPostTrainingGPUCount(config, maxUsefulGPUs),
  )
  if (!maxUsefulMemory.fits) {
    return {
      numGPUsNeeded: null,
      stateFloorBytes,
      maxUsefulGPUs,
      mode: null,
    }
  }

  let low = 1
  let high = maxUsefulGPUs
  while (low < high) {
    const mid = Math.floor((low + high) / 2)
    const candidate = getPostTrainingMemory(withPostTrainingGPUCount(config, mid))

    if (candidate.fits) {
      high = mid
    } else {
      low = mid + 1
    }
  }

  return {
    numGPUsNeeded: low,
    stateFloorBytes,
    maxUsefulGPUs,
    mode: "data-parallel",
  }
}

// ── Input validation (spec Section 14) ──

function generateInputWarnings(
  config: TrainingConfig,
  architecture: ModelArchitecture,
  moe: MoEConfig,
  totalParams: number,
  parameterCounts: ParameterCounts,
  parallelism: ParallelismConfig,
  numGPUs: number,
  chinchillaRatio: number,
  powerLawOptimalTokens: number,
  effectiveLossTokens: number,
  requestedConfig = config,
): Warning[] {
  const w: Warning[] = []
  const totalTokensValid = isFinitePositive(config.totalTokens)
  const uniqueTokensValid = isFinitePositive(config.uniqueTokens)
  const uniqueTokenRatio =
    totalTokensValid && uniqueTokensValid
      ? config.totalTokens / config.uniqueTokens
      : null
  const effectiveZeroStage =
    parallelism.framework === "fsdp"
      ? resolveFSDPZeroStage(parallelism.fsdpStrategy)
      : parallelism.zeroStage
  const configWithResolvedParallelism = { ...config, parallelism }
  const invalidCPUOffload = hasInvalidCPUOffloadConfig(
    configWithResolvedParallelism,
  )
  const validMoEEnabled = isValidMoEEnabled(moe, architecture.L)

  if (!Number.isFinite(totalParams) || totalParams <= 0)
    w.push({
      severity: "critical",
      category: "compute",
      message: "Parameter count must be positive.",
    })
  if (hasInvalidPretrainingModelInputMode(requestedConfig))
    w.push({
      severity: "critical",
      category: "compute",
      message: "Model input mode must be preset, quick, or detailed.",
    })
  if (
    requestedConfig.model.inputMode === "preset" &&
    !MODEL_PRESETS.some((preset) => preset.id === requestedConfig.model.presetId)
  )
    w.push({
      severity: "critical",
      category: "compute",
      message: "Selected model preset could not be resolved.",
    })
  if (requestedConfig.model.inputMode === "quick") {
    if (
      !Number.isFinite(requestedConfig.model.quickMode.totalParameters) ||
      requestedConfig.model.quickMode.totalParameters <= 0
    )
      w.push({
        severity: "critical",
        category: "compute",
        message: "Quick-mode parameter count must be positive.",
      })
    addIntegerCountWarning(
      w,
      requestedConfig.model.quickMode.totalParameters,
      "compute",
      "Quick-mode parameter count",
    )
    w.push({
      severity: "info",
      category: "compute",
      message:
        "Quick mode infers layers, heads, hidden size, vocabulary, and FFN shape from parameter count. FLOPs are useful for rough planning, but activation, logits, KV-cache, and tensor-parallel divisibility estimates can be off for specific architectures; use Preset or Detailed mode for exact fit checks.",
    })
  }
  if (totalParams > 0 && totalParams < 1e6)
    w.push({
      severity: "warning",
      category: "compute",
      message: "Model has fewer than 1M parameters.",
    })
  if (totalParams > 10e12)
    w.push({
      severity: "warning",
      category: "compute",
      message:
        "Model exceeds 10T parameters — estimates may be unreliable at this scale.",
    })
  if (Number.isFinite(totalParams) && totalParams > 0) {
    const cpuInitBytes = 4 * totalParams

    if (cpuInitBytes > 0.8e12) {
      w.push({
        severity: "warning",
        category: "memory",
        message: `Standard fp32 CPU initialization would materialize about ${(cpuInitBytes / 1e12).toFixed(1)} TB of parameters per node before sharding. Use meta-device, partitioned, or ZeRO-init style initialization unless the host has enough RAM.`,
      })
    }
  }
  addArchitectureDimensionWarnings(w, architecture)
  addKVHeadValidationWarnings(w, architecture)
  if (architecture.attentionVariant === "mla")
    w.push({
      severity: "info",
      category: "compute",
      message:
        "MLA attention uses architecture-specific latent query/KV dimensions that are not exposed in this calculator. Attention FLOPs and KV-shaped estimates use standard hidden-width stand-ins and can be high or low depending on the implementation.",
    })
  if (!totalTokensValid)
    w.push({
      severity: "critical",
      category: "data",
      message: "Total training tokens must be positive.",
    })
  else if (config.totalTokens < 200e9)
    w.push({
      severity: "warning",
      category: "data",
      message:
        "Training on fewer than 200B tokens is usually below the practical data floor for useful pretraining, even when the Chinchilla ratio looks acceptable.",
    })
  addIntegerCountWarning(w, config.totalTokens, "data", "Total training tokens")
  if (!uniqueTokensValid)
    w.push({
      severity: "critical",
      category: "data",
      message: "Unique token count must be positive.",
    })
  addIntegerCountWarning(w, config.uniqueTokens, "data", "Unique token count")
  if (
    totalTokensValid &&
    uniqueTokensValid &&
    config.uniqueTokens > config.totalTokens
  )
    w.push({
      severity: "info",
      category: "data",
      message:
        "Unique token count exceeds total training tokens, so this is treated as less than one epoch over a larger corpus with no data repetition.",
    })
  const effectiveRecommendationTokens =
    Number.isFinite(effectiveLossTokens) && effectiveLossTokens > 0
      ? effectiveLossTokens
      : config.totalTokens
  const powerLawOptimalRatio =
    Number.isFinite(effectiveRecommendationTokens) &&
    Number.isFinite(powerLawOptimalTokens) &&
    powerLawOptimalTokens > 0
      ? effectiveRecommendationTokens / powerLawOptimalTokens
      : null
  if (
    powerLawOptimalRatio !== null &&
    powerLawOptimalRatio > 0 &&
    powerLawOptimalRatio < 1
  )
    w.push({
      severity: "warning",
      category: "data",
      message:
        effectiveRecommendationTokens < config.totalTokens
          ? "Effective token count after repeated-data discounting is below the power-law Chinchilla-optimal target — model may be undertrained despite repeated tokens."
          : "Token count is below the power-law Chinchilla-optimal target — model may be undertrained.",
    })
  if (Number.isFinite(chinchillaRatio) && chinchillaRatio > 5000)
    w.push({
      severity: "critical",
      category: "data",
      message:
        "Extreme overtraining (>5000x Chinchilla). Standard scaling law coefficients are not calibrated for this regime.",
    })
  else if (Number.isFinite(chinchillaRatio) && chinchillaRatio > 500)
    w.push({
      severity: "warning",
      category: "data",
      message:
        "Far beyond Chinchilla optimal (>500x). Loss predictions become less reliable.",
    })
  if (uniqueTokenRatio !== null && uniqueTokenRatio > 40)
    w.push({
      severity: "critical",
      category: "data",
      message: `Training for ${uniqueTokenRatio.toFixed(0)} epochs — additional repetition is effectively wasted compute.`,
    })
  else if (uniqueTokenRatio !== null && uniqueTokenRatio > 4)
    w.push({
      severity: "warning",
      category: "data",
      message: `Training for ${uniqueTokenRatio.toFixed(1)} epochs — repeated data is in the diminishing-returns regime.`,
    })
  if (validMoEEnabled)
    w.push({
      severity: "info",
      category: "data",
      message:
        "MoE scaling guidance uses active parameters with dense Chinchilla-style coefficients. MoE-specific scaling studies suggest the optimal token-to-active-parameter ratio can be lower for large sparse models, so treat the token recommendation as approximate.",
    })
  if (validMoEEnabled)
    w.push({
      severity: "info",
      category: "compute",
      message:
        "MoE FFN parameter and FLOP counts assume three-projection SwiGLU/GeGLU-style experts and dense FFN blocks. Two-projection ReLU/GELU MoE implementations will have a smaller FFN parameter count at the same intermediate size.",
    })
  if (!isFinitePositive(config.microBatchSize))
    w.push({
      severity: "critical",
      category: "compute",
      message: "Micro-batch size must be at least 1.",
    })
  addIntegerCountWarning(w, config.microBatchSize, "compute", "Micro-batch size")
  if (!isFinitePositive(config.gradientAccumulationSteps))
    w.push({
      severity: "critical",
      category: "compute",
      message: "Gradient accumulation steps must be at least 1.",
    })
  addIntegerCountWarning(
    w,
    config.gradientAccumulationSteps,
    "compute",
    "Gradient accumulation steps",
  )
  if (!isFinitePositive(config.sequenceLength))
    w.push({
      severity: "critical",
      category: "compute",
      message: "Sequence length must be positive.",
    })
  else if (config.sequenceLength < 512 || config.sequenceLength > 131072)
    w.push({
      severity: "info",
      category: "compute",
      message: "Sequence length is outside the typical 512–131,072 token planning range.",
    })
  addIntegerCountWarning(w, config.sequenceLength, "compute", "Sequence length")
  if (typeof moe.enabled !== "boolean")
    w.push({
      severity: "critical",
      category: "compute",
      message: "MoE enabled must be true or false.",
    })
  if (moe.enabled === true) {
    if (!isFinitePositive(moe.E) || !Number.isInteger(moe.E))
      w.push({
        severity: "critical",
        category: "compute",
        message: "MoE total experts E must be a positive integer.",
      })
    if (
      !isFinitePositive(moe.topk) ||
      !Number.isInteger(moe.topk) ||
      (Number.isFinite(moe.E) && moe.topk > moe.E)
    )
      w.push({
        severity: "critical",
        category: "compute",
        message: "MoE active experts topk must be a positive integer no larger than E.",
      })
    if (
      !isFinitePositive(moe.L_moe) ||
      !Number.isInteger(moe.L_moe) ||
      moe.L_moe > architecture.L
    )
      w.push({
        severity: "critical",
        category: "compute",
        message:
          "MoE layer count L_moe must be a positive integer no larger than the model layer count.",
      })
    if (!Number.isFinite(moe.E_s) || moe.E_s < 0 || !Number.isInteger(moe.E_s))
      w.push({
        severity: "critical",
        category: "compute",
        message: "MoE shared expert count E_s must be a non-negative integer.",
      })
    if (!Number.isFinite(moe.loadBalanceFactor) || moe.loadBalanceFactor < 1)
      w.push({
        severity: "critical",
        category: "compute",
        message: "MoE load-balance factor must be at least 1.",
      })
    if (
      moe.denseIntermediateSize !== null &&
      (!isFinitePositive(moe.denseIntermediateSize) ||
        !Number.isInteger(moe.denseIntermediateSize))
    )
      w.push({
        severity: "critical",
        category: "compute",
        message: "MoE dense FFN size must be a positive integer when specified.",
      })
    if (
      moe.expertIntermediateSize !== null &&
      (!isFinitePositive(moe.expertIntermediateSize) ||
        !Number.isInteger(moe.expertIntermediateSize))
    )
      w.push({
        severity: "critical",
        category: "compute",
        message: "MoE expert FFN size must be a positive integer when specified.",
      })
  }
  const requestedNumGPUs = resolveExplicitNumGPUs(
    requestedConfig.hardware.numGPUs,
  )

  if (
    requestedConfig.hardware.numGPUs !== null &&
    (!Number.isFinite(requestedConfig.hardware.numGPUs) ||
      requestedConfig.hardware.numGPUs < 1)
  )
    w.push({
      severity: "critical",
      category: "hardware",
      message: "GPU count must be at least 1.",
    })
  addIntegerCountWarning(
    w,
    requestedConfig.hardware.numGPUs,
    "hardware",
    "GPU count",
  )
  if (Number.isFinite(numGPUs) && numGPUs > 100000)
    w.push({
      severity: "warning",
      category: "hardware",
      message: "GPU count exceeds 100,000.",
    })
  if (requestedConfig.hardware.gpu.singleDeviceOnly && requestedNumGPUs > 1)
    w.push({
      severity: "critical",
      category: "hardware",
      message: `${config.hardware.gpu.name} only supports single-device execution.`,
    })
  if (
    requestedConfig.hardware.gpu.singleDeviceOnly &&
    requestedConfig.parallelismMode === "manual" &&
    getParallelWorldSize(requestedConfig.parallelism) > 1
  )
    w.push({
      severity: "critical",
      category: "parallelism",
      message:
        "Manual multi-rank parallelism is unavailable on single-device hardware; estimates force DP=TP=PP=CP=EP=1.",
    })
  if (
    config.parallelismMode === "manual" &&
    config.hardware.targetTrainingDays !== null &&
    Number.isFinite(config.hardware.targetTrainingDays) &&
    config.hardware.targetTrainingDays > 0
  )
    w.push({
      severity: "info",
      category: "hardware",
      message:
        "Target training days applies only in auto parallelism; manual estimates use the configured world size.",
    })
  if (
    hasInvalidTargetTrainingDays(requestedConfig)
  )
    w.push({
      severity: "critical",
      category: "hardware",
      message: "Target training days must be positive when set.",
    })
  addPrecisionSupportWarnings(w, config.precision, config.hardware.gpu)
  addCustomGPUThroughputWarnings(
    w,
    requestedConfig.hardware.inputMode,
    requestedConfig.hardware.gpu,
    requestedConfig.precision,
  )
  const requestedOptimizerProfile = getOptimizerProfileDefinition(
    requestedConfig.optimizer,
  )
  if (requestedOptimizerProfile && !requestedOptimizerProfile.supportsPretraining)
    w.push({
      severity: "critical",
      category: "compute",
      message: `${requestedOptimizerProfile.name} is fine-tuning only and is not a valid pretraining optimizer. Pretraining estimates are disabled until a pretraining optimizer is selected.`,
    })
  const adamWFP8FallbackMessage = getAdamWFP8FallbackMessage(config)
  if (adamWFP8FallbackMessage !== null)
    w.push({
      severity: "warning",
      category: "precision",
      message: adamWFP8FallbackMessage,
    })
  const fp8StorageInfoMessage = getFP8StorageInfoMessage(config)
  if (fp8StorageInfoMessage !== null)
    w.push({
      severity: "info",
      category: "precision",
      message: fp8StorageInfoMessage,
    })
  addFP8KernelSpeedupWarnings(w, config)
  if (hasInvalidChunkedCrossEntropyFlag(requestedConfig))
    w.push({
      severity: "critical",
      category: "memory",
      message: "Chunked cross-entropy must be true or false.",
    })
  if (hasInvalidTorchCompileFlag(requestedConfig))
    w.push({
      severity: "critical",
      category: "memory",
      message: "torch.compile must be true or false.",
    })
  if (hasInvalidParallelismFramework(requestedConfig))
    w.push({
      severity: "critical",
      category: "parallelism",
      message:
        "Parallelism framework must be Megatron, DeepSpeed, PyTorch FSDP, or Hugging Face Trainer.",
    })
  if (
    parallelism.framework === "fsdp" &&
    config.gradientPrecision === "bf16"
  )
    w.push({
      severity: "info",
      category: "precision",
      message:
        "PyTorch FSDP upcasts gradients to full precision by default before the optimizer step. The BF16 gradient estimate assumes keep_low_precision_grads=true or an equivalent low-precision-gradient optimizer path.",
    })
  if (
    parallelism.framework === "fsdp" &&
    !config.ampAutocast &&
    config.precision !== "fp32" &&
    !(
      config.precision === "fp8" &&
      config.fp8.storageMode === "ms-amp"
    )
  )
    w.push({
      severity: "info",
      category: "memory",
      message:
        "Native PyTorch FSDP mixed precision is modeled with fp32 resident parameter shards and up to two transient low-precision all-gathered wrapping units, rather than a persistent low-precision parameter copy plus separate fp32 master weights.",
    })
  if (config.ampAutocast)
    w.push({
      severity: "info",
      category: "memory",
      message:
        "AMP autocast is modeled with fp32 resident parameters and no separate master-weight copy. Activation memory uses the selected autocast precision, so model-state savings differ from explicit bf16/fp16 distributed training.",
    })
  if (isFinitePositive(config.microBatchSize) && config.microBatchSize <= 2)
    w.push({
      severity: "info",
      category: "compute",
      message:
        "Micro-batch size ≤ 2 may significantly reduce MFU due to memory-bandwidth-bound matmuls.",
    })
  if (
    config.mfuOverride !== null &&
    (!Number.isFinite(config.mfuOverride) ||
      config.mfuOverride <= 0 ||
      config.mfuOverride > MAX_MFU_OVERRIDE)
  )
    w.push({
      severity: "critical",
      category: "compute",
      message:
        "MFU override must be greater than 0 and at most the calibrated 70% upper range.",
    })
  if (
    !Number.isFinite(config.pricing.costPerGPUHour) ||
    config.pricing.costPerGPUHour < 0
  )
    w.push({
      severity: "critical",
      category: "cost",
      message: "Cost per GPU-hour must be a non-negative finite value.",
    })
  if (
    !Number.isFinite(config.pricing.checkpointRetentionCount) ||
    config.pricing.checkpointRetentionCount < 0
  )
    w.push({
      severity: "critical",
      category: "cost",
      message: "Checkpoint retention count cannot be negative.",
    })
  addIntegerCountWarning(
    w,
    config.pricing.checkpointRetentionCount,
    "cost",
    "Checkpoint retention count",
  )
  if (
    !Number.isFinite(config.pricing.storagePricePerGBMonth) ||
    config.pricing.storagePricePerGBMonth < 0
  )
    w.push({
      severity: "critical",
      category: "cost",
      message: "Storage price cannot be negative.",
    })
  if (
    !Number.isFinite(config.pricing.datasetStorageGB) ||
    config.pricing.datasetStorageGB < 0
  )
    w.push({
      severity: "critical",
      category: "cost",
      message: "Dataset storage must be a non-negative finite value.",
    })
  if (
    !Number.isFinite(config.failureModel.failureRatePerInstancePerDay) ||
    config.failureModel.failureRatePerInstancePerDay < 0 ||
    !Number.isFinite(config.failureModel.recoveryTimeHours) ||
    config.failureModel.recoveryTimeHours < 0 ||
    !Number.isFinite(config.failureModel.checkpointFrequencyPerDay) ||
    config.failureModel.checkpointFrequencyPerDay < 0
  )
    w.push({
      severity: "critical",
      category: "cost",
      message: "Failure rate, recovery time, and checkpoint frequency must be non-negative finite values.",
    })
  const surfacesFailureAdjustedTime = shouldSurfaceFailureAdjustedTime(numGPUs)

  if (
    surfacesFailureAdjustedTime &&
    Number.isFinite(config.failureModel.failureRatePerInstancePerDay) &&
    config.failureModel.failureRatePerInstancePerDay > 0 &&
    Number.isFinite(config.failureModel.checkpointFrequencyPerDay) &&
    config.failureModel.checkpointFrequencyPerDay <= 0
  )
    w.push({
      severity: "critical",
      category: "cost",
      message:
        "Failure recovery needs a positive checkpoint frequency; set checkpoint frequency above 0/day or failure-adjusted training time diverges.",
    })
  if (
    surfacesFailureAdjustedTime &&
    Number.isFinite(config.failureModel.failureRatePerInstancePerDay) &&
    config.failureModel.failureRatePerInstancePerDay > 0 &&
    Number.isFinite(config.failureModel.checkpointFrequencyPerDay) &&
    config.failureModel.checkpointFrequencyPerDay > 0 &&
    Number.isFinite(config.pricing.checkpointRetentionCount) &&
    config.pricing.checkpointRetentionCount <= 0
  )
    w.push({
      severity: "critical",
      category: "cost",
      message:
        "Failure recovery needs at least one retained checkpoint; set checkpoint retention to 1 or more, or failure-adjusted training time diverges.",
    })

  if (!VALID_ZERO_COMMUNICATION_BUCKET_MODES.has(config.zeroCommunication.mode))
    w.push({
      severity: "critical",
      category: "memory",
      message:
        "ZeRO communication bucket mode must be HF auto, DeepSpeed defaults, or custom.",
    })
  if (typeof config.zeroCommunication.overlapComm !== "boolean")
    w.push({
      severity: "critical",
      category: "memory",
      message: "ZeRO communication overlap must be true or false.",
    })
  if (config.zeroCommunication.mode === "custom") {
    addNonNegativeIntegerWarning(
      w,
      config.zeroCommunication.allgatherBucketSizeElements,
      "memory",
      "Custom ZeRO allgather bucket size",
    )
    addNonNegativeIntegerWarning(
      w,
      config.zeroCommunication.reduceBucketSizeElements,
      "memory",
      "Custom ZeRO reduce bucket size",
    )
    addNonNegativeIntegerWarning(
      w,
      config.zeroCommunication.prefetchBucketSizeElements,
      "memory",
      "Custom ZeRO prefetch bucket size",
    )
  }

  if (config.activationCheckpointing === "partial") {
    const maxCheckpointedLayersPerStage =
      getMaxTransformerLayersPerPipelineStage(architecture, parallelism)

    if (
      config.partialCheckpointDepth === null ||
      !isFinitePositive(config.partialCheckpointDepth)
    )
      w.push({
        severity: "critical",
        category: "memory",
        message: "Partial checkpointing depth must be at least 1.",
      })
    addIntegerCountWarning(
      w,
      config.partialCheckpointDepth,
      "memory",
      "Partial checkpointing depth",
    )
    if (
      Number.isFinite(config.partialCheckpointDepth) &&
      config.partialCheckpointDepth !== null &&
      config.partialCheckpointDepth > maxCheckpointedLayersPerStage
    )
      w.push({
        severity: "critical",
        category: "memory",
        message: `Partial checkpointing depth must not exceed ${maxCheckpointedLayersPerStage.toLocaleString()} transformer layer${maxCheckpointedLayersPerStage === 1 ? "" : "s"} per pipeline stage for the current PP=${normalizeParallelismDegree(
          parallelism.N_pp,
        )} layout.`,
      })
  }

  // Manual parallelism validation
  if (config.parallelismMode === "manual") {
    const invalidDegrees = [
      parallelism.N_dp,
      parallelism.N_tp,
      parallelism.N_pp,
      parallelism.N_cp,
      parallelism.N_ep,
      parallelism.VP,
    ].some((degree) => !isFinitePositive(degree))

    if (invalidDegrees)
      w.push({
        severity: "critical",
        category: "parallelism",
        message: "Manual parallelism degrees must be positive finite values.",
      })
    const nonIntegerDegrees = [
      parallelism.N_dp,
      parallelism.N_tp,
      parallelism.N_pp,
      parallelism.N_cp,
      parallelism.N_ep,
      parallelism.VP,
    ].some((degree) => Number.isFinite(degree) && !Number.isInteger(degree))

    if (nonIntegerDegrees)
      w.push({
        severity: "critical",
        category: "parallelism",
        message: "Manual parallelism degrees must be integers.",
      })

    if (config.model.inputMode === "detailed") {
      const hiddenAlignment = validateHiddenDimAlignment(architecture.d)
      if (!hiddenAlignment.valid)
        w.push({
          severity: "warning",
          category: "parallelism",
          message: hiddenAlignment.message,
        })
    }

    const dff = resolveTPShardedFFNWidth(architecture, moe)
    const tp = validateTPDivisibility(
      parallelism.N_tp,
      architecture.d,
      architecture.a,
      architecture.a_kv,
      dff,
    )
    if (!tp.valid)
      w.push({ severity: "critical", category: "parallelism", message: tp.message })
    const pp = validatePPDivisibility(parallelism.N_pp, architecture.L)
    if (!pp.valid)
      w.push({ severity: "critical", category: "parallelism", message: pp.message })
    else if (usesEmbeddingAwarePipelinePartition(parallelism.N_pp, architecture.L))
      w.push({
        severity: "info",
        category: "parallelism",
        message: `PP=${parallelism.N_pp} uses embedding-aware partitioning: input and output embedding stages are treated as virtual layers, so first and last stages carry fewer transformer blocks.`,
      })
    const paddedVocab = calculateVocabPadding(architecture.V, parallelism.N_tp)
    if (
      parallelism.N_tp > 1 &&
      Number.isFinite(architecture.V) &&
      architecture.V > 0 &&
      Number.isInteger(architecture.V) &&
      Number.isFinite(paddedVocab) &&
      paddedVocab > architecture.V
    )
      w.push({
        severity: "info",
        category: "parallelism",
        message: `Vocabulary padded from ${architecture.V.toLocaleString()} to ${paddedVocab.toLocaleString()} for TP=${parallelism.N_tp}.`,
      })
    const ws = validateWorldSize(parallelism, numGPUs)
    if (!ws.valid)
      w.push({ severity: "critical", category: "parallelism", message: ws.message })
    const zp = validateZeroPPCompatibility(
      effectiveZeroStage,
      parallelism.N_pp,
      parallelism.framework,
    )
    if (!zp.valid)
      w.push({ severity: "critical", category: "parallelism", message: zp.message })
    if (
      parallelism.framework === "fsdp" &&
      effectiveZeroStage === 2 &&
      parallelism.N_pp > 1 &&
      !usesAFABSchedule(parallelism, config.gradientAccumulationSteps)
    )
      w.push({
        severity: "critical",
        category: "parallelism",
        message:
          "FSDP SHARD_GRAD_OP / HYBRID_SHARD_ZERO2 with PP is only modeled under the AFAB fallback condition (num_microbatches < 2 x N_pp). Use DeepSpeed ZeRO-1 or FSDP NO_SHARD for 1F1B/interleaved PP.",
      })
    const tpEpSp = validateTensorExpertSequenceParallelism(
      parallelism,
      validMoEEnabled,
    )
    if (!tpEpSp.valid)
      w.push({
        severity: "critical",
        category: "parallelism",
        message: tpEpSp.message,
      })
    if (usesAFABSchedule(parallelism, config.gradientAccumulationSteps)) {
      w.push({
        severity: "info",
        category: "parallelism",
        message:
          "FSDP SHARD_GRAD_OP + PP will fall back to the AFAB schedule here. This relaxes the 1F1B microbatch minimum but increases activation residency.",
      })
      if (parallelism.VP > 1) {
        w.push({
          severity: "info",
          category: "parallelism",
          message:
            "Virtual pipeline chunks are ignored for this AFAB schedule; VP only reduces bubble for interleaved 1F1B.",
        })
      }
    } else {
      const mb = validateMicrobatches(
        config.gradientAccumulationSteps,
        parallelism.N_pp,
        parallelism.VP,
      )
      if (!mb.valid)
        w.push({
          severity: "critical",
          category: "parallelism",
          message: mb.message,
        })
    }
    if (
      validMoEEnabled &&
      parallelism.N_pp > 1 &&
      moe.L_moe > 0 &&
      moe.L_moe < architecture.L
    )
      w.push({
        severity: "info",
        category: "memory",
        message:
          "MoE memory under PP assumes MoE layers are distributed evenly across pipeline stages. If MoE layers are clustered, the peak stage can require more VRAM than shown.",
      })
    const localParallelismGroupSize = getParallelismLocalGroupSize(
      config.hardware.gpu,
    )
    if (parallelism.N_tp > localParallelismGroupSize)
      w.push({
        severity: "critical",
        category: "parallelism",
        message: `N_tp=${parallelism.N_tp} exceeds the local high-bandwidth group size of ${localParallelismGroupSize}.`,
      })
    if (!moe.enabled && parallelism.N_ep > 1)
      w.push({
        severity: "warning",
        category: "parallelism",
        message: "Expert parallelism is only meaningful for MoE models.",
      })
    if (
      validMoEEnabled &&
      parallelism.N_ep > 1 &&
      moe.E % parallelism.N_ep !== 0
    )
      w.push({
        severity: "critical",
        category: "parallelism",
        message: `N_ep=${parallelism.N_ep} must divide the total expert count E=${moe.E}.`,
      })
    const expertDataParallelNumerator = calculateExpertDataParallelNumerator(
      config,
      parallelism,
    )
    if (
      validMoEEnabled &&
      parallelism.N_ep > 1 &&
      !hasIntegerExpertDataParallelDegree(config, parallelism)
    )
      w.push({
        severity: "critical",
        category: "parallelism",
        message: `N_ep=${parallelism.N_ep} must divide ${
          usesFSDPHybridShard(parallelism)
            ? "the effective hybrid shard degree × N_tp"
            : "N_dp × N_cp × N_tp"
        } (${expertDataParallelNumerator}) so expert data parallelism is an integer.`,
      })
    if (
      validMoEEnabled &&
      parallelism.N_ep > 1 &&
      parallelism.N_tp * parallelism.N_ep > localParallelismGroupSize
    )
      w.push({
        severity: "critical",
        category: "parallelism",
        message: `N_tp × N_ep must stay within the local high-bandwidth group (${localParallelismGroupSize}) for expert traffic.`,
      })
    addManualStateShardDivisibilityWarnings(
      w,
      parameterCounts,
      architecture,
      moe,
      config,
      parallelism,
      effectiveZeroStage,
    )
    const cp = validateContextParallelDivisibility(
      parallelism.N_cp,
      config.sequenceLength,
    )
    if (!cp.valid)
      w.push({
        severity: "critical",
        category: "parallelism",
        message: cp.message,
      })
    if (
      parallelism.N_cp > 1 &&
      config.sequenceLength / parallelism.N_cp < 2048
    )
      w.push({
        severity: "warning",
        category: "parallelism",
        message: `CP=${parallelism.N_cp} leaves fewer than ~2K tokens per rank, which can hurt arithmetic intensity.`,
      })
    if (
      parallelism.N_cp > 1 &&
      (config.hardware.gpu.interconnect === "pcie" ||
        config.hardware.gpu.interconnect === "none" ||
        parallelism.N_tp * parallelism.N_cp > localParallelismGroupSize)
    )
      w.push({
        severity: "warning",
        category: "parallelism",
        message: `CP=${parallelism.N_cp} adds high-bandwidth sequence-shard traffic; scaling may be poor when CP extends beyond a node or runs on PCIe-only GPUs.`,
      })
    if (
      parallelism.N_tp > 1 &&
      (config.hardware.gpu.interconnect === "pcie" ||
        config.hardware.gpu.interconnect === "none")
    )
      w.push({
        severity: "warning",
        category: "parallelism",
        message: `TP=${parallelism.N_tp} on PCIe-only GPU will be severely bandwidth-limited.`,
      })
    if (parallelism.N_tp > 1 && config.hardware.gpu.id === "rtx-3090")
      w.push({
        severity: "info",
        category: "parallelism",
        message:
          "RTX 3090 TP=2 assumes a paired NVLink bridge (~112.5 GB/s), which is much slower than datacenter NVLink and is not present in unbridged multi-GPU builds.",
      })
    if (parallelism.N_tp > 1 && config.hardware.gpu.id === "h100-nvl")
      w.push({
        severity: "info",
        category: "parallelism",
        message:
          "H100 NVL TP assumes a paired NVLink bridge; larger H100 NVL hosts are typically multiple bridge pairs, not one all-to-all NVLink domain.",
      })
    const bubble = calculatePipelineBubble(
      parallelism.N_pp,
      config.gradientAccumulationSteps,
      getEffectivePipelineBubbleVP(
        parallelism,
        config.gradientAccumulationSteps,
      ),
    )
    if (parallelism.N_pp > 1 && bubble > 0.5)
      w.push({
        severity: "warning",
        category: "parallelism",
        message: `Pipeline bubble is ${(bubble * 100).toFixed(1)}%. Increase gradient accumulation steps to reduce idle time. Default MFU includes this schedule efficiency; manual MFU overrides should include it too.`,
      })
    else if (parallelism.N_pp > 1 && bubble > 0.2)
      w.push({
        severity: "info",
        category: "parallelism",
        message: `Pipeline bubble is ${(bubble * 100).toFixed(1)}%. A common rule of thumb is num_microbatches ≥ ${4 * parallelism.N_pp}. Default MFU includes this schedule efficiency; manual MFU overrides should include it too.`,
      })
    if (effectiveZeroStage === 3)
      w.push({
        severity: "info",
        category: "parallelism",
        message:
          "ZeRO-3 / FULL_SHARD maximizes memory savings but adds extra communication overhead and can reduce throughput.",
      })
  }

  if (config.cpuOffload === "optimizer-only" && effectiveZeroStage < 1)
    w.push({
      severity: "critical",
      category: "memory",
      message:
        "Optimizer offload requires ZeRO-1, ZeRO-2, ZeRO-3, or an equivalent FSDP sharding strategy.",
    })
  if (config.cpuOffload === "optimizer-and-params" && effectiveZeroStage !== 3)
    w.push({
      severity: "critical",
      category: "memory",
      message:
        "Parameter offload requires ZeRO-3 or FSDP FULL_SHARD / HYBRID_SHARD.",
    })
  if (config.cpuOffload !== "none" && !invalidCPUOffload) {
    const offloadEfficiency = calculateCPUOffloadEfficiency(config)
    const efficiencyLabel =
      offloadEfficiency > 0 && Number.isFinite(offloadEfficiency)
        ? ` Modeled throughput efficiency is ${(offloadEfficiency * 100).toFixed(1)}% before other communication overheads.`
        : ""

    w.push({
      severity: "warning",
      category: "memory",
      message:
        `CPU offloading reduces GPU memory pressure but slows training because optimizer or parameter traffic shifts onto the host interconnect.${efficiencyLabel} The training-time estimate does not apply this as a separate multiplier; lower the MFU override to include it.`,
    })
    w.push({
      severity: "info",
      category: "memory",
      message:
        "CPU offloading here applies only to model states. Activation tensors, recomputation working memory, logits peaks, and communication buffers are still modeled as GPU-resident; activation CPU offload is a separate technique and is not modeled.",
    })
  }

  return w
}

// ── Export formatters ──

function fmtBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "--"
  if (bytes === 0) return "0 B"
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`
  return "< 1 KB"
}

function fmtCount(n: number): string {
  if (!Number.isFinite(n)) return "--"
  if (Math.abs(n) >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return n.toLocaleString()
}

function fmtFLOPs(flops: number): string {
  if (!Number.isFinite(flops) || flops < 0) return "--"
  if (flops >= 1e21) return `${(flops / 1e21).toFixed(2)} ZFLOPs`
  if (flops >= 1e18) return `${(flops / 1e18).toFixed(2)} EFLOPs`
  if (flops >= 1e15) return `${(flops / 1e15).toFixed(2)} PFLOPs`
  if (flops >= 1e12) return `${(flops / 1e12).toFixed(2)} TFLOPs`
  return `${(flops / 1e9).toFixed(2)} GFLOPs`
}

function fmtMultiplier(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(2)}x` : "--"
}

function fmtFractionPercent(value: number): string {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "--"
}

function fmtBatchRelation(
  relation: PretrainingOutput["batchEfficiency"]["relation"],
): string {
  if (relation === "below") return "below B_crit, time-inefficient"
  if (relation === "above") return "above B_crit, compute-inefficient"
  if (relation === "near") return "near B_crit"
  return "B_crit unavailable"
}

function fmtDuration(hours: number): string {
  if (!Number.isFinite(hours) || hours < 0) return "--"
  if (hours === 0) return "0 min"
  if (hours >= 24 * 365) return `${(hours / (24 * 365)).toFixed(1)} years`
  if (hours >= 24) return `${(hours / 24).toFixed(1)} days`
  if (hours >= 1) return `${hours.toFixed(1)} hr`

  const minutes = hours * 60
  return minutes >= 1 ? `${Math.round(minutes)} min` : "< 1 min"
}

function fmtCurrency(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "--"
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`
  if (value >= 1e3) return `$${Math.round(value).toLocaleString()}`
  return `$${value.toFixed(2)}`
}

const WARNING_PRIORITY: Record<Warning["severity"], number> = {
  critical: 0,
  warning: 1,
  info: 2,
}

const WARNING_LABEL: Record<Warning["severity"], string> = {
  critical: "Error",
  warning: "Warning",
  info: "Info",
}

function formatWarningsMarkdown(warnings: Warning[]): string[] {
  if (warnings.length === 0) return []

  const sortedWarnings = [...warnings].sort(
    (left, right) =>
      WARNING_PRIORITY[left.severity] - WARNING_PRIORITY[right.severity],
  )

  return [
    "",
    "## Warnings",
    ...sortedWarnings.map(
      (warning) =>
        `- ${WARNING_LABEL[warning.severity]} (${warning.category}): ${warning.message}`,
    ),
  ]
}

function formatDataRepetitionMarkdown(
  dataRepetition: PretrainingOutput["dataRepetition"],
): string[] {
  if (!dataRepetition.hasRepetition) return []

  return [
    "",
    "## Data Repetition",
    `- Epochs: ${Number.isFinite(dataRepetition.epochs) ? dataRepetition.epochs.toFixed(1) : "--"}`,
    `- Effective Data Ceiling: ${fmtCount(dataRepetition.effectiveDataCeiling)} tokens`,
    `- Assessment: ${dataRepetition.recommendation}`,
  ]
}

function formatPostTrainingMemoryItemsMarkdown(
  items: PostTrainingModelMemoryLineItem[],
): string[] {
  const sortedItems = [...items]
    .filter((item) => Number.isFinite(item.bytes) && item.bytes > 0)
    .sort((left, right) => right.bytes - left.bytes)

  if (sortedItems.length === 0) return []

  const itemTotal = Math.max(
    sortedItems.reduce((sum, item) => sum + item.bytes, 0),
    1,
  )

  return [
    "",
    "## Memory Line Items",
    ...sortedItems.map((item) => {
      const share = (item.bytes / itemTotal) * 100
      return [
        `- ${item.label}: ${fmtBytes(item.bytes)}`,
        `(${share.toFixed(1)}% of listed items)`,
      ].join(" ")
    }),
  ]
}

function formatPostTrainingGPURequirementMarkdown(
  output: PostTrainingOutput,
): string {
  if (output.numGPUsNeeded === null) {
    return "No data-parallel fit"
  }

  const count = fmtCount(output.numGPUsNeeded)
  if (output.numGPUsNeededMode === "state-sharded-lower-bound") {
    return `${count} (ideal ZeRO-3/FSDP state-sharded lower bound)`
  }

  if (output.numGPUsNeededMode === "data-parallel") {
    return `${count} data-parallel GPU${output.numGPUsNeeded === 1 ? "" : "s"}`
  }

  return count
}

function serializeCalculatorOutput(output: CalculatorOutput): string {
  return JSON.stringify(
    output,
    (_key, value: unknown) => {
      if (typeof value === "number" && !Number.isFinite(value)) {
        if (Number.isNaN(value)) return "NaN"
        return value > 0 ? "Infinity" : "-Infinity"
      }

      return value
    },
    2,
  )
}

function generatePretrainingMarkdown(o: PretrainingOutput): string {
  const hasActive = o.parameterCounts.active !== o.parameterCounts.total
  const implementationHasActive =
    o.implementationParameterCounts.active !==
    o.implementationParameterCounts.total
  const hasImplementationPadding =
    o.implementationParameterCounts.total !== o.parameterCounts.total ||
    o.implementationParameterCounts.active !== o.parameterCounts.active
  return [
    "# GPU Calculator — Pretraining Results\n",
    "## Model",
    `- Model Parameters: ${fmtCount(o.parameterCounts.total)}${hasActive ? ` total, ${fmtCount(o.parameterCounts.active)} active` : ""}`,
    hasImplementationPadding
      ? `- TP-Padded Implementation Parameters: ${fmtCount(o.implementationParameterCounts.total)}${implementationHasActive ? ` total, ${fmtCount(o.implementationParameterCounts.active)} active` : ""}`
      : null,
    "",
    "## Compute",
    `- Total FLOPs: ${fmtFLOPs(o.computeEstimate.totalFLOPs)}`,
    `- Chinchilla Ratio: ${fmtMultiplier(o.chinchilla.ratio)} (20x basis: ${fmtCount(o.chinchilla.parameterCount)} params)`,
    `- Predicted Loss: ${Number.isFinite(o.predictedLossNats) ? o.predictedLossNats.toFixed(3) : "--"} nats (${fmtCount(o.chinchilla.effectiveLossTokens)} effective tokens, ${o.chinchilla.coefficientRowLabel})`,
    `- Attention Overhead: ${fmtFractionPercent(o.attentionOverheadFraction)}`,
    "",
    "## Memory per GPU",
    `- Parameters: ${fmtBytes(o.memory.parameters)}`,
    `- Gradients: ${fmtBytes(o.memory.gradients)}`,
    `- Optimizer States: ${fmtBytes(o.memory.optimizerStates)}`,
    `- Activations: ${fmtBytes(o.memory.activations)}`,
    `- Buffers: ${fmtBytes(o.memory.communicationBuffers)}`,
    `- Total: ${fmtBytes(o.memory.total)} / ${fmtBytes(o.memory.usableCapacity)} usable`,
    `- Fits: ${o.memory.fits ? "Yes" : "No"}`,
    `- Minimum VRAM Floor: ${fmtBytes(o.minVRAMFloor)}`,
    "",
    "## Parallelism",
    `- Strategy: ${o.parallelismRecommendation.strategyLabel}`,
    `- Pipeline Bubble: ${fmtFractionPercent(o.pipelineBubbleFraction)}`,
    `- Effective GPUs: ${fmtCount(o.effectiveNumGPUs)}`,
    `- Minimum GPUs: ${fmtCount(o.minGPUsNeeded)}`,
    "",
    "## Batch",
    `- Global Batch: ${fmtCount(o.globalBatchSize.sequences)} sequences / ${fmtCount(o.globalBatchSize.tokens)} tokens`,
    `- Maximum Micro-Batch: ${fmtCount(o.maxMicroBatchSize)} sequences per GPU`,
    `- Critical Batch: ${fmtCount(o.batchEfficiency.criticalBatchTokens)} tokens (${fmtBatchRelation(o.batchEfficiency.relation)})`,
    `- Compute Multiplier Above Optimum: ${fmtMultiplier(o.batchEfficiency.computeMultiplier)}`,
    `- Actual Compute Above Optimum: ${fmtFractionPercent(o.batchEfficiency.wastedComputeFraction)} of actual run`,
    ...formatDataRepetitionMarkdown(o.dataRepetition),
    "",
    "## Training Time",
    `- Theoretical: ${fmtDuration(o.trainingTime.theoreticalHours)}`,
    o.trainingTime.failureAdjustedHours != null
      ? `- Failure-Adjusted: ${fmtDuration(o.trainingTime.failureAdjustedHours)}`
      : null,
    `- Throughput: ${fmtCount(o.tokensPerSecond)} tok/s`,
    `- Steps: ${fmtCount(o.trainingTime.totalSteps)}`,
    "",
    "## Cost",
    `- Compute: ${fmtCurrency(o.cost.computeCost)}`,
    o.cost.actualComputeCost != null &&
    o.cost.actualComputeCost !== o.cost.computeCost
      ? `- Actual Compute: ${fmtCurrency(o.cost.actualComputeCost)}`
      : null,
    `- Storage: ${fmtCurrency(o.cost.storageCost)}`,
    `- Failure Overhead: ${fmtCurrency(o.cost.failureOverheadCost)}`,
    `- Checkpoints: ${fmtCount(o.cost.numCheckpoints)} saves, ${fmtBytes(o.cost.averageCheckpointStorage)} average retained, ${fmtBytes(o.cost.peakCheckpointStorage)} peak retained`,
    o.cost.datasetStorageBytes > 0
      ? `- Dataset Storage: ${fmtBytes(o.cost.datasetStorageBytes)}`
      : null,
    `- Total: ${fmtCurrency(o.cost.totalCost)}`,
    ...formatWarningsMarkdown(o.warnings),
    "",
    "---",
    "Generated by LLM Training GPU Calculator",
  ]
    .filter((line) => line !== null)
    .join("\n")
}

function generatePostTrainingMarkdown(o: PostTrainingOutput): string {
  return [
    "# GPU Calculator — Post-Training Results\n",
    "## Memory per GPU",
    `- Peak Working Set: ${fmtBytes(o.memory.total)} / ${fmtBytes(o.memory.usableCapacity)} usable (allocator-adjusted)`,
    `- Fits: ${o.memory.fits ? "Yes" : "No"}`,
    `- Free Headroom: ${fmtBytes(o.memory.freeHeadroom)}`,
    `- Parameters: ${fmtBytes(o.memory.parameters)}`,
    `- Gradients: ${fmtBytes(o.memory.gradients)}`,
    `- Optimizer States: ${fmtBytes(o.memory.optimizerStates)}`,
    `- Activations: ${fmtBytes(o.memory.activations)}`,
    `- Buffers: ${fmtBytes(o.memory.communicationBuffers)}`,
    `- GPUs Needed: ${formatPostTrainingGPURequirementMarkdown(o)}`,
    ...formatPostTrainingMemoryItemsMarkdown(o.memory.items),
    "",
    "## Training Time",
    `- Estimated: ${fmtDuration(o.trainingTime.theoreticalHours)}`,
    `- Throughput: ${fmtCount(o.trainingTime.tokensPerSecond)} tok/s`,
    `- ${o.stepMarkdownLabel}: ${fmtCount(o.trainingTime.totalSteps)}`,
    `- ${o.stepTimeLabel}: ${
      Number.isFinite(o.trainingTime.secondsPerStep)
        ? `${o.trainingTime.secondsPerStep.toFixed(2)} s`
        : "--"
    }`,
    "",
    "## Cost",
    `- Compute: ${fmtCurrency(o.cost.computeCost)}`,
    `- Total: ${fmtCurrency(o.cost.totalCost)}`,
    ...formatWarningsMarkdown(o.warnings),
    "",
    "---",
    "Generated by LLM Training GPU Calculator",
  ].join("\n")
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

const tabs: { key: CalculatorTab; label: string; description: string }[] = [
  {
    key: "pretraining",
    label: "Pretraining",
    description:
      "Configure model, data, hardware, and parallelism for pretraining runs.",
  },
  {
    key: "post-training",
    label: "Post-Training",
    description:
      "SFT, DPO, PPO, and GRPO — configure fine-tuning method, approach, and resources.",
  },
]

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function GpuCalculator() {
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  )
  const { resolvedTheme } = useTheme()
  const isDark = mounted && resolvedTheme === "dark"

  const [activeTab, setActiveTab] = useState<CalculatorTab>("pretraining")
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>(
    DEFAULT_TRAINING_CONFIG,
  )
  const [postTrainingConfig, setPostTrainingConfig] =
    useState<PostTrainingConfig>(DEFAULT_POST_TRAINING_CONFIG)
  const [copied, setCopied] = useState<"text" | "json" | null>(null)

  const colors = useMemo(
    () => ({
      bg: isDark
        ? "oklch(0.14 0.005 260)"
        : "oklch(0.993 0.003 80)",
      cardBg: isDark
        ? "oklch(0.185 0.008 260)"
        : "oklch(0.99 0.002 80)",
      text: isDark
        ? "oklch(0.93 0.004 80)"
        : "oklch(0.155 0.004 260)",
      textSecondary: isDark
        ? "oklch(0.60 0.010 260)"
        : "oklch(0.50 0.010 260)",
      border: isDark
        ? "oklch(0.28 0.010 260)"
        : "oklch(0.915 0.006 80)",
      accent: isDark
        ? "oklch(0.72 0.12 180)"
        : "oklch(0.52 0.135 180)",
      accentMuted: isDark
        ? "oklch(0.22 0.035 180)"
        : "oklch(0.96 0.022 180)",
      panel: isDark
        ? "oklch(0.16 0.006 260 / 0.85)"
        : "oklch(0.985 0.003 80 / 0.92)",
      warning: isDark
        ? "oklch(0.80 0.12 80)"
        : "oklch(0.56 0.14 80)",
      warningBg: isDark
        ? "oklch(0.22 0.04 80)"
        : "oklch(0.97 0.025 80)",
      warningBorder: isDark
        ? "oklch(0.35 0.06 80)"
        : "oklch(0.90 0.06 80)",
    }),
    [isDark],
  )

  const stats = [
    { label: "GPU Presets", value: GPU_SPECS.length, icon: Server },
    { label: "Model Presets", value: MODEL_PRESETS.length, icon: Layers3 },
    {
      label: "Optimizer Profiles",
      value: OPTIMIZER_PROFILES.length,
      icon: Boxes,
    },
  ]

  // ═══════════════════════════════════════════════════════════════════════════
  // PRETRAINING CALCULATION PIPELINE
  // ═══════════════════════════════════════════════════════════════════════════

  const resolvedTrainingModel = useMemo(
    () => resolvePretrainingModel(trainingConfig),
    [trainingConfig],
  )

  const computeEstimate = useMemo(
    () =>
      calculateFLOPs(
        resolvedTrainingModel.parameterCounts,
        {
          totalTokens: trainingConfig.totalTokens,
          sequenceLength: trainingConfig.sequenceLength,
        },
        resolvedTrainingModel.architecture,
        resolvedTrainingModel.moe,
      ),
    [
      resolvedTrainingModel,
      trainingConfig.totalTokens,
      trainingConfig.sequenceLength,
    ],
  )

  const scalingLawParameterCount = resolvedTrainingModel.moe.enabled
    ? resolvedTrainingModel.parameterCounts.active
    : resolvedTrainingModel.parameterCounts.total

  const numGPUs = useMemo(
    () =>
      resolveRequestedNumGPUs(
        trainingConfig,
        computeEstimate.totalFLOPs,
        resolvedTrainingModel.parameterCounts,
        resolvedTrainingModel.architecture,
        resolvedTrainingModel.moe,
      ),
    [
      trainingConfig,
      computeEstimate.totalFLOPs,
      resolvedTrainingModel.parameterCounts,
      resolvedTrainingModel.architecture,
      resolvedTrainingModel.moe,
    ],
  )

  const gpuCountDerivedFromTarget =
    trainingConfig.parallelismMode === "auto" &&
    trainingConfig.hardware.targetTrainingDays !== null &&
    Number.isFinite(trainingConfig.hardware.targetTrainingDays) &&
    trainingConfig.hardware.targetTrainingDays > 0

  const normalizedTrainingParallelism = useMemo(
    () =>
      normalizeParallelismConfig(
        trainingConfig.parallelism,
        resolvedTrainingModel.moe.enabled,
      ),
    [trainingConfig.parallelism, resolvedTrainingModel.moe.enabled],
  )

  const resolvedTrainingConfig = useMemo(
    (): TrainingConfig => ({
      ...trainingConfig,
      model: {
        ...trainingConfig.model,
        architecture: resolvedTrainingModel.architecture,
        moe: resolvedTrainingModel.moe,
      },
      optimizer: trainingConfig.optimizer,
      parallelism: trainingConfig.hardware.gpu.singleDeviceOnly
        ? forceSingleDeviceParallelism(normalizedTrainingParallelism)
        : normalizedTrainingParallelism,
      hardware: {
        ...trainingConfig.hardware,
        numGPUs,
      },
    }),
    [trainingConfig, resolvedTrainingModel, normalizedTrainingParallelism, numGPUs],
  )

  const chinchillaAnalysis = useMemo(
    () =>
      calculateChinchillaAnalysis(
        scalingLawParameterCount,
        trainingConfig.totalTokens,
        trainingConfig.uniqueTokens,
      ),
    [
      scalingLawParameterCount,
      trainingConfig.totalTokens,
      trainingConfig.uniqueTokens,
    ],
  )

  const parallelismRecommendation = useMemo((): ParallelismRecommendation => {
    const arch = resolvedTrainingModel.architecture
    const moe = resolvedTrainingModel.moe
    const gpu = resolvedTrainingConfig.hardware.gpu

    if (hasInvalidExplicitTrainingGPUCount(resolvedTrainingConfig)) {
      return {
        config: resolvedTrainingConfig.parallelism,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid GPU count",
        reasoning: ["GPU count must be a positive integer."],
        warnings: [],
      }
    }

    if (hasInvalidManualWorldSize(resolvedTrainingConfig)) {
      return {
        config: resolvedTrainingConfig.parallelism,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid manual world size",
        reasoning: [
          "Manual parallelism world size must match the configured GPU count.",
        ],
        warnings: [],
      }
    }

    if (hasInvalidPretrainingModelInputMode(resolvedTrainingConfig)) {
      return {
        config: resolvedTrainingConfig.parallelism,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid model input mode",
        reasoning: ["Model input mode must be preset, quick, or detailed."],
        warnings: [],
      }
    }

    if (
      hasInvalidTrainingHardware(
        resolvedTrainingConfig.hardware.inputMode,
        gpu,
        resolvedTrainingConfig.precision,
      )
    ) {
      return {
        config: resolvedTrainingConfig.parallelism,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid custom GPU",
        reasoning: ["Custom GPU hardware fields are invalid."],
        warnings: [],
      }
    }

    const p = resolvedTrainingConfig.parallelism

    if (hasInvalidParallelismFramework(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid parallelism framework",
        reasoning: [
          "Parallelism framework must be megatron, deepspeed, fsdp, or hf_trainer.",
        ],
        warnings: [],
      }
    }

    if (hasInvalidParallelismMode(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid parallelism mode",
        reasoning: ["Parallelism mode must be auto or manual."],
        warnings: [],
      }
    }

    if (hasInvalidSequenceParallelismMode(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid sequence parallelism mode",
        reasoning: [
          "Sequence parallelism mode must be auto, enabled, or disabled.",
        ],
        warnings: [],
      }
    }

    if (hasInvalidFlashAttentionFlag(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid Flash Attention flag",
        reasoning: ["Flash Attention must be true or false."],
        warnings: [],
      }
    }

    if (hasInvalidAMPAutocastFlag(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid AMP autocast flag",
        reasoning: ["AMP autocast must be true or false."],
        warnings: [],
      }
    }

    if (hasInvalidChunkedCrossEntropyFlag(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid chunked cross-entropy flag",
        reasoning: ["Chunked cross-entropy must be true or false."],
        warnings: [],
      }
    }

    if (hasInvalidTorchCompileFlag(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid torch.compile flag",
        reasoning: ["torch.compile must be true or false."],
        warnings: [],
      }
    }

    if (hasInvalidZeROCommunicationConfig(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid ZeRO communication",
        reasoning: [
          "ZeRO communication buckets and overlap settings must be valid.",
        ],
        warnings: [],
      }
    }

    if (resolvedTrainingConfig.parallelismMode === "auto") {
      return recommendParallelism(
        resolvedTrainingModel.parameterCounts,
        arch,
        resolvedTrainingConfig,
        gpu,
        numGPUs,
        moe,
      )
    }

    // Manual mode — wrap user config in a ParallelismRecommendation
    if (hasInvalidManualParallelismDegrees(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid manual parallelism",
        reasoning: ["Manual parallelism configuration contains invalid degrees."],
        warnings: [],
      }
    }

    if (hasInvalidManualShardingMode(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid sharding mode",
        reasoning: ["Manual ZeRO/FSDP sharding mode is invalid."],
        warnings: [],
      }
    }

    if (hasInvalidManualTensorParallelismTopology(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid tensor parallelism",
        reasoning: [
          "Manual tensor parallelism requires N_tp to divide hidden size, attention heads, KV heads, and the TP-sharded FFN width.",
        ],
        warnings: [],
      }
    }

    if (
      hasInvalidManualTensorExpertSequenceParallelismTopology(
        resolvedTrainingConfig,
      )
    ) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid TP/EP sequence parallelism",
        reasoning: [
          "Manual TP+EP MoE layouts require sequence parallelism enabled or auto.",
        ],
        warnings: [],
      }
    }

    if (hasInvalidManualPipelineTopology(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid pipeline topology",
        reasoning: ["Manual pipeline parallelism configuration is invalid."],
        warnings: [],
      }
    }

    if (hasInvalidManualContextParallelismTopology(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid context parallelism",
        reasoning: [
          "Manual context parallelism requires sequence length to divide evenly across CP ranks.",
        ],
        warnings: [],
      }
    }

    if (hasInvalidManualExpertParallelismTopology(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid expert parallelism",
        reasoning: [
          "Manual expert parallelism topology is invalid for the selected MoE model.",
        ],
        warnings: [],
      }
    }

    if (hasInvalidCPUOffloadConfig(resolvedTrainingConfig)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid CPU offload placement",
        reasoning: ["Manual CPU offload placement is invalid."],
        warnings: [],
      }
    }

    if (hasInvalidPretrainingOptimizer(resolvedTrainingConfig.optimizer)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid pretraining optimizer",
        reasoning: ["Selected optimizer is not valid for pretraining."],
        warnings: [],
      }
    }

    if (hasInvalidGradientPrecision(resolvedTrainingConfig.gradientPrecision)) {
      return {
        config: p,
        minGPUs: Number.POSITIVE_INFINITY,
        minVRAMFloor: Number.POSITIVE_INFINITY,
        pipelineBubbleFraction: Number.POSITIVE_INFINITY,
        strategyLabel: "Invalid gradient precision",
        reasoning: ["Gradient precision must be fp32 or bf16."],
        warnings: [],
      }
    }

    const bubble = calculatePipelineBubble(
      p.N_pp,
      resolvedTrainingConfig.gradientAccumulationSteps,
      getEffectivePipelineBubbleVP(
        p,
        resolvedTrainingConfig.gradientAccumulationSteps,
      ),
    )
    const moeEnabled = moe.enabled && moe.E > 0
    const parts: string[] = [`DP=${p.N_dp}`]
    if (p.N_tp > 1) parts.push(`TP=${p.N_tp}`)
    if (moeEnabled && p.N_ep > 1) parts.push(`EP=${p.N_ep}`)
    if (p.N_pp > 1) parts.push(`PP=${p.N_pp}`)
    if (p.N_cp > 1) parts.push(`CP=${p.N_cp}`)
    parts.push(
      p.framework === "fsdp" && p.fsdpStrategy !== null
        ? `FSDP ${p.fsdpStrategy}`
        : `ZeRO-${p.zeroStage}`,
    )

    const manualWorldSize = resolveParallelWorldSize(p)
    const configForFloor: TrainingConfig = {
      ...resolvedTrainingConfig,
      parallelism: p,
      hardware: {
        ...resolvedTrainingConfig.hardware,
        numGPUs: manualWorldSize,
      },
    }
    const minVRAMFloor = calculateMinGPUVRAMFloor(
      applyVocabPaddingToCounts(
        resolvedTrainingModel.parameterCounts,
        resolvedTrainingModel.architecture,
        p.N_tp,
      ),
      configForFloor,
    )

    return {
      config: p,
      minGPUs: manualWorldSize,
      minVRAMFloor,
      pipelineBubbleFraction: bubble,
      strategyLabel: parts.join(", "),
      reasoning: ["Manual parallelism configuration."],
      warnings: [],
    }
  }, [resolvedTrainingConfig, resolvedTrainingModel, numGPUs])

  const effectiveConfig = useMemo((): TrainingConfig => {
    const hasInvalidManualParallelism =
      hasInvalidTrainingGPUCount(resolvedTrainingConfig) ||
      hasInvalidPretrainingModelInputMode(resolvedTrainingConfig) ||
      hasInvalidParallelismFramework(resolvedTrainingConfig) ||
      hasInvalidParallelismMode(resolvedTrainingConfig) ||
      hasInvalidSequenceParallelismMode(resolvedTrainingConfig) ||
      hasInvalidAMPAutocastFlag(resolvedTrainingConfig) ||
      hasInvalidChunkedCrossEntropyFlag(resolvedTrainingConfig) ||
      hasInvalidFlashAttentionFlag(resolvedTrainingConfig) ||
      hasInvalidTorchCompileFlag(resolvedTrainingConfig) ||
      hasInvalidManualParallelismDegrees(resolvedTrainingConfig) ||
      hasInvalidManualShardingMode(resolvedTrainingConfig)
    const parallelWorldSize = hasInvalidManualParallelism
      ? Number.POSITIVE_INFINITY
      : resolveParallelWorldSize(parallelismRecommendation.config)

    return {
      ...resolvedTrainingConfig,
      parallelism: parallelismRecommendation.config,
      hardware: {
        ...resolvedTrainingConfig.hardware,
        numGPUs:
          resolvedTrainingConfig.parallelismMode === "auto"
            ? Math.max(numGPUs, parallelWorldSize)
            : parallelWorldSize,
      },
    }
  }, [resolvedTrainingConfig, parallelismRecommendation.config, numGPUs])
  const effectiveTrainingNumGPUs = hasInvalidTrainingGPUCount(effectiveConfig)
    ? Number.POSITIVE_INFINITY
    : resolveExplicitNumGPUs(effectiveConfig.hardware.numGPUs)

  const paddedParameterCounts = useMemo(
    () =>
      hasInvalidManualParallelismDegrees(effectiveConfig)
        ? infiniteParameterCounts(resolvedTrainingModel.parameterCounts)
        : applyVocabPaddingToCounts(
            resolvedTrainingModel.parameterCounts,
            resolvedTrainingModel.architecture,
            parallelismRecommendation.config.N_tp,
          ),
    [resolvedTrainingModel, parallelismRecommendation.config.N_tp, effectiveConfig],
  )

  const effectiveComputeEstimate = useMemo(
    () =>
      calculateFLOPs(
        paddedParameterCounts,
        {
          totalTokens: trainingConfig.totalTokens,
          sequenceLength: trainingConfig.sequenceLength,
        },
        resolvedTrainingModel.architecture,
        resolvedTrainingModel.moe,
      ),
    [
      paddedParameterCounts,
      trainingConfig.totalTokens,
      trainingConfig.sequenceLength,
      resolvedTrainingModel.architecture,
      resolvedTrainingModel.moe,
    ],
  )

  const activationSchedule = useMemo(
    () =>
      resolveActivationSchedule(
        effectiveConfig.parallelism,
        effectiveConfig.gradientAccumulationSteps,
      ),
    [effectiveConfig.parallelism, effectiveConfig.gradientAccumulationSteps],
  )

  const memoryBreakdown = useMemo(() => {
    return calculateTotalMemoryPerGPU(
      paddedParameterCounts,
      effectiveConfig,
      resolvedTrainingModel.architecture,
      resolvedTrainingModel.moe,
      effectiveConfig.hardware.gpu,
      activationSchedule,
    )
  }, [
    paddedParameterCounts,
    effectiveConfig,
    resolvedTrainingModel.architecture,
    resolvedTrainingModel.moe,
    activationSchedule,
  ])

  const trainingTime = useMemo(
    () =>
      calculateTrainingTime(
        effectiveComputeEstimate,
        effectiveConfig,
        paddedParameterCounts.active,
      ),
    [effectiveComputeEstimate, effectiveConfig, paddedParameterCounts.active],
  )

  const costEstimate = useMemo(
    () =>
      calculateCost(
        trainingTime,
        effectiveConfig,
        paddedParameterCounts.total,
      ),
    [trainingTime, effectiveConfig, paddedParameterCounts.total],
  )

  const dataRepetition = useMemo(
    () =>
      analyzeDataRepetition(
        trainingConfig.totalTokens,
        trainingConfig.uniqueTokens,
      ),
    [trainingConfig.totalTokens, trainingConfig.uniqueTokens],
  )

  const globalBatchSize = useMemo(() => {
    const hasInvalidBatchShape =
      hasInvalidTrainingGPUCount(effectiveConfig) ||
      hasInvalidPretrainingModelInputMode(effectiveConfig) ||
      hasInvalidTrainingHardware(
        effectiveConfig.hardware.inputMode,
        effectiveConfig.hardware.gpu,
        effectiveConfig.precision,
      ) ||
      hasInvalidAMPAutocastFlag(effectiveConfig) ||
      hasInvalidChunkedCrossEntropyFlag(effectiveConfig) ||
      hasInvalidFlashAttentionFlag(effectiveConfig) ||
      hasInvalidTorchCompileFlag(effectiveConfig) ||
      hasInvalidParallelismFramework(effectiveConfig) ||
      hasInvalidParallelismMode(effectiveConfig) ||
      hasInvalidSequenceParallelismMode(effectiveConfig) ||
      hasInvalidManualParallelismDegrees(effectiveConfig) ||
      hasInvalidManualTensorParallelismTopology(effectiveConfig) ||
      hasInvalidManualTensorExpertSequenceParallelismTopology(effectiveConfig) ||
      hasInvalidManualContextParallelismTopology(effectiveConfig) ||
      hasInvalidManualExpertParallelismTopology(effectiveConfig) ||
      hasInvalidManualShardingMode(effectiveConfig) ||
      hasInvalidManualPipelineTopology(effectiveConfig) ||
      hasInvalidCPUOffloadConfig(effectiveConfig) ||
      hasInvalidGradientPrecision(effectiveConfig.gradientPrecision) ||
      hasInvalidPretrainingOptimizer(effectiveConfig.optimizer) ||
      !Number.isFinite(trainingConfig.microBatchSize) ||
      trainingConfig.microBatchSize <= 0 ||
      !Number.isInteger(trainingConfig.microBatchSize) ||
      !Number.isFinite(trainingConfig.gradientAccumulationSteps) ||
      trainingConfig.gradientAccumulationSteps <= 0 ||
      !Number.isInteger(trainingConfig.gradientAccumulationSteps) ||
      !Number.isFinite(trainingConfig.sequenceLength) ||
      trainingConfig.sequenceLength <= 0 ||
      !Number.isInteger(trainingConfig.sequenceLength)

    if (hasInvalidBatchShape) {
      return {
        sequences: Number.POSITIVE_INFINITY,
        tokens: Number.POSITIVE_INFINITY,
      }
    }

    const N_dp = normalizeParallelismDegree(parallelismRecommendation.config.N_dp)
    const microBatchSize = normalizeParallelismDegree(
      trainingConfig.microBatchSize,
    )
    const gradientAccumulationSteps = normalizeParallelismDegree(
      trainingConfig.gradientAccumulationSteps,
    )
    const sequences =
      microBatchSize * gradientAccumulationSteps * N_dp
    return { sequences, tokens: sequences * trainingConfig.sequenceLength }
  }, [
    trainingConfig.microBatchSize,
    trainingConfig.gradientAccumulationSteps,
    trainingConfig.sequenceLength,
    parallelismRecommendation.config.N_dp,
    effectiveConfig,
  ])

  const batchEfficiency = useMemo(
    () =>
      calculateCriticalBatchSize(
        chinchillaAnalysis.predictedLossNats,
        globalBatchSize.tokens,
      ),
    [chinchillaAnalysis.predictedLossNats, globalBatchSize.tokens],
  )

  const maxMicroBatchSize = useMemo(
    () =>
      estimateMaxMicroBatch(
        paddedParameterCounts,
        effectiveConfig,
        resolvedTrainingModel.architecture,
        resolvedTrainingModel.moe,
        effectiveConfig.hardware.gpu,
        activationSchedule,
        parallelismRecommendation.minVRAMFloor,
      ),
    [
      paddedParameterCounts,
      effectiveConfig,
      resolvedTrainingModel.architecture,
      resolvedTrainingModel.moe,
      activationSchedule,
      parallelismRecommendation.minVRAMFloor,
    ],
  )

  const moeSparsity = useMemo((): MoESparsityMetrics | null => {
    const { total, active } = resolvedTrainingModel.parameterCounts
    if (
      !resolvedTrainingModel.moe.enabled ||
      !Number.isFinite(total) ||
      !Number.isFinite(active) ||
      total <= 0 ||
      active <= 0
    )
      return null
    const activeFraction = active / total
    return {
      sparsityRatio: Math.max(0, Math.min(1, 1 - activeFraction)),
      efficiencyGain: total / active,
      loadBalanceFactor: resolvedTrainingModel.moe.loadBalanceFactor,
    }
  }, [resolvedTrainingModel])

  const pretrainingWarnings = useMemo((): Warning[] => {
    const inputW = generateInputWarnings(
      resolvedTrainingConfig,
      resolvedTrainingModel.architecture,
      resolvedTrainingModel.moe,
      resolvedTrainingModel.parameterCounts.total,
      resolvedTrainingModel.parameterCounts,
      parallelismRecommendation.config,
      numGPUs,
      chinchillaAnalysis.ratio,
      chinchillaAnalysis.powerLawOptimalTokens,
      chinchillaAnalysis.effectiveLossTokens,
      trainingConfig,
    )
    if (
      gpuCountDerivedFromTarget &&
      trainingConfig.hardware.targetTrainingDays !== null &&
      Number.isFinite(trainingTime.theoreticalDays)
    ) {
      const targetDays = trainingConfig.hardware.targetTrainingDays
      const targetTooLoose =
        trainingTime.theoreticalDays < targetDays * 0.98
      const gpuMessage =
        effectiveTrainingNumGPUs === numGPUs
          ? targetTooLoose
            ? `Minimum feasible auto layout uses ${effectiveTrainingNumGPUs.toLocaleString()} GPUs and estimates ${trainingTime.theoreticalDays.toFixed(2)} days, faster than the ${targetDays.toFixed(1)}-day target.`
            : `Using ${effectiveTrainingNumGPUs.toLocaleString()} GPUs to target roughly ${targetDays.toFixed(1)} training days.`
          : `Target-time estimate starts at ${numGPUs.toLocaleString()} GPUs, but the effective auto layout uses ${effectiveTrainingNumGPUs.toLocaleString()} GPUs after memory and topology constraints.`

      inputW.unshift({
        severity: "info",
        category: "hardware",
        message: gpuMessage,
      })
    }
    const memW: Warning[] = []
    if (!memoryBreakdown.fits) {
      const totalExceedsUsable =
        memoryBreakdown.total > memoryBreakdown.usableCapacity
      const floorExceedsUsable =
        parallelismRecommendation.minVRAMFloor > memoryBreakdown.usableCapacity

      memW.push({
        severity: "critical",
        category: "memory",
        message: totalExceedsUsable
          ? `Per-GPU memory (${(memoryBreakdown.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memoryBreakdown.usableCapacity / 1e9).toFixed(1)} GB).`
          : floorExceedsUsable
            ? `The largest parameter-unit working set (${(parallelismRecommendation.minVRAMFloor / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memoryBreakdown.usableCapacity / 1e9).toFixed(1)} GB), even though the summed steady-state estimate is ${(memoryBreakdown.total / 1e9).toFixed(1)} GB.`
            : `Per-GPU memory does not fit within usable capacity (${(memoryBreakdown.usableCapacity / 1e9).toFixed(1)} GB).`,
      })
    }
    if (!effectiveComputeEstimate.simplifiedFormulaAccurate) {
      inputW.push({
        severity: "warning",
        category: "compute",
        message: `The sequence-dependent attention term is material here (${(effectiveComputeEstimate.attentionOverheadFraction * 100).toFixed(1)}% over model FLOPs), so the calculator is using the full PaLM FLOPs formula instead of 6ΨD.`,
      })
    }
    if (trainingTime.failureMultiplier !== null) {
      if (!Number.isFinite(trainingTime.failureMultiplier)) {
        inputW.push({
          severity: "critical",
          category: "cost",
          message:
            "Failure-adjusted training time diverges for the current failure rate, recovery time, checkpoint cadence, and cluster size.",
        })
      } else if (trainingTime.failureMultiplier > 2) {
        inputW.push({
          severity: "warning",
          category: "cost",
          message: `Failure overhead more than doubles training time (${trainingTime.failureMultiplier.toFixed(1)}x). Increase checkpoint frequency, reduce recovery time, or lower the cluster size.`,
        })
      }
    }
    const checkpointFrequency =
      effectiveConfig.failureModel.checkpointFrequencyPerDay
    if (
      Number.isFinite(checkpointFrequency) &&
      checkpointFrequency > 0 &&
      Number.isFinite(trainingTime.secondsPerStep) &&
      trainingTime.secondsPerStep > 0 &&
      trainingTime.totalSteps > 0
    ) {
      const checkpointIntervalSeconds = 86400 / checkpointFrequency

      if (checkpointIntervalSeconds < trainingTime.secondsPerStep) {
        inputW.push({
          severity: "warning",
          category: "cost",
          message: `Checkpoint frequency (${checkpointFrequency.toLocaleString()}/day) is faster than the optimizer-step cadence (${fmtDuration(trainingTime.secondsPerStep / 3600)} per step). Failure and storage estimates cap recoverable checkpoints at one per completed optimizer step.`,
        })
      }
    }
    if (
      Number.isFinite(trainingConfig.totalTokens) &&
      trainingConfig.totalTokens > 0 &&
      Number.isInteger(trainingConfig.totalTokens) &&
      Number.isFinite(globalBatchSize.tokens) &&
      globalBatchSize.tokens > 0 &&
      trainingTime.totalSteps > 0
    ) {
      const fullBatchSteps = Math.floor(
        trainingConfig.totalTokens / globalBatchSize.tokens,
      )
      const finalBatchTokens =
        trainingConfig.totalTokens - fullBatchSteps * globalBatchSize.tokens
      const hasPartialFinalBatch =
        finalBatchTokens > globalBatchSize.tokens * 1e-9 &&
        finalBatchTokens < globalBatchSize.tokens * (1 - 1e-9)

      if (trainingTime.totalSteps === 1 && hasPartialFinalBatch) {
        inputW.push({
          severity: "info",
          category: "data",
          message: `Total tokens (${fmtCount(trainingConfig.totalTokens)}) are below the configured global batch (${fmtCount(globalBatchSize.tokens)} tokens). The run is modeled as one partial optimizer step; frameworks that pad, drop, or reuse examples can see a different effective batch.`,
        })
      } else if (hasPartialFinalBatch && trainingTime.totalSteps <= 100) {
        inputW.push({
          severity: "info",
          category: "data",
          message: `Total tokens are not an integer multiple of the configured global batch. Step count rounds up to ${trainingTime.totalSteps.toLocaleString()} with a final partial step (${fmtCount(finalBatchTokens)} of ${fmtCount(globalBatchSize.tokens)} tokens); align tokens or batch size if exact step cadence matters.`,
        })
      }
    }
    if (
      gpuCountDerivedFromTarget &&
      trainingConfig.hardware.targetTrainingDays !== null &&
      Number.isFinite(trainingConfig.hardware.targetTrainingDays) &&
      trainingConfig.hardware.targetTrainingDays > 0 &&
      Number.isFinite(trainingTime.theoreticalDays) &&
      trainingTime.theoreticalDays >
        trainingConfig.hardware.targetTrainingDays * 1.02
    ) {
      const targetDays = trainingConfig.hardware.targetTrainingDays
      const scheduleNote =
        effectiveConfig.parallelism.N_pp > 1
          ? ` The selected PP=${effectiveConfig.parallelism.N_pp} layout has a ${(parallelismRecommendation.pipelineBubbleFraction * 100).toFixed(1)}% pipeline bubble.`
          : ""

      inputW.push({
        severity: "warning",
        category: "hardware",
        message: `Target training time is ${targetDays.toFixed(2)} days, but the selected layout estimates ${trainingTime.theoreticalDays.toFixed(2)} days after schedule efficiency and memory-driven topology are applied.${scheduleNote}`,
      })
    }
    if (
      trainingConfig.parallelismMode === "manual" &&
      !resolvedTrainingModel.moe.enabled &&
      trainingConfig.parallelism.N_ep > 1
    ) {
      inputW.push({
        severity: "info",
        category: "parallelism",
        message:
          "Expert parallelism is ignored because the resolved model is dense; effective N_ep is 1 for memory, time, and cost estimates.",
      })
    }
    if (
      Number.isFinite(effectiveTrainingNumGPUs) &&
      effectiveTrainingNumGPUs >= 16000
    ) {
      inputW.push({
        severity: "warning",
        category: "hardware",
        message:
          "At 16K+ GPUs, the default 1% instance-day failure-rate assumption may understate real interruption rates; calibrate the failure model from cluster logs.",
      })
    }
    return [...memW, ...inputW, ...parallelismRecommendation.warnings]
  }, [
    resolvedTrainingConfig,
    resolvedTrainingModel,
    parallelismRecommendation,
    effectiveTrainingNumGPUs,
    numGPUs,
    chinchillaAnalysis.ratio,
    chinchillaAnalysis.powerLawOptimalTokens,
    chinchillaAnalysis.effectiveLossTokens,
    memoryBreakdown,
    effectiveComputeEstimate,
    trainingTime.failureMultiplier,
    trainingTime.theoreticalDays,
    trainingTime.secondsPerStep,
    trainingTime.totalSteps,
    globalBatchSize.tokens,
    gpuCountDerivedFromTarget,
    trainingConfig,
    effectiveConfig.failureModel.checkpointFrequencyPerDay,
    effectiveConfig.parallelism.N_pp,
  ])

  const pretrainingOutput = useMemo(
    (): PretrainingOutput => ({
      parameterCounts: resolvedTrainingModel.parameterCounts,
      implementationParameterCounts: paddedParameterCounts,
      computeEstimate: effectiveComputeEstimate,
      chinchilla: chinchillaAnalysis,
      memory: memoryBreakdown,
      effectiveNumGPUs: effectiveTrainingNumGPUs,
      minGPUsNeeded: parallelismRecommendation.minGPUs,
      minVRAMFloor: parallelismRecommendation.minVRAMFloor,
      parallelismRecommendation,
      pipelineBubbleFraction:
        parallelismRecommendation.pipelineBubbleFraction,
      trainingTime,
      tokensPerSecond: trainingTime.tokensPerSecond,
      cost: costEstimate,
      globalBatchSize,
      checkpointSize: costEstimate.checkpointSize,
      attentionOverheadFraction:
        effectiveComputeEstimate.attentionOverheadFraction,
      predictedLossNats: chinchillaAnalysis.predictedLossNats,
      maxMicroBatchSize,
      dataRepetition,
      moeSparsity,
      batchEfficiency,
      warnings: pretrainingWarnings,
    }),
    [
      effectiveComputeEstimate,
      chinchillaAnalysis,
      memoryBreakdown,
      effectiveTrainingNumGPUs,
      parallelismRecommendation,
      trainingTime,
      costEstimate,
      globalBatchSize,
      maxMicroBatchSize,
      dataRepetition,
      moeSparsity,
      batchEfficiency,
      pretrainingWarnings,
      paddedParameterCounts,
      resolvedTrainingModel.parameterCounts,
    ],
  )

  // ═══════════════════════════════════════════════════════════════════════════
  // POST-TRAINING CALCULATION PIPELINE
  // ═══════════════════════════════════════════════════════════════════════════

  const resolvedPostTrainingConfig = useMemo(
    () => resolvePostTrainingConfig(postTrainingConfig),
    [postTrainingConfig],
  )

  const postTrainingOutput = useMemo((): PostTrainingOutput => {
    const cfg = resolvedPostTrainingConfig
    const gpu = cfg.hardware.gpu
    const hasInvalidGPUCount = hasInvalidPostTrainingGPUCount(cfg)
    const hasInvalidHardware =
      hasInvalidTrainingHardware(
        cfg.hardware.inputMode,
        gpu,
        cfg.precision,
      )
    const hasInvalidSemanticConfig =
      hasInvalidPostTrainingOptimizer(cfg.optimizer) ||
      hasInvalidGradientPrecision(cfg.gradientPrecision) ||
      hasInvalidPostTrainingModelShape(cfg) ||
      hasInvalidFP8Config(cfg) ||
      hasInvalidPostTrainingKVCachePrecision(cfg) ||
      hasInvalidChunkedCrossEntropyFlag(cfg) ||
      hasInvalidPostTrainingApproachConfig(cfg) ||
      hasInvalidPostTrainingMethodConfig(cfg) ||
      hasInvalidQLoRAQuantizationBits(cfg) ||
      hasInvalidLoRARank(cfg) ||
      hasInvalidLoRAAlpha(cfg) ||
      hasInvalidPostTrainingTrainablePercentage(cfg) ||
      ((cfg.approach === "lora" || cfg.approach === "qlora") &&
        hasInvalidLoRATargetModules(cfg.lora))
    const ptGPUs = hasInvalidGPUCount
      ? Number.POSITIVE_INFINITY
      : resolveExplicitNumGPUs(cfg.hardware.numGPUs)

    const memory = getPostTrainingMemory(cfg)

    // Compute + time. MoE bases do matmuls only on the active (routed + shared)
    // experts per token, so compute uses Ψ_active; memory still uses Ψ_total.
    const computeParams = resolvePostTrainingComputeParameterCount(cfg)
    const compute = calculatePostTrainingCompute(
      cfg.method,
      computeParams,
      cfg,
    )
    const generationFeasibility = estimateMaxConcurrentGenerations(cfg, memory)
    const effectiveComputeGPUs = estimatePostTrainingMaxEffectiveComputeGPUs(
      cfg,
      ptGPUs,
    )
    const fPeakFLOPS =
      getEffectiveTrainingTFLOPS(
        gpu,
        cfg.precision,
        cfg.fp8,
      ) * 1e12
    const generationTokens =
      cfg.method === "ppo" || cfg.method === "grpo" ? compute.totalTokens : 0
    const generationFLOPs =
      Number.isFinite(computeParams) && Number.isFinite(generationTokens)
        ? (2 * computeParams +
            estimatePostTrainingMoELoadBalanceFLOPsPerToken(
              computeParams,
              cfg,
              2,
            ) +
            2 * estimateLoRAAdapterParameterCount(computeParams, cfg)) *
          generationTokens
        : generationTokens === 0
          ? 0
          : Number.POSITIVE_INFINITY
    const nonGenerationFLOPs =
      Number.isFinite(compute.totalFLOPs) && Number.isFinite(generationFLOPs)
        ? Math.max(0, compute.totalFLOPs - generationFLOPs)
        : Number.POSITIVE_INFINITY
    const generationSeconds = estimatePostTrainingGenerationSeconds(
      cfg,
      computeParams,
      generationFeasibility,
    )
    const nonGenerationSeconds = estimatePostTrainingBatchedFLOPSeconds(
      cfg,
      nonGenerationFLOPs,
      computeParams,
      fPeakFLOPS,
      ptGPUs,
    )
    const qloraPenaltySeconds =
      cfg.approach === "qlora"
        ? estimatePostTrainingBatchedQLoRAPenaltySeconds(
            cfg,
            estimateQLoRAAffectedNonGenerationFLOPs(
              cfg,
              computeParams,
              compute.totalTokens,
            ),
            nonGenerationFLOPs,
            computeParams,
            fPeakFLOPS,
            ptGPUs,
          )
        : 0
    const theoSec =
      hasInvalidGPUCount ||
      hasInvalidHardware ||
      hasInvalidSemanticConfig
        ? Number.POSITIVE_INFINITY
        : nonGenerationSeconds +
          qloraPenaltySeconds +
          generationSeconds

    const totalTokens = compute.totalTokens
    const datasetSizeExamples = getFinitePositiveIntegerOrNull(
      cfg.datasetSizeExamples,
    )
    const epochs = getFinitePositiveOrNull(cfg.epochs)
    const batchSize = resolvePostTrainingBatchSize(cfg)
    const totalSteps =
      datasetSizeExamples !== null && epochs !== null && batchSize !== null
        ? Math.max(1, Math.ceil((datasetSizeExamples * epochs) / batchSize))
        : 0
    const stepLabels = getPostTrainingStepLabels(cfg.method)

    const time: TrainingTimeEstimate = {
      theoreticalDays: theoSec / 86400,
      theoreticalHours: theoSec / 3600,
      failureAdjustedDays: null,
      failureAdjustedHours: null,
      failureMultiplier: null,
      tokensPerSecond:
        Number.isFinite(theoSec) &&
        theoSec > 0 &&
        Number.isFinite(totalTokens)
          ? totalTokens / theoSec
          : 0,
      totalSteps,
      secondsPerStep:
        totalSteps > 0
          ? Number.isFinite(theoSec)
            ? theoSec / totalSteps
            : Number.POSITIVE_INFINITY
          : 0,
    }

    const postTrainingCostPerGPUHour =
      Number.isFinite(cfg.costPerGPUHour) && cfg.costPerGPUHour >= 0
        ? cfg.costPerGPUHour
        : null
    const computeCost = calculateGPUHourlyCost(
      ptGPUs,
      theoSec / 3600,
      postTrainingCostPerGPUHour,
    )
    const cost: CostEstimate = {
      computeCost,
      actualComputeCost: computeCost,
      storageCost: 0,
      failureOverheadCost: 0,
      totalCost: computeCost,
      checkpointSize: 0,
      numCheckpoints: 0,
      peakCheckpointStorage: 0,
      averageCheckpointStorage: 0,
      datasetStorageBytes: 0,
    }

    const requiredGpuEstimate = estimatePostTrainingRequiredGPUs(cfg)

    const warnings: Warning[] = []
    addPrecisionSupportWarnings(warnings, cfg.precision, gpu)
    addPostTrainingInputWarnings(warnings, cfg, postTrainingConfig)
    if (
      datasetSizeExamples !== null &&
      epochs !== null &&
      batchSize !== null &&
      totalSteps > 0
    ) {
      const totalExamples = datasetSizeExamples * epochs
      const fullBatches = Math.floor(totalExamples / batchSize)
      const finalBatchExamples = totalExamples - fullBatches * batchSize
      const hasPartialFinalBatch =
        finalBatchExamples > batchSize * 1e-9 &&
        finalBatchExamples < batchSize * (1 - 1e-9)

      if (totalSteps === 1 && hasPartialFinalBatch) {
        warnings.push({
          severity: "info",
          category: "data",
          message: `Dataset examples × epochs (${fmtCount(totalExamples)}) are below the configured batch size (${fmtCount(batchSize)}). The run is modeled as one partial ${stepLabels.singular}; frameworks that pad, drop, or resample examples can see a different effective batch.`,
        })
      } else if (hasPartialFinalBatch && totalSteps <= 100) {
        warnings.push({
          severity: "info",
          category: "data",
          message: `${stepLabels.markdownLabel} round up to ${totalSteps.toLocaleString()} with a final partial ${stepLabels.singular} (${fmtCount(finalBatchExamples)} of ${fmtCount(batchSize)} examples); align dataset size, epochs, or batch size if exact step cadence matters.`,
        })
      }
    }
    if (cfg.method === "ppo") {
      if (
        Number.isFinite(cfg.ppo.updateEpochs) &&
        cfg.ppo.updateEpochs >= 1 &&
        Number.isInteger(cfg.ppo.updateEpochs)
      ) {
        warnings.push({
          severity: "info",
          category: "compute",
          message: `PPO step count reports rollout batches. Policy, critic, and reference/KL optimizer work is modeled as ${cfg.ppo.updateEpochs.toLocaleString()} update epoch${cfg.ppo.updateEpochs === 1 ? "" : "s"} per rollout batch, so inner optimizer passes are not the displayed step count.`,
        })
      }
    }
    if (
      !hasInvalidGPUCount &&
      !hasInvalidHardware &&
      !hasInvalidSemanticConfig &&
      effectiveComputeGPUs < ptGPUs
    ) {
      warnings.push({
        severity: "warning",
        category: "compute",
        message: `Configured ${ptGPUs.toLocaleString()} GPUs, but the largest actual post-training batch exposes about ${effectiveComputeGPUs.toLocaleString()} independent training item${effectiveComputeGPUs === 1 ? "" : "s"}. Non-generation time scaling is capped at ${effectiveComputeGPUs.toLocaleString()} effective GPU${effectiveComputeGPUs === 1 ? "" : "s"}.`,
      })
    }
    if (
      !hasInvalidGPUCount &&
      !hasInvalidHardware &&
      !hasInvalidSemanticConfig &&
      !memory.fits
    ) {
      warnings.push({
        severity: "critical",
        category: "memory",
        message:
          requiredGpuEstimate.mode === "data-parallel" &&
          requiredGpuEstimate.numGPUsNeeded !== null
            ? `Per-GPU memory (${(memory.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memory.usableCapacity / 1e9).toFixed(1)} GB). Split the global batch over about ${requiredGpuEstimate.numGPUsNeeded.toLocaleString()} data-parallel GPUs to fit.`
            : requiredGpuEstimate.mode === "state-sharded-lower-bound" &&
                requiredGpuEstimate.numGPUsNeeded !== null
              ? `Per-GPU memory (${(memory.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memory.usableCapacity / 1e9).toFixed(1)} GB). Replicated model states are too large for this GPU; an ideal ZeRO-3/FSDP state-sharded lower bound is about ${requiredGpuEstimate.numGPUsNeeded.toLocaleString()} GPUs before activations, KV cache, largest-layer gathers, and communication.`
              : requiredGpuEstimate.stateFloorBytes > memory.usableCapacity
              ? `Per-GPU memory (${(memory.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memory.usableCapacity / 1e9).toFixed(1)} GB). Replicated model states and framework overhead alone require ${(requiredGpuEstimate.stateFloorBytes / 1e9).toFixed(1)} GB per GPU, so adding data-parallel GPUs will not make this fit without sharding, offload, or a smaller model.`
              : `Per-GPU memory (${(memory.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memory.usableCapacity / 1e9).toFixed(1)} GB). Even after splitting the batch across ${requiredGpuEstimate.maxUsefulGPUs.toLocaleString()} useful data-parallel GPUs, the per-GPU working set remains too large.`,
      })
    }
    if (
      !hasInvalidGPUCount &&
      !hasInvalidHardware &&
      !hasInvalidSemanticConfig &&
      generationFeasibility !== null &&
      generationFeasibility.requestedBatch > generationFeasibility.maxBatch
    ) {
      warnings.push({
        severity: "warning",
        category: "generation",
        message: formatGenerationCapacityWarning(cfg, generationFeasibility),
      })
    }
    if (
      !hasInvalidGPUCount &&
      !hasInvalidHardware &&
      !hasInvalidSemanticConfig &&
      generationFeasibility !== null &&
      (cfg.method === "ppo" || cfg.method === "grpo") &&
      ptGPUs > 1
    ) {
      warnings.push({
        severity: "info",
        category: "generation",
        message:
          "PPO/GRPO generation time assumes data-parallel serving replicas: each active GPU holds a full policy copy and serves its local completions. Tensor- or pipeline-parallel rollout engines can shift the latency and memory tradeoff, but are not modeled in post-training estimates.",
      })
    }
    const generationCrossoverBatch = estimateGenerationCrossoverBatch(cfg)
    const localGenerationBatch =
      generationFeasibility !== null
        ? estimateLocalGenerationBatch(cfg, generationFeasibility.requestedBatch)
        : 0
    if (
      !hasInvalidGPUCount &&
      !hasInvalidHardware &&
      !hasInvalidSemanticConfig &&
      generationFeasibility !== null &&
      generationCrossoverBatch !== null &&
      localGenerationBatch > 0 &&
      localGenerationBatch < generationCrossoverBatch * 0.25 &&
      generationFeasibility.maxBatch >
        generationFeasibility.requestedBatch * 1.5
    ) {
      warnings.push({
        severity: "info",
        category: "generation",
        message: `Autoregressive decode is likely memory-bandwidth-bound at ${generationFeasibility.requestedBatch.toLocaleString()} concurrent generation${generationFeasibility.requestedBatch === 1 ? "" : "s"} (~${localGenerationBatch.toLocaleString()} per active GPU); the estimated per-GPU memory/compute crossover on ${gpu.name} is about ${Math.round(generationCrossoverBatch).toLocaleString()}. If rollout quality and memory headroom allow, increasing concurrent generations can improve GPU utilization.`,
      })
    }

    return {
      memory,
      numGPUsNeeded: requiredGpuEstimate.numGPUsNeeded,
      numGPUsNeededMode: requiredGpuEstimate.mode,
      trainingTime: time,
      stepCountLabel: stepLabels.countLabel,
      stepTimeLabel: stepLabels.timeLabel,
      stepMarkdownLabel: stepLabels.markdownLabel,
      cost,
      warnings,
    }
  }, [resolvedPostTrainingConfig, postTrainingConfig])

  // ═══════════════════════════════════════════════════════════════════════════
  // EXPORT
  // ═══════════════════════════════════════════════════════════════════════════

  const currentOutput = activeTab === "pretraining" ? pretrainingOutput : postTrainingOutput

  const handleCopyText = useCallback(() => {
    const text =
      activeTab === "pretraining"
        ? generatePretrainingMarkdown(pretrainingOutput)
        : generatePostTrainingMarkdown(postTrainingOutput)
    navigator.clipboard.writeText(text).then(() => {
      setCopied("text")
      setTimeout(() => setCopied(null), 1500)
    })
  }, [activeTab, pretrainingOutput, postTrainingOutput])

  const handleCopyJSON = useCallback(() => {
    navigator.clipboard
      .writeText(serializeCalculatorOutput(currentOutput))
      .then(() => {
        setCopied("json")
        setTimeout(() => setCopied(null), 1500)
      })
  }, [currentOutput])

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════

  if (!mounted) {
    return (
      <div className="rounded-2xl border border-border bg-surface/70 p-8 backdrop-blur-sm">
        <div className="animate-pulse space-y-5">
          <div className="h-6 w-48 rounded-full bg-accent-soft" />
          <div className="h-14 rounded-xl bg-surface" />
          <div className="grid gap-5 lg:grid-cols-2">
            <div className="h-72 rounded-xl bg-surface" />
            <div className="h-72 rounded-xl bg-surface" />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className="overflow-hidden rounded-2xl border"
      style={{
        backgroundColor: colors.cardBg,
        borderColor: colors.border,
        color: colors.text,
      }}
    >
      {/* ── Header ── */}
      <div
        className="border-b px-6 py-7 sm:px-8 sm:py-8"
        style={{ borderColor: colors.border }}
      >
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-[72ch]">
            <p
              className="text-xs font-medium uppercase tracking-[0.2em]"
              style={{ color: colors.accent }}
            >
              GPU Calculator
            </p>
            <h2
              className="mt-4"
              style={{ fontFamily: "var(--font-display)", fontWeight: 280, letterSpacing: "-0.025em", lineHeight: 1.2 }}
            >
              Estimate GPU requirements for LLM training
            </h2>
            <p
              className="mt-4 text-sm"
              style={{ color: colors.textSecondary, lineHeight: 1.85 }}
            >
              Configure model architecture, training setup, and hardware to
              get memory breakdown, parallelism recommendation, training
              time, and cost estimates.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            {stats.map(({ label, value, icon: Icon }) => (
              <div
                key={label}
                className="rounded-xl border px-4 py-3.5"
                style={{
                  borderColor: colors.border,
                  backgroundColor: colors.panel,
                }}
              >
                <div
                  className="flex items-center gap-2 text-[10px] uppercase tracking-[0.18em]"
                  style={{ color: colors.textSecondary }}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {label}
                </div>
                <div
                  className="mt-2.5 text-xl tabular-nums"
                  style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}
                >
                  {value}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Tabs ── */}
      <div
        className="border-b px-5 py-4 sm:px-7"
        style={{ borderColor: colors.border }}
      >
        <div
          className="inline-flex gap-1 rounded-lg p-1"
          style={{ backgroundColor: isDark ? "oklch(0.18 0.006 260)" : "oklch(0.96 0.003 80)" }}
        >
          {tabs.map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className="no-theme-transition relative rounded-md px-5 py-2 text-sm font-medium"
              style={{
                color:
                  activeTab === tab.key
                    ? colors.accent
                    : colors.textSecondary,
                backgroundColor:
                  activeTab === tab.key ? colors.cardBg : "transparent",
                boxShadow:
                  activeTab === tab.key
                    ? `0 1px 3px ${isDark ? "rgba(0,0,0,0.35)" : "rgba(0,0,0,0.06)"}, 0 0 0 1px ${colors.border}`
                    : "none",
                transition: "all 200ms cubic-bezier(0.22, 1, 0.36, 1)",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <p
          className="mt-3 text-sm"
          style={{ color: colors.textSecondary, lineHeight: 1.85 }}
        >
          {tabs.find((tab) => tab.key === activeTab)?.description}
        </p>
      </div>

      {/* ── Main grid: inputs | results ── */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
        className="grid gap-4 p-4 sm:gap-5 sm:p-5 lg:grid-cols-[1.1fr_0.9fr]"
      >
        {/* ── Input panel ── */}
        <section
          className="gpu-calc-scroll rounded-xl border p-5 sm:p-6 lg:max-h-[82vh] lg:overflow-y-auto"
          style={{
            borderColor: colors.border,
            backgroundColor: colors.panel,
          }}
        >
          {activeTab === "pretraining" ? (
            <PretrainingPanel
              config={trainingConfig}
              onChange={setTrainingConfig}
              colors={colors}
              activeParameterCount={resolvedTrainingModel.parameterCounts.active}
              effectiveNumGPUs={effectiveTrainingNumGPUs}
              gpuCountDerivedFromTarget={gpuCountDerivedFromTarget}
              autoParallelismRecommendation={parallelismRecommendation}
            />
          ) : (
            <PostTrainingPanel
              config={postTrainingConfig}
              onChange={setPostTrainingConfig}
              colors={colors}
            />
          )}
        </section>

        {/* ── Results panel ── */}
        <section
          className="gpu-calc-scroll rounded-xl border p-5 sm:p-6 lg:max-h-[82vh] lg:overflow-y-auto"
          style={{
            borderColor: colors.border,
            backgroundColor: colors.panel,
          }}
        >
          {/* Export header bar */}
          <div
            className="sticky top-0 z-10 -mx-5 -mt-5 mb-5 flex items-center justify-between rounded-t-xl border-b px-5 py-3.5 backdrop-blur-xl sm:-mx-6 sm:-mt-6 sm:px-6"
            style={{
              borderColor: colors.border,
              backgroundColor: isDark
                ? "oklch(0.14 0.005 260 / 0.9)"
                : "oklch(0.993 0.003 80 / 0.92)",
            }}
          >
            <div
              className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em]"
              style={{ color: colors.accent }}
            >
              <Gauge className="h-4 w-4" />
              Results
            </div>
            <div className="flex gap-1.5">
              <button
                type="button"
                onClick={handleCopyText}
                className="no-theme-transition flex items-center gap-1 rounded-md px-2.5 py-1.5 text-[11px] font-medium"
                style={{
                  backgroundColor: copied === "text" ? colors.accentMuted : "transparent",
                  color: copied === "text" ? colors.accent : colors.textSecondary,
                  transition: "all 150ms ease",
                }}
              >
                {copied === "text" ? (
                  <Check className="h-3 w-3" />
                ) : (
                  <FileText className="h-3 w-3" />
                )}
                {copied === "text" ? "Copied" : "Text"}
              </button>
              <button
                type="button"
                onClick={handleCopyJSON}
                className="no-theme-transition flex items-center gap-1 rounded-md px-2.5 py-1.5 text-[11px] font-medium"
                style={{
                  backgroundColor: copied === "json" ? colors.accentMuted : "transparent",
                  color: copied === "json" ? colors.accent : colors.textSecondary,
                  transition: "all 150ms ease",
                }}
              >
                {copied === "json" ? (
                  <Check className="h-3 w-3" />
                ) : (
                  <ClipboardCopy className="h-3 w-3" />
                )}
                {copied === "json" ? "Copied" : "JSON"}
              </button>
            </div>
          </div>

          <ResultsSummary output={currentOutput} isDark={isDark} />
        </section>
      </motion.div>
    </div>
  )
}
