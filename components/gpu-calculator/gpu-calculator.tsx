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
  CalculatorTab,
  CostEstimate,
  MemoryBreakdown,
  MoESparsityMetrics,
  ModelArchitecture,
  MoEConfig,
  ParameterCounts,
  ParallelismConfig,
  ParallelismRecommendation,
  PostTrainingConfig,
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
} from "./formulas/compute"
import {
  calculateTotalMemoryPerGPU,
  calculateMinGPUVRAMFloor,
  resolvePostTrainingOptimizerProfile,
  calculatePostTrainingActivationMemory,
  calculatePostTrainingForwardWorkingMemory,
  calculateLoRAMemory,
  calculateQLoRAMemory,
  calculateDPOMemory,
  calculatePPOMemory,
  calculateGRPOMemory,
} from "./formulas/memory"
import {
  calculateTrainingTime,
  calculateCost,
  getDefaultMFU,
  calculateGenerationTime,
  calculatePostTrainingCompute,
  getEffectiveTrainingTFLOPS,
  resolveTrainingMFU,
  calculateCPUOffloadEfficiency,
} from "./formulas/cost"
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
} from "./formulas/parallelism"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const QLORA_THROUGHPUT_PENALTY = 1.75

function resolveExplicitNumGPUs(numGPUs: number | null | undefined): number {
  return typeof numGPUs === "number" && Number.isFinite(numGPUs) && numGPUs > 0
    ? Math.round(numGPUs)
    : 1
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
      severity: "warning",
      category: "precision",
      message: `${gpu.name} does not support BF16. Use FP16 with loss scaling.`,
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
    warnings.push({
      severity: "info",
      category: "precision",
      message: gpu.supportsTF32
        ? "FP32 mode uses TF32 tensor-core throughput where available, but tensors still occupy FP32 memory. Model states and activations are estimated at 4 bytes per element."
        : "FP32 mode stores tensors in full precision, so model states and activations are estimated at 4 bytes per element.",
    })
  }

  if (precision === "fp8" && !gpu.supportsFP8) {
    warnings.push({
      severity: "warning",
      category: "precision",
      message: `${gpu.name} does not support FP8 kernels. Estimates assume BF16-class throughput and storage instead.`,
    })
  }
}

function addPostTrainingInputWarnings(
  warnings: Warning[],
  config: PostTrainingConfig,
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

  addKVHeadValidationWarnings(warnings, config.baseModel.architecture)

  warnings.push({
    severity: "info",
    category: "memory",
    message:
      "Post-training activation memory assumes full activation checkpointing with one-layer recompute workspace. Runs without activation checkpointing can require substantially more VRAM.",
  })

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
        "Partial full fine-tuning assumes the frozen portion is a contiguous set of layers whose backward pass can be skipped. If trainable weights are spread through the model, compute and activation memory can be close to full fine-tuning.",
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

  if (config.baseModel.architecture.attentionVariant === "mla") {
    warnings.push({
      severity: "info",
      category: "compute",
      message:
        "MLA models use architecture-specific latent KV dimensions that are not exposed in this calculator. Attention and generation KV-cache estimates fall back to full hidden-width assumptions and may be conservative.",
    })
  }

  if (!Number.isFinite(config.batchSize) || config.batchSize < 1) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "Batch size must be at least 1.",
    })
  }

  if (!Number.isFinite(config.hardware.numGPUs) || config.hardware.numGPUs < 1) {
    warnings.push({
      severity: "critical",
      category: "hardware",
      message: "GPU count must be at least 1.",
    })
  }

  if (
    config.hardware.gpu.singleDeviceOnly &&
    resolveExplicitNumGPUs(config.hardware.numGPUs) > 1
  ) {
    warnings.push({
      severity: "critical",
      category: "hardware",
      message: `${config.hardware.gpu.name} only supports single-device execution.`,
    })
  }

  if (!Number.isFinite(config.costPerGPUHour) || config.costPerGPUHour < 0) {
    warnings.push({
      severity: "critical",
      category: "cost",
      message: "Cost per GPU-hour must be a non-negative finite value.",
    })
  }

  if (
    config.optimizer === "adamw-fp8" &&
    (config.precision !== "fp8" || !config.hardware.gpu.supportsFP8)
  ) {
    warnings.push({
      severity: "warning",
      category: "precision",
      message:
        "AdamW FP8 storage requires FP8 precision on FP8-capable hardware. Estimates fall back to AdamW mixed-precision optimizer storage.",
    })
  }

  if (config.precision === "fp8" && config.hardware.gpu.supportsFP8) {
    warnings.push({
      severity: "info",
      category: "precision",
      message:
        config.optimizer === "adamw-fp8"
          ? "Post-training AdamW FP8 assumes MS-AMP-style persistent FP8 parameter and gradient storage for trainable models. TransformerEngine-style FP8 kernels would use bf16/fp16 model-state memory instead."
          : "Post-training FP8 is modeled as a throughput setting here; model weights, activations, and frozen reference/reward models remain estimated at bf16/fp16 storage size.",
    })
  }

  if (
    (config.approach === "lora" || config.approach === "qlora") &&
    config.lora.targetModules.length === 0
  ) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "At least one LoRA target module must be selected.",
    })
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

  if (config.method === "dpo") {
    warnings.push({
      severity: "info",
      category: "memory",
      message:
        "DPO estimates keep the reference-policy path in the training loop. Pipelines that precompute fixed reference log-probs can discard the reference model after the cache pass and avoid repeated-epoch reference scoring; that optimization is not modeled here.",
    })
  }

  if (config.approach === "qlora") {
    warnings.push({
      severity: "info",
      category: "memory",
      message:
        "QLoRA GPU memory excludes transient loading/dequantization and CPU RAM requirements. Loading a quantized checkpoint can still require substantial host memory and short-lived extra GPU buffers.",
    })
  }

  if (
    (config.approach === "lora" || config.approach === "qlora") &&
    (!Number.isFinite(config.lora.rank) || config.lora.rank < 1)
  ) {
    warnings.push({
      severity: "critical",
      category: "compute",
      message: "LoRA rank must be at least 1.",
    })
  }

  if (
    config.method === "grpo" &&
    (!Number.isFinite(config.grpo.groupSize) || config.grpo.groupSize < 2)
  ) {
    warnings.push({
      severity: "critical",
      category: "generation",
      message: "GRPO group size must be at least 2.",
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
      message: "PPO critic and reward model parameter counts must be positive.",
    })
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
}

function isFinitePositive(value: number): boolean {
  return Number.isFinite(value) && value > 0
}

function getFinitePositiveOrNull(value: number): number | null {
  return Number.isFinite(value) && value > 0 ? value : null
}

function addKVHeadValidationWarnings(
  warnings: Warning[],
  architecture: ModelArchitecture,
): void {
  const { a, a_kv } = architecture

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
  }
}

function optimizerProfileUsesMasterWeights(config: TrainingConfig): boolean {
  if (config.ampAutocast) {
    return false
  }

  const profile = OPTIMIZER_PROFILES.find(
    (candidate) => candidate.id === config.optimizer,
  )
  const variant =
    config.gradientPrecision === "bf16" ? profile?.bf16Grad : profile?.fp32Grad

  return (variant?.masterWeightBytes ?? 0) > 0
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

function resolveFFNWidth(arch: ModelArchitecture, moe: MoEConfig): number {
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
  return (
    parallelism.framework === "fsdp" &&
    parallelism.N_pp > 1 &&
    parallelism.zeroStage === 2 &&
    numMicrobatches < 2 * parallelism.N_pp
  )
}

function getEffectivePipelineBubbleVP(
  parallelism: ParallelismConfig,
  numMicrobatches: number,
): number {
  return usesAFABSchedule(parallelism, numMicrobatches)
    ? 1
    : parallelism.VP
}

function estimateMaxMicroBatch(
  memory: MemoryBreakdown,
  currentBatch: number,
  minVRAMFloor = 0,
): number {
  if (
    Number.isFinite(minVRAMFloor) &&
    minVRAMFloor > memory.usableCapacity
  ) {
    return 0
  }

  if (memory.activations <= 0 || currentBatch <= 0)
    return Math.max(1, currentBatch)
  const perSample = memory.activations / currentBatch
  if (perSample <= 0) return currentBatch
  const nonAct =
    memory.parameters +
    memory.gradients +
    memory.optimizerStates +
    memory.communicationBuffers +
    memory.frameworkOverhead
  const available = memory.usableCapacity / 1.04 - nonAct
  return available <= 0 ? 0 : Math.max(1, Math.floor(available / perSample))
}

function scaleParameterCounts(
  counts: ParameterCounts,
  targetTotal: number | null,
  targetActive: number | null,
): ParameterCounts {
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
  const paddedVocab = calculateVocabPadding(architecture.V, tpDegree)

  if (paddedVocab === architecture.V) {
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

  const architecture =
    config.model.inputMode === "quick"
      ? estimateParametersQuick(config.model.quickMode.totalParameters)
      : preset?.architecture ?? config.model.architecture
  const moe =
    config.model.inputMode === "preset"
      ? preset?.moe ?? disableMoEConfig(config.model.moe)
      : config.model.inputMode === "quick"
        ? disableMoEConfig(config.model.moe)
        : config.model.moe
  const rawCounts = calculateParameterCount(architecture, moe, config.sequenceLength)

  if (config.model.inputMode === "quick") {
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
    return {
      architecture,
      moe,
      parameterCounts: scaleParameterCounts(
        rawCounts,
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
  const method = config.approach === "mezo" ? "sft" : config.method
  const optimizer =
    config.approach === "mezo"
      ? "mezo"
      : config.optimizer === "mezo"
        ? "adamw-mixed"
        : config.optimizer

  if (config.baseModel.inputMode === "preset") {
    const preset =
      MODEL_PRESETS.find((candidate) => candidate.id === config.baseModel.presetId) ??
      null

    if (!preset) {
      return { ...config, method, optimizer }
    }

    return {
      ...config,
      method,
      optimizer,
      baseModel: {
        ...config.baseModel,
        parameterCount: preset.parameterCount,
        architecture: preset.architecture,
        moe: preset.moe ?? disableMoEConfig(config.baseModel.moe),
      },
    }
  }

  return {
    ...config,
    method,
    optimizer,
    baseModel: {
      ...config.baseModel,
      architecture: estimateParametersQuick(config.baseModel.parameterCount),
      moe: disableMoEConfig(config.baseModel.moe),
    },
  }
}

function resolvePostTrainingComputeParameterCount(
  config: PostTrainingConfig,
): number {
  if (config.baseModel.moe.enabled && config.baseModel.moe.activeParameterCount) {
    return config.baseModel.moe.activeParameterCount
  }

  const counts = calculateParameterCount(
    config.baseModel.architecture,
    config.baseModel.moe,
    config.sequenceLength,
  )

  if (!Number.isFinite(counts.total) || counts.total <= 0) {
    return config.baseModel.parameterCount
  }

  return counts.active * (config.baseModel.parameterCount / counts.total)
}

function resolveRequestedNumGPUs(
  config: TrainingConfig,
  totalFLOPs: number,
  activeParams: number,
): number {
  const explicitNumGPUs = resolveExplicitNumGPUs(config.hardware.numGPUs)
  const targetDays = config.hardware.targetTrainingDays

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

  let guess = explicitNumGPUs

  for (let iteration = 0; iteration < 8; iteration += 1) {
    const mfu = resolveTrainingMFU(config, activeParams, guess)
    const offloadEfficiency = calculateCPUOffloadEfficiency(config)
    const next = Math.max(
      1,
      Math.ceil(
        totalFLOPs /
          Math.max(secondsBudget * fPeakFLOPS * mfu * offloadEfficiency, 1),
      ),
    )

    if (next === guess) {
      return next
    }

    guess = next
  }

  return guess
}

function getParallelWorldSize(parallelism: ParallelismConfig): number {
  return (
    parallelism.N_dp *
    parallelism.N_tp *
    parallelism.N_pp *
    parallelism.N_cp *
    parallelism.N_ep
  )
}

function resolveParallelWorldSize(parallelism: ParallelismConfig): number {
  return resolveExplicitNumGPUs(getParallelWorldSize(parallelism))
}

function hasIntegerExpertDataParallelDegree(
  parallelism: ParallelismConfig,
): boolean {
  const numerator = parallelism.N_dp * parallelism.N_tp

  return (
    Number.isFinite(numerator) &&
    Number.isFinite(parallelism.N_ep) &&
    parallelism.N_ep > 0 &&
    numerator % parallelism.N_ep === 0
  )
}

function resolveTrainableParameterCount(config: PostTrainingConfig): number {
  const percentage = config.trainableParameterPercentage
  const ratio =
    percentage === null || !Number.isFinite(percentage) || percentage <= 0
      ? 1
      : Math.min(percentage, 100) / 100
  return config.baseModel.parameterCount * ratio
}

function resolvePostTrainingGRPOGroupSize(config: PostTrainingConfig): number {
  return Number.isFinite(config.grpo.groupSize)
    ? Math.max(1, config.grpo.groupSize)
    : 1
}

function getPostTrainingParallelWorkItems(config: PostTrainingConfig): number {
  const batch = Number.isFinite(config.batchSize)
    ? Math.max(0, config.batchSize)
    : 0

  if (config.method === "grpo") {
    return batch * resolvePostTrainingGRPOGroupSize(config)
  }

  return batch
}

function getPostTrainingMemorySplitLimit(config: PostTrainingConfig): number {
  const batch = Number.isFinite(config.batchSize)
    ? Math.max(0, Math.ceil(config.batchSize))
    : 0

  if (config.method === "grpo") {
    const groupSize = Number.isFinite(config.grpo.groupSize)
      ? Math.max(1, Math.ceil(config.grpo.groupSize))
      : 1

    return Math.max(1, batch * groupSize)
  }

  return Math.max(1, batch)
}

function getKVCacheBytesPerElement(config: PostTrainingConfig): number {
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

  const arch = config.baseModel.architecture
  const kvHeads = arch.a_kv ?? arch.a
  const sequenceLength = getFinitePositiveOrNull(config.sequenceLength)
  const headDim =
    Number.isFinite(arch.a) &&
    arch.a > 0 &&
    Number.isFinite(arch.d) &&
    arch.d > 0
      ? arch.d / arch.a
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
    sequenceLength !== null ? 16 * sequenceLength : 0
  const generationBytesPerSequence = kvPerSequence + rolloutBytesPerSequence
  const generationAvailableBytes =
    memory.usableCapacity / 1.04 -
    memory.parameters -
    memory.gradients -
    memory.optimizerStates -
    memory.frameworkOverhead
  const maxBatchPerGPU =
    generationBytesPerSequence > 0
      ? Math.floor(generationAvailableBytes / generationBytesPerSequence)
      : Number.POSITIVE_INFINITY
  const maxBatch = Number.isFinite(maxBatchPerGPU)
    ? maxBatchPerGPU * resolveExplicitNumGPUs(config.hardware.numGPUs)
    : maxBatchPerGPU
  const batchSize = getFinitePositiveOrNull(config.batchSize) ?? 0
  const requestedBatch =
    config.method === "grpo"
      ? resolvePostTrainingGRPOGroupSize(config) * batchSize
      : batchSize

  return {
    requestedBatch,
    maxBatch,
    rounds:
      maxBatch > 0 && Number.isFinite(maxBatch)
        ? Math.max(1, Math.ceil(requestedBatch / maxBatch))
        : Number.POSITIVE_INFINITY,
  }
}

function estimateGenerationCrossoverBatch(
  config: PostTrainingConfig,
): number | null {
  if (config.method !== "ppo" && config.method !== "grpo") {
    return null
  }

  const gpu = config.hardware.gpu
  const fPeakTFLOPS =
    config.precision === "fp32"
      ? gpu.supportsTF32 && gpu.tf32TFLOPS !== null
        ? gpu.tf32TFLOPS
        : gpu.halfPrecisionTFLOPS / 8
      : gpu.halfPrecisionTFLOPS
  const fPeakFLOPS = fPeakTFLOPS * 1e12
  const bandwidthBytesPerSecond = gpu.memoryBandwidthGBps * 1e9 * 0.9
  const weightBytes = config.precision === "fp32" ? 4 : 2

  if (
    !Number.isFinite(fPeakFLOPS) ||
    fPeakFLOPS <= 0 ||
    !Number.isFinite(bandwidthBytesPerSecond) ||
    bandwidthBytesPerSecond <= 0
  ) {
    return null
  }

  return (weightBytes * fPeakFLOPS) / (2 * bandwidthBytesPerSecond)
}

function estimatePostTrainingGenerationSeconds(
  config: PostTrainingConfig,
  policyParams: number,
  feasibility: GenerationFeasibilityEstimate | null,
): number {
  if (config.method !== "ppo" && config.method !== "grpo") {
    return 0
  }

  if (feasibility === null || feasibility.requestedBatch <= 0) {
    return 0
  }

  if (!Number.isFinite(feasibility.maxBatch) || feasibility.maxBatch <= 0) {
    return Number.POSITIVE_INFINITY
  }

  const datasetSizeExamples = getFinitePositiveOrNull(
    config.datasetSizeExamples,
  )
  const epochs = getFinitePositiveOrNull(config.epochs)
  const batchSize = getFinitePositiveOrNull(config.batchSize)
  const sequenceLength = getFinitePositiveOrNull(config.sequenceLength)

  if (
    datasetSizeExamples === null ||
    epochs === null ||
    batchSize === null ||
    sequenceLength === null
  ) {
    return Number.POSITIVE_INFINITY
  }

  const batchPerRound = Math.min(feasibility.requestedBatch, feasibility.maxBatch)
  const rounds = Math.max(1, Math.ceil(feasibility.requestedBatch / batchPerRound))
  const promptBatches = Math.max(
    1,
    Math.ceil((datasetSizeExamples * epochs) / batchSize),
  )
  // The UI exposes a single sequence length for post-training. Treat it as the
  // generated/scored token horizon and avoid inventing a separate prompt split.
  const perRound = calculateGenerationTime(
    policyParams,
    config,
    batchPerRound,
    sequenceLength,
    0,
  )

  return perRound.totalSeconds * rounds * promptBatches
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

  const ppoUpdateEpochs = Number.isFinite(config.ppo.updateEpochs)
    ? Math.max(1, config.ppo.updateEpochs)
    : 1
  const actorBaseFLOPsPerToken =
    config.method === "sft"
      ? 4 * policyParams
      : config.method === "dpo"
        ? 6 * policyParams
        : config.method === "ppo"
          ? ppoUpdateEpochs * 6 * policyParams
          : 6 * policyParams

  return actorBaseFLOPsPerToken * totalTokens
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
  const totalParamCount = config.baseModel.parameterCount
  const trainableParamCount = resolveTrainableParameterCount(config)
  const frozenParamCount = Math.max(totalParamCount - trainableParamCount, 0)
  const parameters = totalParamCount * optimizer.parameterBytes
  const gradients = trainableParamCount * optimizer.betaGrad
  const optimizerStates = trainableParamCount * optimizer.kOpt
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
      trainableParamCount * optimizer.parameterBytes + gradients + optimizerStates,
    frozenModels: frozenParamCount * optimizer.parameterBytes,
    loraAdapter: 0,
    ppoBuffers: 0,
    items: [
      {
        label:
          frozenParamCount > 0
            ? "Trainable model parameters"
            : "Model parameters",
        category: "trainable",
        bytes: trainableParamCount * optimizer.parameterBytes,
      },
      ...(frozenParamCount > 0
        ? [
            {
              label: "Frozen model parameters",
              category: "frozen" as const,
              bytes: frozenParamCount * optimizer.parameterBytes,
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
  const totalParamCount = config.baseModel.parameterCount
  const trainableParamCount = resolveTrainableParameterCount(config)
  const frozenParamCount = Math.max(totalParamCount - trainableParamCount, 0)
  const parameters = totalParamCount * wb
  const activations = calculatePostTrainingForwardWorkingMemory(
    config.baseModel.architecture,
    config,
  )
  const frameworkOverhead = 1e9
  const total = (parameters + activations + frameworkOverhead) * 1.04
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
    trainableModels: trainableParamCount * wb,
    frozenModels: frozenParamCount * wb,
    loraAdapter: 0,
    ppoBuffers: 0,
    items: [
      {
        label:
          frozenParamCount > 0
            ? "Trainable model parameters"
            : "Model parameters",
        category: "trainable",
        bytes: trainableParamCount * wb,
      },
      ...(frozenParamCount > 0
        ? [
            {
              label: "Frozen model parameters",
              category: "frozen" as const,
              bytes: frozenParamCount * wb,
            },
          ]
        : []),
      { label: "Activations", category: "buffer", bytes: activations },
    ],
  }
}

function getPostTrainingMemory(
  config: PostTrainingConfig,
): PostTrainingMemoryBreakdown {
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
} {
  const maxUsefulGPUs = getPostTrainingMemorySplitLimit(config)
  const oneGpuMemory = getPostTrainingMemory(withPostTrainingGPUCount(config, 1))
  const stateFloorBytes = getPostTrainingStateFloorBytes(oneGpuMemory)

  if (stateFloorBytes > oneGpuMemory.usableCapacity) {
    return {
      numGPUsNeeded: null,
      stateFloorBytes,
      maxUsefulGPUs,
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
  }
}

// ── Input validation (spec Section 14) ──

function generateInputWarnings(
  config: TrainingConfig,
  architecture: ModelArchitecture,
  moe: MoEConfig,
  totalParams: number,
  parallelism: ParallelismConfig,
  numGPUs: number,
  chinchillaRatio: number,
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

  if (!Number.isFinite(totalParams) || totalParams <= 0)
    w.push({
      severity: "critical",
      category: "compute",
      message: "Parameter count must be positive.",
    })
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
  addKVHeadValidationWarnings(w, architecture)
  if (architecture.attentionVariant === "mla")
    w.push({
      severity: "info",
      category: "compute",
      message:
        "MLA attention uses architecture-specific latent KV dimensions that are not exposed in this calculator. Attention FLOPs and KV-shaped estimates use a full hidden-width fallback and may be conservative.",
    })
  if (!totalTokensValid)
    w.push({
      severity: "critical",
      category: "data",
      message: "Total training tokens must be positive.",
    })
  if (!uniqueTokensValid)
    w.push({
      severity: "critical",
      category: "data",
      message: "Unique token count must be positive.",
    })
  if (
    totalTokensValid &&
    uniqueTokensValid &&
    config.uniqueTokens > config.totalTokens
  )
    w.push({
      severity: "critical",
      category: "data",
      message: "Unique tokens U must be less than or equal to total tokens D.",
    })
  if (
    Number.isFinite(chinchillaRatio) &&
    chinchillaRatio > 0 &&
    chinchillaRatio < 1
  )
    w.push({
      severity: "warning",
      category: "data",
      message:
        "Token count is below 1x Chinchilla optimal — model will be severely undertrained.",
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
      severity: "warning",
      category: "data",
      message: `Training for ${uniqueTokenRatio.toFixed(0)} epochs — additional repetition is effectively wasted compute.`,
    })
  else if (uniqueTokenRatio !== null && uniqueTokenRatio > 4)
    w.push({
      severity: "warning",
      category: "data",
      message: `Training for ${uniqueTokenRatio.toFixed(1)} epochs — repeated data is in the diminishing-returns regime.`,
    })
  if (moe.enabled)
    w.push({
      severity: "info",
      category: "data",
      message:
        "MoE scaling guidance uses active parameters with dense Chinchilla-style coefficients. MoE-specific scaling studies suggest the optimal token-to-active-parameter ratio can be lower for large sparse models, so treat the token recommendation as approximate.",
    })
  if (!isFinitePositive(config.microBatchSize))
    w.push({
      severity: "critical",
      category: "compute",
      message: "Micro-batch size must be at least 1.",
    })
  if (!isFinitePositive(config.gradientAccumulationSteps))
    w.push({
      severity: "critical",
      category: "compute",
      message: "Gradient accumulation steps must be at least 1.",
    })
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
  if (moe.enabled) {
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
      !isFinitePositive(moe.denseIntermediateSize)
    )
      w.push({
        severity: "critical",
        category: "compute",
        message: "MoE dense FFN size must be positive when specified.",
      })
    if (
      moe.expertIntermediateSize !== null &&
      !isFinitePositive(moe.expertIntermediateSize)
    )
      w.push({
        severity: "critical",
        category: "compute",
        message: "MoE expert FFN size must be positive when specified.",
      })
  }
  if (!isFinitePositive(numGPUs))
    w.push({
      severity: "critical",
      category: "hardware",
      message: "GPU count must be at least 1.",
    })
  if (Number.isFinite(numGPUs) && numGPUs > 100000)
    w.push({
      severity: "warning",
      category: "hardware",
      message: "GPU count exceeds 100,000.",
    })
  if (config.hardware.gpu.singleDeviceOnly && numGPUs > 1)
    w.push({
      severity: "critical",
      category: "hardware",
      message: `${config.hardware.gpu.name} only supports single-device execution.`,
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
  addPrecisionSupportWarnings(w, config.precision, config.hardware.gpu)
  if (
    config.optimizer === "adamw-fp8" &&
    (config.precision !== "fp8" || !config.hardware.gpu.supportsFP8)
  )
    w.push({
      severity: "warning",
      category: "precision",
      message:
        "AdamW FP8 storage requires FP8 precision on FP8-capable hardware. Estimates fall back to AdamW mixed-precision optimizer storage.",
    })
  if (config.precision === "fp8" && config.hardware.gpu.supportsFP8)
    w.push({
      severity: "info",
      category: "precision",
      message:
        config.optimizer === "adamw-fp8" &&
        config.fp8.storageMode === "ms-amp"
          ? "MS-AMP FP8 storage reduces parameter and gradient memory only; activations and output logits remain estimated at bf16/fp16 size."
          : "TransformerEngine-style FP8 is modeled as kernel throughput only; model states, activations, and output logits remain estimated at bf16/fp16 size.",
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
    optimizerProfileUsesMasterWeights(config)
  )
    w.push({
      severity: "info",
      category: "memory",
      message:
        "FSDP mixed-precision model-state memory is shown with ZeRO-style low-precision parameter plus sharded master-weight categories. Native PyTorch FSDP keeps resident sharded parameters in full precision, so profile a real run for exact category splits.",
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
      config.mfuOverride > 1)
  )
    w.push({
      severity: "critical",
      category: "compute",
      message: "MFU override must be greater than 0 and at most 100%.",
    })
  else if (config.mfuOverride !== null && config.mfuOverride > 0.7)
    w.push({
      severity: "warning",
      category: "compute",
      message:
        "MFU override is above the calculator's calibrated 10-70% range, so time and cost may be optimistic.",
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

    if (config.model.inputMode === "detailed") {
      const hiddenAlignment = validateHiddenDimAlignment(architecture.d)
      if (!hiddenAlignment.valid)
        w.push({
          severity: "warning",
          category: "parallelism",
          message: `${hiddenAlignment.message}, causing significant tensor-core inefficiency.`,
        })
    }

    const dff = resolveFFNWidth(architecture, moe)
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
    const tpEpSp = validateTensorExpertSequenceParallelism(parallelism, moe.enabled)
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
          severity: "warning",
          category: "parallelism",
          message: mb.message,
        })
    }
    if (
      moe.enabled &&
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
    if (parallelism.N_tp > config.hardware.gpu.gpusPerNode)
      w.push({
        severity: "critical",
        category: "parallelism",
        message: `N_tp=${parallelism.N_tp} exceeds the per-node high-bandwidth group size of ${config.hardware.gpu.gpusPerNode}.`,
      })
    if (!moe.enabled && parallelism.N_ep > 1)
      w.push({
        severity: "warning",
        category: "parallelism",
        message: "Expert parallelism is only meaningful for MoE models.",
      })
    if (moe.enabled && parallelism.N_ep > 1 && moe.E % parallelism.N_ep !== 0)
      w.push({
        severity: "critical",
        category: "parallelism",
        message: `N_ep=${parallelism.N_ep} must divide the total expert count E=${moe.E}.`,
      })
    if (
      moe.enabled &&
      parallelism.N_ep > 1 &&
      !hasIntegerExpertDataParallelDegree(parallelism)
    )
      w.push({
        severity: "critical",
        category: "parallelism",
        message: `N_ep=${parallelism.N_ep} must divide N_dp × N_tp (${parallelism.N_dp * parallelism.N_tp}) so expert data parallelism is an integer.`,
      })
    if (
      moe.enabled &&
      parallelism.N_ep > 1 &&
      parallelism.N_tp * parallelism.N_ep > config.hardware.gpu.gpusPerNode
    )
      w.push({
        severity: "critical",
        category: "parallelism",
        message: `N_tp × N_ep must stay within the per-node GPU group (${config.hardware.gpu.gpusPerNode}) for expert traffic.`,
      })
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
        parallelism.N_tp * parallelism.N_cp > config.hardware.gpu.gpusPerNode)
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
        message: `Pipeline bubble is ${(bubble * 100).toFixed(1)}%. Increase gradient accumulation steps to reduce idle time.`,
      })
    else if (parallelism.N_pp > 1 && bubble > 0.2)
      w.push({
        severity: "info",
        category: "parallelism",
        message: `Pipeline bubble is ${(bubble * 100).toFixed(1)}%. A common rule of thumb is num_microbatches ≥ ${4 * parallelism.N_pp}.`,
      })
    if (effectiveZeroStage === 3)
      w.push({
        severity: "info",
        category: "parallelism",
        message:
          "ZeRO-3 / FULL_SHARD maximizes memory savings but adds extra communication overhead and can reduce throughput.",
      })
    if (
      config.cpuOffload === "optimizer-and-params" &&
      effectiveZeroStage !== 3
    )
      w.push({
        severity: "critical",
        category: "memory",
        message:
          'Parameter offload requires ZeRO-3 or FSDP FULL_SHARD / HYBRID_SHARD.',
      })
    if (config.cpuOffload !== "none") {
      const offloadEfficiency = calculateCPUOffloadEfficiency(config)
      const efficiencyLabel =
        offloadEfficiency > 0 && Number.isFinite(offloadEfficiency)
          ? ` Modeled throughput efficiency is ${(offloadEfficiency * 100).toFixed(1)}% before other communication overheads.`
          : ""

      w.push({
        severity: "warning",
        category: "memory",
        message:
          `CPU offloading reduces GPU memory pressure but slows training because optimizer or parameter traffic shifts onto the host interconnect.${efficiencyLabel}`,
      })
    }
  }

  return w
}

// ── Export formatters ──

function fmtBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "--"
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`
  return `${(bytes / 1e3).toFixed(0)} KB`
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

function fmtDuration(hours: number): string {
  if (!Number.isFinite(hours) || hours < 0) return "--"
  if (hours >= 24) return `${(hours / 24).toFixed(1)} days`
  return `${hours.toFixed(1)} hours`
}

function fmtCurrency(value: number, cents = false): string {
  if (!Number.isFinite(value) || value < 0) return "--"
  return cents ? `$${value.toFixed(2)}` : `$${Math.round(value).toLocaleString()}`
}

function generatePretrainingMarkdown(o: PretrainingOutput): string {
  const hasActive = o.parameterCounts.active !== o.parameterCounts.total
  return [
    "# GPU Calculator — Pretraining Results\n",
    "## Model",
    `- Parameters: ${fmtCount(o.parameterCounts.total)}${hasActive ? ` total, ${fmtCount(o.parameterCounts.active)} active` : ""}`,
    "",
    "## Compute",
    `- Total FLOPs: ${fmtFLOPs(o.computeEstimate.totalFLOPs)}`,
    `- Chinchilla Ratio: ${fmtMultiplier(o.chinchilla.ratio)}`,
    `- Predicted Loss: ${Number.isFinite(o.predictedLossNats) ? o.predictedLossNats.toFixed(3) : "--"} nats`,
    `- Attention Overhead: ${fmtFractionPercent(o.attentionOverheadFraction)}`,
    "",
    "## Memory per GPU",
    `- Parameters: ${fmtBytes(o.memory.parameters)}`,
    `- Gradients: ${fmtBytes(o.memory.gradients)}`,
    `- Optimizer States: ${fmtBytes(o.memory.optimizerStates)}`,
    `- Activations: ${fmtBytes(o.memory.activations)}`,
    `- Buffers: ${fmtBytes(o.memory.communicationBuffers)}`,
    `- Total: ${fmtBytes(o.memory.total)} / ${fmtBytes(o.memory.usableCapacity)} usable`,
    "",
    "## Parallelism",
    `- Strategy: ${o.parallelismRecommendation.strategyLabel}`,
    `- Pipeline Bubble: ${fmtFractionPercent(o.pipelineBubbleFraction)}`,
    `- Minimum GPUs: ${fmtCount(o.minGPUsNeeded)}`,
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
    `- Storage: ${fmtCurrency(o.cost.storageCost, true)}`,
    `- Total: ${fmtCurrency(o.cost.totalCost)}`,
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
    `- Total: ${fmtBytes(o.memory.total)} / ${fmtBytes(o.memory.usableCapacity)} usable`,
    `- Fits: ${o.memory.fits ? "Yes" : "No"}`,
    `- GPUs Needed: ${o.numGPUsNeeded ?? "No data-parallel fit"}`,
    "",
    "## Training Time",
    `- Estimated: ${fmtDuration(o.trainingTime.theoreticalHours)}`,
    `- Throughput: ${fmtCount(o.trainingTime.tokensPerSecond)} tok/s`,
    "",
    "## Cost",
    `- Total: ${fmtCurrency(o.cost.totalCost)}`,
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
        resolvedTrainingModel.parameterCounts.active,
      ),
    [
      trainingConfig,
      computeEstimate.totalFLOPs,
      resolvedTrainingModel.parameterCounts.active,
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
      parallelism: normalizedTrainingParallelism,
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
    const p = resolvedTrainingConfig.parallelism
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
      resolvedTrainingModel.parameterCounts,
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
    const parallelWorldSize = resolveParallelWorldSize(
      parallelismRecommendation.config,
    )

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

  const paddedParameterCounts = useMemo(
    () =>
      applyVocabPaddingToCounts(
        resolvedTrainingModel.parameterCounts,
        resolvedTrainingModel.architecture,
        parallelismRecommendation.config.N_tp,
      ),
    [resolvedTrainingModel, parallelismRecommendation.config.N_tp],
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

  const memoryBreakdown = useMemo(() => {
    const activationSchedule = usesAFABSchedule(
      effectiveConfig.parallelism,
      effectiveConfig.gradientAccumulationSteps,
    )
      ? "afab"
      : "none"

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
    const N_dp = parallelismRecommendation.config.N_dp
    const sequences =
      trainingConfig.microBatchSize *
      trainingConfig.gradientAccumulationSteps *
      N_dp
    return { sequences, tokens: sequences * trainingConfig.sequenceLength }
  }, [
    trainingConfig.microBatchSize,
    trainingConfig.gradientAccumulationSteps,
    trainingConfig.sequenceLength,
    parallelismRecommendation.config.N_dp,
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
        memoryBreakdown,
        trainingConfig.microBatchSize,
        parallelismRecommendation.minVRAMFloor,
      ),
    [
      memoryBreakdown,
      trainingConfig.microBatchSize,
      parallelismRecommendation.minVRAMFloor,
    ],
  )

  const moeSparsity = useMemo((): MoESparsityMetrics | null => {
    if (
      !resolvedTrainingModel.moe.enabled ||
      resolvedTrainingModel.parameterCounts.total <= 0 ||
      resolvedTrainingModel.parameterCounts.active <= 0
    )
      return null
    return {
      sparsityRatio:
        resolvedTrainingModel.parameterCounts.active /
        resolvedTrainingModel.parameterCounts.total,
      efficiencyGain:
        resolvedTrainingModel.parameterCounts.total /
        resolvedTrainingModel.parameterCounts.active,
      loadBalanceFactor: resolvedTrainingModel.moe.loadBalanceFactor,
    }
  }, [resolvedTrainingModel])

  const pretrainingWarnings = useMemo((): Warning[] => {
    const inputW = generateInputWarnings(
      resolvedTrainingConfig,
      resolvedTrainingModel.architecture,
      resolvedTrainingModel.moe,
      resolvedTrainingModel.parameterCounts.total,
      parallelismRecommendation.config,
      numGPUs,
      chinchillaAnalysis.ratio,
    )
    if (gpuCountDerivedFromTarget && trainingConfig.hardware.targetTrainingDays !== null) {
      inputW.unshift({
        severity: "info",
        category: "hardware",
        message: `Using ${numGPUs.toLocaleString()} GPUs to target roughly ${trainingConfig.hardware.targetTrainingDays.toFixed(1)} training days.`,
      })
    }
    const memW: Warning[] = []
    if (!memoryBreakdown.fits) {
      memW.push({
        severity: "critical",
        category: "memory",
        message: `Per-GPU memory (${(memoryBreakdown.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memoryBreakdown.usableCapacity / 1e9).toFixed(1)} GB).`,
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
    const effectiveNumGPUs = resolveExplicitNumGPUs(
      effectiveConfig.hardware.numGPUs,
    )
    if (effectiveNumGPUs >= 16000) {
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
    effectiveConfig.hardware.numGPUs,
    numGPUs,
    chinchillaAnalysis.ratio,
    memoryBreakdown,
    effectiveComputeEstimate,
    trainingTime.failureMultiplier,
    trainingTime.theoreticalDays,
    gpuCountDerivedFromTarget,
    trainingConfig.hardware.targetTrainingDays,
    trainingConfig.parallelismMode,
    trainingConfig.parallelism.N_ep,
    effectiveConfig.parallelism.N_pp,
  ])

  const pretrainingOutput = useMemo(
    (): PretrainingOutput => ({
      parameterCounts: resolvedTrainingModel.parameterCounts,
      computeEstimate: effectiveComputeEstimate,
      chinchilla: chinchillaAnalysis,
      memory: memoryBreakdown,
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
      resolvedTrainingModel,
      effectiveComputeEstimate,
      chinchillaAnalysis,
      memoryBreakdown,
      parallelismRecommendation,
      trainingTime,
      costEstimate,
      globalBatchSize,
      maxMicroBatchSize,
      dataRepetition,
      moeSparsity,
      batchEfficiency,
      pretrainingWarnings,
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
    const ptGPUs = resolveExplicitNumGPUs(cfg.hardware.numGPUs)

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
    const parallelWorkItems = getPostTrainingParallelWorkItems(cfg)
    const effectiveComputeGPUs = Math.min(
      ptGPUs,
      Math.max(1, Math.ceil(parallelWorkItems)),
    )
    const fPeak =
      getEffectiveTrainingTFLOPS(
        gpu,
        cfg.precision,
        DEFAULT_TRAINING_CONFIG.fp8,
      ) * 1e12
    const mfu = getDefaultMFU(computeParams, effectiveComputeGPUs) * 0.85
    const denom = effectiveComputeGPUs * fPeak * mfu
    const generationTokens =
      cfg.method === "ppo" || cfg.method === "grpo" ? compute.totalTokens : 0
    const generationFLOPs =
      Number.isFinite(computeParams) && Number.isFinite(generationTokens)
        ? 2 * computeParams * generationTokens
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
    const nonGenerationSeconds =
      denom > 0
        ? nonGenerationFLOPs / denom
        : Number.POSITIVE_INFINITY
    const qloraPenaltySeconds =
      cfg.approach === "qlora"
        ? calculateQLoRAPenaltySeconds(
            estimateQLoRAAffectedNonGenerationFLOPs(
              cfg,
              computeParams,
              compute.totalTokens,
            ),
            nonGenerationFLOPs,
            denom,
          )
        : 0
    const generationSecondsWithQLoRAPenalty =
      cfg.approach === "qlora" &&
      (cfg.method === "ppo" || cfg.method === "grpo")
        ? generationSeconds * QLORA_THROUGHPUT_PENALTY
        : generationSeconds
    const theoSec =
      nonGenerationSeconds +
      qloraPenaltySeconds +
      generationSecondsWithQLoRAPenalty

    const totalTokens = compute.totalTokens
    const datasetSizeExamples = getFinitePositiveOrNull(cfg.datasetSizeExamples)
    const epochs = getFinitePositiveOrNull(cfg.epochs)
    const batchSize = getFinitePositiveOrNull(cfg.batchSize)
    const totalSteps =
      datasetSizeExamples !== null && epochs !== null && batchSize !== null
        ? Math.max(1, Math.ceil((datasetSizeExamples * epochs) / batchSize))
        : 0

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
        totalSteps > 0 && Number.isFinite(theoSec) ? theoSec / totalSteps : 0,
    }

    const postTrainingCostPerGPUHour =
      Number.isFinite(cfg.costPerGPUHour) && cfg.costPerGPUHour >= 0
        ? cfg.costPerGPUHour
        : null
    const computeCost =
      postTrainingCostPerGPUHour === null
        ? Number.POSITIVE_INFINITY
        : postTrainingCostPerGPUHour === 0
          ? 0
          : Number.isFinite(theoSec)
            ? ptGPUs * (theoSec / 3600) * postTrainingCostPerGPUHour
            : Number.POSITIVE_INFINITY
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
    }

    const requiredGpuEstimate = estimatePostTrainingRequiredGPUs(cfg)

    const warnings: Warning[] = []
    addPrecisionSupportWarnings(warnings, cfg.precision, gpu)
    addPostTrainingInputWarnings(warnings, cfg)
    if (effectiveComputeGPUs < ptGPUs) {
      warnings.push({
        severity: "warning",
        category: "compute",
        message: `Configured ${ptGPUs.toLocaleString()} GPUs, but batch/method parallelism exposes about ${effectiveComputeGPUs.toLocaleString()} independent training item${effectiveComputeGPUs === 1 ? "" : "s"} per step. Time scaling is capped at ${effectiveComputeGPUs.toLocaleString()} effective GPU${effectiveComputeGPUs === 1 ? "" : "s"}.`,
      })
    }
    if (!memory.fits) {
      warnings.push({
        severity: "critical",
        category: "memory",
        message:
          requiredGpuEstimate.numGPUsNeeded !== null
            ? `Per-GPU memory (${(memory.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memory.usableCapacity / 1e9).toFixed(1)} GB). Split the global batch over about ${requiredGpuEstimate.numGPUsNeeded.toLocaleString()} data-parallel GPUs to fit.`
            : requiredGpuEstimate.stateFloorBytes > memory.usableCapacity
              ? `Per-GPU memory (${(memory.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memory.usableCapacity / 1e9).toFixed(1)} GB). Replicated model states and framework overhead alone require ${(requiredGpuEstimate.stateFloorBytes / 1e9).toFixed(1)} GB per GPU, so adding data-parallel GPUs will not make this fit without sharding, offload, or a smaller model.`
              : `Per-GPU memory (${(memory.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memory.usableCapacity / 1e9).toFixed(1)} GB). Even after splitting the batch across ${requiredGpuEstimate.maxUsefulGPUs.toLocaleString()} useful data-parallel GPUs, the per-GPU working set remains too large.`,
      })
    }
    if (
      generationFeasibility !== null &&
      generationFeasibility.requestedBatch > generationFeasibility.maxBatch
    ) {
      warnings.push({
        severity: "warning",
        category: "generation",
        message:
          cfg.method === "grpo"
            ? `GRPO requests ${generationFeasibility.requestedBatch.toLocaleString()} concurrent generations (batch ${cfg.batchSize} × group ${cfg.grpo.groupSize}), but estimated generation working-set capacity is about ${Math.max(generationFeasibility.maxBatch, 0).toLocaleString()}. Split generation into roughly ${generationFeasibility.rounds.toLocaleString()} rounds or reduce batch/group size.`
            : `PPO requests ${generationFeasibility.requestedBatch.toLocaleString()} concurrent generations, but estimated generation working-set capacity is about ${Math.max(generationFeasibility.maxBatch, 0).toLocaleString()}. Split generation into roughly ${generationFeasibility.rounds.toLocaleString()} rounds or reduce batch size.`,
      })
    }
    const generationCrossoverBatch = estimateGenerationCrossoverBatch(cfg)
    if (
      generationFeasibility !== null &&
      generationCrossoverBatch !== null &&
      generationFeasibility.requestedBatch > 0 &&
      generationFeasibility.requestedBatch < generationCrossoverBatch * 0.25 &&
      generationFeasibility.maxBatch >
        generationFeasibility.requestedBatch * 1.5
    ) {
      warnings.push({
        severity: "info",
        category: "generation",
        message: `Autoregressive decode is likely memory-bandwidth-bound at ${generationFeasibility.requestedBatch.toLocaleString()} concurrent generation${generationFeasibility.requestedBatch === 1 ? "" : "s"}; the estimated memory/compute crossover on ${gpu.name} is about ${Math.round(generationCrossoverBatch).toLocaleString()}. If rollout quality and memory headroom allow, increasing concurrent generations can improve GPU utilization.`,
      })
    }

    return {
      memory,
      numGPUsNeeded: requiredGpuEstimate.numGPUsNeeded,
      trainingTime: time,
      cost,
      warnings,
    }
  }, [resolvedPostTrainingConfig])

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
      .writeText(JSON.stringify(currentOutput, null, 2))
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
              effectiveNumGPUs={numGPUs}
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
