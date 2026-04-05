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
  getOptimizerProfile,
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
  calculatePostTrainingCompute,
  getEffectiveTrainingTFLOPS,
} from "./formulas/cost"
import {
  calculateVocabPadding,
  recommendParallelism,
  validateTPDivisibility,
  validatePPDivisibility,
  validateWorldSize,
  validateZeroPPCompatibility,
  validateMicrobatches,
  calculatePipelineBubble,
} from "./formulas/parallelism"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function normalizeParallelismConfig(
  parallelism: ParallelismConfig,
): ParallelismConfig {
  if (parallelism.framework !== "fsdp") {
    return {
      ...parallelism,
      fsdpStrategy: null,
    }
  }

  const fsdpStrategy = parallelism.fsdpStrategy ?? "FULL_SHARD"

  return {
    ...parallelism,
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

function resolveExpertFFNWidth(arch: ModelArchitecture, moe: MoEConfig): number {
  if (moe.expertIntermediateSize != null) return moe.expertIntermediateSize
  return resolveFFNWidth(arch, moe)
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

function estimateMaxMicroBatch(
  memory: MemoryBreakdown,
  currentBatch: number,
): number {
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
    targetTotal !== null && counts.total > 0 ? targetTotal / counts.total : 1
  const activeScale =
    targetActive !== null && counts.active > 0
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
  const moe = preset?.moe ?? config.model.moe
  const rawCounts = calculateParameterCount(architecture, moe, config.sequenceLength)

  if (config.model.inputMode === "quick") {
    return {
      architecture,
      moe,
      parameterCounts: scaleParameterCounts(
        rawCounts,
        config.model.quickMode.totalParameters,
        config.model.quickMode.totalParameters,
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
        preset.activeParameterCount ?? preset.parameterCount,
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
  if (config.baseModel.inputMode === "preset") {
    const preset =
      MODEL_PRESETS.find((candidate) => candidate.id === config.baseModel.presetId) ??
      null

    if (!preset) {
      return config
    }

    return {
      ...config,
      baseModel: {
        ...config.baseModel,
        parameterCount: preset.parameterCount,
        architecture: preset.architecture,
        moe: preset.moe ?? config.baseModel.moe,
      },
    }
  }

  return {
    ...config,
    baseModel: {
      ...config.baseModel,
      architecture: estimateParametersQuick(config.baseModel.parameterCount),
      moe: {
        ...config.baseModel.moe,
        enabled: false,
      },
    },
  }
}

function resolveRequestedNumGPUs(
  config: TrainingConfig,
  totalFLOPs: number,
  activeParams: number,
): number {
  const explicitNumGPUs = resolveExplicitNumGPUs(config.hardware.numGPUs)
  const targetDays = config.hardware.targetTrainingDays

  if (
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
    const mfu = config.mfuOverride ?? getDefaultMFU(activeParams, guess)
    const next = Math.max(
      1,
      Math.ceil(totalFLOPs / Math.max(secondsBudget * fPeakFLOPS * mfu, 1)),
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

function resolveTrainableParameterCount(config: PostTrainingConfig): number {
  const ratio = Math.max(
    0,
    Math.min(config.trainableParameterPercentage ?? 100, 100),
  ) / 100
  return config.baseModel.parameterCount * ratio
}

// ── Post-training memory dispatchers ──

function calculateSFTFullMemory(
  config: PostTrainingConfig,
): PostTrainingMemoryBreakdown {
  const optimizer = getOptimizerProfile(
    config.optimizer,
    config.gradientPrecision,
  )
  const totalParamCount = config.baseModel.parameterCount
  const trainableParamCount = resolveTrainableParameterCount(config)
  const frozenParamCount = Math.max(totalParamCount - trainableParamCount, 0)
  const parameters = totalParamCount * optimizer.parameterBytes
  const gradients = trainableParamCount * optimizer.betaGrad
  const optimizerStates = trainableParamCount * optimizer.kOpt
  const actBytes = config.precision === "fp32" ? 4 : 2
  const activations =
    config.baseModel.architecture.L *
    config.sequenceLength *
    config.batchSize *
    config.baseModel.architecture.d *
    actBytes
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
  const activations =
    config.baseModel.architecture.L *
    config.sequenceLength *
    config.batchSize *
    config.baseModel.architecture.d *
    wb
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
  const uniqueTokenRatio =
    config.totalTokens > 0 && config.uniqueTokens > 0
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
  if (config.totalTokens <= 0)
    w.push({
      severity: "critical",
      category: "data",
      message: "Total training tokens must be positive.",
    })
  if (config.uniqueTokens <= 0)
    w.push({
      severity: "critical",
      category: "data",
      message: "Unique token count must be positive.",
    })
  if (config.uniqueTokens > config.totalTokens)
    w.push({
      severity: "critical",
      category: "data",
      message: "Unique tokens U must be less than or equal to total tokens D.",
    })
  if (chinchillaRatio > 0 && chinchillaRatio < 1)
    w.push({
      severity: "warning",
      category: "data",
      message:
        "Token count is below 1x Chinchilla optimal — model will be severely undertrained.",
    })
  if (chinchillaRatio > 5000)
    w.push({
      severity: "critical",
      category: "data",
      message:
        "Extreme overtraining (>5000x Chinchilla). Standard scaling law coefficients are not calibrated for this regime.",
    })
  else if (chinchillaRatio > 500)
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
  if (config.microBatchSize < 1)
    w.push({
      severity: "critical",
      category: "compute",
      message: "Micro-batch size must be at least 1.",
    })
  if (config.sequenceLength <= 0)
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
  if (numGPUs < 1)
    w.push({
      severity: "critical",
      category: "hardware",
      message: "GPU count must be at least 1.",
    })
  if (numGPUs > 100000)
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
  if (config.precision === "bf16" && !config.hardware.gpu.supportsBF16)
    w.push({
      severity: "warning",
      category: "precision",
      message: `${config.hardware.gpu.name} does not support BF16. Use FP16 with loss scaling.`,
    })
  if (config.precision === "fp8" && !config.hardware.gpu.supportsFP8)
    w.push({
      severity: "warning",
      category: "precision",
      message: `${config.hardware.gpu.name} does not support FP8 kernels. Estimates assume BF16-class throughput instead.`,
    })
  if (config.microBatchSize <= 2)
    w.push({
      severity: "info",
      category: "compute",
      message:
        "Micro-batch size ≤ 2 may significantly reduce MFU due to memory-bandwidth-bound matmuls.",
    })

  // Manual parallelism validation
  if (config.parallelismMode === "manual") {
    const dff = resolveFFNWidth(architecture, moe)
    const expertDff = resolveExpertFFNWidth(architecture, moe)
    const tp = validateTPDivisibility(
      parallelism.N_tp,
      architecture.a,
      architecture.a_kv,
      dff,
    )
    if (!tp.valid)
      w.push({ severity: "critical", category: "parallelism", message: tp.message })
    if (
      moe.enabled &&
      parallelism.N_tp > 1 &&
      expertDff % parallelism.N_tp !== 0
    )
      w.push({
        severity: "critical",
        category: "parallelism",
        message: `N_tp=${parallelism.N_tp} does not evenly divide expert d_ff=${expertDff}.`,
      })
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
    if (usesAFABSchedule(parallelism, config.gradientAccumulationSteps)) {
      w.push({
        severity: "info",
        category: "parallelism",
        message:
          "FSDP SHARD_GRAD_OP + PP will fall back to the AFAB schedule here. This relaxes the 1F1B microbatch minimum but increases activation residency.",
      })
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
      parallelism.N_tp * parallelism.N_ep > config.hardware.gpu.gpusPerNode
    )
      w.push({
        severity: "critical",
        category: "parallelism",
        message: `N_tp × N_ep must stay within the per-node GPU group (${config.hardware.gpu.gpusPerNode}) for expert traffic.`,
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
      parallelism.VP,
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
    if (config.cpuOffload !== "none")
      w.push({
        severity: "warning",
        category: "memory",
        message:
          "CPU offloading reduces GPU memory pressure but will slow training because optimizer or parameter traffic shifts onto the host interconnect.",
      })
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

function fmtDuration(hours: number): string {
  if (!Number.isFinite(hours) || hours < 0) return "--"
  if (hours >= 24) return `${(hours / 24).toFixed(1)} days`
  return `${hours.toFixed(1)} hours`
}

function generatePretrainingMarkdown(o: PretrainingOutput): string {
  const hasActive = o.parameterCounts.active !== o.parameterCounts.total
  return [
    "# GPU Calculator — Pretraining Results\n",
    "## Model",
    `- Parameters: ${fmtCount(o.parameterCounts.total)}${hasActive ? ` total, ${fmtCount(o.parameterCounts.active)} active` : ""}`,
    "",
    "## Compute",
    `- Total FLOPs: ${(o.computeEstimate.totalFLOPs / 1e21).toFixed(2)} ZFLOPs`,
    `- Chinchilla Ratio: ${o.chinchilla.ratio.toFixed(2)}x`,
    `- Predicted Loss: ${Number.isFinite(o.predictedLossNats) ? o.predictedLossNats.toFixed(3) : "--"} nats`,
    `- Attention Overhead: ${(o.attentionOverheadFraction * 100).toFixed(1)}%`,
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
    `- Pipeline Bubble: ${(o.pipelineBubbleFraction * 100).toFixed(1)}%`,
    `- Minimum GPUs: ${fmtCount(o.minGPUsNeeded)}`,
    "",
    "## Training Time",
    `- Theoretical: ${fmtDuration(o.trainingTime.theoreticalHours)}`,
    o.trainingTime.failureAdjustedHours != null
      ? `- Failure-Adjusted: ${fmtDuration(o.trainingTime.failureAdjustedHours)}`
      : null,
    `- Throughput: ${fmtCount(o.tokensPerSecond)} tok/s`,
    `- Steps: ${o.trainingTime.totalSteps.toLocaleString()}`,
    "",
    "## Cost",
    `- Compute: $${Number.isFinite(o.cost.computeCost) ? Math.round(o.cost.computeCost).toLocaleString() : "--"}`,
    `- Storage: $${Number.isFinite(o.cost.storageCost) ? o.cost.storageCost.toFixed(2) : "--"}`,
    `- Total: $${Number.isFinite(o.cost.totalCost) ? Math.round(o.cost.totalCost).toLocaleString() : "--"}`,
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
    `- GPUs Needed: ${o.numGPUsNeeded}`,
    "",
    "## Training Time",
    `- Estimated: ${fmtDuration(o.trainingTime.theoreticalHours)}`,
    `- Throughput: ${fmtCount(o.trainingTime.tokensPerSecond)} tok/s`,
    "",
    "## Cost",
    `- Total: $${Number.isFinite(o.cost.totalCost) ? Math.round(o.cost.totalCost).toLocaleString() : "--"}`,
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

  const rawComputeEstimate = useMemo(
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

  // Training time uses the simplified 6ΨD formula (model FLOPs only) because
  // MFU defaults are calibrated against it — matching spec Section 15 test cases.
  // The full PaLM attention term (12Lds) is kept in attentionOverheadFraction
  // for display but excluded from totalFLOPs used in the time formula.
  const computeEstimate = useMemo(() => {
    if (rawComputeEstimate.attentionOverheadFraction <= 0) {
      return rawComputeEstimate
    }

    const modelFlopsPerToken =
      rawComputeEstimate.flopsPerToken /
      (1 + rawComputeEstimate.attentionOverheadFraction)

    return {
      ...rawComputeEstimate,
      flopsPerToken: modelFlopsPerToken,
      totalFLOPs: modelFlopsPerToken * trainingConfig.totalTokens,
    }
  }, [rawComputeEstimate, trainingConfig.totalTokens])

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
    trainingConfig.hardware.targetTrainingDays !== null &&
    Number.isFinite(trainingConfig.hardware.targetTrainingDays) &&
    trainingConfig.hardware.targetTrainingDays > 0

  const normalizedTrainingParallelism = useMemo(
    () => normalizeParallelismConfig(trainingConfig.parallelism),
    [trainingConfig.parallelism],
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
        resolvedTrainingModel.parameterCounts.active,
        trainingConfig.totalTokens,
        trainingConfig.uniqueTokens,
      ),
    [
      resolvedTrainingModel.parameterCounts.active,
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
      p.VP,
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

    const configForFloor: TrainingConfig = {
      ...resolvedTrainingConfig,
      parallelism: p,
    }
    const minVRAMFloor = calculateMinGPUVRAMFloor(
      resolvedTrainingModel.parameterCounts,
      configForFloor,
    )

    return {
      config: p,
      minGPUs: numGPUs,
      minVRAMFloor,
      pipelineBubbleFraction: bubble,
      strategyLabel: parts.join(", "),
      reasoning: ["Manual parallelism configuration."],
      warnings: [],
    }
  }, [resolvedTrainingConfig, resolvedTrainingModel, numGPUs])

  const effectiveConfig = useMemo(
    (): TrainingConfig => ({
      ...resolvedTrainingConfig,
      parallelism: parallelismRecommendation.config,
      hardware: {
        ...resolvedTrainingConfig.hardware,
        numGPUs:
          resolvedTrainingConfig.parallelismMode === "auto" &&
          getParallelWorldSize(parallelismRecommendation.config) > numGPUs
            ? getParallelWorldSize(parallelismRecommendation.config)
            : numGPUs,
      },
    }),
    [resolvedTrainingConfig, parallelismRecommendation.config, numGPUs],
  )

  const paddedParameterCounts = useMemo(
    () =>
      applyVocabPaddingToCounts(
        resolvedTrainingModel.parameterCounts,
        resolvedTrainingModel.architecture,
        parallelismRecommendation.config.N_tp,
      ),
    [resolvedTrainingModel, parallelismRecommendation.config.N_tp],
  )

  const memoryBreakdown = useMemo(
    () =>
      calculateTotalMemoryPerGPU(
        paddedParameterCounts,
        effectiveConfig,
        resolvedTrainingModel.architecture,
        resolvedTrainingModel.moe,
        effectiveConfig.hardware.gpu,
      ),
    [
      paddedParameterCounts,
      effectiveConfig,
      resolvedTrainingModel.architecture,
      resolvedTrainingModel.moe,
    ],
  )

  const trainingTime = useMemo(
    () =>
      calculateTrainingTime(
        computeEstimate,
        effectiveConfig,
        resolvedTrainingModel.parameterCounts.active,
      ),
    [computeEstimate, effectiveConfig, resolvedTrainingModel.parameterCounts.active],
  )

  const costEstimate = useMemo(
    () =>
      calculateCost(
        trainingTime,
        effectiveConfig,
        resolvedTrainingModel.parameterCounts.total,
      ),
    [trainingTime, effectiveConfig, resolvedTrainingModel.parameterCounts.total],
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
      estimateMaxMicroBatch(memoryBreakdown, trainingConfig.microBatchSize),
    [memoryBreakdown, trainingConfig.microBatchSize],
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
    return [...memW, ...inputW, ...parallelismRecommendation.warnings]
  }, [
    resolvedTrainingConfig,
    resolvedTrainingModel,
    parallelismRecommendation,
    numGPUs,
    chinchillaAnalysis.ratio,
    memoryBreakdown,
    gpuCountDerivedFromTarget,
    trainingConfig.hardware.targetTrainingDays,
  ])

  const pretrainingOutput = useMemo(
    (): PretrainingOutput => ({
      parameterCounts: resolvedTrainingModel.parameterCounts,
      computeEstimate,
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
        computeEstimate.attentionOverheadFraction,
      predictedLossNats: chinchillaAnalysis.predictedLossNats,
      maxMicroBatchSize,
      dataRepetition,
      moeSparsity,
      batchEfficiency,
      warnings: pretrainingWarnings,
    }),
    [
      resolvedTrainingModel,
      computeEstimate,
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
    const ptGPUs = Math.max(1, cfg.hardware.numGPUs)

    const memory = getPostTrainingMemory(cfg)

    // Compute + time
    const compute = calculatePostTrainingCompute(
      cfg.method,
      cfg.baseModel.parameterCount,
      cfg,
    )
    const fPeak = gpu.halfPrecisionTFLOPS * 1e12
    const mfu =
      getDefaultMFU(cfg.baseModel.parameterCount, ptGPUs) * 0.85
    const denom = ptGPUs * fPeak * mfu
    let theoSec =
      denom > 0
        ? compute.totalFLOPs / denom
        : Number.POSITIVE_INFINITY
    if (cfg.approach === "qlora") theoSec *= 1.75

    const totalTokens =
      cfg.datasetSizeExamples * cfg.epochs * cfg.sequenceLength
    const totalSteps = Math.max(
      1,
      Math.ceil(totalTokens / (cfg.batchSize * cfg.sequenceLength)),
    )

    const time: TrainingTimeEstimate = {
      theoreticalDays: theoSec / 86400,
      theoreticalHours: theoSec / 3600,
      failureAdjustedDays: null,
      failureAdjustedHours: null,
      failureMultiplier: null,
      tokensPerSecond:
        Number.isFinite(theoSec) && theoSec > 0
          ? totalTokens / theoSec
          : 0,
      totalSteps,
      secondsPerStep:
        Number.isFinite(theoSec) ? theoSec / totalSteps : 0,
    }

    const computeCost = ptGPUs * (theoSec / 3600) * cfg.costPerGPUHour
    const cost: CostEstimate = {
      computeCost: Number.isFinite(computeCost) ? computeCost : 0,
      actualComputeCost: Number.isFinite(computeCost) ? computeCost : 0,
      storageCost: 0,
      failureOverheadCost: 0,
      totalCost: Number.isFinite(computeCost) ? computeCost : 0,
      checkpointSize: 0,
      numCheckpoints: 0,
      peakCheckpointStorage: 0,
      averageCheckpointStorage: 0,
    }

    const stateOnlyBytes =
      memory.parameters + memory.gradients + memory.optimizerStates
    const numGPUsNeeded = Math.max(
      1,
      Math.ceil(stateOnlyBytes / Math.max(memory.usableCapacity, 1)),
    )
    const workingSetGPUsNeeded = Math.max(
      1,
      Math.ceil(memory.total / Math.max(memory.usableCapacity, 1)),
    )

    const warnings: Warning[] = []
    if (!memory.fits) {
      warnings.push({
        severity: "critical",
        category: "memory",
        message:
          workingSetGPUsNeeded > numGPUsNeeded
            ? `Per-GPU memory (${(memory.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memory.usableCapacity / 1e9).toFixed(1)} GB). Model states alone imply at least ${numGPUsNeeded} GPUs, but the current working set points closer to ${workingSetGPUsNeeded}.`
            : `Per-GPU memory (${(memory.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memory.usableCapacity / 1e9).toFixed(1)} GB).`,
      })
    }

    return { memory, numGPUsNeeded, trainingTime: time, cost, warnings }
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
