"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
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
import type { CalculatorColors } from "./components/input-controls"
import { PretrainingPanel } from "./components/pretraining-panel"
import { PostTrainingPanel } from "./components/post-training-panel"
import ResultsSummary from "./components/results-summary"
import {
  calculateParameterCount,
  calculateFLOPs,
  calculateChinchillaAnalysis,
  calculateCriticalBatchSize,
  analyzeDataRepetition,
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
} from "./formulas/cost"
import {
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

function resolveNumGPUs(config: TrainingConfig): number {
  const n = config.hardware.numGPUs
  return typeof n === "number" && Number.isFinite(n) && n > 0 ? n : 1
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

// ── Post-training memory dispatchers ──

function calculateSFTFullMemory(
  config: PostTrainingConfig,
): PostTrainingMemoryBreakdown {
  const optimizer = getOptimizerProfile(
    config.optimizer,
    config.gradientPrecision,
  )
  const paramCount = config.baseModel.parameterCount
  const parameters = paramCount * optimizer.parameterBytes
  const gradients = paramCount * optimizer.betaGrad
  const optimizerStates = paramCount * optimizer.kOpt
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
    trainableModels: parameters + gradients + optimizerStates,
    frozenModels: 0,
    loraAdapter: 0,
    ppoBuffers: 0,
    items: [
      { label: "Model parameters", category: "trainable", bytes: parameters },
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
  const parameters = config.baseModel.parameterCount * wb
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
    trainableModels: parameters,
    frozenModels: 0,
    loraAdapter: 0,
    ppoBuffers: 0,
    items: [
      { label: "Model parameters", category: "trainable", bytes: parameters },
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
  totalParams: number,
  parallelism: ParallelismConfig,
  numGPUs: number,
  chinchillaRatio: number,
): Warning[] {
  const w: Warning[] = []
  const arch = config.model.architecture
  const moe = config.model.moe

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
  if (chinchillaRatio > 0 && chinchillaRatio < 1)
    w.push({
      severity: "warning",
      category: "data",
      message:
        "Token count is below 1x Chinchilla optimal — model will be severely undertrained.",
    })
  if (chinchillaRatio > 500)
    w.push({
      severity: "warning",
      category: "data",
      message:
        "Far beyond Chinchilla optimal (>500x). Loss predictions become less reliable.",
    })
  if (chinchillaRatio > 5000)
    w.push({
      severity: "critical",
      category: "data",
      message:
        "Extreme overtraining (>5000x Chinchilla). Standard scaling law coefficients are not calibrated for this regime.",
    })
  if (
    config.uniqueTokens < config.totalTokens &&
    config.totalTokens / config.uniqueTokens > 40
  )
    w.push({
      severity: "warning",
      category: "data",
      message: `Training for ${(config.totalTokens / config.uniqueTokens).toFixed(0)} epochs — past the effective data ceiling of ~16x unique data.`,
    })
  if (numGPUs > 100000)
    w.push({
      severity: "warning",
      category: "hardware",
      message: "GPU count exceeds 100,000.",
    })
  if (config.precision === "bf16" && !config.hardware.gpu.supportsBF16)
    w.push({
      severity: "warning",
      category: "precision",
      message: `${config.hardware.gpu.name} does not support BF16. Use FP16 with loss scaling.`,
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
    const dff = resolveFFNWidth(arch, moe)
    const tp = validateTPDivisibility(
      parallelism.N_tp,
      arch.a,
      arch.a_kv,
      dff,
    )
    if (!tp.valid)
      w.push({ severity: "critical", category: "parallelism", message: tp.message })
    const pp = validatePPDivisibility(parallelism.N_pp, arch.L)
    if (!pp.valid)
      w.push({ severity: "critical", category: "parallelism", message: pp.message })
    const ws = validateWorldSize(parallelism, numGPUs)
    if (!ws.valid)
      w.push({ severity: "critical", category: "parallelism", message: ws.message })
    const zp = validateZeroPPCompatibility(
      parallelism.zeroStage,
      parallelism.N_pp,
      parallelism.framework,
    )
    if (!zp.valid)
      w.push({ severity: "critical", category: "parallelism", message: zp.message })
    const mb = validateMicrobatches(
      config.gradientAccumulationSteps,
      parallelism.N_pp,
      parallelism.VP,
    )
    if (!mb.valid)
      w.push({ severity: "warning", category: "parallelism", message: mb.message })
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
    `- Minimum GPUs: ${o.minGPUsNeeded}`,
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
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    setMounted(true)
  }, [])
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
      bg: isDark ? "#1a1a2e" : "#f8f9fa",
      cardBg: isDark ? "#16213e" : "#ffffff",
      text: isDark ? "#e0e0e0" : "#1a1a2e",
      textSecondary: isDark ? "#a0a0b0" : "#5d6676",
      border: isDark ? "#2a2a4a" : "#dde3ec",
      accent: isDark ? "#83b6ff" : "#1d5fe4",
      accentMuted: isDark
        ? "rgba(131, 182, 255, 0.14)"
        : "rgba(29, 95, 228, 0.08)",
      panel: isDark
        ? "rgba(13, 18, 37, 0.72)"
        : "rgba(245, 247, 250, 0.92)",
      warning: isDark ? "#ffda6a" : "#664d03",
      warningBg: isDark
        ? "rgba(102, 77, 3, 0.15)"
        : "rgba(255, 193, 7, 0.1)",
      warningBorder: isDark
        ? "rgba(255, 218, 106, 0.25)"
        : "rgba(255, 193, 7, 0.4)",
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

  const numGPUs = useMemo(
    () => resolveNumGPUs(trainingConfig),
    [trainingConfig],
  )

  const parameterCounts = useMemo(
    () =>
      calculateParameterCount(
        trainingConfig.model.architecture,
        trainingConfig.model.moe,
        trainingConfig.sequenceLength,
      ),
    [
      trainingConfig.model.architecture,
      trainingConfig.model.moe,
      trainingConfig.sequenceLength,
    ],
  )

  const computeEstimate = useMemo(
    () =>
      calculateFLOPs(
        parameterCounts,
        {
          totalTokens: trainingConfig.totalTokens,
          sequenceLength: trainingConfig.sequenceLength,
        },
        trainingConfig.model.architecture,
        trainingConfig.model.moe,
      ),
    [
      parameterCounts,
      trainingConfig.totalTokens,
      trainingConfig.sequenceLength,
      trainingConfig.model.architecture,
      trainingConfig.model.moe,
    ],
  )

  const chinchillaAnalysis = useMemo(
    () =>
      calculateChinchillaAnalysis(
        parameterCounts.active,
        trainingConfig.totalTokens,
        trainingConfig.uniqueTokens,
      ),
    [
      parameterCounts.active,
      trainingConfig.totalTokens,
      trainingConfig.uniqueTokens,
    ],
  )

  const parallelismRecommendation = useMemo((): ParallelismRecommendation => {
    const arch = trainingConfig.model.architecture
    const moe = trainingConfig.model.moe
    const gpu = trainingConfig.hardware.gpu

    if (trainingConfig.parallelismMode === "auto") {
      return recommendParallelism(
        parameterCounts,
        arch,
        trainingConfig,
        gpu,
        numGPUs,
        moe,
      )
    }

    // Manual mode — wrap user config in a ParallelismRecommendation
    const p = trainingConfig.parallelism
    const bubble = calculatePipelineBubble(
      p.N_pp,
      trainingConfig.gradientAccumulationSteps,
      p.VP,
    )
    const moeEnabled = moe.enabled && moe.E > 0
    const parts: string[] = [`DP=${p.N_dp}`]
    if (p.N_tp > 1) parts.push(`TP=${p.N_tp}`)
    if (moeEnabled && p.N_ep > 1) parts.push(`EP=${p.N_ep}`)
    if (p.N_pp > 1) parts.push(`PP=${p.N_pp}`)
    if (p.N_cp > 1) parts.push(`CP=${p.N_cp}`)
    parts.push(`ZeRO-${p.zeroStage}`)

    const configForFloor: TrainingConfig = {
      ...trainingConfig,
      parallelism: p,
    }
    const minVRAMFloor = calculateMinGPUVRAMFloor(
      parameterCounts,
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
  }, [parameterCounts, trainingConfig, numGPUs])

  const effectiveConfig = useMemo(
    (): TrainingConfig => ({
      ...trainingConfig,
      parallelism: parallelismRecommendation.config,
    }),
    [trainingConfig, parallelismRecommendation.config],
  )

  const memoryBreakdown = useMemo(
    () =>
      calculateTotalMemoryPerGPU(
        parameterCounts,
        effectiveConfig,
        trainingConfig.model.architecture,
        trainingConfig.model.moe,
        trainingConfig.hardware.gpu,
      ),
    [
      parameterCounts,
      effectiveConfig,
      trainingConfig.model.architecture,
      trainingConfig.model.moe,
      trainingConfig.hardware.gpu,
    ],
  )

  const trainingTime = useMemo(
    () =>
      calculateTrainingTime(
        computeEstimate,
        effectiveConfig,
        parameterCounts.active,
      ),
    [computeEstimate, effectiveConfig, parameterCounts.active],
  )

  const costEstimate = useMemo(
    () =>
      calculateCost(trainingTime, effectiveConfig, parameterCounts.total),
    [trainingTime, effectiveConfig, parameterCounts.total],
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
      !trainingConfig.model.moe.enabled ||
      parameterCounts.total <= 0 ||
      parameterCounts.active <= 0
    )
      return null
    return {
      sparsityRatio: parameterCounts.active / parameterCounts.total,
      efficiencyGain: parameterCounts.total / parameterCounts.active,
      loadBalanceFactor: trainingConfig.model.moe.loadBalanceFactor,
    }
  }, [trainingConfig.model.moe, parameterCounts])

  const pretrainingWarnings = useMemo((): Warning[] => {
    const inputW = generateInputWarnings(
      trainingConfig,
      parameterCounts.total,
      parallelismRecommendation.config,
      numGPUs,
      chinchillaAnalysis.ratio,
    )
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
    trainingConfig,
    parameterCounts.total,
    parallelismRecommendation,
    numGPUs,
    chinchillaAnalysis.ratio,
    memoryBreakdown,
  ])

  const pretrainingOutput = useMemo(
    (): PretrainingOutput => ({
      parameterCounts,
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
      parameterCounts,
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

  const postTrainingOutput = useMemo((): PostTrainingOutput => {
    const cfg = postTrainingConfig
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

    const numGPUsNeeded = memory.fits
      ? ptGPUs
      : Math.max(
          1,
          Math.ceil(memory.total / Math.max(memory.usableCapacity, 1)),
        )

    const warnings: Warning[] = []
    if (!memory.fits) {
      warnings.push({
        severity: "critical",
        category: "memory",
        message: `Per-GPU memory (${(memory.total / 1e9).toFixed(1)} GB) exceeds usable capacity (${(memory.usableCapacity / 1e9).toFixed(1)} GB).`,
      })
    }

    return { memory, numGPUsNeeded, trainingTime: time, cost, warnings }
  }, [postTrainingConfig])

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
      {/* ── Header ── */}
      <div
        className="border-b px-6 py-6 sm:px-8"
        style={{
          borderColor: colors.border,
          background: `linear-gradient(135deg, ${colors.accentMuted}, transparent 65%)`,
        }}
      >
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-2xl">
            <p
              className="text-sm font-medium uppercase tracking-[0.24em]"
              style={{ color: colors.accent }}
            >
              GPU Calculator
            </p>
            <h2
              className="mt-3 text-3xl"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Estimate GPU requirements for LLM training
            </h2>
            <p
              className="mt-3 text-sm leading-6"
              style={{ color: colors.textSecondary }}
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
                className="rounded-2xl border px-4 py-3"
                style={{
                  borderColor: colors.border,
                  backgroundColor: colors.panel,
                }}
              >
                <div
                  className="flex items-center gap-2 text-xs uppercase tracking-[0.22em]"
                  style={{ color: colors.textSecondary }}
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </div>
                <div className="mt-3 text-2xl font-semibold">{value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Tabs ── */}
      <div
        className="border-b px-4 py-4 sm:px-6"
        style={{ borderColor: colors.border }}
      >
        <div className="flex flex-wrap gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className="rounded-full px-4 py-2 text-sm font-medium transition-colors"
              style={{
                color:
                  activeTab === tab.key
                    ? colors.accent
                    : colors.textSecondary,
                backgroundColor:
                  activeTab === tab.key ? colors.accentMuted : "transparent",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <p
          className="mt-3 text-sm"
          style={{ color: colors.textSecondary }}
        >
          {tabs.find((tab) => tab.key === activeTab)?.description}
        </p>
      </div>

      {/* ── Main grid: inputs | results ── */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.24, ease: "easeOut" }}
        className="grid gap-4 p-4 sm:p-6 lg:grid-cols-[1.1fr_0.9fr]"
      >
        {/* ── Input panel ── */}
        <section
          className="rounded-[1.5rem] border p-5 lg:max-h-[80vh] lg:overflow-y-auto"
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
          className="rounded-[1.5rem] border p-5 lg:max-h-[80vh] lg:overflow-y-auto"
          style={{
            borderColor: colors.border,
            backgroundColor: colors.panel,
          }}
        >
          {/* Export buttons */}
          <div className="mb-4 flex items-center justify-between">
            <div
              className="flex items-center gap-2 text-sm font-medium"
              style={{ color: colors.accent }}
            >
              <Gauge className="h-4 w-4" />
              Results
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleCopyText}
                className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors hover:opacity-80"
                style={{
                  borderColor: colors.border,
                  color: copied === "text" ? colors.accent : colors.textSecondary,
                }}
              >
                {copied === "text" ? (
                  <Check className="h-3.5 w-3.5" />
                ) : (
                  <FileText className="h-3.5 w-3.5" />
                )}
                {copied === "text" ? "Copied" : "Copy text"}
              </button>
              <button
                type="button"
                onClick={handleCopyJSON}
                className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors hover:opacity-80"
                style={{
                  borderColor: colors.border,
                  color: copied === "json" ? colors.accent : colors.textSecondary,
                }}
              >
                {copied === "json" ? (
                  <Check className="h-3.5 w-3.5" />
                ) : (
                  <ClipboardCopy className="h-3.5 w-3.5" />
                )}
                {copied === "json" ? "Copied" : "Copy JSON"}
              </button>
            </div>
          </div>

          <ResultsSummary output={currentOutput} isDark={isDark} />
        </section>
      </motion.div>
    </div>
  )
}
