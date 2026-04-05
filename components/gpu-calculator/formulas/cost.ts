import type {
  ComputeEstimate,
  CostEstimate,
  FP8Config,
  FailureModelConfig,
  GPUSpec,
  PostTrainingConfig,
  PostTrainingMethod,
  TrainingConfig,
  TrainingPrecision,
  TrainingTimeEstimate,
} from "../types"
import { MFU_DEFAULTS, OPTIMIZER_PROFILES } from "../constants"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Effective peak TFLOPS for the training-time formula based on precision.
 *
 * - bf16 / fp16 → half-precision tensor-core TFLOPS (the GPU spec's primary
 *   matmul rate).
 * - fp32 → TF32 TFLOPS on Ampere+ GPUs (TF32 is enabled by default in
 *   PyTorch ≥ 1.12). Pre-Ampere / AMD / Apple: approximate as
 *   halfPrecision / 8 (matches V100 FP16-TC 125 / FP32 15.7 ≈ 8×).
 * - fp8 → BF16 TFLOPS × empirical speedup factor (default 1.3×). The raw
 *   FP8 peak (e.g. 1 979 for H100) must NOT be used — doing so would
 *   underestimate training time by ~50 % (spec Section 6.2).
 */
function getEffectiveTFLOPS(
  gpu: GPUSpec,
  precision: TrainingPrecision,
  fp8Config: FP8Config,
): number {
  switch (precision) {
    case "bf16":
    case "fp16":
      return gpu.halfPrecisionTFLOPS
    case "fp32":
      if (gpu.supportsTF32 && gpu.tf32TFLOPS !== null) {
        return gpu.tf32TFLOPS
      }
      return gpu.halfPrecisionTFLOPS / 8
    case "fp8":
      return gpu.halfPrecisionTFLOPS * fp8Config.kernelSpeedupFactor
  }
}

/** Bytes per parameter for weight storage at the given precision. */
function getBytesPerParam(precision: TrainingPrecision): number {
  return precision === "fp32" ? 4 : 2
}

// ---------------------------------------------------------------------------
// getDefaultMFU — Section 6.3
// ---------------------------------------------------------------------------

/**
 * Look up a reasonable default MFU from the spec's guideline table.
 *
 * Matching priority:
 * 1. Both model-size AND GPU-count ranges match → use that row.
 * 2. Only model-size matches → use that row (user may have atypical GPU count).
 * 3. Nothing matches → 0.40 (medium-model midpoint).
 *
 * Advisory-only rows (e.g. "state-of-the-art") are never auto-selected.
 */
export function getDefaultMFU(params: number, numGPUs: number): number {
  // Pass 1: exact match on both dimensions
  for (const entry of MFU_DEFAULTS) {
    if (entry.advisoryOnly) continue
    const paramsOk =
      (entry.minParams === null || params >= entry.minParams) &&
      (entry.maxParams === null || params < entry.maxParams)
    const gpusOk =
      (entry.minGPUs === null || numGPUs >= entry.minGPUs) &&
      (entry.maxGPUs === null || numGPUs < entry.maxGPUs)
    if (paramsOk && gpusOk) return entry.defaultMFU
  }

  // Pass 2: model-size only
  for (const entry of MFU_DEFAULTS) {
    if (entry.advisoryOnly) continue
    const paramsOk =
      (entry.minParams === null || params >= entry.minParams) &&
      (entry.maxParams === null || params < entry.maxParams)
    if (paramsOk) return entry.defaultMFU
  }

  return 0.4
}

// ---------------------------------------------------------------------------
// calculateFailureAdjustedTime — Section 6.5
// ---------------------------------------------------------------------------

/**
 * Closed-form failure-adjusted training time.
 *
 *   T_actual = T_theory / [1 − f × N_inst × (t_recovery + 1/(2 × f_ckpt))]
 *
 * Returns `null` when the denominator ≤ 0, meaning the failure overhead
 * consumes 100 % of available time and training is infeasible at this scale.
 */
export function calculateFailureAdjustedTime(
  theoreticalDays: number,
  numGPUs: number,
  gpusPerNode: number,
  failureModel: FailureModelConfig,
): { adjustedDays: number; multiplier: number } | null {
  const nInstances = Math.ceil(numGPUs / gpusPerNode)
  const recoveryDays = failureModel.recoveryTimeHours / 24
  const avgLostWorkDays = 1 / (2 * failureModel.checkpointFrequencyPerDay)

  const denominator =
    1 -
    failureModel.failureRatePerInstancePerDay *
      nInstances *
      (recoveryDays + avgLostWorkDays)

  if (denominator <= 0) return null

  return {
    adjustedDays: theoreticalDays / denominator,
    multiplier: 1 / denominator,
  }
}

// ---------------------------------------------------------------------------
// calculateTrainingTime — Section 6.1
// ---------------------------------------------------------------------------

/**
 * Core training-time estimate.
 *
 *   T_seconds = C / (N_gpu × F_peak × MFU)
 *
 * CRITICAL: C is the ideal model FLOPs (6ΨD or the PaLM formula from
 * Section 4). It is NOT adjusted for activation recomputation. MFU already
 * captures the throughput loss from recomputation — adjusting C to 8ΨD while
 * also applying MFU would double-count the overhead (spec Section 6.1).
 *
 * @param compute  – FLOPs estimate from the compute module.
 * @param config   – Full pretraining configuration.
 * @param activeParams – Active parameter count (for MFU lookup; use total
 *   params for dense models, active params for MoE).
 */
export function calculateTrainingTime(
  compute: ComputeEstimate,
  config: TrainingConfig,
  activeParams: number,
): TrainingTimeEstimate {
  const numGPUs = config.hardware.numGPUs ?? 1
  const gpu = config.hardware.gpu

  // F_peak in FLOP/s (convert TFLOPS → FLOPS)
  const fPeakFLOPS = getEffectiveTFLOPS(gpu, config.precision, config.fp8) * 1e12

  // MFU: user override takes precedence over the auto-default
  const mfu = config.mfuOverride ?? getDefaultMFU(activeParams, numGPUs)

  // --- Core formula (Section 6.1) ---
  const C = compute.totalFLOPs
  const tSeconds = C / (numGPUs * fPeakFLOPS * mfu)
  const theoreticalDays = tSeconds / 86400
  const theoreticalHours = tSeconds / 3600

  // Derived throughput metrics
  const totalTokens = C / compute.flopsPerToken
  const tokensPerSecond = totalTokens / tSeconds

  // Training steps from global batch size
  const globalBatchTokens =
    config.microBatchSize *
    config.sequenceLength *
    config.gradientAccumulationSteps *
    config.parallelism.N_dp
  const totalSteps = Math.ceil(totalTokens / globalBatchTokens)
  const secondsPerStep = totalSteps > 0 ? tSeconds / totalSteps : 0

  // --- Failure-adjusted time (Section 6.5) ---
  const failureResult = calculateFailureAdjustedTime(
    theoreticalDays,
    numGPUs,
    gpu.gpusPerNode,
    config.failureModel,
  )

  return {
    theoreticalDays,
    theoreticalHours,
    failureAdjustedDays: failureResult?.adjustedDays ?? null,
    failureAdjustedHours: failureResult
      ? failureResult.adjustedDays * 24
      : null,
    failureMultiplier: failureResult?.multiplier ?? null,
    tokensPerSecond,
    totalSteps,
    secondsPerStep,
  }
}

// ---------------------------------------------------------------------------
// calculateCost — Section 8
// ---------------------------------------------------------------------------

/**
 * Full cost breakdown: compute + checkpoint storage + failure overhead.
 *
 * Checkpoint size is derived from the selected optimizer profile rather than
 * hardcoded to 12Ψ. For mixed-precision optimizers the master weights live
 * inside `kOpt`; for full-fp32 optimizers the stored parameters must be
 * saved alongside optimizer states.
 *
 * @param time        – Training time estimate (theoretical + failure-adjusted).
 * @param config      – Full pretraining configuration.
 * @param totalParams – Total (not active) parameter count — all parameters
 *   are checkpointed, including dormant MoE experts.
 */
export function calculateCost(
  time: TrainingTimeEstimate,
  config: TrainingConfig,
  totalParams: number,
): CostEstimate {
  const numGPUs = config.hardware.numGPUs ?? 1
  const pricing = config.pricing

  // --- Section 8.1: Compute cost (theoretical) ---
  const computeCost = numGPUs * time.theoreticalHours * pricing.costPerGPUHour

  // --- Section 8.3: Actual compute cost (failure-adjusted) ---
  const actualComputeCost =
    time.failureAdjustedHours !== null
      ? numGPUs * time.failureAdjustedHours * pricing.costPerGPUHour
      : null

  const failureOverheadCost =
    actualComputeCost !== null ? actualComputeCost - computeCost : 0

  // --- Checkpoint size from optimizer profile ---
  const profile = OPTIMIZER_PROFILES.find((p) => p.id === config.optimizer)!
  const variant =
    config.gradientPrecision === "bf16" ? profile.bf16Grad : profile.fp32Grad

  // When master weights exist, kOpt already includes them.
  // When there are no master weights (e.g. full-fp32 AdamW), the model
  // parameters must be saved separately alongside optimizer states.
  const checkpointBytesPerParam =
    variant.masterWeightBytes > 0
      ? variant.kOpt
      : variant.parameterBytes + variant.kOpt
  const checkpointSize = checkpointBytesPerParam * totalParams

  // --- Section 8.2: Checkpoint storage cost ---
  const effectiveDays = time.failureAdjustedDays ?? time.theoreticalDays
  const numCheckpoints = Math.ceil(
    effectiveDays * config.failureModel.checkpointFrequencyPerDay,
  )
  const retention = pricing.checkpointRetentionCount

  const peakCheckpointStorage =
    Math.min(numCheckpoints, retention) * checkpointSize

  // Average checkpoints on disk over the training run.
  // Case 1 (numCkpt ≤ retention): ramp from 1 → numCkpt, avg = (n+1)/2.
  // Case 2 (numCkpt > retention): ramp phase then plateau at `retention`,
  //   with a small correction for the early ramp.
  let avgCheckpointCount: number
  if (numCheckpoints <= retention) {
    avgCheckpointCount = (numCheckpoints + 1) / 2
  } else {
    avgCheckpointCount =
      retention - (retention * (retention - 1)) / (2 * numCheckpoints)
  }
  const averageCheckpointStorage = avgCheckpointCount * checkpointSize

  // Storage cost uses SI GB (1 GB = 1e9 bytes), matching AWS S3 billing.
  const avgStorageGB = averageCheckpointStorage / 1e9
  const storageCost =
    pricing.storagePricePerGBMonth * avgStorageGB * (effectiveDays / 30.25)

  // --- Section 8.4: Total ---
  const totalCost = computeCost + storageCost + failureOverheadCost

  return {
    computeCost,
    actualComputeCost,
    storageCost,
    failureOverheadCost,
    totalCost,
    checkpointSize,
    numCheckpoints,
    peakCheckpointStorage,
    averageCheckpointStorage,
  }
}

// ---------------------------------------------------------------------------
// calculatePostTrainingCompute — Section 10.5
// ---------------------------------------------------------------------------

/**
 * Approximate FLOPs for post-training methods.
 *
 * | Method | FLOPs/token | Notes                                    |
 * |--------|-------------|------------------------------------------|
 * | SFT    | 6Ψ          | Same as pretraining (fwd + bwd)          |
 * | DPO    | 8Ψ          | Policy train (6Ψ) + reference fwd (2Ψ)   |
 * | PPO    | ~20Ψ        | Gen + reward + multi-epoch actor/critic  |
 * | GRPO   | ~10Ψ        | Gen + policy train (no critic)           |
 *
 * @param params – Base model parameter count (total, not LoRA-only).
 */
export function calculatePostTrainingCompute(
  method: PostTrainingMethod,
  params: number,
  config: PostTrainingConfig,
): { totalFLOPs: number; flopsPerToken: number } {
  const multipliers: Record<PostTrainingMethod, number> = {
    sft: 6,
    dpo: 8,
    ppo: 20,
    grpo: 10,
  }

  const flopsPerToken = multipliers[method] * params
  const totalTokens =
    config.datasetSizeExamples * config.epochs * config.sequenceLength
  const totalFLOPs = totalTokens * flopsPerToken

  return { totalFLOPs, flopsPerToken }
}

// ---------------------------------------------------------------------------
// calculateGenerationTime — Section 10.3
// ---------------------------------------------------------------------------

/**
 * Wall-clock time for autoregressive generation (PPO / GRPO rollouts).
 *
 *   T_gen = T_prefill + n_tokens × T_decode_per_token
 *
 * Prefill is compute-bound (all prompt tokens processed in one pass).
 * Decode alternates between memory-bandwidth-bound (small batch, dominated
 * by weight loading) and compute-bound (large batch).
 *
 * @param params   – Model parameters (total, for weight loading cost).
 * @param gpu      – GPU spec (TFLOPS, memory bandwidth).
 * @param numGPUs  – GPUs available for generation (TP/sharding).
 * @param precision – Training precision (determines β = bytes per param).
 * @param batchGen – Concurrent generation sequences.
 * @param nTokens  – New tokens to generate per sequence.
 * @param sPrompt  – Prompt length in tokens (per sequence).
 */
export function calculateGenerationTime(
  params: number,
  gpu: GPUSpec,
  numGPUs: number,
  precision: TrainingPrecision,
  batchGen: number,
  nTokens: number,
  sPrompt: number,
): {
  prefillSeconds: number
  decodeSeconds: number
  totalSeconds: number
  isMemoryBound: boolean
} {
  // Generation always uses half-precision peak (bf16/fp16 tensor cores)
  const fPeakFLOPS = gpu.halfPrecisionTFLOPS * 1e12

  // Practical memory bandwidth (spec: apply ~0.87-0.90 efficiency factor)
  const bwMemBps = gpu.memoryBandwidthGBps * 1e9 * 0.9
  const beta = getBytesPerParam(precision)

  // --- Prefill: compute-bound ---
  // Total FLOPs = 2Ψ per token × sPrompt tokens × batchGen sequences
  const prefillSeconds =
    (2 * params * sPrompt * batchGen) / (fPeakFLOPS * numGPUs)

  // --- Decode: per-token cost = max(memory-bound, compute-bound) ---
  // Memory-bound: cost of loading all model weights (same regardless of batch)
  const memBoundTerm = (2 * params * beta) / (bwMemBps * numGPUs)
  // Compute-bound: at large batch, FLOPs dominate over weight-load latency
  const computeBoundTerm = (2 * params * batchGen) / (fPeakFLOPS * numGPUs)

  const decodePerToken = Math.max(memBoundTerm, computeBoundTerm)
  const isMemoryBound = memBoundTerm >= computeBoundTerm

  const decodeSeconds = nTokens * decodePerToken
  const totalSeconds = prefillSeconds + decodeSeconds

  return { prefillSeconds, decodeSeconds, totalSeconds, isMemoryBound }
}
