import type {
  CheckpointingMode,
  CPUOffloadMode,
  FSDPStrategy,
  GradientPrecision,
  GPUSpec,
  KVCachePrecision,
  MemoryBreakdown,
  ModelArchitecture,
  MoEConfig,
  OptimizerMemoryVariant,
  OptimizerType,
  ParameterCounts,
  ParallelismConfig,
  PostTrainingConfig,
  PostTrainingMemoryBreakdown,
  PostTrainingModelMemoryLineItem,
  TrainingConfig,
  ZeROStage,
} from "../types"
import { OPTIMIZER_PROFILES } from "../constants"

// ---------------------------------------------------------------------------
// 1. Optimizer profile lookup (Section 5.1)
// ---------------------------------------------------------------------------

export interface OptimizerValues {
  phi: number
  kOpt: number
  betaGrad: number
  parameterBytes: number
  masterWeightBytes: number
  optimizerStateBytes: number
}

export function getOptimizerProfile(
  optimizer: OptimizerType,
  gradPrecision: GradientPrecision
): OptimizerValues {
  const profile = OPTIMIZER_PROFILES.find((p) => p.id === optimizer)
  if (!profile) {
    throw new Error(`Unknown optimizer: ${optimizer}`)
  }
  const variant: OptimizerMemoryVariant =
    gradPrecision === "bf16" ? profile.bf16Grad : profile.fp32Grad
  return {
    phi: variant.phi,
    kOpt: variant.kOpt,
    betaGrad: variant.betaGrad,
    parameterBytes: variant.parameterBytes,
    masterWeightBytes: variant.masterWeightBytes,
    optimizerStateBytes: variant.optimizerStateBytes,
  }
}

// ---------------------------------------------------------------------------
// 2. Model state memory (Sections 5.1, 5.2, 5.6, 5.7)
// ---------------------------------------------------------------------------

function resolveZeROStage(config: TrainingConfig): ZeROStage {
  const fsdp = config.parallelism.fsdpStrategy
  if (fsdp === null) return config.parallelism.zeroStage
  const map: Record<FSDPStrategy, ZeROStage> = {
    NO_SHARD: 0,
    SHARD_GRAD_OP: 2,
    FULL_SHARD: 3,
    HYBRID_SHARD: 3,
    HYBRID_SHARD_ZERO2: 2,
  }
  return map[fsdp]
}

function isHybridShard(config: TrainingConfig): boolean {
  const fsdp = config.parallelism.fsdpStrategy
  return fsdp === "HYBRID_SHARD" || fsdp === "HYBRID_SHARD_ZERO2"
}

function getEffectiveDPForSharding(config: TrainingConfig): number {
  if (isHybridShard(config)) {
    return config.hardware.gpu.gpusPerNode
  }
  return config.parallelism.N_dp
}

function isSPEnabled(config: TrainingConfig): boolean {
  const sp = config.parallelism.sequenceParallelism
  if (sp === "enabled") return true
  if (sp === "disabled") return false
  // "auto": enable when N_tp > 1
  return config.parallelism.N_tp > 1
}

export interface ModelStateMemoryResult {
  parameters: number
  gradients: number
  optimizerStates: number
  total: number
}

export function calculateModelStateMemory(
  params: ParameterCounts,
  config: TrainingConfig
): ModelStateMemoryResult {
  const opt = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const zeroStage = resolveZeROStage(config)
  const N_tp = config.parallelism.N_tp
  const N_pp = config.parallelism.N_pp
  const N_ep = config.parallelism.N_ep

  // Effective DP degree for ZeRO sharding
  let N_dp_eff = getEffectiveDPForSharding(config)

  // SP + optimizer sharding interaction (Section 5.2):
  // When SP is active, SP ranks participate in optimizer sharding
  const spEnabled = isSPEnabled(config)
  const N_sp = spEnabled ? N_tp : 1

  // Determine per-GPU parameter count accounting for TP and PP
  // PP: most-loaded stage = transformer_params/N_pp + embedding params (Section 5.7)
  const psiTransformerPerStage = (params.total - params.embedding - params.outputProjection - params.positionalEmbedding - params.finalNorm) / N_pp
  const psiEmbedding = params.embedding + params.outputProjection + params.positionalEmbedding + params.finalNorm

  // TP splits weights within a layer (Section 5.6)
  const psiPerGpu = (psiTransformerPerStage + psiEmbedding) / N_tp

  // For MoE models, separate expert and non-expert params for ZeRO sharding
  const isMoE = config.model.moe.enabled && params.moe !== null
  let expertParams = 0
  let nonExpertParams = psiPerGpu

  if (isMoE && params.moe) {
    // Expert params per GPU after TP and PP
    const totalExpertParams = params.moe.expertParameters + params.moe.sharedExpertParameters
    const totalNonExpertParams = params.total - totalExpertParams
    const expertPerGpu = totalExpertParams / (N_tp * N_pp)
    const nonExpertPerGpu = (totalNonExpertParams / N_pp + psiEmbedding * (N_pp > 1 ? 0 : 0)) / N_tp

    // Recalculate: for MoE, embedding is part of non-expert
    expertParams = totalExpertParams / N_tp
    // Non-expert includes transformer attention + norms + embeddings
    nonExpertParams = (totalNonExpertParams / N_pp + (N_pp > 1 ? psiEmbedding * (1 - 1 / N_pp) : 0)) / N_tp
    // Simplified: just use the total per-GPU and split by ratio
    const expertRatio = totalExpertParams / params.total
    expertParams = psiPerGpu * expertRatio
    nonExpertParams = psiPerGpu * (1 - expertRatio)
  }

  // CPU offload adjustments
  const cpuOffload = config.cpuOffload

  // Calculate memory based on ZeRO stage
  let paramBytes: number
  let gradBytes: number
  let optBytes: number

  if (isMoE && N_ep > 1) {
    // MoE + ZeRO: expert and non-expert params use different sharding denominators
    // Expert data parallel degree: N_edp = N_dp * N_tp / N_ep (Section 5.2)
    const N_edp = (N_dp_eff * N_tp) / N_ep
    const N_dp_opt_nonexpert = N_dp_eff * N_sp
    const N_dp_opt_expert = N_edp

    const result = calculateZeROSplit(
      zeroStage,
      opt,
      expertParams,
      nonExpertParams,
      N_dp_opt_nonexpert,
      N_dp_opt_expert,
      cpuOffload
    )
    paramBytes = result.paramBytes
    gradBytes = result.gradBytes
    optBytes = result.optBytes
  } else {
    // Standard (non-MoE or MoE without EP): all params use same sharding degree
    const N_dp_opt = N_dp_eff * N_sp
    const totalPerGpu = psiPerGpu

    switch (zeroStage) {
      case 0:
        paramBytes = totalPerGpu * opt.parameterBytes
        gradBytes = totalPerGpu * opt.betaGrad
        optBytes = totalPerGpu * opt.kOpt
        break
      case 1: {
        paramBytes = totalPerGpu * opt.parameterBytes
        gradBytes = totalPerGpu * opt.betaGrad
        optBytes = (totalPerGpu * opt.kOpt) / N_dp_opt
        break
      }
      case 2: {
        paramBytes = totalPerGpu * opt.parameterBytes
        gradBytes = (totalPerGpu * opt.betaGrad) / N_dp_eff
        optBytes = (totalPerGpu * opt.kOpt) / N_dp_eff
        break
      }
      case 3: {
        paramBytes = (totalPerGpu * opt.parameterBytes) / N_dp_eff
        gradBytes = (totalPerGpu * opt.betaGrad) / N_dp_eff
        optBytes = (totalPerGpu * opt.kOpt) / N_dp_eff
        break
      }
      default:
        paramBytes = totalPerGpu * opt.parameterBytes
        gradBytes = totalPerGpu * opt.betaGrad
        optBytes = totalPerGpu * opt.kOpt
    }

    // CPU offload: subtract offloaded portions from GPU memory
    if (cpuOffload === "optimizer-only") {
      optBytes = 0
    } else if (cpuOffload === "optimizer-and-params" && zeroStage === 3) {
      optBytes = 0
      paramBytes = 0
      gradBytes = 0
    }
  }

  return {
    parameters: paramBytes,
    gradients: gradBytes,
    optimizerStates: optBytes,
    total: paramBytes + gradBytes + optBytes,
  }
}

function calculateZeROSplit(
  zeroStage: ZeROStage,
  opt: OptimizerValues,
  expertParams: number,
  nonExpertParams: number,
  N_dp_nonexpert: number,
  N_dp_expert: number,
  cpuOffload: CPUOffloadMode
): { paramBytes: number; gradBytes: number; optBytes: number } {
  const calcForGroup = (
    psi: number,
    N_dp: number
  ): { p: number; g: number; o: number } => {
    switch (zeroStage) {
      case 0:
        return {
          p: psi * opt.parameterBytes,
          g: psi * opt.betaGrad,
          o: psi * opt.kOpt,
        }
      case 1:
        return {
          p: psi * opt.parameterBytes,
          g: psi * opt.betaGrad,
          o: (psi * opt.kOpt) / N_dp,
        }
      case 2:
        return {
          p: psi * opt.parameterBytes,
          g: (psi * opt.betaGrad) / N_dp,
          o: (psi * opt.kOpt) / N_dp,
        }
      case 3:
        return {
          p: (psi * opt.parameterBytes) / N_dp,
          g: (psi * opt.betaGrad) / N_dp,
          o: (psi * opt.kOpt) / N_dp,
        }
      default:
        return {
          p: psi * opt.parameterBytes,
          g: psi * opt.betaGrad,
          o: psi * opt.kOpt,
        }
    }
  }

  const ne = calcForGroup(nonExpertParams, N_dp_nonexpert)
  const ex = calcForGroup(expertParams, N_dp_expert)

  let paramBytes = ne.p + ex.p
  let gradBytes = ne.g + ex.g
  let optBytes = ne.o + ex.o

  if (cpuOffload === "optimizer-only") {
    optBytes = 0
  } else if (cpuOffload === "optimizer-and-params" && zeroStage === 3) {
    optBytes = 0
    paramBytes = 0
    gradBytes = 0
  }

  return { paramBytes, gradBytes, optBytes }
}

// ---------------------------------------------------------------------------
// 3. Activation memory (Section 5.3)
// ---------------------------------------------------------------------------

function getPerLayerActivationBytes(
  arch: ModelArchitecture,
  config: TrainingConfig,
  isMoELayer: boolean,
  moe: MoEConfig
): number {
  const s = config.sequenceLength
  const b = config.microBatchSize
  const d = arch.d
  const a = arch.a
  const N_tp = config.parallelism.N_tp
  const N_cp = config.parallelism.N_cp
  const checkpointing = config.activationCheckpointing
  const flashAttn = config.flashAttention
  const ampAutocast = config.ampAutocast
  const spEnabled = isSPEnabled(config)

  // Context parallelism: replace s with s/N_cp (Section 5.3)
  const s_eff = s / N_cp

  // d_ff correction: the FFN portion of the 24 coefficient
  // 24 = 8 (attention TP-split) + 16 (FFN TP-split, assumes d_ff=4d)
  // Correct FFN portion: 4 * d_ff / d (instead of 16 when d_ff != 4d)
  const d_ff = arch.d_ff ?? 4 * d
  const ffnCoeff = 4 * d_ff / d // replaces 16 in the decomposition
  const tpSplitCoeff = 8 + ffnCoeff // replaces 24

  // Base linear and attention coefficients
  const baseLinear = ampAutocast ? 36 : 34
  const baseAttnCoeff = ampAutocast ? 6 : 5

  // Attention score term (O(s^2)) — removed by flash attention or selective checkpointing
  const hasAttnScoreTerm =
    !flashAttn && checkpointing !== "selective" && checkpointing !== "full"

  // AMP autocast correction when flash/selective removes attn score:
  // Only the LayerNorm correction remains (+2sbd), not the full attn coefficient change
  const ampLinearDelta = ampAutocast ? 2 : 0

  if (checkpointing === "full") {
    // Full checkpointing: store only layer input (Section 5.3)
    return 2 * s_eff * b * d
  }

  if (checkpointing === "selective") {
    // Selective: drops attention score term, keeps linear activations
    if (N_tp === 1) {
      return s_eff * b * d * (baseLinear)
    } else if (spEnabled) {
      // SP enabled: all activations split by N_tp
      return s_eff * b * d * (baseLinear / N_tp)
    } else {
      // TP without SP: 10 replicated + tpSplitCoeff/N_tp
      const replicatedCoeff = 10 + ampLinearDelta
      return s_eff * b * d * (replicatedCoeff + tpSplitCoeff / N_tp)
    }
  }

  // No checkpointing or partial (partial uses the same per-layer formula for non-checkpointed layers)
  const attnTerm = hasAttnScoreTerm
    ? baseAttnCoeff * a * s_eff / d
    : 0

  if (N_tp === 1) {
    return s_eff * b * d * (baseLinear + attnTerm)
  } else if (spEnabled) {
    // SP enabled: everything divided by N_tp
    return s_eff * b * d * ((baseLinear + attnTerm) / N_tp)
  } else {
    // TP without SP: 10 replicated + (tpSplitCoeff + attn)/N_tp
    const replicatedCoeff = 10 + ampLinearDelta
    const tpAttnTerm = hasAttnScoreTerm
      ? baseAttnCoeff * a * s_eff / (d * N_tp)
      : 0
    return s_eff * b * d * (replicatedCoeff + tpSplitCoeff / N_tp + tpAttnTerm)
  }
}

export function calculateActivationMemory(
  arch: ModelArchitecture,
  config: TrainingConfig,
  moe: MoEConfig,
  params: ParameterCounts
): number {
  const N_pp = config.parallelism.N_pp
  const L = arch.L
  const s = config.sequenceLength
  const b = config.microBatchSize
  const d = arch.d
  const V = arch.V
  const N_tp = config.parallelism.N_tp
  const numMicrobatches = config.gradientAccumulationSteps

  // Layers per pipeline stage
  const layersPerStage = L / N_pp
  const L_moe = moe.enabled ? moe.L_moe : 0
  const L_dense = L - L_moe

  // Dense and MoE layers per stage (uniform distribution)
  const densePerStage = L_dense / N_pp
  const moePerStage = L_moe / N_pp

  // Per-layer activation for dense layers
  const denseActPerLayer = getPerLayerActivationBytes(arch, config, false, moe)

  // Per-layer activation for MoE layers: FFN portion scales by topk/E
  let moeActPerLayer = denseActPerLayer
  if (moe.enabled && moe.E > 0) {
    // Decompose into non-FFN and FFN portions
    // Non-FFN = attention + norms, FFN = MLP activations
    // For MoE layers: FFN activation × topk/E (Section 5.3)
    const d_ff = arch.d_ff ?? 4 * d
    const ffnCoeff = 4 * d_ff / d
    const totalLinearCoeff = config.ampAutocast ? 36 : 34
    // The FFN portion of the linear coefficient is approximately ffnCoeff (which replaces 16 in the 24)
    // Non-FFN portion = totalLinearCoeff - ffnCoeff - 8 = totalLinearCoeff - 8 - ffnCoeff
    // Actually: 34 = 10 (replicated) + 8 (attn TP) + 16 (FFN TP)
    // FFN activation portion: ffnCoeff * s * b * d (possibly /N_tp)
    // Non-FFN: the rest
    const moeScale = moe.topk / moe.E
    // Simpler approach: compute full layer act, then scale FFN portion
    const nonFFNFraction = (totalLinearCoeff - ffnCoeff) / totalLinearCoeff
    const ffnFraction = ffnCoeff / totalLinearCoeff
    moeActPerLayer = denseActPerLayer * (nonFFNFraction + ffnFraction * moeScale)
  }

  // Per-stage activation memory
  const actPerStage = densePerStage * denseActPerLayer + moePerStage * moeActPerLayer

  // Partial checkpointing: first N_recomp layers are fully checkpointed, rest store all
  let totalActPerStage = actPerStage
  if (config.activationCheckpointing === "partial" && config.partialCheckpointDepth !== null) {
    const N_recomp = config.partialCheckpointDepth
    const checkpointedAct = 2 * (config.sequenceLength / config.parallelism.N_cp) * b * d
    const fullLayersPerStage = layersPerStage - N_recomp
    totalActPerStage =
      N_recomp * checkpointedAct + Math.max(0, fullLayersPerStage) * denseActPerLayer
  }

  // 1F1B in-flight microbatch factor (Section 5.3)
  const inflight = Math.min(N_pp, numMicrobatches)
  let totalAct = totalActPerStage * inflight

  // Output logits tensor (Section 5.3)
  const beta = config.precision === "fp32" ? 4 : 2
  if (!config.chunkedCrossEntropy) {
    totalAct += b * (config.sequenceLength / config.parallelism.N_cp) * V * beta
  }

  // Transient recomputation working memory for full checkpointing (Section 5.3)
  if (config.activationCheckpointing === "full") {
    const s_eff = config.sequenceLength / config.parallelism.N_cp
    const attnCoeff = config.flashAttention ? 0 : (config.ampAutocast ? 6 : 5) * arch.a * s_eff / d
    const linearCoeff = config.ampAutocast ? 36 : 34
    const recompWorking = s_eff * b * d * (linearCoeff + attnCoeff)
    totalAct += recompWorking
  }

  return totalAct
}

// ---------------------------------------------------------------------------
// 4. Communication buffers (Section 5.4)
// ---------------------------------------------------------------------------

export function calculateCommunicationBuffers(
  params: ParameterCounts,
  config: TrainingConfig,
  arch: ModelArchitecture
): number {
  const beta = config.precision === "fp32" ? 4 : 2
  const zeroStage = resolveZeROStage(config)
  const N_tp = config.parallelism.N_tp
  const s = config.sequenceLength
  const b = config.microBatchSize
  const d = arch.d
  const V = arch.V

  let buffers = 0

  // ZeRO-3 parameter prefetch buffer (Section 5.4)
  if (zeroStage === 3) {
    const paramsPerLayer =
      params.perLayer.attention + params.perLayer.ffn + params.perLayer.norm
    const prefetchFwd = Math.max(params.embedding, paramsPerLayer) * beta
    const prefetchBwd = 2 * paramsPerLayer * beta
    buffers += Math.max(prefetchFwd, prefetchBwd)
  }

  // ZeRO overlap_comm buffers (Section 5.4)
  if (zeroStage >= 2) {
    const overlapComm = config.zeroCommunication.overlapComm ||
      (zeroStage === 3) // ZeRO-3 defaults overlap_comm to true

    if (overlapComm) {
      // Bucket sizes: use HF Trainer auto-calculated defaults (hidden_size^2)
      // unless user has overridden
      let bucketElements: number
      if (config.zeroCommunication.mode === "deepspeed-defaults") {
        bucketElements = 5e8 // raw DeepSpeed default
      } else if (
        config.zeroCommunication.mode === "custom" &&
        config.zeroCommunication.allgatherBucketSizeElements !== null
      ) {
        bucketElements = config.zeroCommunication.allgatherBucketSizeElements
      } else {
        // HF auto: hidden_size^2
        bucketElements = d * d
      }

      const reduceBucket =
        config.zeroCommunication.mode === "custom" &&
        config.zeroCommunication.reduceBucketSizeElements !== null
          ? config.zeroCommunication.reduceBucketSizeElements
          : bucketElements

      buffers += 4.5 * (bucketElements + reduceBucket) * beta
    }
  }

  // Peak logit memory during loss backward (Section 5.4)
  if (!config.chunkedCrossEntropy) {
    const logitFwd = b * s * V * beta
    const logitGrad = 4 * b * s * V / N_tp
    buffers += logitGrad // The forward logits are already in activation memory
  }

  // TP backward all-gather buffer (Section 5.6)
  if (N_tp > 1) {
    buffers += b * s * d * ((N_tp - 1) / N_tp) * beta
  }

  // PP send/receive buffer
  if (config.parallelism.N_pp > 1) {
    buffers += s * b * d * beta
  }

  // torch.compile overhead (Section 5.4)
  if (config.torchCompile) {
    buffers += 0.1 * params.total * beta
  }

  return buffers
}

// ---------------------------------------------------------------------------
// 5. Total memory per GPU (Section 5.5)
// ---------------------------------------------------------------------------

export function calculateTotalMemoryPerGPU(
  params: ParameterCounts,
  config: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec
): MemoryBreakdown {
  const modelState = calculateModelStateMemory(params, config)
  const activations = calculateActivationMemory(arch, config, moe, params)
  const commBuffers = calculateCommunicationBuffers(params, config, arch)

  // Framework overhead: ~5 GB for Megatron/DeepSpeed, ~2 GB for bare FSDP/HF Trainer
  const framework = config.parallelism.framework
  let frameworkOverheadGB: number
  if (framework === "megatron" || framework === "deepspeed") {
    frameworkOverheadGB = 5
  } else {
    frameworkOverheadGB = 2
  }
  const frameworkOverhead = frameworkOverheadGB * 1e9

  // Sum before CUDA alignment
  const rawTotal =
    modelState.total + activations + commBuffers + frameworkOverhead

  // Apply 1.04x CUDA allocator alignment overhead (Section 5.5)
  const total = rawTotal * 1.04

  // Usable GPU memory: 90% for managed frameworks, 80% for vanilla PyTorch
  const isVanillaPyTorch =
    framework === "hf_trainer" || framework === "fsdp"
  const usableFraction = isVanillaPyTorch ? 0.8 : 0.9
  const gpuCapacity = gpu.memoryGB * 1e9
  const usableCapacity = gpuCapacity * usableFraction

  const fits = total <= usableCapacity
  const freeHeadroom = Math.max(0, usableCapacity - total)

  return {
    parameters: modelState.parameters,
    gradients: modelState.gradients,
    optimizerStates: modelState.optimizerStates,
    activations,
    communicationBuffers: commBuffers,
    frameworkOverhead,
    freeHeadroom,
    total,
    gpuCapacity,
    usableCapacity,
    fits,
  }
}

// Minimum GPU VRAM floor: largest layer × (β + β_grad)
export function calculateMinGPUVRAMFloor(
  params: ParameterCounts,
  config: TrainingConfig
): number {
  const beta = config.precision === "fp32" ? 4 : 2
  const opt = getOptimizerProfile(config.optimizer, config.gradientPrecision)
  const largestLayer =
    params.perLayer.attention + params.perLayer.ffn + params.perLayer.norm
  return largestLayer * (beta + opt.betaGrad)
}

// ---------------------------------------------------------------------------
// 6. Post-training memory functions (Section 10)
// ---------------------------------------------------------------------------

function getPostTrainingOptimizerValues(
  config: PostTrainingConfig
): OptimizerValues {
  return getOptimizerProfile(config.optimizer, config.gradientPrecision)
}

function postTrainingActivationMemory(
  arch: ModelArchitecture,
  config: PostTrainingConfig
): number {
  // Simplified: no checkpointing, single GPU, no TP/SP/CP for post-training
  const s = config.sequenceLength
  const b = config.batchSize
  const d = arch.d
  const a = arch.a
  const beta = config.precision === "fp32" ? 4 : 2
  // Use selective checkpointing approximation (no O(s^2) term)
  return s * b * d * 34
}

// LoRA trainable parameter count (Section 10.1)
export function calculateLoRAParamCount(
  config: PostTrainingConfig
): number {
  const r = config.lora.rank
  const d = config.baseModel.architecture.d
  const M = config.lora.targetModules.length
  const L = config.baseModel.architecture.L
  return 2 * r * d * M * L
}

// LoRA memory (Section 10.1)
export function calculateLoRAMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const psi = config.baseModel.parameterCount
  const arch = config.baseModel.architecture
  const opt = getPostTrainingOptimizerValues(config)
  const psiLora = calculateLoRAParamCount(config)
  const gpu = config.hardware.gpu

  // Base model frozen in bf16
  const baseModelBytes = 2 * psi

  // LoRA adapter: phi_lora × Ψ_lora (params + grads + optimizer)
  // = parameterBytes*Ψ_lora + betaGrad*Ψ_lora + (master+optState)*Ψ_lora
  const loraParamBytes = opt.parameterBytes * psiLora
  const loraGradBytes = opt.betaGrad * psiLora
  const loraOptBytes = opt.kOpt * psiLora
  const loraTotal = loraParamBytes + loraGradBytes + loraOptBytes

  const activations = postTrainingActivationMemory(arch, config)

  const items: PostTrainingModelMemoryLineItem[] = [
    { label: "Base model (frozen, bf16)", category: "frozen", bytes: baseModelBytes },
    { label: "LoRA parameters", category: "adapter", bytes: loraParamBytes },
    { label: "LoRA gradients", category: "adapter", bytes: loraGradBytes },
    { label: "LoRA optimizer states", category: "adapter", bytes: loraOptBytes },
    { label: "Activations", category: "buffer", bytes: activations },
  ]

  const rawTotal = baseModelBytes + loraTotal + activations
  const frameworkOverhead = 2e9
  const total = (rawTotal + frameworkOverhead) * 1.04

  const gpuCapacity = gpu.memoryGB * 1e9
  const usableCapacity = gpuCapacity * 0.9

  return {
    parameters: baseModelBytes + loraParamBytes,
    gradients: loraGradBytes,
    optimizerStates: loraOptBytes,
    activations,
    communicationBuffers: 0,
    frameworkOverhead,
    freeHeadroom: Math.max(0, usableCapacity - total),
    total,
    gpuCapacity,
    usableCapacity,
    fits: total <= usableCapacity,
    trainableModels: loraTotal,
    frozenModels: baseModelBytes,
    loraAdapter: loraTotal,
    ppoBuffers: 0,
    items,
  }
}

// QLoRA memory (Section 10.1)
export function calculateQLoRAMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const psi = config.baseModel.parameterCount
  const arch = config.baseModel.architecture
  const opt = getPostTrainingOptimizerValues(config)
  const psiLora = calculateLoRAParamCount(config)
  const gpu = config.hardware.gpu

  // Base model quantized: 4-bit = ~0.5 bytes/param + 0.01 overhead
  // 8-bit = ~1 byte/param + overhead
  const quantBits = config.lora.quantizationBits ?? 4
  const bytesPerParam = quantBits === 4 ? 0.5 : 1.0
  const overheadFraction = quantBits === 4 ? 0.01 : 0.005
  const baseModelBytes = psi * (bytesPerParam + overheadFraction * 2) // quantization constants

  // LoRA adapter states (same as LoRA)
  const loraParamBytes = opt.parameterBytes * psiLora
  const loraGradBytes = opt.betaGrad * psiLora
  const loraOptBytes = opt.kOpt * psiLora
  const loraTotal = loraParamBytes + loraGradBytes + loraOptBytes

  const activations = postTrainingActivationMemory(arch, config)

  const items: PostTrainingModelMemoryLineItem[] = [
    { label: `Base model (${quantBits}-bit quantized)`, category: "frozen", bytes: baseModelBytes },
    { label: "LoRA parameters", category: "adapter", bytes: loraParamBytes },
    { label: "LoRA gradients", category: "adapter", bytes: loraGradBytes },
    { label: "LoRA optimizer states", category: "adapter", bytes: loraOptBytes },
    { label: "Activations", category: "buffer", bytes: activations },
  ]

  const rawTotal = baseModelBytes + loraTotal + activations
  const frameworkOverhead = 2e9
  const total = (rawTotal + frameworkOverhead) * 1.04

  const gpuCapacity = gpu.memoryGB * 1e9
  const usableCapacity = gpuCapacity * 0.9

  return {
    parameters: baseModelBytes + loraParamBytes,
    gradients: loraGradBytes,
    optimizerStates: loraOptBytes,
    activations,
    communicationBuffers: 0,
    frameworkOverhead,
    freeHeadroom: Math.max(0, usableCapacity - total),
    total,
    gpuCapacity,
    usableCapacity,
    fits: total <= usableCapacity,
    trainableModels: loraTotal,
    frozenModels: baseModelBytes,
    loraAdapter: loraTotal,
    ppoBuffers: 0,
    items,
  }
}

// DPO memory (Section 10.2)
export function calculateDPOMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const psi = config.baseModel.parameterCount
  const arch = config.baseModel.architecture
  const opt = getPostTrainingOptimizerValues(config)
  const gpu = config.hardware.gpu
  const isLoRA = config.approach === "lora" || config.approach === "qlora"
  const isQLoRA = config.approach === "qlora"
  const activations = postTrainingActivationMemory(arch, config) * 2 // 2x for chosen + rejected

  const items: PostTrainingModelMemoryLineItem[] = []
  let trainableModels = 0
  let frozenModels = 0
  let loraAdapter = 0

  if (isLoRA || isQLoRA) {
    // LoRA-as-reference optimization: no separate reference model
    const psiLora = calculateLoRAParamCount(config)
    const baseBytesPerParam = isQLoRA
      ? (config.lora.quantizationBits === 8 ? 1.01 : 0.52)
      : 2
    const baseBytes = psi * baseBytesPerParam

    const loraParamBytes = opt.parameterBytes * psiLora
    const loraGradBytes = opt.betaGrad * psiLora
    const loraOptBytes = opt.kOpt * psiLora
    const loraTotal = loraParamBytes + loraGradBytes + loraOptBytes

    frozenModels = baseBytes
    loraAdapter = loraTotal
    trainableModels = loraTotal

    items.push(
      { label: `Policy base (${isQLoRA ? "quantized" : "frozen bf16"})`, category: "frozen", bytes: baseBytes },
      { label: "LoRA adapter (trainable)", category: "adapter", bytes: loraTotal },
      { label: "Activations (2x for chosen+rejected)", category: "buffer", bytes: activations }
    )
  } else {
    // Full fine-tuning: policy (trainable) + reference (frozen)
    const policyBytes = opt.phi * psi
    const refBytes = 2 * psi

    trainableModels = policyBytes
    frozenModels = refBytes

    items.push(
      { label: "Policy model (trainable)", category: "trainable", bytes: policyBytes },
      { label: "Reference model (frozen, bf16)", category: "frozen", bytes: refBytes },
      { label: "Activations (2x for chosen+rejected)", category: "buffer", bytes: activations }
    )
  }

  // DPO log-prob storage
  const logProbBytes = 2 * config.batchSize * config.sequenceLength * 4
  items.push({ label: "DPO log-prob storage", category: "buffer", bytes: logProbBytes })

  const rawTotal = trainableModels + frozenModels + activations + logProbBytes
  const frameworkOverhead = 2e9
  const total = (rawTotal + frameworkOverhead) * 1.04

  const gpuCapacity = gpu.memoryGB * 1e9
  const usableCapacity = gpuCapacity * 0.9

  return {
    parameters: trainableModels + frozenModels,
    gradients: 0, // included in trainableModels for simplicity
    optimizerStates: 0,
    activations,
    communicationBuffers: logProbBytes,
    frameworkOverhead,
    freeHeadroom: Math.max(0, usableCapacity - total),
    total,
    gpuCapacity,
    usableCapacity,
    fits: total <= usableCapacity,
    trainableModels,
    frozenModels,
    loraAdapter,
    ppoBuffers: logProbBytes,
    items,
  }
}

// PPO memory (Section 10.3)
export function calculatePPOMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const psiActor = config.baseModel.parameterCount
  const psiCritic = config.ppo.criticModelParameterCount
  const psiReward = config.ppo.rewardModelParameterCount
  const arch = config.baseModel.architecture
  const opt = getPostTrainingOptimizerValues(config)
  const gpu = config.hardware.gpu
  const isLoRA = config.approach === "lora" || config.approach === "qlora"

  const activations = postTrainingActivationMemory(arch, config)

  const items: PostTrainingModelMemoryLineItem[] = []
  let trainableModels = 0
  let frozenModels = 0
  let loraAdapter = 0

  if (isLoRA) {
    // LoRA actor: base frozen + LoRA trainable, no separate reference
    const psiLora = calculateLoRAParamCount(config)
    const actorBaseBytes = 2 * psiActor
    const loraParamBytes = opt.parameterBytes * psiLora
    const loraGradBytes = opt.betaGrad * psiLora
    const loraOptBytes = opt.kOpt * psiLora
    const loraTotal = loraParamBytes + loraGradBytes + loraOptBytes

    // Critic still fully trainable
    const criticBytes = opt.phi * psiCritic
    // Reward model frozen
    const rewardBytes = 2 * psiReward

    trainableModels = loraTotal + criticBytes
    frozenModels = actorBaseBytes + rewardBytes
    loraAdapter = loraTotal

    items.push(
      { label: "Actor base (frozen bf16, doubles as reference)", category: "frozen", bytes: actorBaseBytes },
      { label: "Actor LoRA adapter (trainable)", category: "adapter", bytes: loraTotal },
      { label: "Critic model (trainable)", category: "trainable", bytes: criticBytes },
      { label: "Reward model (frozen, bf16)", category: "frozen", bytes: rewardBytes },
      { label: "Activations", category: "buffer", bytes: activations }
    )
  } else {
    // Full fine-tuning: actor + critic trainable, ref + reward frozen
    const actorBytes = opt.phi * psiActor
    const criticBytes = opt.phi * psiCritic
    const refBytes = 2 * psiActor
    const rewardBytes = 2 * psiReward

    trainableModels = actorBytes + criticBytes
    frozenModels = refBytes + rewardBytes

    items.push(
      { label: "Actor (trainable)", category: "trainable", bytes: actorBytes },
      { label: "Critic (trainable)", category: "trainable", bytes: criticBytes },
      { label: "Reference model (frozen, bf16)", category: "frozen", bytes: refBytes },
      { label: "Reward model (frozen, bf16)", category: "frozen", bytes: rewardBytes },
      { label: "Activations", category: "buffer", bytes: activations }
    )
  }

  // PPO rollout buffers: 16 × s bytes per sample × batch_size
  const ppoBufferBytes = 16 * config.sequenceLength * config.batchSize

  // KV cache for generation phase (Section 10.3)
  const a_kv = arch.a_kv ?? arch.a
  const d_kv = arch.d / arch.a // d_head
  const betaCache = config.kvCachePrecision === "int8" ? 1 : 2
  const kvCache =
    config.batchSize * 2 * arch.L * a_kv * d_kv * config.sequenceLength * betaCache

  items.push(
    { label: "PPO rollout buffers", category: "buffer", bytes: ppoBufferBytes },
    { label: "KV cache (generation)", category: "buffer", bytes: kvCache }
  )

  const rawTotal =
    trainableModels + frozenModels + activations + ppoBufferBytes + kvCache
  const frameworkOverhead = 5e9
  const total = (rawTotal + frameworkOverhead) * 1.04

  const gpuCapacity = gpu.memoryGB * 1e9
  const usableCapacity = gpuCapacity * 0.9

  return {
    parameters: trainableModels + frozenModels,
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
    trainableModels,
    frozenModels,
    loraAdapter,
    ppoBuffers: ppoBufferBytes + kvCache,
    items,
  }
}

// GRPO memory (Section 10.4)
export function calculateGRPOMemory(
  config: PostTrainingConfig
): PostTrainingMemoryBreakdown {
  const psi = config.baseModel.parameterCount
  const arch = config.baseModel.architecture
  const opt = getPostTrainingOptimizerValues(config)
  const gpu = config.hardware.gpu
  const isLoRA = config.approach === "lora" || config.approach === "qlora"

  const activations = postTrainingActivationMemory(arch, config)
  const items: PostTrainingModelMemoryLineItem[] = []
  let trainableModels = 0
  let frozenModels = 0
  let loraAdapter = 0

  if (isLoRA) {
    // LoRA with shared reference (Section 10.4)
    const psiLora = calculateLoRAParamCount(config)
    const baseBytes = 2 * psi
    const loraParamBytes = opt.parameterBytes * psiLora
    const loraGradBytes = opt.betaGrad * psiLora
    const loraOptBytes = opt.kOpt * psiLora
    const loraTotal = loraParamBytes + loraGradBytes + loraOptBytes

    trainableModels = loraTotal
    frozenModels = baseBytes
    loraAdapter = loraTotal

    items.push(
      { label: "Policy base (frozen bf16, doubles as reference)", category: "frozen", bytes: baseBytes },
      { label: "LoRA adapter (trainable)", category: "adapter", bytes: loraTotal },
      { label: "Activations", category: "buffer", bytes: activations }
    )
  } else {
    // Full: trainable policy + frozen reference
    const policyBytes = opt.phi * psi
    const refBytes = 2 * psi

    trainableModels = policyBytes
    frozenModels = refBytes

    items.push(
      { label: "Policy model (trainable)", category: "trainable", bytes: policyBytes },
      { label: "Reference model (frozen, bf16)", category: "frozen", bytes: refBytes },
      { label: "Activations", category: "buffer", bytes: activations }
    )
  }

  // KV cache for generation: scales with group size G
  const G = config.grpo.groupSize
  const a_kv = arch.a_kv ?? arch.a
  const d_kv = arch.d / arch.a
  const betaCache = config.kvCachePrecision === "int8" ? 1 : 2
  const kvCache =
    G * config.batchSize * 2 * arch.L * a_kv * d_kv * config.sequenceLength * betaCache

  items.push({ label: `KV cache (generation, G=${G})`, category: "buffer", bytes: kvCache })

  const rawTotal = trainableModels + frozenModels + activations + kvCache
  const frameworkOverhead = 2e9
  const total = (rawTotal + frameworkOverhead) * 1.04

  const gpuCapacity = gpu.memoryGB * 1e9
  const usableCapacity = gpuCapacity * 0.9

  return {
    parameters: trainableModels + frozenModels,
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
    trainableModels,
    frozenModels,
    loraAdapter,
    ppoBuffers: kvCache,
    items,
  }
}
