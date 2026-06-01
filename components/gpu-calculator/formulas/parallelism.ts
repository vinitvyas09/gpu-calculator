/**
 * Parallelism recommendation engine — Spec Sections 9, 5.2, 5.7
 *
 * Pure TypeScript functions. No React, no DOM.
 */
import type {
  FSDPStrategy,
  FrameworkType,
  GPUSpec,
  MemoryBreakdown,
  ModelArchitecture,
  MoEConfig,
  ParameterCounts,
  ParallelismConfig,
  ParallelismRecommendation,
  TrainingConfig,
  Warning,
  ZeROStage,
} from "../types"
import {
  calculateDenseStateShardDegree,
  calculateMinGPUVRAMFloor,
  calculateTotalMemoryPerGPU,
} from "./memory"

export interface ValidationResult {
  valid: boolean
  message: string
}

export type PipelineSchedule = "none" | "1f1b" | "interleaved" | "afab"

export interface ScoredConfiguration {
  config: ParallelismConfig
  score: number
  memory: MemoryBreakdown
  label: string
}

interface SearchStage {
  zeroStage: ZeROStage
  VP: number
  schedule: PipelineSchedule
}

interface Candidate {
  config: ParallelismConfig
  memory: MemoryBreakdown
  label: string
  schedule: PipelineSchedule
  initSpikeBytes: number
  transientFits: boolean
}

interface SearchResult {
  fit: Candidate | null
  attempts: Candidate[]
}

interface SearchOutcome {
  recommended: Candidate | ScoredConfiguration | null
  closestAttempt: Candidate | null
  reasoning: string[]
}

function normalizeDegree(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 1
}

function isSwiGLUStyle(ffnType: ModelArchitecture["ffnType"]): boolean {
  return ffnType === "swiglu" || ffnType === "geglu" || ffnType === "moe"
}

function resolveFFNIntermediateSize(
  arch: ModelArchitecture,
  moe: MoEConfig
): number {
  if (moe.enabled && moe.denseIntermediateSize !== null) {
    return moe.denseIntermediateSize
  }

  if (arch.d_ff !== null) {
    return arch.d_ff
  }

  return isSwiGLUStyle(arch.ffnType)
    ? Math.round((8 / 3) * arch.d)
    : 4 * arch.d
}

function isMoEEnabled(moe: MoEConfig): boolean {
  return moe.enabled && moe.E > 0
}

function isPCIeOnly(gpu: GPUSpec): boolean {
  return gpu.interconnect === "pcie" || gpu.interconnect === "none"
}

function maxTensorParallelDegree(gpu: GPUSpec, numGPUs: number): number {
  if (gpu.id === "rtx-3090") {
    return Math.min(numGPUs, 2)
  }

  return Math.min(numGPUs, gpu.gpusPerNode, 8)
}

function usesAFABSchedule(
  framework: FrameworkType,
  N_pp: number,
  zeroStage: ZeROStage,
  numMicrobatches: number
): boolean {
  return (
    framework === "fsdp" &&
    N_pp > 1 &&
    zeroStage === 2 &&
    numMicrobatches < 2 * N_pp
  )
}

function mapZeROStageToFSDPStrategy(
  zeroStage: ZeROStage
): FSDPStrategy | null {
  switch (zeroStage) {
    case 0:
      return "NO_SHARD"
    case 2:
      return "SHARD_GRAD_OP"
    case 3:
      return "FULL_SHARD"
    case 1:
    default:
      return null
  }
}

function applyFrameworkStage(
  parallelism: ParallelismConfig
): ParallelismConfig {
  if (parallelism.framework !== "fsdp") {
    return { ...parallelism, fsdpStrategy: null }
  }

  return {
    ...parallelism,
    fsdpStrategy: mapZeROStageToFSDPStrategy(parallelism.zeroStage),
  }
}

function buildParallelismConfig(
  baseConfig: TrainingConfig,
  framework: FrameworkType,
  overrides: Partial<ParallelismConfig>
): ParallelismConfig {
  return applyFrameworkStage({
    N_dp: 1,
    N_tp: 1,
    N_pp: 1,
    N_cp: 1,
    N_ep: 1,
    zeroStage: 0,
    fsdpStrategy: null,
    framework,
    sequenceParallelism: baseConfig.parallelism.sequenceParallelism,
    VP: 1,
    ...overrides,
  })
}

function makeStrategyLabel(
  parallelism: ParallelismConfig,
  moeEnabled: boolean
): string {
  const parts: string[] = [`DP=${parallelism.N_dp}`]

  if (parallelism.N_tp > 1) {
    parts.push(`TP=${parallelism.N_tp}`)
  }

  if (moeEnabled && parallelism.N_ep > 1) {
    parts.push(`EP=${parallelism.N_ep}`)
  }

  if (parallelism.N_pp > 1) {
    parts.push(`PP=${parallelism.N_pp}`)
  }

  if (parallelism.N_cp > 1) {
    parts.push(`CP=${parallelism.N_cp}`)
  }

  if (parallelism.N_pp > 1 && parallelism.VP > 1) {
    parts.push(`VP=${parallelism.VP}`)
  }

  parts.push(`ZeRO-${parallelism.zeroStage}`)
  return parts.join(", ")
}

// ─── Constraint Validators ──────────────────────────────────────────────────

export function validateTPDivisibility(
  N_tp: number,
  d: number,
  a: number,
  a_kv: number | null,
  d_ff: number
): ValidationResult {
  if (a_kv !== null) {
    if (!Number.isFinite(a_kv) || a_kv <= 0 || !Number.isInteger(a_kv)) {
      return {
        valid: false,
        message: "KV head count must be a positive integer",
      }
    }

    if (!Number.isFinite(a) || a <= 0 || !Number.isInteger(a)) {
      return {
        valid: false,
        message: "Attention head count must be a positive integer",
      }
    }

    if (a_kv > a) {
      return {
        valid: false,
        message: `KV heads a_kv=${a_kv} cannot exceed attention heads a=${a}`,
      }
    }

    if (a % a_kv !== 0) {
      return {
        valid: false,
        message: `Attention heads a=${a} must be evenly divisible by KV heads a_kv=${a_kv}`,
      }
    }
  }

  if (N_tp <= 1) {
    return { valid: true, message: "No TP active" }
  }

  if (d % N_tp !== 0) {
    return {
      valid: false,
      message: `N_tp=${N_tp} does not evenly divide hidden size d=${d}`,
    }
  }

  if (a % N_tp !== 0) {
    return {
      valid: false,
      message: `N_tp=${N_tp} does not evenly divide attention heads a=${a}`,
    }
  }

  if (a_kv !== null && a_kv % N_tp !== 0) {
    return {
      valid: false,
      message: `N_tp=${N_tp} does not evenly divide KV heads a_kv=${a_kv}`,
    }
  }

  if (d_ff % N_tp !== 0) {
    return {
      valid: false,
      message: `N_tp=${N_tp} does not evenly divide d_ff=${d_ff}`,
    }
  }

  return { valid: true, message: "TP dimensions are compatible" }
}

export function validatePPDivisibility(
  N_pp: number,
  L: number
): ValidationResult {
  if (N_pp <= 1) {
    return { valid: true, message: "No PP active" }
  }

  if (L % N_pp === 0) {
    return {
      valid: true,
      message: `L=${L} divides into ${N_pp} stages (${L / N_pp} layers each)`,
    }
  }

  if (usesEmbeddingAwarePipelinePartition(N_pp, L)) {
    return {
      valid: true,
      message: `Embedding-aware partitioning enabled: (L+2)=${L + 2} divides into ${N_pp} stages`,
    }
  }

  return {
    valid: false,
    message: `Neither L=${L} nor (L+2)=${L + 2} is divisible by N_pp=${N_pp}`,
  }
}

export function usesEmbeddingAwarePipelinePartition(
  N_pp: number,
  L: number
): boolean {
  const pipelineDegree = normalizeDegree(N_pp)

  return (
    pipelineDegree > 1 &&
    Number.isInteger(L) &&
    L % pipelineDegree !== 0 &&
    (L + 2) % pipelineDegree === 0
  )
}

export function validateZeroPPCompatibility(
  zeroStage: ZeROStage,
  N_pp: number,
  framework: FrameworkType
): ValidationResult {
  if (N_pp <= 1) {
    return { valid: true, message: "No PP active" }
  }

  if (framework === "fsdp") {
    if (zeroStage === 3) {
      return {
        valid: false,
        message: "FSDP FULL_SHARD (ZeRO-3) is incompatible with PP.",
      }
    }

    if (zeroStage === 2) {
      return {
        valid: true,
        message:
          "FSDP SHARD_GRAD_OP (ZeRO-2) only pairs with PP under the AFAB schedule conditions.",
      }
    }

    return { valid: true, message: "ZeRO stage compatible with PP" }
  }

  if (zeroStage >= 2) {
    const frameworkLabel =
      framework === "hf_trainer"
        ? "HF Trainer"
        : framework === "megatron"
          ? "Megatron-LM"
          : "DeepSpeed"

    return {
      valid: false,
      message: `${frameworkLabel} ZeRO-${zeroStage} is incompatible with PP. Use ZeRO-0 or ZeRO-1.`,
    }
  }

  return { valid: true, message: "ZeRO stage compatible with PP" }
}

export function validateTensorExpertSequenceParallelism(
  config: ParallelismConfig,
  moeEnabled: boolean
): ValidationResult {
  if (
    !moeEnabled ||
    config.N_tp <= 1 ||
    config.N_ep <= 1 ||
    config.sequenceParallelism !== "disabled"
  ) {
    return {
      valid: true,
      message: "TP/EP sequence-parallelism constraints are compatible",
    }
  }

  return {
    valid: false,
    message:
      "Combining tensor parallelism with expert parallelism requires sequence parallelism. Set sequence parallelism to auto/enabled, or use TP=1 or EP=1.",
  }
}

export function validateContextParallelDivisibility(
  N_cp: number,
  sequenceLength: number
): ValidationResult {
  if (N_cp <= 1) {
    return { valid: true, message: "No CP active" }
  }

  if (!Number.isInteger(N_cp) || N_cp < 1) {
    return {
      valid: false,
      message: `N_cp=${N_cp} must be a positive integer`,
    }
  }

  if (!Number.isFinite(sequenceLength) || sequenceLength <= 0) {
    return {
      valid: false,
      message: "Sequence length must be positive for context parallelism",
    }
  }

  if (sequenceLength % N_cp !== 0) {
    return {
      valid: false,
      message: `Sequence length ${sequenceLength.toLocaleString()} must be divisible by N_cp=${N_cp}`,
    }
  }

  return {
    valid: true,
    message: `Sequence length ${sequenceLength.toLocaleString()} splits into ${(sequenceLength / N_cp).toLocaleString()} tokens per CP rank`,
  }
}

export function validateWorldSize(
  config: ParallelismConfig,
  numGPUs?: number
): ValidationResult {
  const world = getParallelWorldSize(config)

  if (numGPUs === undefined) {
    return {
      valid: Number.isInteger(world) && world >= 1,
      message: `World size = ${world}`,
    }
  }

  if (world !== numGPUs) {
    return {
      valid: false,
      message: `World size ${world} ≠ ${numGPUs} GPUs`,
    }
  }

  return {
    valid: true,
    message: `World size ${world} = ${numGPUs} GPUs`,
  }
}

function getParallelWorldSize(config: ParallelismConfig): number {
  return (
    normalizeDegree(config.N_dp) *
    normalizeDegree(config.N_tp) *
    normalizeDegree(config.N_pp) *
    normalizeDegree(config.N_cp) *
    normalizeDegree(config.N_ep)
  )
}

export function validateMicrobatches(
  numMicrobatches: number,
  N_pp: number,
  VP: number
): ValidationResult {
  if (N_pp <= 1) {
    return { valid: true, message: "No PP active" }
  }

  if (numMicrobatches < N_pp - 1) {
    return {
      valid: false,
      message: `1F1B schedule requires num_microbatches (${numMicrobatches}) ≥ N_pp-1 (${N_pp - 1})`,
    }
  }

  if (VP > 1 && numMicrobatches % N_pp !== 0) {
    return {
      valid: false,
      message: `Interleaved PP requires num_microbatches (${numMicrobatches}) divisible by N_pp (${N_pp})`,
    }
  }

  return { valid: true, message: "Microbatch count valid for PP schedule" }
}

export function validateHiddenDimAlignment(d: number): ValidationResult {
  if (d % 128 !== 0) {
    return {
      valid: false,
      message: `Hidden dimension d=${d} is not aligned to 128`,
    }
  }

  return { valid: true, message: "Hidden dimension aligned to 128" }
}

export function calculateVocabPadding(V: number, N_tp: number): number {
  if (N_tp <= 1) {
    return V
  }

  const alignment = 128 * N_tp
  return Math.ceil(V / alignment) * alignment
}

// ─── Pipeline Bubble ────────────────────────────────────────────────────────

export function calculatePipelineBubble(
  N_pp: number,
  numMicrobatches: number,
  VP: number = 1
): number {
  if (N_pp <= 1) {
    return 0
  }

  if (numMicrobatches <= 0) {
    return 1
  }

  const effectiveVP = Math.max(1, normalizeDegree(VP))
  return (N_pp - 1) / (effectiveVP * numMicrobatches + N_pp - 1)
}

// ─── Memory Helpers ────────────────────────────────────────────────────────

function applyVocabPadding(
  params: ParameterCounts,
  arch: ModelArchitecture,
  N_tp: number
): ParameterCounts {
  const paddedVocab = calculateVocabPadding(arch.V, N_tp)

  if (paddedVocab === arch.V) {
    return params
  }

  const extraEntries = paddedVocab - arch.V
  const embeddingDelta = extraEntries * arch.d
  const outputDelta = arch.tiedEmbeddings ? 0 : extraEntries * arch.d

  return {
    ...params,
    total: params.total + embeddingDelta + outputDelta,
    active: params.active + extraEntries * arch.d,
    embedding: params.embedding + embeddingDelta,
    outputProjection: params.outputProjection + outputDelta,
  }
}

function getEmbeddingParameterCount(params: ParameterCounts): number {
  return (
    params.embedding +
    params.outputProjection +
    params.positionalEmbedding +
    params.finalNorm
  )
}

function uniqueStageMoELayerCountsForLayerCount(
  totalLayers: number,
  moeLayers: number,
  transformerLayers: number
): number[] {
  if (transformerLayers <= 0 || totalLayers <= 0 || moeLayers <= 0) {
    return [0]
  }

  const boundedMoELayers = Math.min(Math.max(0, moeLayers), totalLayers)
  const expectedMoELayers =
    (Math.min(transformerLayers, totalLayers) * boundedMoELayers) / totalLayers
  const minMoELayers = Math.min(
    transformerLayers,
    boundedMoELayers,
    Math.floor(expectedMoELayers)
  )
  const maxMoELayers = Math.min(
    transformerLayers,
    boundedMoELayers,
    Math.ceil(expectedMoELayers)
  )

  return Array.from(new Set([minMoELayers, maxMoELayers]))
}

function getPipelineTransformerLayerCandidates(
  totalLayers: number,
  N_pp: number
): Array<{ transformerLayers: number; boundary: "first" | "last" | "none" }> {
  const pipelineDegree = normalizeDegree(N_pp)

  if (pipelineDegree <= 1) {
    return [{ transformerLayers: totalLayers, boundary: "first" }]
  }

  if (totalLayers % pipelineDegree === 0) {
    const layersPerStage = totalLayers / pipelineDegree
    return [
      { transformerLayers: layersPerStage, boundary: "first" },
      { transformerLayers: layersPerStage, boundary: "last" },
      { transformerLayers: layersPerStage, boundary: "none" },
    ]
  }

  if (usesEmbeddingAwarePipelinePartition(pipelineDegree, totalLayers)) {
    const slotsPerStage = (totalLayers + 2) / pipelineDegree
    const candidates: Array<{
      transformerLayers: number
      boundary: "first" | "last" | "none"
    }> = [
      {
        transformerLayers: Math.max(0, slotsPerStage - 1),
        boundary: "first",
      },
      {
        transformerLayers: Math.max(0, slotsPerStage - 1),
        boundary: "last",
      },
    ]

    if (pipelineDegree > 2) {
      candidates.push({ transformerLayers: slotsPerStage, boundary: "none" })
    }

    return candidates
  }

  const lower = Math.floor(totalLayers / pipelineDegree)
  const upper = Math.ceil(totalLayers / pipelineDegree)

  return [
    { transformerLayers: lower, boundary: "first" },
    { transformerLayers: lower, boundary: "last" },
    { transformerLayers: lower, boundary: "none" },
    { transformerLayers: upper, boundary: "none" },
  ]
}

function calculateLocalParameterCountBeforeZeRO(
  params: ParameterCounts,
  arch: ModelArchitecture,
  moe: MoEConfig,
  parallelism: ParallelismConfig
): number {
  const effectiveParams = applyVocabPadding(
    params,
    arch,
    normalizeDegree(parallelism.N_tp)
  )
  const N_tp = normalizeDegree(parallelism.N_tp)
  const N_pp = normalizeDegree(parallelism.N_pp)
  const N_ep = normalizeDegree(parallelism.N_ep)
  const embeddingTotal = getEmbeddingParameterCount(effectiveParams)
  const firstBoundaryLocal =
    N_pp <= 1
      ? embeddingTotal / N_tp
      : (effectiveParams.embedding + effectiveParams.positionalEmbedding) / N_tp
  const lastBoundaryLocal =
    N_pp <= 1
      ? 0
      : (effectiveParams.outputProjection + effectiveParams.finalNorm) / N_tp
  const routedExpertTotal = effectiveParams.moe?.expertParameters ?? 0
  const sharedExpertTotal = effectiveParams.moe?.sharedExpertParameters ?? 0
  const routerTotal = effectiveParams.moe?.routerParameters ?? 0
  const moeEnabled =
    moe.enabled &&
    effectiveParams.moe !== null &&
    routedExpertTotal + sharedExpertTotal + routerTotal > 0
  const moeLayers = Math.min(Math.max(0, moe.L_moe), arch.L)
  const commonPerLayer =
    effectiveParams.perLayer.attention + effectiveParams.perLayer.norm
  const denseFFNPerLayer = effectiveParams.perLayer.ffn
  const routedExpertPerMoELayer =
    moeLayers > 0 ? routedExpertTotal / moeLayers : 0
  const sharedExpertPerMoELayer =
    moeLayers > 0 ? sharedExpertTotal / moeLayers : 0
  const routerPerMoELayer = moeLayers > 0 ? routerTotal / moeLayers : 0

  return Math.max(
    ...getPipelineTransformerLayerCandidates(arch.L, N_pp).flatMap(
      ({ transformerLayers, boundary }) => {
        const boundaryLocal =
          boundary === "first"
            ? firstBoundaryLocal
            : boundary === "last"
              ? lastBoundaryLocal
              : 0

        return uniqueStageMoELayerCountsForLayerCount(
          arch.L,
          moeEnabled ? moeLayers : 0,
          transformerLayers
        ).map((moeLayersPerStage) => {
          const denseLayers = Math.max(0, transformerLayers - moeLayersPerStage)
          const nonExpertLocal =
            (transformerLayers * commonPerLayer + denseLayers * denseFFNPerLayer) /
              N_tp +
            boundaryLocal
          const routedExpertLocal =
            moeLayersPerStage > 0
              ? (moeLayersPerStage * routedExpertPerMoELayer) / N_ep
              : 0
          const sharedExpertLocal =
            moeLayersPerStage > 0
              ? moeLayersPerStage * sharedExpertPerMoELayer
              : 0
          const routerLocal =
            moeLayersPerStage > 0 ? moeLayersPerStage * routerPerMoELayer : 0

          return (
            nonExpertLocal +
            routedExpertLocal +
            sharedExpertLocal +
            routerLocal
          )
        })
      }
    )
  )
}

function calculateDeepSpeedInitSpikeBytes(
  params: ParameterCounts,
  arch: ModelArchitecture,
  moe: MoEConfig,
  parallelism: ParallelismConfig
): number {
  if (parallelism.framework !== "deepspeed" || parallelism.zeroStage === 0) {
    return 0
  }

  return 4 * calculateLocalParameterCountBeforeZeRO(params, arch, moe, parallelism)
}

function fitsWithTransientBuffers(
  memory: MemoryBreakdown,
  initSpikeBytes: number
): boolean {
  if (initSpikeBytes <= 0) {
    return memory.fits
  }

  return memory.fits && memory.total + initSpikeBytes * 1.04 <= memory.usableCapacity
}

function isParameterGroupEvenlySharded(
  parameterCount: number,
  shardDegree: number
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

function checkMemoryFit(
  params: ParameterCounts,
  baseConfig: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  parallelism: ParallelismConfig,
  schedule: PipelineSchedule
): MemoryBreakdown {
  const effectiveParams = applyVocabPadding(
    params,
    arch,
    normalizeDegree(parallelism.N_tp)
  )
  const configForCandidate = {
    ...baseConfig,
    parallelism,
  }
  const memory = calculateTotalMemoryPerGPU(
    effectiveParams,
    configForCandidate,
    arch,
    moe,
    gpu,
    schedule
  )

  return memory
}

// ─── Search Helpers ────────────────────────────────────────────────────────

function getTPDegrees(
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  numGPUs: number
): number[] {
  const maxTP = maxTensorParallelDegree(gpu, numGPUs)
  const dFF = resolveFFNIntermediateSize(arch, moe)
  const preferredDegrees = [2, 4, 8]

  return preferredDegrees.filter((N_tp) => {
    if (N_tp > maxTP) {
      return false
    }

    if (!validateTPDivisibility(N_tp, arch.d, arch.a, arch.a_kv, dFF).valid) {
      return false
    }

    return true
  })
}

function getPPDegrees(L: number, numGPUs: number): number[] {
  const upperBound = Math.min(numGPUs, L + 2)
  const values: number[] = []

  for (let N_pp = 2; N_pp <= upperBound; N_pp++) {
    if (validatePPDivisibility(N_pp, L).valid) {
      values.push(N_pp)
    }
  }

  return values
}

function getEPCandidates(
  totalExperts: number,
  N_tp: number,
  gpusPerNode: number
): number[] {
  if (totalExperts <= 0) {
    return []
  }

  const maxEP = Math.floor(gpusPerNode / Math.max(1, N_tp))
  const values: number[] = []

  for (let N_ep = 2; N_ep <= Math.min(totalExperts, maxEP); N_ep++) {
    if (totalExperts % N_ep === 0) {
      values.push(N_ep)
    }
  }

  return values.sort((left, right) => right - left)
}

function nearestPowerOfTwo(value: number): number {
  if (value <= 1) {
    return 1
  }

  const lower = 2 ** Math.floor(Math.log2(value))
  const upper = 2 ** Math.ceil(Math.log2(value))
  return value - lower <= upper - value ? lower : upper
}

function getCPDegrees(sequenceLength: number, numGPUs: number): number[] {
  if (sequenceLength < 32768) {
    return []
  }

  const rawDefault = sequenceLength / 8192
  const preferred = Math.max(2, nearestPowerOfTwo(rawDefault))
  const maxCP = Math.min(
    numGPUs,
    Math.max(2, 2 ** Math.floor(Math.log2(Math.max(2, sequenceLength / 2048))))
  )
  const values: number[] = []

  for (let N_cp = 2; N_cp <= maxCP; N_cp *= 2) {
    if (validateContextParallelDivisibility(N_cp, sequenceLength).valid) {
      values.push(N_cp)
    }
  }

  return values.sort((left, right) => {
    const leftDistance = Math.abs(Math.log2(left) - Math.log2(preferred))
    const rightDistance = Math.abs(Math.log2(right) - Math.log2(preferred))
    return leftDistance - rightDistance || left - right
  })
}

function getNoPPStageSearchOrder(framework: FrameworkType): SearchStage[] {
  if (framework === "fsdp") {
    return [
      { zeroStage: 0, VP: 1, schedule: "none" },
      { zeroStage: 2, VP: 1, schedule: "none" },
      { zeroStage: 3, VP: 1, schedule: "none" },
    ]
  }

  return [
    { zeroStage: 1, VP: 1, schedule: "none" },
    { zeroStage: 2, VP: 1, schedule: "none" },
    { zeroStage: 3, VP: 1, schedule: "none" },
  ]
}

function getPPStageSearchOrder(
  framework: FrameworkType,
  N_pp: number,
  numMicrobatches: number,
  baseVP: number
): SearchStage[] {
  if (framework === "fsdp") {
    if (numMicrobatches >= 2 * N_pp) {
      const VP = Math.max(2, normalizeDegree(baseVP))

      return [
        {
          zeroStage: 0,
          VP,
          schedule: "interleaved",
        },
        {
          zeroStage: 0,
          VP: 1,
          schedule: "1f1b",
        },
      ]
    }

    return [{ zeroStage: 2, VP: 1, schedule: "afab" }]
  }

  const VP = Math.max(1, normalizeDegree(baseVP))

  return [
    {
      zeroStage: 1,
      VP,
      schedule: VP > 1 ? "interleaved" : "1f1b",
    },
  ]
}

function validateScheduleForCandidate(
  candidate: ParallelismConfig,
  schedule: PipelineSchedule,
  numMicrobatches: number
): ValidationResult {
  switch (schedule) {
    case "afab":
      return {
        valid: true,
        message: "FSDP SHARD_GRAD_OP + AFAB schedule selected",
      }
    case "interleaved":
      return validateMicrobatches(numMicrobatches, candidate.N_pp, candidate.VP)
    case "1f1b":
      return validateMicrobatches(numMicrobatches, candidate.N_pp, 1)
    case "none":
    default:
      return { valid: true, message: "No PP active" }
  }
}

function evaluateTopology(
  params: ParameterCounts,
  arch: ModelArchitecture,
  baseConfig: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig,
  degrees: {
    N_tp: number
    N_pp: number
    N_cp: number
    N_ep: number
  },
  stageSearch: SearchStage[]
): SearchResult {
  const attempts: Candidate[] = []
  const topology =
    normalizeDegree(degrees.N_tp) *
    normalizeDegree(degrees.N_pp) *
    normalizeDegree(degrees.N_cp) *
    normalizeDegree(degrees.N_ep)

  if (topology > numGPUs || numGPUs % topology !== 0) {
    return { fit: null, attempts }
  }

  if (
    !validateContextParallelDivisibility(
      degrees.N_cp,
      baseConfig.sequenceLength
    ).valid
  ) {
    return { fit: null, attempts }
  }

  if (
    isMoEEnabled(moe) &&
    degrees.N_ep > 1 &&
    degrees.N_tp * degrees.N_ep > gpu.gpusPerNode
  ) {
    return { fit: null, attempts }
  }

  const N_dp = numGPUs / topology
  const numMicrobatches = normalizeDegree(baseConfig.gradientAccumulationSteps)
  const expertDataParallelNumerator =
    N_dp * normalizeDegree(degrees.N_cp) * normalizeDegree(degrees.N_tp)

  if (
    isMoEEnabled(moe) &&
    degrees.N_ep > 1 &&
    expertDataParallelNumerator % normalizeDegree(degrees.N_ep) !== 0
  ) {
    return { fit: null, attempts }
  }

  for (const stage of stageSearch) {
    if (
      !validateZeroPPCompatibility(
        stage.zeroStage,
        degrees.N_pp,
        baseConfig.parallelism.framework
      ).valid
    ) {
      continue
    }

    const parallelism = buildParallelismConfig(
      baseConfig,
      baseConfig.parallelism.framework,
      {
        N_dp,
        N_tp: degrees.N_tp,
        N_pp: degrees.N_pp,
        N_cp: degrees.N_cp,
        N_ep: degrees.N_ep,
        zeroStage: stage.zeroStage,
        VP: stage.VP,
      }
    )

    if (
      !validateTensorExpertSequenceParallelism(
        parallelism,
        isMoEEnabled(moe)
      ).valid
    ) {
      continue
    }

    if (!validateWorldSize(parallelism, numGPUs).valid) {
      continue
    }

    const scheduleValidation = validateScheduleForCandidate(
      parallelism,
      stage.schedule,
      numMicrobatches
    )

    if (!scheduleValidation.valid) {
      continue
    }

    const memory = checkMemoryFit(
      params,
      baseConfig,
      arch,
      moe,
      gpu,
      parallelism,
      stage.schedule
    )
    const initSpikeBytes = calculateDeepSpeedInitSpikeBytes(
      params,
      arch,
      moe,
      parallelism
    )
    const candidate: Candidate = {
      config: parallelism,
      memory,
      label: makeStrategyLabel(parallelism, isMoEEnabled(moe)),
      schedule: stage.schedule,
      initSpikeBytes,
      transientFits: fitsWithTransientBuffers(memory, initSpikeBytes),
    }

    attempts.push(candidate)

    if (memory.fits) {
      return { fit: candidate, attempts }
    }
  }

  return { fit: null, attempts }
}

function estimateMaxMicroBatch(
  memory: MemoryBreakdown,
  currentMicroBatch: number
): number {
  if (memory.activations <= 0 || currentMicroBatch <= 0) {
    return Math.max(1, currentMicroBatch)
  }

  const activationPerSample = memory.activations / currentMicroBatch

  if (activationPerSample <= 0) {
    return Math.max(1, currentMicroBatch)
  }

  const nonActivationTotal =
    memory.parameters +
    memory.gradients +
    memory.optimizerStates +
    memory.communicationBuffers +
    memory.frameworkOverhead
  const availableRaw = memory.usableCapacity / 1.04 - nonActivationTotal

  if (availableRaw <= 0) {
    return 0
  }

  return Math.max(1, Math.floor(availableRaw / activationPerSample))
}

function lowestZeROStage(candidates: Candidate[]): ZeROStage | null {
  if (candidates.length === 0) {
    return null
  }

  return Math.min(
    ...candidates.map((candidate) => candidate.config.zeroStage)
  ) as ZeROStage
}

function filterToLowestStage(candidates: Candidate[]): Candidate[] {
  const lowestStage = lowestZeROStage(candidates)

  if (lowestStage === null) {
    return []
  }

  return candidates.filter((candidate) => candidate.config.zeroStage === lowestStage)
}

function pickClosestAttempt(attempts: Candidate[]): Candidate | null {
  if (attempts.length === 0) {
    return null
  }

  return [...attempts].sort((left, right) => {
    const leftOverage = Math.max(0, left.memory.total - left.memory.usableCapacity)
    const rightOverage = Math.max(0, right.memory.total - right.memory.usableCapacity)

    return (
      leftOverage - rightOverage ||
      left.config.zeroStage - right.config.zeroStage ||
      left.memory.total - right.memory.total
    )
  })[0]
}

function preferTransientSafeCandidates(candidates: Candidate[]): Candidate[] {
  const safeCandidates = candidates.filter((candidate) => candidate.transientFits)
  return safeCandidates.length > 0 ? safeCandidates : candidates
}

function configsMatch(
  left: ParallelismConfig,
  right: ParallelismConfig
): boolean {
  return (
    left.N_dp === right.N_dp &&
    left.N_tp === right.N_tp &&
    left.N_pp === right.N_pp &&
    left.N_cp === right.N_cp &&
    left.N_ep === right.N_ep &&
    left.zeroStage === right.zeroStage &&
    left.framework === right.framework &&
    left.VP === right.VP
  )
}

function pickBestFeasibleCandidate(
  candidates: Candidate[],
  currentMicroBatch: number,
  gradientAccumulationSteps: number
): ScoredConfiguration | null {
  if (candidates.length === 0) {
    return null
  }

  return (
    scoreConfigurations(
      preferTransientSafeCandidates(filterToLowestStage(candidates)),
      currentMicroBatch,
      gradientAccumulationSteps
    )[0] ?? null
  )
}

function chooseCandidateOverFallback(
  candidate: Candidate | null,
  fallbackFits: Candidate[],
  currentMicroBatch: number,
  gradientAccumulationSteps: number
): {
  selected: Candidate | ScoredConfiguration | null
  comparison: "none" | "better_stage" | "same_stage"
} {
  if (candidate === null) {
    return { selected: null, comparison: "none" }
  }

  const fallbackStage = lowestZeROStage(fallbackFits)

  if (fallbackStage === null || candidate.config.zeroStage < fallbackStage) {
    return { selected: candidate, comparison: "better_stage" }
  }

  if (candidate.config.zeroStage > fallbackStage) {
    return { selected: null, comparison: "none" }
  }

  const comparisonPool = [...filterToLowestStage(fallbackFits), candidate]
  const best = pickBestFeasibleCandidate(
    comparisonPool,
    currentMicroBatch,
    gradientAccumulationSteps
  )

  if (best !== null && configsMatch(best.config, candidate.config)) {
    return { selected: best, comparison: "same_stage" }
  }

  return { selected: null, comparison: "none" }
}

function searchTensorStrategies(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig,
  tpDegrees: number[]
): {
  recommended: Candidate | null
  fallbackFits: Candidate[]
  attempts: Candidate[]
} {
  const attempts: Candidate[] = []
  const fallbackFits: Candidate[] = []
  const moeEnabled = isMoEEnabled(moe)
  const searchTPDegrees = moeEnabled ? [1, ...tpDegrees] : tpDegrees

  for (const N_tp of searchTPDegrees) {
    const epOrder = moeEnabled
      ? [1, ...getEPCandidates(moe.E, N_tp, gpu.gpusPerNode)]
      : [1]
    let bestAtCurrentTP: Candidate | null = null

    for (const N_ep of epOrder) {
      const result = evaluateTopology(
        params,
        arch,
        config,
        gpu,
        numGPUs,
        moe,
        {
          N_tp,
          N_pp: 1,
          N_cp: 1,
          N_ep,
        },
        getNoPPStageSearchOrder(config.parallelism.framework)
      )

      attempts.push(...result.attempts)

      if (result.fit === null) {
        continue
      }

      if (
        bestAtCurrentTP === null ||
        result.fit.config.zeroStage < bestAtCurrentTP.config.zeroStage
      ) {
        bestAtCurrentTP = result.fit
      }

      if (result.fit.config.zeroStage <= 1) {
        return {
          recommended: result.fit,
          fallbackFits,
          attempts,
        }
      }
    }

    if (bestAtCurrentTP !== null) {
      fallbackFits.push(bestAtCurrentTP)
    }
  }

  return {
    recommended: null,
    fallbackFits,
    attempts,
  }
}

function searchPipelineStrategies(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig,
  N_tp: number,
  ppDegrees: number[]
): SearchResult {
  const attempts: Candidate[] = []
  const moeEnabled = isMoEEnabled(moe)
  const numMicrobatches = normalizeDegree(config.gradientAccumulationSteps)

  for (const N_pp of ppDegrees) {
    const epOrder = moeEnabled
      ? [1, ...getEPCandidates(moe.E, N_tp, gpu.gpusPerNode)]
      : [1]
    let bestAtCurrentPP: Candidate | null = null

    for (const N_ep of epOrder) {
      const result = evaluateTopology(
        params,
        arch,
        config,
        gpu,
        numGPUs,
        moe,
        {
          N_tp,
          N_pp,
          N_cp: 1,
          N_ep,
        },
        getPPStageSearchOrder(
          config.parallelism.framework,
          N_pp,
          numMicrobatches,
          config.parallelism.VP
        )
      )

      attempts.push(...result.attempts)

      if (result.fit === null) {
        continue
      }

      if (
        bestAtCurrentPP === null ||
        result.fit.config.zeroStage < bestAtCurrentPP.config.zeroStage
      ) {
        bestAtCurrentPP = result.fit
      }

      if (result.fit.config.zeroStage <= 1) {
        return { fit: result.fit, attempts }
      }
    }

    if (bestAtCurrentPP !== null) {
      return { fit: bestAtCurrentPP, attempts }
    }
  }

  return { fit: null, attempts }
}

function searchContextStrategies(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig,
  N_tp: number,
  ppDegrees: number[],
  cpDegrees: number[]
): SearchResult {
  const attempts: Candidate[] = []
  const moeEnabled = isMoEEnabled(moe)

  for (const N_cp of cpDegrees) {
    let bestAtCurrentCP: Candidate | null = null

    for (const N_pp of [1, ...ppDegrees]) {
      const epOrder = moeEnabled
        ? [1, ...getEPCandidates(moe.E, N_tp, gpu.gpusPerNode)]
        : [1]

      for (const N_ep of epOrder) {
        const result = evaluateTopology(
          params,
          arch,
          config,
          gpu,
          numGPUs,
          moe,
          {
            N_tp,
            N_pp,
            N_cp,
            N_ep,
          },
          N_pp > 1
            ? getPPStageSearchOrder(
                config.parallelism.framework,
                N_pp,
                normalizeDegree(config.gradientAccumulationSteps),
                config.parallelism.VP
              )
            : getNoPPStageSearchOrder(config.parallelism.framework)
        )

        attempts.push(...result.attempts)

        if (result.fit === null) {
          continue
        }

        if (
          bestAtCurrentCP === null ||
          result.fit.config.zeroStage < bestAtCurrentCP.config.zeroStage
        ) {
          bestAtCurrentCP = result.fit
        }

        if (result.fit.config.zeroStage <= 1) {
          return { fit: result.fit, attempts }
        }
      }
    }

    if (bestAtCurrentCP !== null) {
      return { fit: bestAtCurrentCP, attempts }
    }
  }

  return { fit: null, attempts }
}

function searchRecommendation(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig
): SearchOutcome {
  const reasoning: string[] = []
  const attemptedCandidates: Candidate[] = []
  const fallbackFits: Candidate[] = []
  const moeEnabled = isMoEEnabled(moe)
  const pcieOnly = isPCIeOnly(gpu)
  const tpDegrees = getTPDegrees(arch, moe, gpu, numGPUs)
  const ppDegrees = getPPDegrees(arch.L, numGPUs)
  const cpDegrees = getCPDegrees(config.sequenceLength, numGPUs)
  const framework = config.parallelism.framework
  const singleFitDataParallelDegree = gpu.singleDeviceOnly
    ? 1
    : Math.max(1, numGPUs)

  const singleGPUConfig = buildParallelismConfig(config, framework, {
    N_dp: 1,
    N_tp: 1,
    N_pp: 1,
    N_cp: 1,
    N_ep: 1,
    zeroStage: 0,
    VP: 1,
  })
  const singleGPUCandidate: Candidate = {
    config: buildParallelismConfig(config, framework, {
      N_dp: singleFitDataParallelDegree,
      N_tp: 1,
      N_pp: 1,
      N_cp: 1,
      N_ep: 1,
      zeroStage: 0,
      VP: 1,
    }),
    memory: checkMemoryFit(
      params,
      config,
      arch,
      moe,
      gpu,
      singleGPUConfig,
      "none"
    ),
    label: makeStrategyLabel(
      buildParallelismConfig(config, framework, {
        N_dp: singleFitDataParallelDegree,
        N_tp: 1,
        N_pp: 1,
        N_cp: 1,
        N_ep: 1,
        zeroStage: 0,
        VP: 1,
      }),
      moeEnabled
    ),
    schedule: "none",
    initSpikeBytes: 0,
    transientFits: true,
  }

  if (gpu.singleDeviceOnly || numGPUs === 1) {
    reasoning.push(
      gpu.singleDeviceOnly
        ? `${gpu.name} supports single-device training only.`
        : "Only one GPU is available, so no model parallelism can be introduced."
    )

    return {
      recommended: singleGPUCandidate.memory.fits ? singleGPUCandidate : null,
      closestAttempt: singleGPUCandidate,
      reasoning,
    }
  }

  if (singleGPUCandidate.memory.fits) {
    reasoning.push(
      "The model fits on one GPU with room for activations, so pure data parallelism is sufficient."
    )

    return {
      recommended: singleGPUCandidate,
      closestAttempt: null,
      reasoning,
    }
  }

  const dpResult = evaluateTopology(
    params,
    arch,
    config,
    gpu,
    numGPUs,
    moe,
    {
      N_tp: 1,
      N_pp: 1,
      N_cp: 1,
      N_ep: 1,
    },
    getNoPPStageSearchOrder(config.parallelism.framework)
  )

  attemptedCandidates.push(...dpResult.attempts)

  if (dpResult.fit !== null) {
    if (pcieOnly || dpResult.fit.config.zeroStage <= 1) {
      reasoning.push(
        pcieOnly
          ? "Pure data parallelism fits, and PCIe-only GPUs should prefer ZeRO over TP."
          : `Pure data parallelism fits with ZeRO-${dpResult.fit.config.zeroStage}, so lower-overhead model sharding is unnecessary.`
      )

      return {
        recommended: dpResult.fit,
        closestAttempt: null,
        reasoning,
      }
    }

    fallbackFits.push(dpResult.fit)
    reasoning.push(
      `Pure data parallelism only fits with ZeRO-${dpResult.fit.config.zeroStage}; trying model parallelism to recover a lower ZeRO stage.`
    )
  } else {
    reasoning.push(
      "Pure data parallelism does not fit in GPU memory, so model sharding is required."
    )
  }

  if (!pcieOnly || dpResult.fit === null) {
    const tensorSearch = searchTensorStrategies(
      params,
      arch,
      config,
      gpu,
      numGPUs,
      moe,
      tpDegrees
    )

    attemptedCandidates.push(...tensorSearch.attempts)

    if (tensorSearch.recommended !== null) {
      reasoning.push(
        moeEnabled
          ? "A low-stage TP/EP configuration fits, so pipeline parallelism is unnecessary."
          : "A low-stage TP configuration fits, so pipeline parallelism is unnecessary."
      )

      return {
        recommended: tensorSearch.recommended,
        closestAttempt: null,
        reasoning,
      }
    }

    fallbackFits.push(...tensorSearch.fallbackFits)

    if (tensorSearch.fallbackFits.length > 0) {
      reasoning.push(
        moeEnabled
          ? "TP/EP reduce memory pressure, but the fitting options still require ZeRO-2/3; increasing PP next."
          : "TP reduces memory pressure, but the fitting options still require ZeRO-2/3; increasing PP next."
      )
    } else if (tpDegrees.length > 0 || moeEnabled) {
      reasoning.push(
        moeEnabled
          ? "TP/EP without PP still do not fit the model in memory."
          : "TP without PP still does not fit the model in memory."
      )
    }
  }

  const ppBaseTP = tpDegrees[tpDegrees.length - 1] ?? 1

  if (ppDegrees.length > 0) {
    const ppSearch = searchPipelineStrategies(
      params,
      arch,
      config,
      gpu,
      numGPUs,
      moe,
      ppBaseTP,
      ppDegrees
    )

    attemptedCandidates.push(...ppSearch.attempts)

    const ppChoice = chooseCandidateOverFallback(
      ppSearch.fit,
      fallbackFits,
      config.microBatchSize,
      normalizeDegree(config.gradientAccumulationSteps)
    )

    if (ppChoice.selected !== null) {
      reasoning.push(
        ppChoice.comparison === "better_stage"
          ? "Adding PP restored a lower ZeRO stage, so the smallest PP degree that fits was selected."
          : "Adding PP produced the best throughput among equally sharded candidates."
      )

      return {
        recommended: ppChoice.selected,
        closestAttempt: null,
        reasoning,
      }
    }

    if (ppSearch.fit !== null) {
      reasoning.push(
        "Pipeline parallelism did not improve on the earlier feasible candidates."
      )
    }
  }

  if (cpDegrees.length > 0) {
    const cpSearch = searchContextStrategies(
      params,
      arch,
      config,
      gpu,
      numGPUs,
      moe,
      ppBaseTP,
      ppDegrees,
      cpDegrees
    )

    attemptedCandidates.push(...cpSearch.attempts)

    const cpChoice = chooseCandidateOverFallback(
      cpSearch.fit,
      fallbackFits,
      config.microBatchSize,
      normalizeDegree(config.gradientAccumulationSteps)
    )

    if (cpChoice.selected !== null) {
      reasoning.push(
        `Sequence length ${config.sequenceLength.toLocaleString()} is long enough to justify context parallelism.`
      )

      return {
        recommended: cpChoice.selected,
        closestAttempt: null,
        reasoning,
      }
    }
  }

  if (fallbackFits.length > 0) {
    const bestFallback = pickBestFeasibleCandidate(
      fallbackFits,
      config.microBatchSize,
      normalizeDegree(config.gradientAccumulationSteps)
    )

    if (bestFallback !== null) {
      reasoning.push(
        "No later topology improved on the earlier feasible candidates, so the best fallback is returned."
      )

      return {
        recommended: bestFallback,
        closestAttempt: null,
        reasoning,
      }
    }
  }

  reasoning.push("No feasible configuration fits within the available VRAM.")
  return {
    recommended: null,
    closestAttempt: pickClosestAttempt(attemptedCandidates),
    reasoning,
  }
}

function hasFeasibleRecommendation(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig
): boolean {
  return (
    searchRecommendation(params, arch, config, gpu, numGPUs, moe).recommended !==
    null
  )
}

function findMinimumGPUCount(
  params: ParameterCounts,
  config: TrainingConfig,
  arch: ModelArchitecture,
  moe: MoEConfig,
  gpu: GPUSpec,
  currentNumGPUs: number
): number {
  if (gpu.singleDeviceOnly) {
    return 1
  }

  const cap = 4096
  let upperBound = Math.max(1, currentNumGPUs)
  let feasibleWithinCap = hasFeasibleRecommendation(
    params,
    arch,
    config,
    gpu,
    upperBound,
    moe
  )

  while (upperBound < cap && !feasibleWithinCap) {
    upperBound *= 2
    upperBound = Math.min(upperBound, cap)
    feasibleWithinCap = hasFeasibleRecommendation(
      params,
      arch,
      config,
      gpu,
      upperBound,
      moe
    )
  }

  if (!feasibleWithinCap) {
    return Number.POSITIVE_INFINITY
  }

  for (let candidateGPUs = 1; candidateGPUs <= upperBound; candidateGPUs++) {
    if (hasFeasibleRecommendation(params, arch, config, gpu, candidateGPUs, moe)) {
      return candidateGPUs
    }
  }

  return Number.POSITIVE_INFINITY
}

// ─── Throughput Scoring ─────────────────────────────────────────────────────

export function scoreConfigurations(
  configs: Array<{
    config: ParallelismConfig
    memory: MemoryBreakdown
    label: string
    schedule?: PipelineSchedule
  }>,
  currentMicroBatch: number,
  gradientAccumulationSteps: number
): ScoredConfiguration[] {
  const numMicrobatches = normalizeDegree(gradientAccumulationSteps)

  return configs
    .map(({ config, memory, label, schedule = "none" }) => {
      const maxBatch = estimateMaxMicroBatch(memory, currentMicroBatch)
      const isPureDP =
        config.N_tp === 1 &&
        config.N_pp === 1 &&
        config.N_cp === 1 &&
        config.N_ep === 1
      const isTPOnly =
        config.N_tp > 1 &&
        config.N_dp === 1 &&
        config.N_pp === 1 &&
        config.N_cp === 1 &&
        config.N_ep === 1

      let score: number

      if (isPureDP && config.zeroStage !== 3) {
        score = maxBatch * config.N_dp * 1.5
      } else if (isPureDP && config.zeroStage === 3) {
        score = maxBatch * config.N_dp
      } else if (isTPOnly) {
        score = maxBatch
      } else {
        score = maxBatch * config.N_dp
      }

      if (config.N_pp > 1) {
        if (schedule === "afab") {
          score *= 0.75
        } else {
          score *= 1 - calculatePipelineBubble(config.N_pp, numMicrobatches, config.VP)
        }
      }

      return { config, score, memory, label }
    })
    .sort((left, right) => {
      return (
        right.score - left.score ||
        left.config.zeroStage - right.config.zeroStage ||
        right.config.N_ep - left.config.N_ep ||
        right.config.N_dp - left.config.N_dp ||
        left.config.N_tp - right.config.N_tp ||
        left.config.N_pp - right.config.N_pp ||
        left.config.N_cp - right.config.N_cp
      )
    })
}

// ─── Main Recommendation Engine ─────────────────────────────────────────────

export function recommendParallelism(
  params: ParameterCounts,
  arch: ModelArchitecture,
  config: TrainingConfig,
  gpu: GPUSpec,
  numGPUs: number,
  moe: MoEConfig
): ParallelismRecommendation {
  const warnings: Warning[] = []
  const moeEnabled = isMoEEnabled(moe)
  const pcieOnly = isPCIeOnly(gpu)
  const currentSearchOutcome = searchRecommendation(
    params,
    arch,
    config,
    gpu,
    numGPUs,
    moe
  )

  if (!validateHiddenDimAlignment(arch.d).valid) {
    warnings.push({
      severity: "warning",
      category: "parallelism",
      message: `Hidden dimension d=${arch.d} is not aligned to 128, causing significant tensor-core inefficiency.`,
    })
  }

  const fallbackDataParallelDegree = gpu.singleDeviceOnly
    ? 1
    : Math.max(1, numGPUs)
  const defaultConfig = buildParallelismConfig(
    config,
    config.parallelism.framework,
    {
      N_dp: fallbackDataParallelDegree,
      N_tp: 1,
      N_pp: 1,
      N_cp: 1,
      N_ep: 1,
      zeroStage: 0,
      VP: 1,
    },
  )
  const minGPUs = gpu.singleDeviceOnly
    ? currentSearchOutcome.recommended === null
      ? Number.POSITIVE_INFINITY
      : 1
    : findMinimumGPUCount(
        params,
        config,
        arch,
        moe,
        gpu,
        Math.max(1, numGPUs)
      )
  const minimumSearchOutcome =
    !gpu.singleDeviceOnly &&
    currentSearchOutcome.recommended === null &&
    Number.isFinite(minGPUs) &&
    minGPUs > numGPUs
      ? searchRecommendation(params, arch, config, gpu, minGPUs, moe)
      : null
  const searchOutcome =
    minimumSearchOutcome !== null && minimumSearchOutcome.recommended !== null
      ? minimumSearchOutcome
      : currentSearchOutcome

  const chosen =
    searchOutcome.recommended ??
    currentSearchOutcome.closestAttempt ?? {
      config: defaultConfig,
      memory: checkMemoryFit(
        params,
        config,
        arch,
        moe,
        gpu,
        defaultConfig,
        "none"
      ),
      label: makeStrategyLabel(defaultConfig, moeEnabled),
      schedule: "none" as const,
      initSpikeBytes: 0,
      transientFits: true,
    }

  const parallelism = chosen.config
  const recommendedWorldSize = getParallelWorldSize(parallelism)
  const chosenInitSpikeBytes =
    "initSpikeBytes" in chosen
      ? chosen.initSpikeBytes
      : calculateDeepSpeedInitSpikeBytes(params, arch, moe, parallelism)
  const chosenTransientFits =
    "transientFits" in chosen
      ? chosen.transientFits
      : fitsWithTransientBuffers(chosen.memory, chosenInitSpikeBytes)
  const minVRAMFloor = calculateMinGPUVRAMFloor(
    applyVocabPadding(params, arch, parallelism.N_tp),
    { ...config, parallelism }
  )
  const pipelineBubbleFraction = calculatePipelineBubble(
    parallelism.N_pp,
    normalizeDegree(config.gradientAccumulationSteps),
    parallelism.VP
  )
  const paddedVocab = calculateVocabPadding(arch.V, parallelism.N_tp)

  if (currentSearchOutcome.recommended === null) {
    warnings.push({
      severity: "critical",
      category: "memory",
      message: gpu.singleDeviceOnly
        ? `Model does not fit on ${gpu.name}. This hardware is single-device only, so reduce the model, micro-batch size, or sequence length.`
        : `Model does not fit on ${numGPUs}× ${gpu.name}. Increase GPU count or reduce the model, batch size, or sequence length.`,
    })
  }

  if (!Number.isFinite(minGPUs)) {
    warnings.push({
      severity: "critical",
      category: "parallelism",
      message: gpu.singleDeviceOnly
        ? "No feasible single-device layout was found for the current micro-batch, sequence length, and checkpointing settings."
        : "No feasible auto-parallelism layout was found within 4,096 GPUs for the current micro-batch, sequence length, and checkpointing settings.",
    })
  } else if (recommendedWorldSize > numGPUs) {
    warnings.push({
      severity: "info",
      category: "hardware",
      message: `Showing the minimum feasible auto layout at ${recommendedWorldSize.toLocaleString()} GPUs. Memory, time, and cost estimates below use that cluster size rather than the currently selected ${numGPUs.toLocaleString()} GPUs.`,
    })
  }

  if (parallelism.N_pp > 1) {
    if (usesEmbeddingAwarePipelinePartition(parallelism.N_pp, arch.L)) {
      warnings.push({
        severity: "info",
        category: "parallelism",
        message: `PP=${parallelism.N_pp} uses embedding-aware partitioning: input and output embedding stages are treated as virtual layers, so first and last stages carry fewer transformer blocks.`,
      })
    }

    const scheduleValidation = validateScheduleForCandidate(
      parallelism,
      parallelism.VP > 1 ? "interleaved" : "1f1b",
      normalizeDegree(config.gradientAccumulationSteps)
    )

    if (
      usesAFABSchedule(
        parallelism.framework,
        parallelism.N_pp,
        parallelism.zeroStage,
        normalizeDegree(config.gradientAccumulationSteps)
      )
    ) {
      warnings.push({
        severity: "info",
        category: "parallelism",
        message: "FSDP SHARD_GRAD_OP + PP uses the AFAB schedule here; this removes the 1F1B microbatch minimum but increases activation residency.",
      })
    } else if (!scheduleValidation.valid) {
      warnings.push({
        severity: "critical",
        category: "parallelism",
        message: scheduleValidation.message,
      })
    }

    if (pipelineBubbleFraction > 0.5) {
      warnings.push({
        severity: "warning",
        category: "parallelism",
        message: `Pipeline bubble is ${(pipelineBubbleFraction * 100).toFixed(1)}%. Increase gradient accumulation steps to reduce idle time. Default MFU includes this schedule efficiency; manual MFU overrides should include it too.`,
      })
    } else if (pipelineBubbleFraction > 0.2) {
      warnings.push({
        severity: "info",
        category: "parallelism",
        message: `Pipeline bubble is ${(pipelineBubbleFraction * 100).toFixed(1)}%. A common rule of thumb is num_microbatches ≥ ${4 * parallelism.N_pp}. Default MFU includes this schedule efficiency; manual MFU overrides should include it too.`,
      })
    }

    if (parallelism.VP > 1) {
      warnings.push({
        severity: "info",
        category: "parallelism",
        message: `VP=${parallelism.VP} lowers the pipeline bubble but increases PP communication and activation residency.`,
      })
    }

    if (moeEnabled && moe.L_moe > 0 && moe.L_moe < arch.L) {
      warnings.push({
        severity: "info",
        category: "memory",
        message:
          "MoE memory under PP assumes MoE layers are distributed evenly across pipeline stages. If MoE layers are clustered, the peak stage can require more VRAM than shown.",
      })
    }
  }

  if (parallelism.zeroStage === 3) {
    warnings.push({
      severity: "info",
      category: "parallelism",
      message: "ZeRO-3 maximizes memory efficiency but adds the highest communication overhead.",
    })
  }

  if (moeEnabled && config.parallelism.sequenceParallelism === "disabled") {
    warnings.push({
      severity: "info",
      category: "parallelism",
      message:
        "Sequence parallelism is disabled, so auto mode skips combined TP+EP MoE layouts. Enable sequence parallelism to allow tensor-parallel expert-parallel candidates.",
    })
  }

  if (chosenInitSpikeBytes > 0) {
    const spikeGB = (chosenInitSpikeBytes / 1e9).toFixed(1)

    warnings.push({
      severity: chosenTransientFits ? "info" : "warning",
      category: "memory",
      message: chosenTransientFits
        ? `DeepSpeed initialization adds a transient ~${spikeGB} GB fp32 parameter buffer before sharding.`
        : `DeepSpeed steady-state memory fits, but initialization adds a transient ~${spikeGB} GB fp32 parameter buffer that can OOM without partitioned init.`,
    })
  }

  if (parallelism.N_tp > 1 && pcieOnly) {
    warnings.push({
      severity: "warning",
      category: "parallelism",
      message: `TP=${parallelism.N_tp} on PCIe-only GPUs is bandwidth-limited relative to NVLink-equipped systems.`,
    })
  }

  if (parallelism.N_tp > 1 && gpu.id === "rtx-3090") {
    warnings.push({
      severity: "info",
      category: "parallelism",
      message:
        "RTX 3090 TP=2 assumes a paired NVLink bridge (~112.5 GB/s), which is much slower than datacenter NVLink and is not present in unbridged multi-GPU builds.",
    })
  }

  if (
    parallelism.N_cp > 1 &&
    (pcieOnly || parallelism.N_tp * parallelism.N_cp > gpu.gpusPerNode)
  ) {
    warnings.push({
      severity: "warning",
      category: "parallelism",
      message: `CP=${parallelism.N_cp} introduces additional high-bandwidth traffic; scaling may be poor when CP extends beyond a node or runs on PCIe-only GPUs.`,
    })
  }

  if (parallelism.N_tp > 1 && paddedVocab > arch.V) {
    warnings.push({
      severity: "info",
      category: "parallelism",
      message: `Vocabulary padded from ${arch.V.toLocaleString()} to ${paddedVocab.toLocaleString()} for TP=${parallelism.N_tp}.`,
    })
  }

  const denseStateShardDegree = calculateDenseStateShardDegree({
    ...config,
    parallelism,
  })
  const effectiveParams = applyVocabPadding(
    params,
    arch,
    normalizeDegree(parallelism.N_tp)
  )
  const expertParameterCount =
    moeEnabled && effectiveParams.moe !== null
      ? effectiveParams.moe.expertParameters +
        effectiveParams.moe.sharedExpertParameters
      : 0
  const nonExpertParameterCount = Math.max(
    0,
    effectiveParams.total - expertParameterCount
  )

  if (
    parallelism.zeroStage > 0 &&
    !isParameterGroupEvenlySharded(
      nonExpertParameterCount,
      denseStateShardDegree
    )
  ) {
    warnings.push({
      severity: "info",
      category: "parallelism",
      message: `Non-expert parameter count is not evenly divisible by dense state shard degree N_dp × N_cp = ${denseStateShardDegree}; some frameworks will pad shards automatically.`,
    })
  }

  const expertStateShardDegree =
    (denseStateShardDegree * normalizeDegree(parallelism.N_tp)) /
    normalizeDegree(parallelism.N_ep)
  if (
    parallelism.zeroStage > 0 &&
    moeEnabled &&
    expertParameterCount > 0 &&
    !isParameterGroupEvenlySharded(
      expertParameterCount,
      expertStateShardDegree
    )
  ) {
    warnings.push({
      severity: "info",
      category: "parallelism",
      message: `Expert parameter count is not evenly divisible by expert state shard degree N_edp = ${expertStateShardDegree}; some frameworks will pad shards automatically.`,
    })
  }

  if (minVRAMFloor > gpu.memoryGB * 1e9 * 0.8) {
    warnings.push({
      severity: "warning",
      category: "memory",
      message: "The largest-layer working set is close to the minimum usable VRAM floor even with full sharding.",
    })
  }

  return {
    config: parallelism,
    minGPUs,
    minVRAMFloor,
    pipelineBubbleFraction,
    strategyLabel: makeStrategyLabel(parallelism, moeEnabled),
    reasoning:
      recommendedWorldSize > numGPUs
        ? [
            `Current ${numGPUs.toLocaleString()}× ${gpu.name} selection does not fit; showing the minimum feasible auto layout at ${recommendedWorldSize.toLocaleString()} GPUs.`,
            ...searchOutcome.reasoning,
          ]
        : searchOutcome.reasoning,
    warnings,
  }
}
