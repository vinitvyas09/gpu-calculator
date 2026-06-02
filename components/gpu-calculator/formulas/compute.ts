/**
 * Parameter counting + compute estimation — Spec Sections 3, 4
 *
 * Pure TypeScript functions. No React, no DOM.
 */
import type {
  ModelArchitecture,
  MoEConfig,
  ParameterCounts,
  ComputeEstimate,
  ChinchillaAnalysis,
  BatchSizeAnalysis,
  DataRepetitionAnalysis,
} from "../types"
import { QUICK_MODE_LOOKUP, CHINCHILLA_COEFFICIENTS } from "../constants"

// ─── Helpers ──────────────────────────────────────────────────────────────────

const VALID_FFN_TYPES = new Set(["standard", "swiglu", "geglu", "moe"])
const VALID_NORM_TYPES = new Set(["layernorm", "rmsnorm"])
const VALID_POSITIONAL_EMBEDDINGS = new Set([
  "learned",
  "rope",
  "alibi",
  "none",
])
const VALID_ATTENTION_VARIANTS = new Set(["mha", "gqa", "mqa", "mla"])

export function getInvalidArchitectureEnumMessages(
  arch: ModelArchitecture
): string[] {
  const messages: string[] = []

  if (!VALID_FFN_TYPES.has(arch.ffnType)) {
    messages.push("FFN type must be standard, SwiGLU, GeGLU, or MoE.")
  }

  if (!VALID_NORM_TYPES.has(arch.normType)) {
    messages.push("Norm type must be LayerNorm or RMSNorm.")
  }

  if (!VALID_POSITIONAL_EMBEDDINGS.has(arch.posEmbedding)) {
    messages.push("Positional embedding type must be learned, RoPE, ALiBi, or none.")
  }

  if (!VALID_ATTENTION_VARIANTS.has(arch.attentionVariant)) {
    messages.push("Attention variant must be MHA, GQA, MQA, or MLA.")
  }

  return messages
}

function hasInvalidArchitectureEnums(arch: ModelArchitecture): boolean {
  return getInvalidArchitectureEnumMessages(arch).length > 0
}

export function hasInvalidArchitectureConfig(
  arch: ModelArchitecture,
  sequenceLength?: number,
): boolean {
  const normalizedArch = normalizeAttentionVariantHeads(arch)
  const { d, L, a, V, posEmbedding } = normalizedArch
  const a_kv = normalizedArch.a_kv ?? a
  const hasInvalidLearnedPositionLength =
    posEmbedding === "learned" &&
    sequenceLength !== undefined &&
    !isFinitePositiveInteger(sequenceLength)

  return (
    !isFinitePositive(d) ||
    !isFinitePositive(L) ||
    !isFinitePositive(a) ||
    !isFinitePositive(a_kv) ||
    !isFinitePositive(V) ||
    hasInvalidArchitectureEnums(normalizedArch) ||
    !Number.isInteger(d) ||
    !Number.isInteger(L) ||
    !Number.isInteger(a) ||
    !Number.isInteger(a_kv) ||
    !Number.isInteger(V) ||
    (normalizedArch.d_ff !== null &&
      (!isFinitePositive(normalizedArch.d_ff) ||
        !Number.isInteger(normalizedArch.d_ff))) ||
    hasInvalidExplicitHeadDim(normalizedArch) ||
    ((normalizedArch.d_head === null || normalizedArch.d_head === undefined) &&
      d % a !== 0) ||
    hasInvalidLearnedPositionLength ||
    a_kv > a ||
    a % a_kv !== 0
  )
}

/** True when the FFN uses 3 projections (gate + up + down): SwiGLU, GeGLU, or MoE experts. */
function isSwiGLUStyle(ffnType: string): boolean {
  return ffnType === "swiglu" || ffnType === "geglu" || ffnType === "moe"
}

/** FFN parameter count: 2·d·d_ff (standard) or 3·d·d_ff (SwiGLU/GeGLU). */
function computeFFNParams(d: number, dFF: number, swiGLU: boolean): number {
  return swiGLU ? 3 * d * dFF : 2 * d * dFF
}

/** Resolve d_ff when null, falling back to 4d (standard) or round(8d/3) (SwiGLU). */
function resolveIntermediateSize(
  dFF: number | null,
  d: number,
  swiGLU: boolean
): number {
  if (dFF !== null) return dFF
  return swiGLU ? Math.round((8 / 3) * d) : 4 * d
}

/** Round to the nearest multiple of m. */
function roundToMultiple(value: number, m: number): number {
  return Math.round(value / m) * m
}

function roundToAlignedHiddenSize(value: number, alignment: number): number {
  return Math.max(alignment, roundToMultiple(value, alignment))
}

function resolveDefaultGQAKVHeads(heads: number): number {
  const safeHeads =
    Number.isFinite(heads) && heads > 0 ? Math.max(1, Math.floor(heads)) : 1
  let kvHeads = Math.min(8, safeHeads)

  while (kvHeads > 1 && safeHeads % kvHeads !== 0) {
    kvHeads--
  }

  return Math.max(1, kvHeads)
}

export function normalizeAttentionVariantHeads(
  arch: ModelArchitecture
): ModelArchitecture {
  if (!Number.isFinite(arch.a) || arch.a <= 0) {
    return arch
  }

  const queryHeads = Math.max(1, Math.floor(arch.a))

  if (arch.attentionVariant === "mha") {
    return arch.a_kv === queryHeads ? arch : { ...arch, a_kv: queryHeads }
  }

  if (arch.attentionVariant === "mqa") {
    return arch.a_kv === 1 ? arch : { ...arch, a_kv: 1 }
  }

  if (arch.attentionVariant === "mla") {
    return arch.a_kv === null ? arch : { ...arch, a_kv: null }
  }

  if (arch.a_kv === null || arch.a_kv === undefined) {
    return { ...arch, a_kv: resolveDefaultGQAKVHeads(queryHeads) }
  }

  return arch
}

function hasInvalidExplicitHeadDim(arch: ModelArchitecture): boolean {
  return (
    arch.d_head !== null &&
    arch.d_head !== undefined &&
    (!Number.isFinite(arch.d_head) ||
      arch.d_head <= 0 ||
      !Number.isInteger(arch.d_head))
  )
}

export function hasInvalidMoEConfig(moe: MoEConfig, layerCount: number): boolean {
  if (!moe.enabled) {
    return false
  }

  return (
    !Number.isFinite(layerCount) ||
    layerCount <= 0 ||
    !Number.isInteger(layerCount) ||
    !Number.isFinite(moe.E) ||
    moe.E <= 0 ||
    !Number.isInteger(moe.E) ||
    !Number.isFinite(moe.topk) ||
    moe.topk <= 0 ||
    !Number.isInteger(moe.topk) ||
    moe.topk > moe.E ||
    !Number.isFinite(moe.L_moe) ||
    moe.L_moe <= 0 ||
    !Number.isInteger(moe.L_moe) ||
    moe.L_moe > layerCount ||
    !Number.isFinite(moe.E_s) ||
    moe.E_s < 0 ||
    !Number.isInteger(moe.E_s) ||
    !Number.isFinite(moe.loadBalanceFactor) ||
    moe.loadBalanceFactor < 1 ||
    (moe.denseIntermediateSize !== null &&
      (!Number.isFinite(moe.denseIntermediateSize) ||
        moe.denseIntermediateSize <= 0 ||
        !Number.isInteger(moe.denseIntermediateSize))) ||
    (moe.expertIntermediateSize !== null &&
      (!Number.isFinite(moe.expertIntermediateSize) ||
        moe.expertIntermediateSize <= 0 ||
        !Number.isInteger(moe.expertIntermediateSize)))
  )
}

/**
 * Attention projection width used in the PaLM attention term.
 *
 * The current public architecture type does not expose a dedicated attention
 * projection width, so we default to d_model. If a future caller provides an
 * explicit width or head dimension on the runtime object, prefer that.
 */
function resolveAttentionProjectionWidth(arch: ModelArchitecture): number {
  const extendedArch = arch as ModelArchitecture & {
    attentionProjectionWidth?: number | null
    headDim?: number | null
  }

  if (
    extendedArch.attentionProjectionWidth !== null &&
    extendedArch.attentionProjectionWidth !== undefined
  ) {
    return typeof extendedArch.attentionProjectionWidth === "number" &&
      Number.isFinite(extendedArch.attentionProjectionWidth) &&
      extendedArch.attentionProjectionWidth > 0
      ? extendedArch.attentionProjectionWidth
      : Number.POSITIVE_INFINITY
  }

  if (
    arch.d_head !== null &&
    arch.d_head !== undefined
  ) {
    return typeof arch.d_head === "number" &&
      Number.isFinite(arch.d_head) &&
      arch.d_head > 0
      ? arch.a * arch.d_head
      : Number.POSITIVE_INFINITY
  }

  if (
    extendedArch.headDim !== null &&
    extendedArch.headDim !== undefined
  ) {
    return typeof extendedArch.headDim === "number" &&
      Number.isFinite(extendedArch.headDim) &&
      extendedArch.headDim > 0
      ? arch.a * extendedArch.headDim
      : Number.POSITIVE_INFINITY
  }

  return arch.d
}

function getCorrectedChinchillaCoefficients() {
  return (
    CHINCHILLA_COEFFICIENTS.find((row) => row.id === "chinchilla-corrected") ??
    CHINCHILLA_COEFFICIENTS[0]
  )
}

function isFinitePositive(value: number): boolean {
  return Number.isFinite(value) && value > 0
}

function isFinitePositiveInteger(value: number): boolean {
  return isFinitePositive(value) && Number.isInteger(value)
}

function invalidParameterCounts(): ParameterCounts {
  return {
    total: Number.POSITIVE_INFINITY,
    active: Number.POSITIVE_INFINITY,
    embedding: 0,
    outputProjection: 0,
    positionalEmbedding: 0,
    finalNorm: 0,
    perLayer: {
      attention: Number.POSITIVE_INFINITY,
      ffn: 0,
      norm: 0,
    },
    moe: null,
  }
}

function invalidComputeEstimate(): ComputeEstimate {
  return {
    totalFLOPs: Number.POSITIVE_INFINITY,
    flopsPerToken: Number.POSITIVE_INFINITY,
    attentionOverheadFraction: 0,
    simplifiedFormulaAccurate: false,
    moeLoadBalanceFactor: Number.POSITIVE_INFINITY,
  }
}

/** Human-readable token count (e.g. "1.5T", "300B"). */
function formatTokens(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1).replace(/\.0$/, "")}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(1).replace(/\.0$/, "")}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1).replace(/\.0$/, "")}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1).replace(/\.0$/, "")}K`
  return n.toFixed(0)
}

// ─── 1. Parameter Count (Sections 3.1, 3.4) ──────────────────────────────────

/**
 * Calculate total and active parameter counts for a decoder-only transformer.
 *
 * Dense models (Section 3.1):
 *   Per-layer attention (GQA): Ψ_attn = 2d²(1 + a_kv/a)
 *   Per-layer FFN:             Ψ_ffn  = 2·d·d_ff (standard) | 3·d·d_ff (SwiGLU)
 *   Per-layer norm:            4d (LayerNorm) | 2d (RMSNorm)
 *
 * MoE models (Section 3.4):
 *   Total includes all E + E_s experts; active uses topk + E_s.
 *   Active excludes lookup-style non-matmul tables (input/positional embeddings).
 *   Output projection always included in active (matmul, even when tied).
 *
 * @param sequenceLength  Needed for learned positional embeddings; ignored for RoPE/ALiBi.
 */
export function calculateParameterCount(
  arch: ModelArchitecture,
  moe: MoEConfig,
  sequenceLength?: number
): ParameterCounts {
  const normalizedArch = normalizeAttentionVariantHeads(arch)
  const { d, L, a, V, ffnType, normType, posEmbedding, tiedEmbeddings } =
    normalizedArch
  const a_kv = normalizedArch.a_kv ?? a // Default to MHA when null (e.g. MLA)
  if (
    hasInvalidArchitectureConfig(normalizedArch, sequenceLength) ||
    hasInvalidMoEConfig(moe, L)
  ) {
    return invalidParameterCounts()
  }

  // ── Per-layer attention: Q/O use a×d_head; K/V use a_kv×d_head. ──
  const attentionProjectionWidth =
    resolveAttentionProjectionWidth(normalizedArch)
  const attentionPerLayer =
    2 * d * attentionProjectionWidth * (1 + a_kv / a)

  // ── Per-layer norm: 2 norms × (scale + optional bias) ──
  const normPerLayer = normType === "rmsnorm" ? 2 * d : 4 * d

  // ── Non-layer parameters ──
  const embedding = V * d
  const outputProjection = tiedEmbeddings ? 0 : V * d
  const positionalEmbedding =
    posEmbedding === "learned" && Number.isFinite(sequenceLength)
      ? Math.max(0, sequenceLength ?? 0) * d
      : 0
  const finalNorm = normType === "rmsnorm" ? d : 2 * d
  // For compute (FLOPs): lm_head matmul is always V×d, even when tied.
  // Lookup tables (token / learned positional embeddings) are excluded.
  const outputProjForCompute = V * d

  // ── Dense model ──
  if (!moe.enabled) {
    const swiGLU = isSwiGLUStyle(ffnType)
    const effectiveDFF = resolveIntermediateSize(normalizedArch.d_ff, d, swiGLU)
    const ffnPerLayer = computeFFNParams(d, effectiveDFF, swiGLU)
    const perLayerTotal = attentionPerLayer + ffnPerLayer + normPerLayer

    const total =
      L * perLayerTotal +
      embedding +
      outputProjection +
      positionalEmbedding +
      finalNorm

    const active = L * perLayerTotal + outputProjForCompute

    return {
      total,
      active,
      embedding,
      outputProjection,
      positionalEmbedding,
      finalNorm,
      perLayer: {
        attention: attentionPerLayer,
        ffn: ffnPerLayer,
        norm: normPerLayer,
      },
      moe: null,
    }
  }

  // ── MoE model (Section 3.4) ──
  const totalExperts = Number.isFinite(moe.E) ? Math.max(0, moe.E) : 0
  const sharedExperts = Number.isFinite(moe.E_s) ? Math.max(0, moe.E_s) : 0
  const L_moe =
    totalExperts > 0 && Number.isFinite(moe.L_moe)
      ? Math.min(Math.max(0, moe.L_moe), L)
      : 0
  const activeRoutedExperts =
    Number.isFinite(moe.topk) && moe.topk > 0
      ? Math.min(moe.topk, totalExperts)
      : totalExperts
  const L_dense = Math.max(0, L - L_moe)

  // Expert FFN (always SwiGLU for modern MoE architectures)
  const expertDFF = resolveIntermediateSize(moe.expertIntermediateSize, d, true)
  const ffnPerExpert = computeFFNParams(d, expertDFF, true)

  // Dense-layer FFN (preserves the arch's FFN type)
  const denseSwiGLU = isSwiGLUStyle(ffnType)
  const denseDFF = resolveIntermediateSize(
    moe.denseIntermediateSize ?? normalizedArch.d_ff,
    d,
    denseSwiGLU
  )
  const ffnPerDenseLayer = computeFFNParams(d, denseDFF, denseSwiGLU)

  // Router: d × E per MoE layer
  const routerPerMoELayer = d * totalExperts

  const denseLayerParams = attentionPerLayer + ffnPerDenseLayer + normPerLayer

  // Total per MoE layer: all E routed + E_s shared experts
  const moeLayerParams =
    attentionPerLayer +
    (totalExperts + sharedExperts) * ffnPerExpert +
    routerPerMoELayer +
    normPerLayer

  const total =
    L_dense * denseLayerParams +
    L_moe * moeLayerParams +
    embedding +
    outputProjection +
    positionalEmbedding +
    finalNorm

  // Active per MoE layer: topk + E_s experts (Section 3.4 shared-expert rule)
  const activeRoutedExpertParameters =
    L_moe * activeRoutedExperts * ffnPerExpert
  const activeMoELayerParams =
    attentionPerLayer +
    activeRoutedExperts * ffnPerExpert +
    sharedExperts * ffnPerExpert +
    routerPerMoELayer +
    normPerLayer

  const active =
    L_dense * denseLayerParams +
    L_moe * activeMoELayerParams +
    outputProjForCompute

  return {
    total,
    active,
    embedding,
    outputProjection,
    positionalEmbedding,
    finalNorm,
    perLayer: {
      attention: attentionPerLayer,
      ffn: ffnPerDenseLayer,
      norm: normPerLayer,
    },
    moe: {
      expertParameters: L_moe * totalExperts * ffnPerExpert,
      routerParameters: L_moe * routerPerMoELayer,
      sharedExpertParameters: L_moe * sharedExperts * ffnPerExpert,
      activeRoutedExpertParameters,
    },
  }
}

// ─── 2. Quick Estimate (Sections 3.2, 11.1) ──────────────────────────────────

/**
 * Infer a plausible ModelArchitecture from just a total parameter count.
 *
 * 1. Look up heads (a) and layers (L) from the Quick Mode table (Section 11.1).
 * 2. Solve d = √(Ψ / 12L) and round to the nearest multiple of 128.
 * 3. Fill in family defaults (GPT-style < 2B, modern open-weights ≥ 2B).
 *
 * Intentionally approximate: expect 10-20% error for standard models,
 * 20-40% for GQA/large-vocab architectures. Use Preset or Detailed mode
 * for precision.
 */
export function estimateParametersQuick(
  totalParams: number
): ModelArchitecture {
  const safeTotalParams = isFinitePositive(totalParams) ? totalParams : 1e9
  const row =
    QUICK_MODE_LOOKUP.find(
      (r) => safeTotalParams >= r.minParams && safeTotalParams < r.maxParams
    ) ?? QUICK_MODE_LOOKUP[QUICK_MODE_LOOKUP.length - 1]

  const L = row.layers
  const a = row.heads
  const isModern = row.family === "modern-open-weights"

  // Section 3.2 / 11.1: Ψ ≈ 12Ld² → d = √(Ψ / 12L), rounded to nearest 128.
  const d = roundToAlignedHiddenSize(Math.sqrt(safeTotalParams / (12 * L)), 128)

  const d_ff = isModern ? roundToAlignedHiddenSize((8 / 3) * d, 128) : 4 * d
  const a_kv = isModern ? resolveDefaultGQAKVHeads(a) : a

  return {
    d,
    L,
    a,
    a_kv,
    d_ff,
    V: isModern ? 128000 : 50000,
    ffnType: isModern ? "swiglu" : "standard",
    normType: isModern ? "rmsnorm" : "layernorm",
    posEmbedding: isModern ? "rope" : "learned",
    attentionVariant: a_kv < a ? "gqa" : "mha",
    tiedEmbeddings: !isModern,
  }
}

// ─── 3. FLOPs Estimation (Section 4.1) ───────────────────────────────────────

/**
 * Total training FLOPs using the PaLM formula:
 *
 *   C = (6·Ψ_active + 12·L·d_attn·s) × D
 *
 * - Ψ_active excludes lookup-style embedding tables (input / learned positional).
 * - d_attn = n_heads × d_head (not necessarily d_model).
 * - MoE: load_balance_factor applied only to expert FLOPs.
 */
export function calculateFLOPs(
  params: ParameterCounts,
  config: { totalTokens: number; sequenceLength: number },
  arch: ModelArchitecture,
  moe: MoEConfig
): ComputeEstimate {
  const { totalTokens: D, sequenceLength: s } = config
  const normalizedArch = normalizeAttentionVariantHeads(arch)
  const { L } = normalizedArch

  if (
    hasInvalidArchitectureConfig(normalizedArch, s) ||
    hasInvalidMoEConfig(moe, L) ||
    !isFinitePositiveInteger(D) ||
    !isFinitePositiveInteger(s)
  ) {
    return invalidComputeEstimate()
  }

  const attentionProjectionWidth =
    resolveAttentionProjectionWidth(normalizedArch)

  // ── Model FLOPs per token (6Ψ_active term) ──
  const baseModelFLOPsPerToken = isFinitePositive(params.active)
    ? 6 * params.active
    : Number.POSITIVE_INFINITY
  let modelFLOPsPerToken = baseModelFLOPsPerToken
  let effectiveLoadBalanceFactor = 1.0

  if (moe.enabled && params.moe && Number.isFinite(modelFLOPsPerToken)) {
    const routedExpertActiveParams = Number.isFinite(
      params.moe.activeRoutedExpertParameters
    )
      ? Math.max(0, params.moe.activeRoutedExpertParameters)
      : 0
    effectiveLoadBalanceFactor = isFinitePositive(moe.loadBalanceFactor)
      ? Math.max(1, moe.loadBalanceFactor)
      : 1

    // Load-balance overhead applies only to routed expert FLOPs.
    // Shared experts are always active and should not be inflated.
    modelFLOPsPerToken +=
      6 * routedExpertActiveParams * (effectiveLoadBalanceFactor - 1)
  }

  // ── Attention quadratic FLOPs per token (12·L·d_attn·s) ──
  const attentionFLOPsPerToken =
    isFinitePositive(L) &&
    isFinitePositive(attentionProjectionWidth) &&
    isFinitePositiveInteger(s)
      ? 12 * L * attentionProjectionWidth * s
      : Number.POSITIVE_INFINITY

  // ── Totals ──
  const flopsPerToken =
    Number.isFinite(modelFLOPsPerToken) &&
    Number.isFinite(attentionFLOPsPerToken)
      ? modelFLOPsPerToken + attentionFLOPsPerToken
      : Number.POSITIVE_INFINITY
  const totalFLOPs =
    isFinitePositiveInteger(D) && Number.isFinite(flopsPerToken)
      ? flopsPerToken * D
      : Number.POSITIVE_INFINITY
  const attentionOverheadFraction =
    Number.isFinite(modelFLOPsPerToken) &&
    modelFLOPsPerToken > 0 &&
    Number.isFinite(attentionFLOPsPerToken)
      ? attentionFLOPsPerToken / modelFLOPsPerToken
      : 0

  return {
    totalFLOPs,
    flopsPerToken,
    attentionOverheadFraction,
    // Treat 6ΨD as planning-accurate only when the omitted attention term is
    // small relative to model FLOPs.
    simplifiedFormulaAccurate: attentionOverheadFraction <= 0.1,
    moeLoadBalanceFactor: effectiveLoadBalanceFactor,
  }
}

// ─── 4. Chinchilla Scaling-Law Analysis (Section 4.3) ────────────────────────

/**
 * Select the coefficient row from the sensitivity table that best matches
 * the user's D/N ratio. Ranges are contiguous from 0 to ∞.
 */
function selectCoefficientRow(ratio: number) {
  const selectable = CHINCHILLA_COEFFICIENTS.filter((c) => c.autoSelectable)
  for (const row of selectable) {
    const min = row.autoSelectMinDNRatio ?? -Infinity
    const max = row.autoSelectMaxDNRatio ?? Infinity
    if (ratio >= min && ratio <= max) return row
  }

  return getCorrectedChinchillaCoefficients()
}

/**
 * Chinchilla scaling-law analysis: predicted loss, compute-optimal allocation,
 * and practical recommendation.
 *
 * @param totalParams  Model parameters Ψ (use Ψ_active for MoE).
 * @param tokens       Total training tokens D.
 * @param uniqueTokens Unique tokens U (optional; caps the loss data term under repetition).
 */
export function calculateChinchillaAnalysis(
  totalParams: number,
  tokens: number,
  uniqueTokens?: number
): ChinchillaAnalysis {
  if (
    !Number.isFinite(totalParams) ||
    totalParams <= 0 ||
    !isFinitePositiveInteger(tokens) ||
    (uniqueTokens !== undefined && !isFinitePositiveInteger(uniqueTokens))
  ) {
    const fallbackRow = getCorrectedChinchillaCoefficients()

    return {
      parameterCount: Number.NaN,
      ratio: Number.NaN,
      recommendedTokenCount: Number.NaN,
      powerLawOptimalTokens: Number.NaN,
      optimalModelSize: Number.NaN,
      predictedLossNats: Number.NaN,
      effectiveLossTokens: Number.NaN,
      coefficientRowId: fallbackRow.id,
      coefficientRowLabel: fallbackRow.label,
      recommendation:
        "Enter positive finite parameter and token counts to compute scaling-law guidance.",
    }
  }

  const N = totalParams
  const D = tokens
  const repeatedUniqueTokens =
    uniqueTokens !== undefined &&
    Number.isFinite(uniqueTokens) &&
    uniqueTokens > 0 &&
    uniqueTokens < D
      ? uniqueTokens
      : null
  const hasRepeatedData = repeatedUniqueTokens !== null
  const effectiveLossTokens = hasRepeatedData
    ? Math.min(D, 16 * repeatedUniqueTokens)
    : D
  const tokensPerParamRatio = D / N
  const twentyXTokenCount = 20 * N
  const chinchillaRatio = D / twentyXTokenCount

  const row = selectCoefficientRow(tokensPerParamRatio)
  const { alpha, beta, A, B, E } = row

  // ── Loss prediction: L(N,D_eff) = E + A/N^α + B/D_eff^β ──
  const predictedLossNats =
    E + A / Math.pow(N, alpha) + B / Math.pow(effectiveLossTokens, beta)

  // ── Power-law optimal: D_opt = 8.62 × N^1.041 (Section 4.3) ──
  const powerLawOptimalTokens = 8.62 * Math.pow(N, 1.041)
  const recommendedTokenCount = powerLawOptimalTokens
  const recommendationTokens = effectiveLossTokens
  const recommendedRatio = recommendationTokens / recommendedTokenCount
  const usesDiscountedRecommendation = recommendationTokens < D
  const discountedRecommendationNote = usesDiscountedRecommendation
    ? " after repeated-data discounting"
    : ""
  const recommendationTokenLabel = usesDiscountedRecommendation
    ? "effective tokens"
    : "tokens"

  // ── Exact compute-optimal allocation for implied budget C = 6ND ──
  // Section 4.3 specifies using the corrected Epoch AI coefficients here.
  const corrected = getCorrectedChinchillaCoefficients()
  const correctedAlpha = corrected.alpha
  const correctedBeta = corrected.beta
  const correctedA = corrected.A
  const correctedB = corrected.B
  const C = 6 * N * D
  const sumExp = correctedAlpha + correctedBeta
  const optimalModelSize =
    Math.pow(
      (correctedAlpha * correctedA) / (correctedBeta * correctedB),
      1 / sumExp
    ) * Math.pow(C / 6, correctedBeta / sumExp)
  const optimalTokenCount =
    Math.pow(
      (correctedBeta * correctedB) / (correctedAlpha * correctedA),
      1 / sumExp
    ) * Math.pow(C / 6, correctedAlpha / sumExp)

  // ── Recommendation ──
  const recommendationParts: string[] = []

  if (recommendedRatio < 1) {
    recommendationParts.push(
      `Undertrained relative to Chinchilla${discountedRecommendationNote}. A better target is roughly ${formatTokens(powerLawOptimalTokens)} ${recommendationTokenLabel}.`
    )
  } else if (recommendedRatio <= 1.5) {
    recommendationParts.push(
      `Near the Chinchilla compute-optimal frontier${discountedRecommendationNote}.`
    )
  } else if (recommendedRatio <= 25) {
    recommendationParts.push(
      `Above the compute-optimal frontier${discountedRecommendationNote}. This can be a deliberate inference-efficiency tradeoff when the extra tokens are sufficiently fresh.`
    )
  } else {
    recommendationParts.push(
      `Far beyond the original Chinchilla regime${discountedRecommendationNote}. Loss estimates are lower-confidence at this overtraining ratio.`
    )
  }

  if (D < 200e9) {
    recommendationParts.push(
      "Training on fewer than 200B tokens is usually a practical quality floor, even when the Chinchilla ratio looks acceptable."
    )
  }

  recommendationParts.push(
    `The simple 20x reference is ${formatTokens(twentyXTokenCount)} tokens; the power-law recommendation is ${formatTokens(powerLawOptimalTokens)}.`
  )

  recommendationParts.push(
    `At the same training-compute budget, the exact corrected Chinchilla allocation is about ${formatTokens(optimalModelSize)} parameters and ${formatTokens(optimalTokenCount)} tokens.`
  )

  recommendationParts.push(
    "Predicted loss is calibrated to MassiveText-style data, so use the nats value for relative planning rather than as an absolute target for a different data mix."
  )

  // Note data repetition if relevant
  if (hasRepeatedData) {
    const epochs = D / repeatedUniqueTokens

    if (epochs > 16) {
      recommendationParts.push(
        `Loss prediction is capped at about ${formatTokens(effectiveLossTokens)} effective tokens because repeated data past ~16x unique tokens provides negligible marginal value.`
      )
    } else if (epochs > 4) {
      recommendationParts.push(
        `Data repeats about ${epochs.toFixed(0)}x; the loss prediction assumes unique data and becomes less reliable when D >> U.`
      )
    }
  }

  return {
    parameterCount: N,
    ratio: chinchillaRatio,
    recommendedTokenCount,
    powerLawOptimalTokens,
    optimalModelSize,
    predictedLossNats,
    effectiveLossTokens,
    coefficientRowId: row.id,
    coefficientRowLabel: row.label,
    recommendation: recommendationParts.join(" "),
  }
}

// ─── 5. Critical Batch Size (Section 4.4) ────────────────────────────────────

/**
 * Critical batch size and batch-efficiency metrics (McCandlish et al., 2018;
 * Kaplan et al., 2020).
 *
 *   B_crit(L) = B* / L^(1/α_B)
 *
 * where B* = 2×10⁸ tokens, α_B = 0.21.
 *
 * @param loss        Predicted training loss (nats) from the Chinchilla formula.
 * @param batchTokens Actual global batch size in tokens = b × s × G × N_dp.
 */
export function calculateCriticalBatchSize(
  loss: number,
  batchTokens: number
): BatchSizeAnalysis {
  const B_STAR = 2.0e8
  const ALPHA_B = 0.21

  if (
    !Number.isFinite(loss) ||
    loss <= 0 ||
    !isFinitePositiveInteger(batchTokens)
  ) {
    return {
      criticalBatchTokens: Number.NaN,
      actualBatchTokens: Number.NaN,
      relation: "unknown",
      computeMultiplier: Number.NaN,
      wastedComputeFraction: Number.NaN,
    }
  }

  // B_crit = B* / L^(1/α_B)
  const criticalBatchTokens = B_STAR / Math.pow(loss, 1 / ALPHA_B)

  // Compute multiplier above optimum: C/C_min = 1 + B_tok/B_crit
  const computeMultiplier = 1 + batchTokens / criticalBatchTokens

  // Wasted-compute fraction: B_tok / (B_tok + B_crit)
  const wastedComputeFraction =
    batchTokens / (batchTokens + criticalBatchTokens)

  const batchRatio = batchTokens / criticalBatchTokens
  let relation: BatchSizeAnalysis["relation"]
  if (batchRatio < 0.5) {
    relation = "below"
  } else if (batchRatio > 2) {
    relation = "above"
  } else {
    relation = "near"
  }

  return {
    criticalBatchTokens,
    actualBatchTokens: batchTokens,
    relation,
    computeMultiplier,
    wastedComputeFraction,
  }
}

// ─── 6. Data Repetition Analysis (Section 4.5) ──────────────────────────────

/**
 * Assess data repetition severity and effective data ceiling.
 *
 * Thresholds (Muennighoff et al., 2023):
 *   < 1 epoch:  none     — sub-epoch pass over a larger corpus
 *   = 1 epoch:  none     — one unique pass
 *   ≤ 4 epochs: info     — near-full value
 *   ≤ 40 epochs: warning — rapidly diminishing returns
 *   > 40 epochs: critical — past effective ceiling
 *
 * Effective data ceiling: ~16× unique tokens (additional repeats ≈ zero value).
 */
export function analyzeDataRepetition(
  totalTokens: number,
  uniqueTokens: number
): DataRepetitionAnalysis {
  if (
    !isFinitePositiveInteger(totalTokens) ||
    !isFinitePositiveInteger(uniqueTokens)
  ) {
    return {
      epochs: Number.NaN,
      hasRepetition: false,
      severity: "critical",
      effectiveDataCeiling: Number.NaN,
      recommendation:
        "Enter positive finite total and unique token counts to analyze data repetition.",
    }
  }

  const epochs = totalTokens / uniqueTokens
  const hasRepetition = epochs > 1
  // Maximum effective contribution saturates at ~16× unique tokens
  const effectiveDataCeiling = 16 * uniqueTokens

  let severity: DataRepetitionAnalysis["severity"]
  let recommendation: string

  if (epochs < 1) {
    severity = "none"
    recommendation = `Less than one epoch (${epochs.toFixed(2)}x) over the available corpus; no data repetition is modeled.`
  } else if (epochs === 1) {
    severity = "none"
    recommendation = "One epoch over unique data; no data repetition is modeled."
  } else if (epochs <= 4) {
    severity = "info"
    recommendation = `Training for ${epochs.toFixed(1)} epochs. Repetition has near-full value at this level.`
  } else if (epochs <= 16) {
    severity = "warning"
    recommendation =
      `Training for ${epochs.toFixed(1)} epochs — diminishing returns from repeated data. ` +
      "Consider acquiring more unique data or reducing total tokens."
  } else if (epochs <= 40) {
    severity = "warning"
    recommendation =
      `Training for ${epochs.toFixed(1)} epochs — you are past the effective data ceiling of ~16x unique data. ` +
      `Additional training beyond ~${formatTokens(effectiveDataCeiling)} provides negligible benefit.`
  } else {
    severity = "critical"
    recommendation =
      `Training for ${epochs.toFixed(0)} epochs — well past the effective ceiling of ~16x unique tokens. ` +
      `Effective training capped at ~${formatTokens(effectiveDataCeiling)}.`
  }

  return {
    epochs,
    hasRepetition,
    severity,
    effectiveDataCeiling,
    recommendation,
  }
}
