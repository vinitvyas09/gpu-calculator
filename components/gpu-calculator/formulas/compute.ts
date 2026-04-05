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

function greatestCommonDivisor(a: number, b: number): number {
  let x = Math.abs(a)
  let y = Math.abs(b)

  while (y !== 0) {
    const remainder = x % y
    x = y
    y = remainder
  }

  return x || 1
}

function leastCommonMultiple(a: number, b: number): number {
  return Math.abs(a * b) / greatestCommonDivisor(a, b)
}

function roundToCompatibleHiddenSize(
  value: number,
  alignment: number,
  heads: number
): number {
  const compatibleMultiple =
    heads > 0 ? leastCommonMultiple(alignment, heads) : alignment

  return Math.max(compatibleMultiple, roundToMultiple(value, compatibleMultiple))
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
    d_head?: number | null
  }

  if (
    typeof extendedArch.attentionProjectionWidth === "number" &&
    Number.isFinite(extendedArch.attentionProjectionWidth) &&
    extendedArch.attentionProjectionWidth > 0
  ) {
    return extendedArch.attentionProjectionWidth
  }

  const explicitHeadDim =
    extendedArch.d_head ?? extendedArch.headDim ?? null

  if (
    typeof explicitHeadDim === "number" &&
    Number.isFinite(explicitHeadDim) &&
    explicitHeadDim > 0
  ) {
    return arch.a * explicitHeadDim
  }

  return arch.d
}

function getCorrectedChinchillaCoefficients() {
  return (
    CHINCHILLA_COEFFICIENTS.find((row) => row.id === "chinchilla-corrected") ??
    CHINCHILLA_COEFFICIENTS[0]
  )
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
 *   Input embedding excluded from active (lookup, 0 FLOPs).
 *   Output projection always included in active (matmul, even when tied).
 *
 * @param sequenceLength  Needed for learned positional embeddings; ignored for RoPE/ALiBi.
 */
export function calculateParameterCount(
  arch: ModelArchitecture,
  moe: MoEConfig,
  sequenceLength?: number
): ParameterCounts {
  const { d, L, a, V, ffnType, normType, posEmbedding, tiedEmbeddings } = arch
  const a_kv = arch.a_kv ?? a // Default to MHA when null (e.g. MLA)

  // ── Per-layer attention: Q(d²) + K(d²·a_kv/a) + V(d²·a_kv/a) + O(d²) ──
  const attentionPerLayer = 2 * d * d * (1 + a_kv / a)

  // ── Per-layer norm: 2 norms × (scale + optional bias) ──
  const normPerLayer = normType === "rmsnorm" ? 2 * d : 4 * d

  // ── Non-layer parameters ──
  const embedding = V * d
  const outputProjection = tiedEmbeddings ? 0 : V * d
  const positionalEmbedding =
    posEmbedding === "learned" ? (sequenceLength ?? 0) * d : 0
  const finalNorm = normType === "rmsnorm" ? d : 2 * d
  // For compute (FLOPs): lm_head matmul is always V×d, even when tied
  const outputProjForCompute = V * d

  // ── Dense model ──
  if (!moe.enabled) {
    const swiGLU = isSwiGLUStyle(ffnType)
    const effectiveDFF = resolveIntermediateSize(arch.d_ff, d, swiGLU)
    const ffnPerLayer = computeFFNParams(d, effectiveDFF, swiGLU)
    const perLayerTotal = attentionPerLayer + ffnPerLayer + normPerLayer

    const total =
      L * perLayerTotal +
      embedding +
      outputProjection +
      positionalEmbedding +
      finalNorm

    // Active: exclude input embedding (lookup = 0 FLOPs), include output matmul
    const active =
      L * perLayerTotal +
      outputProjForCompute +
      positionalEmbedding +
      finalNorm

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
  const { E, topk, L_moe, E_s } = moe
  const L_dense = L - L_moe

  // Expert FFN (always SwiGLU for modern MoE architectures)
  const expertDFF = resolveIntermediateSize(moe.expertIntermediateSize, d, true)
  const ffnPerExpert = computeFFNParams(d, expertDFF, true)

  // Dense-layer FFN (preserves the arch's FFN type)
  const denseSwiGLU = isSwiGLUStyle(ffnType)
  const denseDFF = resolveIntermediateSize(
    moe.denseIntermediateSize ?? arch.d_ff,
    d,
    denseSwiGLU
  )
  const ffnPerDenseLayer = computeFFNParams(d, denseDFF, denseSwiGLU)

  // Router: d × E per MoE layer
  const routerPerMoELayer = d * E

  const denseLayerParams = attentionPerLayer + ffnPerDenseLayer + normPerLayer

  // Total per MoE layer: all E routed + E_s shared experts
  const moeLayerParams =
    attentionPerLayer +
    (E + E_s) * ffnPerExpert +
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
  const activeMoELayerParams =
    attentionPerLayer +
    (topk + E_s) * ffnPerExpert +
    routerPerMoELayer +
    normPerLayer

  const active =
    L_dense * denseLayerParams +
    L_moe * activeMoELayerParams +
    outputProjForCompute +
    positionalEmbedding +
    finalNorm

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
      expertParameters: L_moe * E * ffnPerExpert,
      routerParameters: L_moe * routerPerMoELayer,
      sharedExpertParameters: L_moe * E_s * ffnPerExpert,
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
  const row =
    QUICK_MODE_LOOKUP.find(
      (r) => totalParams >= r.minParams && totalParams < r.maxParams
    ) ?? QUICK_MODE_LOOKUP[QUICK_MODE_LOOKUP.length - 1]

  const L = row.layers
  const a = row.heads
  const isModern = row.family === "modern-open-weights"

  // Section 3.2: Ψ ≈ 12Ld² → d = √(Ψ / 12L), aligned to 128 and head-divisible.
  const d = roundToCompatibleHiddenSize(
    Math.sqrt(totalParams / (12 * L)),
    128,
    a
  )

  const d_ff = isModern ? Math.round((8 / 3) * d) : 4 * d
  const a_kv = isModern ? Math.min(8, a) : a

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
 * - Ψ_active excludes the input embedding (lookup = 0 FLOPs).
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
  const { L } = arch
  const attentionProjectionWidth = resolveAttentionProjectionWidth(arch)

  // ── Model FLOPs per token (6Ψ_active term) ──
  let modelFLOPsPerToken = 6 * params.active
  let effectiveLoadBalanceFactor = 1.0

  if (moe.enabled && params.moe) {
    const routedExpertActiveParams =
      moe.E > 0 ? params.moe.expertParameters * (moe.topk / moe.E) : 0
    effectiveLoadBalanceFactor = moe.loadBalanceFactor

    // Load-balance overhead applies only to routed expert FLOPs.
    // Shared experts are always active and should not be inflated.
    modelFLOPsPerToken +=
      6 * routedExpertActiveParams * (effectiveLoadBalanceFactor - 1)
  }

  // ── Attention quadratic FLOPs per token (12·L·d_attn·s) ──
  const attentionFLOPsPerToken = 12 * L * attentionProjectionWidth * s

  // ── Totals ──
  const flopsPerToken = modelFLOPsPerToken + attentionFLOPsPerToken
  const totalFLOPs = flopsPerToken * D

  return {
    totalFLOPs,
    flopsPerToken,
    attentionOverheadFraction: attentionFLOPsPerToken / flopsPerToken,
    // Simplified 6ΨD is accurate when d > s/12 (Section 4.1)
    simplifiedFormulaAccurate: attentionProjectionWidth > s / 12,
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
    if (ratio >= min && ratio < max) return row
  }
  // Fallback: corrected Chinchilla (should not reach here — ranges cover [0, ∞))
  return CHINCHILLA_COEFFICIENTS[0]
}

/**
 * Chinchilla scaling-law analysis: predicted loss, compute-optimal allocation,
 * and practical recommendation.
 *
 * @param totalParams  Model parameters Ψ (use Ψ_active for MoE).
 * @param tokens       Total training tokens D.
 * @param uniqueTokens Unique tokens U (optional; affects recommendation text only).
 */
export function calculateChinchillaAnalysis(
  totalParams: number,
  tokens: number,
  uniqueTokens?: number
): ChinchillaAnalysis {
  const N = totalParams
  const D = tokens
  const tokensPerParamRatio = D / N
  const chinchillaRatio = D / (20 * N)

  const row = selectCoefficientRow(tokensPerParamRatio)
  const { alpha, beta, A, B, E } = row

  // ── Loss prediction: L(N,D) = E + A/N^α + B/D^β ──
  const predictedLossNats =
    E + A / Math.pow(N, alpha) + B / Math.pow(D, beta)

  // ── Chinchilla 20× rule ──
  const recommendedTokenCount = 20 * N

  // ── Power-law optimal: D_opt = 8.62 × N^1.041 (Section 4.3) ──
  const powerLawOptimalTokens = 8.62 * Math.pow(N, 1.041)

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

  if (chinchillaRatio < 1) {
    recommendationParts.push(
      `Undertrained relative to Chinchilla. A better target is roughly ${formatTokens(powerLawOptimalTokens)} tokens.`
    )
  } else if (chinchillaRatio <= 1.5) {
    recommendationParts.push(
      "Near the Chinchilla compute-optimal frontier."
    )
  } else if (chinchillaRatio <= 25) {
    recommendationParts.push(
      "Above the compute-optimal frontier. This can be a deliberate inference-efficiency tradeoff."
    )
  } else {
    recommendationParts.push(
      "Far beyond the original Chinchilla regime. Loss estimates are lower-confidence at this overtraining ratio."
    )
  }

  if (D < 200e9) {
    recommendationParts.push(
      "Training on fewer than 200B tokens is usually a practical quality floor, even when the Chinchilla ratio looks acceptable."
    )
  }

  recommendationParts.push(
    `At the same training-compute budget, the exact corrected Chinchilla allocation is about ${formatTokens(optimalModelSize)} parameters and ${formatTokens(optimalTokenCount)} tokens.`
  )

  // Note data repetition if relevant
  if (uniqueTokens !== undefined && uniqueTokens < D) {
    const epochs = D / uniqueTokens
    if (epochs > 4) {
      recommendationParts.push(
        `Data repeats about ${epochs.toFixed(0)}x; the loss prediction assumes unique data and becomes less reliable when D >> U.`
      )
    }
  }

  return {
    ratio: chinchillaRatio,
    recommendedTokenCount,
    powerLawOptimalTokens,
    optimalModelSize,
    predictedLossNats,
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

  // B_crit = B* / L^(1/α_B)
  const criticalBatchTokens = B_STAR / Math.pow(loss, 1 / ALPHA_B)

  // Compute multiplier above optimum: C/C_min = 1 + B_tok/B_crit
  const computeMultiplier = 1 + batchTokens / criticalBatchTokens

  // Wasted-compute fraction: B_tok / (B_tok + B_crit)
  const wastedComputeFraction =
    batchTokens / (batchTokens + criticalBatchTokens)

  // Classify relation with 10% tolerance band for "at"
  const batchRatio = batchTokens / criticalBatchTokens
  let relation: BatchSizeAnalysis["relation"]
  if (batchRatio < 0.9) {
    relation = "below"
  } else if (batchRatio > 1.1) {
    relation = "above"
  } else {
    relation = "at"
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
 *   ≤ 1 epoch:  none     — all unique
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
  const epochs = totalTokens / uniqueTokens
  const hasRepetition = epochs > 1
  // Maximum effective contribution saturates at ~16× unique tokens
  const effectiveDataCeiling = 16 * uniqueTokens

  let severity: DataRepetitionAnalysis["severity"]
  let recommendation: string

  if (epochs <= 1) {
    severity = "none"
    recommendation = "No data repetition — all tokens are unique."
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
