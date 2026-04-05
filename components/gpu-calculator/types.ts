// ─── Model Architecture ──────────────────────────────────────────────────────

/** FFN type determines parameter count and FLOP formulas (Section 3.1) */
export type FFNType = "standard" | "swiglu" | "geglu"

/** Normalization type affects per-layer param count (Section 3.1) */
export type NormType = "layernorm" | "rmsnorm"

/** Positional embedding type (Section 3.1) */
export type PositionalEmbeddingType = "learned" | "rope" | "alibi" | "none"

/** Full model architecture specification (Section 3.1, 2) */
export interface ModelArchitecture {
  /** d — hidden dimension (d_model) */
  d: number
  /** L — number of transformer layers */
  L: number
  /** a (n_heads) — number of attention heads */
  a: number
  /** a_kv (n_kv) — number of KV heads (GQA/MQA) */
  a_kv: number
  /** d_ff — FFN intermediate dimension */
  d_ff: number
  /** V — vocabulary size */
  V: number
  /** s — sequence length */
  s: number
  /** FFN type: standard, SwiGLU, or GeGLU */
  ffnType: FFNType
  /** Normalization type */
  normType: NormType
  /** Positional embedding type */
  posEmbedding: PositionalEmbeddingType
  /** Whether input/output embeddings share weights */
  tiedEmbeddings: boolean
}

/** MoE configuration (Section 3.4) */
export interface MoEConfig {
  /** Whether MoE is enabled */
  enabled: boolean
  /** E — total number of experts per MoE layer */
  E: number
  /** topk — experts activated per token */
  topk: number
  /** L_moe — number of MoE layers (may be < L) */
  L_moe: number
  /** E_s — shared (always-active) experts, e.g. DeepSeek-v2/v3 */
  E_s: number
  /** Load balance overhead factor (default 1.1 = 10%) */
  loadBalanceFactor: number
}

// ─── GPU Hardware ────────────────────────────────────────────────────────────

/** GPU interconnect type */
export type InterconnectType = "nvlink" | "pcie" | "none"

/** GPU vendor */
export type GPUVendor = "nvidia" | "amd" | "apple"

/** GPU category for grouping in selector */
export type GPUCategory = "datacenter" | "consumer" | "amd" | "apple"

/** GPU hardware specification (Section 7) */
export interface GPUSpec {
  /** Display name */
  name: string
  /** Vendor */
  vendor: GPUVendor
  /** Category for UI grouping */
  category: GPUCategory
  /** VRAM in GB */
  vram: number
  /** BF16 tensor core TFLOPS (dense) */
  tflops_bf16: number
  /** TF32 tensor core TFLOPS (dense); null if not supported */
  tflops_tf32: number | null
  /** FP8 tensor core TFLOPS (dense); null if not supported */
  tflops_fp8: number | null
  /** HBM bandwidth in GB/s */
  memBandwidth: number
  /** NVLink bidirectional bandwidth in GB/s; null if PCIe only */
  nvlinkBandwidth: number | null
  /** Thermal design power in watts */
  tdp: number
  /** Typical GPUs per node (8 for DGX, 1 for consumer) */
  gpusPerNode: number
  /** Interconnect type */
  interconnect: InterconnectType
}

// ─── Training Configuration ──────────────────────────────────────────────────

/** Training precision (Section 11.2, input 3) */
export type TrainingPrecision = "fp32" | "bf16" | "fp16" | "fp8"

/** Optimizer choice (Section 5.1) */
export type OptimizerType =
  | "adamw_fp32"
  | "adamw_mixed"
  | "adamw_8bit"
  | "adamw_fp8"
  | "adam_mini"
  | "sgd_momentum"
  | "sgd_no_momentum"
  | "adafactor"
  | "lion"
  | "lamb"
  | "mezo"

/** Gradient accumulation precision (Section 11.2, input 4a) */
export type GradientPrecision = "fp32" | "bf16"

/** Activation checkpointing mode (Section 11.2, input 8) */
export type CheckpointingMode = "none" | "selective" | "full" | "partial"

/** ZeRO/FSDP sharding stage (Section 5.2) */
export type ZeROStage = 0 | 1 | 2 | 3

/** FSDP sharding strategy names (map to ZeRO equivalents, Section 5.2) */
export type FSDPStrategy =
  | "NO_SHARD"
  | "SHARD_GRAD_OP"
  | "FULL_SHARD"
  | "HYBRID_SHARD"
  | "HYBRID_SHARD_ZERO2"

/** Distributed training framework (Section 11.2, input 21) */
export type FrameworkType = "megatron" | "deepspeed" | "fsdp" | "hf_trainer"

/** CPU offloading mode (Section 5.2, input 24) */
export type CPUOffloadMode = "none" | "optimizer" | "optimizer_params"

/** FP8 backend mode (Section 5.1, input 29) */
export type FP8Mode = "transformer_engine" | "ms_amp"

/** KV cache precision for post-training generation (Section 11.2, input 30) */
export type KVCachePrecision = "bf16" | "fp16" | "int8"

/** Inter-node bandwidth preset (Section 7, input 26) */
export type InterNodeBandwidthPreset = "hdr_200" | "ndr_400" | "custom"

/** Parallelism configuration (Section 9) */
export interface ParallelismConfig {
  /** N_tp — tensor parallel degree */
  N_tp: number
  /** N_pp — pipeline parallel degree */
  N_pp: number
  /** N_dp — data parallel degree */
  N_dp: number
  /** N_cp — context parallel degree */
  N_cp: number
  /** N_ep — expert parallel degree (MoE) */
  N_ep: number
  /** ZeRO/FSDP sharding stage */
  zeroStage: ZeROStage
  /** Distributed training framework */
  framework: FrameworkType
  /** Sequence parallelism (= N_tp when enabled) */
  sequenceParallelism: boolean
  /** Virtual pipeline chunks for interleaved PP schedule */
  VP: number
}

/** Full pretraining configuration — maps to all inputs in Section 11.2 */
export interface TrainingConfig {
  // ─── Model specification (input 1) ───
  /** Input mode for model specification */
  modelInputMode: "quick" | "preset" | "detailed"
  /** Model architecture (filled by preset, quick estimate, or user) */
  architecture: ModelArchitecture
  /** MoE configuration (input 18) */
  moe: MoEConfig

  // ─── Dataset (inputs 2, 2a) ───
  /** D — total training tokens (may include repeats) */
  totalTokens: number
  /** U — unique training tokens (defaults to D) */
  uniqueTokens: number

  // ─── Precision & Optimizer (inputs 3, 4, 4a) ───
  /** Training precision */
  precision: TrainingPrecision
  /** Optimizer type */
  optimizer: OptimizerType
  /** Gradient accumulation precision */
  gradientPrecision: GradientPrecision

  // ─── Batch & Sequence (inputs 5, 6, 7) ───
  /** b — micro-batch size per GPU */
  microBatchSize: number
  /** s — sequence length */
  sequenceLength: number
  /** G — gradient accumulation steps */
  gradAccumSteps: number

  // ─── Memory optimizations (inputs 8, 9) ───
  /** Activation checkpointing mode */
  checkpointing: CheckpointingMode
  /** Flash Attention on/off */
  flashAttention: boolean

  // ─── Hardware (inputs 10, 11, 12) ───
  /** GPU specification */
  gpu: GPUSpec
  /** Number of GPUs (N_gpu); null = auto-compute minimum */
  numGPUs: number | null
  /** Target training time in days; null = compute from GPU count */
  targetTrainingDays: number | null

  // ─── Efficiency (input 13) ───
  /** MFU override (0.10 to 0.70); null = use smart default */
  mfuOverride: number | null

  // ─── Parallelism (input 14) ───
  /** Whether to auto-recommend parallelism or use manual config */
  parallelismMode: "auto" | "manual"
  /** Parallelism configuration */
  parallelism: ParallelismConfig

  // ─── Cost (input 15) ───
  /** Cost per GPU-hour in USD */
  costPerGPUHour: number

  // ─── Advanced inputs (16-32) ───
  /** N_recomp — partial checkpointing depth (input 19) */
  partialCheckpointDepth: number | null
  /** AMP autocast toggle (input 23) */
  ampAutocast: boolean
  /** CPU offloading mode (input 24) */
  cpuOffload: CPUOffloadMode
  /** ZeRO communication bucket mode (input 25) */
  overlapComm: boolean
  /** Inter-node bandwidth in GB/s (input 26) */
  interNodeBandwidth: number
  /** torch.compile toggle (input 27) */
  torchCompile: boolean
  /** Chunked cross-entropy toggle (input 28) */
  chunkedCrossEntropy: boolean
  /** FP8 effective kernel speedup factor (input 29) */
  fp8SpeedupFactor: number
  /** FP8 backend mode (input 29) */
  fp8Mode: FP8Mode
  /** KV cache precision (input 30) */
  kvCachePrecision: KVCachePrecision
  /** Checkpoint retention count (input 31) */
  checkpointRetention: number
  /** f — failure rate (failures/instance/day) (input 32) */
  failureRate: number
  /** t_recovery — recovery time in hours (input 32) */
  recoveryTimeHours: number
  /** f_checkpoint — checkpoint saves per day (input 32) */
  checkpointFrequency: number
  /** Storage price $/GB/month (Section 8.2) */
  storagePricePerGBMonth: number
}

// ─── Post-Training Configuration ─────────────────────────────────────────────

/** Post-training method (Section 11.3) */
export type PostTrainingMethod = "sft" | "dpo" | "ppo" | "grpo"

/** Fine-tuning approach (Section 11.3, input 3) */
export type FineTuningApproach = "full" | "lora" | "qlora" | "mezo"

/** LoRA configuration (Section 11.3, input 4) */
export interface LoRAConfig {
  /** LoRA rank r */
  rank: number
  /** LoRA alpha scaling */
  alpha: number
  /** Target modules (e.g. ["q_proj", "v_proj"]) */
  targetModules: string[]
}

/** PPO-specific configuration (Section 11.3, input 5) */
export interface PPOConfig {
  /** Critic model parameter count */
  criticModelParams: number
  /** Reward model parameter count */
  rewardModelParams: number
}

/** GRPO-specific configuration (Section 11.3, input 6) */
export interface GRPOConfig {
  /** Group size G for GRPO sampling */
  groupSize: number
}

/** Full post-training configuration (Section 11.3) */
export interface PostTrainingConfig {
  /** Base model (preset or param count) — input 1 */
  architecture: ModelArchitecture
  /** MoE configuration for the base model */
  moe: MoEConfig
  /** Post-training method — input 2 */
  method: PostTrainingMethod
  /** Fine-tuning approach — input 3 */
  approach: FineTuningApproach
  /** LoRA config — input 4 */
  loraConfig: LoRAConfig
  /** Trainable parameter percentage (input 4a; 100 for full, auto for LoRA) */
  trainablePercentage: number
  /** PPO config — input 5 */
  ppoConfig: PPOConfig
  /** GRPO config — input 6 */
  grpoConfig: GRPOConfig
  /** Dataset size in examples — input 7 */
  datasetSize: number
  /** Training epochs — input 8 */
  epochs: number
  /** Sequence length — input 9 */
  sequenceLength: number
  /** Micro-batch size — input 9 */
  microBatchSize: number
  /** GPU specification — input 10 */
  gpu: GPUSpec
  /** Number of GPUs — input 10 */
  numGPUs: number
  /** Training precision */
  precision: TrainingPrecision
  /** Optimizer */
  optimizer: OptimizerType
  /** Gradient precision */
  gradientPrecision: GradientPrecision
  /** Cost per GPU-hour */
  costPerGPUHour: number
}

// ─── Optimizer Profile ───────────────────────────────────────────────────────

/** Optimizer memory profile (Section 5.1) */
export interface OptimizerProfile {
  /** Display name */
  name: string
  /** Optimizer type key */
  type: OptimizerType
  /** Φ — total bytes per parameter (with fp32 gradients) */
  phi_fp32grad: number
  /** Φ — total bytes per parameter (with bf16 gradients) */
  phi_bf16grad: number
  /** K_opt — optimizer state bytes per param (the portion sharded in ZeRO-1) */
  K_opt: number
  /** β_grad for fp32 gradients */
  beta_grad_fp32: number
  /** β_grad for bf16 gradients */
  beta_grad_bf16: number
  /** Breakdown description */
  breakdown: string
  /** Whether this optimizer supports pretraining */
  supportsPretraining: boolean
}

// ─── Output Types ────────────────────────────────────────────────────────────

/** Parameter count breakdown (Section 3) */
export interface ParameterCounts {
  /** Ψ — total parameters */
  total: number
  /** Ψ_active — active parameters per token (= total for dense) */
  active: number
  /** Per-layer breakdown */
  perLayer: {
    attention: number
    ffn: number
    norm: number
  }
  /** Non-layer parameters */
  embedding: number
  outputProjection: number
  positionalEmbedding: number
  finalNorm: number
  /** MoE-specific (null if dense) */
  moe: {
    totalExperts: number
    routerParams: number
    sharedExpertParams: number
  } | null
}

/** Compute estimate (Section 4) */
export interface ComputeEstimate {
  /** C — total training FLOPs */
  totalFLOPs: number
  /** FLOPs per token (6Ψ + 12Lds) */
  flopsPerToken: number
  /** Attention overhead: 12Lds / 6Ψ as a fraction */
  attentionOverheadFraction: number
  /** Whether simplified 6ΨD is accurate (d > s/12) */
  simplifiedAccurate: boolean
  /** MoE load balance overhead applied */
  moeLoadBalanceOverhead: number
}

/** Chinchilla scaling analysis (Section 4.3) */
export interface ChinchillaAnalysis {
  /** D / (20 × Ψ) ratio */
  chinchillaRatio: number
  /** Predicted training loss in nats */
  predictedLoss: number
  /** Compute-optimal token count for this model size */
  optimalTokens: number
  /** Compute-optimal model size for the total compute budget */
  optimalModelSize: number
  /** Which coefficient row was used */
  coefficientRow: string
  /** D_optimal from power-law fit: 8.62 × N^1.041 */
  powerLawOptimalTokens: number
}

/** Critical batch size analysis (Section 4.4) */
export interface BatchSizeAnalysis {
  /** B_crit in tokens */
  criticalBatchSize: number
  /** User's B_tok */
  actualBatchTokens: number
  /** Whether B_tok > B_crit */
  aboveCritical: boolean
  /** Compute multiplier above optimum: 1 + B_tok/B_crit */
  computeMultiplier: number
  /** Wasted-compute fraction: B_tok / (B_tok + B_crit) */
  wastedComputeFraction: number
}

/** Data repetition analysis (Section 4.5) */
export interface DataRepetitionAnalysis {
  /** Number of epochs: D / U */
  epochs: number
  /** Whether U < D (repetition is happening) */
  hasRepetition: boolean
  /** Warning severity */
  severity: "none" | "info" | "warning" | "critical"
  /** Maximum effective data: ~16× unique tokens */
  effectiveDataCeiling: number
}

/** Memory breakdown per GPU (Section 5) */
export interface MemoryBreakdown {
  /** Parameters in bytes */
  parameters: number
  /** Gradients in bytes */
  gradients: number
  /** Optimizer states in bytes */
  optimizerStates: number
  /** Activations in bytes */
  activations: number
  /** Communication buffers in bytes */
  communicationBuffers: number
  /** Framework overhead (CUDA alignment + misc) in bytes */
  frameworkOverhead: number
  /** Total memory per GPU in bytes */
  totalPerGPU: number
  /** GPU VRAM in bytes */
  gpuVRAM: number
  /** Free headroom in bytes (can be negative if oversubscribed) */
  freeHeadroom: number
  /** Whether memory fits on GPU */
  fits: boolean
  /** Usable VRAM fraction (0.9 for most, 0.8 for vanilla PyTorch) */
  usableVRAMFraction: number
}

/** Post-training memory breakdown (Section 10) */
export interface PostTrainingMemoryBreakdown extends MemoryBreakdown {
  /** Trainable model memory */
  trainableModel: number
  /** Frozen model memory (inference-only, no grads/optimizer) */
  frozenModel: number
  /** LoRA adapter memory (if applicable) */
  loraAdapter: number | null
  /** PPO-specific buffers (if applicable) */
  ppoBuffers: number | null
}

/** Parallelism recommendation (Section 9) */
export interface ParallelismRecommendation {
  /** Recommended parallelism config */
  config: ParallelismConfig
  /** Minimum GPUs needed (memory-constrained) */
  minGPUs: number
  /** Minimum GPU VRAM floor (bytes) — largest transformer block */
  minVRAMFloor: number
  /** Pipeline bubble overhead fraction */
  pipelineBubbleFraction: number
  /** Explanation of why this config was chosen */
  reasoning: string[]
  /** Warnings about the configuration */
  warnings: Warning[]
}

/** Training time estimate (Section 6) */
export interface TrainingTimeEstimate {
  /** Theoretical training time in days (no failures) */
  theoreticalDays: number
  /** Failure-adjusted training time in days (Section 6.5) */
  failureAdjustedDays: number | null
  /** Failure time multiplier */
  failureMultiplier: number | null
  /** Tokens per second throughput */
  tokensPerSecond: number
  /** Training steps */
  totalSteps: number
  /** Time per step in seconds */
  secondsPerStep: number
}

/** Cost estimate (Section 8) */
export interface CostEstimate {
  /** Compute cost: N_gpu × hours × $/hr */
  computeCost: number
  /** Checkpoint storage cost */
  storageCost: number
  /** Failure overhead cost */
  failureOverheadCost: number
  /** Total cost */
  totalCost: number
  /** Checkpoint size in bytes */
  checkpointSize: number
  /** Number of checkpoints saved */
  numCheckpoints: number
}

/** MoE sparsity metrics (Section 11.2, output 18) */
export interface MoESparsityMetrics {
  /** Ψ_active / Ψ_total */
  sparsityRatio: number
  /** Ψ_total / Ψ_active */
  efficiencyGain: number
  /** Load balance overhead applied */
  loadBalanceOverhead: number
}

/** Warning with severity (Section 14) */
export interface Warning {
  /** Severity level */
  severity: "info" | "warning" | "error"
  /** Category for grouping */
  category:
    | "memory"
    | "precision"
    | "parallelism"
    | "compute"
    | "data"
    | "hardware"
    | "cost"
    | "batch"
  /** Human-readable message */
  message: string
}

/** Global batch size (Section 11.2, output 12) */
export interface GlobalBatchSize {
  /** B_seq — batch size in sequences: b × G × N_dp */
  sequences: number
  /** B_tok — batch size in tokens: b × s × G × N_dp */
  tokens: number
}

/** Complete calculator output for pretraining (Section 11.2 outputs) */
export interface CalculatorOutput {
  /** 1. Total parameter count */
  parameterCounts: ParameterCounts
  /** 2. Total FLOPs */
  computeEstimate: ComputeEstimate
  /** 3. Chinchilla ratio and recommendation */
  chinchillaAnalysis: ChinchillaAnalysis
  /** 4. Memory breakdown per GPU */
  memoryBreakdown: MemoryBreakdown
  /** 5. Minimum GPUs needed (memory) */
  minGPUs: number
  /** 6. Minimum GPU VRAM floor */
  minVRAMFloor: number
  /** 7. Recommended parallelism */
  parallelismRecommendation: ParallelismRecommendation
  /** 8. Pipeline bubble overhead */
  pipelineBubbleFraction: number
  /** 9. Training time estimate */
  trainingTime: TrainingTimeEstimate
  /** 10. Tokens/second */
  tokensPerSecond: number
  /** 11. Cost breakdown */
  costEstimate: CostEstimate
  /** 12. Global batch size */
  globalBatchSize: GlobalBatchSize
  /** 13. Checkpoint size in bytes */
  checkpointSize: number
  /** 14. Attention overhead percentage */
  attentionOverheadPercent: number
  /** 15. Predicted training loss */
  predictedLoss: number
  /** 16. Maximum micro-batch size */
  maxMicroBatchSize: number
  /** 17. Data repetition analysis */
  dataRepetition: DataRepetitionAnalysis
  /** 18. MoE sparsity metrics (null if dense) */
  moeMetrics: MoESparsityMetrics | null
  /** 19. Batch size efficiency */
  batchSizeAnalysis: BatchSizeAnalysis
  /** All warnings */
  warnings: Warning[]
}

/** Complete calculator output for post-training (Section 11.3 outputs) */
export interface PostTrainingOutput {
  /** Memory breakdown per GPU */
  memoryBreakdown: PostTrainingMemoryBreakdown
  /** Number of GPUs needed */
  numGPUs: number
  /** Estimated training time */
  trainingTime: TrainingTimeEstimate
  /** Estimated cost */
  costEstimate: CostEstimate
  /** Warnings */
  warnings: Warning[]
}

// ─── Model Preset ────────────────────────────────────────────────────────────

/** Model preset (Section 3.3) */
export interface ModelPreset {
  /** Display name */
  name: string
  /** Architecture details */
  architecture: ModelArchitecture
  /** Approximate parameter count (for display) */
  approxParams: number
  /** Whether this is a MoE model */
  moe: MoEConfig | null
}

// ─── Quick Mode ──────────────────────────────────────────────────────────────

/** Quick mode lookup entry (Section 11.1) */
export interface QuickModeLookup {
  /** Minimum parameter count (inclusive) */
  minParams: number
  /** Maximum parameter count (exclusive, Infinity for last) */
  maxParams: number
  /** Inferred number of attention heads */
  heads: number
  /** Inferred number of layers */
  layers: number
}

// ─── Chinchilla Coefficients ─────────────────────────────────────────────────

/** Chinchilla/scaling law coefficient row (Section 4.3) */
export interface ChinchillaCoefficients {
  /** Label for this coefficient set */
  label: string
  /** Max D/N ratio this row applies to */
  maxDNRatio: number
  /** alpha exponent */
  alpha: number
  /** beta exponent */
  beta: number
  /** A coefficient */
  A: number
  /** B coefficient */
  B: number
  /** E irreducible loss */
  E: number
}

// ─── MFU Defaults ────────────────────────────────────────────────────────────

/** MFU default lookup (Section 6.3) */
export interface MFUDefault {
  /** Label for this tier */
  label: string
  /** Minimum parameter count */
  minParams: number
  /** Maximum parameter count */
  maxParams: number
  /** Minimum GPU count */
  minGPUs: number
  /** Maximum GPU count */
  maxGPUs: number
  /** Default MFU value (as fraction, e.g. 0.35) */
  mfu: number
}

// ─── Cloud Pricing ───────────────────────────────────────────────────────────

/** Cloud pricing preset (Section 8.1) */
export interface CloudPricingPreset {
  /** GPU name */
  gpu: string
  /** Low end of price range $/hr/GPU */
  priceLow: number
  /** High end of price range $/hr/GPU */
  priceHigh: number
  /** Default (midpoint) */
  priceDefault: number
}

/** Reference cloud instance (Section 8.1) */
export interface CloudInstance {
  /** Cloud provider */
  provider: string
  /** Instance type name */
  instanceType: string
  /** GPU model */
  gpu: string
  /** GPU count */
  gpuCount: number
  /** VRAM per GPU in GB */
  vramPerGPU: number
  /** On-demand price $/hr */
  pricePerHour: number
}

// ─── Calculator Tab ──────────────────────────────────────────────────────────

/** Active calculator tab */
export type CalculatorTab = "pretraining" | "post-training"
