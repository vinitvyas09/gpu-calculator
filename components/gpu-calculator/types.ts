export type CalculatorTab = "pretraining" | "post-training"

export type FFNType = "standard" | "swiglu" | "geglu" | "moe"
export type NormType = "layernorm" | "rmsnorm"
export type PositionalEmbeddingType = "learned" | "rope" | "alibi" | "none"
export type AttentionVariant = "mha" | "gqa" | "mqa" | "mla"
export type ModelInputMode = "quick" | "preset" | "detailed"
export type QuickModeFamily = "dense-gpt-style" | "modern-open-weights"
export type BaseModelInputMode = "preset" | "parameter-count"

export interface ModelArchitecture {
  d: number
  L: number
  a: number
  d_head?: number | null
  a_kv: number | null
  d_ff: number | null
  V: number
  ffnType: FFNType
  normType: NormType
  posEmbedding: PositionalEmbeddingType
  attentionVariant: AttentionVariant
  tiedEmbeddings: boolean
}

export interface MoEConfig {
  enabled: boolean
  E: number
  topk: number
  L_moe: number
  E_s: number
  loadBalanceFactor: number
  expertIntermediateSize: number | null
  denseIntermediateSize: number | null
  activeParameterCount: number | null
}

export interface QuickModeConfig {
  totalParameters: number
  family: QuickModeFamily | null
  inferredHeads: number | null
  inferredLayers: number | null
  hiddenSizeRoundingMultiple: number
}

export interface ModelSelection {
  inputMode: ModelInputMode
  presetId: string | null
  quickMode: QuickModeConfig
  architecture: ModelArchitecture
  moe: MoEConfig
}

export type GPUVendor = "nvidia" | "amd" | "apple"
export type GPUCategory =
  | "nvidia-datacenter"
  | "nvidia-consumer"
  | "amd-datacenter"
  | "apple-silicon"
export type InterconnectType = "nvlink" | "pcie" | "xgmi" | "none"
export type GPUMemoryType = "vram" | "unified"
export type GPUInputMode = "preset" | "custom"

export interface GPUSpec {
  id: string
  name: string
  vendor: GPUVendor
  category: GPUCategory
  memoryType: GPUMemoryType
  memoryGB: number
  halfPrecisionTFLOPS: number
  halfPrecisionFormat: "bf16" | "fp16"
  tf32TFLOPS: number | null
  fp32TFLOPS?: number | null
  fp8TFLOPS: number | null
  memoryBandwidthGBps: number
  nvlinkBandwidthGBps: number | null
  tdpWatts: number | null
  gpusPerNode: number
  interconnect: InterconnectType
  singleDeviceOnly: boolean
  supportsBF16: boolean
  supportsTF32: boolean
  supportsFP8: boolean
}

export type TrainingPrecision = "fp32" | "bf16" | "fp16" | "fp8"
export type GradientPrecision = "fp32" | "bf16"
export type CheckpointingMode = "none" | "selective" | "full" | "partial"
export type ParallelismMode = "auto" | "manual"
export type SequenceParallelismMode = "auto" | "enabled" | "disabled"
export type ZeROStage = 0 | 1 | 2 | 3
export type FSDPStrategy =
  | "NO_SHARD"
  | "SHARD_GRAD_OP"
  | "FULL_SHARD"
  | "HYBRID_SHARD"
  | "HYBRID_SHARD_ZERO2"
export type FrameworkType = "megatron" | "deepspeed" | "fsdp" | "hf_trainer"
export type CPUOffloadMode = "none" | "optimizer-only" | "optimizer-and-params"
export type InterNodeBandwidthPreset = "hdr-200" | "ndr-400" | "custom"
export type KVCachePrecision = "bf16" | "fp16" | "int8"
export type FP8StorageMode = "transformer-engine" | "ms-amp"
export type ZeROCommunicationBucketMode =
  | "hf-auto"
  | "deepspeed-defaults"
  | "custom"

export type OptimizerType =
  | "adamw-fp32"
  | "adamw-mixed"
  | "adamw-mixed-bf16-states"
  | "adamw-mixed-no-master"
  | "adamw-fp8"
  | "adamw-8bit"
  | "adam-mini"
  | "sgd-momentum"
  | "sgd-no-momentum"
  | "adafactor"
  | "lion"
  | "lamb"
  | "mezo"

export interface ParallelismConfig {
  N_tp: number
  N_pp: number
  N_dp: number
  N_cp: number
  N_ep: number
  zeroStage: ZeROStage
  fsdpStrategy: FSDPStrategy | null
  framework: FrameworkType
  sequenceParallelism: SequenceParallelismMode
  VP: number
}

export interface HardwareSelection {
  inputMode: GPUInputMode
  gpuId: string | null
  gpu: GPUSpec
  numGPUs: number | null
  targetTrainingDays: number | null
  interNodeBandwidthPreset: InterNodeBandwidthPreset
  interNodeBandwidthGBps: number
}

export interface ZeROCommunicationConfig {
  mode: ZeROCommunicationBucketMode
  overlapComm: boolean
  allgatherBucketSizeElements: number | null
  reduceBucketSizeElements: number | null
  prefetchBucketSizeElements: number | null
}

export interface FP8Config {
  enabled: boolean
  kernelSpeedupFactor: number
  storageMode: FP8StorageMode
}

export interface FailureModelConfig {
  failureRatePerInstancePerDay: number
  recoveryTimeHours: number
  checkpointFrequencyPerDay: number
}

export interface PricingConfig {
  costPerGPUHour: number
  checkpointRetentionCount: number
  storagePricePerGBMonth: number
  datasetStorageGB: number
  cloudPricingPresetId: string | null
  cloudInstanceId: string | null
}

export interface TrainingConfig {
  model: ModelSelection
  totalTokens: number
  uniqueTokens: number
  precision: TrainingPrecision
  optimizer: OptimizerType
  gradientPrecision: GradientPrecision
  microBatchSize: number
  sequenceLength: number
  gradientAccumulationSteps: number
  activationCheckpointing: CheckpointingMode
  partialCheckpointDepth: number | null
  flashAttention: boolean
  hardware: HardwareSelection
  mfuOverride: number | null
  parallelismMode: ParallelismMode
  parallelism: ParallelismConfig
  pricing: PricingConfig
  ampAutocast: boolean
  cpuOffload: CPUOffloadMode
  zeroCommunication: ZeROCommunicationConfig
  torchCompile: boolean
  chunkedCrossEntropy: boolean
  fp8: FP8Config
  failureModel: FailureModelConfig
}

export type PostTrainingMethod = "sft" | "dpo" | "ppo" | "grpo"
export type FineTuningApproach = "full" | "lora" | "qlora" | "mezo"
export type LoRATargetModule =
  | "q_proj"
  | "k_proj"
  | "v_proj"
  | "o_proj"
  | "gate_proj"
  | "up_proj"
  | "down_proj"

export interface BaseModelSelection {
  inputMode: BaseModelInputMode
  presetId: string | null
  parameterCount: number
  architecture: ModelArchitecture
  moe: MoEConfig
}

export interface LoRAConfig {
  rank: number
  alpha: number
  targetModules: LoRATargetModule[]
  quantizationBits: 4 | 8 | null
}

export interface PPOConfig {
  criticModelParameterCount: number
  rewardModelParameterCount: number
  updateEpochs: number
}

export interface GRPOConfig {
  groupSize: number
}

export interface PostTrainingHardwareSelection {
  inputMode: GPUInputMode
  gpuId: string | null
  gpu: GPUSpec
  numGPUs: number
}

export interface PostTrainingConfig {
  baseModel: BaseModelSelection
  method: PostTrainingMethod
  approach: FineTuningApproach
  lora: LoRAConfig
  trainableParameterPercentage: number | null
  ppo: PPOConfig
  grpo: GRPOConfig
  datasetSizeExamples: number
  epochs: number
  sequenceLength: number
  batchSize: number
  hardware: PostTrainingHardwareSelection
  precision: TrainingPrecision
  optimizer: OptimizerType
  gradientPrecision: GradientPrecision
  chunkedCrossEntropy: boolean
  fp8: FP8Config
  costPerGPUHour: number
  kvCachePrecision: KVCachePrecision
}

export interface OptimizerMemoryVariant {
  parameterBytes: number
  betaGrad: number
  masterWeightBytes: number
  optimizerStateBytes: number
  kOpt: number
  phi: number
  breakdown: string
}

export interface OptimizerProfile {
  id: OptimizerType
  name: string
  description: string
  fp32Grad: OptimizerMemoryVariant
  bf16Grad: OptimizerMemoryVariant
  supportsPretraining: boolean
  supportsPostTraining: boolean
  fixedGradientStorage: boolean
}

export interface ParameterCounts {
  total: number
  active: number
  embedding: number
  outputProjection: number
  positionalEmbedding: number
  finalNorm: number
  perLayer: {
    attention: number
    ffn: number
    norm: number
  }
  moe: {
    expertParameters: number
    routerParameters: number
    sharedExpertParameters: number
    activeRoutedExpertParameters: number
  } | null
}

export interface ComputeEstimate {
  totalFLOPs: number
  flopsPerToken: number
  attentionOverheadFraction: number
  simplifiedFormulaAccurate: boolean
  moeLoadBalanceFactor: number
}

export interface ChinchillaAnalysis {
  parameterCount: number
  ratio: number
  recommendedTokenCount: number
  powerLawOptimalTokens: number
  optimalModelSize: number
  predictedLossNats: number
  effectiveLossTokens: number
  coefficientRowId: string
  coefficientRowLabel: string
  recommendation: string
}

export interface BatchSizeAnalysis {
  criticalBatchTokens: number
  actualBatchTokens: number
  relation: "below" | "near" | "above" | "unknown"
  computeMultiplier: number
  wastedComputeFraction: number
}

export interface DataRepetitionAnalysis {
  epochs: number
  hasRepetition: boolean
  severity: "none" | "info" | "warning" | "critical"
  effectiveDataCeiling: number
  recommendation: string
}

export interface MemoryBreakdown {
  parameters: number
  gradients: number
  optimizerStates: number
  activations: number
  communicationBuffers: number
  frameworkOverhead: number
  freeHeadroom: number
  total: number
  gpuCapacity: number
  usableCapacity: number
  fits: boolean
}

export interface PostTrainingModelMemoryLineItem {
  label: string
  category: "trainable" | "frozen" | "adapter" | "buffer"
  bytes: number
}

export interface PostTrainingMemoryBreakdown extends MemoryBreakdown {
  trainableModels: number
  frozenModels: number
  loraAdapter: number
  ppoBuffers: number
  items: PostTrainingModelMemoryLineItem[]
}

export interface Warning {
  severity: "info" | "warning" | "critical"
  category:
    | "memory"
    | "precision"
    | "parallelism"
    | "compute"
    | "data"
    | "hardware"
    | "cost"
    | "generation"
  message: string
}

export interface ParallelismRecommendation {
  config: ParallelismConfig
  minGPUs: number
  minVRAMFloor: number
  pipelineBubbleFraction: number
  strategyLabel: string
  reasoning: string[]
  warnings: Warning[]
}

export interface TrainingTimeEstimate {
  theoreticalDays: number
  theoreticalHours: number
  failureAdjustedDays: number | null
  failureAdjustedHours: number | null
  failureMultiplier: number | null
  tokensPerSecond: number
  totalSteps: number
  secondsPerStep: number
}

export interface CostEstimate {
  computeCost: number
  actualComputeCost: number | null
  storageCost: number
  failureOverheadCost: number
  totalCost: number
  checkpointSize: number
  numCheckpoints: number
  peakCheckpointStorage: number
  averageCheckpointStorage: number
  datasetStorageBytes: number
}

export interface MoESparsityMetrics {
  sparsityRatio: number
  efficiencyGain: number
  loadBalanceFactor: number
}

export interface GlobalBatchSize {
  sequences: number
  tokens: number
}

export interface PretrainingOutput {
  /** Raw model parameter counts before implementation padding. */
  parameterCounts: ParameterCounts
  /** Counts after implementation padding, such as TP vocabulary padding. */
  implementationParameterCounts: ParameterCounts
  computeEstimate: ComputeEstimate
  chinchilla: ChinchillaAnalysis
  memory: MemoryBreakdown
  minGPUsNeeded: number
  minVRAMFloor: number
  parallelismRecommendation: ParallelismRecommendation
  pipelineBubbleFraction: number
  trainingTime: TrainingTimeEstimate
  tokensPerSecond: number
  cost: CostEstimate
  globalBatchSize: GlobalBatchSize
  checkpointSize: number
  attentionOverheadFraction: number
  predictedLossNats: number
  maxMicroBatchSize: number
  dataRepetition: DataRepetitionAnalysis
  moeSparsity: MoESparsityMetrics | null
  batchEfficiency: BatchSizeAnalysis
  warnings: Warning[]
}

export type CalculatorOutput = PretrainingOutput | PostTrainingOutput
export type PostTrainingGPURequirementMode =
  | "data-parallel"
  | "state-sharded-lower-bound"

export interface PostTrainingOutput {
  memory: PostTrainingMemoryBreakdown
  numGPUsNeeded: number | null
  numGPUsNeededMode: PostTrainingGPURequirementMode | null
  trainingTime: TrainingTimeEstimate
  stepCountLabel: string
  stepTimeLabel: string
  stepMarkdownLabel: string
  cost: CostEstimate
  warnings: Warning[]
}

export interface ModelPreset {
  id: string
  name: string
  parameterCount: number
  activeParameterCount: number | null
  defaultSequenceLength: number
  architecture: ModelArchitecture
  moe: MoEConfig | null
  notes: string | null
}

export interface QuickModeLookup {
  id: string
  minParams: number
  maxParams: number
  heads: number
  layers: number
  family: QuickModeFamily
}

export interface ChinchillaCoefficients {
  id: string
  label: string
  alpha: number
  beta: number
  A: number
  B: number
  E: number
  autoSelectMinDNRatio: number | null
  autoSelectMaxDNRatio: number | null
  autoSelectable: boolean
}

export interface MFUDefault {
  id: string
  label: string
  minParams: number | null
  maxParams: number | null
  minGPUs: number | null
  maxGPUs: number | null
  mfuLow: number
  mfuHigh: number
  defaultMFU: number
  advisoryOnly: boolean
}

export interface CloudPricingPreset {
  id: string
  gpuId: string
  label: string
  priceLow: number
  priceHigh: number
  priceDefault: number
}

export interface CloudInstance {
  id: string
  provider: string
  instanceType: string
  gpuId: string
  gpuCount: number
  vramPerGPU: number
  pricePerHour: number
}
