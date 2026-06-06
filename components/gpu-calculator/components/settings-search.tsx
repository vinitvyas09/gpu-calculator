"use client"

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react"
import { motion, AnimatePresence, useReducedMotion } from "framer-motion"
import { Search, CornerDownLeft } from "lucide-react"
import type { CalculatorTab } from "../types"
import type { CalculatorColors } from "./input-controls"

// ═══════════════════════════════════════════════════════════════════════════
// settings-search.tsx — the ⌘K control palette (Phase 6, plan §5).
//
// A keyboard-first "jump to any setting" surface built on the EXISTING
// primitives (no cmdk/radix — plan §0.6). It searches a static control
// registry (CONTROL_REGISTRY below) and, on select, takes the user to the
// matching control: switch tab → open its owning Layer → focus the field.
//
// The host (GpuCalculator) owns the ⌘K keybinding + the visibility atom and
// supplies a single atomic navigation callback (onNavigate), implemented as
// navigateToControl in the shell. Routing is one operation on purpose: a
// cross-tab jump must switch the tab, open the owning Layer KEYED TO THE
// DESTINATION TAB, and focus the control only after the switched-in panel has
// mounted — three separate calls would read the host's stale (pre-switch)
// activeTab and write the wrong tab's layer key / focus an unmounted field. The
// registry maps every field's data-field-id to its owning layer + tab; the
// dev-only `assertRegistryFresh` tripwire guarantees a moved/renamed control can
// never silently fall out of search.
// ═══════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// Control registry
// ---------------------------------------------------------------------------

export interface ControlRegistryEntry {
  /** Control id — matches the data-field-id rendered on the control's wrapper. */
  id: string
  /** Plain-word-first searchable label shown in the results list. */
  label: string
  /** Owning Layer.id → opened + scrolled before focusing. Omitted ⇒ the control
   *  lives outside the LayerStack (e.g. an Essentials field), so only the
   *  tab-switch + focus steps run. */
  layerId?: string
  /** Which phase tab the control belongs to. "both" ⇒ no tab switch needed
   *  (the control renders on either tab — e.g. an Essentials primitive). */
  tab: CalculatorTab | "both"
  /** Extra search terms (acronyms, symbols) folded into the match. */
  keywords?: string[]
}

/**
 * The static control registry — one entry per user-facing control (Stage B).
 *
 * `id` matches the `data-field-id` the primitive renders (the ⌘K focus target);
 * `label` is the Appendix-A plain-word-first label; `layerId` is the owning
 * Layer (undefined for Essentials / custom-GPU-disclosure controls that live
 * outside the LayerStack); `tab` drives the tab-switch on select ("both" for a
 * field that renders on either tab — e.g. #GPUs, precision — so no switch is
 * needed). `keywords` fold in old-label fragments, acronyms, and the config
 * field name so a search like "tokens-per-param" or "n_tp" still finds it.
 *
 * Conditional controls (only visible under some config — manual parallelism,
 * fp8, MoE, partial checkpointing, custom-GPU mode, LoRA/PPO/GRPO) are listed
 * unconditionally and flagged via keywords; the palette does not model the
 * condition, so focusing a hidden control may only open its owning layer (the
 * gating control is then in view), which is acceptable (plan §5).
 *
 * Disclosure-parent rows ("customizeAdapter", "customGpuSpecs") target a
 * data-field-id placed on the disclosure wrapper: there is no inner
 * data-field-input, so the palette scroll-focuses the open disclosure.
 *
 * Kept FRESH against the DOM by `assertRegistryFresh` (every rendered
 * data-field-id must appear here, or dev console.errors).
 */
export const CONTROL_REGISTRY: ControlRegistryEntry[] = [
  // ── Essentials — shared across both tabs (no tab switch on select) ──
  {
    id: "gpuId",
    label: "GPU",
    tab: "both",
    keywords: ["gpu", "accelerator", "device", "hardware", "preset", "gpuId"],
  },
  {
    id: "numGPUs",
    label: "Number of GPUs",
    tab: "both",
    keywords: ["gpu count", "how many gpus", "numGPUs", "devices", "scale"],
  },
  {
    id: "costPerGPUHour",
    label: "Cost per GPU-hour ($/hr)",
    tab: "both",
    keywords: ["price", "$/hr", "hourly", "rate", "cost", "costPerGPUHour"],
  },

  // ── Essentials — pretraining only ──
  {
    id: "presetId",
    label: "Model",
    tab: "pretraining",
    keywords: ["model preset", "preset", "architecture", "presetId"],
  },
  {
    id: "quickModeTotalParameters",
    label: "Parameters (quick)",
    tab: "pretraining",
    keywords: ["total parameters", "param count", "size", "quick", "totalParameters"],
  },
  {
    id: "totalTokens",
    label: "Total training tokens (D)",
    tab: "pretraining",
    keywords: ["training tokens", "dataset tokens", "D", "totalTokens"],
  },
  {
    id: "targetTrainingDays",
    label: "Target training days (optional)",
    tab: "pretraining",
    keywords: ["target days", "deadline", "wall clock", "targetTrainingDays"],
  },

  // ── Essentials — post-training only ──
  {
    id: "baseModel-presetId",
    label: "Base model",
    tab: "post-training",
    keywords: ["base model", "preset", "baseModel", "presetId"],
  },
  {
    id: "baseModel-parameterCount",
    label: "Parameters (base model)",
    tab: "post-training",
    keywords: ["parameter count", "by size", "param", "parameterCount"],
  },
  {
    id: "method",
    label: "Method",
    tab: "post-training",
    keywords: ["sft", "dpo", "ppo", "grpo", "rlhf", "fine-tuning method", "method"],
  },
  {
    id: "approach",
    label: "Approach",
    tab: "post-training",
    keywords: ["full", "lora", "qlora", "mezo", "adapter", "approach"],
  },
  {
    id: "trainableParameterPercentage",
    label: "Trainable parameters (%)",
    tab: "post-training",
    keywords: ["trainable %", "frozen layers", "percent", "trainableParameterPercentage"],
  },
  {
    id: "customizeAdapter",
    label: "Customize adapter (LoRA)",
    tab: "post-training",
    keywords: ["lora", "qlora", "adapter", "rank", "alpha", "target modules"],
  },
  {
    id: "lora-rank",
    label: "LoRA rank (r)",
    tab: "post-training",
    keywords: ["lora", "rank", "r", "adapter capacity", "lora.rank"],
  },
  {
    id: "lora-alpha",
    label: "LoRA alpha",
    tab: "post-training",
    keywords: ["lora", "alpha", "scaling", "lora.alpha"],
  },
  {
    id: "lora-quantizationBits",
    label: "Quantization bits",
    tab: "post-training",
    keywords: ["qlora", "quantization", "4-bit", "8-bit", "nf4", "quantizationBits"],
  },
  {
    id: "ppo-criticModelParameterCount",
    label: "Critic model parameters",
    tab: "post-training",
    keywords: ["ppo", "critic", "value model", "criticModelParameterCount"],
  },
  {
    id: "ppo-rewardModelParameterCount",
    label: "Reward model parameters",
    tab: "post-training",
    keywords: ["ppo", "reward model", "rewardModelParameterCount"],
  },
  {
    id: "ppo-updateEpochs",
    label: "PPO update epochs",
    tab: "post-training",
    keywords: ["ppo", "update epochs", "optimization epochs", "updateEpochs"],
  },
  {
    id: "grpo-groupSize",
    label: "GRPO group size (G)",
    tab: "post-training",
    keywords: ["grpo", "group size", "G", "groupSize"],
  },
  {
    id: "grpo-rewardModelParameterCount",
    label: "Reward model parameters (GRPO)",
    tab: "post-training",
    keywords: ["grpo", "reward model", "rule-based", "rewardModelParameterCount"],
  },
  {
    id: "datasetSizeExamples",
    label: "Dataset size",
    tab: "post-training",
    keywords: ["examples", "pairs", "prompts", "dataset", "datasetSizeExamples"],
  },
  {
    id: "epochs",
    label: "Epochs",
    tab: "post-training",
    keywords: ["epochs", "passes", "epochs"],
  },

  // ── Layer 3 · Parallelism — pretraining only ──
  {
    id: "parallelismMode",
    label: "How to choose the GPU layout — Auto · Manual",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["mode", "auto", "manual", "layout", "parallelismMode"],
  },
  {
    id: "parallelism-framework",
    label: "Framework",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["deepspeed", "megatron", "fsdp", "hf trainer", "framework"],
  },
  {
    id: "parallelism-N_tp",
    label: "Tensor parallel (TP)",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["tensor parallel", "TP", "N_tp", "megatron"],
  },
  {
    id: "parallelism-N_pp",
    label: "Pipeline parallel (PP)",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["pipeline parallel", "PP", "N_pp", "stages"],
  },
  {
    id: "parallelism-N_dp",
    label: "Data parallel (DP)",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["data parallel", "DP", "N_dp"],
  },
  {
    id: "parallelism-zeroStage",
    label: "ZeRO stage",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["zero", "stage", "deepspeed", "sharding", "zeroStage"],
  },
  {
    id: "parallelism-fsdpStrategy",
    label: "FSDP strategy",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["fsdp", "full shard", "shard grad", "fsdpStrategy"],
  },
  {
    id: "parallelism-N_cp",
    label: "Context parallel (CP)",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["context parallel", "CP", "N_cp", "long sequence"],
  },
  {
    id: "parallelism-VP",
    label: "Virtual pipeline chunks (VP)",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["virtual pipeline", "interleave", "VP", "chunks"],
  },
  {
    id: "parallelism-sequenceParallelism",
    label: "Sequence parallelism",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["sequence parallel", "sequenceParallelism"],
  },
  {
    id: "zeroCommunication-overlapComm",
    label: "Overlap communication",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["overlap", "communication", "comm", "overlapComm"],
  },
  {
    id: "interNodeBandwidth-mode",
    label: "Inter-node bandwidth",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["inter-node", "bandwidth", "network", "interNodeBandwidth"],
  },
  {
    id: "interNodeBandwidth-customGBps",
    label: "Custom bandwidth (GB/s)",
    layerId: "parallelism",
    tab: "pretraining",
    keywords: ["custom bandwidth", "GB/s", "customGBps"],
  },

  // ── Layer 4 · Model architecture ──
  {
    id: "sequenceLength",
    label: "Sequence length (s)",
    layerId: "architecture",
    tab: "both",
    keywords: ["sequence length", "context", "s", "tokens", "sequenceLength"],
  },
  {
    id: "flashAttention",
    label: "Flash Attention",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["flash attention", "fused attention", "flashAttention"],
  },
  // Layer 4 · detailed-mode architecture grid (pretraining, detailed mode only)
  {
    id: "architecture-d",
    label: "Hidden size (d)",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["hidden dim", "d_model", "d", "width", "architecture.d"],
  },
  {
    id: "architecture-L",
    label: "Layers (L)",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["layers", "depth", "L", "architecture.L"],
  },
  {
    id: "architecture-a",
    label: "Attention heads (a)",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["attention heads", "heads", "a", "architecture.a"],
  },
  {
    id: "architecture-d_head",
    label: "Head size (d_head)",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["head dim", "d_head", "per-head", "architecture.d_head"],
  },
  {
    id: "architecture-a_kv",
    label: "Key/value heads (a_kv)",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["kv heads", "gqa", "a_kv", "key value", "architecture.a_kv"],
  },
  {
    id: "architecture-d_ff",
    label: "Feed-forward size (d_ff)",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["ffn dim", "d_ff", "feed forward", "intermediate", "architecture.d_ff"],
  },
  {
    id: "architecture-V",
    label: "Vocabulary size (V)",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["vocab", "vocabulary", "V", "architecture.V"],
  },
  {
    id: "architecture-ffnType",
    label: "Feed-forward type",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["ffn type", "swiglu", "geglu", "moe", "ffnType"],
  },
  {
    id: "architecture-normType",
    label: "Normalization",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["norm type", "layernorm", "rmsnorm", "normType"],
  },
  {
    id: "architecture-posEmbedding",
    label: "Positional encoding",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["positional", "rope", "alibi", "learned", "posEmbedding"],
  },
  {
    id: "architecture-attentionVariant",
    label: "Attention variant",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["attention variant", "mha", "gqa", "mqa", "mla", "attentionVariant"],
  },
  {
    id: "moe-enabled",
    label: "Mixture of Experts (MoE)",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["mixture of experts", "moe", "sparse", "enable", "moe.enabled"],
  },
  {
    id: "architecture-tiedEmbeddings",
    label: "Tied embeddings",
    layerId: "architecture",
    tab: "pretraining",
    keywords: ["tied embeddings", "weight tying", "tiedEmbeddings"],
  },

  // ── Layer 5 · Precision & optimizer ──
  {
    id: "precision",
    label: "Precision",
    layerId: "precision",
    tab: "both",
    keywords: ["precision", "bf16", "fp16", "fp32", "fp8", "number format"],
  },
  {
    id: "optimizer",
    label: "Optimizer",
    layerId: "precision",
    tab: "both",
    keywords: ["optimizer", "adamw", "adam", "lion", "optimizer states"],
  },
  {
    id: "gradientPrecision",
    label: "Gradient precision",
    layerId: "precision",
    tab: "both",
    keywords: ["gradient precision", "grad precision", "gradientPrecision"],
  },
  {
    id: "chunkedCrossEntropy",
    label: "Chunked cross-entropy",
    layerId: "precision",
    tab: "both",
    keywords: ["chunked cross entropy", "logits", "chunkedCrossEntropy"],
  },
  {
    id: "fp8-kernelSpeedupFactor",
    label: "FP8 kernel speedup",
    layerId: "precision",
    tab: "both",
    keywords: ["fp8", "kernel speedup", "kernelSpeedupFactor"],
  },
  {
    id: "fp8-storageMode",
    label: "FP8 storage mode",
    layerId: "precision",
    tab: "both",
    keywords: ["fp8", "storage mode", "transformer engine", "ms-amp", "storageMode"],
  },
  {
    id: "microBatchSize",
    label: "Micro-batch size (b)",
    layerId: "precision",
    tab: "pretraining",
    keywords: ["micro batch", "b", "per-gpu batch", "microBatchSize"],
  },
  {
    id: "gradientAccumulationSteps",
    label: "Gradient accumulation steps (G)",
    layerId: "precision",
    tab: "pretraining",
    keywords: ["gradient accumulation", "grad accum", "G", "gradientAccumulationSteps"],
  },
  {
    id: "activationCheckpointing",
    label: "Activation checkpointing",
    layerId: "precision",
    tab: "pretraining",
    keywords: ["activation checkpointing", "recompute", "gradient checkpointing", "activationCheckpointing"],
  },
  {
    id: "partialCheckpointDepth",
    label: "Checkpointed layers per stage",
    layerId: "precision",
    tab: "pretraining",
    keywords: ["partial checkpoint", "recompute depth", "N_recomp", "partialCheckpointDepth"],
  },
  {
    id: "cpuOffload",
    label: "CPU offloading",
    layerId: "precision",
    tab: "pretraining",
    keywords: ["cpu offload", "offloading", "ram", "cpuOffload"],
  },
  {
    id: "ampAutocast",
    label: "AMP autocast",
    layerId: "precision",
    tab: "pretraining",
    keywords: ["amp", "autocast", "mixed precision", "ampAutocast"],
  },
  {
    id: "torchCompile",
    label: "torch.compile",
    layerId: "precision",
    tab: "pretraining",
    keywords: ["torch compile", "compile", "torchCompile"],
  },
  {
    id: "mfuOverride-toggle",
    label: "Override MFU estimate",
    layerId: "precision",
    tab: "pretraining",
    keywords: ["mfu", "override", "utilization", "mfuOverride"],
  },
  {
    id: "mfuOverride",
    label: "MFU override",
    layerId: "precision",
    tab: "pretraining",
    keywords: ["mfu", "override slider", "utilization", "mfuOverride"],
  },
  {
    id: "kvCachePrecision",
    label: "KV-cache precision",
    layerId: "precision",
    tab: "post-training",
    keywords: ["kv cache", "kv-cache precision", "int8", "kvCachePrecision"],
  },

  // ── Layer 6 · Data & scaling ──
  {
    id: "uniqueTokens",
    label: "Unique tokens (U)",
    layerId: "data",
    tab: "pretraining",
    keywords: ["unique tokens", "U", "corpus", "epochs", "uniqueTokens"],
  },
  {
    id: "batchSize",
    label: "Batch size",
    layerId: "data",
    tab: "post-training",
    keywords: ["batch size", "batch", "batchSize"],
  },

  // ── Layer 7 · Cost detail & failures — pretraining only ──
  {
    id: "pricing-cloudInstanceId",
    label: "Cloud instance (optional)",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["cloud instance", "instance", "aws", "gcp", "cloudInstanceId"],
  },
  {
    id: "pricing-cloudPricingPresetId",
    label: "Pricing preset",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["pricing preset", "cloud pricing", "cloudPricingPresetId"],
  },
  {
    id: "zeroCommunication-mode",
    label: "ZeRO communication buckets",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["zero", "communication buckets", "bucket sizing", "zeroCommunication"],
  },
  {
    id: "zeroCommunication-allgatherBucketSizeElements",
    label: "All-gather bucket (elements)",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["allgather", "all-gather", "bucket", "allgatherBucketSizeElements"],
  },
  {
    id: "zeroCommunication-reduceBucketSizeElements",
    label: "Reduce bucket (elements)",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["reduce", "bucket", "reduceBucketSizeElements"],
  },
  {
    id: "zeroCommunication-prefetchBucketSizeElements",
    label: "Prefetch bucket (elements)",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["prefetch", "bucket", "prefetchBucketSizeElements"],
  },
  {
    id: "pricing-checkpointRetentionCount",
    label: "Checkpoints to keep",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["checkpoint retention", "keep checkpoints", "checkpointRetentionCount"],
  },
  {
    id: "failureModel-checkpointFrequencyPerDay",
    label: "Checkpoint frequency (/day)",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["checkpoint frequency", "per day", "checkpointFrequencyPerDay"],
  },
  {
    id: "pricing-storagePricePerGBMonth",
    label: "Storage price ($/GB/mo)",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["storage price", "$/GB/mo", "storagePricePerGBMonth"],
  },
  {
    id: "pricing-datasetStorageGB",
    label: "Dataset storage (GB)",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["dataset storage", "GB", "datasetStorageGB"],
  },
  {
    id: "failureModel-failureRatePerInstancePerDay",
    label: "Failure rate (/instance/day)",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["failure rate", "per instance", "failureRatePerInstancePerDay"],
  },
  {
    id: "failureModel-recoveryTimeHours",
    label: "Recovery time (hours)",
    layerId: "cost",
    tab: "pretraining",
    keywords: ["recovery time", "restart", "recoveryTimeHours"],
  },

  // ── Layer 8 · MoE — pretraining only ──
  {
    id: "moe-E",
    label: "Total experts (E)",
    layerId: "moe",
    tab: "pretraining",
    keywords: ["total experts", "E", "experts", "moe.E"],
  },
  {
    id: "moe-topk",
    label: "Active experts per token (top-k)",
    layerId: "moe",
    tab: "pretraining",
    keywords: ["active experts", "top-k", "topk", "routing", "moe.topk"],
  },
  {
    id: "moe-L_moe",
    label: "MoE layers (L_moe)",
    layerId: "moe",
    tab: "pretraining",
    keywords: ["moe layers", "L_moe", "moe.L_moe"],
  },
  {
    id: "moe-E_s",
    label: "Shared experts (E_s)",
    layerId: "moe",
    tab: "pretraining",
    keywords: ["shared experts", "E_s", "moe.E_s"],
  },
  {
    id: "moe-loadBalanceFactor",
    label: "Load-balance factor",
    layerId: "moe",
    tab: "pretraining",
    keywords: ["load balance", "routing imbalance", "loadBalanceFactor"],
  },
  {
    id: "moe-denseIntermediateSize",
    label: "Dense FFN size",
    layerId: "moe",
    tab: "pretraining",
    keywords: ["dense ffn", "intermediate", "denseIntermediateSize"],
  },
  {
    id: "moe-expertIntermediateSize",
    label: "Expert FFN size",
    layerId: "moe",
    tab: "pretraining",
    keywords: ["expert ffn", "intermediate", "expertIntermediateSize"],
  },
  {
    id: "parallelism-N_ep",
    label: "Expert parallel (EP)",
    layerId: "moe",
    tab: "pretraining",
    keywords: ["expert parallel", "EP", "N_ep", "moe parallelism"],
  },

  // ── Custom-GPU disclosure + spec fields — both tabs, custom GPU mode only ──
  {
    id: "customGpuSpecs",
    label: "Custom GPU specs",
    tab: "both",
    keywords: ["custom gpu", "custom hardware", "specs", "vram", "tflops"],
  },
  {
    id: "gpu-vendor",
    label: "Vendor",
    tab: "both",
    keywords: ["vendor", "nvidia", "amd", "apple", "gpu.vendor"],
  },
  {
    id: "gpu-category",
    label: "Category",
    tab: "both",
    keywords: ["category", "datacenter", "consumer", "gpu.category"],
  },
  {
    id: "gpu-memoryType",
    label: "Memory type",
    tab: "both",
    keywords: ["memory type", "unified", "vram", "memoryType"],
  },
  {
    id: "gpu-halfPrecisionFormat",
    label: "Half-precision format",
    tab: "both",
    keywords: ["half precision", "bf16", "fp16", "halfPrecisionFormat"],
  },
  {
    id: "gpu-memoryGB",
    label: "VRAM (GB)",
    tab: "both",
    keywords: ["vram", "memory", "GB", "memoryGB"],
  },
  {
    id: "gpu-halfPrecisionTFLOPS",
    label: "Dense BF16/FP16 (TFLOPS)",
    tab: "both",
    keywords: ["tflops", "bf16", "fp16", "throughput", "halfPrecisionTFLOPS"],
  },
  {
    id: "gpu-memoryBandwidthGBps",
    label: "Memory bandwidth (GB/s)",
    tab: "both",
    keywords: ["memory bandwidth", "GB/s", "memoryBandwidthGBps"],
  },
  {
    id: "gpu-tf32TFLOPS",
    label: "Dense TF32 (TFLOPS)",
    tab: "both",
    keywords: ["tf32", "tflops", "tf32TFLOPS"],
  },
  {
    id: "gpu-fp32TFLOPS",
    label: "FP32 (TFLOPS)",
    tab: "both",
    keywords: ["fp32", "tflops", "fp32TFLOPS"],
  },
  {
    id: "gpu-fp8TFLOPS",
    label: "Dense FP8 (TFLOPS)",
    tab: "both",
    keywords: ["fp8", "tflops", "fp8TFLOPS"],
  },
  {
    id: "gpu-gpusPerNode",
    label: "GPUs per node",
    tab: "both",
    keywords: ["gpus per node", "node", "gpusPerNode"],
  },
  {
    id: "gpu-interconnect",
    label: "Interconnect",
    tab: "both",
    keywords: ["interconnect", "nvlink", "pcie", "xgmi", "gpu.interconnect"],
  },
  {
    id: "gpu-supportsBF16",
    label: "Supports BF16",
    tab: "both",
    keywords: ["supports bf16", "bf16", "supportsBF16"],
  },
  {
    id: "gpu-supportsTF32",
    label: "Supports TF32",
    tab: "both",
    keywords: ["supports tf32", "tf32", "supportsTF32"],
  },
  {
    id: "gpu-supportsFP8",
    label: "Supports FP8",
    tab: "both",
    keywords: ["supports fp8", "fp8", "supportsFP8"],
  },
  {
    id: "gpu-singleDeviceOnly",
    label: "Single-device only",
    tab: "both",
    keywords: ["single device", "apple silicon", "singleDeviceOnly"],
  },
]

// ---------------------------------------------------------------------------
// Dev-only registry staleness tripwire
// ---------------------------------------------------------------------------

/**
 * Asserts the registry is FRESH against what's actually on screen: scans the
 * DOM for rendered `[data-field-id]` controls and `console.error`s any that are
 * missing from `registry`. One-way by design — every rendered control MUST be
 * findable via ⌘K (rendered ⊆ registry); the reverse is allowed (a registry
 * entry for a control hidden on the current tab is expected and fine).
 *
 * No-ops outside development and on the server. The host calls this in dev
 * after mount and after each tab switch (when the rendered field set changes).
 */
export function assertRegistryFresh(registry: ControlRegistryEntry[]): void {
  if (process.env.NODE_ENV === "production") return
  if (typeof document === "undefined") return

  const known = new Set(registry.map((entry) => entry.id))
  const missing = new Set<string>()
  document.querySelectorAll<HTMLElement>("[data-field-id]").forEach((node) => {
    const id = node.getAttribute("data-field-id")
    if (id && !known.has(id)) missing.add(id)
  })

  if (missing.size > 0) {
    console.error(
      `[settings-search] CONTROL_REGISTRY is stale — ${missing.size} rendered ` +
        `control(s) are not in the ⌘K registry and cannot be found by search: ` +
        `${[...missing].join(", ")}. Add an entry to CONTROL_REGISTRY for each.`,
    )
  }
}

// ---------------------------------------------------------------------------
// Filtering — case-insensitive substring across label + keyword tokens
// ---------------------------------------------------------------------------

function matches(entry: ControlRegistryEntry, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (!q) return true
  const haystack = [entry.label, ...(entry.keywords ?? [])]
    .join(" ")
    .toLowerCase()
  // Every whitespace-separated token in the query must appear somewhere — a
  // simple "fuzzy-ish" AND over substrings (so "gpu count" matches "Number of
  // GPUs" via keywords without needing exact ordering).
  return q.split(/\s+/).every((token) => haystack.includes(token))
}

// ---------------------------------------------------------------------------
// SettingsSearch — the ⌘K palette
// ---------------------------------------------------------------------------

export interface SettingsSearchProps {
  colors: CalculatorColors
  registry: ControlRegistryEntry[]
  /**
   * Navigate to a selected control in ONE atomic host operation: switch the
   * tab (unless `entry.tab === "both"`), open the owning Layer keyed to the
   * destination tab (when `entry.layerId` is set), and focus the control once
   * the switched-in panel has mounted. Kept as a single callback so the host
   * never reads its stale pre-switch `activeTab` mid-navigation.
   */
  onNavigate: (entry: ControlRegistryEntry) => void
  /** Controlled visibility — the host owns the ⌘K keybinding. */
  open: boolean
  onOpenChange: (next: boolean) => void
}

export function SettingsSearch({
  colors,
  registry,
  onNavigate,
  open,
  onOpenChange,
}: SettingsSearchProps) {
  const reduce = useReducedMotion()
  const titleId = useId()
  const listboxId = useId()

  const [query, setQuery] = useState("")
  const [activeIndex, setActiveIndex] = useState(0)

  const dialogRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLUListElement>(null)
  // The element focused immediately before the palette opened — Esc / select
  // restore focus here so keyboard users return to where they were.
  const restoreFocusRef = useRef<HTMLElement | null>(null)

  const results = useMemo(
    () => registry.filter((entry) => matches(entry, query)),
    [registry, query],
  )

  // Read-time clamp so a narrowing query can never leave activeIndex past the
  // end (mirrors SearchableSelect's safeActiveIndex pattern).
  const safeActiveIndex = Math.min(activeIndex, Math.max(0, results.length - 1))

  // ── Reset query + capture the opener each time the palette opens ──
  // The state resets + input focus are deferred to the next frame (after the
  // entrance mounts), which also keeps setState out of the synchronous effect
  // body (react-hooks/set-state-in-effect).
  useEffect(() => {
    if (!open) return
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null
    const id = requestAnimationFrame(() => {
      setQuery("")
      setActiveIndex(0)
      inputRef.current?.focus()
    })
    return () => cancelAnimationFrame(id)
  }, [open])

  // ── Scroll the active option into view as it moves ──
  useEffect(() => {
    if (!open) return
    const active = results[safeActiveIndex]
    if (!active) return
    listRef.current
      ?.querySelector<HTMLElement>(`[data-opt-id="${CSS.escape(active.id)}"]`)
      ?.scrollIntoView({ block: "nearest" })
  }, [open, safeActiveIndex, results])

  const close = useCallback(
    (restore = true) => {
      onOpenChange(false)
      if (restore) restoreFocusRef.current?.focus()
    },
    [onOpenChange],
  )

  const select = useCallback(
    (entry: ControlRegistryEntry | undefined) => {
      if (!entry) return
      // Close WITHOUT restoring focus — navigation moves focus to the control.
      onOpenChange(false)
      // Hand the whole entry to the host's single atomic navigator: it switches
      // tab, opens the owning layer keyed to the DESTINATION tab, and focuses
      // the control after the switched-in panel mounts. Doing tab-switch / open
      // / focus as separate ordered calls is what broke cross-tab jumps (the
      // host read its stale activeTab between them).
      onNavigate(entry)
    },
    [onOpenChange, onNavigate],
  )

  // ── Focus trap: keep Tab/Shift+Tab inside the dialog while open ──
  const onDialogKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === "Escape") {
        e.preventDefault()
        close()
        return
      }
      if (e.key === "Tab") {
        const focusables = dialogRef.current?.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        )
        if (!focusables || focusables.length === 0) return
        const first = focusables[0]
        const last = focusables[focusables.length - 1]
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault()
          last.focus()
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault()
          first.focus()
        }
      }
    },
    [close],
  )

  // ── Combobox key handling: ↑/↓ move the active option, Enter selects ──
  const onInputKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "ArrowDown") {
        e.preventDefault()
        setActiveIndex((i) => (results.length ? (i + 1) % results.length : 0))
      } else if (e.key === "ArrowUp") {
        e.preventDefault()
        setActiveIndex((i) =>
          results.length ? (i - 1 + results.length) % results.length : 0,
        )
      } else if (e.key === "Enter") {
        e.preventDefault()
        select(results[safeActiveIndex])
      } else if (e.key === "Home") {
        e.preventDefault()
        setActiveIndex(0)
      } else if (e.key === "End") {
        e.preventDefault()
        setActiveIndex(Math.max(0, results.length - 1))
      }
    },
    [results, safeActiveIndex, select],
  )

  const activeOptionId = results[safeActiveIndex]
    ? `${listboxId}-opt-${results[safeActiveIndex].id}`
    : undefined

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="settings-search"
          className="fixed inset-0 z-[100] flex items-start justify-center px-4 pt-[12vh]"
          initial={reduce ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={reduce ? { opacity: 1 } : { opacity: 0 }}
          transition={{ duration: reduce ? 0 : 0.12, ease: "easeOut" }}
        >
          {/* Scrim — click to dismiss (restores focus to the opener). */}
          <div
            className="absolute inset-0 backdrop-blur-sm"
            style={{ backgroundColor: "oklch(0 0 0 / 0.45)" }}
            onClick={() => close()}
            aria-hidden="true"
          />

          {/* Dialog */}
          <motion.div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            onKeyDown={onDialogKeyDown}
            initial={reduce ? false : { opacity: 0, y: 8, scale: 0.99 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reduce ? { opacity: 1 } : { opacity: 0, y: 8, scale: 0.99 }}
            transition={{ duration: reduce ? 0 : 0.16, ease: [0.22, 1, 0.36, 1] }}
            className="relative w-full max-w-xl overflow-hidden rounded-xl border shadow-2xl"
            style={{
              backgroundColor: colors.cardBg,
              borderColor: colors.border,
              color: colors.text,
            }}
          >
            <h2 id={titleId} className="sr-only">
              Search settings
            </h2>

            {/* Search input (combobox) */}
            <div
              className="flex items-center gap-2.5 border-b px-4 py-3"
              style={{ borderColor: colors.border }}
            >
              <Search
                className="h-4 w-4 shrink-0"
                style={{ color: colors.textSecondary }}
              />
              <input
                ref={inputRef}
                type="text"
                role="combobox"
                aria-expanded={open}
                aria-controls={listboxId}
                aria-activedescendant={activeOptionId}
                aria-autocomplete="list"
                aria-label="Search settings"
                placeholder="Search settings…"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value)
                  setActiveIndex(0)
                }}
                onKeyDown={onInputKeyDown}
                className="w-full bg-transparent text-sm focus:outline-none"
                style={{ color: colors.text, fontFamily: "var(--font-mono)" }}
              />
            </div>

            {/* Results (listbox) */}
            <ul
              ref={listRef}
              id={listboxId}
              role="listbox"
              aria-label="Settings"
              className="max-h-72 overflow-y-auto py-1"
            >
              {results.length === 0 && (
                <li
                  className="px-4 py-6 text-center text-xs"
                  style={{ color: colors.textSecondary }}
                >
                  {registry.length === 0
                    ? "No searchable settings registered yet."
                    : "No matching settings."}
                </li>
              )}
              {results.map((entry, index) => {
                const isActive = index === safeActiveIndex
                return (
                  <li
                    key={entry.id}
                    id={`${listboxId}-opt-${entry.id}`}
                    role="option"
                    aria-selected={isActive}
                    data-opt-id={entry.id}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => select(entry)}
                    className="flex cursor-pointer items-center justify-between gap-3 px-4 py-2.5 text-sm"
                    style={{
                      backgroundColor: isActive
                        ? colors.accentMuted
                        : "transparent",
                      color: isActive ? colors.accent : colors.text,
                    }}
                  >
                    <span className="min-w-0 truncate">{entry.label}</span>
                    {entry.tab !== "both" && (
                      <span
                        className="shrink-0 rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em]"
                        style={{
                          backgroundColor: colors.accentMuted,
                          color: colors.accent,
                        }}
                      >
                        {entry.tab === "pretraining" ? "Pretrain" : "Post-train"}
                      </span>
                    )}
                  </li>
                )
              })}
            </ul>

            {/* Footer hint row — the keys */}
            <div
              className="flex items-center gap-4 border-t px-4 py-2.5 text-[10px]"
              style={{
                borderColor: colors.border,
                color: colors.textSecondary,
              }}
            >
              <HintKey label="navigate">
                <KeyCap colors={colors}>↑</KeyCap>
                <KeyCap colors={colors}>↓</KeyCap>
              </HintKey>
              <HintKey label="select">
                <KeyCap colors={colors}>
                  <CornerDownLeft className="h-3 w-3" />
                </KeyCap>
              </HintKey>
              <HintKey label="close">
                <KeyCap colors={colors}>esc</KeyCap>
              </HintKey>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// ---------------------------------------------------------------------------
// HintKey / KeyCap — footer key-caps + their action labels
// ---------------------------------------------------------------------------
function HintKey({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-flex items-center gap-0.5">{children}</span>
      <span className="uppercase tracking-[0.08em]">{label}</span>
    </span>
  )
}

function KeyCap({
  colors,
  children,
}: {
  colors: CalculatorColors
  children: React.ReactNode
}) {
  return (
    <kbd
      className="inline-flex min-w-[1.25rem] items-center justify-center rounded border px-1 py-px text-[10px]"
      style={{
        borderColor: colors.border,
        color: colors.textSecondary,
        fontFamily: "var(--font-mono)",
      }}
    >
      {children}
    </kbd>
  )
}
