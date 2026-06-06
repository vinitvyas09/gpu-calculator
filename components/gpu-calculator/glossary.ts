// ---------------------------------------------------------------------------
// Glossary — the single source of truth for every concept the UI teaches.
//
// Keys are stable identifiers referenced by <Term termKey="..."> (term.tsx) and
// by the layer summary lines. Definitions are 1-2 sentences, novice-readable,
// non-circular, and grounded in spec/research/*.md (citations: spec bundle
// Appendix B§1). No definition is duplicated inline anywhere else — every label
// or summary that explains a term imports GLOSSARY from here.
// ---------------------------------------------------------------------------
export interface GlossaryEntry {
  /** Display label (e.g. "MFU (Model FLOPs Utilization)"). */
  term: string
  /** 1-2 sentence plain-English definition. */
  def: string
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
  parameters: {
    term: "Parameters",
    def: "The learned weights inside the model — the numbers it adjusts during training. Model size is usually quoted as a parameter count (e.g. 7B = 7 billion).",
  },
  tokens: {
    term: "Tokens",
    def: "The chunks of text a model reads and writes — roughly a word or word-piece. Training data and context length are both measured in tokens.",
  },
  uniqueTokens: {
    term: "Unique tokens",
    def: "How much distinct text your dataset actually contains, before any repetition. If total training tokens exceed this, the model sees some data more than once (multiple epochs).",
  },
  sequenceLength: {
    term: "Sequence length",
    def: "How many tokens the model processes at once in a single training example — its context window during training. Longer sequences cost more memory and compute, growing steeply with attention.",
  },
  microBatch: {
    term: "Micro-batch size",
    def: "The number of sequences one GPU processes in a single forward/backward pass. Kept small to fit in memory; the real (global) batch is built up from many micro-batches.",
  },
  gradientAccumulation: {
    term: "Gradient accumulation",
    def: "Running several micro-batches and summing their gradients before updating the weights, so you can reach a large effective batch size without the memory of one giant batch.",
  },
  globalBatchSize: {
    term: "Global batch size",
    def: "The effective batch the optimizer actually updates on: micro-batch × gradient-accumulation steps × data-parallel GPUs. This is the number that affects how the model learns.",
  },
  precision: {
    term: "Precision (number format)",
    def: "How many bits each number uses. Lower precision (fp16, bf16, fp8) is faster and uses less memory than fp32, at some risk to numerical stability.",
  },
  bf16: {
    term: "bf16 (bfloat16)",
    def: "A 16-bit format with the same large exponent range as fp32 but fewer precision bits. The default for modern training because it rarely needs loss scaling to stay stable.",
  },
  fp16: {
    term: "fp16 (half precision)",
    def: "A 16-bit format with more precision bits than bf16 but a much smaller range, so it often needs loss scaling to keep small gradients from underflowing to zero.",
  },
  fp32: {
    term: "fp32 (single precision)",
    def: "The classic 32-bit format. Most accurate and most memory-hungry; modern training keeps an fp32 'master copy' of weights even when computing in 16-bit.",
  },
  fp8: {
    term: "fp8 (8-bit float)",
    def: "An 8-bit format on the newest hardware (H100 and later) that can roughly double matmul throughput. Used for the heavy matrix multiplies while a higher-precision copy preserves accuracy.",
  },
  mixedPrecision: {
    term: "Mixed precision",
    def: "Computing the forward/backward pass in 16-bit for speed and memory, while keeping an fp32 master copy of the weights and accumulating into fp32 for stability. This is why Adam-style training costs ~16 bytes per parameter, not 4.",
  },
  optimizerStates: {
    term: "Optimizer states",
    def: "Extra per-parameter memory the optimizer keeps between steps — for Adam, the running mean and variance plus the fp32 master weights. Often the single largest slice of training memory.",
  },
  activationCheckpointing: {
    term: "Activation checkpointing",
    def: "Saving memory by discarding most intermediate activations during the forward pass and recomputing them in the backward pass. Trades extra compute (~30% more) for large memory savings.",
  },
  selectiveCheckpointing: {
    term: "Selective recomputation",
    def: "A lighter form of checkpointing that only recomputes the cheap-but-memory-heavy attention activations, leaving the expensive matmuls stored. Cuts the worst activation memory with minimal extra compute.",
  },
  flashAttention: {
    term: "Flash attention",
    def: "An attention algorithm that never writes the full N×N attention matrix to GPU memory, computing it in fast on-chip tiles instead. Gives the exact same result while removing attention's quadratic memory cost.",
  },
  mfu: {
    term: "MFU (Model FLOPs Utilization)",
    def: "The fraction of a GPU's peak math throughput your training actually achieves, after communication, memory, and bubble overhead. Real LLM runs typically land around 35-55%.",
  },
  tflops: {
    term: "TFLOPS",
    def: "Trillions of floating-point operations per second — a measure of raw math throughput. GPUs are rated at a peak TFLOPS; MFU tells you how much of it you reach.",
  },
  vram: {
    term: "VRAM",
    def: "The memory on a GPU that must hold the weights, gradients, optimizer states, and activations during training. Running out of it ('OOM') is the most common reason a run won't fit.",
  },
  kvCache: {
    term: "KV cache",
    def: "Stored key/value vectors from earlier tokens so a model doesn't recompute them while generating text. It matters for generation and for methods like PPO/GRPO that produce samples during training.",
  },
  dataParallel: {
    term: "Data parallelism (DP)",
    def: "Putting a full copy of the model on each GPU and feeding each a different slice of the batch, then averaging gradients. The simplest way to scale, but every GPU needs to hold the whole model.",
  },
  tensorParallel: {
    term: "Tensor parallelism (TP)",
    def: "Splitting each weight matrix across GPUs so they jointly compute one layer, syncing with an all-reduce. Shrinks per-GPU memory but needs fast interconnect, so it's usually kept within a node. Must evenly divide the attention heads.",
  },
  pipelineParallel: {
    term: "Pipeline parallelism (PP)",
    def: "Splitting the model's layers into stages on different GPUs, passing activations down the line like an assembly line. Scales across nodes well, but leaves some GPUs idle (the 'bubble').",
  },
  contextParallel: {
    term: "Context parallelism (CP)",
    def: "Splitting a single long sequence across GPUs along the token dimension, so each holds part of the context. Lets you train on very long sequences that wouldn't fit on one GPU.",
  },
  expertParallel: {
    term: "Expert parallelism (EP)",
    def: "Placing different experts of a Mixture-of-Experts model on different GPUs and routing each token to the GPUs holding its chosen experts. The standard way to scale MoE models.",
  },
  virtualPipeline: {
    term: "Virtual pipeline (interleaving)",
    def: "Giving each pipeline GPU several small, non-contiguous chunks of layers instead of one big block, which shrinks the idle pipeline bubble at the cost of a bit more communication.",
  },
  zero: {
    term: "ZeRO",
    def: "A DeepSpeed technique that removes the wasteful duplication in data parallelism by partitioning optimizer states, gradients, and parameters across GPUs instead of replicating them. Comes in three stages of increasing savings.",
  },
  zeroStages: {
    term: "ZeRO stages",
    def: "Stage 1 shards optimizer states (~4× less model-state memory), Stage 2 also shards gradients (~8×), and Stage 3 also shards parameters (memory falls roughly linearly with GPU count). Stages 1-2 add almost no extra communication; Stage 3 adds about 50%.",
  },
  fsdp: {
    term: "FSDP",
    def: "PyTorch's Fully Sharded Data Parallel — its native equivalent of ZeRO that shards parameters, gradients, and optimizer states across data-parallel GPUs. FULL_SHARD ≈ ZeRO-3, SHARD_GRAD_OP ≈ ZeRO-2.",
  },
  distributedStrategy: {
    term: "Distributed strategy",
    def: "How post-training model states are placed across GPUs. Replicated (DDP) keeps full parameters, gradients, and optimizer states on every GPU, so adding GPUs only splits the batch. Sharded (FSDP / ZeRO-3) divides those states across all GPUs — frozen base weights included — at the cost of extra all-gather communication and a transient working buffer (modeled as 2× the largest transformer block, capped at 1B parameters). Activations always stay per-GPU.",
  },
  pipelineBubble: {
    term: "Pipeline bubble",
    def: "The idle time at the start and end of each step when pipeline stages are waiting to be filled or drained. More micro-batches (or virtual pipeline) shrink it; a common rule of thumb is at least 4× the number of pipeline stages.",
  },
  moe: {
    term: "Mixture of Experts (MoE)",
    def: "A model where each layer has many 'expert' sub-networks but each token only uses a few of them, so total parameters can be huge while compute per token stays modest. DeepSeek-V3 and Mixtral are examples.",
  },
  experts: {
    term: "Experts",
    def: "The parallel sub-networks inside an MoE layer. There can be hundreds; a router sends each token to only a small number of them, so most experts sit idle for any given token.",
  },
  topK: {
    term: "Top-k routing",
    def: "How many experts each token is sent to per MoE layer (often 1 or 2). Higher k means more experts active per token — more quality, more compute and communication.",
  },
  router: {
    term: "Router (gating)",
    def: "The small network in each MoE layer that scores the experts for a token and picks the top-k to run. Its choices determine which experts do the work.",
  },
  loadBalancing: {
    term: "Load balancing (MoE)",
    def: "Keeping tokens spread evenly across experts so none is overloaded while others idle. Imbalance wastes capacity and is usually discouraged with an auxiliary loss.",
  },
  gqa: {
    term: "GQA (Grouped-Query Attention)",
    def: "Letting several query heads share one key/value head, between full multi-head and single-head attention. Shrinks the KV cache and memory with little quality loss; used by Llama 2/3 70B.",
  },
  mqa: {
    term: "MQA (Multi-Query Attention)",
    def: "The extreme of GQA where all query heads share a single key/value head. Smallest KV cache and fastest generation, at some cost to quality.",
  },
  mla: {
    term: "MLA (Multi-head Latent Attention)",
    def: "DeepSeek's attention variant that compresses keys and values into a small shared latent vector, cutting KV-cache memory dramatically while keeping multi-head expressiveness.",
  },
  rope: {
    term: "RoPE (Rotary Position Embedding)",
    def: "A way of encoding token positions by rotating the query/key vectors, rather than adding a learned position vector. The de-facto standard in modern LLMs and friendly to long-context extension.",
  },
  swiglu: {
    term: "SwiGLU",
    def: "A gated feed-forward layer that multiplies two projections, one passed through a SiLU activation. It outperforms a plain MLP, which is why its hidden dimension is usually sized to ~⅔ × the naive width.",
  },
  tiedEmbeddings: {
    term: "Tied embeddings",
    def: "Reusing the same weight matrix for the input token embedding and the output projection (the LM head). Saves parameters and memory; common in smaller models like GPT-2.",
  },
  chinchillaOptimal: {
    term: "Chinchilla-optimal",
    def: "The compute-optimal balance between model size and training data found by the Chinchilla study: for a fixed compute budget, scale parameters and tokens together (about 20 tokens per parameter). Bigger isn't better if it's undertrained.",
  },
  tokensPerParameter: {
    term: "Tokens per parameter",
    def: "Training tokens divided by parameter count. Around 20 is the classic compute-optimal sweet spot; far below it underuses the model, far above it spends compute for diminishing loss gains.",
  },
  predictedLoss: {
    term: "Predicted loss (nats)",
    def: "An estimate of the model's final training loss in nats (natural-log units), from the Chinchilla scaling law for your size and token budget. Lower is better; it's a rough forecast, not a guarantee.",
  },
  criticalBatchSize: {
    term: "Critical batch size",
    def: "The batch size beyond which adding more parallel data stops speeding up training and mostly wastes compute. It grows as the loss falls, so larger batches pay off later in a run.",
  },
  sft: {
    term: "SFT (Supervised Fine-Tuning)",
    def: "Fine-tuning a base model on curated input→output examples so it follows instructions or a target style. The usual first step of post-training, and the cheapest.",
  },
  dpo: {
    term: "DPO (Direct Preference Optimization)",
    def: "Aligning a model from pairs of preferred vs. rejected responses directly, without training a separate reward model. Simpler and lighter than PPO, though it keeps a frozen reference copy in memory.",
  },
  ppo: {
    term: "PPO (Proximal Policy Optimization)",
    def: "A reinforcement-learning method (the classic RLHF algorithm) that improves the model against a reward signal while a critic estimates value. Powerful but memory-heavy: it holds the policy, a reference model, a reward model, and a critic at once.",
  },
  criticModel: {
    term: "Critic (value model)",
    def: "In PPO, a helper network that estimates the expected future reward of a partial response, used to reduce variance in the updates. It's roughly the size of the model being trained, so it adds substantial memory.",
  },
  rewardModel: {
    term: "Reward model",
    def: "A model that scores how good a response is, trained from human preferences. PPO and GRPO query it during training to decide which outputs to reinforce.",
  },
  grpo: {
    term: "GRPO (Group Relative Policy Optimization)",
    def: "An RL method that drops PPO's critic and instead samples a group of responses per prompt, scoring each relative to the group average. Cheaper than PPO because there's no value network to hold.",
  },
  lora: {
    term: "LoRA (Low-Rank Adaptation)",
    def: "Freezing the base model and training only small low-rank 'adapter' matrices added to chosen layers. Slashes trainable parameters and optimizer memory while reaching near-full-fine-tuning quality.",
  },
  loraRank: {
    term: "LoRA rank (r)",
    def: "The size of the LoRA adapter's bottleneck — higher rank means more capacity and more trainable parameters. Common values are 8 to 64.",
  },
  loraAlpha: {
    term: "LoRA alpha",
    def: "A scaling factor that controls how strongly the LoRA adapter is applied to the frozen base weights. It's typically set to a small multiple of the rank.",
  },
  loraTargetModules: {
    term: "LoRA target modules",
    def: "Which weight matrices get adapters — e.g. the attention query/value projections, or all linear layers. Targeting more modules raises quality and trainable-parameter count.",
  },
  qlora: {
    term: "QLoRA",
    def: "LoRA on top of a base model stored in 4-bit (NF4), dequantized on the fly for the math. It let a 65B model fine-tune on a single 48 GB GPU with no quality loss versus 16-bit.",
  },
  mezo: {
    term: "MeZO (zeroth-order)",
    def: "A fine-tuning method that estimates gradients from forward passes alone, so it stores no gradients or optimizer states — only the parameters. Extremely memory-light, but slower to converge.",
  },
  checkpoint: {
    term: "Checkpoint",
    def: "A saved snapshot of the model (and optimizer state) written to storage during training so a crashed run can resume. Frequent checkpoints cost storage but reduce lost work after a failure.",
  },
  failureOverhead: {
    term: "Failure overhead",
    def: "Extra wall-clock time and cost from hardware failures: detecting the crash, restarting, and re-doing the work since the last checkpoint. It grows with GPU count and run length.",
  },
}
