# GPU Calculator Research Sources

Comprehensive list of open source tools, calculators, blog posts, papers, and resources related to LLM training GPU/memory estimation.

---

## Standalone Calculators & Tools (GitHub)


| Name                                   | URL                                                                                                                    | What It Does                                                                            |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [x] EleutherAI Cookbook (calc/)            | [https://github.com/EleutherAI/cookbook/tree/main/calc](https://github.com/EleutherAI/cookbook/tree/main/calc)         | `calc_transformer_flops.py`, `calc_transformer_mem.py`, `calc_transformer_params.py`    |
| [x] cli99/llm-analysis                     | [https://github.com/cli99/llm-analysis](https://github.com/cli99/llm-analysis)                                         | Latency and memory analysis for training/inference given model, GPU, parallelism config |
| [x] RahulSChand/gpu_poor                   | [https://github.com/RahulSChand/gpu_poor](https://github.com/RahulSChand/gpu_poor)                                     | Token/s and GPU memory for any LLM; supports llama.cpp/ggml/bnb/QLoRA quantization      |
| [x] isEmmanuelOlowe/llm-cost-estimator     | [https://github.com/isEmmanuelOlowe/llm-cost-estimator](https://github.com/isEmmanuelOlowe/llm-cost-estimator)         | VRAM, FLOPs, tokens/s, and cloud costs (AWS/GCP/Azure)                                  |
| [x] debjitpaul/flop_calculator             | [https://github.com/debjitpaul/flop_calculator](https://github.com/debjitpaul/flop_calculator)                         | FLOPs and MFU for dense Transformer and MoE architectures                               |
| [x] MrYxJ/calculate-flops.pytorch          | [https://github.com/MrYxJ/calculate-flops.pytorch](https://github.com/MrYxJ/calculate-flops.pytorch)                   | Theoretical FLOPs, MACs, and parameters for neural networks including Transformers      |
| [x] hunkim/llm_gpu_cal                     | [https://github.com/hunkim/llm_gpu_cal](https://github.com/hunkim/llm_gpu_cal)                                         | Simple GPU calculator for LLM pre-training and fine-tuning                              |
| [x] JGalego/llm-calc                       | [https://github.com/JGalego/llm-calc](https://github.com/JGalego/llm-calc)                                             | Estimates compute power and storage to train and host a model                           |
| [x] manuelescobar-dev/LLM-Tools            | [https://github.com/manuelescobar-dev/LLM-Tools](https://github.com/manuelescobar-dev/LLM-Tools)                       | Calculator for LLM system requirements (memory to run or train)                         |
| [x] shchoice/LLM-GPU-Memory-Estimator      | [https://github.com/shchoice/LLM-GPU-Memory-Estimator](https://github.com/shchoice/LLM-GPU-Memory-Estimator)           | Open-source calculator for LLM GPU memory requirements                                  |
| [x] TechNavii/LLM-Memory-Calculator        | [https://github.com/TechNavii/LLM-Memory-Calculator](https://github.com/TechNavii/LLM-Memory-Calculator)               | Web app for LLM memory and performance metrics across GPU configs and Apple Silicon     |
| [x] AndreaPi/llm-memory-calculator         | [https://github.com/AndreaPi/llm-memory-calculator](https://github.com/AndreaPi/llm-memory-calculator)                 | Script to estimate memory for training, finetuning, or inference                        |
| [x] erans/selfhostllm                      | [https://github.com/erans/selfhostllm](https://github.com/erans/selfhostllm)                                           | GPU memory requirements and max concurrent requests for self-hosted inference           |
| [x] GPUforLLM/llm-vram-calculator          | [https://github.com/GPUforLLM/llm-vram-calculator](https://github.com/GPUforLLM/llm-vram-calculator)                   | VRAM calculator accounting for GGUF overhead, GQA context memory, offloading            |
| [x] thisismindo/llm-vram-estimator         | [https://github.com/thisismindo/llm-vram-estimator](https://github.com/thisismindo/llm-vram-estimator)                 | VRAM and inference performance for 138+ predefined LLM models                           |
| [x] taehokim20/LLMem                       | [https://github.com/taehokim20/LLMem](https://github.com/taehokim20/LLMem)                                             | GPU memory estimation for fine-tuning with distributed methods (1.6% error)             |
| [x] TitaniumMonkey/LLM_Hardware_Calculator | [https://github.com/TitaniumMonkey/LLM_Hardware_Calculator](https://github.com/TitaniumMonkey/LLM_Hardware_Calculator) | GPU VRAM and disk space; fetches real-time model data from Hugging Face                 |
| [x] adarshxs/TokenTally                    | [https://github.com/adarshxs/TokenTally](https://github.com/adarshxs/TokenTally)                                       | Token cost estimation across LLM platforms                                              |
| [x] qoofyk/LLM_Sizing_Guide                | [https://github.com/qoofyk/LLM_Sizing_Guide](https://github.com/qoofyk/LLM_Sizing_Guide)                               | Memory footprint, capacity, and latency for LLMs on VMware Private AI                   |


## HuggingFace Spaces & Web Apps


| Name                                     | URL                                                                                                                                                          | What It Does                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| [x] HF Accelerate Model Memory Usage         | [https://huggingface.co/spaces/hf-accelerate/model-memory-usage](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)                             | Estimate memory for any HF model by dtype                               |
| [x] NyxKrage LLM VRAM Calculator             | [https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator)                         | Enter HF model, quantization format, context length for VRAM estimate   |
| [x] Adam Casson Transformer FLOPs Calculator | [https://huggingface.co/spaces/adamcasson/transformer-flops-calculator](https://huggingface.co/spaces/adamcasson/transformer-flops-calculator)               | Interactive FLOPs calculator (OpenAI and DeepMind methods)              |
| [x] Lambda LLM Calculator                    | [https://huggingface.co/spaces/lambdabrendan/Lambda-LLM-Calculator](https://huggingface.co/spaces/lambdabrendan/Lambda-LLM-Calculator)                       | VRAM for inference, full training, and fine-tuning across GPUs          |
| [x] LLM Training Time and Cost Calculator    | [https://huggingface.co/spaces/ghost613/LLM-Training-Time-and-Cost-Calculator](https://huggingface.co/spaces/ghost613/LLM-Training-Time-and-Cost-Calculator) | Training time and cost given GPU, model size, dataset                   |
| [x] dhlak LLM VRAM Calculator                | [https://huggingface.co/spaces/dhlak/llm-vram-calc](https://huggingface.co/spaces/dhlak/llm-vram-calc)                                                       | VRAM for training or inference given model, GPU, batch size, seq length |
| [x] LLM Finetuning Memory Calculator         | [https://huggingface.co/spaces/aelrefai/llm-finetuning-memory-calculator](https://huggingface.co/spaces/aelrefai/llm-finetuning-memory-calculator)           | Finetuning memory with QLoRA/LoRA/full                                  |
| [x] Can It Run LLM?                          | [https://huggingface.co/spaces/Vokturz/can-it-run-llm](https://huggingface.co/spaces/Vokturz/can-it-run-llm)                                                 | Checks if a given GPU can run a specific model                          |
| [x] LipikaAggarwal LLM Memory Estimator      | [https://lipikaaggarwal.github.io/LLM-Memory-Estimator/](https://lipikaaggarwal.github.io/LLM-Memory-Estimator/)                                             | Web-based GPU memory calculator                                         |
| [x] deadjoe LLM Memory Calculator            | [https://deadjoe.github.io/llm-memory-calculator/](https://deadjoe.github.io/llm-memory-calculator/)                                                         | Web-based LLM memory calculator                                         |


## Utilities Inside Larger Frameworks


| Name                                 | URL                                                                                                                                                                                | What It Does                                                            |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [x] DeepSpeed Memory Estimation          | [https://deepspeed.readthedocs.io/en/latest/memory.html](https://deepspeed.readthedocs.io/en/latest/memory.html)                                                                   | `estimate_zero2/3_model_states_mem_needs_*` functions                   |
| [x] HuggingFace Accelerate estimate.py   | [https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/estimate.py](https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/estimate.py) | `accelerate estimate-memory` CLI command                                |
| [x] NVIDIA Megatron-LM                   | [https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)                                                                                                     | Memory profiling and training config tools for large-scale transformers |
| [x] NVIDIA TransformerEngine             | [https://github.com/NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine)                                                                                         | FP8/FP4 precision library with memory-related utilities                 |
| [x] NVIDIA NeMo                          | [https://github.com/NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo)                                                                                                         | AutoConfigurator parallelism heuristics; NeMo-Aligner PPO/DPO memory    |
| [x] MosaicML LLM Foundry (benchmarking)  | [https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/benchmarking/README.md](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/benchmarking/README.md)   | Throughput benchmarks with MFU/HFU formulas for 125M-70B models         |
| [x] facebookresearch/fvcore (flop_count) | [https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md)                                 | Operator-level and module-level FLOPs counting for PyTorch              |
| [x] Stonesjtu/pytorch_memlab             | [https://github.com/Stonesjtu/pytorch_memlab](https://github.com/Stonesjtu/pytorch_memlab)                                                                                         | Line-profiler-style CUDA memory profiler for PyTorch                    |
| [x] HuggingFace Nanotron                 | [https://github.com/huggingface/nanotron](https://github.com/huggingface/nanotron)                                                                                                 | Training framework with memory estimation utilities                     |


## Notebooks


| Name                                      | URL                                                                                                                                                  | What It Does                                                      |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [x] karpathy/nanoGPT transformer_sizing.ipynb | [https://github.com/karpathy/nanoGPT/blob/master/transformer_sizing.ipynb](https://github.com/karpathy/nanoGPT/blob/master/transformer_sizing.ipynb) | FLOPs, parameters, peak memory, checkpoint size, MFU              |
| [x] karpathy/llm.c                            | [https://github.com/karpathy/llm.c](https://github.com/karpathy/llm.c)                                                                               | LLM training in raw C/CUDA with FLOPs and memory benchmarking     |
| [x] EleutherAI/nanoGPT-mup scaling_laws.ipynb | [https://github.com/EleutherAI/nanoGPT-mup/blob/master/scaling_laws.ipynb](https://github.com/EleutherAI/nanoGPT-mup/blob/master/scaling_laws.ipynb) | Chinchilla scaling law results and compute-optimal model guidance |


## Scaling Laws Tools


| Name                            | URL                                                                                                      | What It Does                                                              |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [x] kyo-takano/chinchilla           | [https://github.com/kyo-takano/chinchilla](https://github.com/kyo-takano/chinchilla)                     | Toolkit for scaling law research; compute-optimal allocation              |
| [x] nikhilsardana/beyond-chinchilla | [https://github.com/nikhilsardana/beyond-chinchilla](https://github.com/nikhilsardana/beyond-chinchilla) | Given two of {compute, params, tokens}, returns the third + expected loss |
| [x] shehper/scaling_laws            | [https://github.com/shehper/scaling_laws](https://github.com/shehper/scaling_laws)                       | Implementation of Kaplan et al. scaling laws using nanoGPT                |
| [x] huggingface/datablations        | [https://github.com/huggingface/datablations](https://github.com/huggingface/datablations)               | 400 training runs exploring data repetition and compute budgets           |


---

## Blog Posts & Technical Articles

### Memory Estimation & GPU Requirements


| Title                                                          | URL                                                                                                                                                                                                                                                                                                                | What It Covers                                                                        |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| [x] Transformer Math 101 (EleutherAI)                              | [https://blog.eleuther.ai/transformer-math/](https://blog.eleuther.ai/transformer-math/)                                                                                                                                                                                                                           | THE canonical reference: C=6PD, activation memory, parameter counting, throughput     |
| [x] How to Train Really Large Models on Many GPUs (Lilian Weng)    | [https://lilianweng.github.io/posts/2021-09-25-train-large/](https://lilianweng.github.io/posts/2021-09-25-train-large/)                                                                                                                                                                                           | Survey of parallelism paradigms, mixed precision, memory-saving techniques            |
| [x] Transformer Memory Arithmetic (erees.dev)                      | [https://erees.dev/transformer-memory/](https://erees.dev/transformer-memory/)                                                                                                                                                                                                                                     | Byte-level accounting of every memory component during GPT training                   |
| [x] Model Training Anatomy (HuggingFace)                           | [https://huggingface.co/docs/transformers/model_memory_anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy)                                                                                                                                                                                     | All GPU memory components during training: weights, optimizer, gradients, activations |
| [x] Efficient Training on a Single GPU (HuggingFace)               | [https://huggingface.co/docs/transformers/perf_train_gpu_one](https://huggingface.co/docs/transformers/perf_train_gpu_one)                                                                                                                                                                                         | Gradient accumulation, checkpointing, mixed precision, memory-efficient optimizers    |
| [x] Optimizing Memory for Training LLMs (Sebastian Raschka)        | [https://sebastianraschka.com/blog/2023/pytorch-memory-optimization.html](https://sebastianraschka.com/blog/2023/pytorch-memory-optimization.html)                                                                                                                                                                 | 9 techniques to reduce memory ~20x in PyTorch                                         |
| [x] Estimating vRAM (Hamel Husain)                                 | [https://hamel.dev/notes/llm/finetuning/estimating_vram.html](https://hamel.dev/notes/llm/finetuning/estimating_vram.html)                                                                                                                                                                                         | Practical notebook for VRAM estimation in fine-tuning                                 |
| [x] Train Big, Plan Smart (Shreyans92)                             | [https://shreyans92.github.io/2025-05-23-LLMMemory/](https://shreyans92.github.io/2025-05-23-LLMMemory/)                                                                                                                                                                                                           | Memory estimation walkthrough for LLaMA 3-8B with NeMo                                |
| [x] GPT Training Memory Estimation - NeMo Practice (Jianbin Chang) | [https://shjwudp.github.io/blog/2023/gpt-training-memory-estimation-nemo-training-practice/](https://shjwudp.github.io/blog/2023/gpt-training-memory-estimation-nemo-training-practice/)                                                                                                                           | Memory profiling for GPT models on DGX-A100                                           |
| [x] Understanding GPU Memory for Training LLMs (Max Shap)          | [https://medium.com/@maxshapp/understanding-and-estimating-gpu-memory-demands-for-training-llms-in-practise-c5ef20a4baff](https://medium.com/@maxshapp/understanding-and-estimating-gpu-memory-demands-for-training-llms-in-practise-c5ef20a4baff)                                                                 | DeepSpeed stages, practical memory reduction strategies                               |
| [x] Estimating Memory Requirements of Transformers (Schartz)       | [https://schartz.github.io/blog/estimating-memory-requirements-of-transformers/](https://schartz.github.io/blog/estimating-memory-requirements-of-transformers/)                                                                                                                                                   | Clean formulas for model memory and activation memory                                 |
| [x] Calculating Memory Footprint for LLMs (Shawn/Medium)           | [https://medium.com/@xiaxiami/calculating-memory-footprint-for-large-language-models-llms-a-complete-guide-98ac3fdfdbf6](https://medium.com/@xiaxiami/calculating-memory-footprint-for-large-language-models-llms-a-complete-guide-98ac3fdfdbf6)                                                                   | End-to-end memory footprint calculation guide                                         |
| [x] Estimate GPU Memory for LLM Fine-Tuning (Red Hat)              | [https://developers.redhat.com/articles/2026/03/04/estimate-gpu-memory-llm-fine-tuning-red-hat-ai](https://developers.redhat.com/articles/2026/03/04/estimate-gpu-memory-llm-fine-tuning-red-hat-ai)                                                                                                               | Memory formulas with Red Hat's `memory_estimator.py`                                  |
| [x] Decoding High-Bandwidth Memory (Google Cloud)                  | [https://cloud.google.com/blog/topics/developers-practitioners/decoding-high-bandwidth-memory-a-practical-guide-to-gpu-memory-for-fine-tuning-ai-models/](https://cloud.google.com/blog/topics/developers-practitioners/decoding-high-bandwidth-memory-a-practical-guide-to-gpu-memory-for-fine-tuning-ai-models/) | GPU memory for fine-tuning with HBM breakdown                                         |
| [x] Which GPU for Deep Learning (Tim Dettmers)                     | [https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)                                                                                                                                                                                 | GPU recommendation guide with memory and cost/performance analysis                    |


### FLOPs, MFU & Training Efficiency


| Title                                                                | URL                                                                                                                                                                                                          | What It Covers                                                             |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| [x] Transformer FLOPs (Adam Casson)                                      | [https://www.adamcasson.com/posts/transformer-flops](https://www.adamcasson.com/posts/transformer-flops)                                                                                                     | Detailed FLOPs derivation comparing OpenAI and DeepMind methods            |
| [x] The FLOPs Calculus of Language Model Training (Bahdanau)             | [https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)             | Why naive FLOPs counting underestimates training time                      |
| [x] Demystifying 6ND FLOPs (Dominik Farhan)                              | [https://dominikfarhan.com/2025/09/14/demystify-6nd/](https://dominikfarhan.com/2025/09/14/demystify-6nd/)                                                                                                   | Step-by-step derivation of the 6ND approximation                           |
| [x] How to Calculate FLOPs in Transformers (Gao Hongnan)                 | [https://www.gaohongnan.com/playbook/training/how_to_calculate_flops_in_transformer_based_models.html](https://www.gaohongnan.com/playbook/training/how_to_calculate_flops_in_transformer_based_models.html) | Per-layer and total FLOPs with the 6ND formula                             |
| [x] Understanding FLOPs, MFU, and Computational Efficiency (Debjit Paul) | [https://debjitpaul.github.io/blog/2025/compute/](https://debjitpaul.github.io/blog/2025/compute/)                                                                                                           | FLOPs counting, MFU calculations, MoE architectures                        |
| [x] Using Model Flops Utilization (MFU) (Jaideep Ray)                    | [https://medium.com/better-ml/using-model-flops-utilization-mfu-7b17de07faec](https://medium.com/better-ml/using-model-flops-utilization-mfu-7b17de07faec)                                                   | MFU as a hardware-agnostic training efficiency metric                      |
| [x] Model FLOPs Utilization (Glenn Klockwood)                            | [https://www.glennklockwood.com/garden/MFU](https://www.glennklockwood.com/garden/MFU)                                                                                                                       | MFU definition comparing Google and Meta formulations                      |
| [x] I Paid for the Whole GPU (Modal Blog)                                | [https://modal.com/blog/gpu-utilization-guide](https://modal.com/blog/gpu-utilization-guide)                                                                                                                 | GPU utilization framework across allocation, kernel, and arithmetic levels |
| [x] Transformer Training Costs (Continuum Labs)                          | [https://training.continuumlabs.ai/infrastructure/data-and-memory/transformer-training-costs](https://training.continuumlabs.ai/infrastructure/data-and-memory/transformer-training-costs)                   | Memory formulas with different recomputation strategies                    |


### Distributed Training & Parallelism


| Title                                                         | URL                                                                                                                                                                                                                                                                                              | What It Covers                                                  |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| [x] Parallelism Methods (HuggingFace)                             | [https://huggingface.co/docs/transformers/perf_train_gpu_many](https://huggingface.co/docs/transformers/perf_train_gpu_many)                                                                                                                                                                     | Combining data, tensor, pipeline, and ZeRO parallelism          |
| [x] ZeRO & DeepSpeed (Microsoft Research)                         | [https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) | How ZeRO partitions optimizer states, gradients, and parameters |
| [x] DeepSpeed ZeRO Tutorial                                       | [https://www.deepspeed.ai/tutorials/zero/](https://www.deepspeed.ai/tutorials/zero/)                                                                                                                                                                                                             | ZeRO Stage 1/2/3 with configs and memory reduction mechanics    |
| [x] Going Deep on DeepSpeed (Stephen Diehl)                       | [https://www.stephendiehl.com/posts/deepspeed/](https://www.stephendiehl.com/posts/deepspeed/)                                                                                                                                                                                                   | ZeRO stages and their 4x/8x/16x memory reduction ratios         |
| [x] ZeRO-Infinity (Microsoft Research)                            | [https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/)           | Offloading to NVMe for trillion-parameter training              |
| [x] FSDP Announcement (Meta Engineering)                          | [https://engineering.fb.com/2021/07/15/open-source/fsdp/](https://engineering.fb.com/2021/07/15/open-source/fsdp/)                                                                                                                                                                               | Sharding params/gradients/optimizer states                      |
| [x] PyTorch FSDP API (PyTorch Blog)                               | [https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)                                                                                                                                   | FSDP as a drop-in DDP replacement                               |
| [x] Everything about Distributed Training (Sumanth R Hegde)       | [https://sumanthrh.com/post/distributed-and-efficient-finetuning/](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)                                                                                                                                                             | DeepSpeed ZeRO vs FSDP, practical guidelines                    |
| [x] Fit More and Train Faster with ZeRO (HuggingFace/Stas Bekman) | [https://huggingface.co/blog/zero-deepspeed-fairscale](https://huggingface.co/blog/zero-deepspeed-fairscale)                                                                                                                                                                                     | Practical walkthrough of ZeRO with memory savings demos         |
| [x] Scaling Llama 3 Training (Meta, ISCA 2025)                    | [https://dl.acm.org/doi/10.1145/3695053.3731410](https://dl.acm.org/doi/10.1145/3695053.3731410)                                                                                                                                                                                                 | 4D parallelism for Llama 3 405B on 16K H100s                    |
| [x] Scaling LM Training to 1T Params with Megatron (NVIDIA Blog)  | [https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)                                                                         | Blog companion to Megatron-LM papers                            |
| [x] Parallelism and Memory Optimization (Yue Shui)                | [https://syhya.github.io/posts/2025-03-01-train-llm/](https://syhya.github.io/posts/2025-03-01-train-llm/)                                                                                                                                                                                       | Summary of all distributed parallel training techniques         |


### Activation Memory & Checkpointing


| Title                                                       | URL                                                                                                                                                                                                                                                  | What It Covers                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| [x] Activation Recomputation (NVIDIA NeMo)                      | [https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/optimizations/activation_recomputation.html](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/optimizations/activation_recomputation.html) | Full vs selective recomputation with config examples |
| [x] Activation Checkpointing Techniques (PyTorch Blog)          | [https://pytorch.org/blog/activation-checkpointing-techniques/](https://pytorch.org/blog/activation-checkpointing-techniques/)                                                                                                                       | `torch.utils.checkpoint` and selective checkpointing |
| [x] Gradient Checkpointing (MLWorks/Medium)                     | [https://medium.com/mlworks/gradient-checkpointing-the-unsung-hero-of-llm-training-ac2bbe5d4396](https://medium.com/mlworks/gradient-checkpointing-the-unsung-hero-of-llm-training-ac2bbe5d4396)                                                     | O(N) to O(sqrt(N)) memory reduction                  |
| [x] Gradient Accumulation and Checkpointing (Aman's AI Journal) | [https://aman.ai/primers/ai/grad-accum-checkpoint/](https://aman.ai/primers/ai/grad-accum-checkpoint/)                                                                                                                                               | Memory tradeoff analysis for both techniques         |


### Mixed Precision


| Title                                              | URL                                                                                                                                                                                                | What It Covers                                                |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| [x] Train With Mixed Precision (NVIDIA Docs)           | [https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)                       | Canonical guide on FP16/BF16 mixed precision and loss scaling |
| [x] Mixed Precision Training Blog (NVIDIA)             | [https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/)                               | FP32 master weights, FP16 forward/backward, loss scaling      |
| [x] Mixed Precision Training in PyTorch (PyTorch Blog) | [https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) | torch.amp: bfloat16 vs float16, TF32 mode                     |


### Scaling Laws


| Title                                                  | URL                                                                                                                                                            | What It Covers                                                      |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [x] Chinchilla Scaling Laws in Plain English               | [https://lifearchitect.ai/chinchilla/](https://lifearchitect.ai/chinchilla/)                                                                                   | Accessible non-technical explanation                                |
| [x] How Long Should You Train Your LM (Databricks)         | [https://www.databricks.com/blog/how-long-should-you-train-your-language-model](https://www.databricks.com/blog/how-long-should-you-train-your-language-model) | Modified scaling law accounting for inference cost                  |
| [x] Scaling Laws for LLMs: GPT-3 to o3 (Cameron Wolfe)     | [https://cameronrwolfe.substack.com/p/llm-scaling-laws](https://cameronrwolfe.substack.com/p/llm-scaling-laws)                                                 | Evolution from Kaplan through Chinchilla to modern reasoning models |
| [x] Language Model Scaling Laws and GPT-3 (Cameron Wolfe)  | [https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt](https://cameronrwolfe.substack.com/p/language-model-scaling-laws-and-gpt)           | Original OpenAI scaling laws and GPT-3 implications                 |
| [x] PaLM: Efficiently Training Massive LMs (Cameron Wolfe) | [https://cameronrwolfe.substack.com/p/palm-efficiently-training-massive](https://cameronrwolfe.substack.com/p/palm-efficiently-training-massive)               | PaLM's 46.2% MFU (57.8% HFU) on 6144 TPUv4 chips                   |


### Inference (for reference)


| Title                                     | URL                                                                                                    | What It Covers                                                    |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| [x] Transformer Inference Arithmetic (kipply) | [https://kipp.ly/transformer-inference-arithmetic/](https://kipp.ly/transformer-inference-arithmetic/) | FLOPs vs memory boundedness, KV cache costs, arithmetic intensity |


---

## Academic Papers

### Foundational Formulas


| Paper                                                                       | URL                                                                  | Why It Matters                                                              |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| [x] Scaling Laws for Neural Language Models (Kaplan et al., 2020)               | [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361) | Power-law scaling relationships between loss, model size, data, and compute |
| [x] Training Compute-Optimal LLMs / Chinchilla (Hoffmann et al., 2022)          | [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556) | D_optimal ~ 20N; trained 400+ models to fit scaling laws                    |
| [ ] ZeRO: Memory Optimizations (Rajbhandari et al., 2019)                       | [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054) | Memory partitioning formulas for optimizer states, gradients, parameters    |
| [x] Efficient Large-Scale LM Training with Megatron-LM (Narayanan et al., 2021) | [https://arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473) | Activation memory formulas with TP/PP/DP; trains 1T parameter model         |
| [x] Megatron-LM: Training with Model Parallelism (Shoeybi et al., 2019)         | [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053) | Original intra-layer tensor parallelism for transformers                    |
| [x] Reducing Activation Recomputation (Korthikanti et al., 2022)                | [https://arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198) | Activation memory 5x reduction; sequence parallelism + selective recompute  |
| [x] Mixed Precision Training (Micikevicius et al., 2018)                        | [https://arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740) | FP16+FP32 training memory model (master weights in FP32, compute in FP16)   |
| [x] An Empirical Model of Large-Batch Training (McCandlish et al., 2018)        | [https://arxiv.org/abs/1812.06162](https://arxiv.org/abs/1812.06162) | Gradient noise scale for predicting optimal batch size                      |
| [x] PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)      | [https://arxiv.org/abs/2204.02311](https://arxiv.org/abs/2204.02311) | 540B model; reports MFU of 46.2% and HFU of 57.8%                           |
| [ ] FlashAttention (Dao et al., 2022)                                           | [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135) | O(n^2) to O(n) attention memory; 10-20x savings at long sequences           |
| [ ] QLoRA (Dettmers et al., 2023)                                               | [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314) | 65B fine-tuning on single 48GB GPU using 4-bit quantization + LoRA          |


### Memory Estimation Research


| Paper                                                                    | URL                                                                  | Why It Matters                                                                |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [x] LLMem: Estimating GPU Memory for Fine-Tuning (Kim et al., 2024)          | [https://arxiv.org/abs/2404.10933](https://arxiv.org/abs/2404.10933) | Shows existing methods underestimate peak memory; 1.6% error rate             |
| [x] Understanding Performance and Cost of LLM Fine-Tuning (Xia et al., 2024) | [https://arxiv.org/abs/2408.04693](https://arxiv.org/abs/2408.04693) | Analytical model for fine-tuning throughput and cost including MoE            |
| [x] Comprehensive Performance Modeling for Foundation Models (2024)          | [https://arxiv.org/abs/2410.0273](https://arxiv.org/abs/2410.0273)   | Systematic FLOPs, memory accesses, and communication for every transformer op |
| [x] Memory Analysis of DeepSeek Model Training (Zhang & Su, 2025)            | [https://arxiv.org/abs/2502.07846](https://arxiv.org/abs/2502.07846) | Detailed memory breakdown for DeepSeek-v2/v3 with 3D parallelism              |
| [x] Scaling Data-Constrained Language Models (Muennighoff et al., 2023)      | [https://arxiv.org/abs/2305.16264](https://arxiv.org/abs/2305.16264) | Scaling laws accounting for data repetition                                   |


### Surveys


| Paper                                                              | URL                                                                  | Why It Matters                                                              |
| ------------------------------------------------------------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| [x] Survey on Memory-Efficient Transformer Training (2025)             | [https://arxiv.org/abs/2501.11847](https://arxiv.org/abs/2501.11847) | Systematic review of memory-saving techniques at algorithm/system/HW levels |
| [x] Survey on Efficient Training of Transformers (Zhuang et al., 2023) | [https://arxiv.org/abs/2302.01107](https://arxiv.org/abs/2302.01107) | First systematic overview of efficient transformer training (IJCAI 2023)    |


---

## Books & Comprehensive References


| Name                                               | URL                                                                                                                                        | What It Covers                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| [ ] JAX/Google DeepMind Scaling Book                   | [https://jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/)                                                           | Full book: TPU/GPU hardware, transformer math, parallelism, efficient training  |
| [ ] ML Engineering Open Book (Stas Bekman)             | [https://github.com/stas00/ml-engineering](https://github.com/stas00/ml-engineering)                                                       | From BLOOM-176B experience: GPU memory anatomy, performance tuning, parallelism |
| [ ] PyTorch Training Performance Guide (ResidentMario) | [https://residentmario.github.io/pytorch-training-performance-guide/](https://residentmario.github.io/pytorch-training-performance-guide/) | Mixed precision, checkpointing, distributed training in PyTorch                 |


---

## Training Reports & Retrospectives


| Name                                      | URL                                                                                                                                                                                                                                                    | What It Covers                                            |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| [x] BLOOM-176B Training Logs (BigScience)     | [https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml)                                                                                   | Full training scripts, SLURM configs, engineering lessons |
| [x] The Llama 3 Herd of Models (Meta)         | [https://ai.meta.com/blog/meta-llama-3/](https://ai.meta.com/blog/meta-llama-3/)                                                                                                                                                                       | Training details including 4D parallelism on 16K H100s    |
| [x] PaLM: Scaling to 540B Parameters (Google) | [https://research.google/blog/pathways-language-model-palm-scaling-to-540-billion-parameters-for-breakthrough-performance/](https://research.google/blog/pathways-language-model-palm-scaling-to-540-billion-parameters-for-breakthrough-performance/) | 6144 TPUv4 chips, 46.2% MFU (57.8% HFU)                  |
| [x] FlashAttention-3 (Tri Dao)                | [https://tridao.me/blog/2024/flash3/](https://tridao.me/blog/2024/flash3/)                                                                                                                                                                             | 75% H100 utilization (740 TFLOP/s)                        |


---

## Podcasts & Talks


| Name                                                    | URL                                                                                          | What It Covers                                                    |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [ ] The Mathematics of Training LLMs (Latent Space Podcast) | [https://www.latent.space/p/transformers-math](https://www.latent.space/p/transformers-math) | Q. Anthony on 6PD formula, memory breakdown, distributed training |

