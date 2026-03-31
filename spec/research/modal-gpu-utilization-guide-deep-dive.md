# Deep Dive: "I Paid for the Whole GPU" -- Modal Blog GPU Utilization Guide

**Source**: https://modal.com/blog/gpu-utilization-guide
**Authors**: Modal Team (Charles Frye et al.)
**Type**: Technical blog post
**Reviewed**: 2026-03-31

---

## 1. Executive Summary

This blog proposes a **three-level framework** for decomposing GPU utilization into allocation, kernel, and arithmetic levels. The key thesis: "GPU utilization" is ambiguous because it conflates three distinct metrics that measure fundamentally different things. The framework provides a conceptual taxonomy for understanding where throughput is lost, with each level having different measurement tools, optimization strategies, and typical efficiency ranges.

**Relevance to the calculator spec**: The spec currently treats MFU as a single opaque knob (Section 6.3). This blog's decomposition could enrich the spec's explanation of *why* MFU falls in the 30-55% range and provide structure for the "sources of throughput loss" cascade already in Section 6.3. However, Level 1 (allocation utilization) is an infrastructure/billing concern outside the calculator's scope, and Level 2 (kernel utilization) is largely subsumed by the existing throughput degradation analysis.

---

## 2. The Three-Level GPU Utilization Framework

### Level 1: GPU Allocation Utilization

**Formula**:
```
GPU Allocation Utilization = GPU-seconds running application code / GPU-seconds paid for
```

**What it measures**: The fraction of *purchased* GPU time during which application code is actually running. This is a billing/infrastructure metric, not a hardware performance metric.

**Typical values**:
- Industry majority: < 70% at peak demand
- Banana serverless GPU platform: ~20% aggregate
- Modal (claimed): > 90% aggregate

**Sources of loss at this level**:
- GPU purchasing/commissioning/decommissioning latency (physical hardware constraints)
- Cloud pricing models requiring multi-month or multi-year commitments
- OS configuration, health checks, code/data transfer between allocation and execution
- Demand variability -- provisioned capacity must exceed peak demand, creating idle time during troughs

**Relevance to calculator**: **Low**. This is an infrastructure cost efficiency metric. The calculator's training time formula (T = C / (N_gpu * F_peak * MFU)) assumes GPUs are already allocated and running. Allocation utilization would only matter for a *cost* calculator that accounts for idle GPU time in cloud billing, which is outside the current spec scope.

### Level 2: GPU Kernel Utilization

**Formula**:
```
GPU Kernel Utilization = GPU-seconds running kernels / GPU-seconds paid for
```

**What it measures**: The fraction of time the GPU is executing *any* kernel (including memory copies, not just compute kernels). This is what `nvidia-smi` reports as "GPU utilization" via NVML.

**Critical insight**: This metric "does not care whether the code we're running on the GPU is exercising the hardware's actual capacity." A kernel that performs a single addition on one thread counts the same as a kernel saturating all SMs with tensor core operations.

**Sources of loss at this level** (gaps between kernels):
- **Host overhead**: CPU cannot feed work to GPU fast enough. At millisecond-scale steps, Python becomes a bottleneck. Even at microsecond scale, CUDA C++ API scheduling latency matters.
- **Non-GPU supporting work**: Network/disk I/O for inputs/outputs, downloading model weights, writing logs.
- **Slow host operations blocking GPU work**: Python logging or other slow operations on the host blocking the critical path for kernel launches.

**Measurement tools**:
- `nvidia-smi` (wraps NVML) -- reports this as "GPU utilization"
- PyTorch Profiler -- produces traces where "periods where no kernels are running appear as empty strips in the timelines of CUDA streams"

**Optimization strategies**:
- **CUDA Graphs**: "Convert a sequence of kernel launches into a DAG that only needs to be launched once" -- eliminates per-kernel launch overhead for repeated execution patterns.
- **Request batching**: Aggregate more work per kernel launch to amortize host overhead.
- **Asynchronous host work**: Prevent slow host operations from blocking GPU-bound work.

**Relevance to calculator**: **Medium**. The spec already captures kernel launch overhead as item #3 in the throughput degradation cascade (Section 6.3: "Framework and kernel launch overhead, down to 20-40% of peak"). The blog adds specificity about *why* this happens (host-side bottlenecks, Python overhead) and *how* to fix it (CUDA Graphs, batching), which enriches the qualitative explanation but does not change any formulas.

### Level 3: Arithmetic (Model FLOP/s) Utilization

**Formula**:
```
Model FLOP/s Utilization (MFU) = Model FLOP/s throughput achieved / FLOP/s bandwidth paid for
```

**What it measures**: The ratio of useful compute throughput to peak theoretical compute throughput. This is the metric the spec already uses in Section 6.2.

**Typical values** (from the blog):
- Raw matrix multiplications: 70-80% MFU
- Meta LLaMA 3 405B training: 38-41% MFU
- DeepSeek-v3 training: ~20-30% MFU (no official number)

**Sources of loss at this level**:
- **Low arithmetic intensity**: For memory-bandwidth-bound workloads, FLOP/s throughput is limited by how fast data can be moved, not how fast it can be computed. "Memory bandwidth is generally many times lower than the device's FLOP/s bandwidth, especially in recent generations."
- **Non-matmul operations**: Operations like softmax, LayerNorm, activation functions are memory-bandwidth-bound.
- **Inter-node communication**: "Much of the shortfall is due to the need for inter-node communication in large training jobs."

**Arithmetic intensity concept** (key definition from the blog):
```
Arithmetic Intensity = Algorithm's FLOP/s throughput / Algorithm's byte/s throughput
```

The blog notes that foundation model inference has "low" arithmetic intensity, "perhaps a few FLOPs per byte." Two strategies to improve it:
1. **Algorithmic rewrites** -- e.g., online softmax in Flash Attention increases arithmetic intensity by restructuring computation to perform more FLOPs per byte moved.
2. **Batching** -- increases FLOPs more than memory bytes for most neural network workloads, but adds per-task latency.

**Measurement tools**:
- NVIDIA DCGM (`dcgm`) with metrics:
  - `DCGM_FI_PROF_DRAM_ACTIVE` -- DRAM-to-SRAM memory bandwidth utilization
  - `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` -- Tensor Core utilization
- Blog notes: "Measuring Model FLOP/s Utilization is tricky" -- often requires pen-and-paper analysis.

**Relevance to calculator**: **High** -- this maps directly to the spec's MFU (Section 6.2). The DCGM metric names and the arithmetic intensity concept are useful additions.

---

## 3. How the Three Levels Compose

The blog establishes a **hierarchical dependency** but does NOT provide an explicit multiplicative decomposition (i.e., it does NOT claim `overall_utilization = L1 * L2 * L3`).

The stated relationship is one-directional implication:
> "Instances that aren't running application code or that aren't running GPU kernels cannot achieve a high MFU, so low GPU Allocation Utilization or low GPU Kernel Utilization imply low Model FLOP/s Utilization."

This is a **necessary condition** chain:
```
High MFU requires High Kernel Utilization requires High Allocation Utilization
```

But it is NOT a sufficient condition chain -- high allocation and kernel utilization do not guarantee high MFU (the GPU could be running inefficient kernels at high duty cycle).

**Why there is no multiplicative decomposition**: The three metrics use different denominators and measurement approaches. Allocation utilization is a time-based billing metric. Kernel utilization is a time-based hardware metric. MFU is a FLOP/s-based performance metric. They don't naturally multiply to give a single overall number. The blog is providing a *diagnostic taxonomy*, not a compositional formula.

---

## 4. CPU Benchmark Analogy (One Billion Row Challenge)

The blog uses a CPU benchmark to illustrate why MFU is unintuitive even for optimized code:

**Setup**: Process 1 billion rows with 3 FLOPs each = 3 billion total FLOPs. Hardware: AMD EPYC 7502P (32 cores, 3.35 GHz, 256-bit AVX2 SIMD). Leading result: ~1 second.

**Naive MFU calculation** (1 FLOP/cycle):
```
Peak = 8 cores * 3.35 GHz = 26.8 GFLOP/s
MFU = 3 GFLOP/s / 26.8 GFLOP/s = ~10%
```

**Correct MFU calculation** (accounting for AVX2 SIMD, 16 FLOPs/core/cycle for FP32):
```
Peak = 8 cores * 3.35 GHz * 16 FLOP/cycle = ~428 GFLOP/s
MFU = 3 GFLOP/s / 428 GFLOP/s = under 1%
```

**Lesson**: Even highly optimized code achieves < 1% of theoretical hardware capability when the workload has low arithmetic intensity. This parallels why GPU MFU for training is 30-55% -- the gap is structural, not just a sign of poor optimization.

**Relevance to calculator**: This is a pedagogical example, not a formula. It could be useful for documentation/tooltips explaining to users why MFU of 40% is actually good.

---

## 5. Quantitative Data Compilation

### MFU Reference Points (from blog)
| Workload | MFU | Notes |
|---|---|---|
| Raw matrix multiplication on GPU | 70-80% | Theoretical ceiling for real workloads |
| Meta LLaMA 3 405B training | 38-41% | State-of-the-art large-scale training |
| DeepSeek-v3 training | ~20-30% | Estimated, no official number |
| CPU 1BRC benchmark (AVX2 basis) | < 1% | Memory-bound workload |

### Allocation Utilization Reference Points
| Organization Type | Allocation Util | Notes |
|---|---|---|
| Industry majority at peak demand | < 70% | From "State of AI Infrastructure 2024" report |
| Banana (serverless GPUs) | ~20% | Now sunset |
| Modal (claimed) | > 90% | Aggregate across all users |

### Key Observation on QoS
The blog notes that GPU applications show "much less variability" than transactional databases, permitting higher utilization (90%+) without quality-of-service degradation. This is because GPU workloads (especially training) have predictable, steady-state demand patterns.

---

## 6. Tools and Metrics Reference

### NVIDIA Monitoring Stack
| Tool | Metric Level | What It Reports |
|---|---|---|
| `nvidia-smi` (NVML) | Kernel Utilization | Fraction of time *any* kernel is running on the GPU |
| `dcgm` `DCGM_FI_PROF_DRAM_ACTIVE` | Arithmetic | DRAM-to-SRAM memory bandwidth utilization |
| `dcgm` `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Arithmetic | Tensor Core utilization (pipeline active fraction) |
| `dcgm` `DCGM_FI_PROF_*` metrics | Various | All profiling metrics prefixed with `DCGM_FI_PROF` are relevant |
| PyTorch Profiler | Kernel Utilization | Application traces showing idle CUDA streams |

### Critical Warning About nvidia-smi
The blog explicitly warns that `nvidia-smi` GPU utilization is NOT MFU and is NOT arithmetic utilization. It measures kernel occupancy time, which can be 100% even when the GPU is running trivial kernels that use a fraction of the hardware's FLOP/s capability. This is a common misunderstanding.

---

## 7. Key Optimization Techniques Mentioned

### For Kernel Utilization
- **CUDA Graphs**: Convert sequential kernel launches into a DAG launched once. Eliminates per-kernel launch overhead. Referenced PyTorch blog: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- **Request batching**: Aggregate GPU work to amortize host overhead.
- **Async host operations**: Prevent slow host-side work from blocking GPU kernel launches.

### For Arithmetic Utilization
- **Flash Attention**: Algorithmic rewrite using online softmax that increases arithmetic intensity -- performs more FLOPs per byte of memory moved. Directly cited: https://arxiv.org/abs/2205.14135
- **Batching for inference**: Increases FLOPs more than memory bytes for neural network workloads, improving arithmetic intensity at the cost of latency.
- **High-quality open-source kernels**: Using CuBLAS, PyTorch, vLLM rather than writing custom kernels.

### For Detecting Deadlocks
- **Power/heat monitoring**: Communication kernels on GPUs are "subject to faults" that "frequently manifest as deadlock." Blog recommends monitoring GPU power draw and heat to detect these, since a deadlocked GPU will show abnormally low power consumption.

---

## 8. What Is Novel / Non-Obvious

### 8.1 The three-level taxonomy itself
While individual levels are well-known, the explicit framing as three distinct meanings of "GPU utilization" -- each with different measurement tools, optimization strategies, and typical ranges -- is a useful pedagogical contribution. The spec's Section 6.3 already has the throughput degradation cascade, but it doesn't explicitly separate "are we running kernels?" from "are those kernels efficient?" in the same structured way.

### 8.2 nvidia-smi as misleading metric
The explicit call-out that `nvidia-smi` reports kernel utilization (not MFU) is important. Many practitioners equate nvidia-smi "100% GPU utilization" with "I'm getting full performance." The spec could include a warning about this in a tooltip or documentation note.

### 8.3 DCGM metric names
The specific `DCGM_FI_PROF_DRAM_ACTIVE` and `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` metric names are useful for users who want to profile their actual training runs. These are concrete, actionable references not currently in the spec.

### 8.4 Arithmetic intensity as the bridge between memory and compute bounds
The definition `Arithmetic Intensity = FLOP/s throughput / byte/s throughput` and the observation that modern GPUs have a much higher compute-to-bandwidth ratio than most workloads can exploit is the roofline model concept (though the blog doesn't use that term). The spec mentions arithmetic intensity in two places (Section 6.3 for small micro-batch MFU degradation, and Section 10 for RL generation being memory-bandwidth-bound) but doesn't define it or explain it as a general concept.

### 8.5 QoS argument for high allocation utilization
The observation that GPU workloads tolerate high utilization without QoS degradation (unlike databases) is relevant for cost modeling -- it means users should target near-100% GPU allocation for training runs, unlike traditional server provisioning where you leave headroom.

---

## 9. Relationship to Existing Spec

### What the spec already covers well
- MFU formula and definition (Section 6.2) -- matches the blog's Level 3
- Sources of throughput loss cascade (Section 6.3) -- covers tensor core gap, non-matmul ops, framework overhead, communication, I/O
- MFU reference points for LLaMA 3 405B (Section 6.3) -- matches the blog's data
- Small micro-batch MFU degradation (Section 6.3) -- addresses arithmetic intensity implicitly
- Warning about MFU as single comprehensive efficiency knob (Section 6.3) -- aligns with blog's point about not double-counting

### What the blog adds that the spec could incorporate

**Minor additions (enriching existing content)**:
1. **nvidia-smi warning**: Add a note in Section 6.3 or as a tooltip: "nvidia-smi reports kernel utilization (fraction of time any kernel is running), NOT MFU. A GPU showing 100% nvidia-smi utilization may still have very low MFU if kernels are inefficient or memory-bound."
2. **DCGM metric names**: Add to Section 6.3 or Section 7 as a profiling reference: users can check `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` for tensor core utilization and `DCGM_FI_PROF_DRAM_ACTIVE` for memory bandwidth utilization.
3. **Arithmetic intensity definition**: Add a brief definition in Section 6.3 near the small micro-batch warning: "Arithmetic intensity (FLOPs per byte of memory moved) determines whether a workload is compute-bound or memory-bound. Modern GPUs need ~150-300 FLOPs/byte to saturate compute; when micro-batch size is small, matmuls drop below this threshold." (The spec already has this number in Section 10 for RL generation.)
4. **DeepSeek-v3 MFU data point**: Add to the MFU reference table: "DeepSeek-v3: ~20-30% MFU (estimated, no official number)" as a contrast showing that not all large models achieve the 35-45% range.
5. **Raw matmul MFU ceiling**: Add "70-80%" as the empirical ceiling even for pure matmul workloads, which provides context for why training MFU of 40% is actually reasonable (it's ~50-57% of the matmul ceiling).

**Not recommended for inclusion**:
- Level 1 (allocation utilization): Out of scope for the calculator. This is an infrastructure/billing concern.
- Level 2 (kernel utilization) as a separate formula: The spec already captures this as part of the throughput degradation cascade. Adding it as a separate metric would add complexity without enabling any new calculations.
- The three-level framework as a formal decomposition: The spec's single-MFU approach with qualitative breakdown is sufficient and avoids false precision. The blog itself doesn't provide a multiplicative composition formula.
- The CPU 1BRC analogy: Interesting for pedagogy but not needed in the spec.

---

## 10. References Cited by the Blog

All external references from the post:

| Reference | URL | Relevance |
|---|---|---|
| Modal GPU Glossary | https://modal.com/gpu-glossary | Definitions reference |
| Flash Attention paper | https://arxiv.org/abs/2205.14135 | Arithmetic intensity improvement |
| Horace He PyTorch talk | https://www.youtube.com/watch?v=139UPjoq7Kw&t=1236s | GPU performance optimization |
| Abhinav Upadhyay "GPU Computing" | https://blog.codingconfessions.com/p/gpu-computing | GPU computing fundamentals |
| Stas Bekman ML Engineering Open Book | https://github.com/stas00/ml-engineering/ | Comprehensive ML engineering |
| Stas Bekman MFU vs HFU guide | https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#mfu-vs-hfu | MFU definition reference |
| Si Boehm CUDA matrix multiplication | https://siboehm.com/articles/22/CUDA-MMM | Kernel optimization worklog |
| Pranjal Shankhdhar CUBLAS on H100 | https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog | Tensor core optimization |
| NVIDIA DCGM documentation | https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#profiling-metrics | Profiling metrics reference |
| LLaMA 3 paper | https://arxiv.org/abs/2407.21783 | MFU data source |
| One Billion Row Challenge | https://github.com/gunnarmorling/1brc | CPU benchmark |
| CUDA Graphs PyTorch blog | https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ | Kernel launch optimization |
| DeepSeek-v3 analysis (SemiAnalysis) | https://semianalysis.com/2025/01/31/deepseek-debates/ | MFU estimate source |
| State of AI Infrastructure 2024 | https://ai-infrastructure.org/the-state-of-ai-infrastructure-at-scale-2024/ | Industry utilization data |

---

## 11. Verdict: Spec Changes Needed

**Overall assessment**: This blog is primarily a **pedagogical taxonomy** rather than a source of new formulas or calculations. Its main contribution is the three-level framework for understanding GPU utilization, which provides good conceptual scaffolding but does not change the calculator's computational model.

**Recommended spec additions** (all minor, enriching existing Section 6.3):

1. Add a brief note distinguishing `nvidia-smi` kernel utilization from MFU, warning users not to conflate them.
2. Add DCGM metric names (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE`) as profiling references.
3. Add a brief definition of arithmetic intensity near the small micro-batch warning.
4. Add DeepSeek-v3 ~20-30% MFU as an additional data point in the MFU reference table.
5. Add the 70-80% raw matmul MFU ceiling as context for the throughput loss cascade.

**No new formulas, sections, or calculator features are needed.** The spec's existing single-MFU approach with the qualitative throughput loss breakdown is the right level of abstraction for a calculator. The three-level framework is useful background knowledge but does not translate into additional calculator inputs or outputs.
