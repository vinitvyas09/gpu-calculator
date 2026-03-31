# Deep Dive: Scaling Laws for Neural Language Models (Kaplan et al., 2020)

**Paper**: arXiv:2001.08361
**Authors**: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (OpenAI)
**Date**: January 2020
**Local implementation reference**: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/scaling_laws_repo/kaplan_scaling_laws.ipynb`

---

## 1. Summary of Relevance to Our Calculator

This paper establishes power-law relationships between language model loss and three variables: model parameters (N), dataset tokens (D), and compute (C). While the Chinchilla paper (2022) superseded Kaplan's compute-optimal allocation recommendations, several Kaplan results remain the canonical reference in the field:

1. The **6ND FLOPs approximation** (already in our spec at Section 4.1)
2. The **critical batch size formula** B_crit(L) (already in our spec at Section 4.4)
3. The **L(N,D) parametric loss function** (superseded by Chinchilla's formulation in our spec)
4. The **C_min vs C relationship** (useful for adjusting compute estimates based on batch size choices)
5. The **L(N,S) formula** for predicting loss from partial training (not yet in our spec)
6. The **N_opt, B_opt, S_min power-law relationships** with compute (partially in our spec)

---

## 2. All Scaling Law Formulas

### 2.1 Single-Variable Power Laws

**Loss vs. parameters (fixed large D):**
```
L(N) = (N_c / N)^alpha_N
```

**Loss vs. dataset size (fixed large N):**
```
L(D) = (D_c / D)^alpha_D
```

**Loss vs. compute (fixed batch size, naive allocation):**
```
L(C) = (C_c / C)^alpha_C
```

**Loss vs. minimum compute (compute-efficient training, B << B_crit):**
```
L(C_min) = (C_c_min / C_min)^alpha_C_min
```

### 2.2 Multi-Variable Loss Functions

**Loss as function of N and D (the "combined" scaling law):**
```
L(N, D) = [ (N_c / N)^(alpha_N / alpha_D) + D_c / D ]^alpha_D
```

This is a key formula. It captures the interplay between model size and data size, including overfitting behavior. When D is much larger than needed, the D_c/D term vanishes and we recover L(N). When N is much larger than needed, the (N_c/N) term vanishes and we recover L(D).

**Loss as function of N and minimum training steps (for B >> B_crit):**
```
L(N, S_min) = (N_c / N)^alpha_N + (S_c / S_min)^alpha_S
```

This additive form (not multiplicative) means that loss is bounded below by both the model size limit and the training duration limit. The additive structure is important: you cannot compensate for a too-small model by training longer, or vice versa.

### 2.3 Critical Batch Size

**Definition**: The batch size at which training is equally efficient in compute and time.

**Derivation from the compute-time trade-off curve:**
```
(S / S_min - 1) * (E / E_min - 1) = 1
```

Where:
- S = actual training steps at batch size B
- E = total training examples processed (E = B * S)
- S_min = minimum steps (time-efficient limit, B -> infinity)
- E_min = minimum examples (compute-efficient limit, B -> 0)

**Critical batch size:**
```
B_crit = E_min / S_min
```

At B = B_crit, training requires 2 * E_min examples and 2 * S_min steps (a 2x overhead on both compute and time relative to their respective minima).

**B_crit as function of loss:**
```
B_crit(L) = B_star / L^(1 / alpha_B)
```

### 2.4 Compute Adjustment Formulas

**Minimum compute (given actual compute at batch size B):**
```
C_min = C / (1 + B / B_crit(L))
```

Where C = 6 * N * B * S is the actual compute used. This formula converts from "actual FLOPs used at a particular batch size" to "the minimum FLOPs that would have been needed if training at B << B_crit." It is useful for normalizing compute across runs with different batch sizes.

**Minimum serial steps (given actual steps at batch size B):**
```
S_min = S / (1 + B_crit(L) / B)
```

This is the dual adjustment: it converts from actual steps to the minimum steps that would have been needed if training at B >> B_crit.

### 2.5 Optimal Allocation Power Laws

**Optimal model size as function of compute (naive, fixed batch size):**
```
N_opt(C) = N_e * C^p_N
```

**Optimal model size as function of minimum compute:**
```
N_opt(C_min) = N_e_min * C_min^p_N_min
```

**Batch size and steps scaling with minimum compute:**
```
B_opt ~ C_min^(alpha_C_min / alpha_B)
S_min ~ C_min^(alpha_C_min / alpha_S)
D_opt ~ C_min^(alpha_C_min / alpha_D)  [implied]
```

### 2.6 Overfitting Quantification

**Excess loss from insufficient data:**
```
delta_L ≈ [ 1 + (N / N_c)^(alpha_N / alpha_D) * (D_c / D) ]^alpha_D - 1
```

**Minimum data to keep overfitting penalty below ~0.02:**
```
D >= 5 × 10^3 × N^0.74
```

This is derived from L(N,D) - L(N,infinity). It says that data requirements grow sub-linearly with model size.

### 2.7 Early Stopping Lower Bound

**Minimum steps before early stopping to reach L(N,D):**
```
S_stop(N, D) >= S_c / [ L(N, D) - L(N, infinity) ]^(1 / alpha_S)
```

Where L(N, infinity) is the loss at convergence (infinite data).

### 2.8 FLOPs Counting

**Total training compute:**
```
C = 6 * N * E  (where E = total training examples = B * S)
```

Equivalently, C = 6 * N * D where D = total tokens processed.

**Forward pass FLOPs per token (detailed breakdown):**
```
C_forward ≈ 2N + 2 * n_layer * n_ctx * d_attn
```

Where:
- N = non-embedding parameter count
- n_layer = number of layers
- n_ctx = context length (sequence length)
- d_attn = attention dimension per head * number of heads

The first term (2N) covers all matrix multiplications in the model. The second term (2 * n_layer * n_ctx * d_attn) covers the attention score computation (Q*K^T) and the value reduction (scores * V). This is the same as the PaLM formula (already in our spec) written in Kaplan's notation.

**Backward pass is ~2x forward**, giving total per-token FLOPs = 6N + 6 * n_layer * n_ctx * d_attn, which matches the PaLM formula 6N + 12Lds when you account for the 2x in the detailed per-operation counting.

**Non-embedding parameter count (for standard GPT):**
```
N ≈ 2 * d_model * n_layer * (2 * d_attn + d_ff)
```

Where d_ff is typically 4 * d_model, giving:
```
N ≈ 12 * n_layer * d_model^2
```

This is the same quick estimate already in our spec (Section 3.2).

**PF-day conversion:**
```
1 PF-day = 8.64 × 10^19 FLOPs
```

(= 10^15 FLOPS × 86400 seconds/day)

---

## 3. Complete Table of Fitted Coefficients

### 3.1 Kaplan et al. Original Values

| Parameter | Symbol | Value | Units | Context |
|-----------|--------|-------|-------|---------|
| Loss vs N exponent | alpha_N | 0.076 | dimensionless | From L(N) fit |
| Loss vs N scale | N_c | 8.8 x 10^13 | non-embedding params | From L(N) fit |
| Loss vs D exponent | alpha_D | 0.095 | dimensionless | From L(D) fit |
| Loss vs D scale | D_c | 5.4 x 10^13 | tokens | From L(D) fit |
| Loss vs C exponent (naive) | alpha_C | 0.057 | dimensionless | Fixed batch size |
| Loss vs C scale (naive) | C_c | 1.6 x 10^7 | PF-days | Fixed batch size |
| Loss vs C_min exponent | alpha_C_min | 0.050 | dimensionless | Compute-efficient |
| Loss vs C_min scale | C_c_min | 3.1 x 10^8 | PF-days | Compute-efficient |
| Loss vs steps exponent | alpha_S | 0.76 | dimensionless | From L(N,S) fit |
| Steps scale | S_c | 2.1 x 10^3 | steps | From L(N,S) fit |
| Critical batch size scale | B_star | 2.0 x 10^8 | tokens | From B_crit(L) fit |
| Critical batch size exponent | alpha_B | 0.21 | dimensionless | From B_crit(L) fit |
| N_opt vs C coefficient | N_e | 1.6 x 10^9 | - | Naive allocation |
| N_opt vs C exponent | p_N | 0.88 | - | Naive allocation |
| N_opt vs C_min coefficient | N_e_min | 1.3 x 10^9 | - | Compute-efficient |
| N_opt vs C_min exponent | p_N_min | 0.73 | - | Compute-efficient |

### 3.2 L(N,D) Joint Fit Coefficients (different from single-variable fits)

| Parameter | L(N) fit | L(D) fit | L(N,D) joint fit |
|-----------|----------|----------|-------------------|
| alpha_N | 0.076 | - | 0.076 |
| alpha_D | - | 0.095 | 0.103 |
| N_c | 8.8 x 10^13 | - | 6.4 x 10^13 |
| D_c | - | 5.4 x 10^13 | 1.8 x 10^13 |

Note: The joint L(N,D) fit gives different coefficients than the individual L(N) and L(D) fits, particularly for D_c and alpha_D. The paper notes this is expected and that the joint fit better captures the interaction between model size and data.

### 3.3 L(N,S) Fit Coefficients

| Parameter | Value | Context |
|-----------|-------|---------|
| N_c | 6.5 x 10^13 | From L(N,S) fit (slightly different from L(N) fit) |
| alpha_N | 0.077 | From L(N,S) fit |
| S_c | 2.1 x 10^3 | Minimum steps scale |
| alpha_S | 0.76 | Steps exponent |

### 3.4 Compute-Optimal Scaling Exponents

| Quantity | Scales as | Exponent | Interpretation |
|----------|-----------|----------|----------------|
| N_opt | C_min^0.73 | 0.73 | Most compute goes to model size |
| B_opt | C_min^0.24 | 0.24 | Some goes to batch size |
| S_min | C_min^0.03 | 0.03 | Almost none goes to training steps |
| D_opt | C_min^0.27 | 0.27 | = B_opt exponent (D = B * S) |

**Derived relationship** (verified algebraically):
```
alpha_C_min = 1 / (1/alpha_S + 1/alpha_B + 1/alpha_N) ≈ 0.050
```

This confirms internal consistency of the fitted parameters.

**Key practical implication**: When compute increases 100x:
- N_opt increases ~29x (100^0.73)
- B_opt increases ~3x (100^0.24)
- S_min increases ~1.07x (100^0.03) -- essentially unchanged
- D_opt increases ~3.5x (100^0.27)

This is the basis for Kaplan's recommendation to "make models bigger, not train longer." Chinchilla later showed this was wrong because Kaplan's training runs did not adequately explore the data axis.

---

## 4. Independent Reproduction Results (from local scaling_laws_repo)

The local repository reproduces Kaplan's results at a smaller scale (10^5 to 10^7 parameters instead of 10^4 to 10^9). The fact that the power-law exponents approximately match despite ~100x smaller models provides confidence in the robustness of these relationships.

| Parameter | Kaplan et al. | Reproduction | Match? |
|-----------|--------------|--------------|--------|
| alpha_N (L(N)) | 0.076 | 0.082 | Close |
| alpha_N (L(N,D)) | 0.076 | 0.076 | Exact |
| alpha_D (L(N,D)) | 0.103 | 0.122 | Moderate |
| alpha_C_min | 0.050 | 0.056 | Close |
| p_N_min (N_opt ~ C^p) | 0.73 | 0.72 | Very close |
| alpha_B | 0.21 | 0.23 | Close |
| B_opt exponent | 0.24 | 0.24 | Exact |
| S_min exponent | 0.03 | 0.04 | Close |

---

## 5. What is Already in Our Spec vs. What is New

### Already in the spec:
- 6ND FLOPs approximation (Section 4.1) -- complete and well-documented
- PaLM per-token formula with quadratic attention correction (Section 4.1) -- complete
- Critical batch size B_crit(L) formula and coefficients (Section 4.4) -- complete
- Chinchilla L(N,D) superseding Kaplan (Section 4.3) -- complete
- Historical context of Kaplan vs Chinchilla (Section 4.3) -- complete
- 12Ld_model s formula and note about 6LCT variant (Section 4.1) -- complete
- N ≈ 12 * L * d^2 quick estimate (Section 3.2) -- complete

### NOT yet in the spec (potential additions):

#### 5.1 L(N, S_min) Formula for Predicting Loss from Partial Training
```
L(N, S_min) = (N_c / N)^alpha_N + (S_c / S_min)^alpha_S
```
With alpha_S = 0.76, S_c = 2.1 x 10^3.

**Why it matters for our calculator**: This formula allows predicting the loss at any point during training given the model size and how many steps have been completed. The calculator could show an "expected loss trajectory" curve for a given (N, total_steps) configuration, which would help users understand:
- How much loss improvement remains (diminishing returns curve)
- Whether the model has approximately converged
- The cost of early stopping vs. training to completion

The additive form means L(N, S) > (N_c/N)^alpha_N always, so there is a hard floor set by model size that no amount of training can overcome.

#### 5.2 C_min Adjustment Formula
```
C_min = C / (1 + B / B_crit(L))
```

**Why it matters**: When users specify a batch size that is above the critical batch size, this formula quantifies how much extra compute they are "wasting." The calculator could show:
- "Your batch size of X is Y% above B_crit, meaning ~Z% of your compute budget is above the efficient frontier"
- The formula C_min = C / (1 + B/B_crit) gives the compute that would have been sufficient if training at B << B_crit

**Dual formula for time:**
```
S_min = S / (1 + B_crit(L) / B)
```

And conversely, when B < B_crit:
- "Your batch size is below B_crit, meaning you're spending more wall-clock steps than necessary"

#### 5.3 Overfitting Data Requirement
```
D >= 5 x 10^3 * N^0.74
```

**Why it matters**: This is a Kaplan-specific data floor. While our spec uses Chinchilla's D_optimal ≈ 20N as the main recommendation, this formula gives a **minimum** data requirement to avoid overfitting -- a distinct and lower threshold. For example:
- N = 1B: D >= 5e3 * (1e9)^0.74 ≈ 3.5B tokens (vs. 20B for Chinchilla-optimal)
- N = 7B: D >= 5e3 * (7e9)^0.74 ≈ 16B tokens (vs. 140B for Chinchilla-optimal)

This could be useful as a "minimum viable training data" warning.

#### 5.4 Early Stopping Lower Bound
```
S_stop(N, D) >= S_c / [ L(N, D) - L(N, infinity) ]^(1/alpha_S)
```

This gives a lower bound on the number of training steps needed to reach the loss predicted by L(N,D). It could be useful for sanity-checking user-specified training step counts.

#### 5.5 Scaling of B_opt and S_min with Compute

The exponents showing how batch size and training steps scale with compute:
```
B_opt ~ C_min^0.24
S_min ~ C_min^0.03
```

These are useful for the calculator's batch size recommendation: given a compute budget, the calculator could suggest an optimal batch size in addition to the Chinchilla-optimal model size and token count.

However, note that Chinchilla found different compute-optimal allocations (N ~ C^0.50 instead of C^0.73), so the B_opt and S_min exponents from Kaplan may also be inaccurate. The calculator should use these with appropriate caveats.

---

## 6. Key Quantitative Details Not in the Spec

### 6.1 Non-Embedding Parameter Counting

The paper is very specific that N refers to **non-embedding parameters**. For a GPT-2 style model:
```
N_non_embedding = N_total - n_vocab * d_model - n_positions * d_model
```

But with weight tying (lm_head shares weights with token embedding), the token embedding weights ARE used as a matmul in the output layer, so they should be counted for FLOPs but position embeddings should not. The local implementation's `get_num_params()` method subtracts only position embeddings:

```python
def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.transformer.wpe.weight.numel()
    return n_params
```

Our spec already handles this correctly in Section 4.1's embedding exclusion discussion.

### 6.2 Learning Rate Schedule

Kaplan used a specific learning rate formula that depends on model size:
```
lr_max = 0.003239 - 0.0001395 * ln(N)
```

This is from Appendix D.1 of the paper. It is implemented in the local repo's configurator.py:
```python
learning_rate = 0.003239 - 0.0001395 * math.log(N)
```

This is relevant for the calculator if we ever add a "recommended learning rate" feature.

### 6.3 MFU Calculation in the Implementation

The local model.py includes an MFU (Model FLOPs Utilization) calculation:

```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
    N = self.get_num_params()
    cfg = self.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    flops_achieved = flops_per_iter * (1.0/dt)
    flops_promised = 312e12  # A100 bfloat16 peak
    mfu = flops_achieved / flops_promised
    return mfu
```

This confirms the formula: `FLOPs_per_token = 6N + 12*L*H*Q*T` where `12*L*H*Q*T = 12*L*d*s` (since H*Q = n_heads * head_dim = d_model, and T = block_size = sequence length). This matches our spec's PaLM formula.

### 6.4 PF-day Conversion

```
1 PF-day = 8.64 x 10^19 FLOPs
```

The local implementation uses this in the notebook:
```python
[6 * N * step['iter'] * tokens_per_iter / 8.64e19 for step in run_history[1:]]
```

This is a standard conversion already implied in our spec but worth having as an explicit constant.

### 6.5 The alpha_C_min Consistency Relationship

Kaplan derives that the compute-efficient loss scaling exponent is:
```
alpha_C_min = 1 / (1/alpha_S + 1/alpha_B + 1/alpha_N)
```

Plugging in: 1 / (1/0.76 + 1/0.21 + 1/0.076) = 1 / (1.316 + 4.762 + 13.158) = 1/19.236 = 0.052, which is close to the fitted 0.050.

This is a self-consistency check and could be used to validate scaling law parameters.

---

## 7. Edge Cases and Gotchas

### 7.1 Small D Regime
The L(N,D) formula breaks down for very small datasets (D < ~10^7 tokens in the reproduction). This is because overfitting happens so early that the model enters a different regime. The calculator should not use these scaling laws for datasets below ~10M tokens.

### 7.2 Kaplan Underfitting Warning
The paper's most important finding -- that N should scale as C^0.73 -- was later shown to be wrong (Chinchilla finds C^0.50). This happened because Kaplan's models were **all undertrained on data**, meaning the data axis was never properly explored. At the time, GPT-3 (175B params, 300B tokens, D/N = 1.7:1) was considered normal. By Chinchilla standards, GPT-3 should have been trained on 3.5T tokens.

Our spec correctly notes this in Section 4.3's "Historical Context" paragraph.

### 7.3 Loss Units
All losses in the paper are in nats (natural log base), not bits. The cross-entropy loss is computed using log base e. To convert to bits: L_bits = L_nats / ln(2) ≈ L_nats / 0.693.

### 7.4 Batch Size Units
B_crit is measured in **tokens**, not in sequences. The batch size in the formula B_crit(L) = B_star / L^(1/alpha_B) with B_star = 2 x 10^8 gives a result in tokens. If the user's batch size is specified as sequences, multiply by sequence length first.

### 7.5 The C = 6NE vs C = 6ND Notational Confusion
Kaplan uses E for "total training examples" (= B * S, where each example is a sequence of tokens). The more common modern notation is D for "total training tokens." The relationship is:
```
E (examples) * n_ctx (tokens per example) = D (total tokens)
```

So C = 6*N*E*n_ctx = 6*N*D. The notebook implementation uses `C = 6*N*step*tokens_per_iter` which correctly multiplies by the number of tokens per step.

---

## 8. What's Unique / Non-Obvious

### 8.1 Additive vs. Multiplicative Loss Decomposition
Kaplan's L(N,S) = (N_c/N)^alpha_N + (S_c/S)^alpha_S is **additive**, while L(N,D) has a more complex nested power-law structure. The additive form for L(N,S) implies that training longer cannot compensate for an undersized model -- the loss floor is set by N alone. This is important for the calculator: if a user specifies a very large number of training steps with a small model, the calculator can show that additional training provides diminishing returns and the bottleneck is model size.

### 8.2 The 100x Compute -> 27x Model / 3x Batch Rule
When compute increases 100x, the optimal allocation is:
- 27x to model size (100^0.73)
- 3x to batch size (100^0.24)
- 1.07x to training steps (100^0.03)

This was Kaplan's most influential finding (later revised by Chinchilla). Even though the exponents differ, the **qualitative insight** that training steps barely need to increase is robust across both Kaplan and Chinchilla.

### 8.3 Critical Batch Size Grows During Training
Since B_crit depends on loss L, and loss decreases during training, B_crit increases during training. A batch size that is above B_crit early in training might be below B_crit later. Some advanced training systems dynamically increase batch size during training to track B_crit. The calculator could show B_crit at both the initial loss and the predicted final loss.

### 8.4 Transfer Learning Constant Offset
The paper shows that out-of-distribution performance follows the same scaling law as in-distribution performance, but with a constant additive loss offset:
```
L_OOD = L_in_distribution + constant
```

The constant depends on the distribution shift but not on model size. This means scaling predictions transfer across domains.

---

## 9. Recommendations for the Spec

### High Value (should add):
1. **C_min adjustment formula** (Section 5.2): `C_min = C / (1 + B / B_crit(L))` -- allows the calculator to show compute efficiency given the user's batch size choice. This is a one-line formula that adds significant value.

2. **PF-day constant**: `1 PF-day = 8.64 x 10^19 FLOPs` -- useful for displaying compute in human-readable units.

### Medium Value (consider adding):
3. **L(N, S_min) formula** (Section 5.1): For showing training trajectory curves. This would require alpha_S and S_c from Kaplan. However, since Chinchilla has its own L(N,D) formula that we already use, adding a separate L(N,S) formula may create confusion. Consider adding it only if the calculator includes a "training curve visualization" feature.

4. **Overfitting data floor** (Section 5.3): `D >= 5e3 * N^0.74` as a warning for insufficient training data, distinct from Chinchilla-optimal.

### Low Value (skip):
5. Kaplan's L(N,D) formula -- we already use Chinchilla's version which is more accurate.
6. N_opt(C) = N_e * C^p_N -- superseded by Chinchilla.
7. Learning rate formula -- too model-specific and not within our calculator's scope.
8. Transfer learning offset -- interesting but not actionable for the calculator.

---

## 10. Key File Paths

- Paper HTML: `https://ar5iv.labs.arxiv.org/html/2001.08361`
- Local reproduction notebook: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/scaling_laws_repo/kaplan_scaling_laws.ipynb`
- Local model with MFU calculation: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/scaling_laws_repo/model.py` (lines 289-302)
- Local configurator with lr formula: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/scaling_laws_repo/configurator.py` (line 66)
- Local train.py with FLOPs computation: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/scaling_laws_repo/train.py` (line 111, 339)
- Current spec: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/spec/llm-training-gpu-calculator-spec.md`
