# Deep Dive: An Empirical Model of Large-Batch Training (McCandlish et al., 2018)

**Paper**: arXiv:1812.06162
**Authors**: Sam McCandlish, Jared Kaplan, Dario Amodei, and the OpenAI Dota Team
**Date**: December 2018
**Relevance**: Introduces the gradient noise scale and critical batch size concepts used to determine optimal data parallelism and batch size selection for LLM training.

---

## 1. Core Concept: The Gradient Noise Scale

### 1.1 Definition (Full Noise Scale)

The **gradient noise scale** B_noise is defined as:

```
B_noise = tr(H * Sigma) / (G^T * H * G)
```

Where:
- G = true gradient (full-batch gradient over the data distribution)
- H = true Hessian of the loss at the current parameters
- Sigma = per-example gradient covariance matrix, defined as:

```
Sigma(theta) = cov_{x ~ rho}(grad_theta L_x(theta))
             = E_{x ~ rho}[(grad_theta L_x)(grad_theta L_x)^T] - G * G^T
```

This is computationally expensive because it requires the Hessian H.

### 1.2 Simplified Noise Scale (Practical)

The **simplified noise scale** B_simple assumes the Hessian is a multiple of the identity (well-conditioned optimization):

```
B_simple = tr(Sigma) / |G|^2
```

This equals: (sum of variances of individual gradient components) / (squared global norm of the gradient).

Intuitively, B_simple measures **how large the gradient is compared to its variance** -- the scale at which the estimated and true gradient become close in L2 space.

The normalized L2 distance between the estimated gradient and the true gradient is:

```
E[|G_est - G|^2] / |G|^2 = (1/B) * tr(Sigma) / |G|^2 = B_simple / B
```

### 1.3 Relationship Between the Two

In practice, B_simple and B_noise typically differ only by a small constant multiplicative factor. B_simple is much cheaper to compute and is the recommended practical measurement. The paper uses B_simple for most empirical work.

---

## 2. Key Formulas

### 2.1 Optimal Step Size as a Function of Batch Size

**Equation 2.6**: The optimal learning rate (step size) given batch size B:

```
epsilon_opt(B) = epsilon_max / (1 + B_noise / B)
```

Where epsilon_max = |G|^2 / (G^T * H * G) is the optimal step size with the noiseless true gradient.

**Behavior**:
- When B << B_noise: epsilon_opt ~ epsilon_max * (B / B_noise), scales linearly with B
- When B >> B_noise: epsilon_opt ~ epsilon_max, saturates

### 2.2 Optimal Loss Improvement Per Step

**Equation 2.7**: The optimal improvement in loss from a single gradient step at batch size B:

```
Delta_L_opt(B) = Delta_L_max / (1 + B_noise / B)
```

Where Delta_L_max = (1/2) * |G|^4 / (G^T * H * G) is the maximum possible improvement (with the true gradient).

**This is the central formula of the paper.** It has two regimes:
- **B << B_noise**: Delta_L ~ Delta_L_max * (B / B_noise). Doubling B nearly doubles the progress per step. This is the **perfect scaling** regime for data parallelism.
- **B >> B_noise**: Delta_L ~ Delta_L_max. Increasing B gives no additional progress per step -- you are just wasting compute. This is the **ineffective scaling** regime.
- **B ~ B_noise**: The turning point. Training speed drops to ~50% of maximum possible.

### 2.3 Gradient Estimated from a Batch

**Equation 2.1**:
```
G_est(theta) = (1/B) * sum_{i=1}^{B} grad_theta L_{x_i}(theta);    x_i ~ rho
```

**Equation 2.2**: Expected value and covariance of the estimated gradient:
```
E[G_est] = G(theta)
cov(G_est) = (1/B) * Sigma(theta)
```

The variance scales inversely with batch size B -- this is the fundamental reason larger batches give better gradient estimates.

### 2.4 Training Speed/Efficiency Tradeoff

**Equation 2.11** (the hyperbolic tradeoff law): Averaged over a full training run, the relationship between time-efficiency and data-efficiency follows:

```
(S / S_min - 1) = (E / E_min - 1)^(-1)
```

Equivalently:
```
(S / S_min - 1) * (E / E_min - 1) = 1
```

Where:
- S = actual number of optimization steps to reach a target loss
- S_min = minimum possible steps (infinitely large batch, maximally time-efficient)
- E = actual number of training examples processed (E = B * S for fixed batch size)
- E_min = minimum possible examples (batch size = 1, maximally compute-efficient)

This is a **hyperbola in (S, E) space**. It generates the Pareto frontier curves seen in Figure 1 and Figure 7 of the paper.

### 2.5 Critical Batch Size Definition

**Equation 2.12**:

```
B_crit = E_min / S_min
```

When B = B_crit, both sides of Equation 2.11 equal 1, meaning:
- Training takes **2x S_min** steps (twice the minimum steps)
- Training processes **2x E_min** examples (twice the minimum data)

This is the natural compromise: you waste exactly a factor of 2 in both time and compute relative to their respective optima.

The model predicts: **B_crit ~ B_noise** (where B_noise is appropriately averaged over the training run).

### 2.6 Expressing S and E as Functions of B

From the tradeoff law, at fixed batch size B with E = B*S:

```
S(B) = S_min * (1 + B_crit / B)
E(B) = E_min * (1 + B / B_crit)
```

These show:
- Steps S: Starts at 2 * S_min when B = B_crit, approaches S_min as B -> infinity
- Examples E: Starts at 2 * E_min when B = B_crit, grows linearly with B above B_crit

---

## 3. How to Measure the Noise Scale in Practice

### 3.1 Zero-Overhead Method (Appendix A.1)

This method works **for free** in data-parallel training by comparing gradient norms at two different batch sizes.

**Step 1**: From Equation A.1, the expected gradient norm for a batch of size B is:
```
E[|G_est|^2] = |G|^2 + (1/B) * tr(Sigma)
```

**Step 2**: Given gradient estimates from two batch sizes B_small and B_big (e.g., before and after averaging across data-parallel workers), compute unbiased estimates:

```
|G|^2 = (1 / (B_big - B_small)) * (B_big * |G_{B_big}|^2 - B_small * |G_{B_small}|^2)

S = (1 / (1/B_small - 1/B_big)) * (|G_{B_small}|^2 - |G_{B_big}|^2)
```

**Equation A.2**: These satisfy E[|G|^2] = |G|^2 and E[S] = tr(Sigma).

**Step 3**: B_simple = S / |G|^2. However, this ratio is biased (E[x/y] >= E[x]/E[y]). So in practice:
- Compute |G|^2 and S on every training step
- Maintain separate exponentially-weighted moving averages (EMA) of each
- The ratio of the EMAs gives a good estimate of B_simple

**In data-parallel training**: B_small is the "local" batch size on each worker (before all-reduce), and B_big is the "global" batch size (after all-reduce). You compute |G_{B_small}|^2 from the local gradient norm and |G_{B_big}|^2 from the globally averaged gradient norm. This requires essentially zero extra computation.

### 3.2 Line Search Method (More Accurate, More Expensive)

For each of several batch sizes B, perform a line search to measure Delta_L(B), then fit to Equation 2.7:

```
Delta_L(B) = Delta_L_max / (1 + B_noise / B)
```

This directly estimates B_noise (the full Hessian-weighted noise scale) and also verifies the functional form. Used for validation rather than routine measurement.

---

## 4. Critical Batch Size as a Function of Loss (Kaplan et al. Adaptation)

### 4.1 B_crit Grows During Training

McCandlish et al. (2018) establish that B_crit increases as the loss decreases during training. The noise scale grows because:
- tr(Sigma) stays roughly constant during training
- |G| decreases as the model approaches the loss minimum
- Therefore B_simple = tr(Sigma) / |G|^2 increases

The critical batch size typically increases by **at least an order of magnitude** over the course of a training run.

### 4.2 Power Law Fit from Kaplan et al. (2020)

Kaplan et al. (arXiv:2001.08361) quantified B_crit as a function of training loss for language models:

```
B_crit(L) = B_star / L^(1 / alpha_B)
```

**Fitted constants from Kaplan et al. (Table 5 of their paper)**:
```
alpha_B = 0.21
B_star = 2.0 x 10^8 tokens     (equivalently written as B_* in the paper)
exponent 1/alpha_B ~ 4.76
```

So: B_crit(L) = 2.0 x 10^8 / L^4.76

**Example values** (using these Kaplan constants with cross-entropy loss in nats):
- L = 3.5: B_crit ~ 2.0e8 / 3.5^4.76 ~ 2.0e8 / 436 ~ 460K tokens
- L = 3.0: B_crit ~ 2.0e8 / 3.0^4.76 ~ 2.0e8 / 210 ~ 950K tokens
- L = 2.5: B_crit ~ 2.0e8 / 2.5^4.76 ~ 2.0e8 / 85 ~ 2.4M tokens
- L = 2.0: B_crit ~ 2.0e8 / 2.0^4.76 ~ 2.0e8 / 27 ~ 7.4M tokens

Note: The critical batch size approximately doubles for every 13% decrease in loss.

### 4.3 Independent Replication (shehper/scaling_laws nanoGPT)

An independent replication using nanoGPT on OpenWebText found:
```
alpha_B = 0.23
B_star = 2.2 x 10^7
```

The B_star difference (10^7 vs 10^8) is expected due to different model scales, datasets (OpenWebText vs WebText2), and loss ranges (9.0--5.9 vs 10.0--3.0). The alpha_B exponents agree closely.

### 4.4 B_crit Expressed as a Function of Compute

From Kaplan et al. Table 6, the compute-efficient scaling:
```
B_crit = B_e * C_min^(p_B)
```
Where:
```
B_e = 2.0 x 10^6 tokens
p_B = 0.24
```

This means: as compute grows 100x, the critical batch size grows by about 100^0.24 ~ 3x.

---

## 5. Minimum Compute Adjustment

### 5.1 C_min Formula

When training at batch size B (which may not equal B_crit), the **minimum compute** C_min that would achieve the same loss if training were maximally compute-efficient is:

```
C_min = C / (1 + B / B_crit(L))
```

Where C = 6*N*B*S is the actual compute used.

- If B << B_crit: C_min ~ C (you were already near compute-efficient)
- If B >> B_crit: C_min ~ C * B_crit / B (much of your compute was wasted)
- If B = B_crit: C_min = C/2

### 5.2 Minimum Steps Adjustment

The minimum number of training steps (maximally time-efficient) is related to actual steps S at batch size B by:

```
S_min = S / (1 + B_crit(L) / B)
```

- If B >> B_crit: S_min ~ S (you were already near time-efficient)
- If B << B_crit: S_min ~ S * B / B_crit (you were taking many more steps than necessary)
- If B = B_crit: S_min = S/2

---

## 6. Table 1: Concrete B_crit and Noise Scale Values Across Tasks

| Task | B_crit (Start) | B_crit (Average) | B_simple (Start) | B_simple (Average) |
|---|---|---|---|---|
| **Image Classification** | | | | |
| MNIST | 20 | 200 | 50 | 900 |
| SVHN | 50 | 500 | 300 | 4,000 |
| CIFAR10 | 300 | 900 | 400 | 2,000 |
| ImageNet | 1,000 | 15,000 | 4,000 | 30,000 |
| **Generative/Language Modeling** | | | | |
| Autoencoder (SVHN) | 10 | 40 | 2 | 2 |
| VAE (SVHN) | 10 | 200 | 10 | 10 |
| Billion Word LSTM (per token) | 700 | 100,000 | 1,000 | 150,000 |
| **Reinforcement Learning** | | | | |
| Atari (per frame) | 100--1,000 | 400--8,000 | 100--1,000 | 1,000--20,000 |
| Dota 1v1 (per frame) | 50,000 | 3,000,000 | 100,000 | 300,000 |
| Dota 5v5 (per frame) | (not measured) | >8,000,000 (est.) | 100,000 | 24,000,000 |

Key observations:
- B_simple predicts B_crit at the order of magnitude level across 6 orders of magnitude variation
- The ratio B_simple / B_crit can vary by about an order of magnitude between tasks
- Language modeling (Billion Word): B_crit goes from ~700 tokens early in training to ~100K tokens by the end

---

## 7. Expected Patterns in the Noise Scale

### 7.1 Patterns Confirmed by the Paper

1. **Grows over training**: B increases as |G| decreases while tr(Sigma) stays roughly constant. Larger noise scale later in training means larger batches become useful later.

2. **Larger for more complex/difficult tasks**: Dota 5v5 >> Dota 1v1 >> Atari >> ImageNet >> CIFAR10 >> MNIST. More diverse data means individual gradients are less correlated.

3. **Weak dependence on model size**: At fixed loss values, B_simple is roughly independent of model size. Confirmed on Billion Word LSTMs of sizes 512, 1024, 2048 (Figure 8). Larger models have higher noise scale only because they achieve lower loss.

4. **Depends on learning rate via "temperature"**: The noise scale is proportional to 1/T where T = epsilon / epsilon_max(B). This means inflated learning rates produce artificially low noise scale estimates. Must use a well-tuned learning rate.

### 7.2 Temperature Dependence

**Equation C.1**: The training "temperature" is defined as:
```
T(epsilon, B) = epsilon / epsilon_max(B)
```

**Equation C.2**: In equilibrium:
```
B_noise ~ B_simple ~ 1/T
```

If the learning rate is halved, the noise scale approximately doubles. If the learning rate and batch size are both scaled by the same factor, the noise scale remains unchanged.

In the toy quadratic loss model:
```
B_simple = tr(Sigma) / |G|^2 ~ (B / epsilon) * tr(Sigma) / tr(H^2 * Sigma)
B_noise = tr(H*Sigma) / (G^T * H * G) ~ (B / epsilon) * tr(H * Sigma) / tr(H^3 * Sigma)
```

---

## 8. Dynamically Varying the Batch Size

### 8.1 Theory

When the noise scale varies during training, efficiency can be improved by dynamically adjusting the batch size. The total steps and examples for a dynamically-batched run are:

```
S = integral (1 + B(s)/B(s)) ds        [Equation D.1]
E = integral (B(s) + B(s)) ds
```

Where B(s) is the noise scale at training progress point s and B(s) is the actual batch size.

The optimal batch size schedule is to vary B proportionally to the **square root** of the noise scale:

```
B(s) = sqrt(r * B(s))                  [Equation D.3, D.6]
```

Where r is the "exchange rate" -- a free parameter reflecting the relative value of training time vs compute.

### 8.2 Practical Adaptive Batch Size

The paper proposes dynamically setting:
```
B = sqrt(r * B_simple)
```

with B_simple measured periodically during training.

### 8.3 Efficiency Gain from Adaptive Batching

The Pareto frontier with adaptive batch sizes becomes:

```
S_tot / S_min - 1 = gamma * (E_tot / E_min - 1)^(-1)      [Equation D.4]
```

Where gamma (the "variability parameter") measures how much the noise scale changes:
```
gamma = (integral sqrt(B) ds)^2 / (S_min * E_min)          [Equation D.5]
```

- gamma = 1 when noise scale is constant (no benefit from adaptive batching)
- gamma < 1 when noise scale varies (improvement in Pareto frontier)

For SVHN where B_crit ~ 10*sqrt(s): gamma = 24/25 ~ 0.96, giving only ~4% efficiency improvement from adaptive batching. The benefits are modest unless B_crit changes dramatically.

---

## 9. Learning Rate Scaling Rules

### 9.1 For SGD

From the optimal step size formula (Eq 2.6), the optimal learning rate scales as:

```
epsilon_central(B) = epsilon_* / (1 + B_*/B)^alpha
```

Where:
- alpha = 1 for SGD or momentum
- 0.5 < alpha < 1.0 for Adam or RMSProp

**For SGD (alpha=1)**: This is the **linear scaling rule** -- learning rate scales linearly with batch size up to B ~ B_noise, then saturates.

### 9.2 For Adam

Adam's per-parameter update is approximately:
```
delta_theta_i ~ epsilon * sign(E[G_i]) / sqrt(1 + s_i / E[G_i]^2)
```

If step-to-step noise is dominated by batch noise, s_i scales as 1/B, implying a **square-root scaling rule** (alpha = 0.5). However, beta_2 is often large (0.999), so the second moment accumulator may not adapt to quick noise changes, pushing alpha back toward 1.0.

---

## 10. Relevance to an LLM Training GPU Calculator

### 10.1 What Can Be Directly Used

1. **B_crit(L) = B_star / L^(1/alpha_B)** with Kaplan's constants (B_star = 2.0e8, alpha_B = 0.21) provides an advisory indicator: given the user's batch size and predicted final loss, is the batch size above or below B_crit?

2. **The tradeoff formula** (S/S_min - 1)(E/E_min - 1) = 1 can quantify the compute waste or time waste at any given batch size relative to B_crit.

3. **C_min = C / (1 + B/B_crit)** lets the calculator show how much compute is "wasted" due to batch size being too large, or equivalently, how much faster training could be with a larger batch.

4. **S_min = S / (1 + B_crit/B)** quantifies how many fewer steps would be needed with maximal parallelism.

### 10.2 Calculator Display Suggestions

Given user inputs of model size N, sequence length, global batch size B (in tokens), and training tokens D:
- Compute predicted loss (from Chinchilla or other scaling law)
- Compute B_crit(L) at that loss
- Display ratio B/B_crit:
  - B/B_crit < 0.5: "Batch size is well below critical -- training is time-inefficient. Larger batches would speed up training with minimal compute waste."
  - 0.5 < B/B_crit < 2: "Batch size is near-optimal for compute/time tradeoff."
  - B/B_crit > 2: "Batch size exceeds critical -- training is compute-inefficient. Each additional doubling of batch size gives diminishing returns."
- Compute and display approximate compute waste factor: (1 + B/B_crit) / 2 (ratio of actual compute to minimum compute)
- Compute and display approximate time waste factor: (1 + B_crit/B) / 2 (ratio of actual steps to minimum steps)

### 10.3 Limitations and Caveats

1. **The Kaplan constants are for autoregressive Transformer language models trained on WebText2.** They may not generalize exactly to other architectures, datasets, or tokenizers.

2. **B_crit is measured in tokens, not sequences.** The paper (confirmed by Kaplan scaling_laws replication) shows that B_crit and B_simple depend on total number of tokens, not number of sequences, when sequence length varies.

3. **The noise scale depends on the learning rate.** A poorly-tuned learning rate will give misleading noise scale measurements. The constants assume well-tuned training.

4. **B_crit is advisory, not prescriptive.** Real-world batch size choices are constrained by memory, hardware parallelism, communication costs, and training stability -- B_crit is just one factor.

5. **These scaling laws predate Chinchilla.** The loss values from Kaplan et al. may not match Chinchilla-optimal training. However, B_crit depends on loss value, not on how that loss was achieved, so the formula remains usable.

6. **The noise scale is independent of the dataset size.** The formulas assume B << D (batch size much smaller than dataset size). With sampling without replacement from a finite dataset of size D, the variance instead scales as (1/B - 1/D), which matters when B approaches D.

### 10.4 What is Unique or Non-Obvious

1. **Model size does NOT directly affect B_crit** -- only indirectly through the loss achieved. A 7B model and a 70B model at the same loss value have the same B_crit. This is surprising and empirically validated.

2. **The ratio B_simple/B_crit varies by up to 10x across tasks**, but the order of magnitude always matches. For a calculator, B_simple is a good proxy.

3. **The zero-overhead measurement technique** (comparing gradient norms before and after all-reduce in data-parallel training) means noise scale could be monitored live during training at no computational cost.

4. **As compute grows 100x, B_crit grows only ~3x** (p_B = 0.24). This means training steps S_min barely changes (p_S = 0.03), while model size absorbs most of the additional compute (p_N = 0.73 from Kaplan). The implication for a calculator: even at very large scale, the number of training steps stays roughly in the 10^5 range.

5. **The Pareto frontier is hyperbolic, not linear.** Efficiency does not degrade gradually -- there is a relatively sharp transition around B_crit. The formula (S/S_min - 1)(E/E_min - 1) = 1 fits remarkably well across all tasks tested.

---

## 11. Key Formulas Summary (Quick Reference)

| Formula | Equation | Constants (Kaplan LLM) |
|---|---|---|
| Gradient noise scale (simplified) | B_simple = tr(Sigma) / \|G\|^2 | Measured empirically |
| Gradient noise scale (full) | B_noise = tr(H*Sigma) / (G^T*H*G) | Measured empirically |
| Loss progress per step | Delta_L(B) = Delta_L_max / (1 + B_noise/B) | -- |
| Optimal learning rate | eps_opt(B) = eps_max / (1 + B_noise/B) | -- |
| Time/compute tradeoff | (S/S_min - 1)(E/E_min - 1) = 1 | -- |
| Critical batch size | B_crit = E_min / S_min | -- |
| B_crit as function of loss | B_crit(L) = B_star / L^(1/alpha_B) | B_star=2e8, alpha_B=0.21 |
| B_crit as function of compute | B_crit = B_e * C_min^p_B | B_e=2e6, p_B=0.24 |
| Minimum compute adjustment | C_min = C / (1 + B/B_crit(L)) | -- |
| Minimum steps adjustment | S_min = S / (1 + B_crit(L)/B) | -- |
| Steps at fixed batch size | S(B) = S_min * (1 + B_crit/B) | -- |
| Examples at fixed batch size | E(B) = E_min * (1 + B/B_crit) | -- |
| Noise scale measurement | B_simple ~ EMA(S) / EMA(\|G\|^2) | -- |

---

## Relevant File Paths

- McCandlish et al. PDF: `/Users/vinitvyas09/.claude/projects/-Users-vinitvyas09-code-personal-llm-training-gpu-calculator/d91b4879-75e5-4f21-8298-32855ec485d1/tool-results/webfetch-1774994776993-8bzdnj.pdf`
- Kaplan et al. PDF: `/Users/vinitvyas09/.claude/projects/-Users-vinitvyas09-code-personal-llm-training-gpu-calculator/d91b4879-75e5-4f21-8298-32855ec485d1/tool-results/webfetch-1774994723861-mbbucc.pdf`
- Existing spec B_crit section: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/spec/llm-training-gpu-calculator-spec.md` (lines 438-458)
- Independent nanoGPT replication: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/scaling_laws_repo/kaplan_scaling_laws.ipynb`
- Research output: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/spec/research/mccandlish-large-batch-training-deep-dive.md`
