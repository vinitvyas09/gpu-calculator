# Deep Dive: Training Compute-Optimal Large Language Models (Chinchilla)

**Source**: Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models"
**URL**: https://arxiv.org/abs/2203.15556
**Type**: Research paper (DeepMind, NeurIPS 2022)
**Date reviewed**: 2026-03-31

---

## 1. Executive Summary

This paper established that for compute-optimal training, model size N and training tokens D should be scaled equally (both proportional to C^0.5), superseding Kaplan et al. (2020) who recommended scaling N as C^0.73 and D as C^0.27. The paper trained over 400 models from 70M to 16B parameters on 5B to 500B tokens. The key practical rule: D_optimal is approximately 20N. The paper validated this by training Chinchilla (70B params, 1.4T tokens) which outperformed Gopher (280B params, 300B tokens) using the same compute budget.

**CRITICAL**: The Approach 3 coefficients as published contain errors/rounding issues. Epoch AI's 2024 replication (arXiv:2404.10102) found that the published coefficients imply ~70 tokens/param (not ~20), and provided corrected coefficients that restore consistency with Approaches 1 and 2.

---

## 2. FLOPs Definition (Appendix F)

The paper uses a detailed per-component FLOPs calculation. They use a factor of 2 for multiply-accumulate cost. For the forward pass, they count:

**Embeddings:**
- 2 x seq_len x vocab_size x d_model

**Attention (single layer):**
- Key, query, value projections: 2 x 3 x seq_len x d_model x (key_size x num_heads)
- Key @ Query logits: 2 x seq_len x seq_len x (key_size x num_heads)
- Softmax: 3 x num_heads x seq_len x seq_len
- Softmax @ query reductions: 2 x seq_len x seq_len x (key_size x num_heads)
- Final linear: 2 x seq_len x (key_size x num_heads) x d_model

**Dense block (single layer):**
- 2 x seq_len x (d_model x ffw_size + d_model x ffw_size)

**Final logits:**
- 2 x seq_len x d_model x vocab_size

**Total forward pass FLOPs:**
embeddings + num_layers x (total_attention + dense_block) + logits

**Backward pass:** assumed to be 2x the forward pass FLOPs.

**Total training FLOPs = 3 x forward pass FLOPs** (1x forward + 2x backward).

### Comparison with C = 6ND approximation

Table A4 compares their detailed calculation with the 6ND approximation:

| Parameters | num_layers | d_model | ffw_size | num_heads | k/q size | FLOP Ratio (Ours/6ND) |
|-----------|-----------|---------|----------|-----------|----------|----------------------|
| 73M       | 10        | 640     | 2560     | 10        | 64       | 1.03                 |
| 305M      | 20        | 1024    | 4096     | 16        | 64       | 1.10                 |
| 552M      | 24        | 1280    | 5120     | 10        | 128      | 1.08                 |
| 1.1B      | 26        | 1792    | 7168     | 14        | 128      | 1.04                 |
| 1.6B      | 28        | 2048    | 8192     | 16        | 128      | 1.03                 |
| 6.8B      | 40        | 3584    | 14336    | 28        | 128      | 0.99                 |

**Key finding**: The 6ND approximation is accurate within 3-10% across the full range. For Gopher specifically, their detailed calculation gives 6.3e23 vs the 6ND approximation of 5.76e23.

**For the calculator**: Using C = 6ND is a safe approximation. The paper validates this.

---

## 3. The Core Optimization Problem

Given a fixed FLOPs budget C, find optimal model size N and tokens D:

**Equation (1):**
```
N_opt(C), D_opt(C) = argmin_{N,D s.t. FLOPs(N,D)=C} L(N, D)
```

Where FLOPs(N,D) approximately equals 6ND = C.

---

## 4. Three Approaches to Fitting Scaling Laws

### 4.1 Approach 1: Fix Model Sizes, Vary Training Tokens

**Method**: For a fixed family of models (70M to 10B params), train each for 4 different cosine cycle lengths (horizons varying by 16x). This gives a continuous mapping from FLOP count to training loss. At 1500 logarithmically-spaced FLOP values, find which model achieves lowest loss.

**Result**: Fit power laws N_opt proportional to C^a and D_opt proportional to C^b.

**Found**: a = 0.50, b = 0.50

**Confidence intervals** (10th/90th percentile via bootstrapping):
- a: 0.488 to 0.502
- b: 0.501 to 0.512

### 4.2 Approach 2: IsoFLOP Profiles

**Method**: Vary model size (up to 16B) for 9 fixed training FLOP counts (6e18 to 3e21 FLOPs). For each FLOP budget, find which model size minimizes final loss by fitting a parabola to each IsoFLOP curve.

**Result**: Fit power laws N_opt proportional to C^a and D_opt proportional to C^b.

**Found**: a = 0.49, b = 0.51

**Confidence intervals:**
- a: 0.462 to 0.534
- b: 0.483 to 0.529

### 4.3 Approach 3: Parametric Loss Function

**Method**: Fit all final losses from Approaches 1 and 2 as a parametric function of N and D.

**Equation (2) / Equation (5):**
```
L_hat(N, D) = E + A / N^alpha + B / D^beta
```

Where:
- E = irreducible loss (entropy of natural text / ideal generative process)
- A / N^alpha = approximation error (N-param transformer underperforms ideal)
- B / D^beta = finite data / optimization error (finite training steps on sample)

**Model fitting (Equation 3):**
```
min_{A,B,E,alpha,beta} SUM_i Huber_delta(log L_hat(N_i, D_i) - log L_i)
```

Using L-BFGS algorithm, Huber loss with delta = 10^-3.

**Practical fitting (Equation 11):**
```
min_{a,b,e,alpha,beta} SUM_i Huber_delta(LSE(a - alpha*log(N_i), b - beta*log(D_i), e) - log(L_i))
```

Where LSE is the log-sum-exp operator, and A = exp(a), B = exp(b), E = exp(e).

**Grid of initializations:**
- alpha in {0, 0.5, ..., 2}
- beta in {0, 0.5, ..., 2}
- e in {-1, -0.5, ..., 1}
- a in {0, 5, ..., 25}
- b in {0, 5, ..., 25}

### Published Approach 3 Fitted Coefficients (Equation 10):

```
L(N, D) = E + A / N^0.34 + B / D^0.28
```

With:
- E = 1.69
- A = 406.4
- B = 410.7
- alpha = 0.34
- beta = 0.28

**Note on theoretical expectations from Equation 10 text**: "the parameter/data coefficients are both lower than 1/2; this is expected for the data-efficiency coefficient (but far from the known lower-bound)."

### Compute-Optimal Allocation from Approach 3 (Equation 4):

By minimizing L_hat under the constraint FLOPs(N,D) approximately equals 6ND:

```
N_opt(C) = G * (C/6)^a
D_opt(C) = G^(-1) * (C/6)^b
```

Where:
```
G = (alpha * A / (beta * B))^(1/(alpha+beta))
a = beta / (alpha + beta)
b = alpha / (alpha + beta)
```

**Found**: a = 0.46, b = 0.54

**Confidence intervals:**
- a: 0.454 to 0.455
- b: 0.542 to 0.543

### Summary Table (Table 2): Scaling Exponents Across All Approaches

| Approach                        | a (N_opt ~ C^a) | b (D_opt ~ C^b) |
|--------------------------------|-----------------|-----------------|
| 1. Min over training curves     | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2. IsoFLOP profiles            | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3. Parametric loss              | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| Kaplan et al. (2020)           | 0.73             | 0.27             |

---

## 5. CRITICAL: Known Inconsistency in Approach 3 Coefficients

### The Problem

Epoch AI's replication attempt (Besiroglu et al., arXiv:2404.10102, 2024) identified three critical issues with the published Approach 3 coefficients:

**Issue 1: The published coefficients imply ~70 tokens/param, not ~20.**

Using the published values (E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28), the compute-optimal allocation gives:
- a = beta / (alpha + beta) = 0.28 / 0.62 = 0.452
- b = alpha / (alpha + beta) = 0.34 / 0.62 = 0.548

This means D grows faster than N, and at the scale of Chinchilla (~5.76e23 FLOPs), the optimal D/N ratio is approximately 70, not 20. This contradicts both Approaches 1 and 2 (which predict ~20x) and the actual Chinchilla training (70B params, 1.4T tokens = 20x ratio).

**Issue 2: The optimizer stopped before convergence.**

The L-BFGS algorithm used to fit the parameters stopped prematurely due to a poor choice of loss scale.

**Issue 3: Reported parameters are rounded in a way that introduces substantial bias.**

The confidence intervals in the paper are implausibly narrow (a = 0.454 to 0.455 would require ~600,000 observations, but only ~400 runs were used). The rounding of published coefficients further distorts predictions.

### Corrected Coefficients (Epoch AI, 2024)

| Parameter | Original (Hoffmann) | Corrected (Epoch AI) | Standard Error |
|-----------|--------------------|--------------------|---------------|
| E         | 1.69               | 1.8172             | +/- 0.058     |
| A         | 406.4              | 482.01             | +/- 124.58    |
| B         | 410.7              | 2085.43            | +/- 1293.23   |
| alpha     | 0.34               | 0.3478             | +/- 0.02      |
| beta      | 0.28               | 0.3658             | +/- 0.02      |

**Corrected allocation exponents:**
- a = beta_corrected / (alpha_corrected + beta_corrected) = 0.3658 / 0.7136 = 0.5127
- b = alpha_corrected / (alpha_corrected + beta_corrected) = 0.3478 / 0.7136 = 0.4873

This gives a near-equal scaling (both ~0.5) consistent with Approaches 1 and 2, and implies ~20 tokens/param -- matching the actual Chinchilla training recipe.

**The corrected model achieves lower loss for 90% of all observations** and a likelihood ratio test rejected the original fit at p < 10^-135.

### Recommendation for the Calculator

**Use the corrected Epoch AI coefficients**, not the original published ones:

```
L(N, D) = 1.8172 + 482.01 / N^0.3478 + 2085.43 / D^0.3658
```

**NOTE on B value**: The published Equation 10 text states B = 410.7 with beta = 0.28. The Epoch AI corrected fit gives B = 2085.43 with beta = 0.3658 -- a 5x increase in B accompanied by a 30% increase in beta. The large change in B reflects the fact that the original optimizer stopped prematurely, yielding a poor fit to the data-scaling term. The critical fix is in beta: raising it from 0.28 to 0.37 makes the model significantly more data-efficient, which is what restores the ~20 tokens/param rule and consistency with Approaches 1 and 2.

---

## 6. Compute-Optimal Tables

### Table 3: Estimated Optimal FLOPs/Tokens for Various Model Sizes (Approach 1)

| Parameters   | FLOPs     | FLOPs (in Gopher units) | Tokens       |
|-------------|-----------|------------------------|-------------|
| 400 Million | 1.92e+19  | 1/29,968               | 8.0 Billion  |
| 1 Billion   | 1.21e+20  | 1/4,761                | 20.2 Billion |
| 10 Billion  | 1.23e+22  | 1/46                   | 205.1 Billion|
| 67 Billion  | 5.76e+23  | 1                      | 1.5 Trillion |
| 175 Billion | 3.85e+24  | 6.7                    | 3.7 Trillion |
| 280 Billion | 9.90e+24  | 17.2                   | 5.9 Trillion |
| 520 Billion | 3.43e+25  | 59.5                   | 11.0 Trillion|
| 1 Trillion  | 1.27e+26  | 221.3                  | 21.2 Trillion|
| 10 Trillion | 1.30e+28  | 22515.9                | 216.2 Trillion|

**D/N ratios implied by Table 3:**
- 400M: 8.0B / 0.4B = 20.0x
- 1B: 20.2B / 1B = 20.2x
- 10B: 205.1B / 10B = 20.5x
- 67B: 1500B / 67B = 22.4x
- 175B: 3700B / 175B = 21.1x
- 280B: 5900B / 280B = 21.1x
- 1T: 21200B / 1000B = 21.2x
- 10T: 216200B / 10000B = 21.6x

**The ratio increases slowly with scale, from ~20x at 400M to ~22x at 10T.** This is consistent with the slight difference between alpha and beta (the exponents are not exactly equal).

### Table A3: Approaches 2 and 3 Predictions

| Parameters   | Approach 2 FLOPs | Approach 2 Tokens  | Approach 3 FLOPs | Approach 3 Tokens   |
|-------------|-----------------|-------------------|-----------------|-------------------|
| 400 Million | 1.84e+19        | 7.7 Billion       | 2.21e+19        | 9.2 Billion       |
| 1 Billion   | 1.20e+20        | 20.0 Billion      | 1.62e+20        | 27.1 Billion      |
| 10 Billion  | 1.32e+22        | 219.5 Billion     | 2.46e+22        | 410.1 Billion     |
| 67 Billion  | 6.88e+23        | 1.7 Trillion      | 1.71e+24        | 4.1 Trillion      |
| 175 Billion | 4.54e+24        | 4.3 Trillion      | 1.26e+25        | 12.0 Trillion     |
| 280 Billion | 1.18e+25        | 7.1 Trillion      | 3.52e+25        | 20.1 Trillion     |
| 520 Billion | 4.19e+25        | 13.4 Trillion     | 1.36e+26        | 43.5 Trillion     |
| 1 Trillion  | 1.59e+26        | 26.5 Trillion     | 5.65e+26        | 94.1 Trillion     |
| 10 Trillion | 1.75e+28        | 292.0 Trillion    | 8.55e+28        | 1425.5 Trillion   |

**Key observation**: Approach 3 predicts much more data (and more FLOPs) needed per model size than Approaches 1 and 2. This is the inconsistency stemming from the buggy Approach 3 coefficients. Approach 3 D/N ratios range from 23x at 400M to 143x at 10T -- far above the 20x rule.

---

## 7. Derivation of the 20x Rule

The "D_optimal approximately equals 20N" rule comes from two places:

### From Approaches 1 and 2 (direct empirical fit):
Since a approximately equals b approximately equals 0.5, we have N_opt proportional to C^0.5 and D_opt proportional to C^0.5. Given C = 6ND:
```
C = 6 * N_opt * D_opt
N_opt = k1 * C^0.5
D_opt = k2 * C^0.5
C = 6 * k1 * k2 * C
=> k1 * k2 = 1/6
```

The empirical ratio D_opt/N_opt = k2/k1 is approximately 20, which means k2/k1 = 20, combined with k1*k2 = 1/6, gives k1 = 1/sqrt(120) and k2 = 20/sqrt(120).

### From Approach 3 (analytical, using corrected coefficients):
```
G = (alpha * A / (beta * B))^(1/(alpha+beta))

With corrected values:
G = (0.3478 * 482.01 / (0.3658 * 2085.43))^(1/(0.3478+0.3658))
G = (167.6 / 762.9)^(1/0.7136)
G = (0.2199)^(1.401)
G approximately equals 0.132

N_opt = G * (C/6)^a = 0.132 * (C/6)^0.513
D_opt = (1/G) * (C/6)^b = 7.576 * (C/6)^0.487

D_opt / N_opt = (1/G^2) * (C/6)^(b-a) approximately equals 57.4 * (C/6)^(-0.026)
```

At C = 5.76e23 (Gopher compute):
```
D_opt / N_opt approximately equals 57.4 * (9.6e22)^(-0.026) approximately equals 20
```

This confirms the ~20x rule at Gopher-scale compute. The ratio is not constant but varies slowly with compute budget.

---

## 8. Scaling Results on Alternative Datasets (Appendix C)

Table A2 shows Approach 2 (IsoFLOP) results on C4 and GitHub datasets:

| Dataset    | a (N_opt ~ C^a) | b (D_opt ~ C^b) |
|-----------|-----------------|-----------------|
| MassiveText (main) | 0.49        | 0.51             |
| C4                 | 0.50        | 0.50             |
| GitHub             | 0.53        | 0.47             |
| Kaplan et al.      | 0.73        | 0.27             |

The equal-scaling result holds across datasets.

---

## 9. The 400+ Training Runs

### Model Size Distribution (Table A9)

The paper trained models ranging from:
- **Smallest**: 44M parameters (d_model=512, ffw_size=2048, kv_size=64, 8 heads, 8 layers)
- **Largest**: 16.18B parameters (d_model=5120, ffw_size=20480, kv_size=128, 40 heads, 47 layers)

Total of 55 distinct model architectures listed in Table A9. Each was trained multiple times with different learning rate schedules and token counts.

### Architecture patterns:
- kv_size transitions from 64 to 128 at around 425M parameters
- ffw_size = 4 * d_model consistently
- feed-forward size always 4x the model dimension
- Model depth (n_layers) ranges from 8 to 49

### Training details:
- Learning rate: max of 2e-4 for smallest, 1.25e-4 for largest models
- LR decay: 10x over cosine schedule
- Cosine cycle length: matched to target number of training steps
- Key insight: Setting cosine cycle length > 25% more than training steps degrades performance noticeably

---

## 10. Chinchilla Model Architecture (Table 4)

| Model          | Layers | Num Heads | Key/Value Size | d_model | Max LR   | Batch Size      |
|---------------|--------|-----------|---------------|---------|----------|----------------|
| Gopher 280B   | 80     | 128       | 128           | 16,384  | 4e-5     | 3M -> 6M tokens |
| Chinchilla 70B | 80     | 128       | 128           | 8,192   | 1e-4     | 1.5M -> 3M tokens |

Key differences from Gopher:
- AdamW optimizer (vs Adam for Gopher) -- improves loss and downstream performance
- SentencePiece tokenizer without NFKC normalization
- Forward/backward in bfloat16, float32 copy of weights in distributed optimizer state
- Batch size doubled midway through training
- Same dataset (MassiveText) but different sampling proportions

---

## 11. Key Takeaways for Existing LLMs (Table 1)

The paper showed most contemporary LLMs were significantly undertrained:

| Model         | Size (Params) | Training Tokens | D/N Ratio | Chinchilla-Optimal? |
|--------------|--------------|----------------|----------|-------------------|
| LaMDA        | 137B          | 168B            | 1.2x     | Severely undertrained |
| GPT-3        | 175B          | 300B            | 1.7x     | Severely undertrained |
| Jurassic     | 178B          | 300B            | 1.7x     | Severely undertrained |
| Gopher       | 280B          | 300B            | 1.1x     | Severely undertrained |
| MT-NLG 530B  | 530B          | 270B            | 0.5x     | Severely undertrained |
| **Chinchilla** | **70B**      | **1.4T**        | **20x**  | **Compute-optimal**  |

**For the calculator**: These data points can be used to show users where their planned training sits relative to known models.

---

## 12. Curvature in the FLOP-Loss Frontier (Appendix E)

The paper notes (and Figure A5 shows) that there is **negative curvature** in the log(N_opt) vs log(C) relationship at higher compute budgets. Fitting the first, middle, and final thirds of frontier points separately gives different slopes, suggesting that **even smaller models may be optimal for large FLOP budgets** than the simple power law predicts.

This is acknowledged as a limitation: "we may still be overestimating the optimal size of large models."

**For the calculator**: This means the 20x rule and the power-law fits may be slightly conservative (recommending models that are too large) at very high compute budgets (>10^24 FLOPs).

---

## 13. Comparison with Kaplan et al. (2020)

The key methodological differences that led to different conclusions:

1. **Kaplan fixed training tokens and learning rate schedule for all models.** This prevented modeling the impact of these hyperparameters on loss. Chinchilla varied both model size and training tokens independently.

2. **Kaplan used intermediate loss estimates** (training curves at arbitrary points). For a fixed LR schedule to 130B tokens, intermediate losses at D' << 130B overestimate the loss of a model trained with a schedule matched to D'. This biased Kaplan toward larger models.

3. **Kaplan used smaller models.** Most Kaplan runs had < 100M parameters. Chinchilla used models up to 16B, revealing curvature in the frontier.

At 10^21 FLOPs head-to-head (Section D.4, Figure A4): Chinchilla's Approach 1 predicted 2.86B params as optimal (vs Kaplan's 4.68B). A 2.80B model trained per Chinchilla outperformed a 4.74B model trained per Kaplan.

---

## 14. Optimal Cosine Schedule Length (Appendix B)

An important practical finding: **the cosine cycle length should approximately match the number of training steps.** Overestimating by more than 25% leads to clear drops in performance (Figure A1). This means the learning rate schedule must be decided before training begins (since it determines the total number of steps).

For a model trained over D tokens, the cosine schedule should decay over approximately D tokens (10x LR drop).

---

## 15. What's Unique / Non-Obvious for the Calculator

1. **The D/N ratio is NOT constant.** It varies from ~20x at small scale to ~22x at large scale (Table 3 data). A power-law fit to the optimal D/N ratio vs N or C would be more accurate than a flat 20x.

2. **Approach 3 published coefficients are wrong.** The calculator must use the Epoch AI corrected coefficients (E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658) or else loss predictions and compute-optimal recommendations will be significantly off.

3. **The 6ND approximation is validated** by this paper to within 3-10% accuracy (Table A4). The ratio is closer to 1.0 for larger models.

4. **Cosine schedule coupling**: The scaling law results assume the learning rate schedule is tuned to match the training length. Loss predictions are invalid if the LR schedule is mismatched.

5. **AdamW vs Adam matters**: Chinchilla used AdamW, Gopher used Adam. The paper found AdamW models only surpass Adam models around 80% through training but end significantly better (Figure A7).

6. **Float32 optimizer states**: Despite bfloat16 forward/backward, Chinchilla stores float32 copies of weights in the distributed optimizer state. This has memory implications.

7. **Batch size doubling**: Both Gopher and Chinchilla doubled batch size midway through training. This is standard practice but means the effective training compute calculation must account for non-constant batch size.

8. **Single epoch assumption**: All scaling analysis assumes training on less than one epoch of data. When D exceeds the unique data available, the scaling laws may not hold. MassiveText had enough unique data for 1.4T tokens with most subsets used < 1 epoch (Table A1), but MassiveWeb and Wikipedia were used > 1 epoch.

---

## 16. Formulas Summary for Calculator Implementation

### Loss Prediction (use corrected Epoch AI coefficients):
```python
def chinchilla_loss(N, D):
    """Predict training loss using corrected Chinchilla Approach 3 formula."""
    E = 1.8172
    A = 482.01
    alpha = 0.3478
    B = 2085.43
    beta = 0.3658
    return E + A / (N ** alpha) + B / (D ** beta)
```

### Compute-Optimal Allocation:
```python
import math

def chinchilla_optimal(C):
    """
    Given compute budget C (in FLOPs), return optimal (N, D).
    Uses corrected Epoch AI coefficients.
    """
    # Corrected coefficients
    alpha = 0.3478
    beta = 0.3658
    A = 482.01
    B = 2085.43
    
    # Allocation exponents
    a = beta / (alpha + beta)   # ~0.513, exponent for N
    b = alpha / (alpha + beta)  # ~0.487, exponent for D
    
    # Proportionality constant
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    
    N_opt = G * (C / 6.0) ** a
    D_opt = (1.0 / G) * (C / 6.0) ** b
    
    return N_opt, D_opt
```

### Simple 20x Rule:
```python
def chinchilla_simple(N):
    """Quick estimate: D_optimal is approximately 20 * N."""
    return 20 * N

def chinchilla_flops_simple(N):
    """Quick estimate of compute-optimal FLOPs for model of size N."""
    D = 20 * N
    return 6 * N * D  # = 120 * N^2
```

### Compute from Model + Token Count:
```python
def training_flops(N, D):
    """Estimate total training FLOPs. C = 6 * N * D."""
    return 6 * N * D
```

---

## 17. Comparison: Kaplan vs Chinchilla Scaling Law Formulas

### Kaplan et al. (2020):
```
L(N, D) = [(N_c / N)^(alpha_N / alpha_D) + D_c / D]^alpha_D
```
With: N_c = 8.8e13, alpha_N = 0.076, D_c = 5.4e13, alpha_D = 0.095

Optimal allocation: N_opt proportional to C^0.73, D_opt proportional to C^0.27

### Chinchilla / Hoffmann et al. (2022):
```
L(N, D) = E + A / N^alpha + B / D^beta
```
With corrected: E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658

Optimal allocation: N_opt proportional to C^0.51, D_opt proportional to C^0.49

**The key structural difference**: Chinchilla uses an additive decomposition (separate terms for model and data) while Kaplan uses a multiplicative/nested form. The Chinchilla form is more standard from a statistical learning theory perspective (classical bias-variance decomposition).

---

## 18. Cross-Dataset Consistency

The scaling behavior is consistent across MassiveText, C4, and GitHub (Table A2). The exponents (a, b) are stable; only the proportionality constants change. This means:

- The 20x rule generalizes across datasets
- But the absolute loss values and coefficients (A, B, E) are dataset-specific
- The calculator should note that loss predictions are calibrated to MassiveText and may not directly transfer

---

## 19. File Paths Referenced

- PDF: `/Users/vinitvyas09/.claude/projects/-Users-vinitvyas09-code-personal-llm-training-gpu-calculator/d91b4879-75e5-4f21-8298-32855ec485d1/tool-results/webfetch-1774994741429-vl7bly.pdf`
- Existing scaling laws notebook: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/scaling_laws_repo/kaplan_scaling_laws.ipynb`
- Existing spec: `/Users/vinitvyas09/code/personal/llm-training-gpu-calculator/spec/llm-training-gpu-calculator-spec.md`
