# Information-Theoretic Causal Bounds (CLeaR) — Research-Grade Refactor

This folder is a modular refactor of the original monolithic `fBound.py` into a small codebase that:

- Implements the **debiased loss** in **Definition 3 / Eq. (25)–(26)**, and
- Implements the full estimator pipeline in **Definition 4 (Step 1–6)**,
- Preserves the working design decisions in the script (PyTorch dual nets + sklearn propensity + sklearn regressor).

## What is implemented

### Debiased loss (Def. 12)

The training objective for the dual nets `h_β` and `u_γ` is the **debiased loss** (Definition 3):

- Main dual loss (Eq. (25)):
  \[
  \exp(h(A,X))\left\{\eta_f(A,X) + g^*\left(\frac{\phi(Y)-u(A,X)}{\exp(h(A,X))}\right)\right\} + u(A,X)
  \]
- Debiasing correction term (Eq. (26)):
  \[
  \sum_{a\in\{0,1\}} e_a(X)\exp(h(a,X))\eta_f'(e_a(X))\left(\mathbf{1}(A=a)-e_a(X)\right)
  \]

### Cross-fitting estimator (Definition 4, Step 1–6)

For each fold `k`:

1. Split data into `K` folds.
2. Fit propensity `\hat e^k` on `D^{-k}`.
3. Train dual nets by minimizing the debiased loss **on `D^k`** (paper-faithful).
4. Compute `\hat\lambda_k = exp(\hat h_k)` and `\hat\eta_f^k = B_f(\hat e^k)`.
5. Construct pseudo-outcome:
   \[
   Z_i^k = g^*\left(\frac{\phi(Y_i)-\hat u_k(A_i,X_i)}{\hat\lambda_k(A_i,X_i)}\right)
   \]
   and regress `Z_i^k` on `(A,X)` using `D^k` to obtain `\hat m^k`.
6. Return the bound:
   \[
   \hat\theta_\phi(a,x)= \frac{1}{K}\sum_{k=1}^K \hat\lambda_k(a,x)(\hat\eta_f^k(a,x)+\hat m^k(a,x))+\hat u_k(a,x).
   \]

### Lower bounds

The code computes the lower bound via the “sign-flip” identity:
\[
\theta_\phi^{\text{lower}}(a,x) = -\theta_{-\phi}^{\text{upper}}(a,x).
\]

This is implemented by running the estimator twice: once for `phi`, once for `-phi`.

## File guide

- `divergences.py`  
  Divergence registry providing `B(e)`, `dB(e)`, and `g_star(t)`. Supports: **KL**, **Hellinger**, **Chi2**, **TV**, **JS**, and custom divergences.

- `models.py`  
  Model factories:
  - propensity: `"logistic"`, `"xgboost"` (optional)
  - regression: `"random_forest"`, `"linear"`, `"xgboost"` (optional)
  - includes the PyTorch `TorchMLP` used for dual nets.

- `causal_bound.py`  
  The estimator implementation (`DebiasedCausalBoundEstimator`) and a convenience function `compute_causal_bounds(...)`.

- `data_generating.py`  
  Toy generator with known `GroundTruth(a, X)` (analytic).

- `run_example.py`  
  End-to-end run on simulated data for **KL** and **TV**.

- `test_sanity.py`  
  A small set of tests (3–6) covering reproducibility, shape checks, domain penalties, and bound ordering.

- `utils.py`  
  Seeding, KFold splits, and a utility for macOS thread-safety knobs.

## How to run

From the `project/` directory:

```bash
python run_example.py
python test_sanity.py
