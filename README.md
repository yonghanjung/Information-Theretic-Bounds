# Data-Driven Information-Theoretic Causal Bounds under Unmeasured Confounding

This repo implements the method in `docs/ITB.pdf` (Jung & Kang, 2026). It provides **data-driven** lower and upper bounds on causal estimands under unmeasured confounding without bounded outcomes, sensitivity parameters, instruments/proxies, or full SCM specification.

Target estimands (paper notation):
$$
\theta(a,x)=\mathbb{E}_{Q_{a,x}}[\varphi(Y)],\quad Q_{a,x}=\mathbb{P}(Y\mid do(A=a),X=x),
$$

with 
$$
\theta(a)=\mathbb{E}_{Q_a}[\varphi(Y)]
$$

## Quick start (minimal, fast)

```bash
PYTHONPATH=src python3 - <<'PY'
import numpy as np
import torch
from fbound.utils.data_generating import generate_data
from fbound.estimators.causal_bound import compute_causal_bounds

data = generate_data(n=200, d=3, seed=123, structural_type="linear")
dual_net_config = {
    "hidden_sizes": (16, 16),
    "activation": "relu",
    "dropout": 0.0,
    "h_clip": 10.0,
    "device": "cpu",
}
fit_config = {
    "n_folds": 2,
    "num_epochs": 3,
    "batch_size": 32,
    "lr": 5e-3,
    "weight_decay": 0.0,
    "max_grad_norm": 5.0,
    "eps_propensity": 1e-3,
    "deterministic_torch": True,
    "train_m_on_fold": True,
    "propensity_config": {"C": 1.0, "max_iter": 200, "penalty": "l2", "solver": "lbfgs", "n_jobs": 1},
    "m_config": {"alpha": 1.0},
    "verbose": False,
    "log_every": 1,
}

def phi_identity(y: torch.Tensor) -> torch.Tensor:
    return y

df = compute_causal_bounds(
    Y=data["Y"],
    A=data["A"],
    X=data["X"],
    divergence="KL",
    phi=phi_identity,
    propensity_model="logistic",
    m_model="linear",
    dual_net_config=dual_net_config,
    fit_config=fit_config,
    seed=123,
)
print(df[["lower", "upper"]].head())
PY
```

## End-to-end example

```bash
python3 run_example.py
```

Notes:
- `kth`/`tight_kth` use endpoint-wise order statistics via `aggregate_endpointwise`.

## Validity + NaN semantics (AGENTS P0)

- Upper and lower bounds are computed independently: upper uses `+phi`, lower uses `-phi` then sign-flip.
- `valid_up` and `valid_lo` are tracked separately per divergence.
- Interval validity is `valid_interval = valid_up & valid_lo & isfinite(lower) & isfinite(upper) & (lower <= upper)`.
- If `valid_interval` is false, both endpoints are blanked (set to NaN).
- NaN/inf are never passed into aggregation; endpoints are filtered before sorting.

## Diagnostics (AGENTS P0)

Endpoint-wise aggregation reports per-point diagnostics:
- `n_eff_up`, `n_eff_lo`: effective candidate counts after filtering.
- `k_used_up`, `k_used_lo`: order-statistic index used (default 1).
- `invalid_up`, `invalid_lo`, `nonfinite_upper`, `nonfinite_lower`, `inverted_filtered`: rejection counts.

Run example outputs include validity masks and these diagnostics in the saved tables.

## Theory at a glance (ITB.pdf)

### Divergence bound from propensity (Theorem 1)

For any action `a` and covariates `x` with `P(a|x)>0`:
$$
D_f(P_{a,x}\|Q_{a,x})\le B_f(e_a(x)),\quad B_f(e)=e f(1/e)+(1-e)f(0)
$$
This upper bound depends **only** on the propensity score, making the divergence radius fully data-driven.

Specializations used in the code:
- `KL`: `D_KL(P||Q) ≤ -log e`
- `Hellinger`: `D_H(P||Q) ≤ 1 - sqrt(e)`
- `Chi2`: `D_{chi2}(P||Q) ≤ (1-e)/(2e)`
- `TV`: `D_TV(P||Q) ≤ 1 - e`
- `JS`: `D_JS(P||Q) ≤ B_fJS(e)` (closed form in the paper)

### Dual causal bound (Theorem 2)

Define `g(s)=s f(1/s)` and its convex conjugate `g*(t)`. The upper bound solves:
$$
\theta_{up}(a,x)=\inf_{\lambda>0,\,u\in\mathbb{R}}
\Big\{\lambda\,\eta_f(a,x)+u+\lambda\,\mathbb{E}_{P_{a,x}}\big[g^*\big((\varphi(Y)-u)/\lambda\big)\big]\Big\}.
$$

### Debiased semiparametric estimator (Section 5)

The code minimizes the paper’s risk function (Definition 4) with cross-fitting and a Neyman-orthogonal correction, using:
- PyTorch dual nets for `h(a,x)` and `u(a,x)` with `\lambda(a,x)=exp(h(a,x))`
- sklearn propensity + outcome regressors for nuisances

### Cross-fitting estimator (Definition 4, Step 1–6)

For each fold `k`:

1. Split data into `K` folds.
2. Fit propensity `\hat e^k` on `D^{-k}`.
3. Train dual nets by minimizing the debiased loss **on `D^k`** (paper-faithful).
4. Compute `\hat\lambda_k = exp(\hat h_k)` and `\hat\eta_f^k = B_f(\hat e^k)`.
5. Construct pseudo-outcome:
   $$
   Z_i^k = g^*\left(\frac{\phi(Y_i)-\hat u_k(A_i,X_i)}{\hat\lambda_k(A_i,X_i)}\right)
   $$
   and regress `Z_i^k` on `(A,X)` using `D^k` to obtain `\hat m^k`.
6. Return the bound:
   $$
   \hat\theta_\phi(a,x)= \frac{1}{K}\sum_{k=1}^K \hat\lambda_k(a,x)(\hat\eta_f^k(a,x)+\hat m^k(a,x))+\hat u_k(a,x).
   $$

### Lower bounds

The code computes the lower bound via the “sign-flip” identity:
$$
\theta_\phi^{\text{lower}}(a,x) = -\theta_{-\phi}^{\text{upper}}(a,x).
$$

This is implemented by running the estimator twice: once for `phi`, once for `-phi`.

## File guide

- `src/fbound/estimators/causal_bound.py`  
  Core estimator (`DebiasedCausalBoundEstimator`) and `compute_causal_bounds(...)`.

- `src/fbound/utils/divergences.py`  
  Divergence registry providing `B(e)`, `dB(e)`, and domain-safe `g_star_with_valid(...)`.

- `src/fbound/utils/models.py`  
  Model factories:
  - propensity: `"logistic"`, `"xgboost"` (optional)
  - regression: `"random_forest"`, `"linear"`, `"xgboost"` (optional)
  - includes the PyTorch `TorchMLP` used for dual nets.

- `src/fbound/utils/data_generating.py`  
  Toy generator with known `GroundTruth(a, X)` (analytic).

- `run_example.py`  
  End-to-end run on simulated data for base divergences plus kth/tight_kth aggregation.

- `src/experiments/`  
  Experiment scripts: `plot_*.py` and `load_plot_*.py` (moved from repo root).

- `scripts/reproduce_final_arxiv_plots.py`  
  Reproduces final-arxiv figures from the JSON summaries without rerunning simulations.

- `tests/`  
  Smoke and P0 validity tests (`pytest -q`).

### `run_example.py` outputs

- Prints mean width and empirical coverage for the selected `div` (set inside the script).
- Saves per-sample tables with bounds, validity, and diagnostics:
  - `gstar_bounds_table.csv`: per-divergence lower/upper, `valid_up/valid_lo/valid_interval`, `inverted`,
    g* validity masks, and kth diagnostics (`k_used_up_kth`, `k_used_lo_kth`).
  - `gstar_bounds_any_invalid.csv`: subset where any g* validity flag or interval validity flag is false.
- Saves a summary:
  - `gstar_bounds_summary.csv`: coverage_rate, mean_width, and validity fractions per method
    (`KL`, `TV`, `Hellinger`, `Chi2`, `JS`, `kth`, `tight_kth`, `Manski_empirical`).
