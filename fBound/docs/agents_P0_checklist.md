# AGENTS.md — P0 Compliance Checklist

This checklist extracts the **P0 correctness/stability requirements** from `AGENTS.md` into binary, verifiable items.

## Repository layout (P0)
- [ ] Repo follows the required layout:
  - [ ] `project-root/AGENTS.md` exists at repo root
  - [ ] `project-root/src/estimators/` exists (estimator code)
  - [ ] `project-root/src/utils/` exists (shared utilities)
  - [ ] `project-root/tests/experiments/` exists (experiment scripts)
- [ ] Imports are clean under the `src/` layout (no reliance on working directory side-effects).
- [ ] Experiment/plotting scripts live under `tests/experiments/` (or equivalent experiment folder), not inside estimator modules.
- [ ] If using the recommended refinement, a package directory exists under `src/` (e.g., `src/fbound/...`) and internal imports use that package.

## Validity and NaN policy (P0)

### Independent computations for upper and lower (P0)
- [ ] Upper bound is computed via training/evaluation with **`+phi`**.
- [ ] Lower bound is computed via **an independent run** with **`-phi`**, then negated:
  - [ ] `lower(phi) = - upper(-phi)`
- [ ] Validity is tracked separately:
  - [ ] `valid_up[f, i]` represents validity for the upper computation at point `i` for divergence `f`.
  - [ ] `valid_lo[f, i]` represents validity for the lower computation (via upper on `-phi`) at point `i` for divergence `f`.
- [ ] `valid_up` and `valid_lo` masks are **not coupled** (no early `valid = valid_up & valid_lo` before endpoints exist).

### Interval validity (only when producing an interval) (P0)
- [ ] Per divergence `f`, define inversion at evaluation point `i` as:
  - [ ] `inverted_f[i] = isfinite(lower_f[i]) & isfinite(upper_f[i]) & (lower_f[i] > upper_f[i])`
- [ ] Per divergence `f`, define interval validity at point `i` as:
  - [ ] `valid_interval[f, i] = valid_lo[f, i] & valid_up[f, i] & isfinite(lower_f[i]) & isfinite(upper_f[i]) & (~inverted_f[i])`
- [ ] If `valid_interval[f, i]` is false, the interval view for divergence `f` at point `i` is **blanked** (NaN endpoints).

### Terminate-and-NaN rule (P0)
- [ ] For each divergence `f`, each point `i`, and each bound computation:
  - [ ] If conjugate evaluation is invalid/out-of-domain or non-finite, that point is terminated and the bound value is set to NaN.
- [ ] The terminate-and-NaN rule is applied to:
  - [ ] pseudo-outcomes `Z_i`
  - [ ] per-divergence bound endpoint vectors
  - [ ] any intermediate step that can produce NaN/inf

## Domain-safe conjugate interface (P0)
- [ ] Every divergence implements `g_star_with_valid(t) -> (gstar, valid_mask)`.
- [ ] The input is defined as `t = (phi(Y) - u(A, X)) / lambda(A, X)`.
- [ ] `valid_mask[i]` is true **iff** `t_i` is strictly inside the domain of `g*` with margin `eps > 0`.

### Loss safety (P0)
- [ ] Training does **not** call raw `g_star(t)` on out-of-domain inputs if it can yield inf/NaN.
- [ ] Training computes `(gstar, valid)` via `g_star_with_valid(t)`.
- [ ] Batch loss is finite and constructed using:
  - [ ] `gstar` on valid points
  - [ ] a finite surrogate/barrier + penalty on invalid points
- [ ] Training logs the in-domain fraction per epoch/fold (e.g., `valid_frac_epoch`).

## Annealing-default feasibility steering (P0)
- [ ] Annealing is the **default** optimization technique for dual-net training.
- [ ] Two-stage default schedule is implemented:
  - [ ] Total epochs `E`; stage-1 epochs `E1 = ceil(rho * E)` with default `rho = 0.3`.
  - [ ] Domain penalty weight uses `w_dom(e) = w1` for `e < E1`, else `w2`.
  - [ ] Defaults satisfy `w1 >> w2` with:
    - [ ] `w2 = 1e4`
    - [ ] `w1 = 1e6`
- [ ] Required logging is produced:
  - [ ] `valid_frac_epoch` (or running average) per epoch
  - [ ] `valid_frac_stage1`
  - [ ] `valid_frac_stage2`
  - [ ] number/fraction of invalid pseudo-outcomes excluded from `m`-fitting

## Estimation of m(a, x) from pseudo-outcomes (P0)
- [ ] Pseudo-outcome is `Z_i = g*(t_i)`.
- [ ] `m` is fitted using **only** samples where:
  - [ ] `valid == True` (in-domain)
  - [ ] `Z_i` is finite
- [ ] A minimum number of valid samples per action per fold is enforced (configurable).
- [ ] If insufficient valid support exists:
  - [ ] downstream predictions that depend on `m` are marked NaN (per divergence, per bound computation)

## Aggregation across divergences (P0) — endpoint-wise aggregation (default)

### Pre-filtering and inversion handling (P0)
- [ ] For each divergence `f`, define:
  - [ ] `inverted_f = isfinite(L_f) & isfinite(U_f) & (L_f > U_f)`
- [ ] Filtering is applied **before** sorting/order-statistics.
- [ ] NaN values are never passed into sorting.

### Upper aggregation (P0)
- [ ] Candidate set is:
  - [ ] `U = { U_f : valid_up[f] & isfinite(U_f) & (~inverted_f) }`
- [ ] `kth_upper(k)` returns the k-th **smallest** element of `U`.
- [ ] Default `tight_upper` equals `kth_upper(1)` (i.e., `min(U)`).
- [ ] If `|U| = 0`, aggregated upper is NaN.

### Lower aggregation (P0)
- [ ] Candidate set is:
  - [ ] `L = { L_f : valid_lo[f] & isfinite(L_f) & (~inverted_f) }`
- [ ] `kth_lower(k)` returns the k-th **largest** element of `L`.
- [ ] Default `tight_lower` equals `kth_lower(1)` (i.e., `max(L)`).
- [ ] If `|L| = 0`, aggregated lower is NaN.

### Final interval rule after aggregation (P0)
- [ ] After computing aggregated endpoints `L_agg` and `U_agg`:
  - [ ] If either endpoint is NaN, the aggregated interval is blanked (NaN endpoints).
  - [ ] If `L_agg > U_agg`, the aggregated interval is blanked (NaN endpoints).

### Diagnostics (P0)
- [ ] Diagnostics are reported at each evaluation point:
  - [ ] `n_eff_up = |U|`
  - [ ] `n_eff_lo = |L|`
  - [ ] `k_used_up` (default 1)
  - [ ] `k_used_lo` (default 1)
- [ ] Counts by reason are available (per divergence, aggregated over `f`):
  - [ ] invalid upper (`valid_up = False`)
  - [ ] invalid lower (`valid_lo = False`)
  - [ ] non-finite endpoint
  - [ ] inverted interval filtered out
