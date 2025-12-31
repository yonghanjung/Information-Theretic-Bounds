# AGENTS.md

## Change control / rollback policy (REQUIRED)

Before any agent makes code changes:

1. The user must create a rollback point:
   - either `git commit` (preferred), or at minimum a `git tag` on the current commit.
2. All changes must be done on a new branch (e.g., `refactor/*`).
3. Each logical refactor step must be small and independently revertible:
   - after each step, run tests (or at least run the minimal example),
   - then commit.

Rationale: the refactor touches numerical routines, optimization stability, and aggregation logic; rollback must be trivial.

---

## Priority labels

- **P0**: Must-have correctness/stability requirement. If unmet, outputs are not trustworthy or training can crash.
- **P1**: Strongly recommended improvement. Not strictly required for correctness, but materially improves robustness or usability.

---

## Objective

Implement the method in “Information-Theoretic Causal Bounds under Unmeasured Confounding” as a usable, tested Python package that estimates **lower and upper bounds** on causal estimands:

- Conditional:
  \[
  \theta(a,x)=\mathbb{E}_{Q_{a,x}}[\varphi(Y)], \quad Q_{a,x}=\mathbb{P}(Y \mid do(A=a), X=x)
  \]
- Marginal:
  \[
  \theta(a)=\mathbb{E}_{Q_a}[\varphi(Y)], \quad Q_a=\mathbb{P}(Y \mid do(A=a))
  \]

The package must support unbounded outcomes and heterogeneous effects without instruments/proxies or user-chosen sensitivity parameters, using propensity-based f-divergence radii from the paper.

A P0 engineering requirement is robust handling of **domain-restricted conjugates** \(g^\*\) and explicit NaN/invalid propagation during training, prediction, and aggregation.

---

## Repository layout (P0)

project-root (= fBound)/
├── AGENTS.md
├── README.md
├── docs/                # CLeaR26 paper PDF
├── src/
│   ├── estimators/
│   └── utils/
└── tests/
    └── experiments/     # experiment scripts

Recommended packaging refinement (optional but advised for clean imports):
- place a package directory under src/, e.g. `src/fbound/...`

---

## Core notation

- Treatment \(A \in \{0,1\}\) by default (finite \(\mathcal{A}\) optional).
- Covariates \(X \in \mathbb{R}^k\).
- Outcome \(Y \in \mathbb{R}^d\).
- Observational conditional \(P_{a,x}=\mathbb{P}(Y\mid A=a,X=x)\).
- Interventional conditional \(Q_{a,x}=\mathbb{P}(Y\mid do(A=a),X=x)\).
- Propensity \(e_a(x)=\mathbb{P}(A=a\mid X=x)\).

Propensity-only divergence radius (paper Thm 1):
\[
D_f(P_{a,x}\|Q_{a,x}) \le B_f(e_a(x)),
\quad
B_f(e)= e f(1/e) + (1-e) f(0).
\]
Define \(\eta_f(a,x):=B_f(e_a(x))\).

---

## Estimator (paper Def. 4)

For each fold \(k=1,\dots,K\):

1. Fit propensity \(\hat e^{(-k)}\) on training split \(D_{-k}\).
2. Fit dual nets \(h_k,u_k\) on holdout split \(D_k\) by minimizing the debiased empirical loss.
   - Parameterize \(\lambda(a,x)=\exp(h(a,x)) > 0\).
3. Define pseudo-outcome on \(D_k\):
   \[
   Z_i = g^\*\!\left(\frac{\varphi(Y_i)-\hat u_k(A_i,X_i)}{\hat\lambda_k(A_i,X_i)}\right).
   \]
4. Fit \(m_k(a,x)\approx \mathbb{E}[Z\mid A=a,X=x]\) using **valid pseudo-outcomes only** (see validity policy).
5. Predict fold-specific upper bound using \(\hat m_k\) trained from valid pseudo-outcomes only:
   \[
   \hat\theta_{up}^{(k)}(a,x)=\hat\lambda_k(a,x)\big(\hat\eta_{f,k}(a,x)+\hat m_k(a,x)\big)+\hat u_k(a,x).
   \]
   If \(\hat m_k\) is undefined (insufficient valid support) or any required quantity is non-finite, output NaN for affected points.
6. Average over folds.

Lower bound for \(\varphi\) via sign flip:
\[
\hat\theta_{lo}^{(\varphi)}(a,x) = -\hat\theta_{up}^{(-\varphi)}(a,x).
\]

---

## Validity and NaN policy (P0)

### Independent computations for upper and lower
Upper and lower are computed as **independent processes**:
- Upper bound: training/evaluation with \(+\varphi\).
- Lower bound: training/evaluation with \(-\varphi\), then negation.

Validity must be tracked separately:
- `valid_up[f,i]`: validity of the upper computation (for \(+\varphi\)) at point \(i\).
- `valid_lo[f,i]`: validity of the lower computation (via upper on \(-\varphi\)) at point \(i\).

Do **not** couple these masks.

### Interval validity (only when producing an interval)
Per divergence \(f\), an interval at point \(i\) is valid iff:
- both endpoints exist and are finite, and
- it is non-inverted.

Define:
- `inverted_f[i] = isfinite(lower_f[i]) & isfinite(upper_f[i]) & (lower_f[i] > upper_f[i])`
- `valid_interval[f,i] = valid_lo[f,i] & valid_up[f,i] & isfinite(lower_f[i]) & isfinite(upper_f[i]) & (~inverted_f[i])`

If `valid_interval[f,i]` is false, the interval for divergence \(f\) at point \(i\) must be blanked (NaN endpoints in the interval view).

### Terminate-and-NaN rule
For each divergence \(f\), each point \(i\), and each bound computation:
- If the conjugate evaluation is invalid/out-of-domain or non-finite, terminate that point and set the bound value to NaN.
- This rule applies to pseudo-outcomes \(Z_i\), per-divergence bound vectors, and any intermediate step that can produce NaN/inf.

---

## Domain-safe conjugate interface (P0)

Every divergence must implement:
- `g_star_with_valid(t) -> (gstar, valid_mask)`

Definitions:
- \(t = (\varphi(Y)-u(A,X)) / \lambda(A,X)\)
- `valid_mask[i] = True` iff \(t_i\) is strictly inside the domain of \(g^\*\), with margin \(\varepsilon>0\).

### Loss safety (P0)
Training must not call raw `g_star(t)` on out-of-domain inputs if it can yield inf/NaN.
Instead:
- compute `(gstar, valid)` via `g_star_with_valid(t)`
- construct a finite batch loss using:
  - `gstar` on valid points
  - a finite surrogate/barrier + penalty on invalid points
- log in-domain fraction per epoch/fold.

Purpose: prevents training crashes and enables penalty/annealing to steer solutions into the valid domain.

---

## Annealing-default feasibility steering (P0)

Annealing is the **default** optimization technique for dual-net training.

### Default schedule (two-stage)
Let total epochs be \(E\).
Let \(E_1=\lceil \rho E\rceil\) be Stage-1 epochs with default \(\rho=0.3\).

Domain penalty weight:
\[
w_{\text{dom}}(e)=
\begin{cases}
w_1, & 0 \le e < E_1 \\
w_2, & E_1 \le e < E
\end{cases}
\quad \text{with } w_1 \gg w_2.
\]

Defaults:
- \(w_2 = 10^4\)
- \(w_1 = 10^6\)

Logging (required):
- `valid_frac_epoch` (or running average): in-domain fraction per epoch
- `valid_frac_stage1`, `valid_frac_stage2`
- number/fraction of invalid pseudo-outcomes excluded from m-fitting

---

## Estimation of m(a,x) from pseudo-outcomes (P0)

Pseudo-outcome:
\[
Z_i = g^\*(t_i).
\]

Rules:
1. Fit \(m\) using only samples where `valid=True` and `Z_i` is finite.
2. Enforce a minimum number of valid samples per action per fold (configurable).
3. If insufficient valid support exists, mark downstream predictions dependent on \(m\) as NaN (per divergence and per bound computation).

---

## Aggregation across divergences (P0) — Endpoint-wise aggregation (default)

This repository uses **endpoint-wise order-statistic aggregation** (a pragmatic extension) by default.
It aggregates upper and lower endpoints separately after filtering invalid candidates.

Notation at a fixed evaluation point \((a,x)\) (omitted below):
- per divergence \(f\): lower endpoint \(L_f\), upper endpoint \(U_f\)
- validity masks: `valid_lo[f]`, `valid_up[f]`

Define per divergence:
- `inverted_f = isfinite(L_f) & isfinite(U_f) & (L_f > U_f)`

### Upper aggregation
Define the candidate set:
\[
\mathcal U = \{U_f:\ \texttt{valid_up}[f]\ \&\ \texttt{isfinite}(U_f)\ \&\ \neg \texttt{inverted}_f\}.
\]

- `kth_upper(k)`: k-th **smallest** element of \(\mathcal U\).
- `tight_upper`: `kth_upper(1)` (i.e., \(\min \mathcal U\)).

If \(|\mathcal U| = 0\), set aggregated upper to NaN.

### Lower aggregation
Define the candidate set:
\[
\mathcal L = \{L_f:\ \texttt{valid_lo}[f]\ \&\ \texttt{isfinite}(L_f)\ \&\ \neg \texttt{inverted}_f\}.
\]

- `kth_lower(k)`: k-th **largest** element of \(\mathcal L\).
- `tight_lower`: `kth_lower(1)` (i.e., \(\max \mathcal L\)).

If \(|\mathcal L| = 0\), set aggregated lower to NaN.

### Final interval rule (required)
After computing \(L^{agg}\) and \(U^{agg}\):
- If either endpoint is NaN, the interval is blank (NaN endpoints).
- If \(L^{agg} > U^{agg}\), the interval is blank (NaN endpoints).

### Hard constraints
- Filtering is applied **before sorting/order-statistics**.
- NaN values must never be passed into sorting.

### Diagnostics (required)
Report at each point:
- `n_eff_up`: \(|\mathcal U|\)
- `n_eff_lo`: \(|\mathcal L|\)
- `k_used_up`: the chosen k for upper (default 1)
- `k_used_lo`: the chosen k for lower (default 1)
- counts by reason (per divergence, aggregated over f):
  - invalid upper (`valid_up=False`)
  - invalid lower (`valid_lo=False`)
  - non-finite endpoint
  - inverted interval filtered out

---

## Optimization configuration: batch size rule (user-specified)

Let n be the number of samples used in the dual-net training loop (per fold) and b the batch size:

\[
b(n) :=
\begin{cases}
16, & n \le 1000, \\
32, & 1000 < n \le 5000, \\
64, & 5000 < n \le 10000, \\
\min(128, \lfloor \sqrt{n} \rfloor), & n > 10000.
\end{cases}
\]

Hard constraints:
- \(1 \le b \le n\)
- must fit in memory
- do not increase b beyond this rule unless strictly required for hardware throughput.

Reference implementation:
```python
def choose_batch_size(n: int) -> int:
    if n <= 1000:
        return 16
    elif n <= 5000:
        return 32
    elif n <= 10000:
        return 64
    else:
        return min(128, int(n**0.5))
