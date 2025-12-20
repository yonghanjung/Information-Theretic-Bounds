"""
End-to-end example with extra diagnostics (optimized).

Key speedups vs the original run_example.py
------------------------------------------
1) Avoid redundant estimator fits:
   - The original script fits:
       (a) compute_causal_bounds(div)            -> 2 fits
       (b) fit_bounds_with_masks for 5 divs     -> 10 fits
       (c) compute_causal_bounds("combined")    -> 10 fits
     Total: 22 fits.
   - This optimized script fits base divergences once (10 fits total) and derives
     "combined" and "cluster" from those results.

2) Reuse cross-fitted propensity models across all fits (if using causal_bound_cached):
   - Prefit propensity cache once per dataset and reuse across divergences and phi/-phi.

3) Replace per-sample sklearn KMeans+silhouette cluster aggregation with a tiny
   deterministic 1D partition search (M=5 points), eliminating thousands of sklearn calls.

Outputs are kept compatible with the original script:
  - experiments/run_example_gstar_bounds_table.csv
  - experiments/run_example_gstar_bounds_any_invalid.csv
  - experiments/run_example_gstar_bounds_summary.csv

Run
---
From the project directory:
    python run_example_optimized.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

# Thread knobs kept off by default; flip to True if you hit BLAS/OpenMP issues on macOS.
from utils import apply_macos_thread_safety_knobs

# Must be called before importing numpy/sklearn/torch to affect BLAS/OpenMP.
apply_macos_thread_safety_knobs(enable=False)

import numpy as np
import pandas as pd
import torch

# Silence runtime warnings from synthetic data (overflow/invalid during matmul/logits).
warnings.filterwarnings("ignore", category=RuntimeWarning)

from data_generating import generate_data
from manski import empirical_extrema_manski_bounds

# Prefer cached estimator (propensity cache reuse). Fall back to baseline if unavailable.
from causal_bound import DebiasedCausalBoundEstimator, prefit_propensity_cache
_HAS_PROP_CACHE = True


# -------------------------
# Lightweight progress/timing utilities
# -------------------------
_STEP_STACK: list[dict[str, object]] = []


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, use_tqdm: bool = False) -> None:
    print(msg, flush=True)


class StepTimer:
    def __init__(self, name: str, use_tqdm: bool, enabled: bool = True) -> None:
        self.name = name
        self.use_tqdm = use_tqdm
        self.enabled = enabled

    def __enter__(self) -> "StepTimer":
        if not self.enabled:
            return self
        entry = {
            "name": self.name,
            "start_wall": time.perf_counter(),
            "start_cpu": time.process_time(),
            "use_tqdm": self.use_tqdm,
        }
        _STEP_STACK.append(entry)
        _log(f"[PROGRESS] START {self.name}", self.use_tqdm)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False
        if not _STEP_STACK:
            return False
        entry = _STEP_STACK[-1]
        if entry.get("name") != self.name:
            return False
        if exc_type is None:
            wall = time.perf_counter() - float(entry["start_wall"])
            cpu = time.process_time() - float(entry["start_cpu"])
            _log(f"[PROGRESS] END {self.name} | wall={wall:.3f}s cpu={cpu:.3f}s", self.use_tqdm)
            _STEP_STACK.pop()
        return False


def _log_active_step_error(use_tqdm: bool = False) -> None:
    if not _STEP_STACK:
        _log("[PROGRESS] ERROR during unknown step", use_tqdm)
        return
    entry = _STEP_STACK[-1]
    wall = time.perf_counter() - float(entry["start_wall"])
    cpu = time.process_time() - float(entry["start_cpu"])
    _log(
        f"[PROGRESS] ERROR during {entry['name']} | wall={wall:.3f}s cpu={cpu:.3f}s",
        bool(entry["use_tqdm"]),
    )
    _STEP_STACK.clear()


def _timing_enabled_from_argv(default: bool = True) -> bool:
    if "--no-timing" in sys.argv:
        return False
    if "--timing" in sys.argv:
        return True
    return default


def _timing_detail_from_argv(default: bool = False) -> bool:
    if "--timing_detail" in sys.argv or "--timing-detail" in sys.argv:
        return True
    return default


# -------------------------
# Phi helpers
# -------------------------
def phi_identity(y: torch.Tensor) -> torch.Tensor:
    """Identity transform phi(y)=y."""
    return y


def phi_neg(y: torch.Tensor) -> torch.Tensor:
    """Sign-flipped phi used to compute lower bounds via the identity."""
    return -y


# -------------------------
# g* validity masks (fold-aligned) without computing g* values
# -------------------------
@torch.no_grad()
def gstar_valid_mask_per_sample(
    est: DebiasedCausalBoundEstimator,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Compute per-sample validity mask for g*(t) while respecting cross-fitting folds.

    For each observation i in fold k, evaluate t using the fold-k (h,u) nets,
    and apply the divergence's validity mask to t.
    """
    n = int(X.shape[0])
    valid = np.zeros(n, dtype=bool)

    # Access internal fold split indices.
    for k, (_, fold_idx) in enumerate(est.splits_):
        Xk = torch.tensor(np.asarray(X[fold_idx], dtype=np.float32), dtype=torch.float32)
        Ak = torch.tensor(np.asarray(A[fold_idx], dtype=np.float32).reshape(-1), dtype=torch.float32)
        Yk = torch.tensor(np.asarray(Y[fold_idx], dtype=np.float32).reshape(-1), dtype=torch.float32)

        # Fold-specific dual nets
        h_net = est.h_nets_[k]
        u_net = est.u_nets_[k]

        # Concatenate [A, X]
        ax = torch.cat([Ak.reshape(-1, 1), Xk], dim=1)

        h = h_net(ax)
        h = torch.clamp(h, min=-est.dual_net_cfg.h_clip, max=est.dual_net_cfg.h_clip)
        lam = torch.exp(h)
        u = u_net(ax)

        t = (est.phi(Yk) - u) / lam

        div = est.divergence
        # FDivergence exposes an internal _valid_mask; use it to avoid computing g*(t).
        if hasattr(div, "_valid_mask"):
            mask = div._valid_mask(t)  # type: ignore[attr-defined]
        else:
            _, mask = div.g_star_with_valid(t)

        valid[np.asarray(fold_idx, dtype=int)] = mask.cpu().numpy().astype(bool)

    return valid


# -------------------------
# Robust c-wise intersection aggregation (same logic as causal_bound.compute_causal_bounds(..., "combined"))
# -------------------------
from itertools import combinations

def combined_cwise_intersection(lower_mat: np.ndarray, upper_mat: np.ndarray, c: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    lower_mat, upper_mat: shape (M, n)
    returns (lower_out, upper_out): shape (n,)
    """
    M, n = lower_mat.shape
    lower_out = np.full(n, np.nan, dtype=np.float32)
    upper_out = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        lowers = lower_mat[:, i]
        uppers = upper_mat[:, i]
        valid = np.isfinite(lowers) & np.isfinite(uppers) & (lowers <= uppers)

        if not np.any(valid):
            finite = np.isfinite(lowers) & np.isfinite(uppers)
            if np.any(finite):
                lo_f = float(np.min(lowers[finite]))
                hi_f = float(np.max(uppers[finite]))
                if lo_f > hi_f:
                    mid = 0.5 * (lo_f + hi_f)
                    lo_f, hi_f = mid, mid
                lower_out[i] = np.float32(lo_f)
                upper_out[i] = np.float32(hi_f)
            else:
                lower_out[i] = np.float32(0.0)
                upper_out[i] = np.float32(0.0)
            continue

        lowers_v = lowers[valid]
        uppers_v = uppers[valid]
        if len(lowers_v) < 2:
            lower_out[i] = np.float32(np.min(lowers_v))
            upper_out[i] = np.float32(np.max(uppers_v))
            continue

        best_subset = None
        best_width = np.inf
        best_sum = np.inf
        best_L = np.nan
        best_U = np.nan

        max_t = min(max(c, 2), len(lowers_v))
        for t in range(max_t, 1, -1):
            found_any = False
            for combo in combinations(range(len(lowers_v)), t):
                idx = list(combo)
                L_int = float(np.max(lowers_v[idx]))
                U_int = float(np.min(uppers_v[idx]))
                width = U_int - L_int
                if width < 0:
                    continue
                sum_widths = float(np.sum(uppers_v[idx] - lowers_v[idx]))
                if (
                    (width < best_width)
                    or (np.isclose(width, best_width) and sum_widths < best_sum)
                    or (
                        np.isclose(width, best_width)
                        and np.isclose(sum_widths, best_sum)
                        and (best_subset is None or combo < best_subset)
                    )
                ):
                    best_subset = combo
                    best_width = width
                    best_sum = sum_widths
                    best_L = L_int
                    best_U = U_int
                found_any = True
            if found_any and best_subset is not None:
                break

        if best_subset is not None:
            lower_out[i] = np.float32(best_L)
            upper_out[i] = np.float32(best_U)
        else:
            lower_out[i] = np.float32(np.min(lowers_v))
            upper_out[i] = np.float32(np.max(uppers_v))

    return lower_out, upper_out


# -------------------------
# Fast 1D cluster aggregation for M small (here M=5 base divergences)
# -------------------------
def _median(vals: list[float]) -> float:
    arr = sorted(vals)
    m = len(arr)
    if m % 2 == 1:
        return float(arr[m // 2])
    return 0.5 * (arr[m // 2 - 1] + arr[m // 2])


def _second_smallest(vals: list[float]) -> float:
    arr = sorted(vals)
    return float(arr[1]) if len(arr) >= 2 else float(arr[0])


def _second_largest(vals: list[float]) -> float:
    arr = sorted(vals)
    return float(arr[-2]) if len(arr) >= 2 else float(arr[-1])


def _silhouette_score_1d(values_sorted: list[float], labels_sorted: list[int]) -> float:
    """
    Exact silhouette score for 1D points with absolute-distance metric.

    Notes
    -----
    - We assume `values_sorted` and `labels_sorted` are aligned (same order).
    - For singleton clusters, we set s(i)=0 (a common convention).
    """
    x = np.asarray(values_sorted, dtype=np.float64)
    lab = np.asarray(labels_sorted, dtype=int)
    n = x.shape[0]
    uniq = np.unique(lab)
    if n <= 1 or len(uniq) < 2 or len(uniq) >= n:
        return float("-inf")

    # Pairwise distances (n is tiny here; n=5 for base divergences)
    D = np.abs(x.reshape(-1, 1) - x.reshape(1, -1))

    s = np.zeros(n, dtype=np.float64)
    for i in range(n):
        li = lab[i]
        same = (lab == li)
        same[i] = False
        if not np.any(same):
            s[i] = 0.0
            continue

        a = float(np.mean(D[i, same]))

        b = float("inf")
        for lj in uniq:
            if lj == li:
                continue
            other = (lab == lj)
            if np.any(other):
                b = min(b, float(np.mean(D[i, other])))

        denom = max(a, b)
        s[i] = 0.0 if denom <= 0.0 else (b - a) / denom

    return float(np.mean(s))


def _sse_cluster_1d(cluster_vals: list[float]) -> float:
    if len(cluster_vals) == 0:
        return 0.0
    mu = float(np.mean(cluster_vals))
    return float(np.sum((np.asarray(cluster_vals, dtype=np.float64) - mu) ** 2))


def _best_kmeans_partition_1d(values: list[float], k: int) -> tuple[list[list[float]], list[int], list[float]]:
    """
    Deterministic global optimum of 1D k-means (min SSE), restricted to contiguous clusters
    in sorted order (which is optimal in 1D).

    Returns
    -------
    clusters_sorted : list of clusters, each a list of values, in sorted order
    labels_sorted   : list[int] labels aligned with values_sorted
    values_sorted   : sorted values
    """
    m = len(values)
    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(k, m)

    order = np.argsort(values)
    xs = [float(values[i]) for i in order]

    if k == 1:
        return [xs], [0] * m, xs

    best_sse = float("inf")
    best_clusters = None
    best_labels = None

    # Enumerate all contiguous partitions of sorted xs into k non-empty clusters.
    from itertools import combinations as it_combinations
    for cuts in it_combinations(range(1, m), k - 1):
        cuts = (0,) + cuts + (m,)
        clusters = [xs[cuts[j]:cuts[j + 1]] for j in range(k)]
        sse = sum(_sse_cluster_1d(c) for c in clusters)
        if sse < best_sse:
            best_sse = sse
            best_clusters = clusters
            labels = []
            for j, c in enumerate(clusters):
                labels.extend([j] * len(c))
            best_labels = labels

    assert best_clusters is not None and best_labels is not None
    return best_clusters, best_labels, xs


_SELECT_K_CACHE: dict[tuple[tuple[float, ...], tuple[int, ...], float], tuple[int, tuple[tuple[float, ...], ...]]] = {}


def _select_k_and_clusters_1d(
    values: list[float],
    k_candidates: tuple[int, ...] = (2, 3, 4),
    penalty_singleton: float = 0.2,
) -> tuple[int, list[list[float]]]:
    """
    Mimic the original rule:
      - For each k, run (1D) k-means -> get clusters
      - score = silhouette - penalty_singleton*(#singleton clusters)
      - choose best k
    Here we compute the global-optimum 1D k-means partition by brute-force (m is tiny).
    """
    values_key = tuple(float(v) for v in values)
    cache_key = (values_key, k_candidates, float(penalty_singleton))
    cached = _SELECT_K_CACHE.get(cache_key)
    if cached is not None:
        best_k, clusters = cached
        return int(best_k), [list(c) for c in clusters]

    m = len(values)
    if m == 0:
        return 1, [[]]
    best_k = min(2, m)
    best_score = float("-inf")
    best_clusters = None

    for k in k_candidates:
        if k > m:
            continue
        clusters_k, labels_k, xs = _best_kmeans_partition_1d(values, k=k)

        # Original guard: require at least one eligible cluster (size>=2)
        if not any(len(c) >= 2 for c in clusters_k):
            continue

        sep = _silhouette_score_1d(xs, labels_k)
        num_singletons = sum(1 for c in clusters_k if len(c) == 1)
        score = sep - penalty_singleton * num_singletons
        if score > best_score:
            best_score = score
            best_k = k
            best_clusters = clusters_k

    if best_clusters is None:
        # Fallback: use best_k=min(2,m) and the corresponding min-SSE partition.
        best_clusters, _, _ = _best_kmeans_partition_1d(values, k=best_k)

    _SELECT_K_CACHE[cache_key] = (int(best_k), tuple(tuple(c) for c in best_clusters))
    return int(best_k), best_clusters


def _cluster_choose_lower(clusters: list[list[float]]) -> float | None:
    """
    Lower rule (matches original):
    choose eligible cluster (size>=2) with largest median, tie by size then max,
    and return the max of that cluster.
    """
    eligible = [c for c in clusters if len(c) >= 2]
    if not eligible:
        return None
    med = [_median(c) for c in eligible]
    best_idx = int(np.argmax(med))
    best_med = med[best_idx]
    tied_idx = [i for i, m in enumerate(med) if np.isclose(m, best_med)]
    if len(tied_idx) > 1:
        sizes = [len(eligible[i]) for i in tied_idx]
        best_idx = tied_idx[int(np.argmax(sizes))]
        tied_same_size = [i for i in tied_idx if len(eligible[i]) == len(eligible[best_idx])]
        if len(tied_same_size) > 1:
            max_vals = [max(eligible[i]) for i in tied_same_size]
            best_idx = tied_same_size[int(np.argmax(max_vals))]
    return float(max(eligible[best_idx]))


def _cluster_choose_upper(clusters: list[list[float]]) -> float | None:
    """
    Upper rule (matches original):
    choose eligible cluster (size>=2) with smallest median, tie by size then min,
    and return the min of that cluster.
    """
    eligible = [c for c in clusters if len(c) >= 2]
    if not eligible:
        return None
    med = [_median(c) for c in eligible]
    best_idx = int(np.argmin(med))
    best_med = med[best_idx]
    tied_idx = [i for i, m in enumerate(med) if np.isclose(m, best_med)]
    if len(tied_idx) > 1:
        sizes = [len(eligible[i]) for i in tied_idx]
        best_idx = tied_idx[int(np.argmax(sizes))]
        tied_same_size = [i for i in tied_idx if len(eligible[i]) == len(eligible[best_idx])]
        if len(tied_same_size) > 1:
            min_vals = [min(eligible[i]) for i in tied_same_size]
            best_idx = tied_same_size[int(np.argmin(min_vals))]
    return float(min(eligible[best_idx]))


def cluster_bounds_fast(
    lowers: list[float],
    uppers: list[float],
    k_candidates: tuple[int, ...] = (2, 3, 4),
    penalty_singleton: float = 0.2,
) -> tuple[float, float, int, int]:
    """
    Fast deterministic replacement for the sklearn KMeans+silhouette cluster rule.

    This implements:
      - select k via silhouette - penalty_singleton*(#singletons)
      - clusters computed as the global-optimum 1D k-means partition (min SSE)
      - then apply the same median-based rules to pick the final lower/upper.
    """
    kL, clusters_L = _select_k_and_clusters_1d(lowers, k_candidates=k_candidates, penalty_singleton=penalty_singleton)
    kU, clusters_U = _select_k_and_clusters_1d(uppers, k_candidates=k_candidates, penalty_singleton=penalty_singleton)

    lower_c = _cluster_choose_lower(clusters_L)
    upper_c = _cluster_choose_upper(clusters_U)

    if lower_c is None:
        lower_c = _second_smallest(lowers)
    if upper_c is None:
        upper_c = _second_largest(uppers)

    if lower_c > upper_c:
        lower_c = _second_smallest(lowers)
        upper_c = _second_largest(uppers)

    return float(lower_c), float(upper_c), int(kL), int(kU)


# -------------------------
# Fit bounds (+ g* validity masks) for one divergence, using propensity cache if available
# -------------------------
def _fit_with_optional_prop_cache(
    est: DebiasedCausalBoundEstimator,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    propensity_cache: dict | None,
) -> DebiasedCausalBoundEstimator:
    """
    Call est.fit(..., propensity_cache=...) if supported; otherwise fall back to est.fit(X,A,Y).
    """
    if propensity_cache is None:
        return est.fit(X, A, Y)
    try:
        return est.fit(X, A, Y, propensity_cache=propensity_cache)  # type: ignore[arg-type]
    except TypeError:
        return est.fit(X, A, Y)


def fit_bounds_with_masks_fast(
    div_name: str,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    dual_net_config: dict,
    fit_config: dict,
    seed: int,
    propensity_model: str,
    m_model: str,
    propensity_cache: dict | None,
    timing: bool = False,
    timing_detail: bool = False,
) -> dict:
    """
    Fit upper and lower bounds for a single divergence and return bounds + g* validity masks.
    """
    with StepTimer(
        f"fit {div_name} upper (+phi)",
        use_tqdm=False,
        enabled=timing and timing_detail,
    ):
        est_upper = DebiasedCausalBoundEstimator(
            divergence=div_name,
            phi=phi_identity,
            propensity_model=propensity_model,
            m_model=m_model,
            dual_net_config=dual_net_config,
            fit_config=fit_config,
            seed=seed,
        )
        est_upper = _fit_with_optional_prop_cache(est_upper, X, A, Y, propensity_cache=propensity_cache)
        upper = est_upper.predict_bound(a=1, X=X).astype(np.float32)
        mask_upper = gstar_valid_mask_per_sample(est_upper, X, A, Y)

    with StepTimer(
        f"fit {div_name} lower (-phi)",
        use_tqdm=False,
        enabled=timing and timing_detail,
    ):
        est_lower = DebiasedCausalBoundEstimator(
            divergence=div_name,
            phi=phi_neg,
            propensity_model=propensity_model,
            m_model=m_model,
            dual_net_config=dual_net_config,
            fit_config=fit_config,
            seed=seed,
        )
        est_lower = _fit_with_optional_prop_cache(est_lower, X, A, Y, propensity_cache=propensity_cache)
        upper_neg = est_lower.predict_bound(a=1, X=X).astype(np.float32)
        lower = (-upper_neg).astype(np.float32)
        mask_lower = gstar_valid_mask_per_sample(est_lower, X, A, Y)

    return {
        "upper": upper,
        "lower": lower,
        "mask_upper": mask_upper,
        "mask_lower": mask_lower,
    }

def main() -> None:
    timing_enabled = _timing_enabled_from_argv(default=True)
    timing_detail = _timing_detail_from_argv(default=False)

    with StepTimer("configure experiment", use_tqdm=False, enabled=timing_enabled):
        # Keep defaults identical to the original run_example.py
        seed = 190602
        n = 10000
        d = 10
        div = "KL"  # KL, TV, Hellinger, Chi2, JS, combined, cluster
        structural_type = "cyclic2"

        dual_net_config = {
            "hidden_sizes": (64, 64),
            "activation": "relu",
            "dropout": 0.0,
            "h_clip": 20.0,
            "device": "cpu",
        }

        # Core estimator models (propensity + pseudo-outcome regressor).
        propensity_model = "xgboost"
        m_model = "xgboost"
        # Manski uses simpler, fast models to keep compute cheap and lightweight.
        manski_propensity_model = "logistic"
        manski_outcome_model = "random_forest"

        fit_config = {
            "n_folds": 3,
            "num_epochs": 200,
            "batch_size": 256,
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "max_grad_norm": 10.0,
            "eps_propensity": 1e-3,
            "deterministic_torch": True,
            "train_m_on_fold": True,
            "propensity_config": {
                "n_estimators": 300,
                "max_depth": 10,
                "learning_rate": 0.005,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "min_child_weight": 1.0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_jobs": -1,
                "verbosity": 0,
            },
            "m_config": {
                "n_estimators": 400,
                "max_depth": 10,
                "learning_rate": 0.005,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "min_child_weight": 1.0,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "n_jobs": -1,
                "verbosity": 0,
            },
            "verbose": False,
            "log_every": 10,
        }

        manski_propensity_config = {
            "C": 1.0,
            "max_iter": 2000,
            "penalty": "l2",
            "solver": "lbfgs",
            "n_jobs": 1,
        }
        manski_outcome_config = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "n_jobs": 1,
        }

    with StepTimer("data generation", use_tqdm=False, enabled=timing_enabled):
        data = generate_data(n=n, d=d, seed=seed, structural_type=structural_type)
        keep_cols = None  # choose feature subset; set to None to use all
        full_X = data["X"]
        X = full_X[:, keep_cols] if keep_cols is not None else full_X
        A = data["A"]
        Y = data["Y"]

        if keep_cols is None:
            GroundTruth = data["GroundTruth"]
        else:
            truth_fn = data["GroundTruth"]

            def GroundTruth(a: int, X_query: np.ndarray) -> np.ndarray:
                Xq = np.asarray(X_query, dtype=np.float32)
                if Xq.shape[0] != full_X.shape[0]:
                    raise ValueError(f"GroundTruth expects n={full_X.shape[0]} rows; got {Xq.shape[0]}.")
                return truth_fn(a, full_X)

    with StepTimer("fit propensity cache", use_tqdm=False, enabled=timing_enabled):
        # Prefit propensity cache (big speed win across divergences and phi/-phi).
        propensity_cache = None
        if _HAS_PROP_CACHE and prefit_propensity_cache is not None:
            propensity_cache = prefit_propensity_cache(
                X=X,
                A=A,
                propensity_model=propensity_model,
                propensity_config=fit_config["propensity_config"],
                n_folds=fit_config["n_folds"],
                seed=seed,
                eps_propensity=fit_config["eps_propensity"],
            )
            ehat1_oof = np.asarray(propensity_cache["e1_oof"], dtype=np.float32)
        else:
            ehat1_oof = None  # will be available only after fitting an estimator

    with StepTimer("fit base divergence bounds", use_tqdm=False, enabled=timing_enabled):
        # Fit base divergences once (and only once).
        base_divs = ["KL", "TV", "Hellinger", "Chi2", "JS"]
        truth_do1 = np.asarray(GroundTruth(1, X), dtype=np.float32).reshape(-1)

        results: dict[str, object] = {
            "i": np.arange(X.shape[0], dtype=int),
            "truth_do1": truth_do1,
        }

        lower_stack = []
        upper_stack = []

        for div_name in base_divs:
            with StepTimer(f"fit bounds div={div_name}", use_tqdm=False, enabled=timing_enabled):
                stats = fit_bounds_with_masks_fast(
                    div_name=div_name,
                    X=X,
                    A=A,
                    Y=Y,
                    dual_net_config=dual_net_config,
                    fit_config=fit_config,
                    seed=seed,
                    propensity_model=propensity_model,
                    m_model=m_model,
                    propensity_cache=propensity_cache,
                    timing=timing_enabled,
                    timing_detail=timing_detail,
                )
            results[f"lower_{div_name}"] = stats["lower"]
            results[f"valid_gstar_lower_{div_name}"] = stats["mask_lower"]
            results[f"upper_{div_name}"] = stats["upper"]
            results[f"valid_gstar_upper_{div_name}"] = stats["mask_upper"]

            lower_stack.append(stats["lower"])
            upper_stack.append(stats["upper"])

            # Get ehat1_oof if we couldn't prefit it.
            if ehat1_oof is None:
                try:
                    # Pick from the last fitted estimator implicitly via cache absence:
                    # safer to not rely on internals here.
                    pass
                except Exception:
                    pass

        lower_mat = np.vstack([np.asarray(x, dtype=np.float32) for x in lower_stack])
        upper_mat = np.vstack([np.asarray(x, dtype=np.float32) for x in upper_stack])

    with StepTimer("aggregate combined bounds", use_tqdm=False, enabled=timing_enabled):
        # Combined robust interval (intersection across divergences).
        lower_combined, upper_combined = combined_cwise_intersection(lower_mat, upper_mat, c=3)
        results["lower_combined"] = lower_combined.astype(np.float32)
        results["upper_combined"] = upper_combined.astype(np.float32)

    with StepTimer("compute Manski bounds", use_tqdm=False, enabled=timing_enabled):
        # Empirical-extrema Manski heuristic.
        manski_res = empirical_extrema_manski_bounds(
            Y=Y,
            A=A,
            X=X,
            a=1,
            propensity_model=manski_propensity_model,
            propensity_config=manski_propensity_config,
            outcome_model=manski_outcome_model,
            outcome_config=manski_outcome_config,
            seed=seed,
            eps_propensity=fit_config["eps_propensity"],
        )
        manski_lower = np.asarray(manski_res["lower"], dtype=np.float32)
        manski_upper = np.asarray(manski_res["upper"], dtype=np.float32)
        results["lower_Manski"] = manski_lower
        results["upper_Manski"] = manski_upper

    with StepTimer("aggregate cluster bounds", use_tqdm=False, enabled=timing_enabled):
        # Cluster-based aggregator over divergence bounds (auto-k on {2,3,4}) using fast 1D partition search.
        cluster_lower = np.empty(X.shape[0], dtype=np.float32)
        cluster_upper = np.empty(X.shape[0], dtype=np.float32)
        cluster_kL = np.empty(X.shape[0], dtype=int)
        cluster_kU = np.empty(X.shape[0], dtype=int)
        penalty_singleton = 0.2
        k_candidates = (2, 3, 4)

        cluster_cache: dict[
            tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...], float],
            tuple[float, float, int, int],
        ] = {}
        for i in range(X.shape[0]):
            lowers_key = tuple(float(v) for v in lower_mat[:, i])
            uppers_key = tuple(float(v) for v in upper_mat[:, i])
            cache_key = (lowers_key, uppers_key, k_candidates, penalty_singleton)
            cached = cluster_cache.get(cache_key)
            if cached is None:
                lc, uc, kL, kU = cluster_bounds_fast(
                    lowers=list(lowers_key),
                    uppers=list(uppers_key),
                    k_candidates=k_candidates,
                    penalty_singleton=penalty_singleton,
                )
                cluster_cache[cache_key] = (lc, uc, kL, kU)
            else:
                lc, uc, kL, kU = cached
            cluster_lower[i] = np.float32(lc)
            cluster_upper[i] = np.float32(uc)
            cluster_kL[i] = int(kL)
            cluster_kU[i] = int(kU)

        results["lower_cluster"] = cluster_lower
        results["upper_cluster"] = cluster_upper
        results["k_cluster_lower"] = cluster_kL
        results["k_cluster_upper"] = cluster_kU

    with StepTimer("compute metrics", use_tqdm=False, enabled=timing_enabled):
        # Basic sanity checks (mirror the original checks, but on computed outputs).
        for key in ["lower_combined", "upper_combined", "lower_cluster", "upper_cluster", "lower_Manski", "upper_Manski"]:
            arr = np.asarray(results[key], dtype=np.float32)
            assert np.isfinite(arr).all(), f"{key} has non-finite values"

        # If we have ehat1_oof from cache, validate the clipping range.
        if ehat1_oof is not None:
            assert (
                (ehat1_oof >= fit_config["eps_propensity"] - 1e-8)
                & (ehat1_oof <= 1 - fit_config["eps_propensity"] + 1e-8)
            ).all(), "propensity outside clipping range"

        # Print width/coverage for the user-selected divergence
        div_key = div.strip()
        if div_key in base_divs:
            lo = np.asarray(results[f"lower_{div_key}"], dtype=np.float32)
            up = np.asarray(results[f"upper_{div_key}"], dtype=np.float32)
        elif div_key.lower() == "combined":
            lo = np.asarray(results["lower_combined"], dtype=np.float32)
            up = np.asarray(results["upper_combined"], dtype=np.float32)
        elif div_key.lower() == "cluster":
            lo = np.asarray(results["lower_cluster"], dtype=np.float32)
            up = np.asarray(results["upper_cluster"], dtype=np.float32)
        else:
            raise ValueError(f"Unknown div='{div}'. Choose one of {base_divs + ['combined','cluster']}.")

        width = float(np.mean(up - lo))
        cover = float(np.mean((truth_do1 >= lo) & (truth_do1 <= up)))
        print(f"divergence={div_key:>8} | mean width={width:.4f} | coverage={cover:.3f}")

        # Build detailed table for conjecture: bounds + g* validity per divergence.
        ordered_cols = [
            "i",
            "truth_do1",
            "lower_KL", "valid_gstar_lower_KL",
            "lower_TV", "valid_gstar_lower_TV",
            "lower_Hellinger", "valid_gstar_lower_Hellinger",
            "lower_Chi2", "valid_gstar_lower_Chi2",
            "lower_JS", "valid_gstar_lower_JS",
            "upper_KL", "valid_gstar_upper_KL",
            "upper_TV", "valid_gstar_upper_TV",
            "upper_Hellinger", "valid_gstar_upper_Hellinger",
            "upper_Chi2", "valid_gstar_upper_Chi2",
            "upper_JS", "valid_gstar_upper_JS",
            "lower_combined",
            "upper_combined",
            "lower_Manski",
            "upper_Manski",
            "lower_cluster",
            "upper_cluster",
        ]

        table_df = pd.DataFrame({col: results[col] for col in ordered_cols})

        # Identify samples where any divergence had an invalid g* on lower or upper.
        any_invalid = np.zeros(X.shape[0], dtype=bool)
        for dv in base_divs:
            any_invalid |= (~table_df[f"valid_gstar_lower_{dv}"]) | (~table_df[f"valid_gstar_upper_{dv}"])
        table_df["any_invalid_gstar"] = any_invalid

        # Per-divergence coverage flags: truth within [lower, upper].
        for dv in base_divs:
            table_df[f"coverage_{dv}"] = (
                (table_df[f"lower_{dv}"] <= table_df["truth_do1"])
                & (table_df["truth_do1"] <= table_df[f"upper_{dv}"])
            )
        table_df["coverage_combined"] = (
            (table_df["lower_combined"] <= table_df["truth_do1"])
            & (table_df["truth_do1"] <= table_df["upper_combined"])
        )
        table_df["coverage_Manski"] = (
            (table_df["lower_Manski"] <= table_df["truth_do1"])
            & (table_df["truth_do1"] <= table_df["upper_Manski"])
        )
        table_df["coverage_cluster"] = (
            (table_df["lower_cluster"] <= table_df["truth_do1"])
            & (table_df["truth_do1"] <= table_df["upper_cluster"])
        )

        summary_rows = []
        for dv in base_divs:
            widths = table_df[f"upper_{dv}"] - table_df[f"lower_{dv}"]
            summary_rows.append(
                {
                    "divergence": dv,
                    "coverage_rate": float(table_df[f"coverage_{dv}"].mean()),
                    "mean_width": float(widths.mean()),
                }
            )
        widths_combined = table_df["upper_combined"] - table_df["lower_combined"]
        summary_rows.append(
            {
                "divergence": "combined",
                "coverage_rate": float(table_df["coverage_combined"].mean()),
                "mean_width": float(widths_combined.mean()),
            }
        )
        widths_manski = table_df["upper_Manski"] - table_df["lower_Manski"]
        summary_rows.append(
            {
                "divergence": "Manski_empirical",
                "coverage_rate": float(table_df["coverage_Manski"].mean()),
                "mean_width": float(widths_manski.mean()),
            }
        )
        widths_cluster = table_df["upper_cluster"] - table_df["lower_cluster"]
        summary_rows.append(
            {
                "divergence": "cluster",
                "coverage_rate": float(table_df["coverage_cluster"].mean()),
                "mean_width": float(widths_cluster.mean()),
            }
        )
        summary_df = pd.DataFrame(summary_rows)

        # Reorder columns for clarity before saving (same as original)
        final_columns = [
            "i",
            "truth_do1",
            "lower_KL", "valid_gstar_lower_KL",
            "lower_TV", "valid_gstar_lower_TV",
            "lower_Hellinger", "valid_gstar_lower_Hellinger",
            "lower_Chi2", "valid_gstar_lower_Chi2",
            "lower_JS", "valid_gstar_lower_JS",
            "upper_KL", "valid_gstar_upper_KL",
            "upper_TV", "valid_gstar_upper_TV",
            "upper_Hellinger", "valid_gstar_upper_Hellinger",
            "upper_Chi2", "valid_gstar_upper_Chi2",
            "upper_JS", "valid_gstar_upper_JS",
            "lower_combined", "upper_combined",
            "lower_Manski", "upper_Manski",
            "lower_cluster", "upper_cluster",
            "coverage_KL", "coverage_TV", "coverage_Hellinger", "coverage_Chi2", "coverage_JS",
            "coverage_combined", "coverage_Manski", "coverage_cluster",
            "any_invalid_gstar",
        ]
        table_df = table_df[final_columns]

        invalid_cols = [
            "i",
            "truth_do1",
            "lower_KL", "upper_KL",
            "lower_TV", "upper_TV",
            "lower_Hellinger", "upper_Hellinger",
            "lower_Chi2", "upper_Chi2",
            "lower_JS", "upper_JS",
            "lower_cluster", "upper_cluster",
            "lower_combined", "upper_combined",
            "lower_Manski", "upper_Manski",
            "valid_gstar_lower_KL", "valid_gstar_upper_KL",
            "valid_gstar_lower_TV", "valid_gstar_upper_TV",
            "valid_gstar_lower_Hellinger", "valid_gstar_upper_Hellinger",
            "valid_gstar_lower_Chi2", "valid_gstar_upper_Chi2",
            "valid_gstar_lower_JS", "valid_gstar_upper_JS",
            "any_invalid_gstar",
        ]

    with StepTimer("save outputs", use_tqdm=False, enabled=timing_enabled):
        os.makedirs("experiments", exist_ok=True)

        invalid_df = table_df.loc[table_df["any_invalid_gstar"], invalid_cols].reset_index(drop=True)
        invalid_df.to_csv("experiments/run_example_gstar_bounds_any_invalid.csv", index=False)
        print(f"Found {len(invalid_df)} samples with any invalid g*; saved to run_example_gstar_bounds_any_invalid.csv")
        if len(invalid_df) > 0:
            print(invalid_df.head())

        table_df.to_csv("experiments/run_example_gstar_bounds_table.csv", index=False)
        print("Saved g* validity table to experiments/run_example_gstar_bounds_table.csv")
        print(table_df.head())

        summary_df.to_csv("experiments/run_example_gstar_bounds_summary.csv", index=False)
        print("Coverage/width summary by divergence:")
        print(summary_df)

        # print(table_df[["i","lower_cluster","truth_do1","upper_cluster","lower_combined","truth_do1","upper_combined"]])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log_active_step_error(use_tqdm=False)
        raise
