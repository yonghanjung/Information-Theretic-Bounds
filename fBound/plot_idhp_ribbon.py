"""
Monte Carlo ribbon plot of causal bounds vs X0 using a fixed IHDP evaluation set.

Uses DebiasedCausalBoundEstimator with propensity caching and the sign-flip trick
to compute lower/upper bounds for E[Y | do(A=1), X] at a shared evaluation set X_eval.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **k: x  # type: ignore

from causal_bound import DebiasedCausalBoundEstimator, prefit_propensity_cache

from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

# -------------------------
# Lightweight progress/timing utilities
# -------------------------
_STEP_STACK: list[dict[str, object]] = []


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, use_tqdm: bool) -> None:
    if use_tqdm and hasattr(tqdm, "write"):
        tqdm.write(msg)
    else:
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


def _log_active_step_error(use_tqdm: bool) -> None:
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


CANDIDATE_FILES = ["ihdp_npci_1.csv"]
COLUMN_NAMES = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
LEGACY_COLUMNS = {"t": "treatment", "yf": "y_factual", "ycf": "y_cfactual"}
DEFAULT_D = 10


def load_ihdp_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if set(COLUMN_NAMES).issubset(df.columns):
        return df
    if set(LEGACY_COLUMNS).issubset(df.columns):
        return df.rename(columns=LEGACY_COLUMNS)
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


def _resolve_data_path(data_path: Optional[str]) -> str:
    if data_path:
        return data_path
    return next((path for path in CANDIDATE_FILES if Path(path).exists()), CANDIDATE_FILES[0])


def prepare_ihdp_arrays(df: pd.DataFrame, d: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_full = df[[f"x{i}" for i in range(1, 26)]].to_numpy(dtype=np.float32)
    if d is not None:
        if d < 1 or d > X_full.shape[1]:
            raise ValueError(f"d must be in [1, {X_full.shape[1]}], got d={d}.")
        X = X_full[:, :d]
    else:
        X = X_full
    A = df["treatment"].to_numpy(dtype=np.float32).reshape(-1)
    Y = df["y_factual"].to_numpy(dtype=np.float32).reshape(-1)
    groundtruth = df["mu1"].to_numpy(dtype=np.float32).reshape(-1)
    return A, Y, groundtruth, X

# -------------------------
# Phi definitions
# -------------------------
def phi_identity(y: torch.Tensor) -> torch.Tensor:
    return y


def phi_neg(y: torch.Tensor) -> torch.Tensor:
    return -y


# -------------------------
# Combined (c-wise) intersection aggregator
# -------------------------
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
                L_int = float(np.max(lowers_v[list(combo)]))
                U_int = float(np.min(uppers_v[list(combo)]))
                width = U_int - L_int
                if width < 0:
                    continue
                sum_widths = float(np.sum(uppers_v[list(combo)] - lowers_v[list(combo)]))

                if (
                    (width < best_width)
                    or (np.isclose(width, best_width) and sum_widths < best_sum)
                    or (np.isclose(width, best_width) and np.isclose(sum_widths, best_sum) and (best_subset is None or combo < best_subset))
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
# Kth aggregator (order-statistic bounds)
# -------------------------
def kth(lower_values, upper_values, k):
    lowers = np.asarray(lower_values, dtype=np.float64).reshape(-1)
    uppers = np.asarray(upper_values, dtype=np.float64).reshape(-1)

    if lowers.size != uppers.size:
        raise ValueError("lower_values and upper_values must have same length.")
    n = int(lowers.size)
    if n == 0:
        return float("nan"), float("nan")

    # clamp k into [1, n]
    k = int(k)
    if k < 1:
        k = 1
    if k > n:
        k = n

    # Precompute sorted order-statistics once (O(n log n)).
    # This is simpler and avoids repeated partition work in recursion.
    lowers_sorted = np.sort(lowers)
    uppers_sorted = np.sort(uppers)

    while k >= 1:
        l_k = float(lowers_sorted[k - 1])       # k-th smallest
        u_k = float(uppers_sorted[n - k])       # (n-k+1)-th smallest

        if l_k <= u_k:
            return l_k, u_k
        k -= 1

    # No feasible k found
    return float("nan"), float("nan")


def tight_kth(lower_values, upper_values, k=None):
    lowers = np.asarray(lower_values, dtype=np.float64).reshape(-1)
    uppers = np.asarray(upper_values, dtype=np.float64).reshape(-1)
    if lowers.size != uppers.size:
        raise ValueError("tight_kth requires lower_values and upper_values to have same length.")
    n = int(lowers.size)
    if n == 0:
        return float("nan"), float("nan")
    if k is None:
        k = n
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > n:
        k = n

    last = (float("nan"), float("nan"))
    while k >= 1:
        l_k, u_k = kth(lowers, uppers, k)
        last = (l_k, u_k)
        if l_k <= u_k:
            break
        k -= 1
    return last


# -------------------------
# Cluster aggregator (fast 1D partition search, no sklearn)
# -------------------------
def _median(vals):
    arr = np.sort(np.asarray(vals, dtype=np.float64))
    n = int(arr.shape[0])
    if n % 2 == 1:
        return float(arr[n // 2])
    return float(0.5 * (arr[n // 2 - 1] + arr[n // 2]))


def _silhouette_score_1d(v_sorted: np.ndarray, labels: np.ndarray) -> float:
    v = v_sorted.astype(np.float64, copy=False).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    m = int(v.shape[0])
    uniq = np.unique(labels)
    if uniq.size < 2:
        return -np.inf

    D = np.abs(v.reshape(-1, 1) - v.reshape(1, -1))
    sil = np.zeros(m, dtype=np.float64)
    for i in range(m):
        li = labels[i]
        in_same = np.where(labels == li)[0]
        if in_same.size <= 1:
            sil[i] = 0.0
            continue
        a_i = float(np.mean(D[i, in_same[in_same != i]]))
        b_i = np.inf
        for lj in uniq:
            if lj == li:
                continue
            in_other = np.where(labels == lj)[0]
            if in_other.size == 0:
                continue
            b_i = min(b_i, float(np.mean(D[i, in_other])))
        denom = max(a_i, b_i)
        sil[i] = 0.0 if denom <= 0 else (b_i - a_i) / denom
    return float(np.mean(sil))


def _partition_labels_from_cuts(m: int, cuts: tuple[int, ...]) -> np.ndarray:
    labels = np.empty(m, dtype=int)
    start = 0
    lab = 0
    for c in cuts + (m,):
        labels[start:c] = lab
        start = c
        lab += 1
    return labels


def _sse_from_labels(v_sorted: np.ndarray, labels: np.ndarray) -> float:
    v = v_sorted.astype(np.float64, copy=False)
    sse = 0.0
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue
        vv = v[idx]
        mu = float(np.mean(vv))
        sse += float(np.sum((vv - mu) ** 2))
    return float(sse)


def _best_cluster_partition_1d(values, k_candidates=(2, 3, 4), penalty_singleton=0.2):
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    m = int(v.shape[0])
    if m == 0:
        return []
    if m == 1:
        return [[float(v[0])]]

    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]

    best_score = -np.inf
    best_labels = None
    best_k = None
    best_sse = np.inf
    best_cuts = None

    for k in k_candidates:
        if k < 2 or k > m:
            continue
        for cuts in combinations(tuple(range(1, m)), k - 1):
            labels = _partition_labels_from_cuts(m, cuts)
            sizes = np.bincount(labels, minlength=k)
            if not np.any(sizes >= 2):
                continue
            sil = _silhouette_score_1d(v_sorted, labels)
            if not np.isfinite(sil):
                continue
            num_singletons = int(np.sum(sizes == 1))
            score = float(sil - penalty_singleton * num_singletons)
            sse = _sse_from_labels(v_sorted, labels)

            better = False
            if score > best_score + 1e-12:
                better = True
            elif abs(score - best_score) <= 1e-12:
                if best_labels is None:
                    better = True
                else:
                    best_sizes = np.bincount(best_labels, minlength=int(best_k))
                    best_singletons = int(np.sum(best_sizes == 1))
                    if num_singletons < best_singletons:
                        better = True
                    elif num_singletons == best_singletons:
                        if best_k is None or int(k) < int(best_k):
                            better = True
                        elif int(k) == int(best_k):
                            if sse < best_sse - 1e-12:
                                better = True
                            elif abs(sse - best_sse) <= 1e-12:
                                if best_cuts is None or cuts < best_cuts:
                                    better = True

            if better:
                best_score = score
                best_labels = labels
                best_k = int(k)
                best_sse = float(sse)
                best_cuts = cuts

    if best_labels is None:
        k = 2 if m >= 2 else 1
        cut = (m // 2,)
        labels = _partition_labels_from_cuts(m, cut)
        best_labels = labels
        best_k = k

    clusters = []
    for lab in range(int(best_k)):
        idx = np.where(best_labels == lab)[0]
        clusters.append([float(x) for x in v_sorted[idx]])
    return clusters


def _cluster_choose_lower(clusters):
    eligible = [c for c in clusters if len(c) >= 2]
    if not eligible:
        return None
    med = [_median(c) for c in eligible]
    best = int(np.argmax(med))
    return float(max(eligible[best]))


def _cluster_choose_upper(clusters):
    eligible = [c for c in clusters if len(c) >= 2]
    if not eligible:
        return None
    med = [_median(c) for c in eligible]
    best = int(np.argmin(med))
    return float(min(eligible[best]))


def cluster_per_sample_fast1d(lower_mat, upper_mat, k_candidates=(2, 3, 4), penalty_singleton=0.2):
    M, n = lower_mat.shape
    outL = np.empty(n, dtype=np.float32)
    outU = np.empty(n, dtype=np.float32)

    for i in range(n):
        lowers = lower_mat[:, i].astype(np.float64, copy=False)
        uppers = upper_mat[:, i].astype(np.float64, copy=False)

        cL = _best_cluster_partition_1d(lowers, k_candidates=k_candidates, penalty_singleton=penalty_singleton)
        cU = _best_cluster_partition_1d(uppers, k_candidates=k_candidates, penalty_singleton=penalty_singleton)

        lo = _cluster_choose_lower(cL)
        hi = _cluster_choose_upper(cU)

        lowers_s = np.sort(lowers)
        uppers_s = np.sort(uppers)

        if lo is None:
            lo = float(lowers_s[1] if lowers_s.size >= 2 else lowers_s[0])
        if hi is None:
            hi = float(uppers_s[-2] if uppers_s.size >= 2 else uppers_s[-1])
        if lo > hi:
            lo = float(lowers_s[1] if lowers_s.size >= 2 else lo)
            hi = float(uppers_s[-2] if uppers_s.size >= 2 else hi)

        outL[i] = np.float32(lo)
        outU[i] = np.float32(hi)

    return outL, outU


# -------------------------
# Smoothing helper for plotting (simple moving average over finite points)
# -------------------------
def _smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size < 2:
        return y
    w = int(window)
    w = max(1, min(w, y.size))
    y_arr = np.asarray(y, dtype=float)
    finite_mask = np.isfinite(y_arr)
    if not np.any(finite_mask):
        return y
    # interpolate NaNs for smoothing
    if not np.all(finite_mask):
        idx = np.arange(y_arr.size)
        y_arr[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], y_arr[finite_mask])
    kernel = np.ones(w, dtype=float) / float(w)
    y_smooth = np.convolve(y_arr, kernel, mode="same")
    return y_smooth.astype(np.float32)


def smooth_xy(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    smooth_grid_n: int,
    window: int,
    spline_k: int,
    spline_s: float,
    lowess_frac: float,
    lowess_it: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    # aggregate duplicates
    uniq_x, idx = np.unique(x, return_inverse=True)
    agg_y = np.zeros_like(uniq_x)
    counts = np.zeros_like(uniq_x)
    for k, xv in enumerate(uniq_x):
        mask = idx == k
        agg_y[k] = np.mean(y[mask])
        counts[k] = np.sum(mask)
    x = uniq_x
    y = agg_y
    if x.size == 1:
        return x, y
    x_dense = np.linspace(x.min(), x.max(), int(max(2, smooth_grid_n)), dtype=np.float64)

    if method == "none":
        y_dense = np.interp(x_dense, x, y)
        return x_dense, y_dense

    if method == "spline":
        if x.size <= spline_k:
            y_dense = np.interp(x_dense, x, y)
            return x_dense, y_dense
        s_eff = spline_s
        if s_eff < 0:
            var_y = np.nanvar(y)
            s_eff = float(len(x) * var_y * 0.25)
        spl = UnivariateSpline(x, y, k=int(spline_k), s=s_eff)
        y_dense = spl(x_dense)
        return x_dense, y_dense

    if method == "lowess":
        lo = lowess(y, x, frac=lowess_frac, it=lowess_it, return_sorted=True)
        y_dense = np.interp(x_dense, lo[:, 0], lo[:, 1])
        return x_dense, y_dense

    if method == "moving_avg":
        y_interp = np.interp(x_dense, x, y)
        y_smooth = _smooth_series(y_interp, window)
        return x_dense, y_smooth

    raise ValueError(f"Unknown smooth_method '{method}'")


# -------------------------
# Two-pass fit helper (cached propensity)
# -------------------------
def fit_two_pass_do1_cached(
    EstimatorClass,
    div_name,
    X_tr,
    A_tr,
    Y_tr,
    X_eval,
    dual_net_config,
    fit_config,
    seed,
    propensity_model,
    m_model,
    propensity_cache,
):
    est_pos = EstimatorClass(
        divergence=div_name,
        phi=phi_identity,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
    ).fit(X_tr, A_tr, Y_tr, propensity_cache=propensity_cache)

    est_neg = EstimatorClass(
        divergence=div_name,
        phi=phi_neg,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
    ).fit(X_tr, A_tr, Y_tr, propensity_cache=propensity_cache)

    U = est_pos.predict_bound(a=1, X=X_eval).astype(np.float32)
    L = (-est_neg.predict_bound(a=1, X=X_eval)).astype(np.float32)
    return L, U

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="MC ribbon plot of causal bounds vs X0 using IHDP data.")
    parser.add_argument(
        "--data_path",
        "--data-path",
        dest="data_path",
        type=str,
        default="",
        help="IHDP CSV path (empty uses the first existing candidate).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset name like 'ihdp_npci_1' or 'ihdp_npci_10' (auto-appends .csv).",
    )
    parser.add_argument("--m", type=int, default=20, help="number of MC replicates")
    parser.add_argument("--base_seed", type=int, default=20190602, help="base seed; seed_j = base_seed + j")
    parser.add_argument("--n", type=int, default=500, help="samples per replicate")
    parser.add_argument("--d", type=int, default=DEFAULT_D, help="feature dimension")
    parser.add_argument(
        "--divergence",
        type=str,
        default="kth, tight_kth",
        help="Comma-separated divergences from {KL,TV,Hellinger,Chi2,JS,combined,cluster,kth,tight_kth}.",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Aggregation statistic across MC replicates.",
    )
    parser.add_argument("--min_valid_rate", type=float, default=1.0, help="Minimum validity rate for keeping an eval point.")
    parser.add_argument("--unique_save", action="store_true", default=True, help="Add unique timestamp suffix to outputs.")
    parser.add_argument("--outdir", type=str, default="experiments", help="Output directory.")
    parser.add_argument("--xgb_n_jobs", type=int, default=-1, help="n_jobs for xgboost models (-1 uses all cores).")
    parser.add_argument("--num_epochs", type=int, default=200, help="Dual net epochs.")
    parser.add_argument("--n_folds", type=int, default=2, help="Number of CV folds.")
    parser.add_argument("--eps_propensity", type=float, default=1e-3, help="Propensity clipping epsilon.")
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="Torch intra-op threads (0 uses all available cores).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for dual nets (use cuda if GPU available).",
    )
    parser.add_argument(
        "--eval_dim",
        type=int,
        default=0,
        help="Dimension X[:,eval_dim]",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=500,
        help="Number of evaluation points sampled from IHDP (<=0 uses all rows).",
    )
    parser.add_argument(
        "--eval_axis",
        type=str,
        default="both",
        choices=["x0", "propensity", "both"],
        help="Eval grid axis: X0, propensity e(A=1|X), or both.",
    )
    parser.add_argument("--smooth_window", type=int, default=5, help="Smoothing window (moving average) for plotting.")
    parser.add_argument(
        "--smooth_method",
        type=str,
        default="spline",
        choices=["none", "moving_avg", "spline", "lowess"],
        help="Smoothing method for plotting ribbons.",
    )
    parser.add_argument("--smooth_grid_n", type=int, default=500, help="Number of points in dense grid for smoothing.")
    parser.add_argument("--spline_k", type=int, default=3, help="Spline order (e.g., 3 for cubic).")
    parser.add_argument("--spline_s", type=float, default=-1.0, help="Spline smoothing factor (<0 => auto heuristic).")
    parser.add_argument("--lowess_frac", type=float, default=0.2, help="LOWESS frac parameter.")
    parser.add_argument("--lowess_it", type=int, default=1, help="LOWESS iterations.")
    parser.add_argument("--plot_raw_points", default=True, action="store_true", help="Overlay raw (unsmoothed) points on the plot.")
    parser.add_argument(
        "--plot_raw",
        default = True, 
        action="store_true",
        help="Overlay raw (unsmoothed) lower/upper lines on the plot.",
    )
    parser.add_argument(
        "--kval",
        type=int,
        default=4,
        help="k for kth/tight_kth aggregation (default: number of base divergences).",
    )
    parser.add_argument(
        "--timing",
        dest="timing",
        action="store_true",
        default=True,
        help="Enable progress/timing logs.",
    )
    parser.add_argument(
        "--no-timing",
        dest="timing",
        action="store_false",
        help="Disable progress/timing logs.",
    )
    parser.add_argument(
        "--timing_detail",
        "--timing-detail",
        action="store_true",
        default=False,
        help="Enable per-divergence timing logs.",
    )
    pre_timing = _timing_enabled_from_argv(default=True)
    with StepTimer("parse args", use_tqdm=False, enabled=pre_timing):
        args = parser.parse_args()
    timing_enabled = bool(args.timing)
    timing_detail = bool(args.timing_detail)

    config_timer = StepTimer("configure experiment", use_tqdm=False, enabled=timing_enabled)
    config_timer.__enter__()
    torch_threads = args.torch_threads if args.torch_threads > 0 else max(1, os.cpu_count() or 1)
    try:
        torch.set_num_threads(torch_threads)
    except Exception:
        pass

    if args.dataset:
        args.data_path = args.dataset if args.dataset.endswith(".csv") else f"{args.dataset}.csv"
    args.data_path = _resolve_data_path(args.data_path)
    df = load_ihdp_csv(args.data_path)
    A_all, Y_all, groundtruth_all, X_all = prepare_ihdp_arrays(df, d=args.d)

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if args.unique_save else ""

    def name_with_suffix(base: str, ext: str) -> str:
        fname = f"{base}_{stamp}.{ext}" if stamp else f"{base}.{ext}"
        return os.path.join(args.outdir, fname)

    dual_net_config = {
        "hidden_sizes": (64, 64),
        "activation": "relu",
        "dropout": 0.0,
        "h_clip": 20.0,
        "device": args.device,
    }
    
    propensity_model = "xgboost"
    m_model = "xgboost"

    fit_config = {
        "n_folds": args.n_folds,
        "num_epochs": args.num_epochs,
        "batch_size": 32,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "max_grad_norm": 10.0,
        "eps_propensity": args.eps_propensity,
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
            "n_jobs": args.xgb_n_jobs,
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
            "n_jobs": args.xgb_n_jobs,
            "verbosity": 0,
        },
        "verbose": False,
        "log_every": 10,
    }
    config_timer.__exit__(None, None, None)

    m = args.m
    n = args.n
    d = args.d
    base_divs = ["KL", "TV", "Hellinger", "Chi2", "JS"]
    allowed_divs = set(base_divs + ["combined", "cluster", "kth", "tight_kth"])
    div_list = [div.strip() for div in args.divergence.split(",") if div.strip()]
    if not div_list:
        div_list = ["combined", "cluster"]
    for div in div_list:
        if div not in allowed_divs:
            raise ValueError(f"Unknown divergence '{div}'. Allowed: {sorted(allowed_divs)}")
    needs_base = set()
    if any(div in {"combined", "cluster", "kth", "tight_kth"} for div in div_list):
        needs_base.update(base_divs)
    needs_base.update([div for div in div_list if div in base_divs])

    n_total = int(X_all.shape[0])
    if n > n_total:
        raise ValueError(f"n={n} exceeds IHDP rows {n_total}.")

    # Fixed evaluation set sampled from the IHDP dataset (shared across replicates)
    with StepTimer("build X_eval grid", use_tqdm=False, enabled=timing_enabled):
        eval_seed = args.base_seed + 10**6
        rng_eval = np.random.default_rng(eval_seed)
        if args.n_eval <= 0 or args.n_eval >= n_total:
            eval_idx = np.arange(n_total)
        else:
            eval_idx = rng_eval.choice(n_total, size=args.n_eval, replace=False)
        X_eval = np.asarray(X_all[eval_idx], dtype=np.float32)
        X0_eval = X_eval[:, args.eval_dim]
        truth_eval = np.asarray(groundtruth_all[eval_idx], dtype=np.float32).reshape(-1)

    n_eval = int(X_eval.shape[0])
    truth_mat = np.full((m, n_eval), np.nan, dtype=np.float32)
    seeds = []
    upper_dict = {div: np.full((m, n_eval), np.nan, dtype=np.float32) for div in div_list}
    lower_dict = {div: np.full((m, n_eval), np.nan, dtype=np.float32) for div in div_list}
    valid_dict = {div: np.zeros((m, n_eval), dtype=bool) for div in div_list}

    with StepTimer("MC replicate loop", use_tqdm=True, enabled=timing_enabled):
        for j in tqdm(range(m), desc="MC replicates"):
            seed = args.base_seed + j
            seeds.append(seed)
            with StepTimer(f"replicate {j}", use_tqdm=True, enabled=timing_enabled):
                with StepTimer("data generation", use_tqdm=True, enabled=timing_enabled):
                    rng = np.random.default_rng(seed)
                    train_idx = rng.permutation(n_total)[:n]
                    X_tr = np.asarray(X_all[train_idx], dtype=np.float32)
                    A_tr = np.asarray(A_all[train_idx], dtype=np.float32)
                    Y_tr = np.asarray(Y_all[train_idx], dtype=np.float32)

                with StepTimer("nuisance fits", use_tqdm=True, enabled=timing_enabled):
                    prop_cache = prefit_propensity_cache(
                        X=X_tr,
                        A=A_tr,
                        propensity_model=propensity_model,
                        propensity_config=fit_config["propensity_config"],
                        n_folds=fit_config["n_folds"],
                        seed=seed,
                        eps_propensity=fit_config["eps_propensity"],
                    )

                with StepTimer("bounds per divergence", use_tqdm=True, enabled=timing_enabled):
                    base_outputs = {}
                    for div in needs_base:
                        with StepTimer(
                            f"fit {div}",
                            use_tqdm=True,
                            enabled=timing_enabled and timing_detail,
                        ):
                            L_div, U_div = fit_two_pass_do1_cached(
                                DebiasedCausalBoundEstimator,
                                div,
                                X_tr,
                                A_tr,
                                Y_tr,
                                X_eval,
                                dual_net_config,
                                fit_config,
                                seed=seed,
                                propensity_model=propensity_model,
                                m_model=m_model,
                                propensity_cache=prop_cache,
                            )
                        base_outputs[div] = (L_div, U_div)

                    lower_base = None
                    upper_base = None
                    for div in div_list:
                        if div in base_outputs:
                            L, U = base_outputs[div]
                        elif div == "combined":
                            if lower_base is None:
                                lower_base = np.vstack([base_outputs[b][0] for b in base_divs])
                                upper_base = np.vstack([base_outputs[b][1] for b in base_divs])
                            L, U = combined_cwise_intersection(lower_base, upper_base, c=3)
                        elif div == "cluster":
                            if lower_base is None:
                                lower_base = np.vstack([base_outputs[b][0] for b in base_divs])
                                upper_base = np.vstack([base_outputs[b][1] for b in base_divs])
                            L, U = cluster_per_sample_fast1d(
                                lower_base, upper_base, k_candidates=(2, 3, 4), penalty_singleton=0.2
                            )
                        elif div == "kth":
                            if lower_base is None:
                                lower_base = np.vstack([base_outputs[b][0] for b in base_divs])
                                upper_base = np.vstack([base_outputs[b][1] for b in base_divs])
                            k_val = int(args.kval) if args.kval is not None else int(lower_base.shape[0])
                            if k_val < 1 or k_val > int(lower_base.shape[0]):
                                raise ValueError(f"--kval must be in [1, {int(lower_base.shape[0])}] (got {k_val}).")
                            L = np.empty(lower_base.shape[1], dtype=np.float32)
                            U = np.empty(lower_base.shape[1], dtype=np.float32)
                            for i in range(lower_base.shape[1]):
                                lo, up = kth(lower_base[:, i], upper_base[:, i], k_val)
                                L[i] = np.float32(lo)
                                U[i] = np.float32(up)
                        elif div == "tight_kth":
                            if lower_base is None:
                                lower_base = np.vstack([base_outputs[b][0] for b in base_divs])
                                upper_base = np.vstack([base_outputs[b][1] for b in base_divs])
                            k_val = int(args.kval) if args.kval is not None else int(lower_base.shape[0])
                            if k_val < 1 or k_val > int(lower_base.shape[0]):
                                raise ValueError(f"--kval must be in [1, {int(lower_base.shape[0])}] (got {k_val}).")
                            L = np.empty(lower_base.shape[1], dtype=np.float32)
                            U = np.empty(lower_base.shape[1], dtype=np.float32)
                            for i in range(lower_base.shape[1]):
                                lo, up = tight_kth(lower_base[:, i], upper_base[:, i], k=k_val)
                                L[i] = np.float32(lo)
                                U[i] = np.float32(up)
                        else:
                            raise ValueError(f"Unsupported divergence '{div}'")

                        W = U - L
                        valid = np.isfinite(U) & np.isfinite(L) & (W > 0)

                        upper_dict[div][j, :] = U
                        lower_dict[div][j, :] = L
                        valid_dict[div][j, :] = valid

                with StepTimer("store truth", use_tqdm=True, enabled=timing_enabled):
                    truth_mat[j, :] = truth_eval

    axis_specs = []
    if args.eval_axis in {"x0", "both"}:
        if args.eval_dim < 0 or args.eval_dim >= X_eval.shape[1]:
            raise ValueError(f"--eval_dim must be in [0, {X_eval.shape[1] - 1}] (got {args.eval_dim}).")
        axis_specs.append(("x0", X_eval[:, args.eval_dim], f"X{args.eval_dim}"))

    if args.eval_axis in {"propensity", "both"}:
        with StepTimer("build propensity eval axis", use_tqdm=False, enabled=timing_enabled):
            prop_cache_eval = prefit_propensity_cache(
                X=X_all,
                A=A_all,
                propensity_model=propensity_model,
                propensity_config=fit_config["propensity_config"],
                n_folds=fit_config["n_folds"],
                seed=int(eval_seed + 777),
                eps_propensity=fit_config["eps_propensity"],
            )
            e1_oof = np.asarray(prop_cache_eval["e1_oof"], dtype=np.float32).reshape(-1)
            axis_specs.append(("propensity", e1_oof[eval_idx], "e(A=1|X)"))

    for axis_key, axis_eval, axis_label in axis_specs:
        base_name = "plot_idhp_ribbon" if axis_key == "x0" else f"plot_idhp_ribbon_{axis_key}"

        with StepTimer(f"aggregate + smooth ribbons ({axis_key})", use_tqdm=False, enabled=timing_enabled):
            if args.stat == "mean":
                agg_fn = np.nanmean
            else:
                agg_fn = np.nanmedian

            aggregated_results = []
            for div in div_list:
                upper_mat = upper_dict[div]
                lower_mat = lower_dict[div]
                valid_mat = valid_dict[div]

                coverage_rate = valid_mat.mean(axis=0)
                S_mask = valid_mat.all(axis=0)
                if not np.any(S_mask):
                    for relax in [args.min_valid_rate, 0.9, 0.5]:
                        S_mask = coverage_rate >= relax
                        if np.any(S_mask):
                            break

                lower_masked = np.where(valid_mat, lower_mat, np.nan)
                upper_masked = np.where(valid_mat, upper_mat, np.nan)

                idx_sel = np.where(S_mask)[0]
                if idx_sel.size == 0:
                    warnings.warn(
                        f"No evaluation points met validity criteria for divergence {div}; outputs will be empty."
                    )

                l_bar = agg_fn(lower_masked[:, idx_sel], axis=0)
                u_bar = agg_fn(upper_masked[:, idx_sel], axis=0)
                theta_bar = agg_fn(truth_mat[:, idx_sel], axis=0)
                width_bar = u_bar - l_bar
                valid_rate_sel = coverage_rate[idx_sel]
                axis_sel = axis_eval[idx_sel]

                order = np.argsort(axis_sel)
                axis_plot = axis_sel[order]
                l_plot = l_bar[order]
                u_plot = u_bar[order]
                theta_plot = theta_bar[order]
                width_plot = width_bar[order]
                valid_plot = valid_rate_sel[order]

                # Bound-preserving smoothing via mid/halfwidth reparameterization
                mid_raw = 0.5 * (u_plot + l_plot)
                half_raw = 0.5 * (u_plot - l_plot)
                log_half_raw = np.log(np.clip(half_raw, 1e-6, None))

                x_mid, mid_s = smooth_xy(
                    axis_plot,
                    mid_raw,
                    method=args.smooth_method,
                    smooth_grid_n=args.smooth_grid_n,
                    window=args.smooth_window,
                    spline_k=args.spline_k,
                    spline_s=args.spline_s,
                    lowess_frac=args.lowess_frac,
                    lowess_it=args.lowess_it,
                )

                x_log, log_half_s = smooth_xy(
                    axis_plot,
                    log_half_raw,
                    method=args.smooth_method,
                    smooth_grid_n=args.smooth_grid_n,
                    window=args.smooth_window,
                    spline_k=args.spline_k,
                    spline_s=args.spline_s,
                    lowess_frac=args.lowess_frac,
                    lowess_it=args.lowess_it,
                )

                if x_mid.size == 0:
                    x_s = axis_plot
                    mid_s = mid_raw
                    log_half_s = log_half_raw
                else:
                    x_s = x_mid
                    if x_log.size > 0:
                        log_half_s = np.interp(x_s, x_log, log_half_s)
                    else:
                        log_half_s = np.interp(x_s, axis_plot, log_half_raw)

                half_s = np.exp(log_half_s)
                l_smooth = mid_s - half_s
                u_smooth = mid_s + half_s

                # Smooth truth on same x-grid for plotting
                x_theta, theta_s = smooth_xy(
                    axis_plot,
                    theta_plot,
                    method=args.smooth_method,
                    smooth_grid_n=args.smooth_grid_n,
                    window=args.smooth_window,
                    spline_k=args.spline_k,
                    spline_s=args.spline_s,
                    lowess_frac=args.lowess_frac,
                    lowess_it=args.lowess_it,
                )
                if x_theta.size > 0 and x_s.size > 0:
                    theta_s = np.interp(x_s, x_theta, theta_s)
                elif x_s.size > 0:
                    theta_s = np.interp(x_s, axis_plot, theta_plot)
                else:
                    x_s = axis_plot
                    theta_s = theta_plot

                width_smooth = u_smooth - l_smooth

                aggregated_results.append(
                    {
                        "div": div,
                        "idx_plot": idx_sel[order],
                        "x_raw": axis_plot,
                        "l_raw": l_plot,
                        "u_raw": u_plot,
                        "x_raw": axis_plot,
                        "l_raw": l_plot,
                        "u_raw": u_plot,
                        "theta_raw": theta_plot,
                        "valid_raw": valid_plot,
                        "x_s": x_s,
                        "l_s": l_smooth,
                        "u_s": u_smooth,
                        "theta_s": theta_s,
                        "width_s": width_smooth,
                    }
                )

        with StepTimer(f"save tables ({axis_key})", use_tqdm=False, enabled=timing_enabled):
            # Save table (long format over divergences)
            table_rows = []
            for res in aggregated_results:
                div = res["div"]
                for i, x0, th, lo, up, wd, vr in zip(
                    res["idx_plot"],
                    res["x_raw"],
                    res["theta_raw"],
                    res["l_raw"],
                    res["u_raw"],
                    res["u_raw"] - res["l_raw"],
                    res["valid_raw"],
                ):
                    table_rows.append(
                        {
                            "method": div,
                            "i": int(i),
                            "X0": float(x0),
                            "theta": float(th),
                            "lower": float(lo),
                            "upper": float(up),
                            "width": float(wd),
                            "valid_rate": float(vr),
                        }
                    )
            table_path = name_with_suffix(f"{base_name}_table", "csv")
            try:
                import pandas as pd

                pd.DataFrame(table_rows).to_csv(table_path, index=False)
            except Exception:
                header = "method,i,X0,theta,lower,upper,width,valid_rate"
                with open(table_path, "w") as f:
                    f.write(header + "\n")
                    for row in table_rows:
                        f.write(
                            f"{row['method']},{row['i']},{row['X0']},{row['theta']},{row['lower']},{row['upper']},{row['width']},{row['valid_rate']}\n"
                        )

            # Smoothed table (dense grid)
            sm_table_rows = []
            for res in aggregated_results:
                div = res["div"]
                xg = res["x_s"]
                lg = res["l_s"]
                ug = res["u_s"]
                tg = res["theta_s"]
                wg = ug - lg
                for x0, th, lo, up, wd in zip(xg, tg, lg, ug, wg):
                    sm_table_rows.append(
                        {
                            "method": div,
                            "X0": float(x0),
                            "theta": float(th),
                            "lower": float(lo),
                            "upper": float(up),
                            "width": float(wd),
                        }
                    )
            sm_table_path = name_with_suffix(f"{base_name}_smoothed_table", "csv")
            try:
                import pandas as pd

                pd.DataFrame(sm_table_rows).to_csv(sm_table_path, index=False)
            except Exception:
                header = "method,X0,theta,lower,upper,width"
                with open(sm_table_path, "w") as f:
                    f.write(header + "\n")
                    for row in sm_table_rows:
                        f.write(
                            f"{row['method']},{row['X0']},{row['theta']},{row['lower']},{row['upper']},{row['width']}\n"
                        )

        with StepTimer(f"plot ribbons ({axis_key})", use_tqdm=False, enabled=timing_enabled):
            # Smoothed plot
            plt.figure(figsize=(7.0, 4.0))
            color_map = {
                "combined": "tab:blue",
                "cluster": "tab:orange",
                "kth": "tab:cyan",
                "tight_kth": "tab:olive",
                "KL": "tab:green",
                "TV": "tab:red",
                "Hellinger": "tab:purple",
                "Chi2": "tab:brown",
                "JS": "tab:pink",
            }
            for res in aggregated_results:
                if res["idx_plot"].size == 0:
                    continue
                c = color_map.get(res["div"], None)
                plt.fill_between(
                    res["x_s"], res["l_s"], res["u_s"], alpha=0.2, color=c, label=f"{res['div']} bounds"
                )
                plt.plot(res["x_s"], res["l_s"], color=c, alpha=0.7, linewidth=1.0)
                plt.plot(res["x_s"], res["u_s"], color=c, alpha=0.7, linewidth=1.0)
                if args.plot_raw:
                    plt.plot(res["x_raw"], res["l_raw"], color=c, alpha=0.7, linewidth=1.0)
                    plt.plot(res["x_raw"], res["u_raw"], color=c, alpha=0.7, linewidth=1.0)
                if args.plot_raw_points:
                    plt.scatter(res["x_raw"], res["l_raw"], color=c, alpha=0.2, s=8)
                    plt.scatter(res["x_raw"], res["u_raw"], color=c, alpha=0.2, s=8)
            # Truth line (smoothed on bounds grid of first method)
            if aggregated_results and aggregated_results[0]["x_s"].size > 0:
                plt.plot(
                    aggregated_results[0]["x_s"],
                    aggregated_results[0]["theta_s"],
                    color="k",
                    linewidth=1.5,
                    label="Truth",
                )
            plt.xlabel(axis_label)
            plt.ylabel("E[Y | do(A=1), X]")
            plt.title(f"Causal bounds vs {axis_label} (stat={args.stat}, divs={','.join(div_list)})")
            plt.legend()
            plt.tight_layout()
            fig_path = name_with_suffix(f"{base_name}_smoothed", "png")
            plt.savefig(fig_path, dpi=200)
            plt.close()

            # Raw-only plot
            plt.figure(figsize=(7.0, 4.0))
            label_raw = False
            for res in aggregated_results:
                if res["idx_plot"].size == 0:
                    continue
                c = color_map.get(res["div"], None)
                lbl = f"{res['div']} bounds" if not label_raw else None
                plt.plot(res["x_raw"], res["l_raw"], color=c, alpha=0.7, linewidth=1.0, label=lbl)
                plt.plot(res["x_raw"], res["u_raw"], color=c, alpha=0.7, linewidth=1.0)
                label_raw = True
            if aggregated_results:
                plt.plot(
                    aggregated_results[0]["x_raw"],
                    aggregated_results[0]["theta_raw"],
                    color="k",
                    linewidth=1.5,
                    label="Truth",
                )
            plt.xlabel(axis_label)
            plt.ylabel("E[Y | do(A=1), X]")
            plt.title(f"Raw bounds vs {axis_label} (stat={args.stat}, divs={','.join(div_list)})")
            plt.legend()
            plt.tight_layout()
            fig_raw_path = name_with_suffix(f"{base_name}_raw", "png")
            plt.savefig(fig_raw_path, dpi=200)
            plt.close()

        with StepTimer(f"save artifacts ({axis_key})", use_tqdm=False, enabled=timing_enabled):
            # Artifacts
            artifacts = {
                "X_eval": X_eval,
                "x_axis_eval": axis_eval,
                "eval_axis": axis_key,
                "eval_idx": eval_idx,
                "truth_eval": truth_eval,
                "lower_dict": lower_dict,
                "upper_dict": upper_dict,
                "truth_mat": truth_mat,
                "valid_dict": valid_dict,
                "args": vars(args),
                "n_total": n_total,
                "n_eval": n_eval,
                "data_path": args.data_path,
                "fit_config": fit_config,
                "dual_net_config": dual_net_config,
                "seeds": seeds,
                "aggregated_results": aggregated_results,
                "timestamp": stamp,
                "divergences": div_list,
                "smoothed_table_csv": sm_table_path,
                "smooth_method": args.smooth_method,
                "smooth_grid_n": args.smooth_grid_n,
                "spline_k": args.spline_k,
                "spline_s": args.spline_s,
                "lowess_frac": args.lowess_frac,
                "lowess_it": args.lowess_it,
            }
            artifacts_path = name_with_suffix(f"{base_name}_artifacts", "pkl")
            with open(artifacts_path, "wb") as f:
                pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

            summary = {
                "timestamp": stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "files": {
                    "table_csv": table_path,
                    "smoothed_table_csv": sm_table_path,
                    "plot_png": fig_path,
                    "plot_raw_png": fig_raw_path,
                    "artifacts_pkl": artifacts_path,
                },
                "args": vars(args),
                "n": n,
                "d": d,
                "m": m,
                "n_total": n_total,
                "n_eval": n_eval,
                "data_path": args.data_path,
                "divergences": div_list,
                "stat": args.stat,
                "min_valid_rate": args.min_valid_rate,
                "eval_axis": axis_key,
                "selected_counts": {res["div"]: int(len(res["idx_plot"])) for res in aggregated_results},
                "smooth_method": args.smooth_method,
                "smooth_grid_n": args.smooth_grid_n,
                "spline_k": args.spline_k,
                "spline_s": args.spline_s,
                "lowess_frac": args.lowess_frac,
                "lowess_it": args.lowess_it,
            }
            summary_path = name_with_suffix(f"{base_name}_summary", "json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            log_line = (
                f"[{base_name}] axis={axis_key} ts={summary['timestamp']} m={m} n={n} d={d} stat={args.stat} "
                f"divs={','.join(div_list)} selected={summary['selected_counts']} "
                f"args={vars(args)} files={summary['files']}"
            )
            log_path = name_with_suffix(f"{base_name}_log", "txt")
            with open(log_path, "a") as f:
                f.write(log_line + "\n")
