"""
Monte Carlo evaluation of causal bounds coverage/width/score vs sample size n.

This script fits information-theoretic causal bound estimators (DebiasedCausalBoundEstimator)
on observational samples of size n, across Monte-Carlo replicates, and evaluates bounds on a
FIXED evaluation covariate set X_eval that is shared across all n and replicates.

Fixes addressed (relative to earlier versions):
1) Coverage is reported BOTH unconditional (invalid => failure) and conditional-on-validity.
2) Score penalizes non-coverage in the usual (loss-increasing) direction.
3) Evaluation is performed at the same X_eval across all n (removes changing-x confounding).

Notes:
- Lower bound is computed via sign-flip: L_phi = - U_{-phi}.
- This script does NOT re-implement estimator theory; it uses causal_bound.py.
"""
from __future__ import annotations

import argparse
import datetime
import inspect
import json
import os
import pickle
import sys
import time
import warnings
from itertools import combinations
from typing import Any, Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **k: x  # type: ignore

from causal_bound import DebiasedCausalBoundEstimator, prefit_propensity_cache
from data_generating import generate_data


# -------------------------
# Lightweight progress/timing utilities
# -------------------------
_STEP_STACK: List[Dict[str, Any]] = []


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
def combined_cwise_intersection(lower_mat: np.ndarray, upper_mat: np.ndarray, c: int = 3) -> Tuple[np.ndarray, np.ndarray]:
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
# Cluster aggregator (fast 1D partition search, no sklearn)
# -------------------------
def _median(vals) -> float:
    arr = np.sort(np.asarray(vals, dtype=np.float64))
    n = int(arr.shape[0])
    if n <= 0:
        return float("nan")
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


def _partition_labels_from_cuts(m: int, cuts: Tuple[int, ...]) -> np.ndarray:
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
    """
    Robust per-sample aggregator based on clustering lower/upper values across divergences.
    Operates pointwise across evaluation points.
    """
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

        # Fallbacks if clustering yields no eligible cluster.
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
# Two-pass fit helper (cached propensity)
# -------------------------
def fit_two_pass_do1_cached(
    EstimatorClass,
    div_name: str,
    X_train: np.ndarray,
    A_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    dual_net_config: Dict[str, Any],
    fit_config: Dict[str, Any],
    seed: int,
    propensity_model: Any,
    m_model: Any,
    propensity_cache: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit upper bound estimator for phi(y)=y and phi(y)=-y on training data,
    then evaluate both bounds at X_eval. Returns (L_eval, U_eval).
    """
    est_pos = EstimatorClass(
        divergence=div_name,
        phi=phi_identity,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
    ).fit(X_train, A_train, Y_train, propensity_cache=propensity_cache)

    est_neg = EstimatorClass(
        divergence=div_name,
        phi=phi_neg,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
    ).fit(X_train, A_train, Y_train, propensity_cache=propensity_cache)

    U = est_pos.predict_bound(a=1, X=X_eval).astype(np.float32)
    L = (-est_neg.predict_bound(a=1, X=X_eval)).astype(np.float32)
    return L, U


# -------------------------
# Helper utilities
# -------------------------
def _parse_n_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            raise ValueError(f"Invalid n value: {p}")
    if not out:
        raise ValueError("n_list is empty.")
    return out


def _parse_divergences(raw: str, base_divs: List[str]) -> List[str]:
    allowed = set(base_divs + ["combined", "cluster"])
    divs = [d.strip() for d in raw.split(",") if d.strip()]
    if not divs:
        divs = ["combined"]
    for d in divs:
        if d not in allowed:
            raise ValueError(f"Unknown divergence '{d}'. Allowed: {sorted(allowed)}")
    return divs


def _stat_reduce(arr: np.ndarray, stat: str) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if stat == "mean":
        return float(np.mean(arr))
    return float(np.median(arr))


def _nan_quantile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _unique_suffix(enabled: bool) -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if enabled else ""


def _score_penalized_width(width: float, coverage_uncond: float, lam: float, alpha: float) -> float:
    """
    Replicate-level score = width * (1 + lam * max(0, (1-alpha) - coverage_uncond)).

    - If coverage_uncond >= 1-alpha, score = width.
    - If coverage_uncond < 1-alpha, score increases linearly in shortfall.
    """
    if not np.isfinite(width) or not np.isfinite(coverage_uncond):
        return float("nan")
    target = 1.0 - float(alpha)
    shortfall = max(0.0, target - float(coverage_uncond))
    return float(width * (1.0 + float(lam) * shortfall))


def _call_generate_data_compat(
    n: int,
    d: int,
    seed: int,
    structural_type: str,
    x_range: Optional[float] = None,
    noise_dist: Optional[str] = None,
    _warn_state: Dict[str, bool] = None,
) -> Dict[str, object]:
    """
    Call generate_data(...) but only pass kwargs that exist in the current signature.
    This makes the script compatible with multiple versions of data_generating.py.
    """
    if _warn_state is None:
        _warn_state = {}
    sig = inspect.signature(generate_data)
    params = sig.parameters

    kwargs: Dict[str, Any] = {"n": n, "d": d, "seed": seed, "structural_type": structural_type}
    if x_range is not None and "x_range" in params:
        kwargs["x_range"] = float(x_range)
    if noise_dist is not None and "noise_dist" in params:
        kwargs["noise_dist"] = noise_dist
    elif noise_dist is not None and "noise_dist" not in params and not _warn_state.get("noise_dist", False):
        warnings.warn("generate_data(...) does not accept noise_dist; ignoring --noise_dist.", RuntimeWarning)
        _warn_state["noise_dist"] = True

    return generate_data(**kwargs)


def _make_X_eval(
    *,
    n_eval: int,
    d: int,
    eval_mode: str,
    eval_seed: int,
    structural_type: str,
    x_range: Optional[float],
    noise_dist: Optional[str],
    x0_min: float,
    x0_max: float,
    x_fill: float,
) -> np.ndarray:
    if eval_mode == "sample":
        data_eval = _call_generate_data_compat(
            n=n_eval,
            d=d,
            seed=eval_seed,
            structural_type=structural_type,
            x_range=x_range,
            noise_dist=noise_dist,
            _warn_state={},
        )
        X_eval = np.asarray(data_eval["X"], dtype=np.float32)
        if X_eval.ndim != 2 or X_eval.shape[1] != d:
            raise ValueError(f"generate_data returned X with shape {X_eval.shape}, expected (*,{d}).")
        return X_eval

    if eval_mode == "grid_x0":
        if d <= 0:
            raise ValueError("d must be positive.")
        x0 = np.linspace(float(x0_min), float(x0_max), int(n_eval), dtype=np.float32)
        X_eval = np.full((int(n_eval), int(d)), float(x_fill), dtype=np.float32)
        X_eval[:, 0] = x0
        return X_eval

    raise ValueError(f"Unknown eval_mode: {eval_mode}")


# -------------------------
# Main
# -------------------------


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Coverage/width/score vs n for causal bounds (fixed X_eval).")
    parser.add_argument("--n_list", type=str, required=True, help="Comma-separated sample sizes, e.g. '200,500,1000'.")
    parser.add_argument("--m", type=int, default=5, help="Number of replicates per n.")
    parser.add_argument("--d", type=int, default=5, help="Feature dimension.")
    parser.add_argument(
        "--divergence",
        type=str,
        default="combined",
        help="Comma-separated divergences from {KL,TV,Hellinger,Chi2,JS,combined,cluster}.",
    )
    parser.add_argument(
        "--structural_type",
        type=str,
        default="nonlinear",
        choices=["linear", "nonlinear", "simpson", "cyclic", "cyclic2"],
        help="Data-generating process type.",
    )
    parser.add_argument("--base_seed", type=int, default=123, help="Base seed for training replicates.")
    parser.add_argument("--outdir", type=str, default="experiments", help="Output directory.")
    parser.add_argument("--unique_save", action="store_true", help="Add unique timestamp suffix to outputs.")

    # Evaluation set X_eval
    parser.add_argument("--n_eval", type=int, default=2000, help="Number of fixed evaluation points.")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="sample",
        choices=["sample", "grid_x0"],
        help="How to construct fixed X_eval: sample from DGP marginal or grid over X0.",
    )
    parser.add_argument("--eval_seed", type=int, default=2025, help="Seed for constructing X_eval (sample mode).")
    parser.add_argument("--x0_min", type=float, default=-3.14, help="Min X0 for grid_x0.")
    parser.add_argument("--x0_max", type=float, default=3.14, help="Max X0 for grid_x0.")
    parser.add_argument("--x_eval_fill", type=float, default=0.0, help="Fill value for non-X0 coordinates in grid_x0.")
    parser.add_argument("--x_range", type=float, default=2.0, help="x_range for some DGPs (if supported).")
    # Keep for backward compat; ignored if generate_data doesn't accept it.
    parser.add_argument(
        "--noise_dist",
        type=str,
        default=None,
        help="Optional noise distribution forwarded to generate_data if supported (e.g., 'normal', 't3').",
    )

    # Aggregation / uncertainty
    parser.add_argument(
        "--stat",
        "--width_stat",
        dest="width_stat",
        type=str,
        default="median",
        choices=["mean", "median"],
        help="Within-replicate aggregation over eval points for width (among valid points).",
    )
    parser.add_argument(
        "--stat_over_reps",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Across-replicate aggregation for centers.",
    )
    parser.add_argument(
        "--stat_grid",
        action="store_true",
        help="Run all combinations of --stat/--stat_over_reps in {mean,median}.",
    )
    parser.add_argument("--ci_alpha", type=float, default=0.05, help="Two-sided quantile band level across replicates.")

    # Score
    parser.add_argument("--score_lambda", type=float, default=10.0, help="Penalty lambda for score.")
    parser.add_argument("--score_alpha", type=float, default=0.05, help="Target shortfall alpha (target=1-alpha).")

    # Estimator controls
    parser.add_argument("--n_folds", type=int, default=2, help="CV folds.")
    parser.add_argument("--eps_propensity", type=float, default=1e-3, help="Propensity clipping.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Dual net epochs.")
    parser.add_argument("--xgb_n_jobs", type=int, default=-1, help="n_jobs for xgboost (-1 all cores).")
    parser.add_argument("--torch_threads", type=int, default=0, help="Torch intra-op threads (0 uses all cores).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for dual nets.")
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

    # Torch threads
    config_timer = StepTimer("configure experiment", use_tqdm=False, enabled=timing_enabled)
    config_timer.__enter__()
    torch_threads = args.torch_threads if args.torch_threads > 0 else max(1, os.cpu_count() or 1)
    try:
        torch.set_num_threads(torch_threads)
    except Exception:
        pass

    os.makedirs(args.outdir, exist_ok=True)
    stamp = _unique_suffix(args.unique_save)

    def name_with_suffix(base: str, ext: str) -> str:
        fname = f"{base}_{stamp}.{ext}" if stamp else f"{base}.{ext}"
        return os.path.join(args.outdir, fname)

    n_list = _parse_n_list(args.n_list)
    base_divs = ["KL", "TV", "Hellinger", "Chi2", "JS"]
    div_list = _parse_divergences(args.divergence, base_divs)

    # Build fixed evaluation set X_eval once.
    X_eval = _make_X_eval(
        n_eval=args.n_eval,
        d=args.d,
        eval_mode=args.eval_mode,
        eval_seed=args.eval_seed,
        structural_type=args.structural_type,
        x_range=args.x_range,
        noise_dist=args.noise_dist,
        x0_min=args.x0_min,
        x0_max=args.x0_max,
        x_fill=args.x_eval_fill,
    )
    n_eval = int(X_eval.shape[0])

    # Estimator configs (match plot1-style defaults)
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
        "batch_size": 256,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "max_grad_norm": 10.0,
        "eps_propensity": args.eps_propensity,
        "deterministic_torch": True,
        "train_m_on_fold": True,
        "propensity_config": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.01,
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
            "max_depth": 6,
            "learning_rate": 0.01,
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

    # Determine which base divergences we need to compute.
    needs_base = any(div in {"combined", "cluster"} for div in div_list)
    required_base = base_divs if needs_base else [d for d in div_list if d in base_divs]

    # Storage (per n, per div): arrays are (m, n_eval)
    upper_store = {n: {div: np.full((args.m, n_eval), np.nan, dtype=np.float32) for div in div_list} for n in n_list}
    lower_store = {n: {div: np.full((args.m, n_eval), np.nan, dtype=np.float32) for div in div_list} for n in n_list}
    width_store = {n: {div: np.full((args.m, n_eval), np.nan, dtype=np.float32) for div in div_list} for n in n_list}
    cover_store = {n: {div: np.zeros((args.m, n_eval), dtype=np.int32) for div in div_list} for n in n_list}
    valid_store = {n: {div: np.zeros((args.m, n_eval), dtype=np.int32) for div in div_list} for n in n_list}
    theta_store = {n: np.full((args.m, n_eval), np.nan, dtype=np.float32) for n in n_list}

    seeds_used: List[int] = []

    ci_alpha = float(args.ci_alpha)

    warn_state: Dict[str, bool] = {}

    for idx_n, n in enumerate(n_list):
        n_timer = StepTimer(f"n={n} loop", use_tqdm=True, enabled=timing_enabled)
        n_timer.__enter__()
        try:
            for j in tqdm(range(args.m), desc=f"n={n}", leave=False):
                rep_timer = StepTimer(f"replicate n={n} rep={j}", use_tqdm=True, enabled=timing_enabled)
                rep_timer.__enter__()
                try:
                    # Seed schedule: ensure different n use different seed blocks.
                    seed_tr = int(args.base_seed + 100000 * idx_n + j)
                    seeds_used.append(seed_tr)

                    data_timer = StepTimer("data generation", use_tqdm=True, enabled=timing_enabled)
                    data_timer.__enter__()
                    data = _call_generate_data_compat(
                        n=n,
                        d=args.d,
                        seed=seed_tr,
                        structural_type=args.structural_type,
                        x_range=args.x_range,
                        noise_dist=args.noise_dist,
                        _warn_state=warn_state,
                    )
                    X_tr = np.asarray(data["X"], dtype=np.float32)
                    A_tr = np.asarray(data["A"], dtype=np.float32)
                    Y_tr = np.asarray(data["Y"], dtype=np.float32)

                    # Truth evaluated at fixed X_eval (replicate-specific GroundTruth)
                    theta_eval = np.asarray(data["GroundTruth"](1, X_eval), dtype=np.float32).reshape(-1)
                    if theta_eval.shape[0] != n_eval:
                        raise RuntimeError(f"GroundTruth returned shape {theta_eval.shape}, expected ({n_eval},).")
                    theta_store[n][j, :] = theta_eval
                    data_timer.__exit__(None, None, None)

                    nuisance_timer = StepTimer("nuisance fits", use_tqdm=True, enabled=timing_enabled)
                    nuisance_timer.__enter__()
                    # Prefit/cached propensity on TRAINING data only.
                    prop_cache = prefit_propensity_cache(
                        X=X_tr,
                        A=A_tr,
                        propensity_model=propensity_model,
                        propensity_config=fit_config["propensity_config"],
                        n_folds=fit_config["n_folds"],
                        seed=seed_tr,
                        eps_propensity=fit_config["eps_propensity"],
                    )
                    nuisance_timer.__exit__(None, None, None)

                    bounds_timer = StepTimer("bounds per divergence", use_tqdm=True, enabled=timing_enabled)
                    bounds_timer.__enter__()
                    # Fit required base divergences once per (n,j), evaluate at X_eval.
                    base_outputs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                    for dname in required_base:
                        if timing_enabled and timing_detail:
                            with StepTimer(f"fit div={dname}", use_tqdm=True, enabled=True):
                                L_div, U_div = fit_two_pass_do1_cached(
                                    DebiasedCausalBoundEstimator,
                                    dname,
                                    X_train=X_tr,
                                    A_train=A_tr,
                                    Y_train=Y_tr,
                                    X_eval=X_eval,
                                    dual_net_config=dual_net_config,
                                    fit_config=fit_config,
                                    seed=seed_tr,
                                    propensity_model=propensity_model,
                                    m_model=m_model,
                                    propensity_cache=prop_cache,
                                )
                        else:
                            L_div, U_div = fit_two_pass_do1_cached(
                                DebiasedCausalBoundEstimator,
                                dname,
                                X_train=X_tr,
                                A_train=A_tr,
                                Y_train=Y_tr,
                                X_eval=X_eval,
                                dual_net_config=dual_net_config,
                                fit_config=fit_config,
                                seed=seed_tr,
                                propensity_model=propensity_model,
                                m_model=m_model,
                                propensity_cache=prop_cache,
                            )
                        base_outputs[dname] = (L_div, U_div)
                    bounds_timer.__exit__(None, None, None)

                    coverage_timer = StepTimer("coverage/width computation", use_tqdm=True, enabled=timing_enabled)
                    coverage_timer.__enter__()
                    # Now compute requested divergences (including aggregators).
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
                        else:
                            # Should not happen due to parsing, but keep safe.
                            L, U = fit_two_pass_do1_cached(
                                DebiasedCausalBoundEstimator,
                                div,
                                X_train=X_tr,
                                A_train=A_tr,
                                Y_train=Y_tr,
                                X_eval=X_eval,
                                dual_net_config=dual_net_config,
                                fit_config=fit_config,
                                seed=seed_tr,
                                propensity_model=propensity_model,
                                m_model=m_model,
                                propensity_cache=prop_cache,
                            )

                        width = U - L
                        valid = np.isfinite(L) & np.isfinite(U) & np.isfinite(theta_eval) & (width > 0)
                        covered = valid & (theta_eval >= L) & (theta_eval <= U)

                        # Save stores
                        upper_store[n][div][j, :] = U
                        lower_store[n][div][j, :] = L
                        width_store[n][div][j, :] = width
                        valid_store[n][div][j, :] = valid.astype(np.int32)
                        cover_store[n][div][j, :] = covered.astype(np.int32)
                    coverage_timer.__exit__(None, None, None)
                finally:
                    rep_timer.__exit__(None, None, None)
        finally:
            n_timer.__exit__(None, None, None)

    scalars_timer = StepTimer("precompute replicate scalars", use_tqdm=False, enabled=timing_enabled)
    scalars_timer.__enter__()
    # Precompute per-replicate scalars to avoid repeated reductions.
    covu_store = {n: {div: np.full((args.m,), np.nan, dtype=np.float64) for div in div_list} for n in n_list}
    covc_store = {n: {div: np.full((args.m,), np.nan, dtype=np.float64) for div in div_list} for n in n_list}
    valid_rate_store = {n: {div: np.full((args.m,), np.nan, dtype=np.float64) for div in div_list} for n in n_list}
    n_valid_store = {n: {div: np.zeros((args.m,), dtype=np.int32) for div in div_list} for n in n_list}
    width_mean_store = {n: {div: np.full((args.m,), np.nan, dtype=np.float64) for div in div_list} for n in n_list}
    width_median_store = {n: {div: np.full((args.m,), np.nan, dtype=np.float64) for div in div_list} for n in n_list}
    score_mean_store = {n: {div: np.full((args.m,), np.nan, dtype=np.float64) for div in div_list} for n in n_list}
    score_median_store = {n: {div: np.full((args.m,), np.nan, dtype=np.float64) for div in div_list} for n in n_list}

    for n in n_list:
        for div in div_list:
            for j in range(args.m):
                width = width_store[n][div][j, :]
                valid = valid_store[n][div][j, :].astype(bool)
                covered = cover_store[n][div][j, :].astype(bool)

                covu = float(np.mean(covered))
                covc = float(np.mean(covered[valid])) if np.any(valid) else float("nan")
                valid_rate = float(np.mean(valid))
                n_valid = int(np.sum(valid))
                width_mean = _stat_reduce(width[valid], "mean")
                width_median = _stat_reduce(width[valid], "median")
                score_mean = _score_penalized_width(
                    width=width_mean,
                    coverage_uncond=covu,
                    lam=args.score_lambda,
                    alpha=args.score_alpha,
                )
                score_median = _score_penalized_width(
                    width=width_median,
                    coverage_uncond=covu,
                    lam=args.score_lambda,
                    alpha=args.score_alpha,
                )

                covu_store[n][div][j] = covu
                covc_store[n][div][j] = covc
                valid_rate_store[n][div][j] = valid_rate
                n_valid_store[n][div][j] = n_valid
                width_mean_store[n][div][j] = width_mean
                width_median_store[n][div][j] = width_median
                score_mean_store[n][div][j] = score_mean
                score_median_store[n][div][j] = score_median
    scalars_timer.__exit__(None, None, None)

    def _stat_suffix(stat_within: str, stat_over_reps: str) -> str:
        return f"stat_{stat_within}_over_{stat_over_reps}"

    def _build_rows(stat_within: str, stat_over_reps: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        replicate_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        for idx_n, n in enumerate(n_list):
            for div in div_list:
                covu_list: List[float] = []
                covc_list: List[float] = []
                wid_list: List[float] = []
                sco_list: List[float] = []
                val_list: List[float] = []
                for j in range(args.m):
                    seed_tr = int(args.base_seed + 100000 * idx_n + j)
                    coverage_uncond_j = float(covu_store[n][div][j])
                    coverage_cond_j = float(covc_store[n][div][j])
                    valid_rate_j = float(valid_rate_store[n][div][j])
                    n_valid_j = int(n_valid_store[n][div][j])
                    if stat_within == "mean":
                        width_j = float(width_mean_store[n][div][j])
                        score_j = float(score_mean_store[n][div][j])
                    else:
                        width_j = float(width_median_store[n][div][j])
                        score_j = float(score_median_store[n][div][j])

                    replicate_rows.append(
                        {
                            "divergence": div,
                            "n": int(n),
                            "rep": int(j),
                            "seed": int(seed_tr),
                            "n_eval": int(n_eval),
                            "coverage_uncond": coverage_uncond_j,
                            "coverage_cond": coverage_cond_j,
                            "width": width_j,
                            "score": score_j,
                            "valid_rate": valid_rate_j,
                            "n_valid": n_valid_j,
                        }
                    )

                    covu_list.append(coverage_uncond_j)
                    covc_list.append(coverage_cond_j)
                    wid_list.append(width_j)
                    sco_list.append(score_j)
                    val_list.append(valid_rate_j)

                covu = np.array(covu_list, dtype=float)
                covc = np.array(covc_list, dtype=float)
                wid = np.array(wid_list, dtype=float)
                sco = np.array(sco_list, dtype=float)
                val = np.array(val_list, dtype=float)

                summary_rows.append(
                    {
                        "divergence": div,
                        "n": int(n),
                        "coverage_uncond_center": _stat_reduce(covu, stat_over_reps),
                        "coverage_uncond_ci_low": _nan_quantile(covu, ci_alpha / 2),
                        "coverage_uncond_ci_high": _nan_quantile(covu, 1 - ci_alpha / 2),
                        "coverage_cond_center": _stat_reduce(covc, stat_over_reps),
                        "coverage_cond_ci_low": _nan_quantile(covc, ci_alpha / 2),
                        "coverage_cond_ci_high": _nan_quantile(covc, 1 - ci_alpha / 2),
                        "width_center": _stat_reduce(wid, stat_over_reps),
                        "width_ci_low": _nan_quantile(wid, ci_alpha / 2),
                        "width_ci_high": _nan_quantile(wid, 1 - ci_alpha / 2),
                        "score_center": _stat_reduce(sco, stat_over_reps),
                        "score_ci_low": _nan_quantile(sco, ci_alpha / 2),
                        "score_ci_high": _nan_quantile(sco, 1 - ci_alpha / 2),
                        "valid_center": _stat_reduce(val, stat_over_reps),
                        "valid_ci_low": _nan_quantile(val, ci_alpha / 2),
                        "valid_ci_high": _nan_quantile(val, 1 - ci_alpha / 2),
                        "m": int(args.m),
                        "d": int(args.d),
                        "n_eval": int(n_eval),
                        "structural_type": args.structural_type,
                        "eval_mode": args.eval_mode,
                        "width_stat": stat_within,
                        "stat_over_reps": stat_over_reps,
                        "score_lambda": float(args.score_lambda),
                        "score_alpha": float(args.score_alpha),
                        "ci_alpha": float(args.ci_alpha),
                    }
                )

        return replicate_rows, summary_rows

    stat_pairs = [(args.width_stat, args.stat_over_reps)]
    if args.stat_grid:
        stat_pairs = [("mean", "mean"), ("mean", "median"), ("median", "mean"), ("median", "median")]

    use_suffix = len(stat_pairs) > 1
    replicate_rows_by_stat: Dict[str, List[Dict[str, Any]]] = {}
    summary_rows_by_stat: Dict[str, List[Dict[str, Any]]] = {}
    runs: List[Dict[str, Any]] = []
    default_key = None
    default_files: Dict[str, str] = {}

    # Plots per metric with multiple divergences
    color_map = {
        "combined": "tab:blue",
        "cluster": "tab:orange",
        "KL": "tab:green",
        "TV": "tab:red",
        "Hellinger": "tab:purple",
        "Chi2": "tab:brown",
        "JS": "tab:pink",
    }

    def _plot_metric(rows: List[Dict[str, Any]], metric_key: str, ylabel: str, title: str, fname: str) -> str:
        plt.figure(figsize=(7.2, 4.2))
        for div in div_list:
            sub = [row for row in rows if row["divergence"] == div]
            xs = [row["n"] for row in sub]
            ys = [row[f"{metric_key}_center"] for row in sub]
            lo = [row[f"{metric_key}_ci_low"] for row in sub]
            hi = [row[f"{metric_key}_ci_high"] for row in sub]
            c = color_map.get(div, None)
            yerr = [np.subtract(ys, lo), np.subtract(hi, ys)]
            plt.errorbar(xs, ys, yerr=yerr, marker="o", linestyle="-", color=c, capsize=3, label=div)
        plt.xlabel("Sample size n")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        path = name_with_suffix(fname, "png")
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    replicate_rows = []
    summary_rows = []
    rep_path = ""
    summary_path = ""
    covu_fig = ""
    covc_fig = ""
    valid_fig = ""
    width_fig = ""
    score_fig = ""

    for stat_within, stat_over_reps in stat_pairs:
        stat_timer = StepTimer(
            f"summary+plots stat={stat_within}/{stat_over_reps}",
            use_tqdm=False,
            enabled=timing_enabled,
        )
        stat_timer.__enter__()
        suffix = _stat_suffix(stat_within, stat_over_reps) if use_suffix else ""
        key = suffix or "default"
        rep_rows, sum_rows = _build_rows(stat_within, stat_over_reps)
        replicate_rows_by_stat[key] = rep_rows
        summary_rows_by_stat[key] = sum_rows

        rep_base = "plot_n_coverage_replicates"
        summary_base = "plot_n_coverage_summary"
        covu_base = "plot_n_coverage_cov_uncond"
        covc_base = "plot_n_coverage_cov_cond"
        valid_base = "plot_n_coverage_valid"
        width_base = "plot_n_coverage_width"
        score_base = "plot_n_coverage_score"
        if suffix:
            rep_base = f"{rep_base}_{suffix}"
            summary_base = f"{summary_base}_{suffix}"
            covu_base = f"{covu_base}_{suffix}"
            covc_base = f"{covc_base}_{suffix}"
            valid_base = f"{valid_base}_{suffix}"
            width_base = f"{width_base}_{suffix}"
            score_base = f"{score_base}_{suffix}"

        rep_path_run = name_with_suffix(rep_base, "csv")
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(rep_rows).to_csv(rep_path_run, index=False)
        except Exception:
            header = "divergence,n,rep,seed,n_eval,coverage_uncond,coverage_cond,width,score,valid_rate,n_valid"
            with open(rep_path_run, "w") as f:
                f.write(header + "\n")
                for row in rep_rows:
                    f.write(
                        f"{row['divergence']},{row['n']},{row['rep']},{row['seed']},{row['n_eval']},"
                        f"{row['coverage_uncond']},{row['coverage_cond']},{row['width']},{row['score']},"
                        f"{row['valid_rate']},{row['n_valid']}\n"
                    )

        summary_path_run = name_with_suffix(summary_base, "csv")
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(sum_rows).to_csv(summary_path_run, index=False)
        except Exception:
            header = (
                "divergence,n,"
                "coverage_uncond_center,coverage_uncond_ci_low,coverage_uncond_ci_high,"
                "coverage_cond_center,coverage_cond_ci_low,coverage_cond_ci_high,"
                "width_center,width_ci_low,width_ci_high,"
                "score_center,score_ci_low,score_ci_high,"
                "valid_center,valid_ci_low,valid_ci_high,"
                "m,d,n_eval,structural_type,eval_mode,width_stat,stat_over_reps,score_lambda,score_alpha,ci_alpha"
            )
            with open(summary_path_run, "w") as f:
                f.write(header + "\n")
                for row in sum_rows:
                    f.write(
                        f"{row['divergence']},{row['n']},"
                        f"{row['coverage_uncond_center']},{row['coverage_uncond_ci_low']},{row['coverage_uncond_ci_high']},"
                        f"{row['coverage_cond_center']},{row['coverage_cond_ci_low']},{row['coverage_cond_ci_high']},"
                        f"{row['width_center']},{row['width_ci_low']},{row['width_ci_high']},"
                        f"{row['score_center']},{row['score_ci_low']},{row['score_ci_high']},"
                        f"{row['valid_center']},{row['valid_ci_low']},{row['valid_ci_high']},"
                        f"{row['m']},{row['d']},{row['n_eval']},{row['structural_type']},{row['eval_mode']},"
                        f"{row['width_stat']},{row['stat_over_reps']},{row['score_lambda']},{row['score_alpha']},{row['ci_alpha']}\n"
                    )

        covu_fig_run = _plot_metric(
            sum_rows,
            "coverage_uncond",
            ylabel="Coverage (unconditional)",
            title=f"Coverage (unconditional) vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=covu_base,
        )
        covc_fig_run = _plot_metric(
            sum_rows,
            "coverage_cond",
            ylabel="Coverage (conditional on valid)",
            title=f"Coverage (conditional) vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=covc_base,
        )
        valid_fig_run = _plot_metric(
            sum_rows,
            "valid",
            ylabel="Valid rate",
            title=f"Valid rate vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=valid_base,
        )
        width_fig_run = _plot_metric(
            sum_rows,
            "width",
            ylabel="Width",
            title=f"Width vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=width_base,
        )
        score_fig_run = _plot_metric(
            sum_rows,
            "score",
            ylabel="Score (penalized width)",
            title=f"Score vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=score_base,
        )

        run_files = {
            "replicate_csv": rep_path_run,
            "summary_csv": summary_path_run,
            "coverage_uncond_png": covu_fig_run,
            "coverage_cond_png": covc_fig_run,
            "valid_png": valid_fig_run,
            "width_png": width_fig_run,
            "score_png": score_fig_run,
        }
        runs.append(
            {
                "width_stat": stat_within,
                "stat_over_reps": stat_over_reps,
                "files": run_files,
            }
        )

        if stat_within == args.width_stat and stat_over_reps == args.stat_over_reps and default_key is None:
            default_key = key
            replicate_rows = rep_rows
            summary_rows = sum_rows
            rep_path = rep_path_run
            summary_path = summary_path_run
            covu_fig = covu_fig_run
            covc_fig = covc_fig_run
            valid_fig = valid_fig_run
            width_fig = width_fig_run
            score_fig = score_fig_run
            default_files = run_files
        stat_timer.__exit__(None, None, None)

    if default_key is None and runs:
        default_key = list(summary_rows_by_stat.keys())[0]
        replicate_rows = replicate_rows_by_stat[default_key]
        summary_rows = summary_rows_by_stat[default_key]
        rep_path = runs[0]["files"]["replicate_csv"]
        summary_path = runs[0]["files"]["summary_csv"]
        covu_fig = runs[0]["files"]["coverage_uncond_png"]
        covc_fig = runs[0]["files"]["coverage_cond_png"]
        valid_fig = runs[0]["files"]["valid_png"]
        width_fig = runs[0]["files"]["width_png"]
        score_fig = runs[0]["files"]["score_png"]
        default_files = runs[0]["files"]

    save_timer = StepTimer("save outputs", use_tqdm=False, enabled=timing_enabled)
    save_timer.__enter__()
    # Artifacts
    artifacts = {
        "X_eval": X_eval,
        "upper_store": upper_store,
        "lower_store": lower_store,
        "width_store": width_store,
        "cover_store": cover_store,
        "valid_store": valid_store,
        "theta_store": theta_store,
        "args": vars(args),
        "fit_config": fit_config,
        "dual_net_config": dual_net_config,
        "seeds": seeds_used,
        "summary_rows": summary_rows,
        "replicate_rows": replicate_rows,
        "summary_rows_by_stat": summary_rows_by_stat,
        "replicate_rows_by_stat": replicate_rows_by_stat,
        "stat_pairs": stat_pairs,
        "timestamp": stamp,
        "divergences": div_list,
    }
    artifacts_path = name_with_suffix("plot_n_coverage_artifacts", "pkl")
    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    files = dict(default_files)
    files["artifacts_pkl"] = artifacts_path

    summary_json = {
        "timestamp": stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "files": files,
        "args": vars(args),
        "divergences": div_list,
        "structural_type": args.structural_type,
        "eval_mode": args.eval_mode,
        "n_eval": n_eval,
    }
    if len(runs) > 1:
        summary_json["runs"] = runs
    summary_json_path = name_with_suffix("plot_n_coverage_summary", "json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2)

    log_line = (
        f"[plot_n_coverage] ts={summary_json['timestamp']} divs={','.join(div_list)} struct={args.structural_type} "
        f"eval={args.eval_mode} n_eval={n_eval} n_list={n_list} m={args.m} "
        f"stat_grid={args.stat_grid} args={vars(args)} files={summary_json['files']}"
    )
    log_path = name_with_suffix("plot_n_coverage_log", "txt")
    with open(log_path, "a") as f:
        f.write(log_line + "\n")
    save_timer.__exit__(None, None, None)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log_active_step_error(use_tqdm=False)
        raise
