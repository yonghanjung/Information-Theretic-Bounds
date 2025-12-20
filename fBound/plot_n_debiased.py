"""
Monte Carlo experiment: slow propensity vs debiased/naive bound error rates.

Fits debiased and naive estimators on samples of size n, evaluates bounds on a
fixed X_eval grid, and compares against an oracle bound from a large sample.
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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **k: x  # type: ignore

from causal_bound import DebiasedCausalBoundEstimator, _concat_ax, _predict_proba_class1, prefit_propensity_cache
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
# Naive estimator: omit debiasing correction term
# -------------------------
class NaiveCausalBoundEstimator(DebiasedCausalBoundEstimator):
    def _debiased_loss_batch(self, X, A, Y, e1, e0, h_net, u_net):
        ax = _concat_ax(A, X)
        h_ax = h_net(ax)
        u_ax = u_net(ax)
        h_ax = torch.clamp(h_ax, min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)
        lam_ax = torch.exp(h_ax)

        phi_y = self.phi(Y)
        t = (phi_y - u_ax) / lam_ax
        g_star_val = self.divergence.g_star(t)

        eA = torch.where(A >= 0.5, e1, e0)
        eta = self.divergence.B_torch(eA)

        main = lam_ax * (eta + g_star_val) + u_ax
        loss = main.mean()

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered (NAIVE). divergence={self.divergence.name}."
            )
        return loss


# -------------------------
# Combined (c-wise) intersection aggregator
# -------------------------
def combined_cwise_intersection(lower_mat: np.ndarray, upper_mat: np.ndarray, c: int = 3) -> tuple[np.ndarray, np.ndarray]:
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
# Two-pass fit helper (train on X, evaluate on X_eval)
# -------------------------
def fit_two_pass_do1_cached(
    EstimatorClass,
    div_name,
    X_train,
    A_train,
    Y_train,
    X_eval,
    dual_net_config,
    fit_config,
    seed,
    propensity_model,
    m_model,
    propensity_cache,
    progress_prefix: Optional[str] = None,
):
    if progress_prefix:
        print(f"{progress_prefix} divergence={div_name} pass=pos")
    est_pos = EstimatorClass(
        divergence=div_name,
        phi=phi_identity,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
    ).fit(X_train, A_train, Y_train, propensity_cache=propensity_cache)

    if progress_prefix:
        print(f"{progress_prefix} divergence={div_name} pass=neg")
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
    return L, U, est_pos


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
    if not np.isfinite(width) or not np.isfinite(coverage_uncond):
        return float("nan")
    target = 1.0 - float(alpha)
    shortfall = max(0.0, target - float(coverage_uncond))
    return float(width * (1.0 + float(lam) * shortfall))


def _stat_suffix(stat_within: str, stat_over_reps: str) -> str:
    return f"stat_{stat_within}_over_{stat_over_reps}"


def _call_generate_data_compat(
    n: int,
    d: int,
    seed: int,
    structural_type: str,
    x_range: Optional[float] = None,
    noise_dist: Optional[str] = None,
    _warn_state: Optional[Dict[str, bool]] = None,
) -> Dict[str, object]:
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


class _NoisyPropensityModel:
    def __init__(self, base_model: Any, noise_mean: float, noise_std: float, seed: int, eps: float) -> None:
        self.base_model = base_model
        self.noise_mean = float(noise_mean)
        self.noise_std = float(noise_std)
        self.eps = float(eps)
        self.rng = np.random.default_rng(int(seed))
        self.classes_ = getattr(base_model, "classes_", np.asarray([0, 1], dtype=int))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.base_model.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(f"predict_proba must return shape (n,2+). Got {proba.shape}")
        col = 1
        classes = getattr(self.base_model, "classes_", None)
        if classes is not None:
            classes = np.asarray(classes)
            if 1 in classes:
                col = int(np.where(classes == 1)[0][0])
        p1 = proba[:, col]
        if self.noise_std != 0.0 or self.noise_mean != 0.0:
            noise = self.rng.normal(loc=self.noise_mean, scale=self.noise_std, size=p1.shape)
            p1 = np.clip(p1 + noise, self.eps, 1.0 - self.eps)
        else:
            p1 = np.clip(p1, self.eps, 1.0 - self.eps)

        classes = np.asarray(self.classes_)
        if 1 in classes and 0 in classes:
            idx1 = int(np.where(classes == 1)[0][0])
            idx0 = int(np.where(classes == 0)[0][0])
            out = np.array(proba, copy=True)
            out[:, idx1] = p1
            out[:, idx0] = 1.0 - p1
            return out

        out = np.zeros_like(proba)
        out[:, 0] = 1.0 - p1
        out[:, 1] = p1
        return out


def _apply_propensity_noise(
    prop_cache: Dict[str, Any],
    n: int,
    eps: float,
    noise_beta: float,
    enabled: bool,
    seed: int,
) -> Dict[str, Any]:
    if not enabled:
        return prop_cache

    scale = float(n) ** (-float(noise_beta))
    mean = scale
    std = scale
    rng = np.random.default_rng(int(seed) + 7777)

    e1_oof = np.asarray(prop_cache["e1_oof"], dtype=np.float32).copy()
    noise = rng.normal(loc=mean, scale=std, size=e1_oof.shape).astype(np.float32)
    e1_oof = np.clip(e1_oof + noise, eps, 1.0 - eps).astype(np.float32)

    models = prop_cache["models"]
    noisy_models = []
    for k, m in enumerate(models):
        noisy_models.append(_NoisyPropensityModel(m, mean, std, seed + 991 * (k + 1), eps))

    return {
        "splits": prop_cache["splits"],
        "fold_id": prop_cache["fold_id"],
        "e1_oof": e1_oof,
        "models": noisy_models,
        "seed": prop_cache.get("seed", int(seed)),
        "n_folds": prop_cache.get("n_folds", len(prop_cache["splits"])),
        "eps_propensity": float(eps),
    }


def _predict_propensity_mean(models: List[Any], X: np.ndarray, eps: float) -> np.ndarray:
    if not models:
        raise ValueError("No propensity models provided.")
    preds = np.zeros((X.shape[0],), dtype=np.float64)
    X64 = X.astype(np.float64, copy=False)
    for m in models:
        preds += _predict_proba_class1(m, X64)
    preds /= float(len(models))
    preds = np.clip(preds, eps, 1.0 - eps)
    return preds.astype(np.float32)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def _fit_bounds_for_divs(
    EstimatorClass,
    div_list,
    base_divs,
    X,
    A,
    Y,
    X_eval,
    dual_net_config,
    fit_config,
    seed,
    propensity_model,
    m_model,
    prop_cache,
    progress_prefix: Optional[str] = None,
    timing_label: Optional[str] = None,
    timing: bool = False,
    timing_detail: bool = False,
    use_tqdm: bool = False,
):
    outputs = {}
    needs_base = any(div in {"combined", "cluster"} for div in div_list)
    base_outputs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    label = timing_label or progress_prefix or ""
    prefix = f"{label} " if label else ""
    if needs_base or any(dv in base_divs for dv in div_list):
        for dname in base_divs:
            if dname in div_list or needs_base:
                with StepTimer(
                    f"{prefix}fit {dname}".strip(),
                    use_tqdm=use_tqdm,
                    enabled=timing and timing_detail,
                ):
                    L_div, U_div, _ = fit_two_pass_do1_cached(
                        EstimatorClass,
                        dname,
                        X,
                        A,
                        Y,
                        X_eval,
                        dual_net_config,
                        fit_config,
                        seed=seed,
                        propensity_model=propensity_model,
                        m_model=m_model,
                        propensity_cache=prop_cache,
                        progress_prefix=progress_prefix,
                    )
                base_outputs[dname] = (L_div, U_div)

    lower_base = None
    upper_base = None
    for div in div_list:
        if div in base_outputs:
            L, U = base_outputs[div]
        elif div == "combined":
            with StepTimer(
                f"{prefix}aggregate combined".strip(),
                use_tqdm=use_tqdm,
                enabled=timing and timing_detail,
            ):
                if lower_base is None:
                    lower_base = np.vstack([base_outputs[b][0] for b in base_divs])
                    upper_base = np.vstack([base_outputs[b][1] for b in base_divs])
                L, U = combined_cwise_intersection(lower_base, upper_base, c=3)
        elif div == "cluster":
            with StepTimer(
                f"{prefix}aggregate cluster".strip(),
                use_tqdm=use_tqdm,
                enabled=timing and timing_detail,
            ):
                if lower_base is None:
                    lower_base = np.vstack([base_outputs[b][0] for b in base_divs])
                    upper_base = np.vstack([base_outputs[b][1] for b in base_divs])
                L, U = cluster_per_sample_fast1d(lower_base, upper_base, k_candidates=(2, 3, 4), penalty_singleton=0.2)
        else:
            with StepTimer(
                f"{prefix}fit {div}".strip(),
                use_tqdm=use_tqdm,
                enabled=timing and timing_detail,
            ):
                L, U, _ = fit_two_pass_do1_cached(
                    EstimatorClass,
                    div,
                    X,
                    A,
                    Y,
                    X_eval,
                    dual_net_config,
                    fit_config,
                    seed=seed,
                    propensity_model=propensity_model,
                    m_model=m_model,
                    propensity_cache=prop_cache,
                    progress_prefix=progress_prefix,
                )
        outputs[div] = (L.astype(np.float32), U.astype(np.float32))

    return outputs


# -------------------------
# Main
# -------------------------
def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Slow nuisance experiment: debiased vs naive error rates.")
    parser.add_argument("--n_list", type=str, required=True, help="Comma-separated sample sizes, e.g. '500,1000,2000'.")
    parser.add_argument("--m", type=int, default=30, help="Number of replicates per n.")
    parser.add_argument("--d", type=int, default=5, help="Feature dimension.")
    parser.add_argument(
        "--divergence",
        type=str,
        default="cluster",
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
    parser.add_argument("--n_eval", type=int, default=5000, help="Number of fixed evaluation points.")
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
        default="mean",
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

    # Oracle settings
    parser.add_argument("--n_oracle", type=int, default=200000, help="Oracle sample size for near-truth bounds.")
    parser.add_argument("--oracle_seed", type=int, default=-1, help="Seed for oracle data (negative uses base_seed).")
    parser.add_argument("--oracle_propensity_n_estimators", type=int, default=0, help="Override oracle propensity n_estimators.")
    parser.add_argument("--oracle_propensity_max_depth", type=int, default=0, help="Override oracle propensity max_depth.")
    parser.add_argument("--oracle_num_epochs", type=int, default=50, help="Oracle num_epochs (default 50).")
    parser.add_argument("--oracle_batch_size", type=int, default=1024, help="Oracle batch_size (default 1024).")

    # Slow propensity noise (optional)
    parser.add_argument("--propensity_noise", action="store_true", help="Add N(n^{-beta}, n^{-beta}) noise to propensity.")
    parser.add_argument("--propensity_noise_beta", type=float, default=0.25, help="Noise rate beta for n^{-beta}.")
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

    os.makedirs(args.outdir, exist_ok=True)
    stamp = _unique_suffix(args.unique_save)

    def name_with_suffix(base: str, ext: str) -> str:
        fname = f"{base}_{stamp}.{ext}" if stamp else f"{base}.{ext}"
        return os.path.join(args.outdir, fname)

    n_list = _parse_n_list(args.n_list)
    base_divs = ["KL", "TV", "Hellinger", "Chi2", "JS"]
    div_list = _parse_divergences(args.divergence, base_divs)

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

    # Oracle configuration (optional overrides)
    oracle_fit_config = dict(fit_config)
    oracle_prop_cfg = dict(fit_config["propensity_config"])
    if args.oracle_propensity_n_estimators > 0:
        oracle_prop_cfg["n_estimators"] = int(args.oracle_propensity_n_estimators)
    if args.oracle_propensity_max_depth > 0:
        oracle_prop_cfg["max_depth"] = int(args.oracle_propensity_max_depth)
    oracle_fit_config["propensity_config"] = oracle_prop_cfg
    oracle_fit_config["num_epochs"] = int(args.oracle_num_epochs)
    oracle_fit_config["batch_size"] = int(args.oracle_batch_size)
    config_timer.__exit__(None, None, None)

    warn_state: Dict[str, bool] = {}

    with StepTimer("oracle setup", use_tqdm=False, enabled=timing_enabled):
        # Oracle bounds and propensity (near-truth)
        oracle_seed = int(args.oracle_seed) if args.oracle_seed >= 0 else int(args.base_seed)
        print("Oracle estimator is being computed...")
        data_oracle = _call_generate_data_compat(
            n=int(args.n_oracle),
            d=args.d,
            seed=oracle_seed,
            structural_type=args.structural_type,
            x_range=args.x_range,
            noise_dist=args.noise_dist,
            _warn_state=warn_state,
        )
        X_oracle = np.asarray(data_oracle["X"], dtype=np.float32)
        A_oracle = np.asarray(data_oracle["A"], dtype=np.float32)
        Y_oracle = np.asarray(data_oracle["Y"], dtype=np.float32)

        print("Fitting oracle propensity cache...")
        prop_cache_oracle = prefit_propensity_cache(
            X=X_oracle,
            A=A_oracle,
            propensity_model=propensity_model,
            propensity_config=oracle_fit_config["propensity_config"],
            n_folds=oracle_fit_config["n_folds"],
            seed=oracle_seed,
            eps_propensity=oracle_fit_config["eps_propensity"],
        )

        print(
            "Fitting oracle bounds..."
            f" (batch_size={oracle_fit_config['batch_size']}, num_epochs={oracle_fit_config['num_epochs']})"
        )
        oracle_bounds = _fit_bounds_for_divs(
            DebiasedCausalBoundEstimator,
            div_list,
            base_divs,
            X_oracle,
            A_oracle,
            Y_oracle,
            X_eval,
            dual_net_config,
            oracle_fit_config,
            seed=oracle_seed,
            propensity_model=propensity_model,
            m_model=m_model,
            prop_cache=prop_cache_oracle,
            progress_prefix="[oracle]",
            timing_label="oracle",
            timing=timing_enabled,
            timing_detail=timing_detail,
            use_tqdm=False,
        )
        U_oracle = {div: oracle_bounds[div][1] for div in div_list}
        e_oracle = _predict_propensity_mean(prop_cache_oracle["models"], X_eval, oracle_fit_config["eps_propensity"])
        print("Oracle estimator completed.")

    replicate_rows: List[Dict[str, Any]] = []
    seeds_used: List[int] = []

    total_reps = len(n_list) * args.m
    print(f"Running {total_reps} replicate fits...")
    n_timer = None
    current_n = None
    for t in tqdm(range(total_reps), desc="replicates", leave=False):
        idx_n = t // args.m
        j = t % args.m
        n = n_list[idx_n]
        if n != current_n:
            if n_timer is not None:
                n_timer.__exit__(None, None, None)
            current_n = n
            n_timer = StepTimer(f"n={n} loop", use_tqdm=True, enabled=timing_enabled)
            n_timer.__enter__()
        seed_tr = int(args.base_seed + 100000 * idx_n + j)
        seeds_used.append(seed_tr)
        with StepTimer(f"replicate n={n} rep={j}", use_tqdm=True, enabled=timing_enabled):
            with StepTimer("data generation", use_tqdm=True, enabled=timing_enabled):
                data = _call_generate_data_compat(
                    n=n,
                    d=args.d,
                    seed=seed_tr,
                    structural_type=args.structural_type,
                    x_range=args.x_range,
                    noise_dist=args.noise_dist,
                    _warn_state=warn_state,
                )
                X = np.asarray(data["X"], dtype=np.float32)
                A = np.asarray(data["A"], dtype=np.float32)
                Y = np.asarray(data["Y"], dtype=np.float32)
                theta_eval = np.asarray(data["GroundTruth"](1, X_eval), dtype=np.float32).reshape(-1)

            with StepTimer("nuisance fits", use_tqdm=True, enabled=timing_enabled):
                prop_cache = prefit_propensity_cache(
                    X=X,
                    A=A,
                    propensity_model=propensity_model,
                    propensity_config=fit_config["propensity_config"],
                    n_folds=fit_config["n_folds"],
                    seed=seed_tr,
                    eps_propensity=fit_config["eps_propensity"],
                )
                prop_cache = _apply_propensity_noise(
                    prop_cache,
                    n=n,
                    eps=fit_config["eps_propensity"],
                    noise_beta=args.propensity_noise_beta,
                    enabled=args.propensity_noise,
                    seed=seed_tr,
                )

                e_hat = _predict_propensity_mean(prop_cache["models"], X_eval, fit_config["eps_propensity"])
                prop_rmse = _rmse(e_hat, e_oracle)

            with StepTimer("debiased bounds", use_tqdm=True, enabled=timing_enabled):
                deb_bounds = _fit_bounds_for_divs(
                    DebiasedCausalBoundEstimator,
                    div_list,
                    base_divs,
                    X,
                    A,
                    Y,
                    X_eval,
                    dual_net_config,
                    fit_config,
                    seed=seed_tr,
                    propensity_model=propensity_model,
                    m_model=m_model,
                    prop_cache=prop_cache,
                    timing_label="debiased",
                    timing=timing_enabled,
                    timing_detail=timing_detail,
                    use_tqdm=True,
                )
            with StepTimer("naive bounds", use_tqdm=True, enabled=timing_enabled):
                nai_bounds = _fit_bounds_for_divs(
                    NaiveCausalBoundEstimator,
                    div_list,
                    base_divs,
                    X,
                    A,
                    Y,
                    X_eval,
                    dual_net_config,
                    fit_config,
                    seed=seed_tr,
                    propensity_model=propensity_model,
                    m_model=m_model,
                    prop_cache=prop_cache,
                    timing_label="naive",
                    timing=timing_enabled,
                    timing_detail=timing_detail,
                    use_tqdm=True,
                )

            with StepTimer("metrics", use_tqdm=True, enabled=timing_enabled):
                for div in div_list:
                    Ld, Ud = deb_bounds[div]
                    Ln, Un = nai_bounds[div]

                    valid_d = np.isfinite(Ld) & np.isfinite(Ud) & (Ud - Ld > 0)
                    valid_n = np.isfinite(Ln) & np.isfinite(Un) & (Un - Ln > 0)
                    cov_d = float(np.mean(valid_d & (theta_eval >= Ld) & (theta_eval <= Ud)))
                    cov_n = float(np.mean(valid_n & (theta_eval >= Ln) & (theta_eval <= Un)))

                    width_d_mean = float(np.nanmean((Ud - Ld)[valid_d])) if np.any(valid_d) else float("nan")
                    width_d_median = float(np.nanmedian((Ud - Ld)[valid_d])) if np.any(valid_d) else float("nan")
                    width_n_mean = float(np.nanmean((Un - Ln)[valid_n])) if np.any(valid_n) else float("nan")
                    width_n_median = float(np.nanmedian((Un - Ln)[valid_n])) if np.any(valid_n) else float("nan")
                    width_d = width_d_mean if args.width_stat == "mean" else width_d_median
                    width_n = width_n_mean if args.width_stat == "mean" else width_n_median

                    err_up_d = _rmse(Ud, U_oracle[div])
                    err_up_n = _rmse(Un, U_oracle[div])

                    score_d_mean = _score_penalized_width(width_d_mean, cov_d, args.score_lambda, args.score_alpha)
                    score_d_median = _score_penalized_width(width_d_median, cov_d, args.score_lambda, args.score_alpha)
                    score_n_mean = _score_penalized_width(width_n_mean, cov_n, args.score_lambda, args.score_alpha)
                    score_n_median = _score_penalized_width(width_n_median, cov_n, args.score_lambda, args.score_alpha)
                    score_d = score_d_mean if args.width_stat == "mean" else score_d_median
                    score_n = score_n_mean if args.width_stat == "mean" else score_n_median

                    replicate_rows.append(
                        {
                            "divergence": div,
                            "n": int(n),
                            "rep": int(j),
                            "seed": int(seed_tr),
                            "propensity_rmse": prop_rmse,
                            "err_up_debiased": err_up_d,
                            "err_up_naive": err_up_n,
                            "coverage_debiased": cov_d,
                            "coverage_naive": cov_n,
                            "width_debiased": width_d,
                            "width_debiased_mean": width_d_mean,
                            "width_debiased_median": width_d_median,
                            "width_naive": width_n,
                            "width_naive_mean": width_n_mean,
                            "width_naive_median": width_n_median,
                            "score_debiased": score_d,
                            "score_debiased_mean": score_d_mean,
                            "score_debiased_median": score_d_median,
                            "score_naive": score_n,
                            "score_naive_mean": score_n_mean,
                            "score_naive_median": score_n_median,
                            "valid_rate_debiased": float(np.mean(valid_d)),
                            "valid_rate_naive": float(np.mean(valid_n)),
                        }
                    )
    if n_timer is not None:
        n_timer.__exit__(None, None, None)

    def _build_summary(stat_within: str, stat_over_reps: str) -> List[Dict[str, Any]]:
        summary_rows: List[Dict[str, Any]] = []
        rows_by: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for row in replicate_rows:
            key = (row["divergence"], row["n"])
            rows_by.setdefault(key, []).append(row)
        for div in div_list:
            for n in n_list:
                rows = rows_by.get((div, int(n)), [])
                prop = np.array([r["propensity_rmse"] for r in rows], dtype=float)
                err_d = np.array([r["err_up_debiased"] for r in rows], dtype=float)
                err_n = np.array([r["err_up_naive"] for r in rows], dtype=float)
                cov_d = np.array([r["coverage_debiased"] for r in rows], dtype=float)
                cov_n = np.array([r["coverage_naive"] for r in rows], dtype=float)
                if stat_within == "mean":
                    wid_d = np.array([r["width_debiased_mean"] for r in rows], dtype=float)
                    wid_n = np.array([r["width_naive_mean"] for r in rows], dtype=float)
                    sco_d = np.array([r["score_debiased_mean"] for r in rows], dtype=float)
                    sco_n = np.array([r["score_naive_mean"] for r in rows], dtype=float)
                else:
                    wid_d = np.array([r["width_debiased_median"] for r in rows], dtype=float)
                    wid_n = np.array([r["width_naive_median"] for r in rows], dtype=float)
                    sco_d = np.array([r["score_debiased_median"] for r in rows], dtype=float)
                    sco_n = np.array([r["score_naive_median"] for r in rows], dtype=float)
                val_d = np.array([r["valid_rate_debiased"] for r in rows], dtype=float)
                val_n = np.array([r["valid_rate_naive"] for r in rows], dtype=float)

                summary_rows.append(
                    {
                        "divergence": div,
                        "n": int(n),
                        "propensity_rmse_center": _stat_reduce(prop, stat_over_reps),
                        "propensity_rmse_ci_low": _nan_quantile(prop, args.ci_alpha / 2),
                        "propensity_rmse_ci_high": _nan_quantile(prop, 1 - args.ci_alpha / 2),
                        "err_up_debiased_center": _stat_reduce(err_d, stat_over_reps),
                        "err_up_debiased_ci_low": _nan_quantile(err_d, args.ci_alpha / 2),
                        "err_up_debiased_ci_high": _nan_quantile(err_d, 1 - args.ci_alpha / 2),
                        "err_up_naive_center": _stat_reduce(err_n, stat_over_reps),
                        "err_up_naive_ci_low": _nan_quantile(err_n, args.ci_alpha / 2),
                        "err_up_naive_ci_high": _nan_quantile(err_n, 1 - args.ci_alpha / 2),
                        "coverage_debiased_center": _stat_reduce(cov_d, stat_over_reps),
                        "coverage_debiased_ci_low": _nan_quantile(cov_d, args.ci_alpha / 2),
                        "coverage_debiased_ci_high": _nan_quantile(cov_d, 1 - args.ci_alpha / 2),
                        "coverage_naive_center": _stat_reduce(cov_n, stat_over_reps),
                        "coverage_naive_ci_low": _nan_quantile(cov_n, args.ci_alpha / 2),
                        "coverage_naive_ci_high": _nan_quantile(cov_n, 1 - args.ci_alpha / 2),
                        "width_debiased_center": _stat_reduce(wid_d, stat_over_reps),
                        "width_debiased_ci_low": _nan_quantile(wid_d, args.ci_alpha / 2),
                        "width_debiased_ci_high": _nan_quantile(wid_d, 1 - args.ci_alpha / 2),
                        "width_naive_center": _stat_reduce(wid_n, stat_over_reps),
                        "width_naive_ci_low": _nan_quantile(wid_n, args.ci_alpha / 2),
                        "width_naive_ci_high": _nan_quantile(wid_n, 1 - args.ci_alpha / 2),
                        "score_debiased_center": _stat_reduce(sco_d, stat_over_reps),
                        "score_debiased_ci_low": _nan_quantile(sco_d, args.ci_alpha / 2),
                        "score_debiased_ci_high": _nan_quantile(sco_d, 1 - args.ci_alpha / 2),
                        "score_naive_center": _stat_reduce(sco_n, stat_over_reps),
                        "score_naive_ci_low": _nan_quantile(sco_n, args.ci_alpha / 2),
                        "score_naive_ci_high": _nan_quantile(sco_n, 1 - args.ci_alpha / 2),
                        "valid_rate_debiased_center": _stat_reduce(val_d, stat_over_reps),
                        "valid_rate_debiased_ci_low": _nan_quantile(val_d, args.ci_alpha / 2),
                        "valid_rate_debiased_ci_high": _nan_quantile(val_d, 1 - args.ci_alpha / 2),
                        "valid_rate_naive_center": _stat_reduce(val_n, stat_over_reps),
                        "valid_rate_naive_ci_low": _nan_quantile(val_n, args.ci_alpha / 2),
                        "valid_rate_naive_ci_high": _nan_quantile(val_n, 1 - args.ci_alpha / 2),
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
        return summary_rows

    stat_pairs = [(args.width_stat, args.stat_over_reps)]
    if args.stat_grid:
        stat_pairs = [("mean", "mean"), ("mean", "median"), ("median", "mean"), ("median", "median")]

    summary_rows_by_stat: Dict[str, List[Dict[str, Any]]] = {}
    runs: List[Dict[str, Any]] = []
    default_files: Dict[str, str] = {}

    def _plot_nuisance(rows: List[Dict[str, Any]], fname: str) -> str:
        plt.figure(figsize=(7.2, 4.2))
        for div in div_list:
            sub = [row for row in rows if row["divergence"] == div]
            xs = [row["n"] for row in sub]
            ys = [row["propensity_rmse_center"] for row in sub]
            lo = [row["propensity_rmse_ci_low"] for row in sub]
            hi = [row["propensity_rmse_ci_high"] for row in sub]
            yerr = [np.subtract(ys, lo), np.subtract(hi, ys)]
            plt.errorbar(xs, ys, yerr=yerr, marker="o", linestyle="-", capsize=3, label=f"{div}")

        if rows:
            xs_ref = np.array(sorted({row["n"] for row in rows}), dtype=float)
            if xs_ref.size >= 2:
                y0 = float(rows[0]["propensity_rmse_center"])
                n0 = float(xs_ref[0])
                ref_025 = y0 * (xs_ref / n0) ** (-0.25)
                ref_05 = y0 * (xs_ref / n0) ** (-0.5)
                plt.plot(xs_ref, ref_025, linestyle="--", color="gray", label="n^-1/4 ref")
                plt.plot(xs_ref, ref_05, linestyle=":", color="gray", label="n^-1/2 ref")

        plt.xlabel("Sample size n")
        plt.ylabel("Propensity RMSE")
        plt.title(f"Nuisance error vs n (struct={args.structural_type}, eval={args.eval_mode})")
        plt.legend()
        plt.tight_layout()
        path = name_with_suffix(fname, "png")
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    def _plot_target(rows: List[Dict[str, Any]], fname: str) -> str:
        plt.figure(figsize=(7.2, 4.2))
        for div in div_list:
            sub = [row for row in rows if row["divergence"] == div]
            xs = [row["n"] for row in sub]
            yd = [row["err_up_debiased_center"] for row in sub]
            yn = [row["err_up_naive_center"] for row in sub]
            lo_d = [row["err_up_debiased_ci_low"] for row in sub]
            hi_d = [row["err_up_debiased_ci_high"] for row in sub]
            lo_n = [row["err_up_naive_ci_low"] for row in sub]
            hi_n = [row["err_up_naive_ci_high"] for row in sub]
            plt.errorbar(xs, yd, yerr=[np.subtract(yd, lo_d), np.subtract(hi_d, yd)], marker="o", linestyle="-", capsize=3, label=f"{div} debiased")
            plt.errorbar(xs, yn, yerr=[np.subtract(yn, lo_n), np.subtract(hi_n, yn)], marker="o", linestyle="--", capsize=3, label=f"{div} naive")
        plt.xlabel("Sample size n")
        plt.ylabel("Target RMSE (upper bound)")
        plt.title(f"Target error vs n (struct={args.structural_type}, eval={args.eval_mode})")
        plt.legend()
        plt.tight_layout()
        path = name_with_suffix(fname, "png")
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    def _plot_metric(rows: List[Dict[str, Any]], key_d: str, key_n: str, ylabel: str, title: str, fname: str) -> str:
        plt.figure(figsize=(7.2, 4.2))
        for div in div_list:
            sub = [row for row in rows if row["divergence"] == div]
            xs = [row["n"] for row in sub]
            yd = [row[f"{key_d}_center"] for row in sub]
            yn = [row[f"{key_n}_center"] for row in sub]
            lo_d = [row[f"{key_d}_ci_low"] for row in sub]
            hi_d = [row[f"{key_d}_ci_high"] for row in sub]
            lo_n = [row[f"{key_n}_ci_low"] for row in sub]
            hi_n = [row[f"{key_n}_ci_high"] for row in sub]
            plt.errorbar(xs, yd, yerr=[np.subtract(yd, lo_d), np.subtract(hi_d, yd)], marker="o", linestyle="-", capsize=3, label=f"{div} debiased")
            plt.errorbar(xs, yn, yerr=[np.subtract(yn, lo_n), np.subtract(hi_n, yn)], marker="o", linestyle="--", capsize=3, label=f"{div} naive")
        plt.xlabel("Sample size n")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        path = name_with_suffix(fname, "png")
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    rep_base = "plot_n_debiased_replicates"
    summary_base = "plot_n_debiased_summary"
    nuisance_base = "plot_n_debiased_nuisance"
    target_base = "plot_n_debiased_target"
    width_base = "plot_n_debiased_width"
    coverage_base = "plot_n_debiased_coverage"
    score_base = "plot_n_debiased_score"

    for stat_within, stat_over_reps in stat_pairs:
        step_timer = StepTimer(
            f"summary/plots {stat_within}/{stat_over_reps}",
            use_tqdm=False,
            enabled=timing_enabled,
        )
        step_timer.__enter__()
        suffix = _stat_suffix(stat_within, stat_over_reps) if len(stat_pairs) > 1 else ""
        key = suffix or "default"
        summary_rows = _build_summary(stat_within, stat_over_reps)
        summary_rows_by_stat[key] = summary_rows

        rep_path = name_with_suffix(f"{rep_base}_{suffix}" if suffix else rep_base, "csv")
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(replicate_rows).to_csv(rep_path, index=False)
        except Exception:
            header = (
                "divergence,n,rep,seed,propensity_rmse,err_up_debiased,err_up_naive,coverage_debiased,coverage_naive,"
                "width_debiased,width_debiased_mean,width_debiased_median,width_naive,width_naive_mean,width_naive_median,"
                "score_debiased,score_debiased_mean,score_debiased_median,score_naive,score_naive_mean,score_naive_median,"
                "valid_rate_debiased,valid_rate_naive"
            )
            with open(rep_path, "w") as f:
                f.write(header + "\n")
                for row in replicate_rows:
                    f.write(
                        f"{row['divergence']},{row['n']},{row['rep']},{row['seed']},{row['propensity_rmse']},"
                        f"{row['err_up_debiased']},{row['err_up_naive']},{row['coverage_debiased']},{row['coverage_naive']},"
                        f"{row['width_debiased']},{row['width_debiased_mean']},{row['width_debiased_median']},"
                        f"{row['width_naive']},{row['width_naive_mean']},{row['width_naive_median']},"
                        f"{row['score_debiased']},{row['score_debiased_mean']},{row['score_debiased_median']},"
                        f"{row['score_naive']},{row['score_naive_mean']},{row['score_naive_median']},"
                        f"{row['valid_rate_debiased']},{row['valid_rate_naive']}\n"
                    )

        summary_path = name_with_suffix(f"{summary_base}_{suffix}" if suffix else summary_base, "csv")
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        except Exception:
            header = (
                "divergence,n,propensity_rmse_center,propensity_rmse_ci_low,propensity_rmse_ci_high,"
                "err_up_debiased_center,err_up_debiased_ci_low,err_up_debiased_ci_high,"
                "err_up_naive_center,err_up_naive_ci_low,err_up_naive_ci_high,"
                "coverage_debiased_center,coverage_debiased_ci_low,coverage_debiased_ci_high,"
                "coverage_naive_center,coverage_naive_ci_low,coverage_naive_ci_high,"
                "width_debiased_center,width_debiased_ci_low,width_debiased_ci_high,"
                "width_naive_center,width_naive_ci_low,width_naive_ci_high,"
                "score_debiased_center,score_debiased_ci_low,score_debiased_ci_high,"
                "score_naive_center,score_naive_ci_low,score_naive_ci_high,"
                "valid_rate_debiased_center,valid_rate_debiased_ci_low,valid_rate_debiased_ci_high,"
                "valid_rate_naive_center,valid_rate_naive_ci_low,valid_rate_naive_ci_high,"
                "m,d,n_eval,structural_type,eval_mode,width_stat,stat_over_reps,score_lambda,score_alpha,ci_alpha"
            )
            with open(summary_path, "w") as f:
                f.write(header + "\n")
                for row in summary_rows:
                    f.write(
                        f"{row['divergence']},{row['n']},{row['propensity_rmse_center']},"
                        f"{row['propensity_rmse_ci_low']},{row['propensity_rmse_ci_high']},"
                        f"{row['err_up_debiased_center']},{row['err_up_debiased_ci_low']},{row['err_up_debiased_ci_high']},"
                        f"{row['err_up_naive_center']},{row['err_up_naive_ci_low']},{row['err_up_naive_ci_high']},"
                        f"{row['coverage_debiased_center']},{row['coverage_debiased_ci_low']},{row['coverage_debiased_ci_high']},"
                        f"{row['coverage_naive_center']},{row['coverage_naive_ci_low']},{row['coverage_naive_ci_high']},"
                        f"{row['width_debiased_center']},{row['width_debiased_ci_low']},{row['width_debiased_ci_high']},"
                        f"{row['width_naive_center']},{row['width_naive_ci_low']},{row['width_naive_ci_high']},"
                        f"{row['score_debiased_center']},{row['score_debiased_ci_low']},{row['score_debiased_ci_high']},"
                        f"{row['score_naive_center']},{row['score_naive_ci_low']},{row['score_naive_ci_high']},"
                        f"{row['valid_rate_debiased_center']},{row['valid_rate_debiased_ci_low']},{row['valid_rate_debiased_ci_high']},"
                        f"{row['valid_rate_naive_center']},{row['valid_rate_naive_ci_low']},{row['valid_rate_naive_ci_high']},"
                        f"{row['m']},{row['d']},{row['n_eval']},{row['structural_type']},{row['eval_mode']},"
                        f"{row['width_stat']},{row['stat_over_reps']},{row['score_lambda']},{row['score_alpha']},{row['ci_alpha']}\n"
                    )

        nuisance_fig = _plot_nuisance(summary_rows, f"{nuisance_base}_{suffix}" if suffix else nuisance_base)
        target_fig = _plot_target(summary_rows, f"{target_base}_{suffix}" if suffix else target_base)
        width_fig = _plot_metric(
            summary_rows,
            "width_debiased",
            "width_naive",
            ylabel="Width",
            title=f"Width vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=f"{width_base}_{suffix}" if suffix else width_base,
        )
        cov_fig = _plot_metric(
            summary_rows,
            "coverage_debiased",
            "coverage_naive",
            ylabel="Coverage",
            title=f"Coverage vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=f"{coverage_base}_{suffix}" if suffix else coverage_base,
        )
        score_fig = _plot_metric(
            summary_rows,
            "score_debiased",
            "score_naive",
            ylabel="Score (penalized width)",
            title=f"Score vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=f"{score_base}_{suffix}" if suffix else score_base,
        )

        run_files = {
            "replicate_csv": rep_path,
            "summary_csv": summary_path,
            "nuisance_png": nuisance_fig,
            "target_png": target_fig,
            "width_png": width_fig,
            "coverage_png": cov_fig,
            "score_png": score_fig,
        }
        runs.append({"width_stat": stat_within, "stat_over_reps": stat_over_reps, "files": run_files})

        if stat_within == args.width_stat and stat_over_reps == args.stat_over_reps:
            default_files = run_files
        step_timer.__exit__(None, None, None)

    if not default_files and runs:
        default_files = runs[0]["files"]

    with StepTimer("save artifacts", use_tqdm=False, enabled=timing_enabled):
        artifacts = {
            "args": vars(args),
            "fit_config": fit_config,
            "dual_net_config": dual_net_config,
            "divergences": div_list,
            "seeds": seeds_used,
            "X_eval": X_eval,
            "oracle_U": U_oracle,
            "oracle_e": e_oracle,
            "oracle_seed": oracle_seed,
            "replicate_rows": replicate_rows,
            "summary_rows_by_stat": summary_rows_by_stat,
            "stat_pairs": stat_pairs,
            "timestamp": stamp,
        }
        artifacts_path = name_with_suffix("plot_n_debiased_artifacts", "pkl")
        with open(artifacts_path, "wb") as f:
            pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

        summary_json = {
            "timestamp": stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "files": dict(default_files, artifacts_pkl=artifacts_path),
            "args": vars(args),
            "divergences": div_list,
            "structural_type": args.structural_type,
            "eval_mode": args.eval_mode,
            "n_eval": n_eval,
            "oracle_seed": oracle_seed,
        }
        if len(runs) > 1:
            summary_json["runs"] = runs
        summary_json_path = name_with_suffix("plot_n_debiased_summary", "json")
        with open(summary_json_path, "w") as f:
            json.dump(summary_json, f, indent=2)

        log_line = (
            f"[plot_n_debiased] ts={summary_json['timestamp']} divs={','.join(div_list)} struct={args.structural_type} "
            f"eval={args.eval_mode} n_eval={n_eval} n_list={n_list} m={args.m} "
            f"stat_grid={args.stat_grid} files={summary_json['files']}"
        )
        log_path = name_with_suffix("plot_n_debiased_log", "txt")
        with open(log_path, "a") as f:
            f.write(log_line + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log_active_step_error(use_tqdm=False)
        raise
