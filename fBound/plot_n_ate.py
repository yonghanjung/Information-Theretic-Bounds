"""
ATE-vs-n experiment: compare debiased vs naive ATE bounds coverage and width.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import inspect
import os
import warnings
import zlib
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from causal_bound import DebiasedCausalBoundEstimator, _concat_ax, aggregate_endpointwise, prefit_propensity_cache
from data_generating import generate_data

try:
    from causal_bound import compute_ate_bounds_def6 as _compute_ate_bounds_def6
except Exception:  # pragma: no cover
    _compute_ate_bounds_def6 = None


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
# Endpoint-wise combined aggregator (AGENTS P0 default)
# -------------------------
def combined_endpointwise(lower_mat: np.ndarray, upper_mat: np.ndarray) -> dict[str, np.ndarray]:
    valid_up = np.isfinite(upper_mat)
    valid_lo = np.isfinite(lower_mat)
    return aggregate_endpointwise(
        lower_mat=lower_mat,
        upper_mat=upper_mat,
        valid_up=valid_up,
        valid_lo=valid_lo,
        k_up=1,
        k_lo=1,
    )


# Legacy combined (c-wise) intersection aggregator (explicit opt-in)
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
# Two-pass fit helper (train on X, evaluate on X_eval) for both a=0 and a=1
# -------------------------
def fit_two_pass_do_both_a_cached(
    EstimatorClass,
    div_name,
    X_train,
    A_train,
    Y_train,
    X_eval,
    e_eval,
    e_train_true,
    dual_net_config,
    fit_config,
    seed,
    propensity_model,
    m_model,
    propensity_cache,
    progress_prefix: Optional[str] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    try:
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
        ).fit(X_train, A_train, Y_train, propensity_cache=propensity_cache, e_train_true=e_train_true)

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
        ).fit(X_train, A_train, Y_train, propensity_cache=propensity_cache, e_train_true=e_train_true)

        outputs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for a in (0, 1):
            U = est_pos.predict_bound(a=a, X=X_eval, e_eval=e_eval).astype(np.float32)
            L = (-est_neg.predict_bound(a=a, X=X_eval, e_eval=e_eval)).astype(np.float32)
            outputs[int(a)] = (L, U)

        n_eval = int(np.asarray(X_eval).shape[0])
        assert set(outputs.keys()) == {0, 1}
        for a in (0, 1):
            L, U = outputs[a]
            assert L.shape == (n_eval,)
            assert U.shape == (n_eval,)
            assert L.dtype == np.float32
            assert U.dtype == np.float32
        return outputs
    except Exception as exc:
        method = getattr(EstimatorClass, "__name__", str(EstimatorClass))
        prefix = f", prefix={progress_prefix}" if progress_prefix else ""
        raise RuntimeError(
            f"fit_two_pass_do_both_a_cached failed for method={method}, divergence={div_name}, seed={seed}{prefix}"
        ) from exc


def fit_bounds_for_divs_both_a(
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
    e_eval=None,
    e_train_true=None,
    progress_prefix=None,
) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    bounds: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {0: {}, 1: {}}
    needs_base = any(div in {"combined", "combined_intersection", "cluster", "kth", "tight_kth"} for div in div_list)
    base_outputs: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

    if needs_base or any(dv in base_divs for dv in div_list):
        for dname in base_divs:
            if dname in div_list or needs_base:
                base_outputs[dname] = fit_two_pass_do_both_a_cached(
                    EstimatorClass,
                    dname,
                    X,
                    A,
                    Y,
                    X_eval,
                    e_eval,
                    e_train_true,
                    dual_net_config,
                    fit_config,
                    seed=seed,
                    propensity_model=propensity_model,
                    m_model=m_model,
                    propensity_cache=prop_cache,
                    progress_prefix=progress_prefix,
                )

    for div in div_list:
        if div in base_outputs:
            for a in (0, 1):
                bounds[a][div] = base_outputs[div][a]
            continue

        if div in {"combined", "combined_intersection", "cluster", "kth", "tight_kth"}:
            missing = [b for b in base_divs if b not in base_outputs]
            if missing:
                raise ValueError(f"Missing base divergences for {div}: {missing}")
            for a in (0, 1):
                lower_base = np.vstack([base_outputs[b][a][0] for b in base_divs])
                upper_base = np.vstack([base_outputs[b][a][1] for b in base_divs])
                if div == "combined":
                    agg = combined_endpointwise(lower_base, upper_base)
                    L = agg["lower"].astype(np.float32)
                    U = agg["upper"].astype(np.float32)
                    blanked_frac = float(np.mean(~np.isfinite(L) | ~np.isfinite(U)))
                    print(
                        f"combined diagnostics (a={a}): "
                        f"n_eff_up_mean={float(np.nanmean(agg['n_eff_up'])):.2f}, "
                        f"n_eff_lo_mean={float(np.nanmean(agg['n_eff_lo'])):.2f}, "
                        f"blanked_frac={blanked_frac:.3f}"
                    )
                elif div == "combined_intersection":
                    L, U = combined_cwise_intersection(lower_base, upper_base, c=3)
                elif div == "cluster":
                    L, U = cluster_per_sample_fast1d(lower_base, upper_base, k_candidates=(2, 3, 4), penalty_singleton=0.2)
                elif div == "kth":
                    k_val = int(lower_base.shape[0])
                    L = np.empty(lower_base.shape[1], dtype=np.float32)
                    U = np.empty(lower_base.shape[1], dtype=np.float32)
                    for i in range(lower_base.shape[1]):
                        lo, up = kth(lower_base[:, i], upper_base[:, i], k_val)
                        L[i] = np.float32(lo)
                        U[i] = np.float32(up)
                else:
                    k_val = int(lower_base.shape[0])
                    L = np.empty(lower_base.shape[1], dtype=np.float32)
                    U = np.empty(lower_base.shape[1], dtype=np.float32)
                    for i in range(lower_base.shape[1]):
                        lo, up = tight_kth(lower_base[:, i], upper_base[:, i], k=k_val)
                        L[i] = np.float32(lo)
                        U[i] = np.float32(up)
                bounds[a][div] = (L.astype(np.float32), U.astype(np.float32))
            continue

        div_outputs = fit_two_pass_do_both_a_cached(
            EstimatorClass,
            div,
            X,
            A,
            Y,
            X_eval,
            e_eval,
            e_train_true,
            dual_net_config,
            fit_config,
            seed=seed,
            propensity_model=propensity_model,
            m_model=m_model,
            propensity_cache=prop_cache,
            progress_prefix=progress_prefix,
        )
        for a in (0, 1):
            bounds[a][div] = div_outputs[a]

    return bounds


def derive_seed(base_seed: int, rep: int, n: int, tag: str) -> int:
    h = zlib.adler32(f"{tag}|{rep}|{n}|{base_seed}".encode())
    return int(base_seed + 10_000_000 * int(rep) + 10_000 * int(n) + (h % 10_000))


def _parse_n_list(raw: Union[str, Sequence[object]]) -> List[int]:
    if isinstance(raw, (list, tuple)):
        parts: List[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.extend([p.strip() for p in item.split(",") if p.strip()])
            else:
                parts.append(str(item))
    else:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
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
    allowed = set(base_divs + ["combined", "combined_intersection", "cluster", "kth", "tight_kth"])
    divs = [d.strip() for d in raw.split(",") if d.strip()]
    if not divs:
        divs = ["combined"]
    for d in divs:
        if d not in allowed:
            raise ValueError(f"Unknown divergence '{d}'. Allowed: {sorted(allowed)}")
    return divs


def _unique_suffix(enabled: bool) -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if enabled else ""


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


def _write_csv_rows(rows: List[Dict[str, object]], path: str, columns: List[str]) -> None:
    if not rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
        return
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    except Exception:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in columns})


def _nan_mean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _nan_quantile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _repair_order(ate_lower: float, ate_upper: float) -> Tuple[float, float, int]:
    repaired_order = 0
    if np.isfinite(ate_lower) and np.isfinite(ate_upper) and ate_lower > ate_upper:
        ate_lower, ate_upper = ate_upper, ate_lower
        repaired_order = 1
    return ate_lower, ate_upper, repaired_order


def ate_metrics_from_conditional_bounds(
    L0: np.ndarray,
    U0: np.ndarray,
    L1: np.ndarray,
    U1: np.ndarray,
    tau_eval: np.ndarray,
) -> Dict[str, float]:
    L0 = np.asarray(L0, dtype=np.float64).reshape(-1)
    U0 = np.asarray(U0, dtype=np.float64).reshape(-1)
    L1 = np.asarray(L1, dtype=np.float64).reshape(-1)
    U1 = np.asarray(U1, dtype=np.float64).reshape(-1)
    tau = np.asarray(tau_eval, dtype=np.float64).reshape(-1)

    if not (L0.shape == U0.shape == L1.shape == U1.shape == tau.shape):
        raise ValueError("L0, U0, L1, U1, tau_eval must have the same shape.")

    valid = (
        np.isfinite(L0)
        & np.isfinite(U0)
        & np.isfinite(L1)
        & np.isfinite(U1)
        & np.isfinite(tau)
        & (U0 > L0)
        & (U1 > L1)
    )
    valid_rate = float(np.mean(valid)) if valid.size > 0 else 0.0

    if valid_rate == 0.0:
        return {
            "ate_true": float("nan"),
            "ate_lower": float("nan"),
            "ate_upper": float("nan"),
            "ate_width": float("nan"),
            "coverage": 0.0,
            "valid_rate": 0.0,
            "repaired_order": 0.0,
        }

    ate_true = float(np.mean(tau[valid]))
    ate_lower = float(np.mean(L1[valid] - U0[valid]))
    ate_upper = float(np.mean(U1[valid] - L0[valid]))
    ate_lower, ate_upper, repaired_order = _repair_order(ate_lower, ate_upper)

    if np.isfinite(ate_lower) and np.isfinite(ate_upper):
        ate_width = float(ate_upper - ate_lower)
        coverage = 1.0 if (ate_lower <= ate_true <= ate_upper) else 0.0
    else:
        ate_width = float("nan")
        coverage = 0.0

    return {
        "ate_true": ate_true,
        "ate_lower": ate_lower,
        "ate_upper": ate_upper,
        "ate_width": ate_width,
        "coverage": float(coverage),
        "valid_rate": valid_rate,
        "repaired_order": float(repaired_order),
    }


def _self_test_step3() -> None:
    data = generate_data(n=200, d=2, seed=0, structural_type="cyclic2")
    X = np.asarray(data["X"], dtype=np.float32)
    A = np.asarray(data["A"], dtype=np.float32)
    Y = np.asarray(data["Y"], dtype=np.float32)

    data_eval = generate_data(n=50, d=2, seed=1, structural_type="cyclic2")
    X_eval = np.asarray(data_eval["X"], dtype=np.float32)

    dual_net_config = {
        "hidden_sizes": (16, 16),
        "activation": "relu",
        "dropout": 0.0,
        "h_clip": 20.0,
        "device": "cpu",
    }
    fit_config = {
        "n_folds": 2,
        "num_epochs": 1,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_grad_norm": 10.0,
        "eps_propensity": 1e-3,
        "deterministic_torch": True,
        "train_m_on_fold": True,
        "propensity_config": {
            "C": 1.0,
            "max_iter": 200,
            "penalty": "l2",
            "solver": "lbfgs",
            "n_jobs": 1,
        },
        "m_config": {
            "alpha": 1.0,
        },
        "verbose": False,
        "log_every": 10,
    }

    propensity_model = "logistic"
    m_model = "linear"

    prop_cache = prefit_propensity_cache(
        X=X,
        A=A,
        propensity_model=propensity_model,
        propensity_config=fit_config["propensity_config"],
        n_folds=fit_config["n_folds"],
        seed=0,
        eps_propensity=fit_config["eps_propensity"],
    )

    bounds = fit_two_pass_do_both_a_cached(
        DebiasedCausalBoundEstimator,
        "KL",
        X,
        A,
        Y,
        X_eval,
        e_eval=None,
        e_train_true=None,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=0,
        propensity_model=propensity_model,
        m_model=m_model,
        propensity_cache=prop_cache,
        progress_prefix="self_test_step3",
    )

    for a in (0, 1):
        L, U = bounds[a]
        assert L.shape == (X_eval.shape[0],)
        assert U.shape == (X_eval.shape[0],)
        assert L.dtype == np.float32
        assert U.dtype == np.float32


def _self_test_step4() -> None:
    data = generate_data(n=200, d=2, seed=0, structural_type="cyclic2")
    X = np.asarray(data["X"], dtype=np.float32)
    A = np.asarray(data["A"], dtype=np.float32)
    Y = np.asarray(data["Y"], dtype=np.float32)

    data_eval = generate_data(n=50, d=2, seed=1, structural_type="cyclic2")
    X_eval = np.asarray(data_eval["X"], dtype=np.float32)

    dual_net_config = {
        "hidden_sizes": (16, 16),
        "activation": "relu",
        "dropout": 0.0,
        "h_clip": 20.0,
        "device": "cpu",
    }
    fit_config = {
        "n_folds": 2,
        "num_epochs": 1,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_grad_norm": 10.0,
        "eps_propensity": 1e-3,
        "deterministic_torch": True,
        "train_m_on_fold": True,
        "propensity_config": {
            "C": 1.0,
            "max_iter": 200,
            "penalty": "l2",
            "solver": "lbfgs",
            "n_jobs": 1,
        },
        "m_config": {
            "alpha": 1.0,
        },
        "verbose": False,
        "log_every": 10,
    }

    propensity_model = "logistic"
    m_model = "linear"

    prop_cache = prefit_propensity_cache(
        X=X,
        A=A,
        propensity_model=propensity_model,
        propensity_config=fit_config["propensity_config"],
        n_folds=fit_config["n_folds"],
        seed=0,
        eps_propensity=fit_config["eps_propensity"],
    )

    base_divs_test = ["KL", "TV"]
    div_list_test = ["KL", "combined", "cluster"]
    bounds = fit_bounds_for_divs_both_a(
        DebiasedCausalBoundEstimator,
        div_list_test,
        base_divs_test,
        X,
        A,
        Y,
        X_eval,
        dual_net_config,
        fit_config,
        seed=0,
        propensity_model=propensity_model,
        m_model=m_model,
        prop_cache=prop_cache,
        e_eval=None,
        e_train_true=None,
        progress_prefix="self_test_step4",
    )

    for a in (0, 1):
        assert a in bounds
        for div in div_list_test:
            L, U = bounds[a][div]
            assert L.shape == (X_eval.shape[0],)
            assert U.shape == (X_eval.shape[0],)
            assert L.dtype == np.float32
            assert U.dtype == np.float32


def _self_test_step5() -> None:
    tau = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    L1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    U1 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    L0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    U0 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    metrics = ate_metrics_from_conditional_bounds(L0, U0, L1, U1, tau)
    assert np.isclose(metrics["ate_lower"], 0.5)
    assert np.isclose(metrics["ate_upper"], 2.0)
    assert np.isclose(metrics["ate_true"], 1.0)
    assert metrics["coverage"] == 1.0
    assert np.isclose(metrics["valid_rate"], 1.0)

    tau_bad = np.array([1.0, 1.0], dtype=np.float32)
    L1_bad = np.array([0.0, 0.0], dtype=np.float32)
    U1_bad = np.array([0.0, 0.0], dtype=np.float32)
    L0_bad = np.array([0.0, 0.0], dtype=np.float32)
    U0_bad = np.array([0.0, 0.0], dtype=np.float32)
    metrics_bad = ate_metrics_from_conditional_bounds(L0_bad, U0_bad, L1_bad, U1_bad, tau_bad)
    assert np.isnan(metrics_bad["ate_lower"])
    assert np.isnan(metrics_bad["ate_upper"])
    assert np.isnan(metrics_bad["ate_width"])
    assert np.isnan(metrics_bad["ate_true"])
    assert metrics_bad["coverage"] == 0.0
    assert metrics_bad["valid_rate"] == 0.0

    lo, hi, repaired = _repair_order(2.0, 1.0)
    assert repaired == 1
    assert lo == 1.0
    assert hi == 2.0


# -------------------------
# Main
# -------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ATE interval experiment: debiased vs naive coverage/width vs n."
    )
    parser.add_argument(
        "--n_list",
        type=str,
        nargs="+",
        required=False,
        help="Sample sizes (space or comma-separated), e.g. '500 1000' or '500,1000'.",
    )
    parser.add_argument("--m", type=int, default=30, help="Number of replicates per n.")
    parser.add_argument("--d", type=int, default=5, help="Feature dimension.")
    parser.add_argument("--base_seed", type=int, default=123, help="Base seed for training replicates.")
    parser.add_argument(
        "--structural_type",
        type=str,
        default="nonlinear",
        choices=["linear", "nonlinear", "simpson", "cyclic", "cyclic2"],
        help="Data-generating process type.",
    )
    parser.add_argument("--n_eval", type=int, default=5000, help="Number of fixed evaluation points.")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="sample",
        choices=["sample", "grid_x0"],
        help="How to construct fixed X_eval: sample from DGP marginal or grid over X0.",
    )
    parser.add_argument("--eval_seed", type=int, default=2025, help="Seed for constructing X_eval.")
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
    parser.add_argument(
        "--divergence",
        type=str,
        default="cluster",
        help="Comma-separated divergences from {KL,TV,Hellinger,Chi2,JS,combined,combined_intersection,cluster,kth,tight_kth}.",
    )
    parser.add_argument("--outdir", type=str, default="experiments", help="Output directory.")
    parser.add_argument("--unique_save", action="store_true", help="Add unique timestamp suffix to outputs.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha for summary CIs.")
    parser.add_argument(
        "--include_def6",
        action="store_true",
        help="Include Definition 6 ATE bounds if available.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke-test configuration.")

    # Estimator controls
    parser.add_argument("--n_folds", type=int, default=2, help="CV folds.")
    parser.add_argument("--eps_propensity", type=float, default=1e-3, help="Propensity clipping.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Dual net epochs.")
    parser.add_argument("--xgb_n_jobs", type=int, default=-1, help="n_jobs for xgboost (-1 all cores).")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for dual nets.",
    )
    parser.add_argument("--self_test_step3", action="store_true", help="Run Step 3 self-test.")
    parser.add_argument("--self_test_step4", action="store_true", help="Run Step 4 self-test.")
    parser.add_argument("--self_test_step5", action="store_true", help="Run Step 5 self-test.")
    return parser


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.self_test_step3:
        _self_test_step3()
        return
    if args.self_test_step4:
        _self_test_step4()
        return
    if args.self_test_step5:
        _self_test_step5()
        return
    if not args.n_list and not args.smoke:
        parser.error("--n_list is required unless a self-test flag is set.")
    if args.include_def6 and _compute_ate_bounds_def6 is None:
        print("[WARN] compute_ate_bounds_def6 not available; skipping Definition 6 metrics.")
    if args.smoke:
        n_list = [200, 400]
        m = 2
        n_eval = 200
        n_folds = 2
        num_epochs = 2
    else:
        n_list = _parse_n_list(args.n_list)
        m = int(args.m)
        n_eval = int(args.n_eval)
        n_folds = int(args.n_folds)
        num_epochs = int(args.num_epochs)

    base_divs = ["KL", "TV", "Hellinger", "Chi2", "JS"]
    div_list = _parse_divergences(args.divergence, base_divs)

    os.makedirs(args.outdir, exist_ok=True)
    stamp = _unique_suffix(args.unique_save)

    def name_with_suffix(base: str, ext: str) -> str:
        fname = f"{base}_{stamp}.{ext}" if stamp else f"{base}.{ext}"
        return os.path.join(args.outdir, fname)

    warn_state: Dict[str, bool] = {}
    X_eval = _make_X_eval(
        n_eval=n_eval,
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

    data_eval = _call_generate_data_compat(
        n=n_eval,
        d=args.d,
        seed=args.eval_seed,
        structural_type=args.structural_type,
        x_range=args.x_range,
        noise_dist=args.noise_dist,
        _warn_state=warn_state,
    )
    gt_fn = data_eval.get("GroundTruth")
    if gt_fn is None:
        raise RuntimeError("generate_data(...) did not return GroundTruth.")
    prop_true_fn = data_eval.get("propensity_true")
    if prop_true_fn is None:
        raise RuntimeError("generate_data(...) did not return propensity_true.")

    tau_eval = np.asarray(gt_fn(1, X_eval), dtype=np.float32).reshape(-1) - np.asarray(
        gt_fn(0, X_eval), dtype=np.float32
    ).reshape(-1)
    ate_true_scalar = float(np.mean(tau_eval[np.isfinite(tau_eval)]))
    e_true_eval = np.asarray(prop_true_fn(X_eval), dtype=np.float32).reshape(-1)
    if e_true_eval.shape[0] != n_eval:
        raise RuntimeError(f"propensity_true returned shape {e_true_eval.shape}, expected ({n_eval},).")

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
        "n_folds": n_folds,
        "num_epochs": num_epochs,
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

    if args.smoke:
        fit_config["batch_size"] = 128
        fit_config["propensity_config"]["n_estimators"] = 50
        fit_config["m_config"]["n_estimators"] = 50

    use_def6 = bool(args.include_def6 and _compute_ate_bounds_def6 is not None)
    replicate_rows: List[Dict[str, object]] = []
    for rep in range(m):
        for n in n_list:
            seed_data = derive_seed(args.base_seed, rep, int(n), "data")
            seed_propensity_fit = derive_seed(args.base_seed, rep, int(n), "propensity")
            seed_def6 = derive_seed(args.base_seed, rep, int(n), "def6")
            try:
                data_train = _call_generate_data_compat(
                    n=int(n),
                    d=args.d,
                    seed=seed_data,
                    structural_type=args.structural_type,
                    x_range=args.x_range,
                    noise_dist=args.noise_dist,
                    _warn_state=warn_state,
                )
                X = np.asarray(data_train["X"], dtype=np.float32)
                A = np.asarray(data_train["A"], dtype=np.float32)
                Y = np.asarray(data_train["Y"], dtype=np.float32)

                prop_cache = prefit_propensity_cache(
                    X=X,
                    A=A,
                    propensity_model=propensity_model,
                    propensity_config=fit_config["propensity_config"],
                    n_folds=fit_config["n_folds"],
                    seed=seed_propensity_fit,
                    eps_propensity=fit_config["eps_propensity"],
                )

                deb_bounds = fit_bounds_for_divs_both_a(
                    DebiasedCausalBoundEstimator,
                    div_list,
                    base_divs,
                    X,
                    A,
                    Y,
                    X_eval,
                    dual_net_config,
                    fit_config,
                    seed=seed_propensity_fit,
                    propensity_model=propensity_model,
                    m_model=m_model,
                    prop_cache=prop_cache,
                    e_eval=e_true_eval,
                    e_train_true=None,
                    progress_prefix=f"rep={rep} n={n} debiased",
                )

                nai_bounds = fit_bounds_for_divs_both_a(
                    NaiveCausalBoundEstimator,
                    div_list,
                    base_divs,
                    X,
                    A,
                    Y,
                    X_eval,
                    dual_net_config,
                    fit_config,
                    seed=seed_propensity_fit,
                    propensity_model=propensity_model,
                    m_model=m_model,
                    prop_cache=prop_cache,
                    e_eval=e_true_eval,
                    e_train_true=None,
                    progress_prefix=f"rep={rep} n={n} naive",
                )

                def6_bounds: Dict[str, Dict[str, Any]] = {}
                def6_combined: Optional[Dict[str, float]] = None
                if use_def6:
                    if "combined" in div_list:
                        base_divs_needed = base_divs
                    else:
                        base_divs_needed = [d for d in base_divs if d in div_list]

                    for dname in base_divs_needed:
                        out = _compute_ate_bounds_def6(
                            Y=Y,
                            A=A,
                            divergence=dname,
                            seed=seed_def6,
                            num_epochs=num_epochs,
                            lr=1e-2,
                            weight_decay=0.0,
                            max_grad_norm=10.0,
                            eps_propensity=fit_config["eps_propensity"],
                            h_clip=dual_net_config["h_clip"],
                            device=args.device,
                            verbose=False,
                            log_every=200,
                        )
                        def6_bounds[dname] = out

                    if "combined" in div_list:
                        lows = [float(def6_bounds[d]["ate_lower"]) for d in base_divs]
                        highs = [float(def6_bounds[d]["ate_upper"]) for d in base_divs]
                        L_comb = float(np.max(lows))
                        U_comb = float(np.min(highs))
                        repaired = 0
                        if L_comb > U_comb:
                            L_comb = float(np.min(lows))
                            U_comb = float(np.max(highs))
                            repaired = 1
                        def6_combined = {
                            "ate_lower": L_comb,
                            "ate_upper": U_comb,
                            "ate_width": float(U_comb - L_comb),
                            "ate_order_fixed": float(repaired),
                        }

                for div in div_list:
                    L0d, U0d = deb_bounds[0][div]
                    L1d, U1d = deb_bounds[1][div]
                    metrics_d = ate_metrics_from_conditional_bounds(L0d, U0d, L1d, U1d, tau_eval)

                    L0n, U0n = nai_bounds[0][div]
                    L1n, U1n = nai_bounds[1][div]
                    metrics_n = ate_metrics_from_conditional_bounds(L0n, U0n, L1n, U1n, tau_eval)

                    def6_row = {
                        "def6_ate_lower": float("nan"),
                        "def6_ate_upper": float("nan"),
                        "def6_ate_width": float("nan"),
                        "def6_coverage": float("nan"),
                        "def6_order_fixed": float("nan"),
                    }
                    if use_def6:
                        if div in def6_bounds:
                            L_def = float(def6_bounds[div]["ate_lower"])
                            U_def = float(def6_bounds[div]["ate_upper"])
                            def6_row = {
                                "def6_ate_lower": L_def,
                                "def6_ate_upper": U_def,
                                "def6_ate_width": float(U_def - L_def),
                                "def6_coverage": 1.0 if (L_def <= ate_true_scalar <= U_def) else 0.0,
                                "def6_order_fixed": float(bool(def6_bounds[div].get("ate_order_fixed", False))),
                            }
                        elif div == "combined" and def6_combined is not None:
                            L_def = float(def6_combined["ate_lower"])
                            U_def = float(def6_combined["ate_upper"])
                            def6_row = {
                                "def6_ate_lower": L_def,
                                "def6_ate_upper": U_def,
                                "def6_ate_width": float(def6_combined["ate_width"]),
                                "def6_coverage": 1.0 if (L_def <= ate_true_scalar <= U_def) else 0.0,
                                "def6_order_fixed": float(def6_combined["ate_order_fixed"]),
                            }
                        elif div == "cluster":
                            pass  # Def6 cluster aggregation not defined; keep NaNs.

                    replicate_rows.append(
                        {
                            "divergence": div,
                            "n": int(n),
                            "rep": int(rep),
                            "seed_base": int(args.base_seed),
                            "seed_data": int(seed_data),
                            "seed_propensity_fit": int(seed_propensity_fit),
                            "ate_true": metrics_d["ate_true"],
                            "ate_true_naive": metrics_n["ate_true"],
                            "ate_lower_debiased": metrics_d["ate_lower"],
                            "ate_upper_debiased": metrics_d["ate_upper"],
                            "ate_width_debiased": metrics_d["ate_width"],
                            "coverage_debiased": metrics_d["coverage"],
                            "valid_rate_debiased": metrics_d["valid_rate"],
                            "repaired_order_debiased": metrics_d["repaired_order"],
                            "ate_lower_naive": metrics_n["ate_lower"],
                            "ate_upper_naive": metrics_n["ate_upper"],
                            "ate_width_naive": metrics_n["ate_width"],
                            "coverage_naive": metrics_n["coverage"],
                            "valid_rate_naive": metrics_n["valid_rate"],
                            "repaired_order_naive": metrics_n["repaired_order"],
                            **def6_row,
                        }
                    )
            except Exception as exc:
                print(f"[WARN] rep={rep} n={n} failed: {exc}")
                for div in div_list:
                    replicate_rows.append(
                        {
                            "divergence": div,
                            "n": int(n),
                            "rep": int(rep),
                            "seed_base": int(args.base_seed),
                            "seed_data": int(seed_data),
                            "seed_propensity_fit": int(seed_propensity_fit),
                            "ate_true": float("nan"),
                            "ate_true_naive": float("nan"),
                            "ate_lower_debiased": float("nan"),
                            "ate_upper_debiased": float("nan"),
                            "ate_width_debiased": float("nan"),
                            "coverage_debiased": float("nan"),
                            "valid_rate_debiased": float("nan"),
                            "repaired_order_debiased": float("nan"),
                            "ate_lower_naive": float("nan"),
                            "ate_upper_naive": float("nan"),
                            "ate_width_naive": float("nan"),
                            "coverage_naive": float("nan"),
                            "valid_rate_naive": float("nan"),
                            "repaired_order_naive": float("nan"),
                            "def6_ate_lower": float("nan"),
                            "def6_ate_upper": float("nan"),
                            "def6_ate_width": float("nan"),
                            "def6_coverage": float("nan"),
                            "def6_order_fixed": float("nan"),
                        }
                    )

    rep_path = name_with_suffix("ate_replicates", "csv")
    rep_columns = [
        "divergence",
        "n",
        "rep",
        "seed_base",
        "seed_data",
        "seed_propensity_fit",
        "ate_true",
        "ate_true_naive",
        "ate_lower_debiased",
        "ate_upper_debiased",
        "ate_width_debiased",
        "coverage_debiased",
        "valid_rate_debiased",
        "repaired_order_debiased",
        "ate_lower_naive",
        "ate_upper_naive",
        "ate_width_naive",
        "coverage_naive",
        "valid_rate_naive",
        "repaired_order_naive",
        "def6_ate_lower",
        "def6_ate_upper",
        "def6_ate_width",
        "def6_coverage",
        "def6_order_fixed",
    ]
    _write_csv_rows(replicate_rows, rep_path, rep_columns)

    summary_rows: List[Dict[str, object]] = []
    for div in div_list:
        for n in sorted({int(r["n"]) for r in replicate_rows if r["divergence"] == div}):
            sub = [r for r in replicate_rows if r["divergence"] == div and int(r["n"]) == int(n)]
            cov_d = np.array([r["coverage_debiased"] for r in sub], dtype=float)
            cov_n = np.array([r["coverage_naive"] for r in sub], dtype=float)
            wid_d = np.array([r["ate_width_debiased"] for r in sub], dtype=float)
            wid_n = np.array([r["ate_width_naive"] for r in sub], dtype=float)
            val_d = np.array([r["valid_rate_debiased"] for r in sub], dtype=float)
            val_n = np.array([r["valid_rate_naive"] for r in sub], dtype=float)
            def6_cov = np.array([r["def6_coverage"] for r in sub], dtype=float)
            def6_wid = np.array([r["def6_ate_width"] for r in sub], dtype=float)

            row = {
                "divergence": div,
                "n": int(n),
                "coverage_debiased_center": _nan_mean(cov_d),
                "coverage_debiased_ci_low": _nan_quantile(cov_d, args.alpha / 2),
                "coverage_debiased_ci_high": _nan_quantile(cov_d, 1 - args.alpha / 2),
                "coverage_naive_center": _nan_mean(cov_n),
                "coverage_naive_ci_low": _nan_quantile(cov_n, args.alpha / 2),
                "coverage_naive_ci_high": _nan_quantile(cov_n, 1 - args.alpha / 2),
                "width_debiased_center": _nan_mean(wid_d),
                "width_debiased_ci_low": _nan_quantile(wid_d, args.alpha / 2),
                "width_debiased_ci_high": _nan_quantile(wid_d, 1 - args.alpha / 2),
                "width_naive_center": _nan_mean(wid_n),
                "width_naive_ci_low": _nan_quantile(wid_n, args.alpha / 2),
                "width_naive_ci_high": _nan_quantile(wid_n, 1 - args.alpha / 2),
                "valid_rate_debiased_center": _nan_mean(val_d),
                "valid_rate_debiased_ci_low": _nan_quantile(val_d, args.alpha / 2),
                "valid_rate_debiased_ci_high": _nan_quantile(val_d, 1 - args.alpha / 2),
                "valid_rate_naive_center": _nan_mean(val_n),
                "valid_rate_naive_ci_low": _nan_quantile(val_n, args.alpha / 2),
                "valid_rate_naive_ci_high": _nan_quantile(val_n, 1 - args.alpha / 2),
                "def6_coverage_center": _nan_mean(def6_cov),
                "def6_coverage_ci_low": _nan_quantile(def6_cov, args.alpha / 2),
                "def6_coverage_ci_high": _nan_quantile(def6_cov, 1 - args.alpha / 2),
                "def6_width_center": _nan_mean(def6_wid),
                "def6_width_ci_low": _nan_quantile(def6_wid, args.alpha / 2),
                "def6_width_ci_high": _nan_quantile(def6_wid, 1 - args.alpha / 2),
            }
            summary_rows.append(row)

    summary_path = name_with_suffix("ate_summary", "csv")
    summary_columns = list(summary_rows[0].keys()) if summary_rows else []
    if summary_columns:
        _write_csv_rows(summary_rows, summary_path, summary_columns)
    else:
        _write_csv_rows([], summary_path, ["divergence", "n"])

    def _plot_metric(rows, metric_base, ylabel, title, fname):
        plt.figure(figsize=(8, 5))
        for div in div_list:
            sub = [r for r in rows if r["divergence"] == div]
            sub = sorted(sub, key=lambda r: int(r["n"]))
            if not sub:
                continue
            xs = [int(r["n"]) for r in sub]
            yd = [r[f"{metric_base}_debiased_center"] for r in sub]
            lo_d = [r[f"{metric_base}_debiased_ci_low"] for r in sub]
            hi_d = [r[f"{metric_base}_debiased_ci_high"] for r in sub]
            yn = [r[f"{metric_base}_naive_center"] for r in sub]
            lo_n = [r[f"{metric_base}_naive_ci_low"] for r in sub]
            hi_n = [r[f"{metric_base}_naive_ci_high"] for r in sub]
            plt.errorbar(xs, yd, yerr=[np.subtract(yd, lo_d), np.subtract(hi_d, yd)], marker="o", linestyle="-", capsize=3, label=f"{div} debiased")
            plt.errorbar(xs, yn, yerr=[np.subtract(yn, lo_n), np.subtract(hi_n, yn)], marker="s", linestyle="--", capsize=3, label=f"{div} naive")

        plt.xlabel("Sample size n")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        path = name_with_suffix(fname, "png")
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    _plot_metric(summary_rows, "coverage", "ATE coverage", "ATE coverage vs n", "ate_coverage_vs_n")
    _plot_metric(summary_rows, "width", "ATE width", "ATE width vs n", "ate_width_vs_n")
    _plot_metric(summary_rows, "valid_rate", "Valid rate", "ATE valid rate vs n", "ate_validrate_vs_n")

    if args.smoke:
        if not os.path.exists(rep_path):
            raise RuntimeError("Smoke run failed: ate_replicates.csv not found.")
        with open(rep_path, "r") as f:
            lines = f.readlines()
        if len(lines) <= 1:
            raise RuntimeError("Smoke run failed: ate_replicates.csv is empty.")


if __name__ == "__main__":
    main()
