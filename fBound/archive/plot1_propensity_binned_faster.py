
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
import pickle
import datetime
import warnings
import os
import json
try:
    from tqdm import tqdm  # progress bars
except Exception:
    tqdm = lambda x, **k: x  # fallback no-op iterator

from data_generating import generate_data

# Optimized estimator with propensity caching.
# This script requires causal_bound_cached.py to be available on the Python path.
from causal_bound_cached import DebiasedCausalBoundEstimator, prefit_propensity_cache

# -------------------------
# 1) Phi definitions
# -------------------------
def phi_identity(y: torch.Tensor) -> torch.Tensor:
    return y

def phi_neg(y: torch.Tensor) -> torch.Tensor:
    return -y

# -------------------------
# 2) Fit bounds for do(A=1) via two-pass (phi and -phi)
#    Note: 2 fits total. (We avoid predicting a=0 since this script does not use it.)
#    Key optimization: pass a prefit propensity_cache to skip re-fitting e(X)=P(A=1|X).
# -------------------------
def fit_two_pass_do1_cached(
    EstimatorClass,
    div_name,
    X, A, Y,
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
    ).fit(X, A, Y, propensity_cache=propensity_cache)

    est_neg = EstimatorClass(
        divergence=div_name,
        phi=phi_neg,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
    ).fit(X, A, Y, propensity_cache=propensity_cache)

    # Upper bound for do(A=1)
    upper_a1 = est_pos.predict_bound(a=1, X=X).astype(np.float32)

    # Lower bound by sign flip: lower_phi = - upper_{-phi}
    lower_a1 = (-est_neg.predict_bound(a=1, X=X)).astype(np.float32)

    # Out-of-fold propensity (computed once in the cache)
    ehat1_oof = propensity_cache["e1_oof"].astype(np.float32)

    return lower_a1, upper_a1, ehat1_oof

# -------------------------
# 3) Combined (c-wise) intersection aggregator
#    Copy of the logic in causal_bound.compute_causal_bounds(..., divergence="combined")
# -------------------------
from itertools import combinations

def combined_cwise_intersection(lower_mat, upper_mat, c=3):
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
# 4) Cluster heuristic (FAST 1D implementation)
#
# The original version used sklearn KMeans + silhouette_score inside a per-sample
# loop, which is extremely expensive when n is large (n * ~8 KMeans fits).
#
# Here we provide a drop-in compatible API with three implementations:
#   - impl="fast1d"  (default): exact 1D k-means via exhaustive contiguous partitions
#                               (optimal in 1D for squared loss), + a lightweight
#                               silhouette score implementation.
#   - impl="trimmed": very fast robust rule: lower=2nd-smallest, upper=2nd-largest.
#   - impl="sklearn": original behavior (slow), kept for backward compatibility.
# -------------------------

def _median(vals):
    arr = np.sort(np.asarray(vals, dtype=np.float64))
    n = arr.size
    if n == 0:
        return np.nan
    if n % 2 == 1:
        return float(arr[n // 2])
    return float(0.5 * (arr[n // 2 - 1] + arr[n // 2]))

def _cluster_choose_lower(clusters):
    # choose the cluster (with size>=2) having the largest median, then take its max
    eligible = [c for c in clusters if len(c) >= 2]
    if not eligible:
        return None
    med = [_median(c) for c in eligible]
    best = int(np.argmax(med))
    return float(np.max(eligible[best]))

def _cluster_choose_upper(clusters):
    # choose the cluster (with size>=2) having the smallest median, then take its min
    eligible = [c for c in clusters if len(c) >= 2]
    if not eligible:
        return None
    med = [_median(c) for c in eligible]
    best = int(np.argmin(med))
    return float(np.min(eligible[best]))

def _silhouette_score_1d(x, labels):
    """Lightweight silhouette score for 1D points (Euclidean). Singleton clusters contribute 0."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    m = x.size
    uniq = np.unique(labels)
    if uniq.size < 2 or m < 2:
        return -np.inf

    # Pairwise distances (m x m), m is tiny here (M=5 divergences by default).
    D = np.abs(x[:, None] - x[None, :])

    s = np.zeros(m, dtype=np.float64)
    for i in range(m):
        lab_i = labels[i]
        same = (labels == lab_i)
        same_count = int(np.sum(same))
        if same_count <= 1:
            s[i] = 0.0
            continue
        # mean distance to other points in same cluster (exclude self; D[i,i]=0 so sum works)
        a_i = float(np.sum(D[i, same]) / (same_count - 1))

        b_i = np.inf
        for lab in uniq:
            if lab == lab_i:
                continue
            other = (labels == lab)
            b_i = min(b_i, float(np.mean(D[i, other])))

        denom = max(a_i, b_i)
        s[i] = 0.0 if denom <= 0 else (b_i - a_i) / denom

    return float(np.mean(s))

def _optimal_1d_kmeans_partition(values, k):
    """Exact optimal 1D k-means via contiguous partitions of sorted values."""
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    m = vals.size
    if k <= 1 or k > m:
        raise ValueError("k must be in [2, m] for partitioning.")

    order = np.argsort(vals)
    x = vals[order]

    best_sse = np.inf
    best_labels = None

    # In 1D, optimal clusters are contiguous intervals in sorted order.
    # Enumerate cut points: choose k-1 cuts among positions 1..m-1.
    for cuts in combinations(range(1, m), k - 1):
        boundaries = (0,) + tuple(cuts) + (m,)
        labels = np.empty(m, dtype=int)
        sse = 0.0
        for ci in range(k):
            l = boundaries[ci]
            r = boundaries[ci + 1]
            seg = x[l:r]
            mu = float(np.mean(seg))
            sse += float(np.sum((seg - mu) ** 2))
            labels[l:r] = ci
        if sse < best_sse:
            best_sse = sse
            best_labels = labels.copy()

    if best_labels is None:
        raise RuntimeError("Internal error: no partition found.")
    return x, best_labels

def _clusters_from_labels(x_sorted, labels_sorted):
    clusters = []
    for lab in np.unique(labels_sorted):
        clusters.append(x_sorted[labels_sorted == lab].astype(np.float64).tolist())
    return clusters

def _select_cluster_k_fast(values, k_candidates, penalty_singleton):
    best_k, best_score = None, -np.inf
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    m = vals.size
    if m < 2:
        return 1

    for k in k_candidates:
        if k < 2 or k > m:
            continue
        x_sorted, labels_sorted = _optimal_1d_kmeans_partition(vals, k)
        clusters = _clusters_from_labels(x_sorted, labels_sorted)
        if not any(len(c) >= 2 for c in clusters):
            continue

        sep = _silhouette_score_1d(x_sorted, labels_sorted)
        num_singletons = sum(1 for c in clusters if len(c) == 1)
        score = sep - penalty_singleton * num_singletons
        if score > best_score:
            best_score, best_k = score, k

    if best_k is None:
        best_k = min(2, m)
    return int(best_k)

def cluster_per_sample(
    lower_mat,
    upper_mat,
    seed,
    k_candidates=(2, 3, 4),
    penalty_singleton=0.2,
    impl="fast1d",
):
    """Cluster aggregation to combine M bound intervals into one per sample."""
    lower_mat = np.asarray(lower_mat, dtype=np.float32)
    upper_mat = np.asarray(upper_mat, dtype=np.float32)
    M, n = lower_mat.shape

    outL = np.empty(n, dtype=np.float32)
    outU = np.empty(n, dtype=np.float32)

    use_tqdm = tqdm if callable(tqdm) else (lambda x, **k: x)
    it = use_tqdm(range(n), desc=f"cluster agg ({impl})", leave=False)

    if impl == "trimmed":
        # Robust, extremely fast: assumes at most one divergence is an outlier on each side.
        for i in it:
            lows = np.sort(lower_mat[:, i].astype(np.float64))
            ups = np.sort(upper_mat[:, i].astype(np.float64))
            lo = lows[1] if M >= 2 else lows[0]
            hi = ups[-2] if M >= 2 else ups[0]
            if lo > hi:
                mid = 0.5 * (lo + hi)
                lo, hi = mid, mid
            outL[i] = np.float32(lo)
            outU[i] = np.float32(hi)
        return outL, outU

    if impl == "sklearn":
        # Original slow implementation (kept only for reproducibility comparisons).
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        def _cluster_1d_sklearn(vals, k, rs):
            X1 = np.array(vals, dtype=np.float64).reshape(-1, 1)
            k = max(1, min(int(k), len(vals)))
            km = KMeans(n_clusters=k, n_init=10, random_state=rs)
            labels = km.fit_predict(X1)
            clusters_local = []
            for lab in np.unique(labels):
                clusters_local.append([float(vals[j]) for j in range(len(vals)) if labels[j] == lab])
            return clusters_local, labels

        def _select_k_sklearn(vals, rs):
            best_k_local, best_score_local = None, -np.inf
            X1 = np.array(vals, dtype=np.float64).reshape(-1, 1)
            for k in k_candidates:
                if k > len(vals):
                    continue
                clusters_local, labels = _cluster_1d_sklearn(vals, k, rs)
                if not any(len(c) >= 2 for c in clusters_local):
                    continue
                sep = -np.inf if len(np.unique(labels)) < 2 else silhouette_score(X1, labels)
                num_singletons = sum(1 for c in clusters_local if len(c) == 1)
                score = float(sep) - penalty_singleton * num_singletons
                if score > best_score_local:
                    best_score_local, best_k_local = score, k
            if best_k_local is None:
                best_k_local = min(2, len(vals))
            return best_k_local

        for i in it:
            lows = [float(x) for x in lower_mat[:, i]]
            ups = [float(x) for x in upper_mat[:, i]]
            kL = _select_k_sklearn(lows, rs=seed + 777 + i)
            kU = _select_k_sklearn(ups, rs=seed + 888 + i)
            cL, _ = _cluster_1d_sklearn(lows, kL, rs=seed + 999 + i)
            cU, _ = _cluster_1d_sklearn(ups, kU, rs=seed + 1000 + i)

            lo = _cluster_choose_lower(cL)
            hi = _cluster_choose_upper(cU)

            # fallbacks
            if lo is None:
                lo = sorted(lows)[1] if len(lows) >= 2 else lows[0]
            if hi is None:
                hi = sorted(ups)[-2] if len(ups) >= 2 else ups[0]
            if lo > hi:
                lo = sorted(lows)[1] if len(lows) >= 2 else lo
                hi = sorted(ups)[-2] if len(ups) >= 2 else hi

            outL[i] = np.float32(lo)
            outU[i] = np.float32(hi)

        return outL, outU

    # impl == "fast1d" (default)
    for i in it:
        lows = lower_mat[:, i].astype(np.float64)
        ups = upper_mat[:, i].astype(np.float64)

        if not np.all(np.isfinite(lows)) or not np.all(np.isfinite(ups)):
            # conservative fallback
            lows_s = np.sort(lows[np.isfinite(lows)])
            ups_s = np.sort(ups[np.isfinite(ups)])
            lo = lows_s[1] if lows_s.size >= 2 else (lows_s[0] if lows_s.size == 1 else 0.0)
            hi = ups_s[-2] if ups_s.size >= 2 else (ups_s[0] if ups_s.size == 1 else 0.0)
            if lo > hi:
                mid = 0.5 * (lo + hi)
                lo, hi = mid, mid
            outL[i] = np.float32(lo)
            outU[i] = np.float32(hi)
            continue

        kL = _select_cluster_k_fast(lows, k_candidates, penalty_singleton)
        kU = _select_cluster_k_fast(ups, k_candidates, penalty_singleton)

        xL, labL = _optimal_1d_kmeans_partition(lows, kL) if kL >= 2 else (np.sort(lows), np.zeros(M, dtype=int))
        xU, labU = _optimal_1d_kmeans_partition(ups, kU) if kU >= 2 else (np.sort(ups), np.zeros(M, dtype=int))

        cL = _clusters_from_labels(xL, labL)
        cU = _clusters_from_labels(xU, labU)

        lo = _cluster_choose_lower(cL)
        hi = _cluster_choose_upper(cU)

        # fallbacks
        lows_s = np.sort(lows)
        ups_s = np.sort(ups)
        if lo is None:
            lo = float(lows_s[1]) if M >= 2 else float(lows_s[0])
        if hi is None:
            hi = float(ups_s[-2]) if M >= 2 else float(ups_s[0])
        if lo > hi:
            lo = float(lows_s[1]) if M >= 2 else lo
            hi = float(ups_s[-2]) if M >= 2 else hi

        outL[i] = np.float32(lo)
        outU[i] = np.float32(hi)

    return outL, outU
# -------------------------
# 5) Propensity bin summaries (fixed bins recommended for comparability)
#    Optimized: precompute bin membership once per replicate and reuse for all methods.
# -------------------------
def _prepare_fixed_bins(ehat1, truth, bin_edges):
    ehat1 = np.asarray(ehat1, dtype=np.float32).reshape(-1)
    truth = np.asarray(truth, dtype=np.float32).reshape(-1)
    bin_edges = np.asarray(bin_edges, dtype=float)

    # bin_id in {0,1,...,B-1}; out-of-range will give -1 or B (we drop them by construction via edges).
    bin_id = np.digitize(ehat1, bin_edges, right=False) - 1
    B = len(bin_edges) - 1

    idx_list = []
    n_list = np.empty(B, dtype=int)
    ehat_center = np.empty(B, dtype=np.float32)
    truth_mean = np.empty(B, dtype=np.float32)

    for b in range(B):
        idx = np.where(bin_id == b)[0]
        idx_list.append(idx)
        n_list[b] = int(idx.size)
        if idx.size > 0:
            ehat_center[b] = np.float32(np.mean(ehat1[idx]))
            truth_mean[b] = np.float32(np.mean(truth[idx]))
        else:
            ehat_center[b] = np.float32(np.nan)
            truth_mean[b] = np.float32(np.nan)

    return idx_list, n_list, ehat_center, truth_mean

def summarize_bins_fixed_precomputed(idx_list, n_list, ehat_center, truth_mean, truth, L, U, seed, method_label):
    """
    Fixed-bin summary for one replicate and one set of bounds.
    Returns long-format rows for each method and bin.
    """
    truth = np.asarray(truth, dtype=np.float32).reshape(-1)
    L = np.asarray(L, dtype=np.float32).reshape(-1)
    U = np.asarray(U, dtype=np.float32).reshape(-1)

    B = len(idx_list)
    rows = []
    for b in range(B):
        idx = idx_list[b]
        if idx.size == 0:
            rows.append(
                {
                    "seed": seed,
                    "method": method_label,
                    "bin": b,
                    "n": 0,
                    "ehat_center": np.nan,
                    "truth_mean": np.nan,
                    "L_mean": np.nan,
                    "U_mean": np.nan,
                    "coverage": np.nan,
                    "width": np.nan,
                }
            )
            continue

        Lb = L[idx]
        Ub = U[idx]
        tb = truth[idx]

        rows.append(
            {
                "seed": seed,
                "method": method_label,
                "bin": b,
                "n": int(n_list[b]),
                "ehat_center": float(ehat_center[b]),
                "truth_mean": float(truth_mean[b]),
                "L_mean": float(np.mean(Lb)),
                "U_mean": float(np.mean(Ub)),
                "coverage": float(np.mean((tb >= Lb) & (tb <= Ub))),
                "width": float(np.mean(Ub - Lb)),
            }
        )

    return rows

# -------------------------
# Main: Monte Carlo loop, fixed-bin aggregation, and plotting
# -------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Propensity-binned bounds with Monte Carlo averaging (optimized).")
    parser.add_argument("--m", type=int, default=10, help="number of MC replicates")
    parser.add_argument("--base_seed", type=int, default=190602, help="base seed; seeds = base_seed + r")
    parser.add_argument("--n", type=int, default=2000, help="sample size per replicate")
    parser.add_argument("--d", type=int, default=5, help="feature dimension")
    parser.add_argument("--bins", type=int, default=20, help="number of fixed propensity bins")
    parser.add_argument("--edge_lo", type=float, default=0.05, help="lower edge for propensity binning")
    parser.add_argument("--edge_hi", type=float, default=0.95, help="upper edge for propensity binning")
    parser.add_argument(
        "--methods",
        type=str,
        default="KL,TV,Hellinger,Chi2,JS,combined,cluster",
        help="Comma-separated methods to include in outputs/plots. Combined/cluster require base divergences.",
    )
    parser.add_argument("--unique_save", action="store_true", help="Save outputs with a unique timestamp suffix.")
    parser.add_argument(
        "--stat",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation statistic across seeds for plotting (mean or median).",
    )
    parser.add_argument(
        "--structural_type",
        type=str,
        default="nonlinear",
        choices=["linear", "nonlinear", "simpson"],
        help="Data-generating process type.",
    )
    parser.add_argument(
        "--xgb_n_jobs",
        type=int,
        default=1,
        help="XGBoost n_jobs for propensity and m models (-1 uses all cores).",
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=1,
        help="torch.set_num_threads(...) to control CPU threading in PyTorch.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override fit_config['num_epochs'] for the dual nets (None keeps script default).",
    )
    parser.add_argument(
        "--cluster_impl",
        type=str,
        default="fast1d",
        choices=["fast1d", "trimmed", "sklearn"],
        help="Cluster aggregation implementation: fast1d (recommended), trimmed (fastest), or sklearn (slow).",
    )
    args = parser.parse_args()

    # Threading controls (important for performance and avoiding oversubscription).
    try:
        torch.set_num_threads(max(1, int(args.torch_threads)))
        # interop threads can only be set once in some builds; best-effort
        try:
            torch.set_num_interop_threads(max(1, int(args.torch_threads)))
        except Exception:
            pass
    except Exception:
        pass


    # Monte Carlo loop setup
    m = args.m
    base_seed = args.base_seed
    simulation_seeds = [base_seed + r for r in range(m)]

    n = args.n
    d = args.d
    base_divs = ["KL", "TV", "Hellinger", "Chi2", "JS"]

    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    requested_set = set(requested)

    include_base = [name for name in base_divs if name in requested_set]
    include_combined = "combined" in requested_set
    include_cluster = "cluster" in requested_set

    # Decide what we actually need to fit:
    # - If combined/cluster is requested, we must fit all base divergences.
    # - Otherwise, only fit the requested base divergences.
    if include_combined or include_cluster:
        divs_to_fit = base_divs
    else:
        divs_to_fit = include_base

    bin_edges = np.linspace(args.edge_lo, args.edge_hi, args.bins + 1)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if args.unique_save else ""

    def name_with_suffix(base: str, ext: str) -> str:
        fname = f"{base}_{stamp}.{ext}" if stamp else f"{base}.{ext}"
        return os.path.join("experiments", fname)

    dual_net_config = {
        "hidden_sizes": (64, 64),
        "activation": "relu",
        "dropout": 0.0,
        "h_clip": 20.0,
        "device": "cpu",
    }
    propensity_model = "xgboost"
    m_model = "xgboost"

    fit_config = {
        "n_folds": 3,
        "num_epochs": 80,          # adjust for speed/quality
        "batch_size": 256,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "max_grad_norm": 10.0,
        "eps_propensity": 1e-3,
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
            "n_jobs": 1,
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
            "n_jobs": 1,
            "verbosity": 0,
        },
        "verbose": False,
        "log_every": 10,
    }

    # Optional overrides from CLI
    if args.num_epochs is not None:
        fit_config["num_epochs"] = int(args.num_epochs)

    # XGBoost threading (propensity + m model). If you parallelize across MC seeds, keep this at 1.
    fit_config["propensity_config"]["n_jobs"] = int(args.xgb_n_jobs)
    fit_config["m_config"]["n_jobs"] = int(args.xgb_n_jobs)


    per_seed_rows = []
    os.makedirs("experiments", exist_ok=True)

    print(
        f"Running {m} replicates with fixed bins on [{args.edge_lo}, {args.edge_hi}] "
        f"using divs_to_fit={divs_to_fit} (requested={sorted(requested_set)}) ..."
    )

    for seed in tqdm(simulation_seeds, desc="MC seeds"):
        data = generate_data(n=n, d=d, seed=seed, structural_type=args.structural_type)
        X = data["X"]
        A = data["A"]
        Y = data["Y"]
        truth = data["GroundTruth"](1, X).astype(np.float32)

        # ---- Key optimization: prefit cross-fitted propensity ONCE per replicate ----
        prop_cache = prefit_propensity_cache(
            X=X,
            A=A,
            propensity_model=propensity_model,
            propensity_config=fit_config["propensity_config"],
            n_folds=fit_config["n_folds"],
            seed=seed,
            eps_propensity=fit_config["eps_propensity"],
        )
        ehat1 = prop_cache["e1_oof"].astype(np.float32)

        # Precompute bin membership once per replicate and reuse for all methods.
        idx_list, n_list, ehat_center, truth_mean = _prepare_fixed_bins(ehat1=ehat1, truth=truth, bin_edges=bin_edges)

        # Fit (lower, upper) for each divergence we actually need.
        lower_by_div = {}
        upper_by_div = {}
        for div in divs_to_fit:
            L, U, _ = fit_two_pass_do1_cached(
                DebiasedCausalBoundEstimator,
                div,
                X, A, Y,
                dual_net_config,
                fit_config,
                seed=seed,
                propensity_model=propensity_model,
                m_model=m_model,
                propensity_cache=prop_cache,
            )
            lower_by_div[div] = L
            upper_by_div[div] = U

        # Base-divergence summaries (only those requested)
        for div in include_base:
            if div in lower_by_div:
                per_seed_rows.extend(
                    summarize_bins_fixed_precomputed(
                        idx_list=idx_list,
                        n_list=n_list,
                        ehat_center=ehat_center,
                        truth_mean=truth_mean,
                        truth=truth,
                        L=lower_by_div[div],
                        U=upper_by_div[div],
                        seed=seed,
                        method_label=div,
                    )
                )

        # Combined + cluster require all base divergences
        if include_combined or include_cluster:
            lower_mat = np.vstack([lower_by_div[div] for div in base_divs])
            upper_mat = np.vstack([upper_by_div[div] for div in base_divs])

            if include_combined:
                L_comb, U_comb = combined_cwise_intersection(lower_mat, upper_mat, c=3)
                per_seed_rows.extend(
                    summarize_bins_fixed_precomputed(
                        idx_list=idx_list,
                        n_list=n_list,
                        ehat_center=ehat_center,
                        truth_mean=truth_mean,
                        truth=truth,
                        L=L_comb,
                        U=U_comb,
                        seed=seed,
                        method_label="combined",
                    )
                )

            if include_cluster:
                L_clu, U_clu = cluster_per_sample(lower_mat, upper_mat, seed=seed, impl=args.cluster_impl)
                per_seed_rows.extend(
                    summarize_bins_fixed_precomputed(
                        idx_list=idx_list,
                        n_list=n_list,
                        ehat_center=ehat_center,
                        truth_mean=truth_mean,
                        truth=truth,
                        L=L_clu,
                        U=U_clu,
                        seed=seed,
                        method_label="cluster",
                    )
                )

    bins_by_seed = pd.DataFrame(per_seed_rows)
    bins_by_seed_fname = name_with_suffix("plot1_bins_by_seed", "csv")
    bins_by_seed.to_csv(bins_by_seed_fname, index=False)
    print(f"Saved per-seed bin table to {bins_by_seed_fname}")

    # Averaging across seeds with standard errors (drop empty bins)
    def _sem(x):
        x = np.asarray(x.dropna(), dtype=float)
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan

    if args.stat == "mean":
        agg_func = lambda s: float(np.nanmean(s))
    else:
        agg_func = lambda s: float(np.nanmedian(s))

    bins_avg = (
        bins_by_seed[bins_by_seed["n"] > 0]
        .groupby(["method", "bin"], as_index=False)
        .agg(
            ehat_center=("ehat_center", agg_func),
            truth_mean=("truth_mean", agg_func),
            L_mean=("L_mean", agg_func),
            U_mean=("U_mean", agg_func),
            coverage=("coverage", agg_func),
            width=("width", agg_func),
            coverage_se=("coverage", _sem),
            width_se=("width", _sem),
        )
    )

    bins_avg_fname = name_with_suffix("plot1_bins_avg", "csv")
    bins_avg.to_csv(bins_avg_fname, index=False)
    print(f"Saved averaged bin table to {bins_avg_fname}")

    # --- Plot A: truth + averaged ribbons ---
    figA, axA = plt.subplots(figsize=(7.0, 4.0))
    ribbon_methods = [m for m in requested_set]  # ribbons for all requested methods
    colors_cycle = {
        "combined": "tab:blue",
        "cluster": "tab:orange",
        "KL": "tab:green",
        "TV": "tab:red",
        "Hellinger": "tab:purple",
        "Chi2": "tab:brown",
        "JS": "tab:pink",
    }

    for method in ribbon_methods:
        sub = bins_avg[bins_avg["method"] == method].sort_values("ehat_center")
        if len(sub) == 0:
            continue
        axA.fill_between(
            sub["ehat_center"],
            sub["L_mean"],
            sub["U_mean"],
            alpha=0.25,
            color=colors_cycle.get(method, None),
            label=f"{method} ribbon",
        )
        axA.scatter(sub["ehat_center"], sub["L_mean"], color=colors_cycle.get(method, None), s=10, alpha=0.8, marker="o")
        axA.scatter(sub["ehat_center"], sub["U_mean"], color=colors_cycle.get(method, None), s=10, alpha=0.8, marker="o")

    # Truth line averaged over seeds per bin (same for all methods)
    truth_bins = bins_avg.groupby("bin", as_index=False).agg(
        ehat_center=("ehat_center", "mean"),
        truth_mean=("truth_mean", "mean"),
    )
    axA.plot(truth_bins["ehat_center"], truth_bins["truth_mean"], color="k", label="truth (avg)")
    axA.scatter(truth_bins["ehat_center"], truth_bins["truth_mean"], color="k", s=12, alpha=0.9, marker="x")
    axA.set_xlabel("Propensity estimate ê(x)")
    axA.set_ylabel("E[Y | do(A=1), X]")
    axA.set_title(f"Truth + aggregated ribbons ({args.stat})")
    axA.legend()
    figA.tight_layout()
    figA_fname = name_with_suffix("plot1_propensity_binned_ribbons", "png")
    figA.savefig(figA_fname, dpi=200)
    print(f"Saved {figA_fname}")

    # --- Plot B: coverage within bin ---
    figB, axB = plt.subplots(figsize=(7.0, 4.0))
    plot_methods = [m for m in ["combined", "cluster"] if m in requested_set] + [m for m in include_base]
    for method in plot_methods:
        sub = bins_avg[bins_avg["method"] == method].sort_values("ehat_center")
        if len(sub) == 0:
            continue
        axB.errorbar(
            sub["ehat_center"],
            sub["coverage"],
            yerr=sub["coverage_se"],
            marker="o",
            linestyle="-",
            capsize=3,
            label=method,
        )
    axB.set_xlabel("Propensity estimate ê(x)")
    axB.set_ylabel("Coverage within bin")
    axB.set_ylim(0.0, 1.05)
    axB.set_title(f"Coverage vs propensity ({args.stat})")
    axB.legend()
    figB.tight_layout()
    figB_fname = name_with_suffix("plot1_propensity_binned_coverage", "png")
    figB.savefig(figB_fname, dpi=200)
    print(f"Saved {figB_fname}")

    # --- Plot C: mean width within bin ---
    figC, axC = plt.subplots(figsize=(7.0, 4.0))
    for method in plot_methods:
        sub = bins_avg[bins_avg["method"] == method].sort_values("ehat_center")
        if len(sub) == 0:
            continue
        axC.errorbar(
            sub["ehat_center"],
            sub["width"],
            yerr=sub["width_se"],
            marker="o",
            linestyle="-",
            capsize=3,
            label=method,
        )
    axC.set_xlabel("Propensity estimate ê(x)")
    axC.set_ylabel("Mean width within bin")
    axC.set_title(f"Width vs propensity ({args.stat})")
    axC.legend()
    figC.tight_layout()
    figC_fname = name_with_suffix("plot1_propensity_binned_width", "png")
    figC.savefig(figC_fname, dpi=200)
    print(f"Saved {figC_fname}")

    # Save artifacts for later re-plotting
    artifacts = {
        "bins_by_seed": bins_by_seed,
        "bins_avg": bins_avg,
        "bin_edges": bin_edges,
        "requested_methods": list(requested_set),
        "args": vars(args),
        "stat": args.stat,
        "fit_config": fit_config,
        "timestamp": stamp,
    }
    artifacts_fname = name_with_suffix("plot1_artifacts", "pkl")
    with open(artifacts_fname, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {artifacts_fname} for re-plotting later")

    # Experiment summary logs (one-liner + detailed JSON)
    summary = {
        "timestamp": stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "files": {
            "bins_by_seed_csv": bins_by_seed_fname,
            "bins_avg_csv": bins_avg_fname,
            "ribbons_png": figA_fname,
            "coverage_png": figB_fname,
            "width_png": figC_fname,
            "artifacts_pkl": artifacts_fname,
        },
        "args": vars(args),
        "methods": list(requested_set),
        "stat": args.stat,
        "n": n,
        "d": d,
        "m": m,
        "bin_edges": bin_edges.tolist(),
    }
    summary_line = (
        f"[plot1] ts={summary['timestamp']} m={m} n={n} d={d} stat={args.stat} "
        f"methods={','.join(sorted(requested_set))} bins={len(bin_edges)-1} "
        f"files={summary['files']}"
    )
    with open(name_with_suffix("plot1_experiment_summary", "json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join("experiments", "plot1_experiment_log.txt"), "a") as f:
        f.write(summary_line + "\n")

    print("Done.")
