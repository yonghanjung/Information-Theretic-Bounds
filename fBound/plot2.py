import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings
import argparse
import datetime
import os
import json
from itertools import combinations

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback if tqdm not installed

from data_generating import generate_data

# IMPORTANT:
# - This script expects the optimized estimator module (propensity caching + faster prediction path).
# - If you rename causal_bound_optimized.py -> causal_bound.py, you can switch this import to:
#       from causal_bound import DebiasedCausalBoundEstimator, prefit_propensity_cache
from causal_bound import DebiasedCausalBoundEstimator, prefit_propensity_cache


# -------------------------
# Phi
# -------------------------
def phi_identity(y: torch.Tensor) -> torch.Tensor:
    return y


def phi_neg(y: torch.Tensor) -> torch.Tensor:
    return -y


# -------------------------
# Naive estimator: remove the debiasing correction term (Eq. 26).
# Implement by subclassing and overriding _debiased_loss_batch.
# -------------------------
class NaiveCausalBoundEstimator(DebiasedCausalBoundEstimator):
    def _debiased_loss_batch(self, X, A, Y, e1, e0, h_net, u_net):
        # Copy the parent's "main" term and omit the correction term.
        # This keeps everything else identical.
        from causal_bound_optimized import _concat_ax  # internal helper used in the base class

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
# Two-pass fit for one divergence, returning lower/upper for do(A=1), plus ehat1_oof
# -------------------------
def fit_two_pass_do1(
    EstimatorClass,
    div_name,
    X,
    A,
    Y,
    dual_net_config,
    fit_config,
    seed,
    propensity_model,
    m_model,
    propensity_cache,
    noise_const: float = 0.0,
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

    U = est_pos.predict_bound(a=1, X=X).astype(np.float32)
    L = (-est_neg.predict_bound(a=1, X=X)).astype(np.float32)

    ehat1 = est_pos.e1_hat_oof_.astype(np.float32)
    if noise_const != 0.0:
        eps = float(fit_config.get("eps_propensity", 1e-3))
        scale = (X.shape[0]) ** (-0.25)
        noise = noise_const * np.random.normal(loc=scale, scale=scale, size=ehat1.shape)
        ehat1 = np.clip(ehat1 + noise.astype(np.float32), eps, 1.0 - eps)
    return L, U, ehat1


# -------------------------
# Combined aggregator (unchanged)
# -------------------------
def combined_cwise_intersection(lower_mat, upper_mat, c=3):
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
                        and np.isclose(best_sum, sum_widths)
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
# FAST cluster aggregator (no sklearn per-sample KMeans calls)
# -------------------------
def _median(vals):
    arr = np.sort(np.asarray(vals, dtype=np.float64))
    n = int(arr.shape[0])
    if n % 2 == 1:
        return float(arr[n // 2])
    return float(0.5 * (arr[n // 2 - 1] + arr[n // 2]))


def _silhouette_score_1d(v_sorted: np.ndarray, labels: np.ndarray) -> float:
    '''
    Lightweight silhouette_score for 1D values with Euclidean distance.

    Conventions (matching sklearn):
    - Requires at least 2 clusters (>=2 unique labels).
    - Samples in singleton clusters get silhouette=0.
    '''
    v = v_sorted.astype(np.float64, copy=False).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    m = int(v.shape[0])
    uniq = np.unique(labels)
    if uniq.size < 2:
        return -np.inf

    # Pairwise distance matrix (m <= 5 here).
    D = np.abs(v.reshape(-1, 1) - v.reshape(1, -1))

    sil = np.zeros(m, dtype=np.float64)
    for i in range(m):
        li = labels[i]
        in_same = np.where(labels == li)[0]
        if in_same.size <= 1:
            sil[i] = 0.0
            continue

        # a_i: mean distance to own cluster excluding self
        a_i = float(np.mean(D[i, in_same[in_same != i]]))

        # b_i: min mean distance to other clusters
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
    '''Labels for contiguous partition of sorted values with cut positions in {1,...,m-1}.'''
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


def _best_cluster_partition_1d(
    values,
    k_candidates=(2, 3, 4),
    penalty_singleton=0.2,
):
    '''
    Select a clustering of 1D values by searching all contiguous partitions of the sorted values.

    Objective:
        silhouette_score - penalty_singleton * (#singleton clusters)

    Returns:
        list[list[float]] clusters (each cluster is a list of values).
    '''
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

    # Enumerate all contiguous partitions via cut positions.
    for k in k_candidates:
        if k < 2 or k > m:
            continue
        for cuts in combinations(tuple(range(1, m)), k - 1):
            labels = _partition_labels_from_cuts(m, cuts)
            # Skip if all singleton clusters (only possible when k==m).
            sizes = np.bincount(labels, minlength=k)
            if not np.any(sizes >= 2):
                continue
            sil = _silhouette_score_1d(v_sorted, labels)
            if not np.isfinite(sil):
                continue
            num_singletons = int(np.sum(sizes == 1))
            score = float(sil - penalty_singleton * num_singletons)
            sse = _sse_from_labels(v_sorted, labels)

            # Deterministic tie-breakers:
            # 1) higher score
            # 2) fewer singletons
            # 3) smaller k
            # 4) smaller SSE
            # 5) lexicographically smaller cuts
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
        # Fallback: 2 clusters, cut at median split (deterministic).
        k = 2 if m >= 2 else 1
        cut = (m // 2,)
        labels = _partition_labels_from_cuts(m, cut)
        best_labels = labels
        best_k = k

    # Build clusters from sorted labels.
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

        # Fallbacks mimic the original "second order statistic" logic.
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
# Helper: compute bounds for a method (base, combined, cluster)
# -------------------------
def compute_bounds_for_method(
    method,
    estimator_cls,
    X,
    A,
    Y,
    dual_net_config,
    fit_config,
    seed,
    propensity_model,
    m_model,
    base_divs,
    noise_const,
    propensity_cache,
    cluster_impl: str = "fast1d",
):
    method_up = method.strip()
    if method_up in base_divs:
        return fit_two_pass_do1(
            estimator_cls,
            method_up,
            X,
            A,
            Y,
            dual_net_config,
            fit_config,
            seed,
            propensity_model,
            m_model,
            propensity_cache=propensity_cache,
            noise_const=noise_const,
        )

    # Build matrices over base divergences.
    lowers = []
    uppers = []
    ehat_ref = None
    for div_name in base_divs:
        L, U, ehat1 = fit_two_pass_do1(
            estimator_cls,
            div_name,
            X,
            A,
            Y,
            dual_net_config,
            fit_config,
            seed,
            propensity_model,
            m_model,
            propensity_cache=propensity_cache,
            noise_const=noise_const,
        )
        lowers.append(L)
        uppers.append(U)
        if ehat_ref is None:
            ehat_ref = ehat1

    lower_mat = np.vstack(lowers)
    upper_mat = np.vstack(uppers)

    if method_up.lower() == "combined":
        L_out, U_out = combined_cwise_intersection(lower_mat, upper_mat, c=3)
    elif method_up.lower() == "cluster":
        if cluster_impl.lower() == "sklearn":
            # Backward-compatible: slow, but matches the earlier implementation.
            # Only import sklearn if used.
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            def _cluster_1d_sklearn(values, k, seed_local):
                X1 = np.array(values, dtype=np.float64).reshape(-1, 1)
                k = max(1, min(int(k), len(values)))
                km = KMeans(n_clusters=k, n_init=10, random_state=int(seed_local))
                labels = km.fit_predict(X1)
                clusters = []
                for lab in np.unique(labels):
                    clusters.append([values[i] for i in range(len(values)) if labels[i] == lab])
                return clusters

            def _select_cluster_k_sklearn(values, k_candidates, penalty_singleton, seed_local):
                best_k, best_score = None, -np.inf
                X1 = np.array(values, dtype=np.float64).reshape(-1, 1)
                for k in k_candidates:
                    if k > len(values):
                        continue
                    clusters = _cluster_1d_sklearn(values, k, seed_local)
                    if not any(len(c) >= 2 for c in clusters):
                        continue
                    labels = []
                    for idx, c in enumerate(clusters):
                        labels.extend([idx] * len(c))
                    labels = np.array(labels, dtype=int)
                    sep = -np.inf if len(np.unique(labels)) < 2 else float(silhouette_score(X1, labels))
                    num_singletons = sum(1 for c in clusters if len(c) == 1)
                    score = float(sep - penalty_singleton * num_singletons)
                    if score > best_score:
                        best_score, best_k = score, k
                if best_k is None:
                    best_k = min(2, len(values))
                return int(best_k)

            def cluster_per_sample_sklearn(lower_mat, upper_mat, seed_local, k_candidates=(2, 3, 4), penalty_singleton=0.2):
                M, n = lower_mat.shape
                outL = np.empty(n, dtype=np.float32)
                outU = np.empty(n, dtype=np.float32)
                for i in range(n):
                    lowers_i = [float(x) for x in lower_mat[:, i]]
                    uppers_i = [float(x) for x in upper_mat[:, i]]
                    kL = _select_cluster_k_sklearn(lowers_i, k_candidates, penalty_singleton, seed_local=seed_local + 777 + i)
                    kU = _select_cluster_k_sklearn(uppers_i, k_candidates, penalty_singleton, seed_local=seed_local + 888 + i)
                    cL = _cluster_1d_sklearn(lowers_i, kL, seed_local=seed_local + 999 + i)
                    cU = _cluster_1d_sklearn(uppers_i, kU, seed_local=seed_local + 1000 + i)
                    lo = _cluster_choose_lower(cL)
                    hi = _cluster_choose_upper(cU)
                    lowers_s = np.sort(np.asarray(lowers_i, dtype=np.float64))
                    uppers_s = np.sort(np.asarray(uppers_i, dtype=np.float64))
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

            L_out, U_out = cluster_per_sample_sklearn(lower_mat, upper_mat, seed_local=seed)
        else:
            L_out, U_out = cluster_per_sample_fast1d(lower_mat, upper_mat, seed=seed)
    else:
        raise KeyError(f"Unknown method '{method}'.")

    return L_out.astype(np.float32), U_out.astype(np.float32), ehat_ref


# -------------------------
# Main Monte Carlo
# -------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Monte Carlo ablation: debiased vs naive.")
    parser.add_argument(
        "--divergence",
        type=str,
        default="KL",
        help="One of KL, TV, Hellinger, Chi2, JS, combined, cluster",
    )
    parser.add_argument("--seed0", type=int, default=20251215)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument(
        "--n_list",
        type=str,
        default="400,800,1600",
        help="comma-separated sample sizes",
    )
    parser.add_argument("--R", type=int, default=30, help="MC replicates per n")
    parser.add_argument(
        "--noise_const",
        type=float,
        default=0.0,
        help="Multiply N(n^{-1/4}, n^{-1/4}) noise added to ehat (propensity). (Returned only; does not affect bounds.)",
    )
    parser.add_argument("--unique_save", action="store_true", help="Save outputs with a unique timestamp suffix.")
    parser.add_argument(
        "--stat",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation statistic across replicates (mean or median).",
    )
    parser.add_argument(
        "--cluster_impl",
        type=str,
        default="fast1d",
        choices=["fast1d", "sklearn"],
        help="Cluster aggregator backend for divergence='cluster'. fast1d avoids per-sample sklearn KMeans calls.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-replicate prints (recommended for speed).",
    )
    args = parser.parse_args()

    seed0 = int(args.seed0)
    d = int(args.d)
    div = args.divergence.strip()
    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    R = int(args.R)
    noise_const = float(args.noise_const)
    base_divs = ["KL", "TV", "Hellinger", "Chi2", "JS"]
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

    # Intentionally misspecify propensity to make ablation visible:
    propensity_model = "xgboost"
    m_model = "xgboost"

    fit_config = {
        "n_folds": 3,
        "num_epochs": 100,
        "batch_size": 256,
        "lr": 5e-4,  # slower Adam step to make optimization more conservative
        "weight_decay": 1e-4,
        "max_grad_norm": 10.0,
        "eps_propensity": 1e-3,
        "deterministic_torch": True,
        "train_m_on_fold": True,
        "propensity_config": {
            "n_estimators": 300,
            "max_depth": 10,
            "learning_rate": 0.005,  # slower boosting step
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": 1,
            "verbosity": 0,
        },
        # regression head for Z
        "m_config": {
            "n_estimators": 400,
            "max_depth": 10,
            "learning_rate": 0.005,  # slower boosting step
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

    rows = []
    print(f"Running Monte Carlo ablation (method={div}, cluster_impl={args.cluster_impl}) ...")

    for n in n_list:
        for r in tqdm(range(R), desc=f"n={n}", leave=False):
            seed = seed0 + 1000 * n + r
            data = generate_data(n=n, d=d, seed=seed, structural_type="simpson")
            X, A, Y = data["X"], data["A"], data["Y"]
            truth = data["GroundTruth"](1, X).astype(np.float32)

            # Pre-fit and cache propensity models once per replicate (reused across:
            #   - debiased vs naive, and
            #   - phi vs -phi, and
            #   - all divergences (for combined/cluster).
            prop_cache = prefit_propensity_cache(
                X=X,
                A=A,
                propensity_model=propensity_model,
                propensity_config=fit_config["propensity_config"],
                n_folds=fit_config["n_folds"],
                seed=seed,
                eps_propensity=fit_config["eps_propensity"],
            )

            # Debiased
            Ld, Ud, _ = compute_bounds_for_method(
                method=div,
                estimator_cls=DebiasedCausalBoundEstimator,
                X=X,
                A=A,
                Y=Y,
                dual_net_config=dual_net_config,
                fit_config=fit_config,
                seed=seed,
                propensity_model=propensity_model,
                m_model=m_model,
                base_divs=base_divs,
                noise_const=noise_const,
                propensity_cache=prop_cache,
                cluster_impl=args.cluster_impl,
            )
            cov_d = float(np.mean((truth >= Ld) & (truth <= Ud)))
            wid_d = float(np.mean(Ud - Ld))

            # Naive
            Ln, Un, _ = compute_bounds_for_method(
                method=div,
                estimator_cls=NaiveCausalBoundEstimator,
                X=X,
                A=A,
                Y=Y,
                dual_net_config=dual_net_config,
                fit_config=fit_config,
                seed=seed,
                propensity_model=propensity_model,
                m_model=m_model,
                base_divs=base_divs,
                noise_const=noise_const,
                propensity_cache=prop_cache,
                cluster_impl=args.cluster_impl,
            )
            cov_n = float(np.mean((truth >= Ln) & (truth <= Un)))
            wid_n = float(np.mean(Un - Ln))

            rows.append(
                {
                    "n": n,
                    "rep": r,
                    "coverage_debiased": cov_d,
                    "width_debiased": wid_d,
                    "coverage_naive": cov_n,
                    "width_naive": wid_n,
                }
            )

            if not args.quiet:
                print(
                    f"[n={n}, rep={r}] cov(deb)={cov_d:.3f}, cov(naive)={cov_n:.3f}, "
                    f"w(deb)={wid_d:.3f}, w(naive)={wid_n:.3f}",
                    flush=True,
                )

    os.makedirs("experiments", exist_ok=True)
    res = pd.DataFrame(rows)
    results_fname = name_with_suffix("plot2_mc_results", "csv")
    res.to_csv(results_fname, index=False)

    # Aggregate
    if args.stat == "mean":
        agg_fn = lambda s: float(np.mean(s))
    else:
        agg_fn = lambda s: float(np.median(s))

    agg = (
        res.groupby("n")
        .agg(
            cov_d_mean=("coverage_debiased", agg_fn),
            cov_d_se=("coverage_debiased", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            cov_n_mean=("coverage_naive", agg_fn),
            cov_n_se=("coverage_naive", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            wid_d_mean=("width_debiased", agg_fn),
            wid_d_se=("width_debiased", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            wid_n_mean=("width_naive", agg_fn),
            wid_n_se=("width_naive", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        )
        .reset_index()
    )

    # Plot coverage vs n
    z = 1.96  # 95% CI multiplier
    fig_cov, ax_cov = plt.subplots(figsize=(7.0, 4.0))
    ax_cov.errorbar(agg["n"], agg["cov_d_mean"], yerr=agg["cov_d_se"], marker="o", capsize=3, label="debiased")
    ax_cov.errorbar(agg["n"], agg["cov_n_mean"], yerr=agg["cov_n_se"], marker="o", capsize=3, label="naive")
    ax_cov.fill_between(agg["n"], agg["cov_d_mean"] - z * agg["cov_d_se"], agg["cov_d_mean"] + z * agg["cov_d_se"], alpha=0.2)
    ax_cov.fill_between(agg["n"], agg["cov_n_mean"] - z * agg["cov_n_se"], agg["cov_n_mean"] + z * agg["cov_n_se"], alpha=0.2)
    ax_cov.set_ylim(0.0, 1.05)
    ax_cov.set_xlabel("Sample size n")
    ax_cov.set_ylabel("Coverage")
    ax_cov.set_title(f"Coverage vs n ({div})")
    ax_cov.legend()
    fig_cov.tight_layout()
    cov_fname = name_with_suffix("plot2_mc_ablation_coverage", "png")
    fig_cov.savefig(cov_fname, dpi=200)

    # Plot width vs n
    fig_wid, ax_wid = plt.subplots(figsize=(7.0, 4.0))
    ax_wid.errorbar(agg["n"], agg["wid_d_mean"], yerr=agg["wid_d_se"], marker="o", capsize=3, label="debiased")
    ax_wid.errorbar(agg["n"], agg["wid_n_mean"], yerr=agg["wid_n_se"], marker="o", capsize=3, label="naive")
    ax_wid.fill_between(agg["n"], agg["wid_d_mean"] - z * agg["wid_d_se"], agg["wid_d_mean"] + z * agg["wid_d_se"], alpha=0.2)
    ax_wid.fill_between(agg["n"], agg["wid_n_mean"] - z * agg["wid_n_se"], agg["wid_n_mean"] + z * agg["wid_n_se"], alpha=0.2)
    ax_wid.set_xlabel("Sample size n")
    ax_wid.set_ylabel("Mean interval width")
    ax_wid.set_title(f"Width vs n ({div})")
    ax_wid.legend()
    fig_wid.tight_layout()
    wid_fname = name_with_suffix("plot2_mc_ablation_width", "png")
    fig_wid.savefig(wid_fname, dpi=200)

    # Save artifacts for re-plotting
    artifacts = {
        "results": res,
        "agg": agg,
        "divergence": div,
        "args": vars(args),
        "fit_config": fit_config,
        "timestamp": stamp,
    }
    import pickle

    artifacts_fname = name_with_suffix("plot2_artifacts", "pkl")
    with open(artifacts_fname, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Experiment summary logs
    summary = {
        "timestamp": stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "files": {
            "results_csv": results_fname,
            "coverage_png": cov_fname,
            "width_png": wid_fname,
            "artifacts_pkl": artifacts_fname,
        },
        "divergence": div,
        "settings": {"n_list": n_list, "R": R, "d": d, "cluster_impl": args.cluster_impl},
    }
    summary_fname = name_with_suffix("plot2_summary", "json")
    with open(summary_fname, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Saved: {results_fname}")
    print(f"Saved: {cov_fname}")
    print(f"Saved: {wid_fname}")
    print(f"Saved: {artifacts_fname}")
    print(f"Saved: {summary_fname}")
