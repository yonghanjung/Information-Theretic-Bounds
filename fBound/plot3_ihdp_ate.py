import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from causal_bound import DebiasedCausalBoundEstimator

# -------------------------
# Phi
# -------------------------
def phi_identity(y: torch.Tensor) -> torch.Tensor:
    return y

def phi_neg(y: torch.Tensor) -> torch.Tensor:
    return -y

# -------------------------
# Load IHDP-like .npz
# Supports common keys: x, t, yf, ycf (CEVAE-format)
# Handles shapes (n,d) or (n,d,R)
# -------------------------
def load_ihdp_npz(path, rep_id=0):
    data = np.load(path)
    # Common CEVAE keys
    x = data["x"]            # (n,d) or (n,d,R)
    t = data["t"]            # (n,) or (n,R)
    yf = data["yf"]          # factual outcome (n,) or (n,R)
    ycf = data["ycf"]        # counterfactual outcome (n,) or (n,R)

    def take_rep(arr):
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            # (n,R) -> take rep
            return arr[:, rep_id]
        if arr.ndim == 3:
            # (n,d,R) -> take rep
            return arr[:, :, rep_id]
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    X_full = take_rep(x).astype(np.float32)
    A = take_rep(t).astype(np.int64).reshape(-1)
    yf = take_rep(yf).astype(np.float32).reshape(-1)
    ycf = take_rep(ycf).astype(np.float32).reshape(-1)

    # Reconstruct potential outcomes
    # If A=1: factual is Y1, counterfactual is Y0
    # If A=0: factual is Y0, counterfactual is Y1
    Y1 = np.where(A == 1, yf, ycf).astype(np.float32)
    Y0 = np.where(A == 0, yf, ycf).astype(np.float32)
    Y = yf.astype(np.float32)  # equals A*Y1 + (1-A)*Y0

    return X_full, A, Y, Y0, Y1

# -------------------------
# Two-pass fit, but return bounds for BOTH a=0 and a=1 without refitting
# -------------------------
def fit_two_pass_both_arms(div_name, X, A, Y, dual_net_config, fit_config, seed, propensity_model, m_model):
    est_pos = DebiasedCausalBoundEstimator(
        divergence=div_name,
        phi=phi_identity,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
    ).fit(X, A, Y)

    est_neg = DebiasedCausalBoundEstimator(
        divergence=div_name,
        phi=phi_neg,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
    ).fit(X, A, Y)

    # Upper bounds
    U1 = est_pos.predict_bound(a=1, X=X).astype(np.float32)
    U0 = est_pos.predict_bound(a=0, X=X).astype(np.float32)
    # Lower via sign flip
    L1 = (-est_neg.predict_bound(a=1, X=X)).astype(np.float32)
    L0 = (-est_neg.predict_bound(a=0, X=X)).astype(np.float32)

    ehat1 = est_pos.e1_hat_oof_.astype(np.float32)
    return {"ehat1": ehat1, "a1": (L1, U1), "a0": (L0, U0)}

# -------------------------
# Aggregators (combined + cluster) – reuse from Plot 1 script
# For brevity, import or paste the functions:
#   combined_cwise_intersection(...)
#   cluster_per_sample(...)
# -------------------------
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
                lower_out[i], upper_out[i] = np.float32(lo_f), np.float32(hi_f)
            else:
                lower_out[i], upper_out[i] = np.float32(0.0), np.float32(0.0)
            continue

        lv = lowers[valid]
        uv = uppers[valid]
        if len(lv) < 2:
            lower_out[i], upper_out[i] = np.float32(np.min(lv)), np.float32(np.max(uv))
            continue

        best_subset, best_width, best_sum, best_L, best_U = None, np.inf, np.inf, np.nan, np.nan
        max_t = min(max(c, 2), len(lv))
        for t in range(max_t, 1, -1):
            found_any = False
            for combo in combinations(range(len(lv)), t):
                L_int = float(np.max(lv[list(combo)]))
                U_int = float(np.min(uv[list(combo)]))
                width = U_int - L_int
                if width < 0:
                    continue
                sum_widths = float(np.sum(uv[list(combo)] - lv[list(combo)]))
                if (width < best_width) or (np.isclose(width, best_width) and sum_widths < best_sum):
                    best_subset, best_width, best_sum, best_L, best_U = combo, width, sum_widths, L_int, U_int
                found_any = True
            if found_any and best_subset is not None:
                break

        if best_subset is not None:
            lower_out[i], upper_out[i] = np.float32(best_L), np.float32(best_U)
        else:
            lower_out[i], upper_out[i] = np.float32(np.min(lv)), np.float32(np.max(uv))

    return lower_out, upper_out

def _cluster_1d(values, k, seed):
    X1 = np.array(values, dtype=np.float64).reshape(-1, 1)
    k = max(1, min(k, len(values)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X1)
    clusters = []
    for lab in np.unique(labels):
        clusters.append([values[i] for i in range(len(values)) if labels[i] == lab])
    return clusters

def _median(vals):
    arr = sorted(vals)
    n = len(arr)
    if n % 2 == 1:
        return float(arr[n // 2])
    return 0.5 * (arr[n // 2 - 1] + arr[n // 2])

def _select_cluster_k(values, k_candidates, penalty_singleton, seed):
    best_k, best_score = None, -np.inf
    X1 = np.array(values, dtype=np.float64).reshape(-1, 1)
    for k in k_candidates:
        if k > len(values):
            continue
        clusters = _cluster_1d(values, k, seed=seed)
        if not any(len(c) >= 2 for c in clusters):
            continue
        labels = []
        for idx, c in enumerate(clusters):
            labels.extend([idx] * len(c))
        labels = np.array(labels, dtype=int)
        sep = -np.inf if len(np.unique(labels)) < 2 else silhouette_score(X1, labels)
        num_singletons = sum(1 for c in clusters if len(c) == 1)
        score = sep - penalty_singleton * num_singletons
        if score > best_score:
            best_score, best_k = score, k
    if best_k is None:
        best_k = min(2, len(values))
    return best_k

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

def cluster_per_sample(lower_mat, upper_mat, seed, k_candidates=(2,3,4), penalty_singleton=0.2):
    M, n = lower_mat.shape
    outL = np.empty(n, dtype=np.float32)
    outU = np.empty(n, dtype=np.float32)
    for i in range(n):
        lowers = [float(x) for x in lower_mat[:, i]]
        uppers = [float(x) for x in upper_mat[:, i]]
        kL = _select_cluster_k(lowers, k_candidates, penalty_singleton, seed=seed + 777 + i)
        kU = _select_cluster_k(uppers, k_candidates, penalty_singleton, seed=seed + 888 + i)
        cL = _cluster_1d(lowers, kL, seed=seed + 999 + i)
        cU = _cluster_1d(uppers, kU, seed=seed + 1000 + i)
        lo = _cluster_choose_lower(cL)
        hi = _cluster_choose_upper(cU)
        if lo is None:
            lo = sorted(lowers)[1] if len(lowers) >= 2 else lowers[0]
        if hi is None:
            hi = sorted(uppers)[-2] if len(uppers) >= 2 else uppers[0]
        if lo > hi:
            lo = sorted(lowers)[1] if len(lowers) >= 2 else lo
            hi = sorted(uppers)[-2] if len(uppers) >= 2 else hi
        outL[i], outU[i] = np.float32(lo), np.float32(hi)
    return outL, outU

# -------------------------
# Main IHDP run (one replicate)
# -------------------------
if __name__ == "__main__":
    IHDP_PATH = "ihdp_npci_1-100.train.npz"  # <-- change this
    rep_id = 0
    seed = 20251215

    X_full, A, Y, Y0, Y1 = load_ihdp_npz(IHDP_PATH, rep_id=rep_id)

    # Select observed covariates: keep only d_obs, hide the rest
    d_obs = 10
    # One defensible way: keep the least treatment-predictive features, so hidden confounding is stronger.
    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(X_full, A)
    coef = np.abs(lr.coef_.reshape(-1))
    keep_idx = np.argsort(coef)[:d_obs]  # least predictive of A
    X_obs = X_full[:, keep_idx].astype(np.float32)

    # Standardize observed covariates (helps neural nets)
    X_obs = StandardScaler().fit_transform(X_obs).astype(np.float32)

    true_ate = float(np.mean(Y1 - Y0))

    base_divs = ["KL", "TV", "Hellinger", "Chi2", "JS"]

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
        "num_epochs": 80,
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

    # Fit base divergence bounds once; get both arms from each fit
    L1_list, U1_list = [], []
    L0_list, U0_list = [], []
    for div in base_divs:
        out = fit_two_pass_both_arms(
            div_name=div, X=X_obs, A=A, Y=Y,
            dual_net_config=dual_net_config, fit_config=fit_config,
            seed=seed, propensity_model=propensity_model, m_model=m_model
        )
        L1, U1 = out["a1"]
        L0, U0 = out["a0"]
        L1_list.append(L1); U1_list.append(U1)
        L0_list.append(L0); U0_list.append(U0)

    L1_mat, U1_mat = np.vstack(L1_list), np.vstack(U1_list)
    L0_mat, U0_mat = np.vstack(L0_list), np.vstack(U0_list)

    # Aggregate to combined + cluster, per sample
    L1_comb, U1_comb = combined_cwise_intersection(L1_mat, U1_mat, c=3)
    L0_comb, U0_comb = combined_cwise_intersection(L0_mat, U0_mat, c=3)
    L1_clu,  U1_clu  = cluster_per_sample(L1_mat, U1_mat, seed=seed)
    L0_clu,  U0_clu  = cluster_per_sample(L0_mat, U0_mat, seed=seed)

    # Convert conditional bounds into marginal bounds by averaging over empirical X
    mu1_comb = (float(np.mean(L1_comb)), float(np.mean(U1_comb)))
    mu0_comb = (float(np.mean(L0_comb)), float(np.mean(U0_comb)))
    mu1_clu  = (float(np.mean(L1_clu)),  float(np.mean(U1_clu)))
    mu0_clu  = (float(np.mean(L0_clu)),  float(np.mean(U0_clu)))

    # ATE bounds
    ate_comb = (mu1_comb[0] - mu0_comb[1], mu1_comb[1] - mu0_comb[0])
    ate_clu  = (mu1_clu[0]  - mu0_clu[1],  mu1_clu[1]  - mu0_clu[0])

    summary = pd.DataFrame([
        {"method": "combined", "ATE_L": ate_comb[0], "ATE_U": ate_comb[1], "ATE_width": ate_comb[1]-ate_comb[0], "true_ATE": true_ate},
        {"method": "cluster",  "ATE_L": ate_clu[0],  "ATE_U": ate_clu[1],  "ATE_width": ate_clu[1]-ate_clu[0],  "true_ATE": true_ate},
    ])
    summary.to_csv("plot3_ihdp_summary.csv", index=False)

    # Plot: ATE intervals
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.5))
    y = np.arange(len(summary))
    for i, row in summary.iterrows():
        ax.plot([row["ATE_L"], row["ATE_U"]], [i, i], marker="o")
    ax.axvline(true_ate, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(summary["method"].tolist())
    ax.set_xlabel("ATE")
    ax.set_title(f"IHDP replicate {rep_id}: ATE bounds with hidden confounders (d_obs={d_obs})")
    fig.tight_layout()
    fig.savefig("plot3_ihdp_ate_intervals.png", dpi=200)

    print("Saved plot3_ihdp_ate_intervals.png and plot3_ihdp_summary.csv")
    print(summary)
