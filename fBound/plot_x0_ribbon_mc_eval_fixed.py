"""
Monte Carlo ribbon plot of causal bounds vs X0 using a fixed evaluation grid.

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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **k: x  # type: ignore

from fbound.estimators.causal_bound import (
    DebiasedCausalBoundEstimator,
    aggregate_endpointwise,
    prefit_propensity_cache,
)
from fbound.utils.data_generating import generate_data

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


def smooth_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "spline",
    smooth_grid_n: int = 500,
    window: int = 5,
    spline_k: int = 3,
    spline_s: float = -1.0,
    lowess_frac: float = 0.2,
    lowess_it: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.array([]), np.array([])

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    method = str(method).lower()

    if method == "none":
        return x, y

    if method == "moving_avg":
        window = max(1, int(window))
        if window <= 1 or x.size < 2:
            return x, y
        kernel = np.ones(window, dtype=np.float64) / float(window)
        y_s = np.convolve(y, kernel, mode="same")
        return x, y_s

    if method == "spline":
        if x.size <= int(spline_k):
            return x, y
        s_val = None if float(spline_s) < 0 else float(spline_s)
        grid_n = max(2, int(smooth_grid_n))
        x_grid = np.linspace(float(x.min()), float(x.max()), grid_n)

        def _eval_spline(s_val_inner):
            try:
                spline = UnivariateSpline(x, y, k=int(spline_k), s=s_val_inner)
                y_grid = spline(x_grid)
            except Exception:
                return None
            if not np.all(np.isfinite(y_grid)):
                return None
            return y_grid

        y_grid = _eval_spline(s_val)
        if y_grid is None and s_val is not None:
            y_grid = _eval_spline(None)
        if y_grid is None:
            return x, y
        return x_grid, y_grid

    if method == "lowess":
        try:
            out = lowess(y, x, frac=float(lowess_frac), it=int(lowess_it), return_sorted=True)
            return out[:, 0], out[:, 1]
        except Exception:
            return x, y

    raise ValueError(f"Unknown smoothing method '{method}'.")


def _predict_proba_class1(model: Any, X: np.ndarray) -> np.ndarray:
    """Return P(A=1|X) while respecting sklearn's class ordering."""
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"predict_proba must return shape (n,2+). Got {proba.shape}")
    col = 1
    classes = getattr(model, "classes_", None)
    if classes is not None:
        classes = np.asarray(classes)
        if 1 in classes:
            col = int(np.where(classes == 1)[0][0])
    return proba[:, col]


def _predict_propensity_eval(
    propensity_cache: dict,
    X_eval: np.ndarray,
    eps_propensity: float,
) -> np.ndarray:
    models = propensity_cache.get("models", None)
    if not models:
        raise ValueError("propensity_cache is missing fitted models for eval-axis predictions.")
    Xc = np.asarray(X_eval, dtype=np.float64)
    acc = None
    for model in models:
        e1 = _predict_proba_class1(model, Xc)
        if acc is None:
            acc = e1
        else:
            acc += e1
    e1_mean = acc / float(len(models))
    e1_mean = np.clip(e1_mean, eps_propensity, 1.0 - eps_propensity).astype(np.float32)
    return e1_mean

# -------------------------
# Phi definitions
# -------------------------
def phi_identity(y: torch.Tensor) -> torch.Tensor:
    return y


def phi_neg(y: torch.Tensor) -> torch.Tensor:
    return -y


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



def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="MC ribbon plot of causal bounds vs X0 (fixed evaluation grid).")
    parser.add_argument("--m", type=int, default=20, help="number of MC replicates")
    parser.add_argument("--base_seed", type=int, default=20190602, help="base seed; seed_j = base_seed + j")
    parser.add_argument("--n", type=int, default=5000, help="samples per replicate")
    parser.add_argument("--d", type=int, default=10, help="feature dimension")
    parser.add_argument(
        "--divergence",
        type=str,
        default="kth",
        help="Comma-separated divergences from {KL,TV,Hellinger,Chi2,JS,kth,tight_kth}.",
    )
    parser.add_argument(
        "--structural_type",
        type=str,
        default="cyclic2",
        choices=["linear", "nonlinear", "simpson", "cyclic", "cyclic2", "probit_sine"],
        help="Data-generating process type.",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Aggregation statistic across MC replicates.",
    )
    parser.add_argument(
        "--eval_axis",
        type=str,
        default="x0",
        choices=["x0", "propensity", "both"],
        help="Axis for ribbon plots: X0, estimated propensity, or both.",
    )
    parser.add_argument("--min_valid_rate", type=float, default=1.0, help="Minimum validity rate for keeping an eval point.")
    parser.add_argument("--unique_save", action="store_true", help="Add unique timestamp suffix to outputs.")
    parser.add_argument("--outdir", type=str, default="experiments", help="Output directory.")
    parser.add_argument("--xgb_n_jobs", type=int, default=-1, help="n_jobs for xgboost models (-1 uses all cores).")
    parser.add_argument("--num_epochs", type=int, default=100, help="Dual net epochs.")
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
    parser.add_argument("--n_eval", type=int, default=20, help="Number of evaluation points (sampled).")
    parser.add_argument("--smooth_window", type=int, default=5, help="Smoothing window (moving average) for plotting.")
    parser.add_argument(
        "--noise_dist",
        type=str,
        default="normal",
        choices=["normal", "t3"],
        help="Noise distribution used in generate_data.",
    )
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
    parser.add_argument("--plot_raw_points", action="store_true", help="Overlay raw (unsmoothed) points on the plot.")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Early stopping patience (0 disables).",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation loss improvement to reset early stopping.",
    )
    parser.add_argument(
        "--early_stop_fraction",
        type=float,
        default=0.2,
        help="Fraction of fold used for early-stopping validation.",
    )
    parser.add_argument(
        "--kval",
        type=int,
        default=None,
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

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if args.unique_save else ""

    def name_with_suffix(base: str, ext: str) -> str:
        fname = f"{base}_{stamp}.{ext}" if stamp else f"{base}.{ext}"
        return os.path.join(args.outdir, fname)

    dual_net_config = {
        "hidden_sizes": (64, 64),
        "activation": "relu",
        "dropout": 0.1,
        "h_clip": 20.0,
        "device": args.device,
    }
    
    propensity_model = "xgboost"
    m_model = "xgboost"

    fit_config = {
        "n_folds": args.n_folds,
        "num_epochs": args.num_epochs,
        "batch_size": None,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "max_grad_norm": None,
        "eps_propensity": args.eps_propensity,
        "deterministic_torch": True,
        "train_m_on_fold": True,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stop_fraction": args.early_stop_fraction,
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
    allowed_divs = set(base_divs + ["kth", "tight_kth"])
    div_list = [div.strip() for div in args.divergence.split(",") if div.strip()]
    if not div_list:
        div_list = ["kth"]
    for div in div_list:
        if div not in allowed_divs:
            raise ValueError(f"Unknown divergence '{div}'. Allowed: {sorted(allowed_divs)}")
    needs_base = set()
    if any(div in {"kth", "tight_kth"} for div in div_list):
        needs_base.update(base_divs)
    needs_base.update([div for div in div_list if div in base_divs])

    # Fixed evaluation set sampled from the DGP (shared across replicates)
    with StepTimer("build X_eval grid", use_tqdm=False, enabled=timing_enabled):
        eval_seed = args.base_seed + 10**6
        eval_data = generate_data(
            n=args.n_eval,
            d=d,
            seed=eval_seed,
            structural_type=args.structural_type,
            noise_dist=args.noise_dist,
        )
        X_eval = np.asarray(eval_data["X"], dtype=np.float32)
        X0_eval = X_eval[:, 0]
    need_propensity_axis = args.eval_axis in {"propensity", "both"}

    truth_mat = np.full((m, args.n_eval), np.nan, dtype=np.float32)
    prop_eval_mat = np.full((m, args.n_eval), np.nan, dtype=np.float32) if need_propensity_axis else None
    seeds = []
    upper_dict = {div: np.full((m, args.n_eval), np.nan, dtype=np.float32) for div in div_list}
    lower_dict = {div: np.full((m, args.n_eval), np.nan, dtype=np.float32) for div in div_list}
    valid_dict = {div: np.zeros((m, args.n_eval), dtype=bool) for div in div_list}
    with StepTimer("MC replicate loop", use_tqdm=True, enabled=timing_enabled):
        for j in tqdm(range(m), desc="MC replicates"):
            seed = args.base_seed + j
            seeds.append(seed)
            with StepTimer(f"replicate {j}", use_tqdm=True, enabled=timing_enabled):
                with StepTimer("data generation", use_tqdm=True, enabled=timing_enabled):
                    data = generate_data(
                        n=n,
                        d=d,
                        seed=seed,
                        structural_type=args.structural_type,
                        noise_dist=args.noise_dist,
                    )
                    X_tr = np.asarray(data["X"], dtype=np.float32)
                    A_tr = np.asarray(data["A"], dtype=np.float32)
                    Y_tr = np.asarray(data["Y"], dtype=np.float32)
                    truth_eval = np.asarray(data["GroundTruth"](1, X_eval), dtype=np.float32).reshape(-1)

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
                    if need_propensity_axis and prop_eval_mat is not None:
                        prop_eval_mat[j, :] = _predict_propensity_eval(
                            prop_cache,
                            X_eval,
                            fit_config["eps_propensity"],
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
                        elif div == "kth":
                            if lower_base is None:
                                lower_base = np.vstack([base_outputs[b][0] for b in base_divs])
                                upper_base = np.vstack([base_outputs[b][1] for b in base_divs])
                            k_val = int(args.kval) if args.kval is not None else int(lower_base.shape[0])
                            if k_val < 1 or k_val > int(lower_base.shape[0]):
                                raise ValueError(f"--kval must be in [1, {int(lower_base.shape[0])}] (got {k_val}).")
                            valid_up = np.isfinite(upper_base)
                            valid_lo = np.isfinite(lower_base)
                            agg = aggregate_endpointwise(
                                lower_mat=lower_base,
                                upper_mat=upper_base,
                                valid_up=valid_up,
                                valid_lo=valid_lo,
                                k_up=k_val,
                                k_lo=k_val,
                            )
                            L = agg["lower"].astype(np.float32)
                            U = agg["upper"].astype(np.float32)
                        elif div == "tight_kth":
                            if lower_base is None:
                                lower_base = np.vstack([base_outputs[b][0] for b in base_divs])
                                upper_base = np.vstack([base_outputs[b][1] for b in base_divs])
                            valid_up = np.isfinite(upper_base)
                            valid_lo = np.isfinite(lower_base)
                            agg = aggregate_endpointwise(
                                lower_mat=lower_base,
                                upper_mat=upper_base,
                                valid_up=valid_up,
                                valid_lo=valid_lo,
                                k_up=1,
                                k_lo=1,
                            )
                            L = agg["lower"].astype(np.float32)
                            U = agg["upper"].astype(np.float32)
                        else:
                            raise ValueError(f"Unsupported divergence '{div}'")

                        valid = np.isfinite(U) & np.isfinite(L) & (L <= U)

                        upper_dict[div][j, :] = U
                        lower_dict[div][j, :] = L
                        valid_dict[div][j, :] = valid

                with StepTimer("store truth", use_tqdm=True, enabled=timing_enabled):
                    truth_mat[j, :] = truth_eval

    if args.stat == "mean":
        agg_fn = np.nanmean
    else:
        agg_fn = np.nanmedian

    axis_specs = []
    if args.eval_axis in {"x0", "both"}:
        axis_specs.append(("x0", X0_eval, "X0"))
    if args.eval_axis in {"propensity", "both"}:
        if prop_eval_mat is None:
            raise RuntimeError("Propensity axis requested but estimated propensities are unavailable.")
        axis_eval = agg_fn(prop_eval_mat, axis=0)
        axis_specs.append(("propensity", axis_eval, "e(A=1|X)"))

    for axis_key, axis_eval, axis_label in axis_specs:
        base_name = (
            "plot_x0_ribbon_mc_eval_fixed"
            if axis_key == "x0"
            else "plot_x0_ribbon_mc_eval_fixed_propensity"
        )
        prop_xlim = None
        if axis_key == "propensity":
            finite = np.isfinite(axis_eval)
            if np.any(finite):
                prop_xlim = (float(np.min(axis_eval[finite])), float(np.max(axis_eval[finite])))

        with StepTimer(f"aggregate + smooth ribbons ({axis_key})", use_tqdm=False, enabled=timing_enabled):
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
            # Plot
            plt.figure(figsize=(7.0, 4.0))
            color_map = {
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
                    res["x_s"],
                    res["l_s"],
                    res["u_s"],
                    alpha=0.2,
                    color=c,
                    label=f"{res['div']} bounds",
                )
                plt.plot(res["x_s"], res["l_s"], color=c, alpha=0.7, linewidth=1.0)
                plt.plot(res["x_s"], res["u_s"], color=c, alpha=0.7, linewidth=1.0)
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
            plt.title(
                f"Causal bounds vs {axis_label} (stat={args.stat}, divs={','.join(div_list)}, struct={args.structural_type})"
            )
            if prop_xlim is not None:
                plt.xlim(*prop_xlim)
            plt.legend()
            plt.tight_layout()
            fig_path = name_with_suffix(base_name, "png")
            plt.savefig(fig_path, dpi=200)
            plt.close()

        with StepTimer(f"save artifacts ({axis_key})", use_tqdm=False, enabled=timing_enabled):
            # Artifacts
            artifacts = {
                "X_eval": X_eval,
                "x_axis_eval": axis_eval,
                "eval_axis": axis_key,
                "lower_dict": lower_dict,
                "upper_dict": upper_dict,
                "truth_mat": truth_mat,
                "valid_dict": valid_dict,
                "args": vars(args),
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
                    "artifacts_pkl": artifacts_path,
                },
                "args": vars(args),
                "n": n,
                "d": d,
                "m": m,
                "divergences": div_list,
                "structural_type": args.structural_type,
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
                f"[{base_name}] ts={summary['timestamp']} m={m} n={n} d={d} stat={args.stat} "
                f"divs={','.join(div_list)} struct={args.structural_type} selected={summary['selected_counts']} "
                f"args={vars(args)} files={summary['files']}"
            )
            log_path = name_with_suffix(f"{base_name}_log", "txt")
            with open(log_path, "a") as f:
                f.write(log_line + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        _log_active_step_error(use_tqdm=False)
        raise
