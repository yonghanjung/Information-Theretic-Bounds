"""
Monte Carlo experiment: debiased vs naive bounds with coverage-v2 metrics.

Fits debiased and naive estimators on samples of size n, evaluates bounds on a
fixed X_eval grid, and compares against an oracle bound from a large sample.
This script is a superset of plot_n_coverage_v2.py and includes its metrics and
plots plus debiased diagnostics (oracle error curves, propensity RMSE, etc.).
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
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **k: x  # type: ignore

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fbound.estimators.causal_bound import (
    DebiasedCausalBoundEstimator,
    _concat_ax,
    _predict_proba_class1,
    aggregate_endpointwise,
    prefit_propensity_cache,
)
from fbound.utils.data_generating import generate_data


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


def derive_seed(base_seed: int, rep: int, n: int, tag: str) -> int:
    h = zlib.adler32(f"{tag}|{rep}|{n}|{base_seed}".encode())
    return int(base_seed + 10_000_000 * int(rep) + 10_000 * int(n) + (h % 10_000))


def stable_hash_n(n: int) -> int:
    return int(zlib.adler32(str(int(n)).encode()))

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
    def _debiased_loss_batch(
        self,
        X,
        A,
        Y,
        e1,
        e0,
        h_net,
        u_net,
        domain_penalty_weight: float = 0.0,
    ):
        ax = _concat_ax(A, X)
        h_ax = h_net(ax)
        u_ax = u_net(ax)
        h_ax = torch.clamp(h_ax, min=-self.dual_net_cfg.h_clip, max=self.dual_net_cfg.h_clip)
        min_lambda = self.dual_net_cfg.lambda_min
        if self.divergence.lambda_min_override is not None:
            min_lambda = max(min_lambda, float(self.divergence.lambda_min_override))
        lam_ax = torch.exp(h_ax).clamp(min=min_lambda)

        phi_y = self.phi(Y)
        t = (phi_y - u_ax) / lam_ax
        g_star_val, valid_mask = self.divergence.g_star_with_valid(t)
        valid_mask = valid_mask & torch.isfinite(g_star_val) & torch.isfinite(t)
        g_star_safe = torch.where(valid_mask, g_star_val, torch.zeros_like(g_star_val))
        invalid_pen = (~valid_mask).float()
        domain_pen = domain_penalty_weight * float(self.divergence.domain_penalty_scale) * (
            self.divergence.domain_violation(t).pow(2) + invalid_pen
        )

        eA = torch.where(A >= 0.5, e1, e0)
        eta = self.divergence.B_torch(eA)

        main = lam_ax * (eta + g_star_safe) + u_ax + domain_pen
        loss = main.mean()

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered (NAIVE). divergence={self.divergence.name}."
            )
        valid_count = int(valid_mask.sum().item())
        total_count = int(valid_mask.numel())
        return loss, valid_count, total_count


# -------------------------
# Cluster aggregator (fast 1D partition search, no sklearn)
# -------------------------
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
    e_eval: Optional[np.ndarray],
    e_train_true: Optional[np.ndarray],
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

    U = est_pos.predict_bound(a=1, X=X_eval, e_eval=e_eval).astype(np.float32)
    L = (-est_neg.predict_bound(a=1, X=X_eval, e_eval=e_eval)).astype(np.float32)
    return L, U, est_pos


# -------------------------
# Helper utilities
# -------------------------
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
    allowed = set(base_divs + ["kth", "tight_kth"])
    divs = [d.strip() for d in raw.split(",") if d.strip()]
    if not divs:
        divs = ["kth"]
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


V2_SCORE_LAMBDA = 10.0


def compute_v2_metrics(
    L: np.ndarray,
    U: np.ndarray,
    valid: np.ndarray,
    theta_eval: np.ndarray,
    width_stat: str,
    alpha: float,
) -> Dict[str, float]:
    L = np.asarray(L, dtype=float)
    U = np.asarray(U, dtype=float)
    theta_eval = np.asarray(theta_eval, dtype=float)
    width = U - L
    valid_mask = np.asarray(valid, dtype=bool)
    base_valid = np.isfinite(L) & np.isfinite(U) & np.isfinite(theta_eval) & (width > 0)
    valid_mask = valid_mask & base_valid
    covered = valid_mask & (theta_eval >= L) & (theta_eval <= U)

    coverage_uncond = np.mean((Ud >= theta_eval) & (Ld <= theta_eval))
    coverage_cond = np.mean((Ud >= theta_eval) & (Ld <= theta_eval))
    valid_rate = float(np.mean(valid_mask))
    width_val = _stat_reduce(width[valid_mask], width_stat)
    score = _score_penalized_width(width_val, coverage_uncond, V2_SCORE_LAMBDA, alpha)

    return {
        "coverage_uncond": coverage_uncond,
        "coverage_cond": coverage_cond,
        "width": width_val,
        "valid_rate": valid_rate,
        "score": score,
    }


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


def _make_X_eval_from_ehat(
    X_pool: np.ndarray,
    ehat: np.ndarray,
    n_eval: int,
) -> np.ndarray:
    X_pool = np.asarray(X_pool, dtype=np.float32)
    ehat = np.asarray(ehat, dtype=np.float64).reshape(-1)
    if X_pool.ndim != 2:
        raise ValueError(f"X_pool must be 2D (n,d). Got shape {X_pool.shape}.")
    if X_pool.shape[0] != ehat.shape[0]:
        raise ValueError("X_pool and ehat must have the same length.")
    if n_eval <= 0:
        raise ValueError("n_eval must be positive.")
    n_pool = int(X_pool.shape[0])
    if n_pool == 0:
        raise ValueError("X_pool is empty.")

    order = np.argsort(ehat, kind="mergesort")
    if n_eval == 1:
        positions = np.array([0.5 * float(n_pool - 1)], dtype=np.float64)
    else:
        positions = np.linspace(0.0, float(n_pool - 1), int(n_eval), dtype=np.float64)
    idx = np.clip(np.round(positions).astype(int), 0, n_pool - 1)
    chosen = order[idx]
    return X_pool[chosen]


def _resolve_eval_set(
    eval_mode: str,
    X_eval_fixed: Optional[np.ndarray],
    X_pool: np.ndarray,
    prop_cache: Dict[str, Any],
    prop_true_fn: Any,
    n_eval: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if eval_mode == "grid_ehat":
        e1_oof_raw = prop_cache.get("e1_oof", None)
        if e1_oof_raw is None:
            raise ValueError("prop_cache['e1_oof'] is required for eval_mode='grid_ehat'.")
        e1_oof = np.asarray(e1_oof_raw, dtype=np.float32)
        if e1_oof.ndim != 1:
            raise ValueError(f"prop_cache['e1_oof'] must be 1D. Got shape {e1_oof.shape}.")
        X_eval = _make_X_eval_from_ehat(X_pool, e1_oof, n_eval)
    else:
        if X_eval_fixed is None:
            raise ValueError("X_eval_fixed is required when eval_mode != 'grid_ehat'.")
        X_eval = X_eval_fixed

    e_true_eval = np.asarray(prop_true_fn(X_eval), dtype=np.float32).reshape(-1)
    if e_true_eval.shape[0] != X_eval.shape[0]:
        raise RuntimeError(
            f"propensity_true returned shape {e_true_eval.shape}, expected ({X_eval.shape[0]},)."
        )
    return X_eval, e_true_eval


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
    e_eval: Optional[np.ndarray] = None,
    e_train_true: Optional[np.ndarray] = None,
    progress_prefix: Optional[str] = None,
    timing_label: Optional[str] = None,
    timing: bool = False,
    timing_detail: bool = False,
    use_tqdm: bool = False,
):
    outputs = {}
    needs_base = any(div in {"kth", "tight_kth"} for div in div_list)
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
                base_outputs[dname] = (L_div, U_div)

    lower_base = None
    upper_base = None
    for div in div_list:
        if div in base_outputs:
            L, U = base_outputs[div]
        elif div == "kth":
            with StepTimer(
                f"{prefix}aggregate kth".strip(),
                use_tqdm=use_tqdm,
                enabled=timing and timing_detail,
            ):
                if lower_base is None:
                    lower_base = np.vstack([base_outputs[b][0] for b in base_divs])
                    upper_base = np.vstack([base_outputs[b][1] for b in base_divs])
                k_val = int(args.kval) if args.kval is not None else int(lower_base.shape[0])
                valid_up = np.isfinite(upper_base)
                valid_lo = np.isfinite(lower_base)
                agg = aggregate_endpointwise(
                    lower_base,
                    upper_base,
                    valid_up,
                    valid_lo,
                    k_up=k_val,
                    k_lo=k_val,
                )
                L = agg["lower"]
                U = agg["upper"]
        elif div == "tight_kth":
            with StepTimer(
                f"{prefix}aggregate tight_kth".strip(),
                use_tqdm=use_tqdm,
                enabled=timing and timing_detail,
            ):
                if lower_base is None:
                    lower_base = np.vstack([base_outputs[b][0] for b in base_divs])
                    upper_base = np.vstack([base_outputs[b][1] for b in base_divs])
                valid_up = np.isfinite(upper_base)
                valid_lo = np.isfinite(lower_base)
                agg = aggregate_endpointwise(
                    lower_base,
                    upper_base,
                    valid_up,
                    valid_lo,
                    k_up=1,
                    k_lo=1,
                )
                L = agg["lower"]
                U = agg["upper"]
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
        outputs[div] = (L.astype(np.float32), U.astype(np.float32))

    return outputs


# -------------------------
# Main
# -------------------------


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        description=(
            "Slow nuisance experiment: debiased vs naive error rates. "
            "Superset of plot_n_coverage_v2.py metrics/plots."
        )
    )
    parser.add_argument(
        "--n_list",
        type=str,
        nargs="+",
        required=True,
        help="Sample sizes (space or comma-separated), e.g. '500 1000' or '500,1000'.",
    )
    parser.add_argument("--m", type=int, default=3, help="Number of replicates per n.")
    parser.add_argument("--d", type=int, default=5, help="Feature dimension.")
    parser.add_argument(
        "--kval",
        type=int,
        default=4,
        help="k for kth/tight_kth aggregation (default: number of base divergences).",
    )
    parser.add_argument(
        "--divergence",
        type=str,
        default="kth",
        help="Comma-separated divergences from {KL,TV,Hellinger,Chi2,JS,kth,tight_kth}.",
    )
    parser.add_argument(
        "--structural_type",
        type=str,
        default="nonlinear",
        choices=["linear", "nonlinear", "simpson", "cyclic", "cyclic2", "probit_sine"],
        help="Data-generating process type.",
    )
    parser.add_argument("--base_seed", type=int, default=12345, help="Base seed for training replicates.")
    parser.add_argument("--variantB", action="store_true", default=True, help="Use Variant B nested datasets across n.")
    parser.add_argument("--shuffle_master", action="store_true", default=False, help="Shuffle master dataset per replicate.")
    parser.add_argument(
        "--no-shuffle_master",
        dest="shuffle_master",
        action="store_false",
        help="Disable master dataset shuffle.",
    )
    parser.add_argument("--debug_variantB", action="store_true", default=False, help="Debug Variant B nesting checks.")
    parser.add_argument("--outdir", type=str, default="experiments", help="Output directory.")
    parser.add_argument("--unique_save", action="store_true", default=True, help="Add unique timestamp suffix to outputs.")

    # Evaluation set X_eval
    parser.add_argument("--n_eval", type=int, default=1000, help="Number of fixed evaluation points.")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="sample", # sample?
        choices=["sample", "grid_x0", "grid_ehat"],
        help="How to construct X_eval: sample from DGP, grid over X0, or grid by estimated propensity.",
    )
    parser.add_argument("--eval_seed", type=int, default=190602, help="Seed for constructing X_eval (sample mode).")
    parser.add_argument("--x0_min", type=float, default=-3.14, help="Min X0 for grid_x0.")
    parser.add_argument("--x0_max", type=float, default=3.14, help="Max X0 for grid_x0.")
    parser.add_argument("--x_eval_fill", type=float, default=0.0, help="Fill value for non-X0 coordinates in grid_x0.")
    parser.add_argument("--x_range", type=float, default=2.0, help="x_range for some DGPs (if supported).")
    parser.add_argument(
        "--noise_dist",
        type=str,
        default='normal',
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
        default = True,
        help="Run all combinations of --stat/--stat_over_reps in {mean,median}.",
    )
    parser.add_argument("--ci_alpha", type=float, default=0.05, help="Two-sided quantile band level across replicates.")

    # Score
    parser.add_argument("--score_lambda", type=float, default=10.0, help="Penalty lambda for score.")
    parser.add_argument("--score_alpha", type=float, default=0.05, help="Target shortfall alpha (target=1-alpha).")

    # Estimator controls
    parser.add_argument("--n_folds", type=int, default=2, help="CV folds.")
    parser.add_argument("--eps_propensity", type=float, default=1e-3, help="Propensity clipping.")
    parser.add_argument("--num_epochs", type=int, default=256, help="Dual net epochs.")
    parser.add_argument("--xgb_n_jobs", type=int, default=-1, help="n_jobs for xgboost (-1 all cores).")
    parser.add_argument("--torch_threads", type=int, default=0, help="Torch intra-op threads (0 uses all cores).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for dual nets.")

    # Oracle settings
    parser.add_argument("--n_oracle", type=int, default=20000, help="Oracle sample size for near-truth bounds.")
    parser.add_argument("--oracle_seed", type=int, default=-1, help="Seed for oracle data (negative uses base_seed).")
    parser.add_argument("--oracle_propensity_n_estimators", type=int, default=0, help="Override oracle propensity n_estimators.")
    parser.add_argument("--oracle_propensity_max_depth", type=int, default=0, help="Override oracle propensity max_depth.")
    parser.add_argument("--oracle_num_epochs", type=int, default=200, help="Oracle num_epochs (default 50).")
    parser.add_argument("--oracle_batch_size", type=int, default=256, help="Oracle batch_size (default 1024).")

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
        V2_SCORE_LAMBDA = float(args.score_lambda)
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

    X_eval: Optional[np.ndarray]
    if args.eval_mode == "grid_ehat":
        X_eval = None
        n_eval = int(args.n_eval)
    else:
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

    # True propensity on X_eval from the DGP.
    warn_state: Dict[str, bool] = {}
    data_eval = _call_generate_data_compat(
        n=n_eval,
        d=args.d,
        seed=args.eval_seed,
        structural_type=args.structural_type,
        x_range=args.x_range,
        noise_dist=args.noise_dist,
        _warn_state=warn_state,
    )
    prop_true_fn = data_eval.get("propensity_true")
    if prop_true_fn is None:
        raise RuntimeError("generate_data(...) did not return propensity_true.")
    e_true_eval: Optional[np.ndarray] = None
    if X_eval is not None:
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

    oracle_seed = int(args.oracle_seed) if args.oracle_seed >= 0 else int(args.base_seed)
    U_oracle: Dict[str, np.ndarray] = {
        div: np.full((n_eval,), np.nan, dtype=np.float32) for div in div_list
    }
    e_oracle: Optional[np.ndarray] = None
    if args.eval_mode == "grid_ehat":
        print("[Oracle] Skipped: eval_mode=grid_ehat builds X_eval per replicate.")
    else:
        with StepTimer("oracle setup", use_tqdm=False, enabled=timing_enabled):
            # Oracle bounds and propensity (near-truth)
            if X_eval is None or e_true_eval is None:
                raise RuntimeError("X_eval/e_true_eval required for oracle computation.")
            print("[Oracle] Oracle-A only (Oracle-B disabled/removed).")
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

            prop_true_oracle_fn = data_oracle.get("propensity_true")
            if prop_true_oracle_fn is None:
                raise RuntimeError("generate_data(...) did not return propensity_true for oracle data.")
            e_true_oracle_train = np.asarray(prop_true_oracle_fn(X_oracle), dtype=np.float32).reshape(-1)
            if e_true_oracle_train.shape[0] != X_oracle.shape[0]:
                raise RuntimeError(
                    f"propensity_true returned shape {e_true_oracle_train.shape}, expected ({X_oracle.shape[0]},)."
                )
            print("[ORACLE] using true propensity for training and eval; no propensity model trained")
            prop_cache_oracle = None
            assert prop_cache_oracle is None

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
                prop_cache=None,
                e_train_true=e_true_oracle_train,
                e_eval=e_true_eval,
                progress_prefix="[oracle]",
                timing_label="oracle",
                timing=timing_enabled,
                timing_detail=timing_detail,
                use_tqdm=False,
            )
            U_oracle = {div: oracle_bounds[div][1] for div in div_list}
            e_oracle = e_true_eval
            print("Oracle estimator completed.")

    X_eval_reference: Optional[np.ndarray] = X_eval
    replicate_rows: List[Dict[str, Any]] = []
    seeds_used: List[int] = []

    total_reps = len(n_list) * args.m
    print(f"Running {total_reps} replicate fits...")
    if args.variantB:
        n_list = sorted(n_list)
        n_max = int(max(n_list))
        for j in tqdm(range(args.m), desc="replicates", leave=False):
            seed_mc = int(args.base_seed + j)
            seed_perm = int(seed_mc + 777)
            print(f"[VariantB] rep={j} seed_mc={seed_mc} n_max={n_max}")
            print(f"[VariantB] rep={j} seed_perm={seed_perm} shuffle={args.shuffle_master}")
            data_all = _call_generate_data_compat(
                n=n_max,
                d=args.d,
                seed=seed_mc,
                structural_type=args.structural_type,
                x_range=args.x_range,
                noise_dist=args.noise_dist,
                _warn_state=warn_state,
            )
            X_all = np.asarray(data_all["X"], dtype=np.float32)
            A_all = np.asarray(data_all["A"], dtype=np.float32)
            Y_all = np.asarray(data_all["Y"], dtype=np.float32)
            gt_fn = data_all["GroundTruth"]
            if args.shuffle_master:
                rng = np.random.default_rng(seed_perm)
                perm = rng.permutation(n_max)
                X_all = X_all[perm]
                A_all = A_all[perm]
                Y_all = Y_all[perm]
            for idx_n, n in enumerate(n_list):
                print(f"[VariantB] rep={j} using prefix n={n}")
                hash_n = stable_hash_n(n) % 100000
                seed_propensity_fit = int(seed_mc + hash_n)
                seed_propensity_noise = int(seed_mc + 4242 + hash_n) if args.propensity_noise else float("nan")
                seeds_used.append(seed_propensity_fit)
                with StepTimer(f"replicate n={n} rep={j}", use_tqdm=True, enabled=timing_enabled):
                    with StepTimer("data generation", use_tqdm=True, enabled=timing_enabled):
                        X = X_all[:n]
                        A = A_all[:n]
                        Y = Y_all[:n]
                        if args.debug_variantB:
                            n_chk = min(50, int(n))
                            chk_x = float(np.sum(X_all[:n_chk]))
                            chk_a = float(np.sum(A_all[:n_chk]))
                            chk_y = float(np.sum(Y_all[:n_chk]))
                            print(
                                f"[VariantB-CHK] rep={j} n={int(n)} "
                                f"chkX50={chk_x:.6f} chkA50={chk_a:.6f} chkY50={chk_y:.6f}"
                            )

                    with StepTimer("nuisance fits", use_tqdm=True, enabled=timing_enabled):
                        prop_cache = prefit_propensity_cache(
                            X=X,
                            A=A,
                            propensity_model=propensity_model,
                            propensity_config=fit_config["propensity_config"],
                            n_folds=fit_config["n_folds"],
                            seed=seed_propensity_fit,
                            eps_propensity=fit_config["eps_propensity"],
                        )
                        prop_cache = _apply_propensity_noise(
                            prop_cache,
                            n=n,
                            eps=fit_config["eps_propensity"],
                            noise_beta=args.propensity_noise_beta,
                            enabled=args.propensity_noise,
                            seed=seed_propensity_noise,
                        )
                        X_eval_use, e_true_eval_use = _resolve_eval_set(
                            args.eval_mode,
                            X_eval,
                            X,
                            prop_cache,
                            prop_true_fn,
                            n_eval,
                        )
                        if X_eval_reference is None:
                            X_eval_reference = np.asarray(X_eval_use, dtype=np.float32)
                        theta_eval = np.asarray(gt_fn(1, X_eval_use), dtype=np.float32).reshape(-1)
                        e_oracle_use = e_oracle if e_oracle is not None else e_true_eval_use

                        e_hat = _predict_propensity_mean(prop_cache["models"], X_eval_use, fit_config["eps_propensity"])
                        prop_rmse = _rmse(e_hat, e_oracle_use)
                        prop_rmse_true = _rmse(e_hat, e_true_eval_use)

                    with StepTimer("debiased bounds", use_tqdm=True, enabled=timing_enabled):
                        deb_bounds = _fit_bounds_for_divs(
                            DebiasedCausalBoundEstimator,
                            div_list,
                            base_divs,
                            X,
                            A,
                            Y,
                            X_eval_use,
                            dual_net_config,
                            fit_config,
                            seed=seed_propensity_fit,
                            propensity_model=propensity_model,
                            m_model=m_model,
                            prop_cache=prop_cache,
                            e_eval=None,
                            e_train_true=None,
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
                            X_eval_use,
                            dual_net_config,
                            fit_config,
                            seed=seed_propensity_fit,
                            propensity_model=propensity_model,
                            m_model=m_model,
                            prop_cache=prop_cache,
                            e_eval=None,
                            e_train_true=None,
                            timing_label="naive",
                            timing=timing_enabled,
                            timing_detail=timing_detail,
                            use_tqdm=True,
                        )

                    with StepTimer("metrics", use_tqdm=True, enabled=timing_enabled):
                        for div in div_list:
                            Ld, Ud = deb_bounds[div]
                            Ln, Un = nai_bounds[div]

                            width_d_raw = Ud - Ld
                            width_n_raw = Un - Ln
                            valid_d = np.isfinite(Ld) & np.isfinite(Ud) & (width_d_raw > 0)
                            valid_n = np.isfinite(Ln) & np.isfinite(Un) & (width_n_raw > 0)

                            v2_d = compute_v2_metrics(Ld, Ud, valid_d, theta_eval, args.width_stat, args.score_alpha)
                            v2_n = compute_v2_metrics(Ln, Un, valid_n, theta_eval, args.width_stat, args.score_alpha)

                            cov_d = float(v2_d["coverage_uncond"])
                            cov_n = float(v2_n["coverage_uncond"])

                            width_d_mean = float(np.nanmean(width_d_raw[valid_d])) if np.any(valid_d) else float("nan")
                            width_d_median = float(np.nanmedian(width_d_raw[valid_d])) if np.any(valid_d) else float("nan")
                            width_n_mean = float(np.nanmean(width_n_raw[valid_n])) if np.any(valid_n) else float("nan")
                            width_n_median = float(np.nanmedian(width_n_raw[valid_n])) if np.any(valid_n) else float("nan")
                            width_d = float(v2_d["width"])
                            width_n = float(v2_n["width"])

                            err_up_d = _rmse(Ud, U_oracle[div])
                            err_up_n = _rmse(Un, U_oracle[div])

                            score_d_mean = _score_penalized_width(width_d_mean, cov_d, args.score_lambda, args.score_alpha)
                            score_d_median = _score_penalized_width(width_d_median, cov_d, args.score_lambda, args.score_alpha)
                            score_n_mean = _score_penalized_width(width_n_mean, cov_n, args.score_lambda, args.score_alpha)
                            score_n_median = _score_penalized_width(width_n_median, cov_n, args.score_lambda, args.score_alpha)
                            score_d = float(v2_d["score"])
                            score_n = float(v2_n["score"])

                            replicate_rows.append(
                                {
                                    "divergence": div,
                                    "n": int(n),
                                    "rep": int(j),
                                    "seed": int(seed_propensity_fit),
                                    "seed_mc": seed_mc,
                                    "seed_perm": seed_perm,
                                    "seed_propensity_fit": seed_propensity_fit,
                                    "seed_propensity_noise": seed_propensity_noise,
                                    "propensity_rmse": prop_rmse,
                                    "propensity_rmse_true": prop_rmse_true,
                                    "err_up_debiased": err_up_d,
                                    "err_up_naive": err_up_n,
                                    "coverage_debiased": cov_d,
                                    "coverage_naive": cov_n,
                                    "coverage_uncond_debiased": float(v2_d["coverage_uncond"]),
                                    "coverage_uncond_naive": float(v2_n["coverage_uncond"]),
                                    "coverage_cond_debiased": float(v2_d["coverage_cond"]),
                                    "coverage_cond_naive": float(v2_n["coverage_cond"]),
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
                                    "valid_rate_debiased": float(v2_d["valid_rate"]),
                                    "valid_rate_naive": float(v2_n["valid_rate"]),
                                }
                            )
    else:
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
            seed_mc = int(seed_tr)
            seed_perm = float("nan")
            seed_propensity_fit = int(seed_tr)
            seed_propensity_noise = int(seed_tr) if args.propensity_noise else float("nan")
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

                    X_eval_use, e_true_eval_use = _resolve_eval_set(
                        args.eval_mode,
                        X_eval,
                        X,
                        prop_cache,
                        prop_true_fn,
                        n_eval,
                    )
                    if X_eval_reference is None:
                        X_eval_reference = np.asarray(X_eval_use, dtype=np.float32)
                    theta_eval = np.asarray(data["GroundTruth"](1, X_eval_use), dtype=np.float32).reshape(-1)
                    e_oracle_use = e_oracle if e_oracle is not None else e_true_eval_use

                    e_hat = _predict_propensity_mean(prop_cache["models"], X_eval_use, fit_config["eps_propensity"])
                    prop_rmse = _rmse(e_hat, e_oracle_use)
                    prop_rmse_true = _rmse(e_hat, e_true_eval_use)

                with StepTimer("debiased bounds", use_tqdm=True, enabled=timing_enabled):
                    deb_bounds = _fit_bounds_for_divs(
                        DebiasedCausalBoundEstimator,
                        div_list,
                        base_divs,
                        X,
                        A,
                        Y,
                        X_eval_use,
                        dual_net_config,
                        fit_config,
                        seed=seed_tr,
                        propensity_model=propensity_model,
                        m_model=m_model,
                        prop_cache=prop_cache,
                        e_eval=None,
                        e_train_true=None,
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
                        X_eval_use,
                        dual_net_config,
                        fit_config,
                        seed=seed_tr,
                        propensity_model=propensity_model,
                        m_model=m_model,
                        prop_cache=prop_cache,
                        e_eval=None,
                        e_train_true=None,
                        timing_label="naive",
                        timing=timing_enabled,
                        timing_detail=timing_detail,
                        use_tqdm=True,
                    )

                with StepTimer("metrics", use_tqdm=True, enabled=timing_enabled):
                    for div in div_list:
                        Ld, Ud = deb_bounds[div]
                        Ln, Un = nai_bounds[div]

                        width_d_raw = Ud - Ld
                        width_n_raw = Un - Ln
                        valid_d = np.isfinite(Ld) & np.isfinite(Ud) & (width_d_raw > 0)
                        valid_n = np.isfinite(Ln) & np.isfinite(Un) & (width_n_raw > 0)

                        v2_d = compute_v2_metrics(Ld, Ud, valid_d, theta_eval, args.width_stat, args.score_alpha)
                        v2_n = compute_v2_metrics(Ln, Un, valid_n, theta_eval, args.width_stat, args.score_alpha)

                        cov_d = float(v2_d["coverage_uncond"])
                        cov_n = float(v2_n["coverage_uncond"])

                        width_d_mean = float(np.nanmean(width_d_raw[valid_d])) if np.any(valid_d) else float("nan")
                        width_d_median = float(np.nanmedian(width_d_raw[valid_d])) if np.any(valid_d) else float("nan")
                        width_n_mean = float(np.nanmean(width_n_raw[valid_n])) if np.any(valid_n) else float("nan")
                        width_n_median = float(np.nanmedian(width_n_raw[valid_n])) if np.any(valid_n) else float("nan")
                        width_d = float(v2_d["width"])
                        width_n = float(v2_n["width"])

                        err_up_d = _rmse(Ud, U_oracle[div])
                        err_up_n = _rmse(Un, U_oracle[div])

                        score_d_mean = _score_penalized_width(width_d_mean, cov_d, args.score_lambda, args.score_alpha)
                        score_d_median = _score_penalized_width(width_d_median, cov_d, args.score_lambda, args.score_alpha)
                        score_n_mean = _score_penalized_width(width_n_mean, cov_n, args.score_lambda, args.score_alpha)
                        score_n_median = _score_penalized_width(width_n_median, cov_n, args.score_lambda, args.score_alpha)
                        score_d = float(v2_d["score"])
                        score_n = float(v2_n["score"])

                        replicate_rows.append(
                            {
                                "divergence": div,
                                "n": int(n),
                                "rep": int(j),
                                "seed": int(seed_tr),
                                "seed_mc": seed_mc,
                                "seed_perm": seed_perm,
                                "seed_propensity_fit": seed_propensity_fit,
                                "seed_propensity_noise": seed_propensity_noise,
                                "propensity_rmse": prop_rmse,
                                "propensity_rmse_true": prop_rmse_true,
                                "err_up_debiased": err_up_d,
                                "err_up_naive": err_up_n,
                                "coverage_debiased": cov_d,
                                "coverage_naive": cov_n,
                                "coverage_uncond_debiased": float(v2_d["coverage_uncond"]),
                                "coverage_uncond_naive": float(v2_n["coverage_uncond"]),
                                "coverage_cond_debiased": float(v2_d["coverage_cond"]),
                                "coverage_cond_naive": float(v2_n["coverage_cond"]),
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
                                "valid_rate_debiased": float(v2_d["valid_rate"]),
                                "valid_rate_naive": float(v2_n["valid_rate"]),
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
                prop_true = np.array([r["propensity_rmse_true"] for r in rows], dtype=float)
                err_d = np.array([r["err_up_debiased"] for r in rows], dtype=float)
                err_n = np.array([r["err_up_naive"] for r in rows], dtype=float)
                cov_d = np.array([r["coverage_debiased"] for r in rows], dtype=float)
                cov_n = np.array([r["coverage_naive"] for r in rows], dtype=float)
                covu_d = np.array([r["coverage_uncond_debiased"] for r in rows], dtype=float)
                covu_n = np.array([r["coverage_uncond_naive"] for r in rows], dtype=float)
                covc_d = np.array([r["coverage_cond_debiased"] for r in rows], dtype=float)
                covc_n = np.array([r["coverage_cond_naive"] for r in rows], dtype=float)
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
                        "propensity_rmse_true_center": _stat_reduce(prop_true, stat_over_reps),
                        "propensity_rmse_true_ci_low": _nan_quantile(prop_true, args.ci_alpha / 2),
                        "propensity_rmse_true_ci_high": _nan_quantile(prop_true, 1 - args.ci_alpha / 2),
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
                        "coverage_uncond_debiased_center": _stat_reduce(covu_d, stat_over_reps),
                        "coverage_uncond_debiased_ci_low": _nan_quantile(covu_d, args.ci_alpha / 2),
                        "coverage_uncond_debiased_ci_high": _nan_quantile(covu_d, 1 - args.ci_alpha / 2),
                        "coverage_uncond_naive_center": _stat_reduce(covu_n, stat_over_reps),
                        "coverage_uncond_naive_ci_low": _nan_quantile(covu_n, args.ci_alpha / 2),
                        "coverage_uncond_naive_ci_high": _nan_quantile(covu_n, 1 - args.ci_alpha / 2),
                        "coverage_cond_debiased_center": _stat_reduce(covc_d, stat_over_reps),
                        "coverage_cond_debiased_ci_low": _nan_quantile(covc_d, args.ci_alpha / 2),
                        "coverage_cond_debiased_ci_high": _nan_quantile(covc_d, 1 - args.ci_alpha / 2),
                        "coverage_cond_naive_center": _stat_reduce(covc_n, stat_over_reps),
                        "coverage_cond_naive_ci_low": _nan_quantile(covc_n, args.ci_alpha / 2),
                        "coverage_cond_naive_ci_high": _nan_quantile(covc_n, 1 - args.ci_alpha / 2),
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
            ys_ref = [row["propensity_rmse_center"] for row in sub]
            lo_ref = [row["propensity_rmse_ci_low"] for row in sub]
            hi_ref = [row["propensity_rmse_ci_high"] for row in sub]
            yerr_ref = [np.subtract(ys_ref, lo_ref), np.subtract(hi_ref, ys_ref)]
            plt.errorbar(
                xs,
                ys_ref,
                yerr=yerr_ref,
                marker="o",
                linestyle="-",
                capsize=3,
                label=f"{div} (vs TRUE propensity, ref)",
            )

            ys_true = [row["propensity_rmse_true_center"] for row in sub]
            lo_true = [row["propensity_rmse_true_ci_low"] for row in sub]
            hi_true = [row["propensity_rmse_true_ci_high"] for row in sub]
            yerr_true = [np.subtract(ys_true, lo_true), np.subtract(hi_true, ys_true)]
            plt.errorbar(
                xs,
                ys_true,
                yerr=yerr_true,
                marker="x",
                linestyle="--",
                capsize=3,
                label=f"{div} (vs TRUE propensity, direct)",
            )

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

    def _plot_metric_single(rows: List[Dict[str, Any]], key_d: str, ylabel: str, title: str, fname: str) -> str:
        plt.figure(figsize=(7.2, 4.2))
        for div in div_list:
            sub = [row for row in rows if row["divergence"] == div]
            xs = [row["n"] for row in sub]
            yd = [row[f"{key_d}_center"] for row in sub]
            lo_d = [row[f"{key_d}_ci_low"] for row in sub]
            hi_d = [row[f"{key_d}_ci_high"] for row in sub]
            plt.errorbar(xs, yd, yerr=[np.subtract(yd, lo_d), np.subtract(hi_d, yd)], marker="o", linestyle="-", capsize=3, label=f"{div} debiased")
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
                "divergence,n,rep,seed,seed_mc,seed_perm,seed_propensity_fit,seed_propensity_noise,"
                "propensity_rmse,propensity_rmse_true,err_up_debiased,err_up_naive,"
                "coverage_debiased,coverage_naive,coverage_uncond_debiased,coverage_uncond_naive,"
                "coverage_cond_debiased,coverage_cond_naive,"
                "width_debiased,width_debiased_mean,width_debiased_median,width_naive,width_naive_mean,width_naive_median,"
                "score_debiased,score_debiased_mean,score_debiased_median,score_naive,score_naive_mean,score_naive_median,"
                "valid_rate_debiased,valid_rate_naive"
            )
            with open(rep_path, "w") as f:
                f.write(header + "\n")
                for row in replicate_rows:
                    f.write(
                        f"{row['divergence']},{row['n']},{row['rep']},{row['seed']},"
                        f"{row['seed_mc']},{row['seed_perm']},{row['seed_propensity_fit']},{row['seed_propensity_noise']},"
                        f"{row['propensity_rmse']},{row['propensity_rmse_true']},"
                        f"{row['err_up_debiased']},{row['err_up_naive']},"
                        f"{row['coverage_debiased']},{row['coverage_naive']},"
                        f"{row['coverage_uncond_debiased']},{row['coverage_uncond_naive']},"
                        f"{row['coverage_cond_debiased']},{row['coverage_cond_naive']},"
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
                "propensity_rmse_true_center,propensity_rmse_true_ci_low,propensity_rmse_true_ci_high,"
                "err_up_debiased_center,err_up_debiased_ci_low,err_up_debiased_ci_high,"
                "err_up_naive_center,err_up_naive_ci_low,err_up_naive_ci_high,"
                "coverage_debiased_center,coverage_debiased_ci_low,coverage_debiased_ci_high,"
                "coverage_naive_center,coverage_naive_ci_low,coverage_naive_ci_high,"
                "coverage_uncond_debiased_center,coverage_uncond_debiased_ci_low,coverage_uncond_debiased_ci_high,"
                "coverage_uncond_naive_center,coverage_uncond_naive_ci_low,coverage_uncond_naive_ci_high,"
                "coverage_cond_debiased_center,coverage_cond_debiased_ci_low,coverage_cond_debiased_ci_high,"
                "coverage_cond_naive_center,coverage_cond_naive_ci_low,coverage_cond_naive_ci_high,"
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
                        f"{row['propensity_rmse_true_center']},{row['propensity_rmse_true_ci_low']},{row['propensity_rmse_true_ci_high']},"
                        f"{row['err_up_debiased_center']},{row['err_up_debiased_ci_low']},{row['err_up_debiased_ci_high']},"
                        f"{row['err_up_naive_center']},{row['err_up_naive_ci_low']},{row['err_up_naive_ci_high']},"
                        f"{row['coverage_debiased_center']},{row['coverage_debiased_ci_low']},{row['coverage_debiased_ci_high']},"
                        f"{row['coverage_naive_center']},{row['coverage_naive_ci_low']},{row['coverage_naive_ci_high']},"
                        f"{row['coverage_uncond_debiased_center']},{row['coverage_uncond_debiased_ci_low']},{row['coverage_uncond_debiased_ci_high']},"
                        f"{row['coverage_uncond_naive_center']},{row['coverage_uncond_naive_ci_low']},{row['coverage_uncond_naive_ci_high']},"
                        f"{row['coverage_cond_debiased_center']},{row['coverage_cond_debiased_ci_low']},{row['coverage_cond_debiased_ci_high']},"
                        f"{row['coverage_cond_naive_center']},{row['coverage_cond_naive_ci_low']},{row['coverage_cond_naive_ci_high']},"
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

        covu_base = "plot_n_coverage_uncond_debiased_vs_naive"
        covc_base = "plot_n_coverage_cond_debiased_vs_naive"
        valid_base = "plot_n_valid_rate_debiased_vs_naive"
        width_v2_base = "plot_n_width_debiased_vs_naive"
        score_v2_base = "plot_n_score_debiased_vs_naive"
        covu_only_base = "plot_n_coverage_uncond_debiased_only"
        width_only_base = "plot_n_width_debiased_only"
        if suffix:
            covu_base = f"{covu_base}_{suffix}"
            covc_base = f"{covc_base}_{suffix}"
            valid_base = f"{valid_base}_{suffix}"
            width_v2_base = f"{width_v2_base}_{suffix}"
            score_v2_base = f"{score_v2_base}_{suffix}"
            covu_only_base = f"{covu_only_base}_{suffix}"
            width_only_base = f"{width_only_base}_{suffix}"

        covu_fig_v2 = _plot_metric(
            summary_rows,
            "coverage_uncond_debiased",
            "coverage_uncond_naive",
            ylabel="Coverage (unconditional)",
            title=f"Coverage (unconditional) vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=covu_base,
        )
        covc_fig_v2 = _plot_metric(
            summary_rows,
            "coverage_cond_debiased",
            "coverage_cond_naive",
            ylabel="Coverage (conditional on valid)",
            title=f"Coverage (conditional) vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=covc_base,
        )
        valid_fig_v2 = _plot_metric(
            summary_rows,
            "valid_rate_debiased",
            "valid_rate_naive",
            ylabel="Valid rate",
            title=f"Valid rate vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=valid_base,
        )
        width_fig_v2 = _plot_metric(
            summary_rows,
            "width_debiased",
            "width_naive",
            ylabel="Width",
            title=f"Width vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=width_v2_base,
        )
        score_fig_v2 = _plot_metric(
            summary_rows,
            "score_debiased",
            "score_naive",
            ylabel="Score (penalized width)",
            title=f"Score vs n (struct={args.structural_type}, eval={args.eval_mode})",
            fname=score_v2_base,
        )
        covu_only_fig = _plot_metric_single(
            summary_rows,
            "coverage_uncond_debiased",
            ylabel="Coverage (unconditional)",
            title=f"Coverage (unconditional) vs n (debiased only; struct={args.structural_type}, eval={args.eval_mode})",
            fname=covu_only_base,
        )
        width_only_fig = _plot_metric_single(
            summary_rows,
            "width_debiased",
            ylabel="Width",
            title=f"Width vs n (debiased only; struct={args.structural_type}, eval={args.eval_mode})",
            fname=width_only_base,
        )

        run_files = {
            "replicate_csv": rep_path,
            "summary_csv": summary_path,
            "nuisance_png": nuisance_fig,
            "target_png": target_fig,
            "width_png": width_fig,
            "coverage_png": cov_fig,
            "score_png": score_fig,
            "coverage_uncond_png": covu_fig_v2,
            "coverage_cond_png": covc_fig_v2,
            "valid_rate_png": valid_fig_v2,
            "width_v2_png": width_fig_v2,
            "score_v2_png": score_fig_v2,
            "coverage_uncond_debiased_only_png": covu_only_fig,
            "width_debiased_only_png": width_only_fig,
        }
        runs.append({"width_stat": stat_within, "stat_over_reps": stat_over_reps, "files": run_files})

        if stat_within == args.width_stat and stat_over_reps == args.stat_over_reps:
            default_files = run_files
        step_timer.__exit__(None, None, None)

    if not default_files and runs:
        default_files = runs[0]["files"]

    with StepTimer("save artifacts", use_tqdm=False, enabled=timing_enabled):
        x_eval_note = None
        if args.eval_mode == "grid_ehat":
            x_eval_note = "grid_ehat uses per-replicate eval sets; X_eval stores the first grid."
        artifacts = {
            "args": vars(args),
            "fit_config": fit_config,
            "dual_net_config": dual_net_config,
            "divergences": div_list,
            "seeds": seeds_used,
            "X_eval": X_eval_reference,
            "X_eval_note": x_eval_note,
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
