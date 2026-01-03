"""
Simple end-to-end example.

Generates observational data and ground-truth outcomes, fits bounds for all
divergences, and reports per-sample bounds plus coverage/width on valid intervals.
"""
from __future__ import annotations

import os
import sys
import warnings
import time
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parent
except NameError:
    _ROOT = Path.cwd()
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fbound.utils.utils import apply_macos_thread_safety_knobs

apply_macos_thread_safety_knobs(enable=False)

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from fbound.estimators.causal_bound import (
    DebiasedCausalBoundEstimator,
    _apply_interval_validity,
    aggregate_endpointwise,
    prefit_propensity_cache,
)
from fbound.utils.data_generating import generate_data

warnings.filterwarnings("ignore")


def _log(msg: str) -> None:
    if tqdm is not None and hasattr(tqdm, "write"):
        tqdm.write(msg)
    else:
        print(msg, flush=True)


class StepTimer:
    def __init__(self, name: str) -> None:
        self.name = name
        self._start_wall = 0.0
        self._start_cpu = 0.0

    def __enter__(self) -> "StepTimer":
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        _log(f"[PROGRESS] START {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        wall = time.perf_counter() - self._start_wall
        cpu = time.process_time() - self._start_cpu
        _log(f"[PROGRESS] END {self.name} | wall={wall:.3f}s cpu={cpu:.3f}s")
        return False


def phi_identity(y: torch.Tensor) -> torch.Tensor:
    return y


def phi_neg(y: torch.Tensor) -> torch.Tensor:
    return -y


def _fit_estimator(
    est: DebiasedCausalBoundEstimator,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    propensity_cache: dict | None,
) -> DebiasedCausalBoundEstimator:
    if propensity_cache is None:
        return est.fit(X, A, Y)
    try:
        return est.fit(X, A, Y, propensity_cache=propensity_cache)
    except TypeError:
        return est.fit(X, A, Y)


def fit_bounds_one(
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
) -> dict:
    est_upper = DebiasedCausalBoundEstimator(
        divergence=div_name,
        phi=phi_identity,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
    )
    est_upper = _fit_estimator(est_upper, X, A, Y, propensity_cache)
    upper_raw = est_upper.predict_bound(a=1, X=X).astype(np.float32)

    est_lower = DebiasedCausalBoundEstimator(
        divergence=div_name,
        phi=phi_neg,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_net_config,
        fit_config=fit_config,
        seed=seed,
    )
    est_lower = _fit_estimator(est_lower, X, A, Y, propensity_cache)
    upper_neg = est_lower.predict_bound(a=1, X=X).astype(np.float32)
    lower_raw = (-upper_neg).astype(np.float32)

    valid_up = np.isfinite(upper_raw)
    valid_lo = np.isfinite(lower_raw)
    lower, upper, valid_interval, inverted = _apply_interval_validity(
        lower_raw=lower_raw,
        upper_raw=upper_raw,
        valid_up=valid_up,
        valid_lo=valid_lo,
    )

    return {
        "lower_raw": lower_raw,
        "upper_raw": upper_raw,
        "lower": lower,
        "upper": upper,
        "valid_up": valid_up,
        "valid_lo": valid_lo,
        "valid_interval": valid_interval,
        "inverted": inverted,
    }


def summarize_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    valid_interval: np.ndarray,
) -> dict:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid_interval = np.asarray(valid_interval, dtype=bool)

    finite = np.isfinite(lower) & np.isfinite(upper)
    mask = valid_interval & finite
    n_valid = int(np.count_nonzero(mask))
    if n_valid == 0:
        return {
            "coverage_rate": float("nan"),
            "mean_width": float("nan"),
            "valid_interval_frac": 0.0,
            "n_valid": 0,
        }

    coverage = float(np.mean((truth[mask] >= lower[mask]) & (truth[mask] <= upper[mask])))
    mean_width = float(np.mean(upper[mask] - lower[mask]))
    valid_interval_frac = float(np.mean(mask))
    return {
        "coverage_rate": coverage,
        "mean_width": mean_width,
        "valid_interval_frac": valid_interval_frac,
        "n_valid": n_valid,
    }


def main() -> None:
    with StepTimer("configure experiment"):
        seed = 190702
        n = 1000
        d = 5
        structural_type = "cyclic2"
        k = 3

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
            "num_epochs": 200,
            "batch_size": None,
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "max_grad_norm": None,
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

    with StepTimer("data generation"):
        data = generate_data(
            n=n,
            d=d,
            seed=seed,
            structural_type=structural_type,
            noise_dist="normal",
        )
        X = np.asarray(data["X"], dtype=np.float32)
        A = np.asarray(data["A"], dtype=np.float32)
        Y = np.asarray(data["Y"], dtype=np.float32)
        truth_do1 = np.asarray(data["GroundTruth"](1, X), dtype=np.float32).reshape(-1)

    with StepTimer("prefit propensity cache"):
        propensity_cache = prefit_propensity_cache(
            X=X,
            A=A,
            propensity_model=propensity_model,
            propensity_config=fit_config["propensity_config"],
            n_folds=fit_config["n_folds"],
            seed=seed,
            eps_propensity=fit_config["eps_propensity"],
        )

    results: dict[str, object] = {
        "i": np.arange(X.shape[0], dtype=int),
        "truth_do1": truth_do1,
    }

    lower_raw_stack = []
    upper_raw_stack = []
    valid_up_stack = []
    valid_lo_stack = []

    div_iter = tqdm(base_divs, desc="fit divergences") if tqdm is not None else base_divs
    for div_name in div_iter:
        with StepTimer(f"fit bounds ({div_name})"):
            stats = fit_bounds_one(
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
            )

        results[f"lower_{div_name}"] = stats["lower"]
        results[f"upper_{div_name}"] = stats["upper"]
        results[f"valid_interval_{div_name}"] = stats["valid_interval"]

        lower_raw_stack.append(stats["lower_raw"])
        upper_raw_stack.append(stats["upper_raw"])
        valid_up_stack.append(stats["valid_up"])
        valid_lo_stack.append(stats["valid_lo"])

    lower_mat = np.vstack([np.asarray(x, dtype=np.float32) for x in lower_raw_stack])
    upper_mat = np.vstack([np.asarray(x, dtype=np.float32) for x in upper_raw_stack])
    valid_up_mat = np.vstack([np.asarray(x, dtype=bool) for x in valid_up_stack])
    valid_lo_mat = np.vstack([np.asarray(x, dtype=bool) for x in valid_lo_stack])

    with StepTimer("aggregate kth bounds"):
        agg_kth = aggregate_endpointwise(
            lower_mat=lower_mat,
            upper_mat=upper_mat,
            valid_up=valid_up_mat,
            valid_lo=valid_lo_mat,
            k_up=k,
            k_lo=k,
        )
    lower_kth = agg_kth["lower"].astype(np.float32)
    upper_kth = agg_kth["upper"].astype(np.float32)
    valid_interval_kth = np.isfinite(lower_kth) & np.isfinite(upper_kth) & (lower_kth <= upper_kth)
    results["lower_kth"] = lower_kth
    results["upper_kth"] = upper_kth
    results["valid_interval_kth"] = valid_interval_kth

    with StepTimer("aggregate tight_kth bounds"):
        agg_tight = aggregate_endpointwise(
            lower_mat=lower_mat,
            upper_mat=upper_mat,
            valid_up=valid_up_mat,
            valid_lo=valid_lo_mat,
            k_up=1,
            k_lo=1,
        )
    lower_tight = agg_tight["lower"].astype(np.float32)
    upper_tight = agg_tight["upper"].astype(np.float32)
    valid_interval_tight = np.isfinite(lower_tight) & np.isfinite(upper_tight) & (lower_tight <= upper_tight)
    results["lower_tight_kth"] = lower_tight
    results["upper_tight_kth"] = upper_tight
    results["valid_interval_tight_kth"] = valid_interval_tight

    with StepTimer("compute summaries"):
        summary_rows = []
        for div_name in base_divs + ["kth", "tight_kth"]:
            if div_name == "kth":
                lower = results["lower_kth"]
                upper = results["upper_kth"]
                valid_interval = results["valid_interval_kth"]
            elif div_name == "tight_kth":
                lower = results["lower_tight_kth"]
                upper = results["upper_tight_kth"]
                valid_interval = results["valid_interval_tight_kth"]
            else:
                lower = results[f"lower_{div_name}"]
                upper = results[f"upper_{div_name}"]
                valid_interval = results[f"valid_interval_{div_name}"]

            metrics = summarize_bounds(lower, upper, truth_do1, valid_interval)
            metrics["divergence"] = div_name
            summary_rows.append(metrics)

        summary_df = pd.DataFrame(summary_rows)[
            ["divergence", "coverage_rate", "mean_width", "valid_interval_frac", "n_valid"]
        ]

    ordered_cols = ["i", "truth_do1"]
    for div_name in base_divs:
        ordered_cols.extend(
            [
                f"lower_{div_name}",
                f"upper_{div_name}",
                f"valid_interval_{div_name}",
            ]
        )
    ordered_cols.extend(
        [
            "lower_kth",
            "upper_kth",
            "valid_interval_kth",
            "lower_tight_kth",
            "upper_tight_kth",
            "valid_interval_tight_kth",
        ]
    )

    with StepTimer("save outputs"):
        table_df = pd.DataFrame(results)[ordered_cols]

        os.makedirs("experiments", exist_ok=True)
        table_df.to_csv("experiments/run_example_bounds_table.csv", index=False)
        summary_df.to_csv("experiments/run_example_bounds_summary.csv", index=False)

        print("Saved bounds table to experiments/run_example_bounds_table.csv")
        print(table_df.head())
        print("\nCoverage/mean width on valid intervals:")
        print(summary_df)


if __name__ == "__main__":
    main()
