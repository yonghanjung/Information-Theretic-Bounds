from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from fbound.estimators.causal_bound import compute_causal_bounds

from .artifacts import build_provenance
from .claims import compute_claims
from .config import build_phi, default_example_config
from .report import BoundsReport


def _find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def _default_configs() -> tuple[Dict[str, Any], Dict[str, Any]]:
    base = default_example_config(Path("itbound_bounds.csv"))
    return dict(base["dual_net_config"]), dict(base["fit_config"])


def _require_numeric(frame: pd.DataFrame, cols: Sequence[str], *, label: str) -> None:
    bad = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
    if bad:
        joined = ", ".join(bad)
        raise ValueError(f"{label} must be numeric for MVP. Non-numeric columns: {joined}")


def compute_bounds_diagnostics(bounds_df: pd.DataFrame, *, mode: str, divergence: str) -> Dict[str, Any]:
    lower = bounds_df["lower"].to_numpy(dtype=np.float64)
    upper = bounds_df["upper"].to_numpy(dtype=np.float64)
    width = bounds_df["width"].to_numpy(dtype=np.float64)

    if "valid_interval" in bounds_df.columns:
        valid_interval = bounds_df["valid_interval"].to_numpy(dtype=bool)
    else:
        valid_interval = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)

    if "valid_up" in bounds_df.columns:
        valid_up = bounds_df["valid_up"].to_numpy(dtype=bool)
    else:
        valid_up = np.isfinite(upper)

    if "valid_lo" in bounds_df.columns:
        valid_lo = bounds_df["valid_lo"].to_numpy(dtype=bool)
    else:
        valid_lo = np.isfinite(lower)

    if "inverted" in bounds_df.columns:
        inverted = bounds_df["inverted"].to_numpy(dtype=bool)
    else:
        inverted = np.isfinite(lower) & np.isfinite(upper) & (lower > upper)

    n_rows = int(bounds_df.shape[0])
    n_valid = int(np.count_nonzero(valid_interval))
    valid_rate = float(n_valid / n_rows) if n_rows > 0 else 0.0
    invalid_rate = float(1.0 - valid_rate)

    finite_lower_count = int(np.count_nonzero(np.isfinite(lower)))
    finite_upper_count = int(np.count_nonzero(np.isfinite(upper)))
    finite_width_count = int(np.count_nonzero(np.isfinite(width)))

    non_finite_lower_count = int(np.count_nonzero(~np.isfinite(lower)))
    non_finite_upper_count = int(np.count_nonzero(~np.isfinite(upper)))
    non_finite_width_count = int(np.count_nonzero(~np.isfinite(width)))

    invalid_up_count = int(np.count_nonzero(~valid_up))
    invalid_lo_count = int(np.count_nonzero(~valid_lo))
    inverted_count = int(np.count_nonzero(inverted))
    invalid_interval_count = int(np.count_nonzero(~valid_interval))

    invalid_domain_proxy_count = int(np.count_nonzero((~valid_up) | (~valid_lo)))
    invalid_domain_proxy_rate = float(invalid_domain_proxy_count / n_rows) if n_rows > 0 else 0.0

    return {
        "mode": mode,
        "divergence": str(divergence),
        "interval_validity": {
            "n_rows": n_rows,
            "n_valid_intervals": n_valid,
            "invalid_interval_count": invalid_interval_count,
            "valid_interval_rate": valid_rate,
            "invalid_interval_rate": invalid_rate,
        },
        "nan_propagation": {
            "finite_lower_count": finite_lower_count,
            "finite_upper_count": finite_upper_count,
            "finite_width_count": finite_width_count,
            "non_finite_any_count": int(np.count_nonzero(~np.isfinite(lower) | ~np.isfinite(upper) | ~np.isfinite(width))),
            "by_reason": {
                "non_finite_lower": non_finite_lower_count,
                "non_finite_upper": non_finite_upper_count,
                "non_finite_width": non_finite_width_count,
                "invalid_up_mask": invalid_up_count,
                "invalid_lo_mask": invalid_lo_count,
                "inverted_interval": inverted_count,
                "invalid_interval": invalid_interval_count,
            },
        },
        "invalid_domain": {
            "available": True,
            "rate": invalid_domain_proxy_rate,
            "count": invalid_domain_proxy_count,
            "source": "proxy_from_valid_up_valid_lo_masks",
            "note": "Quick mode exposes a proxy invalid-domain rate from valid_up/valid_lo masks; not a direct conjugate-domain counter.",
        },
        "aggregation": {
            "applicable": False,
            "status": "not_applicable_single_divergence_quick",
            "k_used": None,
            "effective_candidate_counts": None,
            "filtered_count_breakdown": None,
        },
        "missingness": {"status": "stub_v0", "note": "missingness checks not implemented in fit() MVP"},
        "overlap": {"status": "stub_v0", "note": "overlap checks not implemented in fit() MVP"},
    }


def fit(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    mode: str = "paper-default",
    divergence: str = "KL",
    phi: str = "identity",
    propensity_model: str = "logistic",
    m_model: str = "linear",
    dual_overrides: Optional[Dict[str, Any]] = None,
    fit_overrides: Optional[Dict[str, Any]] = None,
    seed: int = 123,
) -> BoundsReport:
    if mode != "paper-default":
        raise ValueError(f"Unsupported mode '{mode}'. Supported: 'paper-default'.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not covariates:
        raise ValueError("covariates must be non-empty.")

    required = [outcome, treatment, *covariates]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    _require_numeric(df, [outcome], label="outcome")
    _require_numeric(df, list(covariates), label="covariates")

    a_vals = np.unique(df[treatment].to_numpy())
    if not set(a_vals.tolist()).issubset({0, 1}):
        raise ValueError(f"treatment must be binary 0/1 for MVP. Found values: {a_vals.tolist()}")

    dual_cfg, fit_cfg = _default_configs()
    if dual_overrides:
        dual_cfg.update(dict(dual_overrides))
    if fit_overrides:
        fit_cfg.update(dict(fit_overrides))

    y = df[outcome].to_numpy()
    a = df[treatment].to_numpy()
    x = df[list(covariates)].to_numpy()

    phi_fn = build_phi(phi)
    bounds = compute_causal_bounds(
        Y=y,
        A=a,
        X=x,
        divergence=divergence,
        phi=phi_fn,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_cfg,
        fit_config=fit_cfg,
        seed=int(seed),
        GroundTruth=None,
    ).sort_values("i").reset_index(drop=True)

    claims = compute_claims(bounds)
    diagnostics = compute_bounds_diagnostics(bounds, mode=mode, divergence=str(divergence))

    try:
        package_version = getattr(importlib.import_module("itbound"), "__version__", "unknown")
    except Exception:
        package_version = "unknown"

    assumptions = {
        "mode": mode,
        "paper_default_math": True,
        "treatment_domain": "binary_0_1",
        "divergence": str(divergence),
    }
    provenance = build_provenance(
        package_version=package_version,
        random_seed=int(seed),
        assumptions=assumptions,
        data_source=df,
        repo_root=_find_repo_root(Path(__file__).resolve().parent),
    )

    return BoundsReport(
        bounds_df=bounds,
        claims=claims,
        diagnostics=diagnostics,
        provenance=provenance,
        warnings=[],
    )


__all__ = ["compute_bounds_diagnostics", "fit"]
