from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

CLAIMS_ENGINE_VERSION = "claims_v0"


def _required_columns(bounds_df: pd.DataFrame) -> None:
    required = ("lower", "upper", "width")
    missing = [c for c in required if c not in bounds_df.columns]
    if missing:
        raise ValueError(f"Missing required bounds columns for claims: {missing}")


def _claim_object(
    *,
    claim_id: str,
    title: str,
    statement: str,
    evidence: Dict[str, Any],
    conditions: list[str],
    caveats: list[str],
) -> Dict[str, Any]:
    return {
        "id": claim_id,
        "title": title,
        "statement": statement,
        "evidence": evidence,
        "conditions": conditions,
        "caveats": caveats,
    }


def compute_claims(bounds_df: pd.DataFrame, *, alpha: float | None = None) -> Dict[str, Any]:
    _required_columns(bounds_df)

    lower = bounds_df["lower"].to_numpy(dtype=np.float64)
    upper = bounds_df["upper"].to_numpy(dtype=np.float64)
    width = bounds_df["width"].to_numpy(dtype=np.float64)

    if "valid_interval" in bounds_df.columns:
        valid_interval = bounds_df["valid_interval"].to_numpy(dtype=bool)
    else:
        valid_interval = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)

    finite_valid = valid_interval & np.isfinite(lower) & np.isfinite(upper) & np.isfinite(width)

    n_rows = int(bounds_df.shape[0])
    n_valid = int(np.count_nonzero(finite_valid))
    invalid_count = int(n_rows - n_valid)
    valid_rate = float(n_valid / n_rows) if n_rows > 0 else 0.0
    invalid_rate = float(1.0 - valid_rate)

    if "inverted" in bounds_df.columns:
        inverted = bounds_df["inverted"].to_numpy(dtype=bool)
        inverted_count = int(np.count_nonzero(inverted))
    else:
        inverted_count = 0

    insufficient = bool(n_valid == 0 or invalid_count > n_valid)

    lower_valid = lower[finite_valid]
    upper_valid = upper[finite_valid]
    width_valid = width[finite_valid]

    if lower_valid.size:
        lower_min = float(np.min(lower_valid))
        upper_max = float(np.max(upper_valid))
    else:
        lower_min = None
        upper_max = None

    if insufficient:
        sign_label = "not_identified"
        sign_statement = "sign not identified (insufficient valid intervals)"
    elif lower_min is not None and lower_min > 0.0:
        sign_label = "positive"
        sign_statement = "effect is positive (robust sign)"
    elif upper_max is not None and upper_max < 0.0:
        sign_label = "negative"
        sign_statement = "effect is negative (robust sign)"
    else:
        sign_label = "not_identified"
        sign_statement = "sign not identified"

    if lower_min is None or upper_max is None:
        range_statement = "insufficient valid intervals"
    else:
        range_statement = f"range over valid intervals: [{lower_min:.6g}, {upper_max:.6g}]"

    if insufficient:
        validity_statement = "insufficient valid intervals"
    else:
        validity_statement = "sufficient valid intervals for summary claims"

    sign_claim = _claim_object(
        claim_id="sign",
        title="Robust Sign Claim",
        statement=sign_statement,
        evidence={
            "lower_min": lower_min,
            "upper_max": upper_max,
            "n_valid_intervals": n_valid,
            "n_rows": n_rows,
        },
        conditions=[
            "Computed from finite, valid bound intervals only.",
            "No additional causal assumptions beyond bound construction.",
        ],
        caveats=[
            "Sign claim is limited to rows with valid intervals.",
            "Sign non-identification does not imply zero effect.",
        ],
    )

    range_claim = _claim_object(
        claim_id="range",
        title="Range Claim",
        statement=range_statement,
        evidence={
            "lower_min": lower_min,
            "upper_max": upper_max,
            "width_mean": float(np.mean(width_valid)) if width_valid.size else None,
        },
        conditions=[
            "Range uses min(lower) and max(upper) over valid intervals.",
            "Rows with non-finite values are excluded.",
        ],
        caveats=[
            "Range summarizes observed rows; it is not a population-wide guarantee.",
        ],
    )

    validity_claim = _claim_object(
        claim_id="validity",
        title="Validity Claim",
        statement=validity_statement,
        evidence={
            "n_rows": n_rows,
            "n_valid_intervals": n_valid,
            "invalid_interval_count": invalid_count,
            "valid_interval_rate": valid_rate,
            "invalid_interval_rate": invalid_rate,
            "inverted_count": inverted_count,
            "non_finite_lower_count": int(np.count_nonzero(~np.isfinite(lower))),
            "non_finite_upper_count": int(np.count_nonzero(~np.isfinite(upper))),
            "non_finite_width_count": int(np.count_nonzero(~np.isfinite(width))),
        },
        conditions=[
            "Validity is based on finite lower/upper/width and valid_interval mask.",
        ],
        caveats=[
            "High invalid or non-finite rates weaken interpretability of sign/range claims.",
        ],
    )

    return {
        "claims_engine_version": CLAIMS_ENGINE_VERSION,
        "alpha": alpha,
        "n_rows": n_rows,
        "n_valid_intervals": n_valid,
        "invalid_interval_count": invalid_count,
        "valid_interval_rate": valid_rate,
        "invalid_interval_rate": invalid_rate,
        "inverted_count": inverted_count,
        "insufficient_valid_intervals": insufficient,
        "sign": {"label": sign_label, "statement": sign_statement},
        "range": {"lower_min": lower_min, "upper_max": upper_max},
        "claims": [sign_claim, range_claim, validity_claim],
    }


__all__ = ["CLAIMS_ENGINE_VERSION", "compute_claims"]
