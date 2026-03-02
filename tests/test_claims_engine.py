import numpy as np
import pandas as pd
import pytest

from itbound.claims import compute_claims


def _claims_by_id(payload: dict) -> dict:
    return {c["id"]: c for c in payload["claims"]}


def test_compute_claims_positive_sign_and_range():
    bounds = pd.DataFrame(
        {
            "lower": [0.10, 0.20, np.nan],
            "upper": [0.40, 0.55, np.nan],
            "width": [0.30, 0.35, np.nan],
            "valid_interval": [True, True, False],
            "inverted": [False, True, False],
        }
    )

    payload = compute_claims(bounds)
    by_id = _claims_by_id(payload)

    assert payload["claims_engine_version"] == "claims_v0"
    assert payload["n_rows"] == 3
    assert payload["n_valid_intervals"] == 2
    assert payload["inverted_count"] == 1
    assert payload["valid_interval_rate"] == pytest.approx(2 / 3)
    assert payload["invalid_interval_rate"] == pytest.approx(1 / 3)
    assert payload["insufficient_valid_intervals"] is False
    assert payload["sign"]["label"] == "positive"
    assert payload["range"]["lower_min"] == pytest.approx(0.10)
    assert payload["range"]["upper_max"] == pytest.approx(0.55)

    for cid in ("sign", "range", "validity"):
        claim = by_id[cid]
        for key in ("id", "title", "statement", "evidence", "conditions", "caveats"):
            assert key in claim

    assert by_id["sign"]["statement"] == "effect is positive (robust sign)"


def test_compute_claims_insufficient_valid_intervals():
    bounds = pd.DataFrame(
        {
            "lower": [np.nan, 0.15, np.nan],
            "upper": [np.nan, 0.40, np.nan],
            "width": [np.nan, 0.25, np.nan],
            "valid_interval": [False, True, False],
        }
    )

    payload = compute_claims(bounds)
    by_id = _claims_by_id(payload)

    assert payload["insufficient_valid_intervals"] is True
    assert payload["n_valid_intervals"] == 1
    assert payload["invalid_interval_count"] == 2
    assert "insufficient valid intervals" in by_id["validity"]["statement"].lower()
    assert payload["sign"]["label"] == "not_identified"
    assert payload["sign"]["statement"].startswith("sign not identified")
