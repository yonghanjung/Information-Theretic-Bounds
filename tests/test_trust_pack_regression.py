import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tiny_regression_df(n: int = 60, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    y = 0.25 + 0.65 * a + 0.2 * x1 - 0.1 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2})


def test_quick_golden_outputs_regression(tmp_path: Path):
    root = _repo_root()
    csv_path = tmp_path / "golden.csv"
    outdir = tmp_path / "golden-out"
    _tiny_regression_df().to_csv(csv_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "itbound",
            "quick",
            "--data",
            str(csv_path),
            "--treatment",
            "a",
            "--outcome",
            "y",
            "--covariates",
            "x1,x2",
            "--outdir",
            str(outdir),
            "--divergence",
            "TV",
            "--num-epochs",
            "2",
            "--n-folds",
            "2",
            "--batch-size",
            "8",
            "--no-plots",
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    results_path = outdir / "results.json"
    assert results_path.exists()

    payload = json.loads(results_path.read_text())

    assert set(payload.keys()) == {"schema_version", "provenance", "bounds", "diagnostics"}
    assert payload["schema_version"] == "results_schema_v0"

    bounds = payload["bounds"]
    assert {"n_rows", "n_valid_intervals", "lower", "upper", "width"}.issubset(bounds.keys())

    diagnostics = payload["diagnostics"]
    assert {"interval_validity", "nan_propagation", "invalid_domain", "aggregation"}.issubset(diagnostics.keys())

    validity = diagnostics["interval_validity"]
    assert "valid_interval_rate" in validity
    assert 0.0 <= float(validity["valid_interval_rate"]) <= 1.0

    nan_prop = diagnostics["nan_propagation"]
    assert {"finite_lower_count", "finite_upper_count", "by_reason"}.issubset(nan_prop.keys())
    assert int(nan_prop["finite_lower_count"]) >= 10
    assert int(nan_prop["finite_upper_count"]) >= 10
    by_reason = nan_prop["by_reason"]
    assert {"non_finite_lower", "non_finite_upper", "invalid_up_mask", "invalid_lo_mask"}.issubset(by_reason.keys())

    invalid_domain = diagnostics["invalid_domain"]
    assert "rate" in invalid_domain
    assert 0.0 <= float(invalid_domain["rate"]) <= 1.0

    aggregation = diagnostics["aggregation"]
    assert aggregation["applicable"] is False
    assert {"k_used", "effective_candidate_counts", "filtered_count_breakdown"}.issubset(aggregation.keys())
