import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _toy_df(n: int = 24, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    y = 0.3 + 0.7 * a + 0.2 * x1 - 0.1 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2})


def _run_quick(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "-m", "itbound", "quick", *args],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_quick_writes_artifact_contract(tmp_path: Path):
    root = _repo_root()
    csv_path = tmp_path / "quick.csv"
    outdir = tmp_path / "quick-out"
    _toy_df().to_csv(csv_path, index=False)

    result = _run_quick(
        root,
        [
            "--data", str(csv_path),
            "--treatment", "a",
            "--outcome", "y",
            "--covariates", "x1,x2",
            "--outdir", str(outdir),
            "--divergence", "KL",
            "--num-epochs", "1",
            "--n-folds", "2",
            "--batch-size", "8",
            "--no-plots",
        ],
    )

    assert result.returncode == 0, result.stderr
    assert (outdir / "summary.txt").exists()
    assert (outdir / "results.json").exists()
    assert (outdir / "claims.json").exists()
    assert (outdir / "claims.md").exists()
    assert (outdir / "plots").is_dir()

    payload = json.loads((outdir / "results.json").read_text())
    assert payload["schema_version"] == "results_schema_v0"

    claims_payload = json.loads((outdir / "claims.json").read_text())
    assert claims_payload["schema_version"] == "results_schema_v0"
    assert "claims" in claims_payload["claims"]


def test_cli_quick_missing_column_error(tmp_path: Path):
    root = _repo_root()
    csv_path = tmp_path / "quick-missing.csv"
    _toy_df().to_csv(csv_path, index=False)

    result = _run_quick(
        root,
        [
            "--data", str(csv_path),
            "--treatment", "a",
            "--outcome", "y_missing",
            "--covariates", "x1,x2",
            "--num-epochs", "1",
            "--n-folds", "2",
            "--batch-size", "8",
            "--no-plots",
        ],
    )

    assert result.returncode == 2
    assert "Missing required columns" in result.stderr


def test_cli_quick_non_binary_treatment_error(tmp_path: Path):
    root = _repo_root()
    csv_path = tmp_path / "quick-nonbinary.csv"
    df = _toy_df()
    df.loc[0, "a"] = 2
    df.to_csv(csv_path, index=False)

    result = _run_quick(
        root,
        [
            "--data", str(csv_path),
            "--treatment", "a",
            "--outcome", "y",
            "--covariates", "x1,x2",
            "--num-epochs", "1",
            "--n-folds", "2",
            "--batch-size", "8",
            "--no-plots",
        ],
    )

    assert result.returncode == 2
    assert "binary" in result.stderr.lower()


def test_cli_quick_non_numeric_covariates_error(tmp_path: Path):
    root = _repo_root()
    csv_path = tmp_path / "quick-nonnumeric.csv"
    df = _toy_df()
    df["cat"] = ["x", "y"] * (len(df) // 2)
    df.to_csv(csv_path, index=False)

    result = _run_quick(
        root,
        [
            "--data", str(csv_path),
            "--treatment", "a",
            "--outcome", "y",
            "--covariates", "x1,cat",
            "--num-epochs", "1",
            "--n-folds", "2",
            "--batch-size", "8",
            "--no-plots",
        ],
    )

    assert result.returncode == 2
    assert "numeric" in result.stderr.lower()
