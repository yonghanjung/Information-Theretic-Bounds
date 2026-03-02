import json
import os
import subprocess
import sys
from pathlib import Path
import builtins

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _toy_frame(n: int = 24, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    y = 0.4 + 0.8 * a + 0.3 * x1 - 0.2 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2})


def test_standard_api_dataframe_outputs(tmp_path: Path):
    from itbound.standard import run_standard_bounds

    outdir = tmp_path / "standard-api"
    result = run_standard_bounds(
        dataframe=_toy_frame(),
        outcome_col="y",
        treatment_col="a",
        covariate_cols=["x1", "x2"],
        divergences=["KL"],
        outdir=outdir,
        write_plots=False,
        write_html=True,
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
    )

    assert result.bounds_path.exists()
    assert result.summary_path.exists()
    assert result.html_path is not None and result.html_path.exists()

    payload = json.loads(result.summary_path.read_text())
    assert payload["claims"]["n_rows"] == 24
    assert "invalid_interval_rate" in payload["claims"]
    assert "diagnostics" in payload


def test_cli_standard_csv_outputs(tmp_path: Path):
    root = _repo_root()
    csv_path = tmp_path / "toy.csv"
    outdir = tmp_path / "standard-cli"
    _toy_frame().to_csv(csv_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "itbound",
            "standard",
            "--csv",
            str(csv_path),
            "--y-col",
            "y",
            "--a-col",
            "a",
            "--x-cols",
            "x1,x2",
            "--outdir",
            str(outdir),
            "--divergences",
            "KL",
            "--num-epochs",
            "1",
            "--n-folds",
            "2",
            "--batch-size",
            "8",
            "--no-plots",
            "--html",
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (outdir / "bounds.csv").exists()
    assert (outdir / "summary.json").exists()
    assert (outdir / "report.html").exists()


def test_standard_api_plot_dependency_graceful(tmp_path: Path, monkeypatch):
    from itbound.standard import run_standard_bounds

    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("matplotlib intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    result = run_standard_bounds(
        dataframe=_toy_frame(),
        outcome_col="y",
        treatment_col="a",
        covariate_cols=["x1", "x2"],
        divergences=["KL"],
        outdir=tmp_path / "no-mpl",
        write_plots=True,
        write_html=False,
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
    )

    assert result.plot_paths == []
    assert any("matplotlib" in warning.lower() for warning in result.warnings)
