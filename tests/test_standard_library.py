import json
import os
import subprocess
import sys
from pathlib import Path
import builtins

import numpy as np
import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _toy_frame(n: int = 24, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    y = 0.4 + 0.8 * a + 0.3 * x1 - 0.2 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2})


def _toy_frame_with_truth(n: int = 24, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    mu0 = 0.25 + 0.2 * x1 - 0.1 * x2
    tau = 0.6 + 0.1 * np.tanh(x1)
    mu1 = mu0 + tau
    y = mu0 + tau * a + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2, "mu0": mu0, "mu1": mu1, "tau_true": tau})


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


def test_cli_standard_tight_kth_outputs(tmp_path: Path):
    root = _repo_root()
    csv_path = tmp_path / "toy-tight-kth.csv"
    outdir = tmp_path / "standard-cli-tight-kth"
    _toy_frame(n=30, seed=19).to_csv(csv_path, index=False)

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
            "KL,JS",
            "--aggregation-mode",
            "tight_kth",
            "--num-epochs",
            "1",
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
    payload = json.loads((outdir / "summary.json").read_text())
    assert payload["diagnostics"]["aggregation_mode"] == "tight_kth"
    assert payload["run_config"]["divergences"] == ["KL", "JS"]


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


def test_standard_api_tight_kth_mode(tmp_path: Path):
    from itbound.standard import run_standard_bounds

    outdir = tmp_path / "standard-tight-kth"
    result = run_standard_bounds(
        dataframe=_toy_frame(n=30, seed=17),
        outcome_col="y",
        treatment_col="a",
        covariate_cols=["x1", "x2"],
        divergences=["KL", "JS"],
        aggregation_mode="tight_kth",
        outdir=outdir,
        write_plots=False,
        write_html=False,
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
    )

    payload = json.loads(result.summary_path.read_text())
    diagnostics = payload["diagnostics"]
    assert diagnostics["aggregation_mode"] == "tight_kth"
    assert "tight_k_start" in diagnostics
    assert len(diagnostics["tight_k_start"]) == 30


def test_standard_api_tight_kth_default_divergences(tmp_path: Path):
    from itbound.standard import DEFAULT_DIVERGENCES, run_standard_bounds

    outdir = tmp_path / "standard-tight-kth-default-divergences"
    result = run_standard_bounds(
        dataframe=_toy_frame(n=30, seed=23),
        outcome_col="y",
        treatment_col="a",
        covariate_cols=["x1", "x2"],
        aggregation_mode="tight_kth",
        outdir=outdir,
        write_plots=False,
        write_html=False,
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
    )

    payload = json.loads(result.summary_path.read_text())
    assert payload["run_config"]["divergences"] == list(DEFAULT_DIVERGENCES)


def test_standard_api_ground_truth_auto_detection(tmp_path: Path):
    from itbound.standard import run_standard_bounds

    outdir = tmp_path / "standard-gt-auto"
    result = run_standard_bounds(
        dataframe=_toy_frame_with_truth(),
        outcome_col="y",
        treatment_col="a",
        covariate_cols=["x1", "x2"],
        divergences=["KL"],
        outdir=outdir,
        write_plots=False,
        write_html=False,
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
    )

    payload = json.loads(result.summary_path.read_text())
    gt = payload["run_config"]["ground_truth_plot"]
    assert gt["source"] == "auto_mu1_mu0"
    assert int(gt["n_truth_points"]) == 24
    assert gt["ground_truth_effect"] is not None


def test_standard_api_ground_truth_explicit_col_precedence(tmp_path: Path):
    from itbound.standard import run_standard_bounds

    df = _toy_frame_with_truth()
    df["gt_override"] = np.full((int(df.shape[0]),), 0.42, dtype=np.float64)

    outdir = tmp_path / "standard-gt-explicit-col"
    result = run_standard_bounds(
        dataframe=df,
        outcome_col="y",
        treatment_col="a",
        covariate_cols=["x1", "x2"],
        divergences=["KL"],
        ground_truth_col="gt_override",
        auto_ground_truth=True,
        outdir=outdir,
        write_plots=False,
        write_html=False,
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
    )

    payload = json.loads(result.summary_path.read_text())
    gt = payload["run_config"]["ground_truth_plot"]
    assert gt["source"] == "explicit_col"
    assert abs(float(gt["ground_truth_effect"]) - 0.42) < 1e-8


def test_standard_api_ground_truth_missing_or_nonnumeric_warns(tmp_path: Path):
    from itbound.standard import run_standard_bounds

    df = _toy_frame_with_truth()
    df["bad_truth"] = ["bad"] * int(df.shape[0])

    outdir_missing = tmp_path / "standard-gt-missing"
    result_missing = run_standard_bounds(
        dataframe=df,
        outcome_col="y",
        treatment_col="a",
        covariate_cols=["x1", "x2"],
        divergences=["KL"],
        ground_truth_col="missing_truth",
        auto_ground_truth=False,
        outdir=outdir_missing,
        write_plots=False,
        write_html=False,
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
    )

    payload_missing = json.loads(result_missing.summary_path.read_text())
    gt_missing = payload_missing["run_config"]["ground_truth_plot"]
    assert gt_missing["source"] == "none"
    assert any("not found" in w for w in gt_missing["warnings"])

    outdir_bad = tmp_path / "standard-gt-bad"
    result_bad = run_standard_bounds(
        dataframe=df,
        outcome_col="y",
        treatment_col="a",
        covariate_cols=["x1", "x2"],
        divergences=["KL"],
        ground_truth_col="bad_truth",
        auto_ground_truth=False,
        outdir=outdir_bad,
        write_plots=False,
        write_html=False,
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
    )

    payload_bad = json.loads(result_bad.summary_path.read_text())
    gt_bad = payload_bad["run_config"]["ground_truth_plot"]
    assert gt_bad["source"] == "none"
    assert any("no finite numeric values" in w for w in gt_bad["warnings"])


def test_standard_write_plots_ground_truth_length_mismatch(tmp_path: Path):
    from itbound.standard import _write_plots

    bounds = pd.DataFrame(
        {
            "i": [0, 1, 2],
            "lower": [0.0, 0.1, 0.2],
            "upper": [1.0, 1.1, 1.2],
            "width": [1.0, 1.0, 1.0],
            "valid_interval": [True, True, True],
        }
    )
    with pytest.raises(ValueError, match="ground_truth_values length must match bounds rows"):
        _write_plots(bounds, tmp_path, ground_truth_values=np.array([0.1, 0.2], dtype=np.float64))
