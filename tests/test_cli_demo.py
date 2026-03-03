import os
import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_demo(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "-m", "itbound", "demo", *args],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _fake_ihdp(path: Path, n: int = 80, seed: int = 41) -> Path:
    rng = np.random.default_rng(seed)
    x = {f"x{i}": rng.normal(size=n) for i in range(1, 26)}
    treatment = (x["x1"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    mu0 = 0.2 + 0.3 * x["x1"] - 0.1 * x["x2"]
    mu1 = mu0 + 0.7 + 0.2 * x["x3"]
    y0 = mu0 + rng.normal(scale=0.1, size=n)
    y1 = mu1 + rng.normal(scale=0.1, size=n)
    y_factual = np.where(treatment == 1, y1, y0)
    y_cfactual = np.where(treatment == 1, y0, y1)

    df = pd.DataFrame(
        {
            "treatment": treatment,
            "y_factual": y_factual,
            "y_cfactual": y_cfactual,
            "mu0": mu0,
            "mu1": mu1,
            **x,
        }
    )
    df.to_csv(path, index=False)
    return path


def test_cli_demo_toy_writes_artifacts(tmp_path: Path):
    root = _repo_root()
    outdir = tmp_path / "demo-toy"
    result = _run_demo(
        root,
        [
            "--scenario",
            "toy",
            "--outdir",
            str(outdir),
            "--toy-n",
            "120",
            "--num-epochs",
            "1",
            "--n-folds",
            "2",
            "--batch-size",
            "8",
            "--no-plots",
        ],
    )
    assert result.returncode == 0, result.stderr

    toy_dir = outdir / "toy"
    assert (toy_dir / "summary.txt").exists()
    assert (toy_dir / "results.json").exists()
    assert (toy_dir / "claims.json").exists()
    summary = outdir / "live_demo_summary.md"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "ground_truth_effect: 0.65" in text


def test_cli_demo_ihdp_with_custom_file(tmp_path: Path):
    root = _repo_root()
    outdir = tmp_path / "demo-ihdp"
    ihdp_path = _fake_ihdp(tmp_path / "ihdp_npci_1.csv")
    result = _run_demo(
        root,
        [
            "--scenario",
            "ihdp",
            "--ihdp-data",
            str(ihdp_path),
            "--outdir",
            str(outdir),
            "--toy-n",
            "120",
            "--num-epochs",
            "1",
            "--n-folds",
            "2",
            "--batch-size",
            "8",
            "--no-plots",
        ],
    )
    assert result.returncode == 0, result.stderr

    ihdp_dir = outdir / "ihdp"
    assert (ihdp_dir / "summary.txt").exists()
    assert (ihdp_dir / "results.json").exists()
    assert (ihdp_dir / "claims.json").exists()
    summary = outdir / "live_demo_summary.md"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "ground_truth_effect:" in text
    assert "truth_coverage_enforced: True" in text


def test_cli_demo_ihdp_disable_enforced_truth_coverage(tmp_path: Path):
    root = _repo_root()
    outdir = tmp_path / "demo-ihdp-no-enforce"
    ihdp_path = _fake_ihdp(tmp_path / "ihdp_npci_disable_enforce.csv")
    result = _run_demo(
        root,
        [
            "--scenario",
            "ihdp",
            "--ihdp-data",
            str(ihdp_path),
            "--no-enforce-truth-coverage",
            "--rounds",
            "2",
            "--n-folds",
            "2",
            "--batch-size",
            "8",
            "--outdir",
            str(outdir),
            "--no-plots",
        ],
    )
    assert result.returncode == 0, result.stderr
    summary = outdir / "live_demo_summary.md"
    text = summary.read_text(encoding="utf-8")
    assert "truth_coverage_enforced: False" in text


def test_cli_demo_rounds_alias_and_toy_n(tmp_path: Path):
    root = _repo_root()
    outdir = tmp_path / "demo-rounds-alias"
    result = _run_demo(
        root,
        [
            "--scenario",
            "toy",
            "--toy-n",
            "80",
            "--rounds",
            "2",
            "--n-folds",
            "2",
            "--batch-size",
            "8",
            "--outdir",
            str(outdir),
            "--no-plots",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert (outdir / "toy" / "results.json").exists()


def test_cli_demo_ihdp_enforce_truth_coverage(tmp_path: Path):
    root = _repo_root()
    outdir = tmp_path / "demo-ihdp-enforced"
    ihdp_path = _fake_ihdp(tmp_path / "ihdp_npci_enforced.csv", n=120)
    result = _run_demo(
        root,
        [
            "--scenario",
            "ihdp",
            "--ihdp-data",
            str(ihdp_path),
            "--enforce-truth-coverage",
            "--rounds",
            "2",
            "--n-folds",
            "2",
            "--batch-size",
            "8",
            "--outdir",
            str(outdir),
            "--no-plots",
        ],
    )
    assert result.returncode == 0, result.stderr
    summary = outdir / "live_demo_summary.md"
    text = summary.read_text(encoding="utf-8")
    assert "truth_coverage_enforced: True" in text
    assert "truth_coverage_plot: 1.0" in text


def test_cli_demo_eval_points_downsamples_artifacts(tmp_path: Path):
    root = _repo_root()
    outdir = tmp_path / "demo-ihdp-eval-points"
    ihdp_path = _fake_ihdp(tmp_path / "ihdp_npci_eval_points.csv", n=120)
    result = _run_demo(
        root,
        [
            "--scenario",
            "ihdp",
            "--ihdp-data",
            str(ihdp_path),
            "--eval-points",
            "25",
            "--rounds",
            "2",
            "--n-folds",
            "2",
            "--batch-size",
            "8",
            "--no-plots",
            "--outdir",
            str(outdir),
        ],
    )
    assert result.returncode == 0, result.stderr
    summary_text = (outdir / "live_demo_summary.md").read_text(encoding="utf-8")
    assert "eval_points_requested: 25" in summary_text
    assert "eval_points_used: 25" in summary_text

    results = json.loads((outdir / "ihdp" / "results.json").read_text(encoding="utf-8"))
    assert int(results["bounds"]["n_rows"]) == 25
