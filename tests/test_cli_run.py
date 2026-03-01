import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _base_config(output_path: Path) -> dict:
    return {
        "data": {
            "synthetic": {
                "n": 20,
                "d": 2,
                "seed": 7,
                "structural_type": "linear",
            }
        },
        "divergence": "KL",
        "phi": "identity",
        "propensity_model": "logistic",
        "m_model": "linear",
        "dual_net_config": {
            "hidden_sizes": [8, 8],
            "activation": "relu",
            "dropout": 0.0,
            "h_clip": 10.0,
            "device": "cpu",
        },
        "fit_config": {
            "n_folds": 2,
            "num_epochs": 1,
            "batch_size": 8,
            "lr": 5e-3,
            "weight_decay": 0.0,
            "max_grad_norm": 5.0,
            "eps_propensity": 1e-3,
            "deterministic_torch": True,
            "train_m_on_fold": True,
            "propensity_config": {
                "C": 1.0,
                "max_iter": 200,
                "penalty": "l2",
                "solver": "lbfgs",
                "n_jobs": 1,
            },
            "m_config": {
                "alpha": 1.0,
            },
            "verbose": False,
            "log_every": 1,
        },
        "seed": 7,
        "output_path": str(output_path),
    }


def test_cli_run_writes_csv(tmp_path: Path):
    root = _repo_root()
    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "bounds.csv"
    cfg = _base_config(out_path)

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [sys.executable, "-m", "itbound", "run", "--config", str(cfg_path)],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    with out_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    assert "lower" in header
    assert "upper" in header
