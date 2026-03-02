import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _toy_frame(n: int = 20, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = (x1 + rng.normal(scale=0.4, size=n) > 0).astype(int)
    y = 0.2 + 0.6 * a + 0.2 * x1 - 0.1 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2})


def test_cli_artifacts_writes_contract_and_schema(tmp_path: Path):
    root = _repo_root()
    csv_path = tmp_path / "toy.csv"
    outdir = tmp_path / "artifacts-out"
    _toy_frame().to_csv(csv_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "itbound",
            "artifacts",
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
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    assert (outdir / "summary.txt").exists()
    assert (outdir / "results.json").exists()
    assert (outdir / "claims.json").exists()
    assert (outdir / "claims.md").exists()
    assert (outdir / "plots").is_dir()

    payload = json.loads((outdir / "results.json").read_text())
    assert payload["schema_version"] == "results_schema_v0"

    assert "provenance" in payload
    prov = payload["provenance"]
    for key in (
        "package_version",
        "git_commit",
        "git_commit_reason",
        "timestamp",
        "random_seed",
        "assumptions",
        "data_hash",
        "data_hash_reason",
        "command_line",
    ):
        assert key in prov

    assert "bounds" in payload
    assert "diagnostics" in payload

    claims_payload = json.loads((outdir / "claims.json").read_text())
    assert claims_payload["schema_version"] == "results_schema_v0"
    assert "claims" in claims_payload["claims"]
