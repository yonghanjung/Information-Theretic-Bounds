import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_cli_example_runs(tmp_path: Path):
    root = _repo_root()
    out_path = tmp_path / "example.csv"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [sys.executable, "-m", "itbound", "example", "--out", str(out_path)],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()
