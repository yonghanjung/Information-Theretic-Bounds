import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_cli_reproduce_dry_run():
    root = _repo_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [sys.executable, "-m", "itbound", "reproduce", "--dry-run"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    expected = str(root / "scripts" / "reproduce_final_arxiv_plots.py")
    assert expected in result.stdout


def test_cli_reproduce_packaged_fallback(monkeypatch, capsys):
    import itbound.cli as cli

    monkeypatch.setattr(cli, "ROOT", Path("/tmp/itbound-missing-root"))
    rc = cli._reproduce(True, "", "", "")
    out = capsys.readouterr().out

    assert rc == 0
    assert "-m itbound.reproduce_final_arxiv_plots" in out
