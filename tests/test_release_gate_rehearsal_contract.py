from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_release_gate_rehearsal_contract():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "release_gate_rehearsal.sh"

    assert script.exists(), "Missing scripts/release_gate_rehearsal.sh"
    assert os.access(script, os.X_OK), "release_gate_rehearsal.sh must be executable"

    text = script.read_text(encoding="utf-8")
    assert "gh auth status" in text
    assert "release_env_bootstrap.sh --check-only" in text
    assert "release_version_sync_check.sh" in text
    assert "git tag -l" in text
    assert "git status --porcelain" in text

    proc = subprocess.run(
        [
            str(script),
            "--check-only",
            "--allow-dirty",
            "--skip-auth-check",
            "--skip-env-check",
            "--skip-version-check",
            "--tag",
            "v0.1.0-gate-check",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "[release-gate] CHECK-ONLY OK" in proc.stdout
