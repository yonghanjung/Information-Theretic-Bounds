from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_release_version_sync_guard_script():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "release_version_sync_check.sh"

    assert script.exists(), "Missing scripts/release_version_sync_check.sh"
    assert os.access(script, os.X_OK), "release_version_sync_check.sh must be executable"

    proc = subprocess.run(
        [str(script)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "[version-sync] OK" in proc.stdout
