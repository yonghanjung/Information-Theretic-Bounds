from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_release_preflight_contract():
    root = _repo_root()
    script = root / "scripts" / "release_preflight.sh"
    assert script.exists(), "Missing preflight script: scripts/release_preflight.sh"
    assert os.access(script, os.X_OK), "Preflight script must be executable"

    text = script.read_text(encoding="utf-8")
    assert 'python3 -m pytest -q -m "not slow"' in text
    assert "python3 -m build" in text
    assert "itbound --help" in text
    assert "itbound example" in text
    assert "itbound quick" in text
    assert "results.json" in text
