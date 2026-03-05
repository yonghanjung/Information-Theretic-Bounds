from __future__ import annotations

import os
from pathlib import Path


def test_release_external_ops_contract_files():
    root = Path(__file__).resolve().parents[1]
    bootstrap = root / "scripts" / "release_env_bootstrap.sh"
    checklist = root / "scripts" / "release_trusted_publisher_checklist.md"

    assert bootstrap.exists(), "Missing scripts/release_env_bootstrap.sh"
    assert os.access(bootstrap, os.X_OK), "release_env_bootstrap.sh must be executable"
    assert checklist.exists(), "Missing scripts/release_trusted_publisher_checklist.md"

    bootstrap_text = bootstrap.read_text(encoding="utf-8")
    assert "testpypi" in bootstrap_text
    assert "pypi" in bootstrap_text
    assert "gh api" in bootstrap_text
    assert "repos/${REPO_SLUG}/environments" in bootstrap_text
    assert "environments" in bootstrap_text

    checklist_text = checklist.read_text(encoding="utf-8")
    assert ".github/workflows/release.yml" in checklist_text
    assert "refs/tags/*" in checklist_text
    assert "Information-Theretic-Bounds" in checklist_text
