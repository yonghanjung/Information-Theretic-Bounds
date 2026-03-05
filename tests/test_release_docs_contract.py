from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_release_docs_contract():
    root = _repo_root()
    runbook = root / "RELEASE_RUNBOOK.md"
    changelog = root / "CHANGELOG.md"

    assert runbook.exists(), "Missing RELEASE_RUNBOOK.md"
    assert changelog.exists(), "Missing CHANGELOG.md"

    runbook_text = runbook.read_text(encoding="utf-8")
    assert "## 0. Release lane checks" in runbook_text
    assert "## 1. Local preflight" in runbook_text
    assert "## 2. Version and changelog freeze" in runbook_text
    assert "## 3. Tag and push" in runbook_text
    assert "## 4. GitHub Actions gate verification" in runbook_text
    assert "## 5. Post-release fresh-venv verification" in runbook_text
    assert "## 6. Rollback notes" in runbook_text

    changelog_text = changelog.read_text(encoding="utf-8")
    assert "## [0.1.0]" in changelog_text
