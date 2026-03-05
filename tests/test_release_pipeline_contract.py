from __future__ import annotations

from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_release_workflow_contract():
    root = _repo_root()
    release_path = root / ".github" / "workflows" / "release.yml"
    assert release_path.exists(), "Missing release workflow: .github/workflows/release.yml"

    payload = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    jobs = payload["jobs"]

    assert "build-dist" in jobs
    assert "publish-testpypi" in jobs
    assert "install-smoke-testpypi" in jobs
    assert "publish-pypi" in jobs

    assert jobs["publish-testpypi"]["needs"] == "build-dist"
    assert jobs["install-smoke-testpypi"]["needs"] == "publish-testpypi"
    assert jobs["publish-pypi"]["needs"] == "install-smoke-testpypi"

    run_text = "\n".join(
        str(step.get("run", ""))
        for step in jobs["install-smoke-testpypi"]["steps"]
        if isinstance(step, dict)
    )
    assert "itbound==" in run_text
    assert "for i in 1 2 3 4 5; do" in run_text
    assert "exit 1" in run_text
