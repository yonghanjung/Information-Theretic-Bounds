from __future__ import annotations

from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_ci_workflow_contract():
    root = _repo_root()
    ci_path = root / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists(), "Missing CI workflow: .github/workflows/ci.yml"

    payload = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
    assert payload["name"] == "CI"

    matrix_versions = payload["jobs"]["test-build"]["strategy"]["matrix"]["python-version"]
    assert matrix_versions == ["3.9", "3.11"]

    step_text = "\n".join(
        str(step.get("run", ""))
        for step in payload["jobs"]["test-build"]["steps"]
        if isinstance(step, dict)
    )
    assert 'python -m pytest -q -m "not slow"' in step_text
    assert "python -m build" in step_text
    assert "itbound --help" in step_text
    assert "itbound example" in step_text
    assert "itbound quick" in step_text
