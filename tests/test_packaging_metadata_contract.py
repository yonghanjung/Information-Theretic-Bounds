from __future__ import annotations

import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_dev_extra_contract_no_twine():
    root = _repo_root()
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"(?ms)^dev\s*=\s*\[(.*?)\]\s*$", text)
    assert match is not None, "Missing [project.optional-dependencies].dev"

    body = match.group(1)
    assert '"pytest"' in body
    assert '"build"' in body
    assert '"twine"' not in body
