from __future__ import annotations

from pathlib import Path


def test_release_gitignore_contract():
    root = Path(__file__).resolve().parents[1]
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert "dist/" in text
    assert "*.egg-info/" in text
