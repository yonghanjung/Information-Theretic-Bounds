from __future__ import annotations

from pathlib import Path


def test_runbook_postrelease_quick_input_contract():
    root = Path(__file__).resolve().parents[1]
    text = (root / "RELEASE_RUNBOOK.md").read_text(encoding="utf-8")

    assert "/tmp/itbound-pypi-quick-input.csv" in text
    assert "python - <<'PY'" in text
    assert "--data /tmp/itbound-pypi-quick-input.csv" in text
