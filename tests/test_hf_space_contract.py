from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPACE_APP = ROOT / "scripts" / "demo" / "hf_space_app.py"
SPACE_README = ROOT / "scripts" / "demo" / "hf_space_README.md"


def _load_space_namespace():
    pytest.importorskip("gradio")
    return runpy.run_path(str(SPACE_APP), run_name="__space_test__")


def test_hf_space_readme_pin_contract():
    text = SPACE_README.read_text(encoding="utf-8")
    assert 'python_version: "3.10"' in text
    assert 'sdk_version: "5.50.0"' in text
    assert "https://arxiv.org/abs/2601.17160" in text
    assert "https://github.com/yonghanjung/Information-Theretic-Bounds" in text


def test_hf_space_api_info_smoke():
    namespace = _load_space_namespace()
    demo = namespace["demo"]
    assert demo.get_api_info()


def test_hf_space_callback_smoke(tmp_path):
    namespace = _load_space_namespace()
    run_space_demo = namespace["run_space_demo"]

    rng = np.random.default_rng(0)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    y = 0.3 + 0.7 * a + 0.2 * x1 - 0.1 * x2 + rng.normal(scale=0.1, size=n)

    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2}).to_csv(csv_path, index=False)

    bounds_preview, width_fig, ribbon_fig, claims_md, archive_path = run_space_demo(
        csv_path=str(csv_path),
        treatment_col="a",
        outcome_col="y",
        covariates_text="x1,x2",
        ribbon_axis_col="propensity",
        divergences=["KL", "TV"],
        aggregation_mode="paper_adaptive_k",
        write_html=False,
    )

    assert not bounds_preview.empty
    assert width_fig is not None
    assert ribbon_fig is not None
    assert ribbon_fig.axes[0].get_xlabel() == "estimated propensity"
    assert ribbon_fig.axes[0].get_title() == "Binned ribbon plot by estimated propensity"
    assert "Claims summary" in claims_md
    assert Path(archive_path).exists()


def test_hf_space_callback_smoke_without_upload_uses_canonical_ihdp():
    namespace = _load_space_namespace()
    run_space_demo = namespace["run_space_demo"]

    bounds_preview, width_fig, ribbon_fig, claims_md, archive_path = run_space_demo(
        csv_path=None,
        treatment_col="treatment",
        outcome_col="y_factual",
        covariates_text="x1,x2,x3,x4,x5",
        ribbon_axis_col="propensity",
        divergences=["KL", "TV"],
        aggregation_mode="paper_adaptive_k",
        write_html=False,
    )

    assert not bounds_preview.empty
    assert width_fig is not None
    assert ribbon_fig is not None
    assert ribbon_fig.axes[0].get_xlabel() == "estimated propensity"
    assert ribbon_fig.axes[0].get_title() == "Binned ribbon plot by estimated propensity"
    assert "canonical IHDP example" in claims_md
    assert Path(archive_path).exists()
