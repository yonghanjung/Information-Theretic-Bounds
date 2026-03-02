from pathlib import Path

import builtins
import numpy as np
import pandas as pd
import pytest

import itbound


def _toy_df(n: int = 20, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    y = 0.3 + 0.7 * a + 0.2 * x1 - 0.1 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"Y": y, "A": a, "X1": x1, "X2": x2})


def test_fit_api_report_interface_and_bounds_df(tmp_path: Path):
    df = _toy_df()
    res = itbound.fit(
        df,
        treatment="A",
        outcome="Y",
        covariates=["X1", "X2"],
        mode="paper-default",
        divergence="KL",
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
        seed=77,
    )

    text = res.summary().lower()
    assert "lower" in text
    assert "upper" in text

    for col in ("lower", "upper", "width"):
        assert col in res.bounds_df.columns

    payload = res.to_json_dict()
    assert payload["schema_version"] == "results_schema_v0"
    assert "provenance" in payload
    assert "bounds" in payload
    assert "diagnostics" in payload

    outdir = tmp_path / "fit-save"
    paths = res.save(outdir, write_plots=False, write_html=False)
    assert paths.summary_txt.exists()
    assert paths.results_json.exists()
    assert paths.claims_json.exists()
    assert paths.claims_md is not None and paths.claims_md.exists()
    assert paths.plots_dir.is_dir()


def test_fit_api_non_numeric_covariates_raise():
    df = _toy_df()
    df["cat"] = ["a", "b"] * (len(df) // 2) + (["a"] if len(df) % 2 else [])

    with pytest.raises(ValueError, match="numeric"):
        itbound.fit(
            df,
            treatment="A",
            outcome="Y",
            covariates=["X1", "cat"],
            fit_overrides={"num_epochs": 1, "n_folds": 2, "batch_size": 8},
        )


def test_fit_save_without_matplotlib_writes_install_hint(tmp_path: Path, monkeypatch):
    df = _toy_df()
    res = itbound.fit(
        df,
        treatment="A",
        outcome="Y",
        covariates=["X1", "X2"],
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
        seed=71,
    )

    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("matplotlib intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    paths = res.save(tmp_path / "fit-save-no-mpl", write_plots=True, write_html=False)
    summary_text = paths.summary_txt.read_text().lower()
    assert "pip install itbound[experiments]" in summary_text


def test_fit_save_writes_bounds_interval_plot_when_matplotlib_available(tmp_path: Path):
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return

    df = _toy_df()
    res = itbound.fit(
        df,
        treatment="A",
        outcome="Y",
        covariates=["X1", "X2"],
        fit_overrides={
            "num_epochs": 1,
            "n_folds": 2,
            "batch_size": 8,
            "verbose": False,
            "log_every": 1,
        },
        seed=73,
    )

    paths = res.save(tmp_path / "fit-save-with-mpl", write_plots=True, write_html=False)
    assert (paths.plots_dir / "bounds_interval.png").exists()
