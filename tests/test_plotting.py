from pathlib import Path

import numpy as np
import pandas as pd

from itbound.plotting import render_plots


def _toy_bounds(n: int = 8) -> pd.DataFrame:
    idx = np.arange(n, dtype=int)
    lower = np.linspace(-0.2, 0.2, num=n)
    upper = lower + 0.4
    valid = np.ones((n,), dtype=bool)
    return pd.DataFrame(
        {
            "i": idx,
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "valid_interval": valid,
        }
    )


def test_render_plots_bounds_interval_if_available(tmp_path: Path):
    bounds = _toy_bounds()
    paths = render_plots(bounds, tmp_path)

    try:
        import matplotlib  # noqa: F401
    except Exception:
        assert paths == []
        return

    assert len(paths) >= 1
    assert any(p.name == "bounds_interval.png" for p in paths)
    assert (tmp_path / "bounds_interval.png").exists()
