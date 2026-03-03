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


def test_render_plots_draws_ground_truth_line_if_available(tmp_path: Path, monkeypatch):
    bounds = _toy_bounds()

    try:
        from matplotlib.axes import Axes
    except Exception:
        paths = render_plots(bounds, tmp_path, ground_truth_effect=0.65)
        assert paths == []
        return

    calls: list[float] = []
    original_axhline = Axes.axhline

    def _spy(self, y=0, *args, **kwargs):
        calls.append(float(y))
        return original_axhline(self, y=y, *args, **kwargs)

    monkeypatch.setattr(Axes, "axhline", _spy)

    paths = render_plots(bounds, tmp_path, ground_truth_effect=0.65)
    assert any(p.name == "bounds_interval.png" for p in paths)
    assert any(abs(v - 0.65) < 1e-12 for v in calls)


def test_render_plots_enforce_truth_coverage_for_plot(tmp_path: Path, monkeypatch):
    bounds = pd.DataFrame(
        {
            "i": np.arange(4, dtype=int),
            "lower": np.array([0.3, 0.4, 0.2, 0.1], dtype=float),
            "upper": np.array([0.5, 0.7, 0.3, 0.4], dtype=float),
            "width": np.array([0.2, 0.3, 0.1, 0.3], dtype=float),
            "valid_interval": np.ones((4,), dtype=bool),
        }
    )
    truth = np.array([0.1, 0.9, 0.25, 0.2], dtype=float)

    try:
        from matplotlib.axes import Axes
    except Exception:
        paths = render_plots(
            bounds,
            tmp_path,
            ground_truth_values=truth,
            enforce_truth_coverage_for_plot=True,
        )
        assert paths == []
        return

    plotted_y: list[np.ndarray] = []
    original_plot = Axes.plot

    def _spy(self, *args, **kwargs):
        if len(args) >= 2:
            plotted_y.append(np.asarray(args[1], dtype=float))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", _spy)
    paths = render_plots(
        bounds,
        tmp_path,
        ground_truth_values=truth,
        enforce_truth_coverage_for_plot=True,
    )
    assert any(p.name == "bounds_interval.png" for p in paths)
    assert len(plotted_y) >= 3
    lower_draw = plotted_y[0]
    upper_draw = plotted_y[1]
    truth_draw = plotted_y[-1]
    assert np.all(lower_draw < truth_draw)
    assert np.all(truth_draw < upper_draw)
