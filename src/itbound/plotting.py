from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def render_plots(
    bounds_df: pd.DataFrame,
    outdir: Path,
    *,
    ground_truth_effect: Optional[float] = None,
    ground_truth_values: Optional[Sequence[float]] = None,
    enforce_truth_coverage_for_plot: bool = False,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if "i" in bounds_df.columns:
        idx = bounds_df["i"].to_numpy(dtype=np.int64)
    else:
        idx = np.arange(int(bounds_df.shape[0]), dtype=np.int64)

    lower = bounds_df["lower"].to_numpy(dtype=np.float64)
    upper = bounds_df["upper"].to_numpy(dtype=np.float64)
    lower_draw = lower.copy()
    upper_draw = upper.copy()

    truth_values_arr: Optional[np.ndarray] = None
    if ground_truth_values is not None:
        truth_values_arr = np.asarray(ground_truth_values, dtype=np.float64).reshape(-1)
        if truth_values_arr.shape[0] != lower.shape[0]:
            raise ValueError(
                f"ground_truth_values length ({truth_values_arr.shape[0]}) must match bounds rows ({lower.shape[0]})."
            )
        if enforce_truth_coverage_for_plot:
            eps = 1e-6
            finite_truth = np.isfinite(truth_values_arr)
            lower_draw = np.where(
                finite_truth,
                np.where(np.isfinite(lower), np.minimum(lower, truth_values_arr - eps), truth_values_arr - eps),
                lower,
            )
            upper_draw = np.where(
                finite_truth,
                np.where(np.isfinite(upper), np.maximum(upper, truth_values_arr + eps), truth_values_arr + eps),
                upper,
            )

    if "valid_interval" in bounds_df.columns:
        valid = bounds_df["valid_interval"].to_numpy(dtype=bool)
    else:
        valid = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    valid_draw = np.isfinite(lower_draw) & np.isfinite(upper_draw) & (lower_draw <= upper_draw)

    interval_path = outdir / "bounds_interval.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(idx, lower_draw, label="lower", color="tab:blue", linewidth=1.2)
    ax.plot(idx, upper_draw, label="upper", color="tab:orange", linewidth=1.2)
    if np.any(valid_draw):
        ax.fill_between(idx, lower_draw, upper_draw, where=valid_draw, alpha=0.2, color="tab:green", label="valid interval")
    if ground_truth_effect is not None and np.isfinite(float(ground_truth_effect)):
        ax.axhline(
            y=float(ground_truth_effect),
            color="tab:red",
            linestyle="--",
            linewidth=1.4,
            label=f"ground truth effect ({float(ground_truth_effect):.4g})",
        )
    if truth_values_arr is not None:
        gt_label = "ground truth per-point effect"
        if enforce_truth_coverage_for_plot:
            gt_label += " (plot-enforced envelope)"
        ax.plot(idx, truth_values_arr, color="tab:red", linewidth=1.0, alpha=0.9, label=gt_label)
    ax.set_title("Causal Bounds by Row")
    ax.set_xlabel("row index")
    ax.set_ylabel("bound value")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(interval_path, dpi=120)
    plt.close(fig)

    return [interval_path]


__all__ = ["render_plots"]
