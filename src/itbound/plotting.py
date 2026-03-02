from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def render_plots(bounds_df: pd.DataFrame, outdir: Path) -> list[Path]:
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

    if "valid_interval" in bounds_df.columns:
        valid = bounds_df["valid_interval"].to_numpy(dtype=bool)
    else:
        valid = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)

    interval_path = outdir / "bounds_interval.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(idx, lower, label="lower", color="tab:blue", linewidth=1.2)
    ax.plot(idx, upper, label="upper", color="tab:orange", linewidth=1.2)
    if np.any(valid):
        ax.fill_between(idx, lower, upper, where=valid, alpha=0.2, color="tab:green", label="valid interval")
    ax.set_title("Causal Bounds by Row")
    ax.set_xlabel("row index")
    ax.set_ylabel("bound value")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(interval_path, dpi=120)
    plt.close(fig)

    return [interval_path]


__all__ = ["render_plots"]
