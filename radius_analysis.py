"""
Plot divergence radius B_f(e) for multiple f-divergences over e in (0, 1).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.fbound.utils.divergences import get_divergence


def _set_mpl_cache_dir() -> None:
    """Avoid matplotlib cache permission issues by defaulting to /tmp."""
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
    if not os.environ.get("XDG_CACHE_HOME"):
        os.environ["XDG_CACHE_HOME"] = "/tmp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot B_f(e) for supported divergences.")
    parser.add_argument("--out", type=str, default="radius_analysis.png", help="Output PNG path.")
    parser.add_argument("--n", type=int, default=2000, help="Number of e points.")
    parser.add_argument("--e_min", type=float, default=1e-4, help="Minimum e value.")
    parser.add_argument("--e_max", type=float, default=0.99999, help="Maximum e value.")
    args = parser.parse_args()

    _set_mpl_cache_dir()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = np.linspace(args.e_min, args.e_max, int(args.n))
    divergences = ["KL", "Hellinger", "Chi2", "TV", "JS"]

    plt.figure(figsize=(7, 4))
    for name in divergences:
        div = get_divergence(name)
        b = div.B_numpy(e.astype(np.float64, copy=False))
        plt.plot(e, b, label=name)

    plt.xlabel("e")
    plt.ylabel("B_f(e)")
    plt.title("Divergence radii B_f(e) for e in (0, 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
