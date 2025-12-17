"""
Reload plot1 artifacts and regenerate plots without rerunning simulations.
Usage:
    python load_plot1.py --artifact plot1_artifacts.pkl
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Reload plot1 artifacts and redraw plots.")
    parser.add_argument("--artifact", type=str, default="plot1_artifacts.pkl", help="path to artifacts pickle")
    args = parser.parse_args()

    with open(args.artifact, "rb") as f:
        art = pickle.load(f)

    bins_avg = art["bins_avg"]
    requested = art.get("requested_methods", ["combined", "cluster"])

    # Plot A: truth + averaged ribbons
    figA, axA = plt.subplots(figsize=(7.0, 4.0))
    ribbon_methods = [m for m in ["combined", "cluster"] if m in requested]
    colors_cycle = {"combined": "tab:blue", "cluster": "tab:orange"}
    for method in ribbon_methods:
        sub = bins_avg[bins_avg["method"] == method].sort_values("ehat_center")
        axA.fill_between(sub["ehat_center"], sub["L_mean"], sub["U_mean"], alpha=0.25, color=colors_cycle.get(method, None), label=f"{method} ribbon")
    truth_bins = bins_avg.groupby("bin", as_index=False).agg(ehat_center=("ehat_center","mean"),
                                                             truth_mean=("truth_mean","mean"))
    axA.plot(truth_bins["ehat_center"], truth_bins["truth_mean"], color="k", label="truth (avg)")
    axA.set_xlabel("Propensity estimate ê(x)")
    axA.set_ylabel("E[Y | do(A=1), X]")
    axA.set_title("Truth + averaged ribbons (reloaded)")
    axA.legend()
    figA.tight_layout()
    figA.savefig("plot1_propensity_binned_ribbons.png", dpi=200)

    # Plot B: coverage
    figB, axB = plt.subplots(figsize=(7.0, 4.0))
    for method in requested:
        sub = bins_avg[bins_avg["method"] == method].sort_values("ehat_center")
        if "coverage_se" in sub:
            axB.errorbar(sub["ehat_center"], sub["coverage"], yerr=sub["coverage_se"], marker="o", linestyle="-", capsize=3, label=method)
        else:
            axB.plot(sub["ehat_center"], sub["coverage"], marker="o", linestyle="-", label=method)
    axB.set_xlabel("Propensity estimate ê(x)")
    axB.set_ylabel("Coverage within bin")
    axB.set_ylim(0.0, 1.05)
    axB.set_title("Coverage vs propensity (reloaded)")
    axB.legend()
    figB.tight_layout()
    figB.savefig("plot1_propensity_binned_coverage.png", dpi=200)

    # Plot C: width
    figC, axC = plt.subplots(figsize=(7.0, 4.0))
    for method in requested:
        sub = bins_avg[bins_avg["method"] == method].sort_values("ehat_center")
        if "width_se" in sub:
            axC.errorbar(sub["ehat_center"], sub["width"], yerr=sub["width_se"], marker="o", linestyle="-", capsize=3, label=method)
        else:
            axC.plot(sub["ehat_center"], sub["width"], marker="o", linestyle="-", label=method)
    axC.set_xlabel("Propensity estimate ê(x)")
    axC.set_ylabel("Mean width within bin")
    axC.set_title("Width vs propensity (reloaded)")
    axC.legend()
    figC.tight_layout()
    figC.savefig("plot1_propensity_binned_width.png", dpi=200)

    print("Reloaded plots saved: plot1_propensity_binned_ribbons.png, _coverage.png, _width.png")


if __name__ == "__main__":
    main()
