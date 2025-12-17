"""
Reload plot2 artifacts and regenerate coverage/width plots without rerunning MC.
Usage:
    python load_plot2.py --artifact plot2_artifacts.pkl
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Reload plot2 artifacts and redraw plots.")
    parser.add_argument("--artifact", type=str, default="plot2_artifacts.pkl", help="path to artifacts pickle")
    args = parser.parse_args()

    with open(args.artifact, "rb") as f:
        art = pickle.load(f)

    agg = art["agg"]
    div = art.get("divergence", "method")

    fig_cov, ax_cov = plt.subplots(figsize=(7.0, 4.0))
    ax_cov.errorbar(agg["n"], agg["cov_d_mean"], yerr=agg["cov_d_se"], marker="o", capsize=3, label="debiased")
    ax_cov.errorbar(agg["n"], agg["cov_n_mean"], yerr=agg["cov_n_se"], marker="o", capsize=3, label="naive")
    ax_cov.set_ylim(0.0, 1.05)
    ax_cov.set_xlabel("Sample size n")
    ax_cov.set_ylabel("Coverage")
    ax_cov.set_title(f"Coverage vs n ({div}) [reloaded]")
    ax_cov.legend()
    fig_cov.tight_layout()
    fig_cov.savefig("plot2_mc_ablation_coverage.png", dpi=200)

    fig_wid, ax_wid = plt.subplots(figsize=(7.0, 4.0))
    ax_wid.errorbar(agg["n"], agg["wid_d_mean"], yerr=agg["wid_d_se"], marker="o", capsize=3, label="debiased")
    ax_wid.errorbar(agg["n"], agg["wid_n_mean"], yerr=agg["wid_n_se"], marker="o", capsize=3, label="naive")
    ax_wid.set_xlabel("Sample size n")
    ax_wid.set_ylabel("Mean interval width")
    ax_wid.set_title(f"Width vs n ({div}) [reloaded]")
    ax_wid.legend()
    fig_wid.tight_layout()
    fig_wid.savefig("plot2_mc_ablation_width.png", dpi=200)

    print("Reloaded plots saved: plot2_mc_ablation_coverage.png and plot2_mc_ablation_width.png")


if __name__ == "__main__":
    main()
