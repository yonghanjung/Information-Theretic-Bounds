"""
Reload plot_x0_ribbon_mc_eval_fixed artifacts and redraw plots without rerunning simulations.
Accepts .pkl/.json/.csv inputs and regenerates the ribbon plot.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _infer_stamp(path: str, pattern: str) -> str:
    m = re.search(pattern, os.path.basename(path))
    if m:
        return m.group(1)
    return ""


def _name_with_suffix(outdir: str, base: str, ext: str, stamp: str) -> str:
    fname = f"{base}_{stamp}.{ext}" if stamp else f"{base}.{ext}"
    return os.path.join(outdir, fname)


def _read_csv_table(path: str, fields: Dict[str, type]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: Dict[str, Any] = {}
            for key, cast in fields.items():
                if key not in row:
                    continue
                if cast is str:
                    out[key] = row[key]
                else:
                    try:
                        out[key] = cast(row[key])
                    except Exception:
                        out[key] = float("nan")
            rows.append(out)
    return rows


def _group_by_method(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        method = row.get("method", "")
        if not method:
            continue
        out.setdefault(method, []).append(row)
    return out


def _build_aggregated_from_smoothed(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group_by_method(rows)
    aggregated = []
    for method, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: float(r.get("X0", 0.0)))
        x_s = np.array([r.get("X0", float("nan")) for r in items_sorted], dtype=np.float32)
        l_s = np.array([r.get("lower", float("nan")) for r in items_sorted], dtype=np.float32)
        u_s = np.array([r.get("upper", float("nan")) for r in items_sorted], dtype=np.float32)
        theta_s = np.array([r.get("theta", float("nan")) for r in items_sorted], dtype=np.float32)
        aggregated.append(
            {
                "div": method,
                "x_s": x_s,
                "l_s": l_s,
                "u_s": u_s,
                "theta_s": theta_s,
            }
        )
    return aggregated


def _load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_artifact(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    summary = None
    artifact = None
    stamp = args.stamp or ""
    artifact_path = args.artifact or ""
    summary_path = args.summary or ""
    table_path = args.table or ""
    smoothed_path = args.smoothed_table or ""

    if summary_path:
        summary = _load_summary(summary_path)
        files = summary.get("files", {})
        if not artifact_path:
            artifact_path = files.get("artifacts_pkl", "") or ""
        if not table_path:
            table_path = files.get("table_csv", "") or ""
        if not smoothed_path:
            smoothed_path = files.get("smoothed_table_csv", "") or ""
        if not stamp:
            stamp = summary.get("timestamp", "") or _infer_stamp(
                summary_path, r"plot_x0_ribbon_mc_eval_fixed_summary_(\d{8}_\d{6})"
            )

    if artifact_path:
        artifact = _load_artifact(artifact_path)
        if not stamp:
            stamp = artifact.get("timestamp", "") or _infer_stamp(
                artifact_path, r"plot_x0_ribbon_mc_eval_fixed_artifacts_(\d{8}_\d{6})"
            )

    if not smoothed_path and stamp:
        base_dir = args.artifact_dir or args.outdir or os.path.dirname(summary_path or artifact_path) or "experiments"
        candidate = os.path.join(base_dir, f"plot_x0_ribbon_mc_eval_fixed_smoothed_table_{stamp}.csv")
        if os.path.exists(candidate):
            smoothed_path = candidate

    if not table_path and stamp:
        base_dir = args.artifact_dir or args.outdir or os.path.dirname(summary_path or artifact_path) or "experiments"
        candidate = os.path.join(base_dir, f"plot_x0_ribbon_mc_eval_fixed_table_{stamp}.csv")
        if os.path.exists(candidate):
            table_path = candidate

    outdir = args.outdir
    if not outdir:
        outdir = os.path.dirname(summary_path or artifact_path or smoothed_path or table_path) or "experiments"

    return {
        "summary": summary,
        "artifact": artifact,
        "stamp": stamp,
        "artifact_path": artifact_path,
        "summary_path": summary_path,
        "table_path": table_path,
        "smoothed_path": smoothed_path,
        "outdir": outdir,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reload plot_x0_ribbon_mc_eval_fixed artifacts and redraw plots.")
    parser.add_argument("--stamp", type=str, default="", help="Timestamp suffix used in output filenames.")
    parser.add_argument("--artifact", type=str, default="", help="Path to artifacts pickle.")
    parser.add_argument("--summary", type=str, default="", help="Path to summary JSON.")
    parser.add_argument("--table", type=str, default="", help="Path to raw table CSV.")
    parser.add_argument("--smoothed_table", type=str, default="", help="Path to smoothed table CSV.")
    parser.add_argument("--artifact_dir", type=str, default="", help="Directory for --stamp lookup.")
    parser.add_argument("--outdir", type=str, default="", help="Output directory for plot.")
    parser.add_argument("--plot_raw_points", action="store_true", help="Overlay raw lower/upper points.")
    parser.add_argument("--title", type=str, default="", help="Override plot title.")
    parser.add_argument("--no_xlabel", action="store_true", help="Disable x-axis label.")
    parser.add_argument("--no_ylabel", action="store_true", help="Disable y-axis label.")
    parser.add_argument("--no_title", action="store_true", help="Disable plot title.")
    parser.add_argument("--no_legend", action="store_true", help="Disable legend.")
    parser.add_argument("--tick_labelsize", type=float, default=14.0, help="Tick label font size.")
    parser.add_argument("--figsize", type=str, default="11,6", help="Figure size as 'width,height'.")
    parser.add_argument("--xlim", type=str, default="", help="x-axis limits as 'min,max'.")
    parser.add_argument("--ylim", type=str, default="", help="y-axis limits as 'min,max'.")

    args = parser.parse_args()
    inputs = _resolve_inputs(args)

    summary = inputs["summary"]
    artifact = inputs["artifact"]
    stamp = inputs["stamp"]
    table_path = inputs["table_path"]
    smoothed_path = inputs["smoothed_path"]
    outdir = inputs["outdir"]

    if artifact is None and not smoothed_path:
        raise RuntimeError("Provide --artifact, --summary, or --smoothed_table to redraw the plot.")

    aggregated_results = None
    div_list: List[str] = []
    stat_label = ""
    struct_label = ""

    if artifact is not None:
        aggregated_results = artifact.get("aggregated_results", None)
        div_list = artifact.get("divergences", []) or []
        args_meta = artifact.get("args", {})
        stat_label = args_meta.get("stat", "")
        struct_label = args_meta.get("structural_type", "")

    if aggregated_results is None and smoothed_path:
        smoothed_rows = _read_csv_table(
            smoothed_path,
            {
                "method": str,
                "X0": float,
                "theta": float,
                "lower": float,
                "upper": float,
                "width": float,
            },
        )
        aggregated_results = _build_aggregated_from_smoothed(smoothed_rows)
        if not div_list:
            div_list = sorted({r["method"] for r in smoothed_rows})

    if summary is not None:
        stat_label = summary.get("stat", stat_label)
        struct_label = summary.get("structural_type", struct_label)
        if not div_list:
            div_list = summary.get("divergences", []) or div_list

    raw_rows: Optional[List[Dict[str, Any]]] = None
    if args.plot_raw_points and table_path:
        raw_rows = _read_csv_table(
            table_path,
            {
                "method": str,
                "X0": float,
                "lower": float,
                "upper": float,
            },
        )

    if not aggregated_results:
        raise RuntimeError("Unable to construct aggregated results for plotting.")

    x_min = None
    x_max = None
    y_min = None
    y_max = None
    if args.xlim:
        try:
            x_min, x_max = [float(x.strip()) for x in args.xlim.split(",")]
        except Exception:
            raise ValueError("--xlim must be in the format 'min,max'.")
    if args.ylim:
        try:
            y_min, y_max = [float(y.strip()) for y in args.ylim.split(",")]
        except Exception:
            raise ValueError("--ylim must be in the format 'min,max'.")

    res = aggregated_results[0]
    x_s = np.asarray(res.get("x_s", []), dtype=np.float32)
    l_s = np.asarray(res.get("l_s", []), dtype=np.float32)
    u_s = np.asarray(res.get("u_s", []), dtype=np.float32)
    theta_s = np.asarray(res.get("theta_s", []), dtype=np.float32)
    mask = np.isfinite(x_s) & np.isfinite(l_s) & np.isfinite(u_s) & np.isfinite(theta_s)
    if x_min is not None and x_max is not None:
        mask &= (x_s >= x_min) & (x_s <= x_max)
    if y_min is not None and y_max is not None:
        mask &= (l_s >= y_min) & (l_s <= y_max)
        mask &= (u_s >= y_min) & (u_s <= y_max)
        mask &= (theta_s >= y_min) & (theta_s <= y_max)
    x_s = x_s[mask]
    l_s = l_s[mask]
    u_s = u_s[mask]
    theta_s = theta_s[mask]
    if x_s.size == 0:
        raise RuntimeError("No curve data found to plot.")

    try:
        fig_w, fig_h = [float(x.strip()) for x in args.figsize.split(",")]
    except Exception:
        raise ValueError("--figsize must be in the format 'width,height' (e.g., 11,6).")
    plt.figure(figsize=(fig_w, fig_h))
    plt.plot(
        x_s,
        l_s,
        color="tab:blue",
        linewidth=3.0,
        linestyle="-",
        label="Lower bound",
        zorder=3,
    )
    plt.plot(
        x_s,
        u_s,
        color="tab:green",
        linewidth=3.0,
        linestyle="-",
        label="Upper bound",
        zorder=3,
    )
    plt.plot(
        x_s,
        theta_s,
        color="k",
        linewidth=2.8,
        linestyle="--",
        label="Ground truth",
        zorder=4,
    )

    plt.grid(True, alpha=0.25)
    plt.tick_params(axis="both", which="major", labelsize=float(args.tick_labelsize))
    if not args.no_xlabel:
        plt.xlabel("Baseline covariate x0")
    if not args.no_ylabel:
        plt.ylabel(
            r"Conditional causal mean $\mu_1(x_0)=\mathbb{E}[Y\mid \mathrm{do}(A=1),X_0=x_0]$"
        )
    if not args.no_title:
        if args.title:
            title = args.title
        else:
            title = "Conditional causal mean vs. baseline covariate: bounds track the true curve"
        plt.title(title)
    if not args.no_legend:
        plt.legend(frameon=True)

    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fig_path = _name_with_suffix(outdir, "load_plot_x0_ribbon_mc_eval_fixed", "png", stamp)
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {fig_path}")
