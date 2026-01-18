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

try:
    from fbound.utils.plotting import DIVERGENCE_COLOR_MAP
except Exception:
    DIVERGENCE_COLOR_MAP = {
        "kth": "tab:cyan",
        "tight_kth": "tab:olive",
        "KL": "tab:green",
        "TV": "tab:red",
        "Hellinger": "tab:purple",
        "Chi2": "tab:brown",
        "JS": "tab:pink",
    }

try:
    from scipy.interpolate import UnivariateSpline
except Exception:
    UnivariateSpline = None

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception:
    lowess = None


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


def smooth_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "spline",
    smooth_grid_n: int = 500,
    window: int = 5,
    spline_k: int = 3,
    spline_s: float = -1.0,
    lowess_frac: float = 0.2,
    lowess_it: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.array([]), np.array([])

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    method = str(method).lower()

    if method == "none":
        return x, y

    if method == "moving_avg":
        window = max(1, int(window))
        if window <= 1 or x.size < 2:
            return x, y
        kernel = np.ones(window, dtype=np.float64) / float(window)
        y_s = np.convolve(y, kernel, mode="same")
        return x, y_s

    if method == "spline":
        if UnivariateSpline is None:
            return x, y
        if x.size <= int(spline_k):
            return x, y
        s_val = None if float(spline_s) < 0 else float(spline_s)
        grid_n = max(2, int(smooth_grid_n))
        x_grid = np.linspace(float(x.min()), float(x.max()), grid_n)

        def _eval_spline(s_val_inner):
            try:
                spline = UnivariateSpline(x, y, k=int(spline_k), s=s_val_inner)
                y_grid = spline(x_grid)
            except Exception:
                return None
            if not np.all(np.isfinite(y_grid)):
                return None
            return y_grid

        y_grid = _eval_spline(s_val)
        if y_grid is None and s_val is not None:
            y_grid = _eval_spline(None)
        if y_grid is None:
            return x, y
        return x_grid, y_grid

    if method == "lowess":
        if lowess is None:
            return x, y
        try:
            out = lowess(y, x, frac=float(lowess_frac), it=int(lowess_it), return_sorted=True)
            return out[:, 0], out[:, 1]
        except Exception:
            return x, y

    raise ValueError(f"Unknown smoothing method '{method}'.")


def _parse_plot_name(raw: str) -> Dict[str, Any]:
    base = os.path.basename(raw)
    name, _ = os.path.splitext(base)
    stamp = ""
    m = re.search(r"(\d{8}_\d{6})$", name)
    if m:
        stamp = m.group(1)
    kind = "ribbon"
    stat = ""
    with_ci = False
    smooth = False
    if "_width_" in name:
        kind = "width"
        m = re.search(r"width_(mean|median)_(ci|noci)_(smooth|raw)", name)
        if m:
            stat = m.group(1)
            with_ci = m.group(2) == "ci"
            smooth = m.group(3) == "smooth"
    elif "_coverage_" in name:
        kind = "coverage"
        m = re.search(r"coverage_(mean|median)_(smooth|raw)", name)
        if m:
            stat = m.group(1)
            smooth = m.group(2) == "smooth"
    return {
        "name": name,
        "stamp": stamp,
        "kind": kind,
        "stat": stat,
        "with_ci": with_ci,
        "smooth": smooth,
        "is_propensity": "propensity" in name,
    }


def _read_width_by_propensity(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    rows = _read_csv_table(
        path,
        {
            "method": str,
            "propensity": float,
            "width_mean": float,
            "width_median": float,
            "width_ci": float,
            "coverage_mean": float,
            "coverage_median": float,
        },
    )
    grouped = _group_by_method(rows)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for method, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: float(r.get("propensity", 0.0)))
        out[method] = {
            "propensity": np.array([r.get("propensity", float("nan")) for r in items_sorted], dtype=np.float64),
            "mean": np.array([r.get("width_mean", float("nan")) for r in items_sorted], dtype=np.float64),
            "median": np.array([r.get("width_median", float("nan")) for r in items_sorted], dtype=np.float64),
            "ci": np.array([r.get("width_ci", float("nan")) for r in items_sorted], dtype=np.float64),
            "coverage_mean": np.array([r.get("coverage_mean", float("nan")) for r in items_sorted], dtype=np.float64),
            "coverage_median": np.array([r.get("coverage_median", float("nan")) for r in items_sorted], dtype=np.float64),
        }
    return out


def _parse_figsize(raw: str, default: tuple[float, float]) -> tuple[float, float]:
    if not raw:
        return default
    try:
        fig_w, fig_h = [float(x.strip()) for x in raw.split(",")]
    except Exception:
        raise ValueError("--figsize must be in the format 'width,height' (e.g., 11,6).")
    return fig_w, fig_h


def _plot_width_by_propensity(
    *,
    width_stats: Dict[str, Dict[str, np.ndarray]],
    div_list: List[str],
    stat_key: str,
    with_ci: bool,
    smooth: bool,
    smooth_cfg: Dict[str, Any],
    title: str,
    ylabel: str,
    tick_labelsize: float,
    label_size: Optional[float],
    title_size: Optional[float],
    show_xlabel: bool,
    show_ylabel: bool,
    show_title: bool,
    show_legend: bool,
    figsize: tuple[float, float],
    out_path: str,
) -> None:
    plt.figure(figsize=figsize)
    ax = plt.gca()
    color_map = DIVERGENCE_COLOR_MAP
    for div in div_list:
        if div not in width_stats:
            continue
        data = width_stats[div]
        prop_grid = data["propensity"]
        w_center = data[stat_key]
        w_ci = data["ci"]
        if not np.isfinite(w_center).any():
            continue
        c = color_map.get(div, None)
        if smooth:
            x_center, w_center_s = smooth_xy(prop_grid, w_center, **smooth_cfg)
            x_ci, w_ci_s = smooth_xy(prop_grid, w_ci, **smooth_cfg)
            if x_center.size == 0:
                x_plot = prop_grid
                w_center_plot = w_center
                if x_ci.size > 0:
                    w_ci_plot = np.interp(x_plot, x_ci, w_ci_s)
                else:
                    w_ci_plot = w_ci
            else:
                x_plot = x_center
                w_center_plot = w_center_s
                if x_ci.size > 0:
                    w_ci_plot = np.interp(x_plot, x_ci, w_ci_s)
                else:
                    w_ci_plot = np.interp(x_plot, prop_grid, w_ci)
        else:
            x_plot = prop_grid
            w_center_plot = w_center
            w_ci_plot = w_ci
        w_ci_plot = np.clip(w_ci_plot, 0.0, None)
        mask = np.isfinite(x_plot) & np.isfinite(w_center_plot)
        if with_ci and np.isfinite(w_ci_plot).any():
            mask &= np.isfinite(w_ci_plot)
            ax.errorbar(
                x_plot[mask],
                w_center_plot[mask],
                yerr=w_ci_plot[mask],
                color=c,
                linewidth=1.8,
                elinewidth=0.8,
                capsize=2,
                alpha=0.8,
                label=div,
            )
        else:
            ax.plot(x_plot[mask], w_center_plot[mask], color=c, linewidth=2.0, label=div)

    ax.tick_params(axis="both", which="major", labelsize=float(tick_labelsize))
    if show_xlabel:
        if label_size is None:
            ax.set_xlabel("e(A=1|X)")
        else:
            ax.set_xlabel("e(A=1|X)", fontsize=label_size)
    if show_ylabel:
        if label_size is None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(ylabel, fontsize=label_size)
    if show_title:
        if title_size is None:
            ax.set_title(title)
        else:
            ax.set_title(title, fontsize=title_size)
    if show_legend:
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_coverage_by_propensity(
    *,
    width_stats: Dict[str, Dict[str, np.ndarray]],
    div_list: List[str],
    cov_key: str,
    smooth: bool,
    smooth_cfg: Dict[str, Any],
    title: str,
    tick_labelsize: float,
    label_size: Optional[float],
    title_size: Optional[float],
    show_xlabel: bool,
    show_ylabel: bool,
    show_title: bool,
    show_legend: bool,
    figsize: tuple[float, float],
    out_path: str,
) -> None:
    plt.figure(figsize=figsize)
    ax = plt.gca()
    color_map = DIVERGENCE_COLOR_MAP
    for div in div_list:
        if div not in width_stats:
            continue
        data = width_stats[div]
        prop_grid = data["propensity"]
        coverage = data[cov_key]
        if not np.isfinite(coverage).any():
            continue
        c = color_map.get(div, None)
        if smooth:
            x_cov, cov_s = smooth_xy(prop_grid, coverage, **smooth_cfg)
            if x_cov.size == 0:
                x_plot = prop_grid
                cov_plot = coverage
            else:
                x_plot = x_cov
                cov_plot = cov_s
        else:
            x_plot = prop_grid
            cov_plot = coverage
        cov_plot = np.clip(cov_plot, 0.0, 1.0)
        mask = np.isfinite(x_plot) & np.isfinite(cov_plot)
        if np.any(mask):
            ax.plot(x_plot[mask], cov_plot[mask], color=c, linewidth=2.0, label=div)

    ax.tick_params(axis="both", which="major", labelsize=float(tick_labelsize))
    if show_xlabel:
        if label_size is None:
            ax.set_xlabel("e(A=1|X)")
        else:
            ax.set_xlabel("e(A=1|X)", fontsize=label_size)
    if show_ylabel:
        if label_size is None:
            ax.set_ylabel("Coverage rate")
        else:
            ax.set_ylabel("Coverage rate", fontsize=label_size)
    if cov_key == "coverage_median":
        ax.set_ylim(0.0, 1.25)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    else:
        ax.set_ylim(0.0, 1.0)
    if show_title:
        if title_size is None:
            ax.set_title(title)
        else:
            ax.set_title(title, fontsize=title_size)
    if show_legend:
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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
    parser.add_argument(
        "--plot_name",
        type=str,
        default="",
        help=(
            "Plot name like plot_x0_ribbon_mc_eval_fixed_propensity_width_mean_ci_smooth_YYYYMMDD_HHMMSS "
            "(used to infer stamp and smoothed table)."
        ),
    )
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
    parser.add_argument("--title_size", type=float, default=0.0, help="Title font size (0 disables).")
    parser.add_argument("--label_size", type=float, default=0.0, help="Axis label font size (0 disables).")
    parser.add_argument("--figsize", type=str, default="11,6", help="Figure size as 'width,height'.")
    parser.add_argument("--xlim", type=str, default="", help="x-axis limits as 'min,max'.")
    parser.add_argument("--ylim", type=str, default="", help="y-axis limits as 'min,max'.")

    args = parser.parse_args()
    plot_spec = _parse_plot_name(args.plot_name) if args.plot_name else {}
    plot_kind = plot_spec.get("kind", "ribbon")
    plot_prefix = (
        "plot_x0_ribbon_mc_eval_fixed_propensity"
        if plot_spec.get("is_propensity", False)
        else "plot_x0_ribbon_mc_eval_fixed"
    )
    if args.plot_name and not args.stamp:
        if plot_spec.get("stamp"):
            args.stamp = plot_spec["stamp"]
        else:
            raise ValueError(f"--plot_name did not include a timestamp: {args.plot_name}")
    if args.plot_name and not args.summary:
        base_dir = args.artifact_dir or "experiments"
        candidate = os.path.join(base_dir, f"{plot_prefix}_summary_{args.stamp}.json")
        if os.path.exists(candidate):
            args.summary = candidate
    if args.plot_name and not args.smoothed_table:
        base_dir = args.artifact_dir or "experiments"
        candidates = [
            f"plot_x0_ribbon_mc_eval_fixed_propensity_smoothed_table_{args.stamp}.csv",
            f"plot_x0_ribbon_mc_eval_fixed_smoothed_table_{args.stamp}.csv",
        ]
        for fname in candidates:
            candidate = os.path.join(base_dir, fname)
            if os.path.exists(candidate):
                args.smoothed_table = candidate
                break
    inputs = _resolve_inputs(args)

    summary = inputs["summary"]
    artifact = inputs["artifact"]
    stamp = inputs["stamp"]
    table_path = inputs["table_path"]
    smoothed_path = inputs["smoothed_path"]
    outdir = inputs["outdir"]

    output_path = None
    if args.plot_name:
        output_path = os.path.join(outdir, f"load_{plot_spec['name']}.png")

    if plot_kind in {"width", "coverage"}:
        base_dir = args.artifact_dir or outdir or os.path.dirname(inputs["summary_path"] or inputs["artifact_path"] or "") or "experiments"
        width_table_path = os.path.join(base_dir, f"{plot_prefix}_width_by_propensity_{stamp}.csv")
        if not os.path.exists(width_table_path):
            raise FileNotFoundError(f"Width table not found: {width_table_path}")
        width_stats = _read_width_by_propensity(width_table_path)
        if not width_stats:
            raise RuntimeError("Width table is empty; cannot redraw width/coverage plot.")

        div_list: List[str] = []
        if summary is not None:
            div_list = summary.get("divergences", []) or []
        if not div_list and artifact is not None:
            div_list = artifact.get("divergences", []) or []
        if not div_list:
            div_list = sorted(width_stats.keys())

        smooth_method = "lowess"
        smooth_grid_n = 500
        smooth_window = 5
        spline_k = 3
        spline_s = -1.0
        lowess_frac = 0.2
        lowess_it = 1
        if summary is not None:
            smooth_method = summary.get("smooth_method", smooth_method)
            smooth_grid_n = summary.get("smooth_grid_n", smooth_grid_n)
            smooth_window = summary.get("smooth_window", smooth_window)
            spline_k = summary.get("spline_k", spline_k)
            spline_s = summary.get("spline_s", spline_s)
            lowess_frac = summary.get("lowess_frac", lowess_frac)
            lowess_it = summary.get("lowess_it", lowess_it)
        if artifact is not None:
            args_meta = artifact.get("args", {})
            smooth_method = args_meta.get("smooth_method", smooth_method)
            smooth_grid_n = args_meta.get("smooth_grid_n", smooth_grid_n)
            smooth_window = args_meta.get("smooth_window", smooth_window)
            spline_k = args_meta.get("spline_k", spline_k)
            spline_s = args_meta.get("spline_s", spline_s)
            lowess_frac = args_meta.get("lowess_frac", lowess_frac)
            lowess_it = args_meta.get("lowess_it", lowess_it)

        smooth_cfg = {
            "method": smooth_method,
            "smooth_grid_n": int(smooth_grid_n),
            "window": int(smooth_window),
            "spline_k": int(spline_k),
            "spline_s": float(spline_s),
            "lowess_frac": float(lowess_frac),
            "lowess_it": int(lowess_it),
        }

        if plot_spec.get("stat") not in {"mean", "median"}:
            raise ValueError(f"Unrecognized stat in plot_name: {args.plot_name}")

        show_title = not args.no_title
        show_xlabel = not args.no_xlabel
        show_ylabel = not args.no_ylabel
        show_legend = not args.no_legend
        label_size = float(args.label_size) if args.label_size and args.label_size > 0 else None
        title_size = float(args.title_size) if args.title_size and args.title_size > 0 else None

        figsize_default = (7.0, 4.0)
        figsize = figsize_default if args.figsize == "11,6" else _parse_figsize(args.figsize, figsize_default)
        out_path = output_path or _name_with_suffix(outdir, "load_plot_x0_ribbon_mc_eval_fixed", "png", stamp)

        if plot_kind == "width":
            stat_key = plot_spec["stat"]
            with_ci = bool(plot_spec.get("with_ci", False))
            smooth = bool(plot_spec.get("smooth", False))
            ylabel = "Median interval width" if stat_key == "median" else "Mean interval width"
            default_title = (
                "Median width vs propensity by divergence"
                if stat_key == "median"
                else "Mean width vs propensity by divergence"
            )
            title = args.title or default_title
            _plot_width_by_propensity(
                width_stats=width_stats,
                div_list=div_list,
                stat_key=stat_key,
                with_ci=with_ci,
                smooth=smooth,
                smooth_cfg=smooth_cfg,
                title=title,
                ylabel=ylabel,
                tick_labelsize=args.tick_labelsize,
                label_size=label_size,
                title_size=title_size,
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel,
                show_title=show_title,
                show_legend=show_legend,
                figsize=figsize,
                out_path=out_path,
            )
        else:
            cov_key = "coverage_mean" if plot_spec["stat"] == "mean" else "coverage_median"
            cov_label = "Mean coverage" if plot_spec["stat"] == "mean" else "Median coverage"
            default_title = f"{cov_label} vs propensity by divergence"
            title = args.title or default_title
            _plot_coverage_by_propensity(
                width_stats=width_stats,
                div_list=div_list,
                cov_key=cov_key,
                smooth=bool(plot_spec.get("smooth", False)),
                smooth_cfg=smooth_cfg,
                title=title,
                tick_labelsize=args.tick_labelsize,
                label_size=label_size,
                title_size=title_size,
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel,
                show_title=show_title,
                show_legend=show_legend,
                figsize=figsize,
                out_path=out_path,
            )
        print(f"Saved plot to {out_path}")
        raise SystemExit(0)

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
    label_size = float(args.label_size) if args.label_size and args.label_size > 0 else None
    title_size = float(args.title_size) if args.title_size and args.title_size > 0 else None
    if not args.no_xlabel:
        if label_size is None:
            plt.xlabel("Baseline covariate x0")
        else:
            plt.xlabel("Baseline covariate x0", fontsize=label_size)
    if not args.no_ylabel:
        ylabel = r"Conditional causal mean $\mu_1(x_0)=\mathbb{E}[Y\mid \mathrm{do}(A=1),X_0=x_0]$"
        if label_size is None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(ylabel, fontsize=label_size)
    if not args.no_title:
        if args.title:
            title = args.title
        else:
            title = "Conditional causal mean vs. baseline covariate: bounds track the true curve"
        if title_size is None:
            plt.title(title)
        else:
            plt.title(title, fontsize=title_size)
    if not args.no_legend:
        plt.legend(frameon=True)

    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    if output_path:
        fig_path = output_path
    else:
        fig_path = _name_with_suffix(outdir, "load_plot_x0_ribbon_mc_eval_fixed", "png", stamp)
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {fig_path}")
