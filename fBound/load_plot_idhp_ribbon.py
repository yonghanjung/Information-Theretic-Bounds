"""
Reload plot_idhp_ribbon artifacts and redraw plots without rerunning simulations.
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

BASE_NAME_X0 = "plot_idhp_ribbon"
BASE_NAME_PROP = "plot_idhp_ribbon_propensity"


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


def _build_aggregated_from_raw(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group_by_method(rows)
    aggregated = []
    for method, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: float(r.get("X0", 0.0)))
        x_raw = np.array([r.get("X0", float("nan")) for r in items_sorted], dtype=np.float32)
        l_raw = np.array([r.get("lower", float("nan")) for r in items_sorted], dtype=np.float32)
        u_raw = np.array([r.get("upper", float("nan")) for r in items_sorted], dtype=np.float32)
        theta_raw = np.array([r.get("theta", float("nan")) for r in items_sorted], dtype=np.float32)
        aggregated.append(
            {
                "div": method,
                "x_raw": x_raw,
                "l_raw": l_raw,
                "u_raw": u_raw,
                "theta_raw": theta_raw,
                "x_s": x_raw,
                "l_s": l_raw,
                "u_s": u_raw,
                "theta_s": theta_raw,
            }
        )
    return aggregated


def _inject_raw_from_table(aggregated: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    grouped = _group_by_method(rows)
    for res in aggregated:
        if "x_raw" in res and "l_raw" in res and "u_raw" in res:
            continue
        method = res.get("div", "")
        items = grouped.get(method, [])
        if not items:
            continue
        items_sorted = sorted(items, key=lambda r: float(r.get("X0", 0.0)))
        res["x_raw"] = np.array([r.get("X0", float("nan")) for r in items_sorted], dtype=np.float32)
        res["l_raw"] = np.array([r.get("lower", float("nan")) for r in items_sorted], dtype=np.float32)
        res["u_raw"] = np.array([r.get("upper", float("nan")) for r in items_sorted], dtype=np.float32)
        res["theta_raw"] = np.array([r.get("theta", float("nan")) for r in items_sorted], dtype=np.float32)


def _load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_artifact(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _has_stamp_files(base_dir: str, base_name: str, stamp: str) -> bool:
    candidates = [
        f"{base_name}_summary_{stamp}.json",
        f"{base_name}_artifacts_{stamp}.pkl",
        f"{base_name}_smoothed_table_{stamp}.csv",
        f"{base_name}_table_{stamp}.csv",
    ]
    return any(os.path.exists(os.path.join(base_dir, fname)) for fname in candidates)


def _path_matches_base(path: str, base_name: str) -> bool:
    if not path:
        return False
    return base_name in os.path.basename(path)


def _axis_to_base(axis: str) -> str:
    return BASE_NAME_PROP if axis == "propensity" else BASE_NAME_X0


def _resolve_base_name(args: argparse.Namespace) -> str:
    for path in (args.summary, args.artifact, args.table, args.smoothed_table):
        if not path:
            continue
        base = os.path.basename(path)
        if BASE_NAME_PROP in base:
            return BASE_NAME_PROP
        if BASE_NAME_X0 in base:
            return BASE_NAME_X0
    if args.stamp:
        base_dir = args.artifact_dir or args.outdir or "experiments"
        if _has_stamp_files(base_dir, BASE_NAME_X0, args.stamp):
            return BASE_NAME_X0
        if _has_stamp_files(base_dir, BASE_NAME_PROP, args.stamp):
            return BASE_NAME_PROP
    return BASE_NAME_X0


def _resolve_inputs(args: argparse.Namespace, base_name: str) -> Dict[str, Any]:
    summary = None
    artifact = None
    stamp = args.stamp or ""
    artifact_path = args.artifact or ""
    summary_path = args.summary or ""
    table_path = args.table or ""
    smoothed_path = args.smoothed_table or ""

    if summary_path and not _path_matches_base(summary_path, base_name):
        summary_path = ""
    if artifact_path and not _path_matches_base(artifact_path, base_name):
        artifact_path = ""
    if table_path and not _path_matches_base(table_path, base_name):
        table_path = ""
    if smoothed_path and not _path_matches_base(smoothed_path, base_name):
        smoothed_path = ""

    if not summary_path and stamp:
        base_dir = args.artifact_dir or args.outdir or "experiments"
        candidate = os.path.join(base_dir, f"{base_name}_summary_{stamp}.json")
        if os.path.exists(candidate):
            summary_path = candidate

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
                summary_path, rf"{re.escape(base_name)}_summary_(\d{{8}}_\d{{6}})"
            )

    if artifact_path:
        artifact = _load_artifact(artifact_path)
        if not stamp:
            stamp = artifact.get("timestamp", "") or _infer_stamp(
                artifact_path, rf"{re.escape(base_name)}_artifacts_(\d{{8}}_\d{{6}})"
            )

    if not smoothed_path and stamp:
        base_dir = args.artifact_dir or args.outdir or os.path.dirname(summary_path or artifact_path) or "experiments"
        candidate = os.path.join(base_dir, f"{base_name}_smoothed_table_{stamp}.csv")
        if os.path.exists(candidate):
            smoothed_path = candidate

    if not table_path and stamp:
        base_dir = args.artifact_dir or args.outdir or os.path.dirname(summary_path or artifact_path) or "experiments"
        candidate = os.path.join(base_dir, f"{base_name}_table_{stamp}.csv")
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


def _resolve_axis_label(summary: Optional[Dict[str, Any]], artifact: Optional[Dict[str, Any]], base_name: str) -> str:
    eval_axis = ""
    args_meta: Dict[str, Any] = {}
    if summary is not None:
        eval_axis = summary.get("eval_axis", "") or eval_axis
        args_meta = summary.get("args", {}) or args_meta
    if artifact is not None:
        eval_axis = artifact.get("eval_axis", "") or eval_axis
        args_meta = artifact.get("args", {}) or args_meta
    if not eval_axis:
        eval_axis = "propensity" if "propensity" in base_name else "x0"
    if eval_axis == "propensity":
        return "e(A=1|X)"
    eval_dim = int(args_meta.get("eval_dim", 0))
    return f"X{eval_dim}"


def _select_result(aggregated: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    if not aggregated:
        raise RuntimeError("No aggregated results found.")
    if not method:
        return aggregated[0]
    for res in aggregated:
        if res.get("div", "") == method:
            return res
    raise RuntimeError(f"Method '{method}' not found in aggregated results.")


def _extract_curve(res: Dict[str, Any], kind: str) -> Dict[str, np.ndarray]:
    suffix = "raw" if kind == "raw" else "s"
    x_key = f"x_{suffix}"
    l_key = f"l_{suffix}"
    u_key = f"u_{suffix}"
    theta_key = f"theta_{suffix}"
    x = np.asarray(res.get(x_key, []), dtype=np.float32)
    l = np.asarray(res.get(l_key, []), dtype=np.float32)
    u = np.asarray(res.get(u_key, []), dtype=np.float32)
    theta = np.asarray(res.get(theta_key, []), dtype=np.float32)
    if x.size == 0 or l.size == 0 or u.size == 0 or theta.size == 0:
        raise RuntimeError(f"Missing {kind} curve data for plotting.")
    return {"x": x, "l": l, "u": u, "theta": theta}


def _apply_limits(
    x: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    xlim: Optional[tuple[float, float]],
    ylim: Optional[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(l) & np.isfinite(u) & np.isfinite(theta)
    if xlim is not None:
        x_min, x_max = xlim
        mask &= (x >= x_min) & (x <= x_max)
    if ylim is not None:
        y_min, y_max = ylim
        mask &= (l >= y_min) & (l <= y_max)
        mask &= (u >= y_min) & (u <= y_max)
        mask &= (theta >= y_min) & (theta <= y_max)
    return x[mask], l[mask], u[mask], theta[mask]


def _parse_limits(raw: str, label: str) -> Optional[tuple[float, float]]:
    if not raw:
        return None
    try:
        low, high = [float(x.strip()) for x in raw.split(",")]
    except Exception:
        raise ValueError(f"{label} must be in the format 'min,max'.")
    return low, high


def _plot_curve(
    *,
    x: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    bound_label: str,
    color: str,
    axis_label: str,
    title: str,
    tick_labelsize: float,
    figsize: tuple[float, float],
    xlim: Optional[tuple[float, float]],
    ylim: Optional[tuple[float, float]],
    no_xlabel: bool,
    no_ylabel: bool,
    no_title: bool,
    no_legend: bool,
    raw_points: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    plt.figure(figsize=figsize)
    plt.fill_between(x, l, u, alpha=0.2, color=color, label=bound_label)
    plt.plot(x, l, color=color, alpha=0.7, linewidth=1.0)
    plt.plot(x, u, color=color, alpha=0.7, linewidth=1.0)
    plt.plot(x, theta, color="k", linewidth=1.5, label="Truth")

    if raw_points is not None:
        x_r = raw_points.get("x")
        l_r = raw_points.get("l")
        u_r = raw_points.get("u")
        if x_r is not None and l_r is not None and u_r is not None:
            mask = np.isfinite(x_r) & np.isfinite(l_r) & np.isfinite(u_r)
            if np.any(mask):
                plt.scatter(x_r[mask], l_r[mask], color=color, alpha=0.2, s=8)
                plt.scatter(x_r[mask], u_r[mask], color=color, alpha=0.2, s=8)

    plt.grid(True, alpha=0.25)
    plt.tick_params(axis="both", which="major", labelsize=float(tick_labelsize))
    if not no_xlabel:
        plt.xlabel(axis_label)
    if not no_ylabel:
        plt.ylabel("E[Y | do(A=1), X]")
    if not no_title:
        plt.title(title)
    if not no_legend:
        plt.legend(frameon=True)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()


def _run_for_base(args: argparse.Namespace, base_name: str) -> bool:
    inputs = _resolve_inputs(args, base_name)
    summary = inputs["summary"]
    artifact = inputs["artifact"]
    stamp = inputs["stamp"]
    table_path = inputs["table_path"]
    smoothed_path = inputs["smoothed_path"]
    outdir = inputs["outdir"]

    if artifact is None and not smoothed_path and not table_path:
        raise RuntimeError("Provide --artifact, --summary, --smoothed_table, or --table to redraw the plot.")

    aggregated_results = None
    div_list: List[str] = []
    stat_label = ""

    if artifact is not None:
        aggregated_results = artifact.get("aggregated_results", None)
        div_list = artifact.get("divergences", []) or []
        args_meta = artifact.get("args", {})
        stat_label = args_meta.get("stat", "")

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

    raw_rows: Optional[List[Dict[str, Any]]] = None
    if table_path:
        raw_rows = _read_csv_table(
            table_path,
            {
                "method": str,
                "X0": float,
                "theta": float,
                "lower": float,
                "upper": float,
                "valid_rate": float,
            },
        )
        raw_aggregated = _build_aggregated_from_raw(raw_rows)
        if aggregated_results is None:
            aggregated_results = raw_aggregated
        else:
            _inject_raw_from_table(aggregated_results, raw_rows)
        if not div_list:
            div_list = sorted({r["method"] for r in raw_rows})

    if summary is not None:
        stat_label = summary.get("stat", stat_label)
        if not div_list:
            div_list = summary.get("divergences", []) or div_list

    if not aggregated_results:
        raise RuntimeError("Unable to construct aggregated results for plotting.")

    res = _select_result(aggregated_results, args.method)
    color_map = {
        "kth": "tab:cyan",
        "tight_kth": "tab:olive",
        "KL": "tab:green",
        "TV": "tab:red",
        "Hellinger": "tab:purple",
        "Chi2": "tab:brown",
        "JS": "tab:pink",
    }
    method_name = str(res.get("div", "") or args.method or "")
    bound_label = f"{method_name} bounds" if method_name else "Bounds"
    color = color_map.get(method_name, "tab:blue")

    xlim = _parse_limits(args.xlim, "--xlim")
    ylim = _parse_limits(args.ylim, "--ylim")
    try:
        fig_w, fig_h = [float(x.strip()) for x in args.figsize.split(",")]
    except Exception:
        raise ValueError("--figsize must be in the format 'width,height' (e.g., 11,6).")

    axis_label = _resolve_axis_label(summary, artifact, base_name)

    plot_raw = bool(args.plot_raw)
    plot_smooth = bool(args.plot_smooth)
    default_smooth = not plot_raw and not plot_smooth

    ran_any = False
    if plot_smooth or default_smooth:
        curve = _extract_curve(res, "smooth")
        x_s, l_s, u_s, theta_s = _apply_limits(
            curve["x"], curve["l"], curve["u"], curve["theta"], xlim, ylim
        )
        if x_s.size == 0:
            if xlim is not None or ylim is not None:
                print("Warning: xlim/ylim removed all points; plotting full curve without limits.")
                x_s, l_s, u_s, theta_s = _apply_limits(
                    curve["x"], curve["l"], curve["u"], curve["theta"], None, None
                )
            if x_s.size == 0:
                raise RuntimeError("No curve data found to plot.")

        raw_points = None
        if args.plot_raw_points:
            try:
                raw_curve = _extract_curve(res, "raw")
                x_r, l_r, u_r, _ = _apply_limits(
                    raw_curve["x"], raw_curve["l"], raw_curve["u"], raw_curve["theta"], xlim, ylim
                )
                raw_points = {"x": x_r, "l": l_r, "u": u_r}
            except RuntimeError:
                raw_points = None

        title = args.title or f"Causal bounds vs {axis_label}"
        if not default_smooth and args.title == "":
            title = f"Smoothed bounds vs {axis_label}"
        _plot_curve(
            x=x_s,
            l=l_s,
            u=u_s,
            theta=theta_s,
            bound_label=bound_label,
            color=color,
            axis_label=axis_label,
            title=title,
            tick_labelsize=args.tick_labelsize,
            figsize=(fig_w, fig_h),
            xlim=xlim,
            ylim=ylim,
            no_xlabel=args.no_xlabel,
            no_ylabel=args.no_ylabel,
            no_title=args.no_title,
            no_legend=args.no_legend,
            raw_points=raw_points,
        )
        os.makedirs(outdir, exist_ok=True)
        suffix = "" if default_smooth else "_smooth"
        fig_base = f"load_{base_name}{suffix}"
        fig_path = _name_with_suffix(outdir, fig_base, "png", stamp)
        plt.savefig(fig_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {fig_path}")
        ran_any = True

    if plot_raw:
        curve = _extract_curve(res, "raw")
        x_r, l_r, u_r, theta_r = _apply_limits(
            curve["x"], curve["l"], curve["u"], curve["theta"], xlim, ylim
        )
        if x_r.size == 0:
            if xlim is not None or ylim is not None:
                print("Warning: xlim/ylim removed all raw points; plotting full raw curve without limits.")
                x_r, l_r, u_r, theta_r = _apply_limits(
                    curve["x"], curve["l"], curve["u"], curve["theta"], None, None
                )
            if x_r.size == 0:
                raise RuntimeError("No raw curve data found to plot.")
        title = args.title or f"Raw bounds vs {axis_label}"
        _plot_curve(
            x=x_r,
            l=l_r,
            u=u_r,
            theta=theta_r,
            bound_label=bound_label,
            color=color,
            axis_label=axis_label,
            title=title,
            tick_labelsize=args.tick_labelsize,
            figsize=(fig_w, fig_h),
            xlim=xlim,
            ylim=ylim,
            no_xlabel=args.no_xlabel,
            no_ylabel=args.no_ylabel,
            no_title=args.no_title,
            no_legend=args.no_legend,
        )
        os.makedirs(outdir, exist_ok=True)
        fig_base = f"load_{base_name}_raw"
        fig_path = _name_with_suffix(outdir, fig_base, "png", stamp)
        plt.savefig(fig_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {fig_path}")
        ran_any = True

    return ran_any

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reload plot_idhp_ribbon artifacts and redraw plots.")
    parser.add_argument("--stamp", type=str, default="", help="Timestamp suffix used in output filenames.")
    parser.add_argument("--artifact", type=str, default="", help="Path to artifacts pickle.")
    parser.add_argument("--summary", type=str, default="", help="Path to summary JSON.")
    parser.add_argument("--table", type=str, default="", help="Path to raw table CSV.")
    parser.add_argument("--smoothed_table", type=str, default="", help="Path to smoothed table CSV.")
    parser.add_argument("--artifact_dir", type=str, default="", help="Directory for --stamp lookup.")
    parser.add_argument("--outdir", type=str, default="", help="Output directory for plot.")
    parser.add_argument(
        "--axis",
        type=str,
        default=None,
        choices=["x0", "propensity", "both"],
        help="Which eval axis to redraw (auto-detects from inputs if omitted).",
    )
    parser.add_argument("--method", type=str, default="", help="Select a divergence/method to plot.")
    parser.add_argument("--plot_raw", action="store_true", help="Write raw curve plot (x_raw vs l/u/theta).")
    parser.add_argument("--plot_smooth", action="store_true", help="Write smooth curve plot (x_s vs l/u/theta).")
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

    if args.axis == "both":
        axes = ["x0", "propensity"]
    elif args.axis in {"x0", "propensity"}:
        axes = [args.axis]
    else:
        axes = [None]

    ran_any = False
    for axis in axes:
        base_name = _axis_to_base(axis) if axis else _resolve_base_name(args)
        try:
            ran_any = _run_for_base(args, base_name) or ran_any
        except RuntimeError as exc:
            if axis is not None and args.stamp:
                print(f"Skipping axis '{axis}': {exc}")
            else:
                raise
    if not ran_any:
        raise RuntimeError("No plots were generated. Check inputs or --stamp.")
