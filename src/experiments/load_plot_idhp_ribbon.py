"""
Reload plot_idhp_ribbon artifacts and redraw a single plot by name.

Example:
  python3 load_plot_idhp_ribbon.py --plot_name plot_idhp_ribbon_propensity_raw_20260109_111750
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fbound.utils.plotting import DIVERGENCE_COLOR_MAP


def _parse_plot_name(raw: str) -> Tuple[str, str, str, str]:
    base_name = os.path.basename(raw)
    name, ext = os.path.splitext(base_name)
    ext = ext.lstrip(".") or "png"
    pat = re.compile(r"^(?P<base>.+)_(?P<kind>raw|smoothed)(?:_(?P<stamp>\d{8}_\d{6}))?$")
    match = pat.match(name)
    if not match:
        raise ValueError(f"Could not parse plot name: {raw}")
    base = match.group("base")
    kind = match.group("kind")
    stamp = match.group("stamp") or ""
    return base, kind, stamp, ext


def _artifact_name(base: str, stamp: str) -> str:
    if stamp:
        return f"{base}_artifacts_{stamp}.pkl"
    return f"{base}_artifacts.pkl"


def _axis_label(eval_axis: str, args: Dict[str, Any]) -> str:
    if eval_axis == "x0":
        eval_dim = int(args.get("eval_dim", 0))
        return f"X{eval_dim}"
    if eval_axis == "propensity":
        return "e(A=1|X)"
    return str(eval_axis)


def _apply_axes_style(ax, *, title: str, xlabel: str, ylabel: str, style: Dict[str, Any]) -> None:
    title_size = style.get("title_size")
    label_size = style.get("label_size")
    tick_size = style.get("tick_size")
    if title:
        if title_size is None:
            ax.set_title(title)
        else:
            ax.set_title(title, fontsize=title_size)
    if xlabel:
        if label_size is None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel:
        if label_size is None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(ylabel, fontsize=label_size)
    if tick_size is not None:
        ax.tick_params(axis="both", labelsize=tick_size)


def _apply_legend(ax, style: Dict[str, Any]) -> None:
    if not style.get("legend", True):
        return
    legend_loc = style.get("legend_loc") or "best"
    legend_size = style.get("legend_size")
    if legend_size is None:
        ax.legend(loc=legend_loc)
    else:
        ax.legend(loc=legend_loc, fontsize=legend_size)


def _parse_legend_labels(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not raw:
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Legend label entry must be key=value. Got: {item!r}")
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        out[key.lower()] = val
    return out


def _legend_label(legend_labels: Dict[str, str], key: str, default: str) -> str:
    if not legend_labels:
        return default
    return legend_labels.get(key.lower(), default)


def _plot_smoothed(
    results: List[Dict[str, Any]],
    *,
    axis_label: str,
    title: str,
    ylabel: str,
    style: Dict[str, Any],
    plot_raw: bool,
    plot_raw_points: bool,
    legend_labels: Dict[str, str],
    path: str,
) -> None:
    plt.figure(figsize=(7.0, 4.0))
    for res in results:
        x_s = np.asarray(res.get("x_s", []), dtype=float)
        if x_s.size == 0:
            continue
        div = res["div"]
        c = DIVERGENCE_COLOR_MAP.get(div)
        l_s = np.asarray(res.get("l_s", []), dtype=float)
        u_s = np.asarray(res.get("u_s", []), dtype=float)
        bounds_label = _legend_label(legend_labels, div, f"{div} bounds")
        plt.fill_between(x_s, l_s, u_s, alpha=0.2, color=c, label=bounds_label)
        plt.plot(x_s, l_s, color=c, alpha=0.7, linewidth=1.0)
        plt.plot(x_s, u_s, color=c, alpha=0.7, linewidth=1.0)
        if plot_raw:
            x_raw = np.asarray(res.get("x_raw", []), dtype=float)
            l_raw = np.asarray(res.get("l_raw", []), dtype=float)
            u_raw = np.asarray(res.get("u_raw", []), dtype=float)
            plt.plot(x_raw, l_raw, color=c, alpha=0.7, linewidth=1.0)
            plt.plot(x_raw, u_raw, color=c, alpha=0.7, linewidth=1.0)
        if plot_raw_points:
            x_raw = np.asarray(res.get("x_raw", []), dtype=float)
            l_raw = np.asarray(res.get("l_raw", []), dtype=float)
            u_raw = np.asarray(res.get("u_raw", []), dtype=float)
            plt.scatter(x_raw, l_raw, color=c, alpha=0.2, s=8)
            plt.scatter(x_raw, u_raw, color=c, alpha=0.2, s=8)

    if results:
        x_s = np.asarray(results[0].get("x_s", []), dtype=float)
        theta_s = np.asarray(results[0].get("theta_s", []), dtype=float)
        if x_s.size > 0:
            truth_label = _legend_label(legend_labels, "Truth", "Truth")
            plt.plot(x_s, theta_s, color="k", linewidth=1.5, label=truth_label)

    _apply_axes_style(
        plt.gca(),
        title=title,
        xlabel=axis_label,
        ylabel=ylabel,
        style=style,
    )
    _apply_legend(plt.gca(), style)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_raw(
    results: List[Dict[str, Any]],
    *,
    axis_label: str,
    title: str,
    ylabel: str,
    style: Dict[str, Any],
    legend_labels: Dict[str, str],
    path: str,
) -> None:
    plt.figure(figsize=(7.0, 4.0))
    for res in results:
        x_raw = np.asarray(res.get("x_raw", []), dtype=float)
        if x_raw.size == 0:
            continue
        div = res["div"]
        c = DIVERGENCE_COLOR_MAP.get(div)
        l_raw = np.asarray(res.get("l_raw", []), dtype=float)
        u_raw = np.asarray(res.get("u_raw", []), dtype=float)
        bounds_label = _legend_label(legend_labels, div, f"{div} bounds")
        plt.fill_between(x_raw, l_raw, u_raw, alpha=0.2, color=c, label=bounds_label)
        plt.plot(x_raw, l_raw, color=c, alpha=0.7, linewidth=1.0)
        plt.plot(x_raw, u_raw, color=c, alpha=0.7, linewidth=1.0)

    if results:
        x_raw = np.asarray(results[0].get("x_raw", []), dtype=float)
        theta_raw = np.asarray(results[0].get("theta_raw", []), dtype=float)
        if x_raw.size > 0:
            truth_label = _legend_label(legend_labels, "Truth", "Truth")
            plt.plot(x_raw, theta_raw, color="k", linewidth=1.5, label=truth_label)

    _apply_axes_style(
        plt.gca(),
        title=title,
        xlabel=axis_label,
        ylabel=ylabel,
        style=style,
    )
    _apply_legend(plt.gca(), style)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reload plot_idhp_ribbon artifacts and redraw a single plot by name."
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        required=True,
        help="Plot name (e.g. plot_idhp_ribbon_propensity_raw_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument("--artifact", type=str, default="", help="Path to plot_idhp_ribbon_*_artifacts_*.pkl")
    parser.add_argument("--artifact_dir", type=str, default="experiments", help="Directory containing artifacts.")
    parser.add_argument("--outdir", type=str, default="experiments", help="Output directory for the redrawn plot.")
    parser.add_argument("--title", type=str, default="", help="Override plot title.")
    parser.add_argument("--xlabel", type=str, default="", help="Override x-axis label.")
    parser.add_argument("--ylabel", type=str, default="", help="Override y-axis label.")
    parser.add_argument("--title_size", type=float, default=0.0, help="Title font size (0 disables).")
    parser.add_argument("--label_size", type=float, default=0.0, help="Axis label font size (0 disables).")
    parser.add_argument("--tick_size", type=float, default=0.0, help="Tick label font size (0 disables).")
    parser.add_argument("--legend", dest="legend", action="store_true", default=True, help="Show legend.")
    parser.add_argument("--no-legend", dest="legend", action="store_false", help="Hide legend.")
    parser.add_argument("--legend_loc", type=str, default="best", help="Legend location (matplotlib loc).")
    parser.add_argument("--legend_size", type=float, default=0.0, help="Legend font size (0 disables).")
    parser.add_argument(
        "--legend_labels",
        type=str,
        default="",
        help="Comma-separated key=value labels (e.g., 'TV=Total Variation,Truth=Ground Truth').",
    )
    parser.add_argument("--plot_raw", dest="plot_raw", action="store_true", default=None, help="Overlay raw lines.")
    parser.add_argument(
        "--no-plot_raw",
        dest="plot_raw",
        action="store_false",
        help="Disable raw line overlay.",
    )
    parser.add_argument(
        "--plot_raw_points",
        dest="plot_raw_points",
        action="store_true",
        default=None,
        help="Overlay raw points.",
    )
    parser.add_argument(
        "--no-plot_raw_points",
        dest="plot_raw_points",
        action="store_false",
        help="Disable raw point overlay.",
    )
    args = parser.parse_args()

    base_name, kind, stamp, ext = _parse_plot_name(args.plot_name)
    title_override = args.title.strip() or None
    xlabel_override = args.xlabel.strip() or None
    ylabel_override = args.ylabel.strip() or None
    style = {
        "title": title_override,
        "xlabel": xlabel_override,
        "ylabel": ylabel_override,
        "title_size": args.title_size if args.title_size > 0 else None,
        "label_size": args.label_size if args.label_size > 0 else None,
        "tick_size": args.tick_size if args.tick_size > 0 else None,
        "legend": bool(args.legend),
        "legend_loc": args.legend_loc,
        "legend_size": args.legend_size if args.legend_size > 0 else None,
    }
    legend_labels = _parse_legend_labels(args.legend_labels)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    artifact_path = args.artifact
    if not artifact_path:
        artifact_path = os.path.join(args.artifact_dir, _artifact_name(base_name, stamp))
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    with open(artifact_path, "rb") as f:
        art = pickle.load(f)

    aggregated_results = art.get("aggregated_results", [])
    if not aggregated_results:
        raise ValueError("Artifacts missing aggregated_results; cannot reconstruct plot.")

    div_list = art.get("divergences") or [res["div"] for res in aggregated_results]
    res_by_div = {res["div"]: res for res in aggregated_results}
    ordered_results = [res_by_div[d] for d in div_list if d in res_by_div]
    for res in aggregated_results:
        if res["div"] not in div_list:
            ordered_results.append(res)

    art_args = art.get("args", {})
    eval_axis = art.get("eval_axis", "")
    axis_label = _axis_label(str(eval_axis), art_args)
    if style["xlabel"]:
        axis_label = style["xlabel"]
    ylabel = style["ylabel"] or "E[Y | do(A=1), X]"
    stat = str(art_args.get("stat", "mean"))
    divs = ",".join(div_list)
    if kind == "raw":
        title = style["title"] or f"Raw bounds vs {axis_label} (stat={stat}, divs={divs})"
    else:
        title = style["title"] or f"Causal bounds vs {axis_label} (stat={stat}, divs={divs})"

    plot_raw = art_args.get("plot_raw", False) if args.plot_raw is None else bool(args.plot_raw)
    plot_raw_points = art_args.get("plot_raw_points", False) if args.plot_raw_points is None else bool(args.plot_raw_points)

    stamp_suffix = f"_{stamp}" if stamp else ""
    output_path = os.path.join(outdir, f"loaded_{base_name}_{kind}{stamp_suffix}.{ext}")

    if kind == "raw":
        _plot_raw(
            ordered_results,
            axis_label=axis_label,
            title=title,
            ylabel=ylabel,
            style=style,
            legend_labels=legend_labels,
            path=output_path,
        )
    else:
        _plot_smoothed(
            ordered_results,
            axis_label=axis_label,
            title=title,
            ylabel=ylabel,
            style=style,
            plot_raw=plot_raw,
            plot_raw_points=plot_raw_points,
            legend_labels=legend_labels,
            path=output_path,
        )

    print(f"[load_plot_idhp_ribbon] wrote: {output_path}")


if __name__ == "__main__":
    main()
