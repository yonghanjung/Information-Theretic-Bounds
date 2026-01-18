"""
Reload plot_n artifacts and redraw a single plot by name.

Example:
  python3 load_plot_n.py --plot_name plot_n_score_debiased_vs_naive_stat_median_over_median_20260115_192202
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_n_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            raise ValueError(f"Invalid n value: {p}")
    if not out:
        raise ValueError("n_list is empty.")
    return out


def _parse_divergences(raw: str, base_divs: List[str]) -> List[str]:
    allowed = set(base_divs + ["kth", "tight_kth"])
    divs = [d.strip() for d in raw.split(",") if d.strip()]
    if not divs:
        divs = ["kth"]
    for d in divs:
        if d not in allowed:
            raise ValueError(f"Unknown divergence '{d}'. Allowed: {sorted(allowed)}")
    return divs


def _stat_reduce(arr: np.ndarray, stat: str) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if stat == "mean":
        return float(np.mean(arr))
    return float(np.median(arr))


def _nan_quantile(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _stat_suffix(stat_within: str, stat_over_reps: str) -> str:
    return f"stat_{stat_within}_over_{stat_over_reps}"


def _parse_plot_name(raw: str) -> Tuple[str, str, str, str]:
    base_name = os.path.basename(raw)
    name, ext = os.path.splitext(base_name)
    ext = ext.lstrip(".") or "png"
    pat = re.compile(
        r"^(?P<base>.+?)(?:_(?P<stat>stat_(?:mean|median)_over_(?:mean|median)))?_(?P<stamp>\d{8}_\d{6})$"
    )
    match = pat.match(name)
    if not match:
        raise ValueError(f"Could not parse plot name: {raw}")
    base = match.group("base")
    stat = match.group("stat") or ""
    stamp = match.group("stamp")
    return base, stat, stamp, ext


def _build_summary(
    *,
    replicate_rows: List[Dict[str, Any]],
    div_list: List[str],
    n_list: List[int],
    stat_within: str,
    stat_over_reps: str,
    ci_alpha: float,
    args_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for div in div_list:
        for n in n_list:
            rows = [r for r in replicate_rows if r["divergence"] == div and r["n"] == n]
            prop = np.array([r["propensity_rmse"] for r in rows], dtype=float)
            prop_true = np.array([r["propensity_rmse_true"] for r in rows], dtype=float)
            err_d = np.array([r["err_up_debiased"] for r in rows], dtype=float)
            err_n = np.array([r["err_up_naive"] for r in rows], dtype=float)
            cov_d = np.array([r["coverage_debiased"] for r in rows], dtype=float)
            cov_n = np.array([r["coverage_naive"] for r in rows], dtype=float)
            covu_d = np.array([r["coverage_uncond_debiased"] for r in rows], dtype=float)
            covu_n = np.array([r["coverage_uncond_naive"] for r in rows], dtype=float)
            covc_d = np.array([r["coverage_cond_debiased"] for r in rows], dtype=float)
            covc_n = np.array([r["coverage_cond_naive"] for r in rows], dtype=float)
            if stat_within == "mean":
                wid_d = np.array([r["width_debiased_mean"] for r in rows], dtype=float)
                wid_n = np.array([r["width_naive_mean"] for r in rows], dtype=float)
                sco_d = np.array([r["score_debiased_mean"] for r in rows], dtype=float)
                sco_n = np.array([r["score_naive_mean"] for r in rows], dtype=float)
            else:
                wid_d = np.array([r["width_debiased_median"] for r in rows], dtype=float)
                wid_n = np.array([r["width_naive_median"] for r in rows], dtype=float)
                sco_d = np.array([r["score_debiased_median"] for r in rows], dtype=float)
                sco_n = np.array([r["score_naive_median"] for r in rows], dtype=float)
            val_d = np.array([r["valid_rate_debiased"] for r in rows], dtype=float)
            val_n = np.array([r["valid_rate_naive"] for r in rows], dtype=float)

            summary_rows.append(
                {
                    "divergence": div,
                    "n": int(n),
                    "propensity_rmse_center": _stat_reduce(prop, stat_over_reps),
                    "propensity_rmse_ci_low": _nan_quantile(prop, ci_alpha / 2),
                    "propensity_rmse_ci_high": _nan_quantile(prop, 1 - ci_alpha / 2),
                    "propensity_rmse_true_center": _stat_reduce(prop_true, stat_over_reps),
                    "propensity_rmse_true_ci_low": _nan_quantile(prop_true, ci_alpha / 2),
                    "propensity_rmse_true_ci_high": _nan_quantile(prop_true, 1 - ci_alpha / 2),
                    "err_up_debiased_center": _stat_reduce(err_d, stat_over_reps),
                    "err_up_debiased_ci_low": _nan_quantile(err_d, ci_alpha / 2),
                    "err_up_debiased_ci_high": _nan_quantile(err_d, 1 - ci_alpha / 2),
                    "err_up_naive_center": _stat_reduce(err_n, stat_over_reps),
                    "err_up_naive_ci_low": _nan_quantile(err_n, ci_alpha / 2),
                    "err_up_naive_ci_high": _nan_quantile(err_n, 1 - ci_alpha / 2),
                    "coverage_debiased_center": _stat_reduce(cov_d, stat_over_reps),
                    "coverage_debiased_ci_low": _nan_quantile(cov_d, ci_alpha / 2),
                    "coverage_debiased_ci_high": _nan_quantile(cov_d, 1 - ci_alpha / 2),
                    "coverage_naive_center": _stat_reduce(cov_n, stat_over_reps),
                    "coverage_naive_ci_low": _nan_quantile(cov_n, ci_alpha / 2),
                    "coverage_naive_ci_high": _nan_quantile(cov_n, 1 - ci_alpha / 2),
                    "coverage_uncond_debiased_center": _stat_reduce(covu_d, stat_over_reps),
                    "coverage_uncond_debiased_ci_low": _nan_quantile(covu_d, ci_alpha / 2),
                    "coverage_uncond_debiased_ci_high": _nan_quantile(covu_d, 1 - ci_alpha / 2),
                    "coverage_uncond_naive_center": _stat_reduce(covu_n, stat_over_reps),
                    "coverage_uncond_naive_ci_low": _nan_quantile(covu_n, ci_alpha / 2),
                    "coverage_uncond_naive_ci_high": _nan_quantile(covu_n, 1 - ci_alpha / 2),
                    "coverage_cond_debiased_center": _stat_reduce(covc_d, stat_over_reps),
                    "coverage_cond_debiased_ci_low": _nan_quantile(covc_d, ci_alpha / 2),
                    "coverage_cond_debiased_ci_high": _nan_quantile(covc_d, 1 - ci_alpha / 2),
                    "coverage_cond_naive_center": _stat_reduce(covc_n, stat_over_reps),
                    "coverage_cond_naive_ci_low": _nan_quantile(covc_n, ci_alpha / 2),
                    "coverage_cond_naive_ci_high": _nan_quantile(covc_n, 1 - ci_alpha / 2),
                    "width_debiased_center": _stat_reduce(wid_d, stat_over_reps),
                    "width_debiased_ci_low": _nan_quantile(wid_d, ci_alpha / 2),
                    "width_debiased_ci_high": _nan_quantile(wid_d, 1 - ci_alpha / 2),
                    "width_naive_center": _stat_reduce(wid_n, stat_over_reps),
                    "width_naive_ci_low": _nan_quantile(wid_n, ci_alpha / 2),
                    "width_naive_ci_high": _nan_quantile(wid_n, 1 - ci_alpha / 2),
                    "score_debiased_center": _stat_reduce(sco_d, stat_over_reps),
                    "score_debiased_ci_low": _nan_quantile(sco_d, ci_alpha / 2),
                    "score_debiased_ci_high": _nan_quantile(sco_d, 1 - ci_alpha / 2),
                    "score_naive_center": _stat_reduce(sco_n, stat_over_reps),
                    "score_naive_ci_low": _nan_quantile(sco_n, ci_alpha / 2),
                    "score_naive_ci_high": _nan_quantile(sco_n, 1 - ci_alpha / 2),
                    "valid_rate_debiased_center": _stat_reduce(val_d, stat_over_reps),
                    "valid_rate_debiased_ci_low": _nan_quantile(val_d, ci_alpha / 2),
                    "valid_rate_debiased_ci_high": _nan_quantile(val_d, 1 - ci_alpha / 2),
                    "valid_rate_naive_center": _stat_reduce(val_n, stat_over_reps),
                    "valid_rate_naive_ci_low": _nan_quantile(val_n, ci_alpha / 2),
                    "valid_rate_naive_ci_high": _nan_quantile(val_n, 1 - ci_alpha / 2),
                    "m": int(args_meta["m"]),
                    "d": int(args_meta["d"]),
                    "n_eval": int(args_meta["n_eval"]),
                    "structural_type": args_meta["structural_type"],
                    "eval_mode": args_meta["eval_mode"],
                    "width_stat": stat_within,
                    "stat_over_reps": stat_over_reps,
                    "score_lambda": float(args_meta["score_lambda"]),
                    "score_alpha": float(args_meta["score_alpha"]),
                    "ci_alpha": float(ci_alpha),
                }
            )
    return summary_rows


def _plot_nuisance(
    rows: List[Dict[str, Any]],
    *,
    div_list: List[str],
    structural_type: str,
    eval_mode: str,
    no_errorbar: bool,
    style: Dict[str, Any],
    path: str,
) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for div in div_list:
        sub = [row for row in rows if row["divergence"] == div]
        xs = [row["n"] for row in sub]
        ys = [row["propensity_rmse_center"] for row in sub]
        if no_errorbar:
            plt.plot(xs, ys, marker="o", linestyle="-", label=f"{div}")
        else:
            lo = [row["propensity_rmse_ci_low"] for row in sub]
            hi = [row["propensity_rmse_ci_high"] for row in sub]
            yerr = [np.subtract(ys, lo), np.subtract(hi, ys)]
            plt.errorbar(xs, ys, yerr=yerr, marker="o", linestyle="-", capsize=3, label=f"{div}")

    if rows:
        xs_ref = np.array(sorted({row["n"] for row in rows}), dtype=float)
        if xs_ref.size >= 2:
            y0 = float(rows[0]["propensity_rmse_center"])
            n0 = float(xs_ref[0])
            ref_025 = y0 * (xs_ref / n0) ** (-0.25)
            ref_05 = y0 * (xs_ref / n0) ** (-0.5)
            plt.plot(xs_ref, ref_025, linestyle="--", color="gray", label="n^-1/4 ref")
            plt.plot(xs_ref, ref_05, linestyle=":", color="gray", label="n^-1/2 ref")

    _apply_axes_style(
        plt.gca(),
        title=style["title"] or f"Nuisance error vs n (struct={structural_type}, eval={eval_mode})",
        xlabel=style["xlabel"] or "Sample size n",
        ylabel=style["ylabel"] or "Propensity RMSE",
        style=style,
    )
    _apply_legend(plt.gca(), style)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_target(
    rows: List[Dict[str, Any]],
    *,
    div_list: List[str],
    structural_type: str,
    eval_mode: str,
    no_errorbar: bool,
    style: Dict[str, Any],
    path: str,
) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for div in div_list:
        sub = [row for row in rows if row["divergence"] == div]
        xs = [row["n"] for row in sub]
        yd = [row["err_up_debiased_center"] for row in sub]
        yn = [row["err_up_naive_center"] for row in sub]
        if no_errorbar:
            plt.plot(xs, yd, marker="o", linestyle="-", label=f"{div} debiased")
            plt.plot(xs, yn, marker="o", linestyle="--", label=f"{div} naive")
        else:
            lo_d = [row["err_up_debiased_ci_low"] for row in sub]
            hi_d = [row["err_up_debiased_ci_high"] for row in sub]
            lo_n = [row["err_up_naive_ci_low"] for row in sub]
            hi_n = [row["err_up_naive_ci_high"] for row in sub]
            plt.errorbar(
                xs,
                yd,
                yerr=[np.subtract(yd, lo_d), np.subtract(hi_d, yd)],
                marker="o",
                linestyle="-",
                capsize=3,
                label=f"{div} debiased",
            )
            plt.errorbar(
                xs,
                yn,
                yerr=[np.subtract(yn, lo_n), np.subtract(hi_n, yn)],
                marker="o",
                linestyle="--",
                capsize=3,
                label=f"{div} naive",
            )
    _apply_axes_style(
        plt.gca(),
        title=style["title"] or f"Target error vs n (struct={structural_type}, eval={eval_mode})",
        xlabel=style["xlabel"] or "Sample size n",
        ylabel=style["ylabel"] or "Target RMSE (upper bound)",
        style=style,
    )
    _apply_legend(plt.gca(), style)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_metric(
    rows: List[Dict[str, Any]],
    *,
    div_list: List[str],
    key_d: str,
    key_n: str,
    ylabel: str,
    title: str,
    no_errorbar: bool,
    style: Dict[str, Any],
    path: str,
) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for div in div_list:
        sub = [row for row in rows if row["divergence"] == div]
        xs = [row["n"] for row in sub]
        yd = [row[f"{key_d}_center"] for row in sub]
        yn = [row[f"{key_n}_center"] for row in sub]
        if no_errorbar:
            plt.plot(xs, yd, marker="o", linestyle="-", label=f"{div} debiased")
            plt.plot(xs, yn, marker="o", linestyle="--", label=f"{div} naive")
        else:
            lo_d = [row[f"{key_d}_ci_low"] for row in sub]
            hi_d = [row[f"{key_d}_ci_high"] for row in sub]
            lo_n = [row[f"{key_n}_ci_low"] for row in sub]
            hi_n = [row[f"{key_n}_ci_high"] for row in sub]
            plt.errorbar(
                xs,
                yd,
                yerr=[np.subtract(yd, lo_d), np.subtract(hi_d, yd)],
                marker="o",
                linestyle="-",
                capsize=3,
                label=f"{div} debiased",
            )
            plt.errorbar(
                xs,
                yn,
                yerr=[np.subtract(yn, lo_n), np.subtract(hi_n, yn)],
                marker="o",
                linestyle="--",
                capsize=3,
                label=f"{div} naive",
            )
    _apply_axes_style(
        plt.gca(),
        title=style["title"] or title,
        xlabel=style["xlabel"] or "Sample size n",
        ylabel=style["ylabel"] or ylabel,
        style=style,
    )
    _apply_legend(plt.gca(), style)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_metric_single(
    rows: List[Dict[str, Any]],
    *,
    div_list: List[str],
    key_d: str,
    ylabel: str,
    title: str,
    style: Dict[str, Any],
    path: str,
) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for div in div_list:
        sub = [row for row in rows if row["divergence"] == div]
        xs = [row["n"] for row in sub]
        yd = [row[f"{key_d}_center"] for row in sub]
        lo_d = [row[f"{key_d}_ci_low"] for row in sub]
        hi_d = [row[f"{key_d}_ci_high"] for row in sub]
        plt.errorbar(
            xs,
            yd,
            yerr=[np.subtract(yd, lo_d), np.subtract(hi_d, yd)],
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"{div} debiased",
        )
    _apply_axes_style(
        plt.gca(),
        title=style["title"] or title,
        xlabel=style["xlabel"] or "Sample size n",
        ylabel=style["ylabel"] or ylabel,
        style=style,
    )
    _apply_legend(plt.gca(), style)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reload plot_n artifacts and redraw a single plot by name."
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        required=True,
        help="Plot name (e.g. plot_n_score_debiased_vs_naive_stat_median_over_median_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument("--artifact", type=str, default="", help="Path to plot_n_debiased_artifacts_*.pkl")
    parser.add_argument("--artifact_dir", type=str, default="experiments", help="Directory containing artifacts.")
    parser.add_argument("--outdir", type=str, default="", help="Output directory for the redrawn plot.")
    parser.add_argument("--no_errorbar", action="store_true", help="Disable error bars.")
    parser.add_argument("--title", type=str, default="", help="Override plot title.")
    parser.add_argument("--xlabel", type=str, default="", help="Override x-axis label.")
    parser.add_argument("--ylabel", type=str, default="", help="Override y-axis label.")
    parser.add_argument("--xtick_title", type=str, default="", help="Alias for --xlabel.")
    parser.add_argument("--ytick_title", type=str, default="", help="Alias for --ylabel.")
    parser.add_argument("--title_size", type=float, default=0.0, help="Title font size (0 disables).")
    parser.add_argument("--label_size", type=float, default=0.0, help="Axis label font size (0 disables).")
    parser.add_argument("--tick_size", type=float, default=0.0, help="Tick label font size (0 disables).")
    parser.add_argument("--legend", dest="legend", action="store_true", default=True, help="Show legend.")
    parser.add_argument("--no-legend", dest="legend", action="store_false", help="Hide legend.")
    parser.add_argument("--legend_loc", type=str, default="best", help="Legend location (matplotlib loc).")
    parser.add_argument("--legend_size", type=float, default=0.0, help="Legend font size (0 disables).")

    parser.add_argument("--divergence", type=str, default="", help="Override divergences list.")
    parser.add_argument("--n_list", type=str, default="", help="Override n_list as comma-separated integers.")
    parser.add_argument("--structural_type", type=str, default="nonlinear", help="Fallback structural type.")
    parser.add_argument("--eval_mode", type=str, default="grid_ehat", help="Fallback eval mode.")
    parser.add_argument("--n_eval", type=int, default=0, help="Fallback n_eval.")
    parser.add_argument("--score_lambda", type=float, default=10.0, help="Fallback score lambda.")
    parser.add_argument("--score_alpha", type=float, default=0.05, help="Fallback score alpha.")
    parser.add_argument("--ci_alpha", type=float, default=0.05, help="Fallback CI alpha.")
    parser.add_argument("--width_stat", type=str, default="mean", help="Fallback width_stat (mean/median).")
    parser.add_argument("--stat_over_reps", type=str, default="mean", help="Fallback stat_over_reps (mean/median).")
    args = parser.parse_args()

    base_name, stat_suffix, stamp, ext = _parse_plot_name(args.plot_name)
    title_override = args.title.strip() or None
    xlabel_override = args.xlabel.strip() or args.xtick_title.strip() or None
    ylabel_override = args.ylabel.strip() or args.ytick_title.strip() or None
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
    outdir = args.outdir if args.outdir else args.artifact_dir
    os.makedirs(outdir, exist_ok=True)

    artifact_path = args.artifact
    if not artifact_path:
        artifact_path = os.path.join(args.artifact_dir, f"plot_n_debiased_artifacts_{stamp}.pkl")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    with open(artifact_path, "rb") as f:
        art = pickle.load(f)

    replicate_rows = art.get("replicate_rows", [])
    if not replicate_rows:
        raise ValueError("Artifacts missing replicate_rows; cannot reconstruct plots.")

    art_args = art.get("args", {})
    base_divs = ["KL", "TV", "Hellinger", "Chi2", "JS"]
    div_list = art.get("divergences") or _parse_divergences(args.divergence, base_divs)
    if args.divergence:
        div_list = _parse_divergences(args.divergence, base_divs)

    n_list = sorted({row["n"] for row in replicate_rows})
    if not n_list and args.n_list:
        n_list = _parse_n_list(args.n_list)

    structural_type = art_args.get("structural_type", args.structural_type)
    eval_mode = art_args.get("eval_mode", args.eval_mode)
    n_eval = int(art_args.get("n_eval", args.n_eval))
    score_lambda = float(art_args.get("score_lambda", args.score_lambda))
    score_alpha = float(art_args.get("score_alpha", args.score_alpha))
    ci_alpha = float(art_args.get("ci_alpha", args.ci_alpha))

    stat_within = args.width_stat
    stat_over_reps = args.stat_over_reps
    if stat_suffix:
        parts = stat_suffix.split("_")
        stat_within = parts[1]
        stat_over_reps = parts[3]
    else:
        stat_within = str(art_args.get("width_stat", args.width_stat))
        stat_over_reps = str(art_args.get("stat_over_reps", args.stat_over_reps))

    args_meta = {
        "m": art_args.get("m", 0),
        "d": art_args.get("d", 0),
        "n_eval": n_eval,
        "structural_type": structural_type,
        "eval_mode": eval_mode,
        "score_lambda": score_lambda,
        "score_alpha": score_alpha,
    }

    summary_rows_by_stat = art.get("summary_rows_by_stat", {})
    stat_key = stat_suffix or "default"
    summary_rows = summary_rows_by_stat.get(stat_key)
    if summary_rows is None:
        summary_rows = _build_summary(
            replicate_rows=replicate_rows,
            div_list=div_list,
            n_list=n_list,
            stat_within=stat_within,
            stat_over_reps=stat_over_reps,
            ci_alpha=ci_alpha,
            args_meta=args_meta,
        )

    output_path = os.path.join(
        outdir,
        f"loaded_{base_name}_{stat_suffix + '_' if stat_suffix else ''}{stamp}.{ext}",
    )

    plot_specs = {
        "plot_n_debiased_nuisance": lambda: _plot_nuisance(
            summary_rows,
            div_list=div_list,
            structural_type=structural_type,
            eval_mode=eval_mode,
            no_errorbar=args.no_errorbar,
            style=style,
            path=output_path,
        ),
        "plot_n_debiased_target": lambda: _plot_target(
            summary_rows,
            div_list=div_list,
            structural_type=structural_type,
            eval_mode=eval_mode,
            no_errorbar=args.no_errorbar,
            style=style,
            path=output_path,
        ),
        "plot_n_debiased_width": lambda: _plot_metric(
            summary_rows,
            div_list=div_list,
            key_d="width_debiased",
            key_n="width_naive",
            ylabel="Width",
            title=f"Width vs n (struct={structural_type}, eval={eval_mode})",
            no_errorbar=args.no_errorbar,
            style=style,
            path=output_path,
        ),
        "plot_n_debiased_coverage": lambda: _plot_metric(
            summary_rows,
            div_list=div_list,
            key_d="coverage_debiased",
            key_n="coverage_naive",
            ylabel="Coverage",
            title=f"Coverage vs n (struct={structural_type}, eval={eval_mode})",
            no_errorbar=args.no_errorbar,
            style=style,
            path=output_path,
        ),
        "plot_n_debiased_score": lambda: _plot_metric(
            summary_rows,
            div_list=div_list,
            key_d="score_debiased",
            key_n="score_naive",
            ylabel="Penalized width",
            title=f"Penalized width vs n (struct={structural_type}, eval={eval_mode})",
            no_errorbar=args.no_errorbar,
            style=style,
            path=output_path,
        ),
        "plot_n_coverage_uncond_debiased_vs_naive": lambda: _plot_metric(
            summary_rows,
            div_list=div_list,
            key_d="coverage_uncond_debiased",
            key_n="coverage_uncond_naive",
            ylabel="Coverage (unconditional)",
            title=f"Coverage (unconditional) vs n (struct={structural_type}, eval={eval_mode})",
            no_errorbar=args.no_errorbar,
            style=style,
            path=output_path,
        ),
        "plot_n_coverage_cond_debiased_vs_naive": lambda: _plot_metric(
            summary_rows,
            div_list=div_list,
            key_d="coverage_cond_debiased",
            key_n="coverage_cond_naive",
            ylabel="Coverage (conditional on valid)",
            title=f"Coverage (conditional) vs n (struct={structural_type}, eval={eval_mode})",
            no_errorbar=args.no_errorbar,
            style=style,
            path=output_path,
        ),
        "plot_n_valid_rate_debiased_vs_naive": lambda: _plot_metric(
            summary_rows,
            div_list=div_list,
            key_d="valid_rate_debiased",
            key_n="valid_rate_naive",
            ylabel="Valid rate",
            title=f"Valid rate vs n (struct={structural_type}, eval={eval_mode})",
            no_errorbar=args.no_errorbar,
            style=style,
            path=output_path,
        ),
        "plot_n_width_debiased_vs_naive": lambda: _plot_metric(
            summary_rows,
            div_list=div_list,
            key_d="width_debiased",
            key_n="width_naive",
            ylabel="Width",
            title=f"Width vs n (struct={structural_type}, eval={eval_mode})",
            no_errorbar=args.no_errorbar,
            style=style,
            path=output_path,
        ),
        "plot_n_score_debiased_vs_naive": lambda: _plot_metric(
            summary_rows,
            div_list=div_list,
            key_d="score_debiased",
            key_n="score_naive",
            ylabel="Penalized width",
            title=f"Penalized width vs n (struct={structural_type}, eval={eval_mode})",
            no_errorbar=args.no_errorbar,
            style=style,
            path=output_path,
        ),
        "plot_n_coverage_uncond_debiased_only": lambda: _plot_metric_single(
            summary_rows,
            div_list=div_list,
            key_d="coverage_uncond_debiased",
            ylabel="Coverage (unconditional)",
            title=(
                f"Coverage (unconditional) vs n (debiased only; "
                f"struct={structural_type}, eval={eval_mode})"
            ),
            style=style,
            path=output_path,
        ),
        "plot_n_width_debiased_only": lambda: _plot_metric_single(
            summary_rows,
            div_list=div_list,
            key_d="width_debiased",
            ylabel="Width",
            title=f"Width vs n (debiased only; struct={structural_type}, eval={eval_mode})",
            style=style,
            path=output_path,
        ),
    }

    plot_fn = plot_specs.get(base_name)
    if plot_fn is None:
        allowed = ", ".join(sorted(plot_specs.keys()))
        raise ValueError(f"Unknown plot base '{base_name}'. Allowed: {allowed}")

    plot_fn()
    print(f"[load_plot_n] wrote: {output_path}")


if __name__ == "__main__":
    main()
