"""
Reload plot_n_debiased artifacts and regenerate plots without rerunning simulations.
Accepts the same CLI as plot_n_debiased.py plus:
  --stamp / --artifact / --artifact_dir for locating the artifacts
  --no_errorbar to disable error bars
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _infer_stamp(artifact_path: str, art: dict) -> str:
    stamp = art.get("timestamp")
    if isinstance(stamp, str) and stamp:
        return stamp
    m = re.search(r"plot_n_debiased_artifacts_(\d{8}_\d{6})", os.path.basename(artifact_path))
    if m:
        return m.group(1)
    return ""


def _name_with_suffix(outdir: str, base: str, ext: str, stamp: str) -> str:
    fname = f"{base}_{stamp}.{ext}" if stamp else f"{base}.{ext}"
    return os.path.join(outdir, fname)


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
            err_d = np.array([r["err_up_debiased"] for r in rows], dtype=float)
            err_n = np.array([r["err_up_naive"] for r in rows], dtype=float)
            cov_d = np.array([r["coverage_debiased"] for r in rows], dtype=float)
            cov_n = np.array([r["coverage_naive"] for r in rows], dtype=float)
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
                    "m": int(args_meta.get("m", 0)),
                    "d": int(args_meta.get("d", 0)),
                    "n_eval": int(args_meta.get("n_eval", 0)),
                    "structural_type": args_meta.get("structural_type", "unknown"),
                    "eval_mode": args_meta.get("eval_mode", "unknown"),
                    "width_stat": stat_within,
                    "stat_over_reps": stat_over_reps,
                    "score_lambda": float(args_meta.get("score_lambda", 0.0)),
                    "score_alpha": float(args_meta.get("score_alpha", 0.0)),
                    "ci_alpha": float(ci_alpha),
                }
            )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Reload plot_n_debiased artifacts and redraw plots.")
    parser.add_argument("--stamp", type=str, default="", help="Timestamp suffix used in output filenames.")
    parser.add_argument("--artifact", type=str, default="", help="Path to artifacts pickle.")
    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="",
        help="Directory to look for artifacts when using --stamp.",
    )

    # Match plot_n_debiased.py CLI (unused args are accepted for compatibility)
    parser.add_argument("--n_list", type=str, default="", help="Comma-separated sample sizes, e.g. '500,1000'.")
    parser.add_argument("--m", type=int, default=30, help="Number of replicates per n.")
    parser.add_argument("--d", type=int, default=5, help="Feature dimension.")
    parser.add_argument(
        "--divergence",
        type=str,
        default="kth",
        help="Comma-separated divergences from {KL,TV,Hellinger,Chi2,JS,kth,tight_kth}.",
    )
    parser.add_argument(
        "--structural_type",
        type=str,
        default="nonlinear",
        choices=["linear", "nonlinear", "simpson", "cyclic", "cyclic2"],
        help="Data-generating process type.",
    )
    parser.add_argument("--base_seed", type=int, default=123, help="Base seed for training replicates.")
    parser.add_argument("--outdir", type=str, default="", help="Output directory.")
    parser.add_argument("--unique_save", action="store_true", help="Add unique timestamp suffix to outputs.")

    parser.add_argument("--n_eval", type=int, default=5000, help="Number of fixed evaluation points.")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="sample",
        choices=["sample", "grid_x0"],
        help="How to construct fixed X_eval: sample from DGP marginal or grid over X0.",
    )
    parser.add_argument("--eval_seed", type=int, default=2025, help="Seed for constructing X_eval (sample mode).")
    parser.add_argument("--x0_min", type=float, default=-3.14, help="Min X0 for grid_x0.")
    parser.add_argument("--x0_max", type=float, default=3.14, help="Max X0 for grid_x0.")
    parser.add_argument("--x_eval_fill", type=float, default=0.0, help="Fill value for non-X0 coordinates in grid_x0.")
    parser.add_argument("--x_range", type=float, default=2.0, help="x_range for some DGPs (if supported).")
    parser.add_argument(
        "--noise_dist",
        type=str,
        default=None,
        help="Optional noise distribution forwarded to generate_data if supported (e.g., 'normal', 't3').",
    )

    parser.add_argument(
        "--stat",
        "--width_stat",
        dest="width_stat",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Within-replicate aggregation over eval points for width (among valid points).",
    )
    parser.add_argument(
        "--stat_over_reps",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Across-replicate aggregation for centers.",
    )
    parser.add_argument(
        "--stat_grid",
        action="store_true",
        help="Run all combinations of --stat/--stat_over_reps in {mean,median}.",
    )
    parser.add_argument("--ci_alpha", type=float, default=0.05, help="Two-sided quantile band level across replicates.")

    parser.add_argument("--score_lambda", type=float, default=10.0, help="Penalty lambda for score.")
    parser.add_argument("--score_alpha", type=float, default=0.05, help="Target shortfall alpha (target=1-alpha).")

    parser.add_argument("--n_folds", type=int, default=2, help="CV folds.")
    parser.add_argument("--eps_propensity", type=float, default=1e-3, help="Propensity clipping.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Dual net epochs.")
    parser.add_argument("--xgb_n_jobs", type=int, default=-1, help="n_jobs for xgboost (-1 all cores).")
    parser.add_argument("--torch_threads", type=int, default=0, help="Torch intra-op threads (0 uses all cores).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for dual nets.")

    parser.add_argument("--n_oracle", type=int, default=200000, help="Oracle sample size for near-truth bounds.")
    parser.add_argument("--oracle_seed", type=int, default=-1, help="Seed for oracle data (negative uses base_seed).")
    parser.add_argument("--oracle_propensity_n_estimators", type=int, default=0, help="Override oracle propensity n_estimators.")
    parser.add_argument("--oracle_propensity_max_depth", type=int, default=0, help="Override oracle propensity max_depth.")
    parser.add_argument("--oracle_num_epochs", type=int, default=50, help="Oracle num_epochs (default 50).")
    parser.add_argument("--oracle_batch_size", type=int, default=1024, help="Oracle batch_size (default 1024).")

    parser.add_argument("--propensity_noise", action="store_true", help="Add N(n^{-beta}, n^{-beta}) noise to propensity.")
    parser.add_argument("--propensity_noise_beta", type=float, default=0.25, help="Noise rate beta for n^{-beta}.")

    parser.add_argument("--no_errorbar", action="store_true", help="Disable error bars in plots.")
    args = parser.parse_args()

    artifact_path = args.artifact
    if not artifact_path:
        if not args.stamp:
            raise ValueError("Provide --artifact or --stamp.")
        base_dir = args.artifact_dir or args.outdir or "."
        artifact_path = os.path.join(base_dir, f"plot_n_debiased_artifacts_{args.stamp}.pkl")

    with open(artifact_path, "rb") as f:
        art = pickle.load(f)

    outdir = args.outdir or os.path.dirname(os.path.abspath(artifact_path)) or "."
    os.makedirs(outdir, exist_ok=True)

    art_args = art.get("args", {})
    stamp = args.stamp or _infer_stamp(artifact_path, art)
    output_stamp = stamp if args.unique_save else ""

    replicate_rows = art.get("replicate_rows", [])
    if not replicate_rows:
        raise ValueError("Artifacts missing replicate_rows; cannot reconstruct plots.")

    base_divs = ["KL", "TV", "Hellinger", "Chi2", "JS"]
    div_list = art.get("divergences") or _parse_divergences(args.divergence, base_divs)

    n_list = sorted({row["n"] for row in replicate_rows})
    if not n_list and args.n_list:
        n_list = _parse_n_list(args.n_list)

    structural_type = art_args.get("structural_type", args.structural_type)
    eval_mode = art_args.get("eval_mode", args.eval_mode)
    n_eval = int(art_args.get("n_eval", args.n_eval))
    score_lambda = float(art_args.get("score_lambda", args.score_lambda))
    score_alpha = float(art_args.get("score_alpha", args.score_alpha))
    ci_alpha = float(art_args.get("ci_alpha", args.ci_alpha))

    args_meta = {
        "m": art_args.get("m", args.m),
        "d": art_args.get("d", args.d),
        "n_eval": n_eval,
        "structural_type": structural_type,
        "eval_mode": eval_mode,
        "score_lambda": score_lambda,
        "score_alpha": score_alpha,
    }

    def _plot_nuisance(rows: List[Dict[str, Any]], fname: str) -> str:
        plt.figure(figsize=(7.2, 4.2))
        for div in div_list:
            sub = [row for row in rows if row["divergence"] == div]
            xs = [row["n"] for row in sub]
            ys = [row["propensity_rmse_center"] for row in sub]
            if args.no_errorbar:
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

        plt.xlabel("Sample size n")
        plt.ylabel("Propensity RMSE")
        plt.title(f"Nuisance error vs n (struct={structural_type}, eval={eval_mode})")
        plt.legend()
        plt.tight_layout()
        path = _name_with_suffix(outdir, fname, "png", output_stamp)
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    def _plot_target(rows: List[Dict[str, Any]], fname: str) -> str:
        plt.figure(figsize=(7.2, 4.2))
        for div in div_list:
            sub = [row for row in rows if row["divergence"] == div]
            xs = [row["n"] for row in sub]
            yd = [row["err_up_debiased_center"] for row in sub]
            yn = [row["err_up_naive_center"] for row in sub]
            if args.no_errorbar:
                plt.plot(xs, yd, marker="o", linestyle="-", label=f"{div} debiased")
                plt.plot(xs, yn, marker="o", linestyle="--", label=f"{div} naive")
            else:
                lo_d = [row["err_up_debiased_ci_low"] for row in sub]
                hi_d = [row["err_up_debiased_ci_high"] for row in sub]
                lo_n = [row["err_up_naive_ci_low"] for row in sub]
                hi_n = [row["err_up_naive_ci_high"] for row in sub]
                plt.errorbar(xs, yd, yerr=[np.subtract(yd, lo_d), np.subtract(hi_d, yd)], marker="o", linestyle="-", capsize=3, label=f"{div} debiased")
                plt.errorbar(xs, yn, yerr=[np.subtract(yn, lo_n), np.subtract(hi_n, yn)], marker="o", linestyle="--", capsize=3, label=f"{div} naive")
        plt.xlabel("Sample size n")
        plt.ylabel("Target RMSE (upper bound)")
        plt.title(f"Target error vs n (struct={structural_type}, eval={eval_mode})")
        plt.legend()
        plt.tight_layout()
        path = _name_with_suffix(outdir, fname, "png", output_stamp)
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    def _plot_metric(rows: List[Dict[str, Any]], key_d: str, key_n: str, ylabel: str, title: str, fname: str) -> str:
        plt.figure(figsize=(7.2, 4.2))
        for div in div_list:
            sub = [row for row in rows if row["divergence"] == div]
            xs = [row["n"] for row in sub]
            yd = [row[f"{key_d}_center"] for row in sub]
            yn = [row[f"{key_n}_center"] for row in sub]
            if args.no_errorbar:
                plt.plot(xs, yd, marker="o", linestyle="-", label=f"{div} debiased")
                plt.plot(xs, yn, marker="o", linestyle="--", label=f"{div} naive")
            else:
                lo_d = [row[f"{key_d}_ci_low"] for row in sub]
                hi_d = [row[f"{key_d}_ci_high"] for row in sub]
                lo_n = [row[f"{key_n}_ci_low"] for row in sub]
                hi_n = [row[f"{key_n}_ci_high"] for row in sub]
                plt.errorbar(xs, yd, yerr=[np.subtract(yd, lo_d), np.subtract(hi_d, yd)], marker="o", linestyle="-", capsize=3, label=f"{div} debiased")
                plt.errorbar(xs, yn, yerr=[np.subtract(yn, lo_n), np.subtract(hi_n, yn)], marker="o", linestyle="--", capsize=3, label=f"{div} naive")
        plt.xlabel("Sample size n")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        path = _name_with_suffix(outdir, fname, "png", output_stamp)
        plt.savefig(path, dpi=200)
        plt.close()
        return path

    stat_pairs = [(args.width_stat, args.stat_over_reps)]
    if args.stat_grid:
        stat_pairs = [("mean", "mean"), ("mean", "median"), ("median", "mean"), ("median", "median")]

    runs: List[Dict[str, Any]] = []
    default_files: Dict[str, str] = {}

    rep_base = "load_plot_n_debiased_replicates"
    summary_base = "load_plot_n_debiased_summary"
    nuisance_base = "load_plot_n_debiased_nuisance"
    target_base = "load_plot_n_debiased_target"
    width_base = "load_plot_n_debiased_width"
    coverage_base = "load_plot_n_debiased_coverage"
    score_base = "load_plot_n_debiased_score"

    for stat_within, stat_over_reps in stat_pairs:
        suffix = _stat_suffix(stat_within, stat_over_reps) if len(stat_pairs) > 1 else ""
        summary_rows = _build_summary(
            replicate_rows=replicate_rows,
            div_list=div_list,
            n_list=n_list,
            stat_within=stat_within,
            stat_over_reps=stat_over_reps,
            ci_alpha=ci_alpha,
            args_meta=args_meta,
        )

        rep_path = _name_with_suffix(outdir, f"{rep_base}_{suffix}" if suffix else rep_base, "csv", output_stamp)
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(replicate_rows).to_csv(rep_path, index=False)
        except Exception:
            header = (
                "divergence,n,rep,seed,propensity_rmse,err_up_debiased,err_up_naive,coverage_debiased,coverage_naive,"
                "width_debiased,width_debiased_mean,width_debiased_median,width_naive,width_naive_mean,width_naive_median,"
                "score_debiased,score_debiased_mean,score_debiased_median,score_naive,score_naive_mean,score_naive_median,"
                "valid_rate_debiased,valid_rate_naive"
            )
            with open(rep_path, "w") as f:
                f.write(header + "\n")
                for row in replicate_rows:
                    f.write(
                        f"{row['divergence']},{row['n']},{row['rep']},{row['seed']},{row['propensity_rmse']},"
                        f"{row['err_up_debiased']},{row['err_up_naive']},{row['coverage_debiased']},{row['coverage_naive']},"
                        f"{row['width_debiased']},{row['width_debiased_mean']},{row['width_debiased_median']},"
                        f"{row['width_naive']},{row['width_naive_mean']},{row['width_naive_median']},"
                        f"{row['score_debiased']},{row['score_debiased_mean']},{row['score_debiased_median']},"
                        f"{row['score_naive']},{row['score_naive_mean']},{row['score_naive_median']},"
                        f"{row['valid_rate_debiased']},{row['valid_rate_naive']}\n"
                    )

        summary_path = _name_with_suffix(outdir, f"{summary_base}_{suffix}" if suffix else summary_base, "csv", output_stamp)
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        except Exception:
            header = (
                "divergence,n,propensity_rmse_center,propensity_rmse_ci_low,propensity_rmse_ci_high,"
                "err_up_debiased_center,err_up_debiased_ci_low,err_up_debiased_ci_high,"
                "err_up_naive_center,err_up_naive_ci_low,err_up_naive_ci_high,"
                "coverage_debiased_center,coverage_debiased_ci_low,coverage_debiased_ci_high,"
                "coverage_naive_center,coverage_naive_ci_low,coverage_naive_ci_high,"
                "width_debiased_center,width_debiased_ci_low,width_debiased_ci_high,"
                "width_naive_center,width_naive_ci_low,width_naive_ci_high,"
                "score_debiased_center,score_debiased_ci_low,score_debiased_ci_high,"
                "score_naive_center,score_naive_ci_low,score_naive_ci_high,"
                "valid_rate_debiased_center,valid_rate_debiased_ci_low,valid_rate_debiased_ci_high,"
                "valid_rate_naive_center,valid_rate_naive_ci_low,valid_rate_naive_ci_high,"
                "m,d,n_eval,structural_type,eval_mode,width_stat,stat_over_reps,score_lambda,score_alpha,ci_alpha"
            )
            with open(summary_path, "w") as f:
                f.write(header + "\n")
                for row in summary_rows:
                    f.write(
                        f"{row['divergence']},{row['n']},{row['propensity_rmse_center']},"
                        f"{row['propensity_rmse_ci_low']},{row['propensity_rmse_ci_high']},"
                        f"{row['err_up_debiased_center']},{row['err_up_debiased_ci_low']},{row['err_up_debiased_ci_high']},"
                        f"{row['err_up_naive_center']},{row['err_up_naive_ci_low']},{row['err_up_naive_ci_high']},"
                        f"{row['coverage_debiased_center']},{row['coverage_debiased_ci_low']},{row['coverage_debiased_ci_high']},"
                        f"{row['coverage_naive_center']},{row['coverage_naive_ci_low']},{row['coverage_naive_ci_high']},"
                        f"{row['width_debiased_center']},{row['width_debiased_ci_low']},{row['width_debiased_ci_high']},"
                        f"{row['width_naive_center']},{row['width_naive_ci_low']},{row['width_naive_ci_high']},"
                        f"{row['score_debiased_center']},{row['score_debiased_ci_low']},{row['score_debiased_ci_high']},"
                        f"{row['score_naive_center']},{row['score_naive_ci_low']},{row['score_naive_ci_high']},"
                        f"{row['valid_rate_debiased_center']},{row['valid_rate_debiased_ci_low']},{row['valid_rate_debiased_ci_high']},"
                        f"{row['valid_rate_naive_center']},{row['valid_rate_naive_ci_low']},{row['valid_rate_naive_ci_high']},"
                        f"{row['m']},{row['d']},{row['n_eval']},{row['structural_type']},{row['eval_mode']},"
                        f"{row['width_stat']},{row['stat_over_reps']},{row['score_lambda']},{row['score_alpha']},{row['ci_alpha']}\n"
                    )

        nuisance_fig = _plot_nuisance(summary_rows, f"{nuisance_base}_{suffix}" if suffix else nuisance_base)
        target_fig = _plot_target(summary_rows, f"{target_base}_{suffix}" if suffix else target_base)
        width_fig = _plot_metric(
            summary_rows,
            "width_debiased",
            "width_naive",
            ylabel="Width",
            title=f"Width vs n (struct={structural_type}, eval={eval_mode})",
            fname=f"{width_base}_{suffix}" if suffix else width_base,
        )
        cov_fig = _plot_metric(
            summary_rows,
            "coverage_debiased",
            "coverage_naive",
            ylabel="Coverage",
            title=f"Coverage vs n (struct={structural_type}, eval={eval_mode})",
            fname=f"{coverage_base}_{suffix}" if suffix else coverage_base,
        )
        score_fig = _plot_metric(
            summary_rows,
            "score_debiased",
            "score_naive",
            ylabel="Score (penalized width)",
            title=f"Score vs n (struct={structural_type}, eval={eval_mode})",
            fname=f"{score_base}_{suffix}" if suffix else score_base,
        )

        run_files = {
            "replicate_csv": rep_path,
            "summary_csv": summary_path,
            "nuisance_png": nuisance_fig,
            "target_png": target_fig,
            "width_png": width_fig,
            "coverage_png": cov_fig,
            "score_png": score_fig,
        }
        runs.append({"width_stat": stat_within, "stat_over_reps": stat_over_reps, "files": run_files})

        if stat_within == args.width_stat and stat_over_reps == args.stat_over_reps:
            default_files = run_files

    if not default_files and runs:
        default_files = runs[0]["files"]

    summary_json = {
        "timestamp": output_stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "files": dict(default_files, artifacts_pkl=artifact_path),
        "args": art_args,
        "divergences": div_list,
        "structural_type": structural_type,
        "eval_mode": eval_mode,
        "n_eval": n_eval,
    }
    if len(runs) > 1:
        summary_json["runs"] = runs

    summary_json_path = _name_with_suffix(outdir, "load_plot_n_debiased_summary", "json", output_stamp)
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2)

    plot_files = [p for k, p in default_files.items() if k.endswith("_png")]
    print(f"Reloaded plots saved: {', '.join(plot_files)}")
    print(f"Summary JSON saved: {summary_json_path}")


if __name__ == "__main__":
    main()
