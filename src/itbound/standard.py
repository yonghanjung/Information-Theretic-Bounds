from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from fbound.estimators.causal_bound import compute_causal_bounds, prefit_propensity_cache

from .claims import compute_claims
from .config import build_phi, default_example_config
from .plotting import render_plots

DEFAULT_DIVERGENCES: tuple[str, ...] = ("KL", "JS", "Hellinger", "TV", "Chi2")


@dataclass(frozen=True)
class StandardRunResult:
    bounds: pd.DataFrame
    claims: Dict[str, Any]
    diagnostics: Dict[str, Any]
    ground_truth_plot: Dict[str, Any]
    bounds_path: Path
    summary_path: Path
    plot_paths: list[Path]
    html_path: Optional[Path]
    warnings: list[str]


def _normalize_divergences(divergences: Optional[Sequence[str] | str]) -> list[str]:
    if divergences is None:
        return list(DEFAULT_DIVERGENCES)
    if isinstance(divergences, str):
        vals = [v.strip() for v in divergences.split(",") if v.strip()]
    else:
        vals = [str(v).strip() for v in divergences if str(v).strip()]
    if not vals:
        raise ValueError("At least one divergence must be provided.")
    return vals


def _load_frame(*, dataframe: Optional[pd.DataFrame], csv_path: Optional[str | Path]) -> pd.DataFrame:
    if dataframe is None and csv_path is None:
        raise ValueError("Provide one data source: dataframe or csv_path.")
    if dataframe is not None and csv_path is not None:
        raise ValueError("Provide exactly one of dataframe or csv_path, not both.")

    if dataframe is not None:
        return dataframe.copy()

    path = Path(csv_path)  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _validate_columns(frame: pd.DataFrame, outcome_col: str, treatment_col: str, covariate_cols: Sequence[str]) -> None:
    required = [outcome_col, treatment_col, *covariate_cols]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(covariate_cols) == 0:
        raise ValueError("covariate_cols must be non-empty.")

    a_vals = np.unique(frame[treatment_col].to_numpy())
    if not set(a_vals.tolist()).issubset({0, 1}):
        raise ValueError(
            "Treatment column must be binary with values in {0,1} for the current estimator. "
            f"Found values: {a_vals.tolist()}"
        )


def _default_model_configs(seed: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    base = default_example_config(Path("itbound_bounds.csv"))
    dual_cfg = dict(base["dual_net_config"])
    fit_cfg = dict(base["fit_config"])
    fit_cfg["deterministic_torch"] = True
    return dual_cfg, fit_cfg


def _coerce_fit_value(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return value


def _compute_divergence_frames(
    *,
    frame: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: Sequence[str],
    divergences: Sequence[str],
    phi: str,
    propensity_model: str,
    m_model: str,
    dual_net_config: Dict[str, Any],
    fit_config: Dict[str, Any],
    seed: int,
) -> list[pd.DataFrame]:
    y = frame[outcome_col].to_numpy()
    a = frame[treatment_col].to_numpy()
    x = frame[list(covariate_cols)].to_numpy()

    phi_fn = build_phi(phi)
    prop_cache = prefit_propensity_cache(
        X=x,
        A=a,
        propensity_model=propensity_model,
        propensity_config=dict(fit_config["propensity_config"]),
        n_folds=int(fit_config["n_folds"]),
        seed=int(seed),
        eps_propensity=float(fit_config["eps_propensity"]),
    )

    out: list[pd.DataFrame] = []
    for div in divergences:
        df = compute_causal_bounds(
            Y=y,
            A=a,
            X=x,
            divergence=div,
            phi=phi_fn,
            propensity_model=propensity_model,
            m_model=m_model,
            dual_net_config=dual_net_config,
            fit_config=fit_config,
            seed=int(seed),
            GroundTruth=None,
            propensity_cache=prop_cache,
        )
        out.append(df.sort_values("i").reset_index(drop=True))
    return out


def _aggregate_divergence_frames(
    frames: Sequence[pd.DataFrame],
    *,
    mode: str,
    fixed_k: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if not frames:
        raise ValueError("frames must be non-empty")
    n = int(frames[0].shape[0])
    for frame in frames:
        if frame.shape[0] != n:
            raise ValueError("All divergence frames must have same number of rows.")

    lower_mat = np.stack([f["lower"].to_numpy(dtype=np.float64) for f in frames], axis=0)
    upper_mat = np.stack([f["upper"].to_numpy(dtype=np.float64) for f in frames], axis=0)
    valid_up_mat = np.stack([f["valid_up"].to_numpy(dtype=bool) for f in frames], axis=0)
    valid_lo_mat = np.stack([f["valid_lo"].to_numpy(dtype=bool) for f in frames], axis=0)

    inverted_chunks = []
    for f in frames:
        if "inverted" in f.columns:
            inverted_chunks.append(f["inverted"].to_numpy(dtype=bool))
        else:
            inverted_chunks.append(np.zeros((n,), dtype=bool))
    inverted_mat = np.stack(inverted_chunks, axis=0)

    lower_out = np.full((n,), np.nan, dtype=np.float64)
    upper_out = np.full((n,), np.nan, dtype=np.float64)
    k_used = np.zeros((n,), dtype=int)
    n_eff_up = np.zeros((n,), dtype=int)
    n_eff_lo = np.zeros((n,), dtype=int)
    invalid_up = np.zeros((n,), dtype=int)
    invalid_lo = np.zeros((n,), dtype=int)
    nonfinite_upper = np.zeros((n,), dtype=int)
    nonfinite_lower = np.zeros((n,), dtype=int)
    inverted_filtered = np.zeros((n,), dtype=int)
    tight_k_start = np.zeros((n,), dtype=int)

    if mode not in {"paper_adaptive_k", "fixed_k_endpoint", "tight_kth"}:
        raise ValueError("aggregation_mode must be one of: paper_adaptive_k, fixed_k_endpoint, tight_kth")
    fixed_k = int(fixed_k)
    if fixed_k <= 0:
        raise ValueError("fixed_k must be >= 1")

    def _tight_kth_endpoint(lo_vals: np.ndarray, up_vals: np.ndarray, *, k_cap: int) -> tuple[float, float, int, int]:
        """Experiment-style tight_kth over endpoint candidates for one row.

        Search from large k down to 1:
          lower = k-th smallest(lower candidates)
          upper = k-th largest(upper candidates)
        and return the first feasible pair (lower <= upper).
        """
        lo_vals = np.asarray(lo_vals, dtype=np.float64).reshape(-1)
        up_vals = np.asarray(up_vals, dtype=np.float64).reshape(-1)
        if lo_vals.size == 0 or up_vals.size == 0:
            return float("nan"), float("nan"), 0, 0

        lo_sorted = np.sort(lo_vals)  # ascending
        up_sorted = np.sort(up_vals)  # ascending
        k_start = int(min(lo_sorted.size, up_sorted.size))
        if int(k_cap) > 1:
            k_start = int(min(k_start, int(k_cap)))

        for k in range(int(k_start), 0, -1):
            lo_k = float(lo_sorted[k - 1])  # k-th smallest
            up_k = float(up_sorted[up_sorted.size - k])  # k-th largest
            if lo_k <= up_k:
                return lo_k, up_k, int(k), int(k_start)
        return float("nan"), float("nan"), 0, int(k_start)

    for i in range(n):
        lo_i = lower_mat[:, i]
        up_i = upper_mat[:, i]
        vu_i = valid_up_mat[:, i]
        vl_i = valid_lo_mat[:, i]
        inv_i = inverted_mat[:, i]

        invalid_up[i] = int(np.count_nonzero(~vu_i))
        invalid_lo[i] = int(np.count_nonzero(~vl_i))
        nonfinite_upper[i] = int(np.count_nonzero(~np.isfinite(up_i)))
        nonfinite_lower[i] = int(np.count_nonzero(~np.isfinite(lo_i)))
        inverted_filtered[i] = int(np.count_nonzero(inv_i))

        lo_mask = vl_i & np.isfinite(lo_i) & (~inv_i)
        up_mask = vu_i & np.isfinite(up_i) & (~inv_i)

        lo_candidates = np.sort(lo_i[lo_mask])[::-1]
        up_candidates = np.sort(up_i[up_mask])
        n_eff_lo[i] = int(lo_candidates.size)
        n_eff_up[i] = int(up_candidates.size)

        if lo_candidates.size == 0 or up_candidates.size == 0:
            continue

        if mode == "fixed_k_endpoint":
            if lo_candidates.size < fixed_k or up_candidates.size < fixed_k:
                continue
            lo_v = float(lo_candidates[fixed_k - 1])
            up_v = float(up_candidates[fixed_k - 1])
            if lo_v <= up_v:
                lower_out[i] = lo_v
                upper_out[i] = up_v
                k_used[i] = fixed_k
            continue

        if mode == "tight_kth":
            lo_v, up_v, k_v, k_start_v = _tight_kth_endpoint(
                lo_i[lo_mask],
                up_i[up_mask],
                k_cap=int(fixed_k),
            )
            tight_k_start[i] = int(k_start_v)
            if np.isfinite(lo_v) and np.isfinite(up_v):
                lower_out[i] = lo_v
                upper_out[i] = up_v
                k_used[i] = int(k_v)
            continue

        k_limit = min(lo_candidates.size, up_candidates.size)
        for k in range(1, int(k_limit) + 1):
            lo_v = float(lo_candidates[k - 1])
            up_v = float(up_candidates[k - 1])
            if lo_v <= up_v:
                lower_out[i] = lo_v
                upper_out[i] = up_v
                k_used[i] = k
                break

    valid_interval = np.isfinite(lower_out) & np.isfinite(upper_out) & (lower_out <= upper_out)
    width = np.where(valid_interval, upper_out - lower_out, np.nan)

    base = frames[0][["i"]].copy()
    base["lower"] = lower_out.astype(np.float32)
    base["upper"] = upper_out.astype(np.float32)
    base["width"] = width.astype(np.float32)
    base["valid_up"] = np.isfinite(upper_out)
    base["valid_lo"] = np.isfinite(lower_out)
    base["valid_interval"] = valid_interval
    base["divergence"] = ",".join(str(v) for v in sorted(set(f["divergence"].iloc[0] for f in frames)))

    diagnostics = {
        "aggregation_mode": mode,
        "k_used": k_used.tolist(),
        "tight_k_start": tight_k_start.tolist(),
        "n_eff_up": n_eff_up.tolist(),
        "n_eff_lo": n_eff_lo.tolist(),
        "filtered_counts": {
            "invalid_upper_mask": invalid_up.tolist(),
            "invalid_lower_mask": invalid_lo.tolist(),
            "non_finite_upper": nonfinite_upper.tolist(),
            "non_finite_lower": nonfinite_lower.tolist(),
            "inverted_interval_filtered": inverted_filtered.tolist(),
        },
    }
    return base, diagnostics


def _resolve_ground_truth_plot_inputs(
    frame: pd.DataFrame,
    *,
    ground_truth_col: Optional[str],
    ground_truth_effect: Optional[float],
    auto_ground_truth: bool,
) -> tuple[Optional[float], Optional[np.ndarray], Dict[str, Any], list[str]]:
    warnings: list[str] = []
    source = "none"
    truth_values: Optional[np.ndarray] = None
    truth_effect: Optional[float] = None

    def _coerce_numeric(col_name: str) -> Optional[np.ndarray]:
        vals = pd.to_numeric(frame[col_name], errors="coerce").to_numpy(dtype=np.float64)
        if vals.ndim != 1 or vals.shape[0] != frame.shape[0]:
            return None
        return vals

    if ground_truth_col:
        if ground_truth_col not in frame.columns:
            warnings.append(
                f"ground_truth_col '{ground_truth_col}' not found; skipping explicit ground truth overlay."
            )
        else:
            vals = _coerce_numeric(ground_truth_col)
            if vals is None:
                warnings.append(
                    f"ground_truth_col '{ground_truth_col}' could not be interpreted as a 1D numeric series."
                )
            else:
                finite = np.isfinite(vals)
                if not np.any(finite):
                    warnings.append(
                        f"ground_truth_col '{ground_truth_col}' has no finite numeric values; skipping explicit overlay."
                    )
                else:
                    truth_values = vals
                    source = "explicit_col"
                    truth_effect = float(np.mean(vals[finite]))

    if source == "none" and auto_ground_truth:
        auto_pairs = [("mu1", "mu0"), ("mu_1", "mu_0")]
        used_pair: Optional[tuple[str, str]] = None
        for lhs, rhs in auto_pairs:
            if lhs in frame.columns and rhs in frame.columns:
                used_pair = (lhs, rhs)
                break
        if used_pair is not None:
            mu1_vals = pd.to_numeric(frame[used_pair[0]], errors="coerce").to_numpy(dtype=np.float64)
            mu0_vals = pd.to_numeric(frame[used_pair[1]], errors="coerce").to_numpy(dtype=np.float64)
            delta = mu1_vals - mu0_vals
            finite = np.isfinite(delta)
            if np.any(finite):
                truth_values = delta
                source = "auto_mu1_mu0"
                truth_effect = float(np.mean(delta[finite]))
            else:
                warnings.append(
                    f"Auto ground-truth columns {used_pair[0]} and {used_pair[1]} contain no finite values."
                )

    if source == "none" and ground_truth_effect is not None:
        try:
            val = float(ground_truth_effect)
            if np.isfinite(val):
                truth_effect = val
                source = "explicit_effect"
            else:
                warnings.append("ground_truth_effect is non-finite; skipping scalar ground truth overlay.")
        except Exception:
            warnings.append("ground_truth_effect could not be parsed as float; skipping scalar ground truth overlay.")

    n_truth_points = 0
    if truth_values is not None:
        n_truth_points = int(np.count_nonzero(np.isfinite(truth_values)))
    elif truth_effect is not None:
        n_truth_points = int(frame.shape[0])

    metadata = {
        "source": source,
        "ground_truth_col": (ground_truth_col if ground_truth_col else None),
        "auto_ground_truth": bool(auto_ground_truth),
        "n_truth_points": int(n_truth_points),
        "ground_truth_effect": (float(truth_effect) if truth_effect is not None else None),
        "warnings": list(warnings),
    }
    return truth_effect, truth_values, metadata, warnings


def _write_plots(
    bounds: pd.DataFrame,
    outdir: Path,
    *,
    ground_truth_effect: Optional[float] = None,
    ground_truth_values: Optional[np.ndarray] = None,
) -> tuple[list[Path], list[str]]:
    if ground_truth_values is not None and int(ground_truth_values.shape[0]) != int(bounds.shape[0]):
        raise ValueError(
            "ground_truth_values length must match bounds rows "
            f"({ground_truth_values.shape[0]} vs {bounds.shape[0]})."
        )
    plot_paths = render_plots(bounds, outdir, ground_truth_effect=ground_truth_effect, ground_truth_values=ground_truth_values)
    warnings: list[str] = []
    if not plot_paths:
        warnings.append(
            "Plot generation skipped: matplotlib unavailable. "
            "Install extras: pip install itbound[experiments]"
        )
    return plot_paths, warnings


def _write_html_report(
    *,
    outdir: Path,
    claims: Dict[str, Any],
    diagnostics: Dict[str, Any],
    summary_path: Path,
    plot_paths: Sequence[Path],
    warnings: Sequence[str],
) -> Path:
    rows = []
    for key, value in claims.items():
        rows.append(f"<tr><th>{key}</th><td>{value}</td></tr>")

    plot_html = ""
    if plot_paths:
        tags = [f'<img src="{p.name}" alt="{p.name}" style="max-width:100%;height:auto;" />' for p in plot_paths]
        plot_html = "\n".join(tags)

    warnings_html = ""
    if warnings:
        entries = "".join(f"<li>{w}</li>" for w in warnings)
        warnings_html = f"<h2>Warnings</h2><ul>{entries}</ul>"

    diag_json = json.dumps(_to_jsonable(diagnostics), indent=2)[:8000]

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>itbound standard report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
    th, td {{ border: 1px solid #ddd; text-align: left; padding: 8px; }}
    th {{ background: #f5f5f5; width: 260px; }}
    pre {{ background: #f7f7f7; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>itbound Standard Report</h1>
  <p>Summary JSON: <code>{summary_path.name}</code></p>
  <h2>Claims</h2>
  <table>{''.join(rows)}</table>
  {warnings_html}
  <h2>Diagnostics (JSON excerpt)</h2>
  <pre>{diag_json}</pre>
  <h2>Plots</h2>
  {plot_html}
</body>
</html>
"""
    html_path = outdir / "report.html"
    html_path.write_text(html)
    return html_path


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def run_standard_bounds(
    *,
    dataframe: Optional[pd.DataFrame] = None,
    csv_path: Optional[str | Path] = None,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: Sequence[str],
    divergences: Optional[Sequence[str] | str] = None,
    phi: str = "identity",
    propensity_model: str = "logistic",
    m_model: str = "linear",
    dual_overrides: Optional[Dict[str, Any]] = None,
    fit_overrides: Optional[Dict[str, Any]] = None,
    seed: int = 123,
    aggregation_mode: str = "paper_adaptive_k",
    fixed_k: int = 1,
    ground_truth_col: Optional[str] = None,
    ground_truth_effect: Optional[float] = None,
    auto_ground_truth: bool = True,
    outdir: str | Path = "itbound-standard-out",
    write_plots: bool = True,
    write_html: bool = False,
) -> StandardRunResult:
    frame = _load_frame(dataframe=dataframe, csv_path=csv_path)
    _validate_columns(frame, outcome_col, treatment_col, covariate_cols)
    truth_effect, truth_values, gt_plot_meta, gt_warnings = _resolve_ground_truth_plot_inputs(
        frame,
        ground_truth_col=ground_truth_col,
        ground_truth_effect=ground_truth_effect,
        auto_ground_truth=bool(auto_ground_truth),
    )

    div_list = _normalize_divergences(divergences)

    dual_cfg, fit_cfg = _default_model_configs(seed=int(seed))
    if dual_overrides:
        dual_cfg.update(dict(dual_overrides))
    if fit_overrides:
        for key, value in dict(fit_overrides).items():
            fit_cfg[key] = _coerce_fit_value(value)

    div_frames = _compute_divergence_frames(
        frame=frame,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        covariate_cols=covariate_cols,
        divergences=div_list,
        phi=phi,
        propensity_model=propensity_model,
        m_model=m_model,
        dual_net_config=dual_cfg,
        fit_config=fit_cfg,
        seed=int(seed),
    )

    bounds_df, diagnostics = _aggregate_divergence_frames(
        div_frames,
        mode=aggregation_mode,
        fixed_k=int(fixed_k),
    )
    claims = compute_claims(bounds_df)

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bounds_path = output_dir / "bounds.csv"
    summary_path = output_dir / "summary.json"
    bounds_df.to_csv(bounds_path, index=False)

    plot_paths: list[Path] = []
    warnings: list[str] = list(gt_warnings)
    if truth_values is not None and int(truth_values.shape[0]) != int(bounds_df.shape[0]):
        raise ValueError(
            "ground truth vector length must match bounds rows "
            f"({truth_values.shape[0]} vs {bounds_df.shape[0]})."
        )
    if write_plots:
        pp, ww = _write_plots(
            bounds_df,
            output_dir,
            ground_truth_effect=truth_effect,
            ground_truth_values=truth_values,
        )
        plot_paths.extend(pp)
        warnings.extend(ww)

    gt_plot_meta = dict(gt_plot_meta)
    gt_warning_list = list(gt_plot_meta.get("warnings", []))
    for msg in warnings:
        if msg not in gt_warning_list:
            gt_warning_list.append(msg)
    gt_plot_meta["warnings"] = gt_warning_list

    summary_payload = {
        "claims": claims,
        "diagnostics": diagnostics,
        "run_config": {
            "divergences": div_list,
            "phi": phi,
            "propensity_model": propensity_model,
            "m_model": m_model,
            "seed": int(seed),
            "aggregation_mode": aggregation_mode,
            "fixed_k": int(fixed_k),
            "n_rows": int(frame.shape[0]),
            "outcome_col": outcome_col,
            "treatment_col": treatment_col,
            "covariate_cols": list(covariate_cols),
            "ground_truth_plot": gt_plot_meta,
        },
        "warnings": warnings,
        "artifacts": {
            "bounds_csv": str(bounds_path),
            "summary_json": str(summary_path),
            "plots": [str(p) for p in plot_paths],
            "html_report": str((output_dir / "report.html")) if write_html else None,
        },
    }
    summary_path.write_text(json.dumps(_to_jsonable(summary_payload), indent=2, allow_nan=False))

    html_path: Optional[Path] = None
    if write_html:
        html_path = _write_html_report(
            outdir=output_dir,
            claims=claims,
            diagnostics=diagnostics,
            summary_path=summary_path,
            plot_paths=plot_paths,
            warnings=warnings,
        )

    return StandardRunResult(
        bounds=bounds_df,
        claims=claims,
        diagnostics=diagnostics,
        ground_truth_plot=gt_plot_meta,
        bounds_path=bounds_path,
        summary_path=summary_path,
        plot_paths=plot_paths,
        html_path=html_path,
        warnings=warnings,
    )


__all__ = ["DEFAULT_DIVERGENCES", "StandardRunResult", "run_standard_bounds"]
