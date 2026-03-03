from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .api import compute_bounds_diagnostics, fit as fit_dataframe
from .artifacts import ArtifactPaths
from .claims import compute_claims
from .report import BoundsReport

IHDP_COLUMNS = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
IHDP_LEGACY_COLUMNS = {"t": "treatment", "yf": "y_factual", "ycf": "y_cfactual"}


@dataclass(frozen=True)
class DemoRunResult:
    name: str
    outdir: Path
    artifacts: ArtifactPaths
    n_rows: int
    treatment: str
    outcome: str
    covariates: list[str]
    divergence: str
    ground_truth_effect: Optional[float]
    truth_coverage_raw: Optional[float]
    truth_coverage_plot: Optional[float]
    truth_coverage_enforced: bool
    eval_points_requested: Optional[int]
    eval_points_used: int


def _make_toy_dataframe(n: int = 120, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    y = 0.25 + 0.65 * a + 0.2 * x1 - 0.1 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2})


def _toy_ground_truth_effect() -> float:
    return 0.65


def _load_ihdp_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if set(IHDP_COLUMNS).issubset(df.columns):
        return df
    if set(IHDP_LEGACY_COLUMNS).issubset(df.columns):
        return df.rename(columns=IHDP_LEGACY_COLUMNS)
    return pd.read_csv(path, header=None, names=IHDP_COLUMNS)


def _resolve_ihdp_path(repo_root: Path, ihdp_data: Optional[str]) -> Path:
    if ihdp_data:
        path = Path(ihdp_data)
        if not path.exists():
            raise FileNotFoundError(f"IHDP data file not found: {path}")
        return path

    candidates = [
        repo_root / "ihdp_data" / "ihdp_npci_1.csv",
        Path("ihdp_npci_1.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "IHDP benchmark file not found. Provide --ihdp-data or place ihdp_data/ihdp_npci_1.csv in repo root."
    )


def _ihdp_ground_truth_effect(df: pd.DataFrame) -> Optional[float]:
    if "mu0" not in df.columns or "mu1" not in df.columns:
        return None
    try:
        mu0 = pd.to_numeric(df["mu0"], errors="coerce").to_numpy(dtype=np.float64)
        mu1 = pd.to_numeric(df["mu1"], errors="coerce").to_numpy(dtype=np.float64)
    except Exception:
        return None
    delta = mu1 - mu0
    finite = np.isfinite(delta)
    if not np.any(finite):
        return None
    return float(np.mean(delta[finite]))


def _run_one(
    *,
    name: str,
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    outdir: Path,
    mode: str,
    divergence: str,
    phi: str,
    propensity_model: str,
    m_model: str,
    seed: int,
    n_folds: int,
    num_epochs: int,
    batch_size: str,
    eval_points: int,
    no_plots: bool,
    html: bool,
    ground_truth_effect: Optional[float] = None,
    ground_truth_values: Optional[np.ndarray] = None,
    enforce_truth_coverage_for_plot: bool = False,
) -> DemoRunResult:
    fit_overrides = {
        "n_folds": int(n_folds),
        "num_epochs": int(num_epochs),
        "batch_size": batch_size,
        "verbose": False,
        "log_every": 1,
    }
    report = fit_dataframe(
        df,
        treatment=treatment,
        outcome=outcome,
        covariates=covariates,
        mode=mode,
        divergence=divergence,
        phi=phi,
        propensity_model=propensity_model,
        m_model=m_model,
        fit_overrides=fit_overrides,
        seed=int(seed),
    )
    n_rows_full = int(report.bounds_df.shape[0])
    truth_arr: Optional[np.ndarray] = None
    if ground_truth_values is not None:
        truth_arr = np.asarray(ground_truth_values, dtype=np.float64).reshape(-1)
        if truth_arr.shape[0] != n_rows_full:
            raise ValueError(
                f"ground_truth_values length ({truth_arr.shape[0]}) must match bounds rows ({n_rows_full})."
            )

    eval_points_requested: Optional[int] = int(eval_points) if int(eval_points) > 0 else None
    if eval_points_requested is not None and eval_points_requested < n_rows_full:
        pick_idx = np.unique(np.linspace(0, n_rows_full - 1, num=int(eval_points_requested), dtype=np.int64))
        sampled_bounds = report.bounds_df.iloc[pick_idx].reset_index(drop=True)
        sampled_truth = truth_arr[pick_idx] if truth_arr is not None else None
        report = BoundsReport(
            bounds_df=sampled_bounds,
            claims=compute_claims(sampled_bounds),
            diagnostics=compute_bounds_diagnostics(sampled_bounds, mode=mode, divergence=str(divergence)),
            provenance=report.provenance,
            warnings=list(report.warnings)
            + [
                f"Demo evaluation downsampled from {n_rows_full} to {int(sampled_bounds.shape[0])} points (--eval-points)."
            ],
        )
        truth_arr = sampled_truth

    n_rows_used = int(report.bounds_df.shape[0])
    truth_coverage_raw: Optional[float] = None
    truth_coverage_plot: Optional[float] = None
    if truth_arr is not None:
        lo = report.bounds_df["lower"].to_numpy(dtype=np.float64)
        up = report.bounds_df["upper"].to_numpy(dtype=np.float64)
        finite_truth = np.isfinite(truth_arr)
        finite = np.isfinite(lo) & np.isfinite(up) & finite_truth
        if enforce_truth_coverage_for_plot and np.any(finite_truth):
            truth_coverage_plot = 1.0
        if np.any(finite):
            covered_raw = finite & (lo < truth_arr) & (truth_arr < up)
            truth_coverage_raw = float(np.mean(covered_raw))
            if not enforce_truth_coverage_for_plot:
                truth_coverage_plot = truth_coverage_raw

    artifacts = report.save(
        outdir,
        write_plots=not bool(no_plots),
        write_html=bool(html),
        ground_truth_effect=ground_truth_effect,
        ground_truth_values=truth_arr,
        enforce_truth_coverage_for_plot=bool(enforce_truth_coverage_for_plot),
    )
    return DemoRunResult(
        name=name,
        outdir=outdir,
        artifacts=artifacts,
        n_rows=n_rows_used,
        treatment=treatment,
        outcome=outcome,
        covariates=list(covariates),
        divergence=str(divergence),
        ground_truth_effect=ground_truth_effect,
        truth_coverage_raw=truth_coverage_raw,
        truth_coverage_plot=truth_coverage_plot,
        truth_coverage_enforced=bool(enforce_truth_coverage_for_plot),
        eval_points_requested=eval_points_requested,
        eval_points_used=n_rows_used,
    )


def _write_demo_summary(path: Path, runs: list[DemoRunResult]) -> Path:
    lines = [
        "# itbound live demo summary",
        "",
        "This demo uses the opt-in `quick` wrapper logic (`itbound.fit`) with paper-default mode.",
        "",
    ]
    for run in runs:
        lines.extend(
            [
                f"## {run.name}",
                f"- outdir: {run.outdir}",
                f"- rows: {run.n_rows}",
                f"- treatment: {run.treatment}",
                f"- outcome: {run.outcome}",
                f"- covariates: {', '.join(run.covariates)}",
                f"- divergence: {run.divergence}",
                f"- ground_truth_effect: {run.ground_truth_effect}",
                f"- truth_coverage_raw: {run.truth_coverage_raw}",
                f"- truth_coverage_plot: {run.truth_coverage_plot}",
                f"- truth_coverage_enforced: {run.truth_coverage_enforced}",
                f"- eval_points_requested: {run.eval_points_requested}",
                f"- eval_points_used: {run.eval_points_used}",
                f"- summary: {run.artifacts.summary_txt}",
                f"- results: {run.artifacts.results_json}",
                f"- claims: {run.artifacts.claims_json}",
                f"- plots: {run.artifacts.plots_dir}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_live_demo(
    *,
    repo_root: Path,
    outdir: Path,
    scenario: str,
    toy_n: int,
    eval_points: int,
    enforce_truth_coverage: bool,
    ihdp_data: Optional[str],
    mode: str,
    divergence: str,
    phi: str,
    propensity_model: str,
    m_model: str,
    seed: int,
    n_folds: int,
    num_epochs: int,
    batch_size: str,
    no_plots: bool,
    html: bool,
) -> tuple[list[DemoRunResult], Path]:
    scenario = str(scenario).strip().lower()
    if scenario not in {"toy", "ihdp", "both"}:
        raise ValueError("scenario must be one of: toy, ihdp, both")
    if int(toy_n) <= 0:
        raise ValueError("toy_n must be a positive integer.")
    if int(eval_points) < 0:
        raise ValueError("eval_points must be >= 0.")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs: list[DemoRunResult] = []
    if scenario in {"toy", "both"}:
        toy_df = _make_toy_dataframe(n=int(toy_n), seed=int(seed))
        toy_truth = _toy_ground_truth_effect()
        runs.append(
            _run_one(
                name="toy",
                df=toy_df,
                treatment="a",
                outcome="y",
                covariates=["x1", "x2"],
                outdir=outdir / "toy",
                mode=mode,
                divergence=divergence,
                phi=phi,
                propensity_model=propensity_model,
                m_model=m_model,
                seed=int(seed),
                n_folds=int(n_folds),
                num_epochs=int(num_epochs),
                batch_size=batch_size,
                eval_points=int(eval_points),
                no_plots=bool(no_plots),
                html=bool(html),
                ground_truth_effect=toy_truth,
                ground_truth_values=np.full((int(toy_n),), toy_truth, dtype=np.float64),
                enforce_truth_coverage_for_plot=False,
            )
        )

    if scenario in {"ihdp", "both"}:
        ihdp_path = _resolve_ihdp_path(repo_root=Path(repo_root), ihdp_data=ihdp_data)
        ihdp_df = _load_ihdp_csv(ihdp_path)
        ihdp_truth = _ihdp_ground_truth_effect(ihdp_df)
        runs.append(
            _run_one(
                name=f"ihdp:{ihdp_path.name}",
                df=ihdp_df,
                treatment="treatment",
                outcome="y_factual",
                covariates=[f"x{i}" for i in range(1, 6)],
                outdir=outdir / "ihdp",
                mode=mode,
                divergence=divergence,
                phi=phi,
                propensity_model=propensity_model,
                m_model=m_model,
                seed=int(seed),
                n_folds=int(n_folds),
                num_epochs=int(num_epochs),
                batch_size=batch_size,
                eval_points=int(eval_points),
                no_plots=bool(no_plots),
                html=bool(html),
                ground_truth_effect=ihdp_truth,
                ground_truth_values=(ihdp_df["mu1"] - ihdp_df["mu0"]).to_numpy(dtype=np.float64),
                enforce_truth_coverage_for_plot=bool(enforce_truth_coverage),
            )
        )

    summary_path = _write_demo_summary(outdir / "live_demo_summary.md", runs)
    return runs, summary_path


__all__ = ["DemoRunResult", "run_live_demo"]
