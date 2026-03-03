from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .api import fit as fit_dataframe
from .artifacts import ArtifactPaths

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
    no_plots: bool,
    html: bool,
    ground_truth_effect: Optional[float] = None,
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
    artifacts = report.save(
        outdir,
        write_plots=not bool(no_plots),
        write_html=bool(html),
        ground_truth_effect=ground_truth_effect,
    )
    return DemoRunResult(
        name=name,
        outdir=outdir,
        artifacts=artifacts,
        n_rows=int(df.shape[0]),
        treatment=treatment,
        outcome=outcome,
        covariates=list(covariates),
        divergence=str(divergence),
        ground_truth_effect=ground_truth_effect,
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

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs: list[DemoRunResult] = []
    if scenario in {"toy", "both"}:
        toy_df = _make_toy_dataframe(n=120, seed=int(seed))
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
                no_plots=bool(no_plots),
                html=bool(html),
                ground_truth_effect=toy_truth,
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
                no_plots=bool(no_plots),
                html=bool(html),
                ground_truth_effect=ihdp_truth,
            )
        )

    summary_path = _write_demo_summary(outdir / "live_demo_summary.md", runs)
    return runs, summary_path


__all__ = ["DemoRunResult", "run_live_demo"]
