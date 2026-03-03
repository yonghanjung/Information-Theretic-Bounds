from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from fbound.estimators.causal_bound import compute_causal_bounds

from .api import fit as fit_dataframe
from .config import ConfigError, build_phi, default_example_config, load_config, resolve_data
from .artifacts import build_provenance, build_result_payload, write_artifact_contract
from .live_demo import run_live_demo
from .standard import DEFAULT_DIVERGENCES, run_standard_bounds
from . import __version__

def _find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
        if (candidate / "scripts" / "reproduce_final_arxiv_plots.py").is_file():
            return candidate
    return start.parents[2]


ROOT = _find_repo_root(Path(__file__).resolve().parent)


def _write_bounds(cfg_path: Path, output_override: Optional[Path]) -> Path:
    cfg = load_config(cfg_path)
    y, a, x, ground_truth = resolve_data(cfg)
    phi_fn = build_phi(cfg["phi"])

    df = compute_causal_bounds(
        Y=y,
        A=a,
        X=x,
        divergence=cfg["divergence"],
        phi=phi_fn,
        propensity_model=cfg["propensity_model"],
        m_model=cfg["m_model"],
        dual_net_config=cfg["dual_net_config"],
        fit_config=cfg["fit_config"],
        seed=int(cfg["seed"]),
        GroundTruth=ground_truth,
    )

    out_path = output_override or Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def _example(out_path: Path) -> Path:
    cfg = default_example_config(out_path)
    y, a, x, ground_truth = resolve_data(cfg)
    phi_fn = build_phi(cfg["phi"])

    df = compute_causal_bounds(
        Y=y,
        A=a,
        X=x,
        divergence=cfg["divergence"],
        phi=phi_fn,
        propensity_model=cfg["propensity_model"],
        m_model=cfg["m_model"],
        dual_net_config=cfg["dual_net_config"],
        fit_config=cfg["fit_config"],
        seed=int(cfg["seed"]),
        GroundTruth=ground_truth,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def _reproduce(dry_run: bool, outdir: str, only: str, final_arxiv_dir: str) -> int:
    script_path = ROOT / "scripts" / "reproduce_final_arxiv_plots.py"
    use_module = not script_path.is_file()
    if dry_run:
        if use_module:
            cmd = [sys.executable, "-m", "itbound.reproduce_final_arxiv_plots", "--dry-run"]
        else:
            cmd = [sys.executable, str(script_path), "--dry-run"]
        if outdir:
            cmd += ["--outdir", outdir]
        if only:
            cmd += ["--only", only]
        if final_arxiv_dir:
            cmd += ["--final_arxiv_dir", final_arxiv_dir]
        print(" ".join(cmd))
        return 0

    try:
        import matplotlib  # noqa: F401
    except Exception:
        print(
            "Missing optional dependency 'matplotlib'. Install extras: pip install itbound[experiments]",
            file=sys.stderr,
        )
        return 2

    if use_module:
        cmd = [sys.executable, "-m", "itbound.reproduce_final_arxiv_plots"]
    else:
        cmd = [sys.executable, str(script_path)]
    if outdir:
        cmd += ["--outdir", outdir]
    if only:
        cmd += ["--only", only]
    if final_arxiv_dir:
        cmd += ["--final_arxiv_dir", final_arxiv_dir]

    subprocess.run(cmd, check=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="itbound", description="itbound CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run bounds from config")
    run_p.add_argument("--config", required=True, type=str, help="Path to YAML/JSON config")
    run_p.add_argument("--out", required=False, type=str, default="", help="Override output path")

    example_p = sub.add_parser("example", help="Run a small synthetic example")
    example_p.add_argument("--out", required=False, type=str, default="itbound_example.csv")

    repro_p = sub.add_parser("reproduce", help="Reproduce arXiv plots")
    repro_p.add_argument("--final-arxiv-dir", dest="final_arxiv_dir", default="experiments/final-arxiv")
    repro_p.add_argument("--outdir", default="")
    repro_p.add_argument("--only", default="")
    repro_p.add_argument("--dry-run", action="store_true")

    std_p = sub.add_parser("standard", help="Standard library run from CSV columns")
    std_p.add_argument("--csv", required=True, type=str, help="Input CSV path")
    std_p.add_argument("--y-col", required=True, type=str, help="Outcome column")
    std_p.add_argument("--a-col", required=True, type=str, help="Treatment column (binary)")
    std_p.add_argument("--x-cols", required=True, type=str, help="Comma-separated covariate columns")
    std_p.add_argument("--outdir", default="itbound-standard-out", type=str)
    std_p.add_argument(
        "--divergences",
        default=",".join(DEFAULT_DIVERGENCES),
        type=str,
        help="Comma-separated divergences (default: KL,JS,Hellinger,TV,Chi2)",
    )
    std_p.add_argument("--phi", default="identity", type=str)
    std_p.add_argument("--propensity-model", default="logistic", type=str)
    std_p.add_argument("--m-model", default="linear", type=str)
    std_p.add_argument("--seed", default=123, type=int)
    std_p.add_argument("--n-folds", default=2, type=int)
    std_p.add_argument("--num-epochs", default=3, type=int)
    std_p.add_argument("--batch-size", default="auto", type=str)
    std_p.add_argument("--aggregation-mode", default="paper_adaptive_k", choices=["paper_adaptive_k", "fixed_k_endpoint"])
    std_p.add_argument("--fixed-k", default=1, type=int)
    std_p.add_argument("--no-plots", action="store_true")
    std_p.add_argument("--html", action="store_true")

    art_p = sub.add_parser("artifacts", help="Write fixed artifact contract outputs")
    art_p.add_argument("--csv", required=True, type=str, help="Input CSV path")
    art_p.add_argument("--y-col", required=True, type=str, help="Outcome column")
    art_p.add_argument("--a-col", required=True, type=str, help="Treatment column (binary)")
    art_p.add_argument("--x-cols", required=True, type=str, help="Comma-separated covariate columns")
    art_p.add_argument("--outdir", default="itbound-artifacts-out", type=str)
    art_p.add_argument(
        "--divergences",
        default=",".join(DEFAULT_DIVERGENCES),
        type=str,
        help="Comma-separated divergences (default: KL,JS,Hellinger,TV,Chi2)",
    )
    art_p.add_argument("--phi", default="identity", type=str)
    art_p.add_argument("--propensity-model", default="logistic", type=str)
    art_p.add_argument("--m-model", default="linear", type=str)
    art_p.add_argument("--seed", default=123, type=int)
    art_p.add_argument("--n-folds", default=2, type=int)
    art_p.add_argument("--num-epochs", default=3, type=int)
    art_p.add_argument("--batch-size", default="auto", type=str)
    art_p.add_argument("--aggregation-mode", default="paper_adaptive_k", choices=["paper_adaptive_k", "fixed_k_endpoint"])
    art_p.add_argument("--fixed-k", default=1, type=int)
    art_p.add_argument("--assumptions", default="", type=str, help="Optional explicit assumptions text")
    art_p.add_argument("--no-plots", action="store_true")
    art_p.add_argument("--html", action="store_true")

    quick_p = sub.add_parser(
        "quick",
        help="Wrapper around itbound.fit (paper-default) that writes artifact contract outputs",
    )
    quick_p.add_argument("--data", required=True, type=str, help="Input CSV path")
    quick_p.add_argument("--treatment", required=True, type=str, help="Treatment column (binary)")
    quick_p.add_argument("--outcome", required=True, type=str, help="Outcome column")
    quick_p.add_argument("--covariates", required=True, type=str, help="Comma-separated covariate columns")
    quick_p.add_argument("--outdir", default="itbound-quick-out", type=str)
    quick_p.add_argument(
        "--mode",
        default="paper-default",
        choices=["paper-default"],
        help="Wrapper mode. Default keeps paper-equivalent math.",
    )
    quick_p.add_argument("--divergence", default="KL", type=str)
    quick_p.add_argument("--phi", default="identity", type=str)
    quick_p.add_argument("--propensity-model", default="logistic", type=str)
    quick_p.add_argument("--m-model", default="linear", type=str)
    quick_p.add_argument("--seed", default=123, type=int)
    quick_p.add_argument("--n-folds", default=2, type=int)
    quick_p.add_argument("--num-epochs", default=3, type=int)
    quick_p.add_argument("--batch-size", default="auto", type=str)
    quick_p.add_argument("--no-plots", action="store_true")
    quick_p.add_argument("--html", action="store_true")

    demo_p = sub.add_parser(
        "demo",
        help="Run a live demo on toy data and/or IHDP benchmark using itbound.fit wrapper",
    )
    demo_p.add_argument(
        "--scenario",
        default="both",
        choices=["toy", "ihdp", "both"],
        help="Demo scenario to run.",
    )
    demo_p.add_argument(
        "--ihdp-data",
        default="",
        type=str,
        help="Optional path to IHDP CSV (default: repo ihdp_data/ihdp_npci_1.csv if available).",
    )
    demo_p.add_argument("--outdir", default="itbound-live-demo", type=str)
    demo_p.add_argument(
        "--mode",
        default="paper-default",
        choices=["paper-default"],
        help="Wrapper mode. Default keeps paper-equivalent math.",
    )
    demo_p.add_argument("--divergence", default="TV", type=str, help="Default TV for stable live demo intervals.")
    demo_p.add_argument("--phi", default="identity", type=str)
    demo_p.add_argument("--propensity-model", default="logistic", type=str)
    demo_p.add_argument("--m-model", default="linear", type=str)
    demo_p.add_argument("--seed", default=123, type=int)
    demo_p.add_argument("--n-folds", default=2, type=int)
    demo_p.add_argument("--num-epochs", default=2, type=int)
    demo_p.add_argument("--batch-size", default="8", type=str)
    demo_p.add_argument("--no-plots", action="store_true")
    demo_p.add_argument("--html", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run":
            cfg_path = Path(args.config)
            out_override = Path(args.out) if args.out else None
            out_path = _write_bounds(cfg_path, out_override)
            print(f"saved: {out_path}")
            return 0
        if args.command == "example":
            out_path = Path(args.out)
            out_path = _example(out_path)
            print(f"saved: {out_path}")
            return 0
        if args.command == "reproduce":
            return _reproduce(args.dry_run, args.outdir, args.only, args.final_arxiv_dir)
        if args.command == "standard":
            x_cols = [x.strip() for x in str(args.x_cols).split(",") if x.strip()]
            if not x_cols:
                raise ConfigError("x-cols must contain at least one covariate column.")

            fit_overrides = {
                "n_folds": int(args.n_folds),
                "num_epochs": int(args.num_epochs),
                "batch_size": args.batch_size,
                "verbose": False,
                "log_every": 1,
            }
            result = run_standard_bounds(
                csv_path=args.csv,
                outcome_col=args.y_col,
                treatment_col=args.a_col,
                covariate_cols=x_cols,
                divergences=args.divergences,
                phi=args.phi,
                propensity_model=args.propensity_model,
                m_model=args.m_model,
                fit_overrides=fit_overrides,
                seed=int(args.seed),
                aggregation_mode=args.aggregation_mode,
                fixed_k=int(args.fixed_k),
                outdir=args.outdir,
                write_plots=not bool(args.no_plots),
                write_html=bool(args.html),
            )
            print(f"saved: {result.bounds_path}")
            print(f"saved: {result.summary_path}")
            if result.html_path is not None:
                print(f"saved: {result.html_path}")
            for warning in result.warnings:
                print(f"warning: {warning}", file=sys.stderr)
            return 0
        if args.command == "artifacts":
            x_cols = [x.strip() for x in str(args.x_cols).split(",") if x.strip()]
            if not x_cols:
                raise ConfigError("x-cols must contain at least one covariate column.")

            fit_overrides = {
                "n_folds": int(args.n_folds),
                "num_epochs": int(args.num_epochs),
                "batch_size": args.batch_size,
                "verbose": False,
                "log_every": 1,
            }
            std = run_standard_bounds(
                csv_path=args.csv,
                outcome_col=args.y_col,
                treatment_col=args.a_col,
                covariate_cols=x_cols,
                divergences=args.divergences,
                phi=args.phi,
                propensity_model=args.propensity_model,
                m_model=args.m_model,
                fit_overrides=fit_overrides,
                seed=int(args.seed),
                aggregation_mode=args.aggregation_mode,
                fixed_k=int(args.fixed_k),
                outdir=args.outdir,
                write_plots=not bool(args.no_plots),
                write_html=False,
            )

            assumptions: object
            if args.assumptions:
                assumptions = args.assumptions
            else:
                assumptions = {
                    "paper_default_math": True,
                    "treatment_domain": "binary_0_1",
                    "aggregation_mode": args.aggregation_mode,
                    "divergences": [v.strip() for v in str(args.divergences).split(",") if v.strip()],
                }

            diag_payload = {
                "standard": std.diagnostics,
                "missingness": {"status": "stub_v0", "note": "missingness checks not implemented in schema v0"},
                "overlap": {"status": "stub_v0", "note": "overlap checks not implemented in schema v0"},
                "warnings": list(std.warnings),
            }
            prov = build_provenance(
                package_version=__version__,
                random_seed=int(args.seed),
                assumptions=assumptions,
                data_source=args.csv,
                repo_root=ROOT,
            )
            payload = build_result_payload(bounds_df=std.bounds, diagnostics=diag_payload, provenance=prov)
            paths = write_artifact_contract(
                outdir=Path(args.outdir),
                payload=payload,
                claims=std.claims,
                warnings=std.warnings,
                plot_paths=std.plot_paths,
                write_html=bool(args.html),
            )

            print(f"saved: {paths.summary_txt}")
            print(f"saved: {paths.results_json}")
            print(f"saved: {paths.claims_json}")
            if paths.claims_md is not None:
                print(f"saved: {paths.claims_md}")
            print(f"saved: {paths.plots_dir}")
            if paths.report_html is not None:
                print(f"saved: {paths.report_html}")
            return 0
        if args.command == "quick":
            covariates = [x.strip() for x in str(args.covariates).split(",") if x.strip()]
            if not covariates:
                raise ConfigError("covariates must contain at least one covariate column.")

            csv_path = Path(args.data)
            if not csv_path.exists():
                raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

            df = pd.read_csv(csv_path)
            fit_overrides = {
                "n_folds": int(args.n_folds),
                "num_epochs": int(args.num_epochs),
                "batch_size": args.batch_size,
                "verbose": False,
                "log_every": 1,
            }
            report = fit_dataframe(
                df,
                treatment=args.treatment,
                outcome=args.outcome,
                covariates=covariates,
                mode=args.mode,
                divergence=args.divergence,
                phi=args.phi,
                propensity_model=args.propensity_model,
                m_model=args.m_model,
                fit_overrides=fit_overrides,
                seed=int(args.seed),
            )
            paths = report.save(
                Path(args.outdir),
                write_plots=not bool(args.no_plots),
                write_html=bool(args.html),
            )
            print(f"saved: {paths.summary_txt}")
            print(f"saved: {paths.results_json}")
            print(f"saved: {paths.claims_json}")
            if paths.claims_md is not None:
                print(f"saved: {paths.claims_md}")
            print(f"saved: {paths.plots_dir}")
            if paths.report_html is not None:
                print(f"saved: {paths.report_html}")
            return 0
        if args.command == "demo":
            runs, summary_path = run_live_demo(
                repo_root=ROOT,
                outdir=Path(args.outdir),
                scenario=args.scenario,
                ihdp_data=(args.ihdp_data or None),
                mode=args.mode,
                divergence=args.divergence,
                phi=args.phi,
                propensity_model=args.propensity_model,
                m_model=args.m_model,
                seed=int(args.seed),
                n_folds=int(args.n_folds),
                num_epochs=int(args.num_epochs),
                batch_size=args.batch_size,
                no_plots=bool(args.no_plots),
                html=bool(args.html),
            )
            for run in runs:
                print(f"demo: {run.name}")
                print(f"saved: {run.artifacts.summary_txt}")
                print(f"saved: {run.artifacts.results_json}")
                print(f"saved: {run.artifacts.claims_json}")
                if run.artifacts.claims_md is not None:
                    print(f"saved: {run.artifacts.claims_md}")
                print(f"saved: {run.artifacts.plots_dir}")
                if run.artifacts.report_html is not None:
                    print(f"saved: {run.artifacts.report_html}")
            print(f"saved: {summary_path}")
            return 0
    except (ConfigError, ValueError, FileNotFoundError, KeyError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0
