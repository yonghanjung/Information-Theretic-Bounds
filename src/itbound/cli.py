from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fbound.estimators.causal_bound import compute_causal_bounds

from .config import ConfigError, build_phi, default_example_config, load_config, resolve_data

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
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0
