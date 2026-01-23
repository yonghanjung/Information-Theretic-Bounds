#!/usr/bin/env python3
"""
Reproduce plots using the JSON summaries in experiments/final-arxiv.

This script replays the plotting scripts without rerunning simulations by
calling load_plot_n.py, load_plot_ribbon.py, and load_plot_idhp_ribbon.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "src" / "experiments"

_PLOT_N_RE = re.compile(
    r"^(?P<base>.+?)(?:_(?P<stat>stat_(?:mean|median)_over_(?:mean|median)))?_(?P<stamp>\d{8}_\d{6})$"
)
_PLOT_IDHP_RE = re.compile(r"^(?P<base>.+)_(?P<kind>raw|smoothed)(?:_(?P<stamp>\d{8}_\d{6}))?$")


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _normalize_only(values: Optional[str]) -> List[str]:
    if not values:
        return []
    return [item.strip() for item in values.split(",") if item.strip()]


def _resolve_summary_paths(final_dir: Path, only: Sequence[str]) -> List[Path]:
    if not only:
        return sorted(p for p in final_dir.glob("*.json") if p.name != "repro_map.json")

    repro_map: Dict[str, str] = {}
    repro_path = final_dir / "repro_map.json"
    if repro_path.exists():
        repro_map = _read_json(repro_path)  # type: ignore[assignment]

    paths: List[Path] = []
    for item in only:
        if item in repro_map:
            paths.append(final_dir / str(repro_map[item]))
            continue
        candidate = Path(item)
        if not candidate.suffix:
            candidate = candidate.with_suffix(".json")
        if not candidate.is_absolute():
            if (final_dir / candidate).exists():
                candidate = final_dir / candidate
            else:
                candidate = ROOT / candidate
        paths.append(candidate)
    return paths


def _collect_plot_files(summary: Dict[str, object]) -> List[str]:
    files = summary.get("files", {})
    if not isinstance(files, dict):
        return []
    out: List[str] = []
    for value in files.values():
        if isinstance(value, list):
            out.extend([str(v) for v in value if isinstance(v, str)])
        elif isinstance(value, str):
            out.append(value)
    return sorted({p for p in out if p.lower().endswith(".png")})


def _plot_kind(plot_name: str) -> str:
    base = os.path.basename(plot_name)
    if base.startswith("plot_n_"):
        return "n"
    if base.startswith("plot_x0_ribbon"):
        return "ribbon"
    if base.startswith("plot_idhp_ribbon"):
        return "idhp"
    return ""


def _expected_loaded_plot_n_name(plot_name: str) -> str:
    name = os.path.basename(plot_name)
    stem, ext = os.path.splitext(name)
    ext = ext.lstrip(".") or "png"
    match = _PLOT_N_RE.match(stem)
    if not match:
        raise ValueError(f"Could not parse plot_n name: {plot_name}")
    stat = match.group("stat") or ""
    stat_suffix = f"{stat}_" if stat else ""
    return f"loaded_{match.group('base')}_{stat_suffix}{match.group('stamp')}.{ext}"


def _expected_loaded_ribbon_name(plot_name: str) -> str:
    name = os.path.basename(plot_name)
    stem, _ = os.path.splitext(name)
    return f"load_{stem}.png"


def _expected_loaded_idhp_name(plot_name: str) -> str:
    name = os.path.basename(plot_name)
    stem, ext = os.path.splitext(name)
    ext = ext.lstrip(".") or "png"
    match = _PLOT_IDHP_RE.match(stem)
    if not match:
        raise ValueError(f"Could not parse IDHP plot name: {plot_name}")
    stamp = match.group("stamp") or ""
    stamp_suffix = f"_{stamp}" if stamp else ""
    return f"loaded_{match.group('base')}_{match.group('kind')}{stamp_suffix}.{ext}"


def _run(cmd: List[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _rename(src: Path, dst: Path, dry_run: bool) -> None:
    if src == dst:
        return
    print(f"mv {src} {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def _dest_path(plot_rel: str, outdir: Optional[str]) -> Path:
    rel_path = Path(plot_rel)
    if outdir:
        return Path(outdir) / rel_path.name
    return ROOT / rel_path


def _run_plot_n(
    *,
    plot_name: str,
    artifact_path: Path,
    dest_path: Path,
    dry_run: bool,
) -> None:
    outdir = dest_path.parent
    cmd = [
        sys.executable,
        str(EXPERIMENTS_DIR / "load_plot_n.py"),
        "--plot_name",
        plot_name,
        "--artifact",
        str(artifact_path),
        "--outdir",
        str(outdir),
    ]
    _run(cmd, dry_run)
    loaded_path = outdir / _expected_loaded_plot_n_name(plot_name)
    if not dry_run and not loaded_path.exists():
        raise FileNotFoundError(f"Expected output missing: {loaded_path}")
    _rename(loaded_path, dest_path, dry_run)


def _run_plot_ribbon(
    *,
    plot_name: str,
    summary_path: Path,
    artifact_path: Optional[Path],
    dest_path: Path,
    dry_run: bool,
) -> None:
    outdir = dest_path.parent
    cmd = [
        sys.executable,
        str(EXPERIMENTS_DIR / "load_plot_ribbon.py"),
        "--plot_name",
        plot_name,
        "--summary",
        str(summary_path),
        "--outdir",
        str(outdir),
    ]
    if artifact_path is not None:
        cmd += ["--artifact", str(artifact_path), "--artifact_dir", str(artifact_path.parent)]
    _run(cmd, dry_run)
    loaded_path = outdir / _expected_loaded_ribbon_name(plot_name)
    if not dry_run and not loaded_path.exists():
        raise FileNotFoundError(f"Expected output missing: {loaded_path}")
    _rename(loaded_path, dest_path, dry_run)


def _run_plot_idhp(
    *,
    plot_name: str,
    artifact_path: Path,
    dest_path: Path,
    dry_run: bool,
) -> None:
    outdir = dest_path.parent
    cmd = [
        sys.executable,
        str(EXPERIMENTS_DIR / "load_plot_idhp_ribbon.py"),
        "--plot_name",
        plot_name,
        "--artifact",
        str(artifact_path),
        "--outdir",
        str(outdir),
    ]
    _run(cmd, dry_run)
    loaded_path = outdir / _expected_loaded_idhp_name(plot_name)
    if not dry_run and not loaded_path.exists():
        raise FileNotFoundError(f"Expected output missing: {loaded_path}")
    _rename(loaded_path, dest_path, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce final-arxiv plots from JSON summaries.")
    parser.add_argument(
        "--final_arxiv_dir",
        type=str,
        default="experiments/final-arxiv",
        help="Directory containing final-arxiv JSON summaries.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory (default: use paths from summary JSON).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated repro_map key(s) or summary JSON filename(s) to run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    args = parser.parse_args()

    final_dir = ROOT / args.final_arxiv_dir
    summaries = _resolve_summary_paths(final_dir, _normalize_only(args.only))
    if not summaries:
        raise FileNotFoundError(f"No summary JSONs found under: {final_dir}")

    seen: Set[str] = set()
    for summary_path in summaries:
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary JSON not found: {summary_path}")
        summary = _read_json(summary_path)
        files = summary.get("files", {})
        artifact_rel = files.get("artifacts_pkl") if isinstance(files, dict) else None
        artifact_path = ROOT / artifact_rel if isinstance(artifact_rel, str) else None

        plot_files = _collect_plot_files(summary)
        for plot_rel in plot_files:
            if plot_rel in seen:
                continue
            seen.add(plot_rel)
            plot_name = os.path.basename(plot_rel)
            dest_path = _dest_path(plot_rel, args.outdir)
            kind = _plot_kind(plot_name)
            if kind == "n":
                if artifact_path is None:
                    raise FileNotFoundError(f"Missing artifacts_pkl in {summary_path}")
                _run_plot_n(
                    plot_name=plot_name,
                    artifact_path=artifact_path,
                    dest_path=dest_path,
                    dry_run=args.dry_run,
                )
            elif kind == "ribbon":
                _run_plot_ribbon(
                    plot_name=plot_name,
                    summary_path=summary_path,
                    artifact_path=artifact_path,
                    dest_path=dest_path,
                    dry_run=args.dry_run,
                )
            elif kind == "idhp":
                if artifact_path is None:
                    raise FileNotFoundError(f"Missing artifacts_pkl in {summary_path}")
                _run_plot_idhp(
                    plot_name=plot_name,
                    artifact_path=artifact_path,
                    dest_path=dest_path,
                    dry_run=args.dry_run,
                )
            else:
                print(f"Skipping unrecognized plot: {plot_name}")


if __name__ == "__main__":
    main()
