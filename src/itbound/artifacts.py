from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

SCHEMA_VERSION = "results_schema_v0"


@dataclass(frozen=True)
class Provenance:
    package_version: str
    git_commit: Optional[str]
    git_commit_reason: Optional[str]
    timestamp: str
    random_seed: int
    assumptions: Any
    data_hash: Optional[str]
    data_hash_reason: Optional[str]
    command_line: Optional[str]


@dataclass(frozen=True)
class ResultPayload:
    schema_version: str
    provenance: Dict[str, Any]
    bounds: Dict[str, Any]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class ArtifactPaths:
    summary_txt: Path
    results_json: Path
    claims_json: Path
    plots_dir: Path
    report_html: Optional[Path]
    claims_md: Optional[Path] = None


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


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_to_jsonable(payload), indent=2, allow_nan=False)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, path)


def _sha256_bytes(content: bytes) -> str:
    h = hashlib.sha256()
    h.update(content)
    return f"sha256:{h.hexdigest()}"


def compute_data_hash(path_or_df: Any) -> tuple[Optional[str], Optional[str]]:
    if isinstance(path_or_df, (str, Path)):
        path = Path(path_or_df)
        if not path.exists():
            return None, f"input path does not exist: {path}"
        if not path.is_file():
            return None, f"input path is not a file: {path}"
        try:
            return _sha256_bytes(path.read_bytes()), None
        except Exception as exc:
            return None, f"failed to hash input file: {exc}"

    if isinstance(path_or_df, pd.DataFrame):
        try:
            data_bytes = path_or_df.to_csv(index=False).encode("utf-8")
            return _sha256_bytes(data_bytes), None
        except Exception as exc:
            return None, f"failed to hash dataframe: {exc}"

    return None, f"unsupported data source for hashing: {type(path_or_df).__name__}"


def _best_effort_git_commit(repo_root: Optional[Path] = None) -> tuple[Optional[str], Optional[str]]:
    cwd = str(repo_root) if repo_root is not None else None
    try:
        raw = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, cwd=cwd)
        commit = raw.decode("utf-8").strip()
        if not commit:
            return None, "git rev-parse returned empty output"
        return commit, None
    except Exception as exc:
        return None, f"git commit unavailable: {exc}"


def _best_effort_command_line() -> Optional[str]:
    if not sys.argv:
        return None
    try:
        return " ".join(shlex.quote(str(arg)) for arg in sys.argv)
    except Exception:
        return None


def _bounds_aggregates(bounds_df: pd.DataFrame) -> Dict[str, Any]:
    valid = bounds_df["valid_interval"].to_numpy(dtype=bool)
    lower = bounds_df["lower"].to_numpy(dtype=np.float64)
    upper = bounds_df["upper"].to_numpy(dtype=np.float64)
    width = bounds_df["width"].to_numpy(dtype=np.float64)

    def _stats(v: np.ndarray) -> Dict[str, Any]:
        if v.size == 0:
            return {"min": None, "max": None, "mean": None, "median": None}
        return {
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
        }

    lo_valid = lower[valid & np.isfinite(lower)]
    up_valid = upper[valid & np.isfinite(upper)]
    wi_valid = width[valid & np.isfinite(width)]

    return {
        "n_rows": int(bounds_df.shape[0]),
        "n_valid_intervals": int(np.count_nonzero(valid)),
        "lower": _stats(lo_valid),
        "upper": _stats(up_valid),
        "width": _stats(wi_valid),
    }


def build_provenance(
    *,
    package_version: str,
    random_seed: int,
    assumptions: Any,
    data_source: Any,
    repo_root: Optional[Path] = None,
) -> Provenance:
    git_commit, git_reason = _best_effort_git_commit(repo_root=repo_root)
    data_hash, data_hash_reason = compute_data_hash(data_source)

    return Provenance(
        package_version=str(package_version),
        git_commit=git_commit,
        git_commit_reason=git_reason,
        timestamp=datetime.now(timezone.utc).isoformat(),
        random_seed=int(random_seed),
        assumptions=assumptions,
        data_hash=data_hash,
        data_hash_reason=data_hash_reason,
        command_line=_best_effort_command_line(),
    )


def build_result_payload(
    *,
    bounds_df: pd.DataFrame,
    diagnostics: Dict[str, Any],
    provenance: Provenance,
) -> ResultPayload:
    return ResultPayload(
        schema_version=SCHEMA_VERSION,
        provenance=asdict(provenance),
        bounds=_bounds_aggregates(bounds_df),
        diagnostics=dict(diagnostics),
    )


def _summary_text(payload: ResultPayload, claims: Dict[str, Any], warnings: list[str]) -> str:
    b = payload.bounds
    lines = [
        "itbound artifact contract summary",
        f"schema_version: {payload.schema_version}",
        f"n_rows: {b.get('n_rows')}",
        f"n_valid_intervals: {b.get('n_valid_intervals')}",
        f"mean_width: {((b.get('width') or {}).get('mean'))}",
        f"claims_keys: {', '.join(sorted(claims.keys()))}",
    ]
    if warnings:
        lines.append("warnings:")
        for w in warnings:
            lines.append(f"- {w}")
    return "\n".join(lines) + "\n"


def _render_html(payload: ResultPayload, claims: Dict[str, Any], rel_plot_files: list[str]) -> str:
    claim_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in sorted(claims.items()))
    plot_html = "\n".join(
        f'<img src="plots/{name}" alt="{name}" style="max-width:100%;height:auto;" />' for name in rel_plot_files
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>itbound report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 960px; }}
    th, td {{ border: 1px solid #ddd; text-align: left; padding: 8px; }}
    th {{ background: #f5f5f5; width: 280px; }}
    pre {{ background: #f7f7f7; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>itbound Artifact Report</h1>
  <p><code>schema_version={payload.schema_version}</code></p>

  <h2>Claims</h2>
  <table>{claim_rows}</table>

  <h2>Payload (excerpt)</h2>
  <pre>{json.dumps(_to_jsonable(asdict(payload)), indent=2)[:8000]}</pre>

  <h2>Plots</h2>
  {plot_html}
</body>
</html>
"""


def _render_claims_markdown(claims: Dict[str, Any]) -> str:
    lines = [
        "# Claims",
        "",
        f"- claims_engine_version: {claims.get('claims_engine_version', 'unknown')}",
        f"- n_rows: {claims.get('n_rows')}",
        f"- n_valid_intervals: {claims.get('n_valid_intervals')}",
        f"- valid_interval_rate: {claims.get('valid_interval_rate')}",
        f"- invalid_interval_rate: {claims.get('invalid_interval_rate')}",
        f"- inverted_count: {claims.get('inverted_count')}",
        "",
    ]

    claim_list = claims.get("claims")
    if isinstance(claim_list, list):
        for item in claim_list:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", item.get("id", "claim")))
            lines.append(f"## {title}")
            lines.append("")
            lines.append(f"- id: {item.get('id')}")
            lines.append(f"- statement: {item.get('statement')}")

            evidence = item.get("evidence", {})
            if isinstance(evidence, dict) and evidence:
                lines.append("- evidence:")
                for key in sorted(evidence.keys()):
                    lines.append(f"  - {key}: {evidence[key]}")

            conditions = item.get("conditions", [])
            if isinstance(conditions, list) and conditions:
                lines.append("- conditions:")
                for cond in conditions:
                    lines.append(f"  - {cond}")

            caveats = item.get("caveats", [])
            if isinstance(caveats, list) and caveats:
                lines.append("- caveats:")
                for caveat in caveats:
                    lines.append(f"  - {caveat}")
            lines.append("")
    else:
        lines.append("No structured claims list found.")
        lines.append("")

    return "\n".join(lines)


def write_artifact_contract(
    *,
    outdir: Path,
    payload: ResultPayload,
    claims: Dict[str, Any],
    warnings: Optional[list[str]] = None,
    plot_paths: Optional[list[Path]] = None,
    write_html: bool = False,
) -> ArtifactPaths:
    warnings = list(warnings or [])
    plot_paths = list(plot_paths or [])

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rel_plot_files: list[str] = []
    for p in plot_paths:
        src = Path(p)
        if not src.exists() or not src.is_file():
            continue
        dst = plots_dir / src.name
        shutil.copy2(src, dst)
        rel_plot_files.append(dst.name)

    results_path = outdir / "results.json"
    claims_path = outdir / "claims.json"
    claims_md_path = outdir / "claims.md"
    summary_path = outdir / "summary.txt"

    write_json_atomic(results_path, asdict(payload))
    write_json_atomic(claims_path, {"schema_version": SCHEMA_VERSION, "claims": claims})
    claims_md_path.write_text(_render_claims_markdown(claims), encoding="utf-8")
    summary_path.write_text(_summary_text(payload, claims=claims, warnings=warnings), encoding="utf-8")

    report_path: Optional[Path] = None
    if write_html:
        report_path = outdir / "report.html"
        report_path.write_text(_render_html(payload, claims=claims, rel_plot_files=rel_plot_files), encoding="utf-8")

    return ArtifactPaths(
        summary_txt=summary_path,
        results_json=results_path,
        claims_json=claims_path,
        plots_dir=plots_dir,
        report_html=report_path,
        claims_md=claims_md_path,
    )


__all__ = [
    "SCHEMA_VERSION",
    "Provenance",
    "ResultPayload",
    "ArtifactPaths",
    "compute_data_hash",
    "build_provenance",
    "build_result_payload",
    "write_json_atomic",
    "write_artifact_contract",
]
