from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .artifacts import ArtifactPaths, Provenance, build_result_payload, write_artifact_contract
from .plotting import render_plots


@dataclass
class BoundsReport:
    bounds_df: pd.DataFrame
    claims: Dict[str, Any]
    diagnostics: Dict[str, Any]
    provenance: Provenance
    warnings: list[str]

    def summary(self) -> str:
        n_rows = int(self.bounds_df.shape[0])
        valid = self.bounds_df["valid_interval"].to_numpy(dtype=bool)
        n_valid = int(np.count_nonzero(valid))

        lower = self.bounds_df["lower"].to_numpy(dtype=np.float64)
        upper = self.bounds_df["upper"].to_numpy(dtype=np.float64)
        width = self.bounds_df["width"].to_numpy(dtype=np.float64)

        valid_lower = lower[np.isfinite(lower)]
        valid_upper = upper[np.isfinite(upper)]
        valid_width = width[np.isfinite(width)]

        def _m(arr: np.ndarray) -> Optional[float]:
            if arr.size == 0:
                return None
            return float(np.mean(arr))

        lines = [
            "BoundsReport",
            f"rows={n_rows}",
            f"valid_intervals={n_valid}",
            f"lower_mean={_m(valid_lower)}",
            f"upper_mean={_m(valid_upper)}",
            f"width_mean={_m(valid_width)}",
        ]
        return "\n".join(lines)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = build_result_payload(
            bounds_df=self.bounds_df,
            diagnostics=self.diagnostics,
            provenance=self.provenance,
        )
        return asdict(payload)

    def save(
        self,
        outdir: str | Path,
        *,
        write_plots: bool = True,
        write_html: bool = False,
        ground_truth_effect: Optional[float] = None,
        ground_truth_values: Optional[np.ndarray] = None,
        enforce_truth_coverage_for_plot: bool = False,
    ) -> ArtifactPaths:
        plot_paths: list[Path] = []
        warnings = list(self.warnings)

        tmpdir = Path(tempfile.mkdtemp(prefix="itbound-report-plots-"))
        try:
            if write_plots:
                plot_paths.extend(
                    render_plots(
                        self.bounds_df,
                        tmpdir,
                        ground_truth_effect=ground_truth_effect,
                        ground_truth_values=ground_truth_values,
                        enforce_truth_coverage_for_plot=enforce_truth_coverage_for_plot,
                    )
                )
                if not plot_paths:
                    warnings.append(
                        "Plot generation skipped: matplotlib unavailable. "
                        "Install extras: pip install itbound[experiments]"
                    )

            payload = build_result_payload(
                bounds_df=self.bounds_df,
                diagnostics=self.diagnostics,
                provenance=self.provenance,
            )
            return write_artifact_contract(
                outdir=Path(outdir),
                payload=payload,
                claims=self.claims,
                warnings=warnings,
                plot_paths=plot_paths,
                write_html=write_html,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


__all__ = ["BoundsReport"]
