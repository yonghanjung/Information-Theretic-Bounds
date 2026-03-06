"""Draft Hugging Face Space app for itbound.

Copy this file to the root of a Gradio Space as `app.py`.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itbound.live_demo import _load_ihdp_csv
from itbound.standard import DEFAULT_DIVERGENCES, run_standard_bounds


CANONICAL_IHDP_COVARIATES = [f"x{i}" for i in range(1, 6)]
CANONICAL_IHDP_URL = (
    "https://raw.githubusercontent.com/yonghanjung/Information-Theretic-Bounds/main/"
    "ihdp_data/ihdp_npci_1.csv"
)


def _detect_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / "ihdp_data" / "ihdp_npci_1.csv").exists():
            return candidate
    return here.parent


REPO_ROOT = _detect_repo_root()


def _claims_markdown(
    claims: dict,
    *,
    source_label: str,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
) -> str:
    lines = [
        "## Run summary",
        f"- Source: {source_label}",
        f"- Treatment / outcome: `{treatment_col}` / `{outcome_col}`",
        f"- Covariates: `{', '.join(covariates)}`",
        "",
        "## Claims summary",
        f"- Robust sign: `{claims['sign']['label']}`",
        f"- Sign statement: {claims['sign']['statement']}",
    ]
    range_payload = claims.get("range", {})
    if range_payload.get("lower_min") is not None and range_payload.get("upper_max") is not None:
        lines.append(
            f"- Overall valid range: `{range_payload['lower_min']:.6g}` to `{range_payload['upper_max']:.6g}`"
        )
    lines.append(f"- Valid interval rate: `{claims['valid_interval_rate']:.3f}`")
    return "\n".join(lines)


def _width_plot(bounds_df: pd.DataFrame):
    width = pd.to_numeric(bounds_df["width"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(width, bins=30, color="#1d4ed8", edgecolor="white")
    ax.set_title("Causal interval width distribution")
    ax.set_xlabel("Width")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _with_x0_alias(frame: pd.DataFrame) -> pd.DataFrame:
    if "x0" in frame.columns or "x1" not in frame.columns:
        return frame
    enriched = frame.copy()
    enriched["x0"] = pd.to_numeric(enriched["x1"], errors="coerce")
    return enriched


def _truth_vector(frame: pd.DataFrame, bounds_df: pd.DataFrame) -> pd.Series | None:
    if {"mu0", "mu1"}.issubset(frame.columns):
        truth = pd.to_numeric(frame["mu1"], errors="coerce") - pd.to_numeric(frame["mu0"], errors="coerce")
    elif "tau_true" in frame.columns:
        truth = pd.to_numeric(frame["tau_true"], errors="coerce")
    else:
        return None

    if "i" not in bounds_df.columns:
        return truth.iloc[: bounds_df.shape[0]].reset_index(drop=True)
    row_ids = pd.to_numeric(bounds_df["i"], errors="coerce").fillna(-1).astype(int)
    valid_rows = row_ids.between(0, frame.shape[0] - 1)
    aligned = pd.Series(index=bounds_df.index, dtype=float)
    aligned.loc[valid_rows] = truth.iloc[row_ids.loc[valid_rows].to_numpy()].to_numpy(dtype=float)
    return aligned


def _ribbon_plot(frame: pd.DataFrame, bounds_df: pd.DataFrame, *, axis_col: str):
    axis_name = axis_col.strip() or "x0"
    if axis_name not in frame.columns:
        return None

    axis_series = pd.to_numeric(frame[axis_name], errors="coerce")
    if "i" in bounds_df.columns:
        row_ids = pd.to_numeric(bounds_df["i"], errors="coerce").fillna(-1).astype(int)
        valid_rows = row_ids.between(0, frame.shape[0] - 1)
        axis_values = pd.Series(index=bounds_df.index, dtype=float)
        axis_values.loc[valid_rows] = axis_series.iloc[row_ids.loc[valid_rows].to_numpy()].to_numpy(dtype=float)
    else:
        axis_values = axis_series.iloc[: bounds_df.shape[0]].reset_index(drop=True)

    ribbon_df = pd.DataFrame(
        {
            axis_name: axis_values.to_numpy(dtype=float),
            "lower": pd.to_numeric(bounds_df["lower"], errors="coerce").to_numpy(dtype=float),
            "upper": pd.to_numeric(bounds_df["upper"], errors="coerce").to_numpy(dtype=float),
        }
    )
    truth = _truth_vector(frame, bounds_df)
    if truth is not None:
        ribbon_df["truth"] = truth.to_numpy(dtype=float)

    ribbon_df = ribbon_df.dropna(subset=[axis_name, "lower", "upper"]).sort_values(axis_name, kind="mergesort")
    if ribbon_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    x = ribbon_df[axis_name].to_numpy(dtype=float)
    lower = ribbon_df["lower"].to_numpy(dtype=float)
    upper = ribbon_df["upper"].to_numpy(dtype=float)
    ax.plot(x, lower, color="tab:blue", linewidth=1.2, label="lower")
    ax.plot(x, upper, color="tab:orange", linewidth=1.2, label="upper")
    valid_band = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    if np.any(valid_band):
        ax.fill_between(x, lower, upper, where=valid_band, alpha=0.2, color="tab:green", label="bound ribbon")
    if "truth" in ribbon_df.columns:
        truth_vals = ribbon_df["truth"].to_numpy(dtype=float)
        truth_mask = np.isfinite(truth_vals)
        if np.any(truth_mask):
            ax.plot(
                x[truth_mask],
                truth_vals[truth_mask],
                color="tab:red",
                linewidth=1.4,
                linestyle="--",
                label="ground truth",
            )
    ax.set_title(f"Ribbon plot by {axis_name}")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Causal effect / bounds")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _save_plot(fig, outdir: Path, filename: str) -> Path | None:
    if fig is None:
        return None
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / filename
    fig.savefig(path, dpi=120)
    return path


def _zip_artifacts(outdir: Path) -> str:
    archive_base = outdir.parent / f"{outdir.name}_artifacts"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=outdir)
    return archive_path


def _parse_covariates(covariates_text: str) -> list[str]:
    return [item.strip() for item in covariates_text.split(",") if item.strip()]


def _load_input_frame(
    *,
    csv_path: str | None,
    outcome_col: str,
    treatment_col: str,
    covariates_text: str,
) -> tuple[pd.DataFrame, list[str], str]:
    covariates = _parse_covariates(covariates_text)
    if csv_path:
        frame = pd.read_csv(csv_path)
        if not covariates:
            excluded = {outcome_col, treatment_col, "y_cfactual", "mu0", "mu1"}
            covariates = [col for col in frame.columns if col not in excluded]
        return _with_x0_alias(frame), covariates, f"uploaded CSV `{Path(csv_path).name}`"

    ihdp_path = _resolve_space_ihdp_path()
    frame = _load_ihdp_csv(ihdp_path)
    if not covariates:
        covariates = list(CANONICAL_IHDP_COVARIATES)
    return _with_x0_alias(frame), covariates, f"canonical IHDP example `{ihdp_path.name}`"


def _resolve_space_ihdp_path() -> Path:
    local_candidate = REPO_ROOT / "ihdp_data" / "ihdp_npci_1.csv"
    if local_candidate.exists():
        return local_candidate

    cache_path = Path(tempfile.gettempdir()) / "itbound-space-ihdp_npci_1.csv"
    if not cache_path.exists():
        urlretrieve(CANONICAL_IHDP_URL, cache_path)
    return cache_path


def run_space_demo(
    csv_path: str | None,
    treatment_col: str,
    outcome_col: str,
    covariates_text: str,
    ribbon_axis_col: str,
    divergences: list[str],
    aggregation_mode: str,
    write_html: bool,
):
    frame, covariates, source_label = _load_input_frame(
        csv_path=csv_path,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        covariates_text=covariates_text,
    )
    if not covariates:
        raise gr.Error("Provide at least one covariate column.")
    if not divergences:
        raise gr.Error("Select at least one divergence.")

    tmpdir = Path(tempfile.mkdtemp(prefix="itbound-space-"))
    outdir = tmpdir / "itbound-demo"
    result = run_standard_bounds(
        dataframe=frame,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        covariate_cols=covariates,
        divergences=divergences,
        aggregation_mode=aggregation_mode,
        outdir=outdir,
        write_plots=True,
        write_html=write_html,
    )

    claims_md = _claims_markdown(
        result.claims,
        source_label=source_label,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariates=covariates,
    )
    bounds_preview = result.bounds.head(50)
    width_fig = _width_plot(result.bounds)
    ribbon_axis = ribbon_axis_col.strip() or "x0"
    ribbon_fig = _ribbon_plot(frame, result.bounds, axis_col=ribbon_axis)
    _save_plot(width_fig, outdir, "bounds_width_histogram.png")
    _save_plot(ribbon_fig, outdir, f"bounds_ribbon_{re.sub(r'[^a-zA-Z0-9_-]+', '_', ribbon_axis)}.png")
    archive_path = _zip_artifacts(outdir)
    return bounds_preview, width_fig, ribbon_fig, claims_md, archive_path


with gr.Blocks(title="itbound Demo") as demo:
    gr.Markdown(
        """
        # itbound Demo

        [Paper](https://arxiv.org/abs/2601.17160) | [GitHub](https://github.com/yonghanjung/Information-Theretic-Bounds) | [PyPI](https://pypi.org/project/itbound/)

        Upload a CSV to compute data-driven lower and upper causal bounds under unmeasured confounding,
        or leave the upload blank to run the canonical IHDP example.
        For a toy uploaded CSV, a common schema is `treatment=a`, `outcome=y`, and `covariates=x1,x2`.
        """
    )

    csv_file = gr.File(label="CSV upload (optional; blank = canonical IHDP example)", type="filepath")
    with gr.Row():
        treatment = gr.Textbox(label="Treatment column", value="treatment")
        outcome = gr.Textbox(label="Outcome column", value="y_factual")
    covariates = gr.Textbox(
        label="Covariate columns (comma-separated)",
        value="x1,x2,x3,x4,x5",
    )
    with gr.Row():
        ribbon_axis = gr.Textbox(label="Ribbon x-axis column", value="x0")
        divergence = gr.CheckboxGroup(
            label="Divergences",
            choices=list(DEFAULT_DIVERGENCES),
            value=["KL", "TV"],
        )
        aggregation = gr.Radio(
            label="Aggregation mode",
            choices=["paper_adaptive_k", "fixed_k_endpoint", "tight_kth"],
            value="paper_adaptive_k",
        )
        html = gr.Checkbox(label="Write HTML report", value=False)

    run_btn = gr.Button("Compute bounds", variant="primary")

    bounds = gr.Dataframe(label="Bounds preview", interactive=False)
    plot = gr.Plot(label="Interval width histogram")
    ribbon = gr.Plot(label="Ribbon plot by x0")
    claims = gr.Markdown()
    bundle = gr.File(label="Download artifact bundle")

    run_btn.click(
        fn=run_space_demo,
        inputs=[csv_file, treatment, outcome, covariates, ribbon_axis, divergence, aggregation, html],
        outputs=[bounds, plot, ribbon, claims, bundle],
    )


if __name__ == "__main__":
    demo.launch()
