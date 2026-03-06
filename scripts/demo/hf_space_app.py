"""Draft Hugging Face Space app for itbound.

Copy this file to the root of a Gradio Space as `app.py`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from itbound.standard import DEFAULT_DIVERGENCES, run_standard_bounds


def _claims_markdown(claims: dict) -> str:
    lines = [
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


def _zip_artifacts(outdir: Path) -> str:
    archive_base = outdir.parent / f"{outdir.name}_artifacts"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=outdir)
    return archive_path


def run_space_demo(
    csv_path: str,
    treatment_col: str,
    outcome_col: str,
    covariates_text: str,
    divergences: list[str],
    aggregation_mode: str,
    write_html: bool,
):
    if not csv_path:
        raise gr.Error("Upload a CSV file first.")

    frame = pd.read_csv(csv_path)
    covariates = [item.strip() for item in covariates_text.split(",") if item.strip()]
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

    claims_md = _claims_markdown(result.claims)
    bounds_preview = result.bounds.head(50)
    width_fig = _width_plot(result.bounds)
    archive_path = _zip_artifacts(outdir)
    return bounds_preview, width_fig, claims_md, archive_path


with gr.Blocks(title="itbound Demo") as demo:
    gr.Markdown(
        """
        # itbound Demo

        Upload a CSV and compute data-driven lower and upper causal bounds under unmeasured confounding.
        This demo is designed for cases where point identification is not credible but causal intervals remain meaningful.
        """
    )

    csv_file = gr.File(label="CSV upload", type="filepath")
    with gr.Row():
        treatment = gr.Textbox(label="Treatment column", value="a")
        outcome = gr.Textbox(label="Outcome column", value="y")
    covariates = gr.Textbox(label="Covariate columns (comma-separated)", value="x1,x2")
    with gr.Row():
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
    claims = gr.Markdown()
    bundle = gr.File(label="Download artifact bundle")

    run_btn.click(
        fn=run_space_demo,
        inputs=[csv_file, treatment, outcome, covariates, divergence, aggregation, html],
        outputs=[bounds, plot, claims, bundle],
    )


if __name__ == "__main__":
    demo.launch()
