"""Draft Hugging Face Space app for itbound.

Copy this file to the root of a Gradio Space as `app.py`.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import gradio as gr
import matplotlib.pyplot as plt
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
        return frame, covariates, f"uploaded CSV `{Path(csv_path).name}`"

    ihdp_path = _resolve_space_ihdp_path()
    frame = _load_ihdp_csv(ihdp_path)
    if not covariates:
        covariates = list(CANONICAL_IHDP_COVARIATES)
    return frame, covariates, f"canonical IHDP example `{ihdp_path.name}`"


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
    archive_path = _zip_artifacts(outdir)
    return bounds_preview, width_fig, claims_md, archive_path


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
