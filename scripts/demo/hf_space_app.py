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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from itbound.live_demo import _load_ihdp_csv
from itbound.standard import DEFAULT_DIVERGENCES, run_standard_bounds


CANONICAL_IHDP_COVARIATES = [f"x{i}" for i in range(1, 6)]
CANONICAL_IHDP_URL = (
    "https://raw.githubusercontent.com/yonghanjung/Information-Theretic-Bounds/main/"
    "ihdp_data/ihdp_npci_1.csv"
)
PROPENSITY_AXIS_NAMES = {"propensity", "estimated_propensity", "e_hat", "ehat"}
RIBBON_Q_BINS = 30


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


def _align_series_to_bounds(values: pd.Series, bounds_df: pd.DataFrame) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if "i" not in bounds_df.columns:
        return numeric.iloc[: bounds_df.shape[0]].reset_index(drop=True)
    row_ids = pd.to_numeric(bounds_df["i"], errors="coerce").fillna(-1).astype(int)
    valid_rows = row_ids.between(0, numeric.shape[0] - 1)
    aligned = pd.Series(index=bounds_df.index, dtype=float)
    aligned.loc[valid_rows] = numeric.iloc[row_ids.loc[valid_rows].to_numpy()].to_numpy(dtype=float)
    return aligned


def _truth_vector(frame: pd.DataFrame, bounds_df: pd.DataFrame) -> pd.Series | None:
    if {"mu0", "mu1"}.issubset(frame.columns):
        truth = pd.to_numeric(frame["mu1"], errors="coerce") - pd.to_numeric(frame["mu0"], errors="coerce")
    elif "tau_true" in frame.columns:
        truth = pd.to_numeric(frame["tau_true"], errors="coerce")
    else:
        return None
    return _align_series_to_bounds(truth, bounds_df)

def _estimate_propensity(frame: pd.DataFrame, *, treatment_col: str, covariates: list[str]) -> pd.Series | None:
    usable_covariates = [col for col in covariates if col in frame.columns]
    if treatment_col not in frame.columns or not usable_covariates:
        return None

    treatment = pd.to_numeric(frame[treatment_col], errors="coerce")
    if treatment.isna().any():
        return None
    a = treatment.astype(int)
    if set(a.unique()) - {0, 1}:
        return None

    x_frame = frame[usable_covariates].apply(pd.to_numeric, errors="coerce")
    x_frame = x_frame.loc[:, x_frame.notna().any(axis=0)]
    if x_frame.empty:
        return None
    x_frame = x_frame.fillna(x_frame.median(numeric_only=True))
    if x_frame.isna().any().any():
        return None

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    try:
        model.fit(x_frame.to_numpy(dtype=float), a.to_numpy(dtype=int))
        propensity = model.predict_proba(x_frame.to_numpy(dtype=float))[:, 1]
    except Exception:
        return None
    return pd.Series(propensity, index=frame.index, dtype=float)


def _resolve_ribbon_axis(
    frame: pd.DataFrame,
    bounds_df: pd.DataFrame,
    *,
    axis_col: str,
    treatment_col: str,
    covariates: list[str],
) -> tuple[str, pd.Series] | None:
    axis_name = axis_col.strip() or "propensity"
    axis_key = axis_name.lower()
    if axis_key in PROPENSITY_AXIS_NAMES:
        propensity = _estimate_propensity(frame, treatment_col=treatment_col, covariates=covariates)
        if propensity is None:
            return None
        return "estimated propensity", _align_series_to_bounds(propensity, bounds_df)
    if axis_name not in frame.columns:
        return None
    return axis_name, _align_series_to_bounds(frame[axis_name], bounds_df)


def _bin_ribbon_frame(ribbon_df: pd.DataFrame, *, axis_label: str) -> tuple[pd.DataFrame, bool]:
    unique_count = int(ribbon_df[axis_label].nunique(dropna=True))
    if unique_count < 10 or ribbon_df.shape[0] < 40:
        return ribbon_df, False

    n_bins = min(RIBBON_Q_BINS, unique_count)
    try:
        qbin = pd.qcut(ribbon_df[axis_label], q=n_bins, duplicates="drop")
    except ValueError:
        return ribbon_df, False

    grouped = ribbon_df.assign(_bin=qbin).dropna(subset=["_bin"]).groupby("_bin", observed=True)
    aggregated = pd.DataFrame(
        {
            axis_label: grouped[axis_label].median(),
            "lower": grouped["lower"].median(),
            "upper": grouped["upper"].median(),
        }
    )
    if "truth" in ribbon_df.columns:
        aggregated["truth"] = grouped["truth"].median()
    return aggregated.reset_index(drop=True), True


def _ribbon_plot(
    frame: pd.DataFrame,
    bounds_df: pd.DataFrame,
    *,
    axis_col: str,
    treatment_col: str,
    covariates: list[str],
):
    resolved = _resolve_ribbon_axis(
        frame,
        bounds_df,
        axis_col=axis_col,
        treatment_col=treatment_col,
        covariates=covariates,
    )
    if resolved is None:
        return None

    axis_label, axis_values = resolved
    ribbon_df = pd.DataFrame(
        {
            axis_label: axis_values.to_numpy(dtype=float),
            "lower": pd.to_numeric(bounds_df["lower"], errors="coerce").to_numpy(dtype=float),
            "upper": pd.to_numeric(bounds_df["upper"], errors="coerce").to_numpy(dtype=float),
        }
    )
    truth = _truth_vector(frame, bounds_df)
    if truth is not None:
        ribbon_df["truth"] = truth.to_numpy(dtype=float)

    ribbon_df = ribbon_df.dropna(subset=[axis_label, "lower", "upper"]).sort_values(axis_label, kind="mergesort")
    if ribbon_df.empty:
        return None
    ribbon_df, used_binning = _bin_ribbon_frame(ribbon_df, axis_label=axis_label)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = ribbon_df[axis_label].to_numpy(dtype=float)
    lower = ribbon_df["lower"].to_numpy(dtype=float)
    upper = ribbon_df["upper"].to_numpy(dtype=float)
    lower_label = "median lower" if used_binning else "lower"
    upper_label = "median upper" if used_binning else "upper"
    ribbon_label = "median bound ribbon" if used_binning else "bound ribbon"
    truth_label = "median ground truth" if used_binning else "ground truth"
    ax.plot(x, lower, color="tab:blue", linewidth=1.2, label=lower_label)
    ax.plot(x, upper, color="tab:orange", linewidth=1.2, label=upper_label)
    valid_band = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    if np.any(valid_band):
        ax.fill_between(x, lower, upper, where=valid_band, alpha=0.2, color="tab:green", label=ribbon_label)
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
                label=truth_label,
            )
    title_prefix = "Binned " if used_binning else ""
    ax.set_title(f"{title_prefix}ribbon plot by {axis_label}")
    ax.set_xlabel(axis_label)
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
    ribbon_axis = ribbon_axis_col.strip() or "propensity"
    ribbon_fig = _ribbon_plot(
        frame,
        result.bounds,
        axis_col=ribbon_axis,
        treatment_col=treatment_col,
        covariates=covariates,
    )
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
        ribbon_axis = gr.Textbox(
            label="Ribbon x-axis column",
            value="propensity",
            info="Recommended: `propensity` for the paper-aligned IHDP view. You can also try a raw covariate such as `x4`.",
        )
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
    ribbon = gr.Plot(label="Binned ribbon plot")
    claims = gr.Markdown()
    bundle = gr.File(label="Download artifact bundle")

    run_btn.click(
        fn=run_space_demo,
        inputs=[csv_file, treatment, outcome, covariates, ribbon_axis, divergence, aggregation, html],
        outputs=[bounds, plot, ribbon, claims, bundle],
    )


if __name__ == "__main__":
    demo.launch()
