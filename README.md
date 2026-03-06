# itbound: Causal Bounds Without Strong Identification Assumptions

[Paper](https://arxiv.org/abs/2601.17160) | [GitHub](https://github.com/yonghanjung/Information-Theretic-Bounds) | [HF Demo](https://huggingface.co/spaces/yonghanjung/itbound-demo) | [PyPI](https://pypi.org/project/itbound/) | [Docs](docs/quickstart.md) | [Citation](CITATION.cff) | [Quickstart](#one-command-quickstart)

Compute valid lower and upper causal bounds under unmeasured confounding when point identification is not credible.

Point causal effects usually require strong assumptions such as no hidden confounding, valid instruments, or other auxiliary structure. When those assumptions are not credible, `itbound` returns causal intervals instead of fragile point estimates, without forcing bounded outcomes or sensitivity-parameter workflows.

![IHDP ribbon example](https://raw.githubusercontent.com/yonghanjung/Information-Theretic-Bounds/main/docs/latex/figures/ribbon_ihdp.png)

## Why it matters
- Point identification is powerful, but brittle when its assumptions are not believable.
- Causal intervals remain meaningful when strong identification assumptions fail.
- Older bound methods often relied on bounded outcomes, discrete settings, or other restrictive inputs that are awkward for modern ML-style datasets.

### Who is this for?
- Researchers who do not believe no-unmeasured-confounding but still want causal information.
- Users who need causal intervals rather than overconfident point estimates.
- People who want a CLI and Python package, not just a theorem.

## One-command quickstart
```bash
python -m pip install itbound
itbound example --out /tmp/itbound_example.csv
itbound quick --data /tmp/itbound_example.csv --treatment a --outcome y --covariates x1,x2 --outdir /tmp/itbound_quick
```

Outputs:
- `results.json`
- `claims.md`
- `plots/`
- `report.html`
- `bounds.csv` and `summary.json` in standard mode

## What you get
- Lower and upper causal bounds under unmeasured confounding
- CLI plus Python API
- Claims and reporting artifacts for sharing results
- Support for practical observational CSV workflows
- Reproducible figure-generation paths for the paper

## Main figure
IHDP is one of the best-known personalized-effect benchmarks. In that setting, `itbound` tracks the true effect trend with lower and upper bounds even when outcomes are not restricted to `[0, 1]`, which is exactly where many older causal-bound workflows become uncomfortable or unusable.

## Reproduce paper
Paper: [Data-Driven Information-Theoretic Causal Bounds under Unmeasured Confounding](https://arxiv.org/abs/2601.17160)

Quick path:

```bash
itbound example --out /tmp/itbound_example.csv
itbound quick --data /tmp/itbound_example.csv --treatment a --outcome y --covariates x1,x2 --outdir /tmp/itbound_quick
```

Standard mode:

```bash
itbound standard --csv /tmp/itbound_example.csv --y-col y --a-col a --x-cols x1,x2 --outdir /tmp/itbound_standard --divergences KL,JS,Hellinger,TV,Chi2 --aggregation-mode paper_adaptive_k --html
```

Paper-figure dry run:

```bash
itbound reproduce --dry-run
```

More detail:
- [docs/quickstart.md](docs/quickstart.md)
- [docs/artifact_contract.md](docs/artifact_contract.md)
- [docs/aggregation_modes.md](docs/aggregation_modes.md)
- [docs/results_schema_v0.md](docs/results_schema_v0.md)

## Citation
Software citation metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@article{jung2026itbound,
  title   = {Data-Driven Information-Theoretic Causal Bounds under Unmeasured Confounding},
  author  = {Jung, Yonghan and Kang, Bogyeong},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.17160}
}
```

## Links
- Paper: <https://arxiv.org/abs/2601.17160>
- PyPI: <https://pypi.org/project/itbound/>
- Repository: <https://github.com/yonghanjung/Information-Theretic-Bounds>
- Hugging Face demo: <https://huggingface.co/spaces/yonghanjung/itbound-demo>
- Quickstart: [docs/quickstart.md](docs/quickstart.md)
- Artifact contract: [docs/artifact_contract.md](docs/artifact_contract.md)

## Demo assets
Live demo preview:

![itbound demo preview](https://raw.githubusercontent.com/yonghanjung/Information-Theretic-Bounds/main/docs/media/quick-demo-v5.gif)

To regenerate the GIF:

```bash
bash scripts/demo/make_quick_demo.sh
```

## Python API
```python
from itbound.standard import run_standard_bounds

result = run_standard_bounds(
    csv_path="/tmp/itbound_toy.csv",
    outcome_col="y",
    treatment_col="a",
    covariate_cols=["x1", "x2"],
    divergences=["KL", "JS", "Hellinger", "TV", "Chi2"],
    aggregation_mode="paper_adaptive_k",
    outdir="/tmp/itbound_standard",
    write_html=True,
)
```

Legacy import remains supported:

```python
import fbound
```

## CLI reference
Run from config:

```bash
itbound run --config docs/cli-config.example.yaml
```

Override output path:

```bash
itbound run --config docs/cli-config.example.yaml --out /tmp/itbound_bounds.csv
```

Artifact contract mode:

```bash
itbound artifacts --csv /tmp/itbound_toy.csv --y-col y --a-col a --x-cols x1,x2 --outdir /tmp/itbound_artifacts --divergences KL
```

## Notes on outputs
- `results.json` is the schema-versioned artifact for quick and artifact-contract flows.
- `claims.md` is the readable report for sharing robust claims.
- `bounds.csv` and `summary.json` are produced in standard mode.
- `report.html` is optional and written when HTML output is enabled.

## Development
```bash
python -m pip install -e .[dev]
python -m pytest -q
python -m build
```

## Troubleshooting
1. If plots do not render, install extras with `python -m pip install 'itbound[experiments]'`.
2. If you only need a fast install check, run `itbound --help`.
3. If you want the demo figures, run from the repo root so the bundled data paths resolve correctly.
