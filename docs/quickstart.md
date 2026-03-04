# Quickstart (CLI + Python)

This page gives a 10-minute success path for new users.

Important:
- `quick` is an opt-in wrapper around `itbound.fit(...)`.
- Default behavior remains paper-equivalent (`mode=paper-default`).
- The config-based workflow (`itbound run --config ...`) is still the canonical paper workflow.

## 1) CLI quick path

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m itbound example --out /tmp/itbound_example.csv
python -m itbound quick --data /tmp/itbound_example.csv --treatment a --outcome y --covariates x1,x2 --outdir /tmp/itbound_quick
```

Artifacts are written to `--outdir` (`/tmp/itbound_quick` above).
See [`docs/artifact_contract.md`](artifact_contract.md) for the file contract.

## 2) Live demo (toy + IHDP benchmark)

IHDP-only (explicit KL + truth-coverage envelope):

```bash
python -m itbound demo --scenario ihdp --divergence KL --enforce-truth-coverage --eval-points 240 --rounds 5 --n-folds 5 --outdir /tmp/itbound_live_demo --batch-size 8
```

IHDP benchmark CSV:

```bash
python -m itbound demo --scenario ihdp --ihdp-data /path/to/ihdp_npci_1.csv --divergence KL --enforce-truth-coverage --eval-points 240 --rounds 5 --n-folds 5 --outdir /tmp/itbound_live_demo --batch-size 8
```

Both scenarios:

```bash
python -m itbound demo --scenario both --toy-n 1000 --divergence KL --enforce-truth-coverage --eval-points 240 --rounds 5 --n-folds 5 --outdir /tmp/itbound_live_demo --batch-size 8
```

Demo output layout:
- `--outdir/toy/` and/or `--outdir/ihdp/` with `summary.txt`, `results.json`, `claims.json`, `claims.md`, `plots/`
- `--outdir/live_demo_summary.md`
- By default, IHDP plotting applies a demo-only truth-aware envelope so each point satisfies `lower < truth < upper` visually.
- Use `--no-enforce-truth-coverage` to disable this visualization aid.
- Use `--eval-points` (for example `240`) to evaluate/render fewer points and reduce plot spikiness in demos.

README includes a toy-demo GIF preview at `docs/media/quick-demo-v5.gif`.
Rebuild it with:

```bash
bash scripts/demo/make_quick_demo.sh
```

## 3) Standard aggregation options (multi-divergence)

```bash
python -m itbound standard \
  --csv /tmp/itbound_example.csv \
  --y-col y \
  --a-col a \
  --x-cols x1,x2 \
  --aggregation-mode tight_kth \
  --outdir /tmp/itbound_standard_tight
```

Aggregation modes:
- `paper_adaptive_k` (default)
- `fixed_k_endpoint` (use `--fixed-k`)
- `tight_kth` (experiment-style tight-kth order-statistics)
  - default divergence set is `KL,JS,Hellinger,TV,Chi2`
  - override with subset via `--divergences` (example: `KL,TV`)

Ground-truth overlays for `standard`/`artifacts` (visualization only):
- `--ground-truth-col <col>`: per-point truth line
- `--ground-truth-effect <float>`: scalar truth line
- auto mode: if `mu1` and `mu0` exist, draw `mu1-mu0` by default
- disable auto mode: `--no-ground-truth-auto`

Subset + explicit truth-column example:

```bash
python -m itbound standard \
  --csv /tmp/itbound_example.csv \
  --y-col y \
  --a-col a \
  --x-cols x1,x2 \
  --divergences KL,TV \
  --aggregation-mode tight_kth \
  --ground-truth-col tau_true \
  --outdir /tmp/itbound_standard_tight_kltv
```

Details: [`docs/aggregation_modes.md`](aggregation_modes.md)

## 4) Python quick path

```python
import pandas as pd
import itbound

df = pd.read_csv("/tmp/itbound_example.csv")

res = itbound.fit(
    df,
    treatment="a",
    outcome="y",
    covariates=["x1", "x2"],
    mode="paper-default",
)

print(res.summary())
res.save("/tmp/itbound_quick_py")
```

## 5) Interpreting outputs safely

- `lower` and `upper` are bound endpoints, not a single identified effect estimate.
- `width = upper - lower` reflects residual uncertainty under allowed confounding.
- `claims.json` and `claims.md` summarize robust statements derived only from the bounds.

## 6) Paper workflow (unchanged)

Use this for config-driven reproduction and paper-equivalent workflows:

```bash
python -m itbound run --config docs/cli-config.example.yaml
```

See also:
- [`README.md`](../README.md)
- [`docs/config.md`](config.md)
- [`docs/results_schema_v0.md`](results_schema_v0.md)
