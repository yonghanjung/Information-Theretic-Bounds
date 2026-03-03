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

Toy-only:

```bash
python -m itbound demo --scenario toy --toy-n 1000 --rounds 5 --n-folds 5 --outdir /tmp/itbound_live_demo --batch-size 8
```

IHDP benchmark CSV:

```bash
python -m itbound demo --scenario ihdp --ihdp-data /path/to/ihdp_npci_1.csv --rounds 5 --n-folds 5 --outdir /tmp/itbound_live_demo --batch-size 8
```

Both scenarios:

```bash
python -m itbound demo --scenario both --toy-n 1000 --rounds 5 --n-folds 5 --outdir /tmp/itbound_live_demo --batch-size 8
```

Demo output layout:
- `--outdir/toy/` and/or `--outdir/ihdp/` with `summary.txt`, `results.json`, `claims.json`, `claims.md`, `plots/`
- `--outdir/live_demo_summary.md`

README includes a toy-demo GIF preview at `docs/media/quick-demo-v5.gif`.
Rebuild it with:

```bash
bash scripts/demo/make_quick_demo.sh
```

## 3) Python quick path

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

## 4) Interpreting outputs safely

- `lower` and `upper` are bound endpoints, not a single identified effect estimate.
- `width = upper - lower` reflects residual uncertainty under allowed confounding.
- `claims.json` and `claims.md` summarize robust statements derived only from the bounds.

## 5) Paper workflow (unchanged)

Use this for config-driven reproduction and paper-equivalent workflows:

```bash
python -m itbound run --config docs/cli-config.example.yaml
```

See also:
- [`README.md`](../README.md)
- [`docs/config.md`](config.md)
- [`docs/results_schema_v0.md`](results_schema_v0.md)
