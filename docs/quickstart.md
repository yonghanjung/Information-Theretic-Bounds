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

## 2) Python quick path

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

## 3) Interpreting outputs safely

- `lower` and `upper` are bound endpoints, not a single identified effect estimate.
- `width = upper - lower` reflects residual uncertainty under allowed confounding.
- `claims.json` and `claims.md` summarize robust statements derived only from the bounds.

## 4) Paper workflow (unchanged)

Use this for config-driven reproduction and paper-equivalent workflows:

```bash
python -m itbound run --config docs/cli-config.example.yaml
```

See also:
- [`README.md`](../README.md)
- [`docs/config.md`](config.md)
- [`docs/results_schema_v0.md`](results_schema_v0.md)
