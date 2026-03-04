# Aggregation Modes (`itbound standard` / `itbound artifacts`)

This page explains endpoint aggregation across multiple divergences.

When you run with multiple divergences (for example `KL,JS,Hellinger,TV,Chi2`), `itbound` aggregates row-wise lower/upper candidates into one interval per row.

## Modes

### `paper_adaptive_k` (default)
- Starts from strict endpoint candidates (`k=1`) and increases `k` until it finds a feasible interval (`lower <= upper`).
- Matches the current paper-default endpoint adaptation behavior in this standard wrapper.

### `fixed_k_endpoint`
- Uses exactly the requested `k` (`--fixed-k`).
- If either side has fewer than `k` valid candidates, that row becomes invalid.

### `tight_kth`
- Uses experiment-style `tight_kth` order-statistics aggregation:
  - `lower = k`-th smallest lower candidate
  - `upper = k`-th largest upper candidate
  - starts from a large `k` and relaxes down until feasible.
- `--fixed-k` acts as an optional upper cap for the starting `k`.
  - Default `--fixed-k 1` means automatic max-k start (recommended).
- Default divergence list is the built-in 5:
  - `KL,JS,Hellinger,TV,Chi2`
- You can override with any explicit subset using `--divergences` (for example `KL,TV`).

## CLI examples

```bash
# Default paper-adaptive aggregation
python -m itbound standard \
  --csv /tmp/toy.csv --y-col y --a-col a --x-cols x1,x2 \
  --divergences KL,JS,Hellinger,TV,Chi2 \
  --aggregation-mode paper_adaptive_k
```

```bash
# Tight-kth aggregation
python -m itbound standard \
  --csv /tmp/toy.csv --y-col y --a-col a --x-cols x1,x2 \
  --aggregation-mode tight_kth
```

```bash
# Tight-kth with a subset override
python -m itbound standard \
  --csv /tmp/toy.csv --y-col y --a-col a --x-cols x1,x2 \
  --divergences KL,TV \
  --aggregation-mode tight_kth
```

```bash
# Fixed endpoint-k aggregation
python -m itbound standard \
  --csv /tmp/toy.csv --y-col y --a-col a --x-cols x1,x2 \
  --divergences KL,JS,Hellinger,TV,Chi2 \
  --aggregation-mode fixed_k_endpoint --fixed-k 2
```

Diagnostics are written to `summary.json` / `results.json` and include:
- `aggregation_mode`
- `k_used`
- `tight_k_start` (for `tight_kth`)
- filtering counts (`invalid_*`, `non_finite_*`, `inverted_interval_filtered`)

Ground-truth plotting metadata in `summary.json` (`run_config.ground_truth_plot`):
- `source`: `explicit_col`, `explicit_effect`, `auto_mu1_mu0`, or `none`
- `n_truth_points`
- `ground_truth_effect`
- `warnings`

Ground-truth overlay options:
- `--ground-truth-col <col>`
- `--ground-truth-effect <float>`
- `--no-ground-truth-auto`
