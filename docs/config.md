# itbound CLI Config Reference

This document explains the YAML/JSON configuration file used by `itbound run`.

## 1) Overview

A config file is a single YAML/JSON object with these top-level keys:

- `data` (required)
- `divergence` (required)
- `propensity_model` (required)
- `m_model` (required)
- `dual_net_config` (required)
- `fit_config` (required)
- `seed` (required)
- `phi` (optional, default: `identity`)
- `output_path` (optional, default: `itbound_bounds.csv`)

The CLI reads the file, validates required fields, and runs bounds estimation.

## 2) File formats

- YAML: `.yml` or `.yaml`
- JSON: `.json`

## 3) `data` section

Exactly **one** of the following must be provided:

### A) Synthetic data

```yaml
data:
  synthetic:
    n: 200
    d: 3
    seed: 123
    structural_type: linear
```

Required keys:
- `n` (int): sample size
- `d` (int): feature dimension

Optional keys:
- `seed` (int): defaults to top-level `seed`
- `structural_type` (str): e.g., `linear`

### B) NPZ file

```yaml
data:
  npz_path: path/to/data.npz
```

The NPZ file must include arrays: `Y`, `A`, `X`.

### C) CSV file

```yaml
data:
  csv_path: path/to/data.csv
  y_col: y
  a_col: a
  x_cols: [x1, x2, x3]
```

Required keys:
- `csv_path` (str)
- `y_col` (str)
- `a_col` (str)
- `x_cols` (list of str)

## 4) Divergence and link function

### `divergence`

Use the same divergence names as the codebase supports (e.g., `KL`, `Hellinger`, `Chi2`, `TV`, `JS`).

```yaml
divergence: KL
```

### `phi`

Currently supported:
- `identity`

```yaml
phi: identity
```

## 5) Nuisance models

### `propensity_model`

Example:

```yaml
propensity_model: logistic
```

### `m_model`

Example:

```yaml
m_model: linear
```

These values must be supported by the underlying `fbound` estimators.

## 6) `dual_net_config`

Configures the neural nets used in the dual formulation.

Example:

```yaml
dual_net_config:
  hidden_sizes: [16, 16]
  activation: relu
  dropout: 0.0
  h_clip: 10.0
  device: cpu
```

Common fields:
- `hidden_sizes`: list of layer widths
- `activation`: `relu` or another supported activation
- `dropout`: float in [0, 1]
- `h_clip`: positive float
- `device`: `cpu` or `cuda`

## 7) `fit_config`

Controls training and cross-fitting.

Example:

```yaml
fit_config:
  n_folds: 2
  num_epochs: 3
  batch_size: 32
  lr: 0.005
  weight_decay: 0.0
  max_grad_norm: 5.0
  eps_propensity: 0.001
  deterministic_torch: true
  train_m_on_fold: true
  propensity_config:
    C: 1.0
    max_iter: 200
    penalty: l2
    solver: lbfgs
    n_jobs: 1
  m_config:
    alpha: 1.0
  verbose: false
  log_every: 1
```

Key fields:
- `n_folds`: number of cross-fitting folds
- `num_epochs`: training epochs
- `batch_size`: mini-batch size
- `lr`: learning rate
- `weight_decay`: weight decay for optimizers
- `max_grad_norm`: gradient clipping
- `eps_propensity`: propensity floor
- `deterministic_torch`: determinism toggle
- `train_m_on_fold`: whether to train outcome model on each fold
- `propensity_config`: sklearn-style options for the propensity model
- `m_config`: sklearn-style options for the outcome model
- `verbose`: print training logs
- `log_every`: logging frequency

## 8) Output and seed

```yaml
seed: 123
output_path: itbound_bounds.csv
```

- `seed`: used for RNG in data generation and training
- `output_path`: where the output CSV is written (overridable with `--out`)

## 9) Full example

See `docs/cli-config.example.yaml` for a complete working configuration.
