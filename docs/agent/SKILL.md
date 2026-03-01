# itbound Skill

Use this skill to compute data-driven causal bounds from configuration files.

## Commands

- Run bounds from config:
  - `itbound run --config <path/to/config.yaml>`
- Run quick synthetic example:
  - `itbound example --out itbound_example.csv`
- Reproduce arXiv plots (dry run):
  - `itbound reproduce --dry-run`

## Config Input (YAML/JSON)

Required keys:
- `data`: one of `synthetic`, `npz_path`, or `csv_path`
- `divergence`
- `propensity_model`
- `m_model`
- `dual_net_config`
- `fit_config`
- `seed`

Optional:
- `phi` (default: `identity`)
- `output_path` (default: `itbound_bounds.csv`)

See `docs/cli-config.example.yaml` for a full example.

## Notes
- This repo keeps `fbound` intact and exposes `import itbound` as a wrapper.
- No deletion is performed by this skill.
