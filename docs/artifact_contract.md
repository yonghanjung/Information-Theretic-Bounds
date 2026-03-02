# Artifact Contract

This document defines the output folder contract used by opt-in standard-library commands (`quick`, `artifacts`) and `BoundsReport.save(...)`.

## Contract directory

All artifacts are written under the user-provided output directory (`--outdir` in CLI).

Required entries:
- `summary.txt`
- `results.json`
- `claims.json`
- `claims.md`
- `plots/` (may be empty when plotting dependencies are missing)

Optional entries:
- `report.html` (only when HTML output is requested)

## File meanings

- `summary.txt`: human-readable run summary and warnings.
- `results.json`: machine-readable results payload (schema-versioned).
- `claims.json`: structured claims payload derived from bounds.
- `claims.md`: readable claims report for sharing/review.
- `plots/`: generated visual artifacts (for example `bounds_interval.png`).
- `report.html`: optional report combining payload and plots.

## Schema and compatibility

`results.json` is versioned and currently uses:
- `schema_version = "results_schema_v0"`

See:
- Human-readable schema: [`docs/results_schema_v0.md`](results_schema_v0.md)
- Machine-readable schema: [`docs/results_schema_v0.json`](results_schema_v0.json)

Compatibility policy summary:
- Existing required fields for `results_schema_v0` remain stable.
- New optional fields may be added.
- Breaking changes require a new schema version.
