# itbound `results.json` Schema v0

This document defines the **stable artifact schema v0** for `results.json`.

- Schema ID: `results_schema_v0`
- Machine-readable JSON Schema: `docs/results_schema_v0.json`
- Intended use: downstream integrations, reproducible reporting, and audit trails.

## Stability Promise (v0)

For schema v0:

1. Required fields listed in this document remain present.
2. Existing field meanings remain stable.
3. New optional fields may be added without breaking compatibility.
4. Breaking changes require a new schema version value and a new schema document.

## Backward Compatibility Policy

- Consumers should check `schema_version`.
- For `schema_version == "results_schema_v0"`, consumers may rely on required fields below.
- Future versions may add fields; consumers should ignore unknown fields.

## Top-level fields

- `schema_version` (string): Must be `results_schema_v0`.
- `provenance` (object): Run metadata and assumptions.
- `bounds` (object): Aggregated bound outputs.
- `diagnostics` (object): Diagnostic payload (may include stubs in v0).

## `provenance` fields (required)

- `package_version` (string): `itbound.__version__` at run time.
- `git_commit` (string|null): Best-effort git commit hash.
- `git_commit_reason` (string|null): Non-null reason when `git_commit` is unavailable.
- `timestamp` (string): Run timestamp in ISO-8601 UTC format.
- `random_seed` (integer): Run seed used by the command.
- `assumptions` (string|object): Explicit assumptions for interpretation.
- `data_hash` (string|null): Best-effort SHA-256 hash.
  - For file input (CSV), this is the file hash.
- `data_hash_reason` (string|null): Non-null reason when `data_hash` is unavailable.
- `command_line` (string|null): Best-effort command invocation string.

## `bounds` fields (required)

- `n_rows` (integer)
- `n_valid_intervals` (integer)
- `lower` (object): `{min,max,mean,median}` numeric-or-null summary stats over valid rows.
- `upper` (object): `{min,max,mean,median}` numeric-or-null summary stats over valid rows.
- `width` (object): `{min,max,mean,median}` numeric-or-null summary stats over valid rows.

## `diagnostics` fields

v0 requires the `diagnostics` object to exist.

- `diagnostics` may include:
  - `missingness` (implemented or stub)
  - `overlap` (implemented or stub)
  - `warnings` (optional)
  - additional nested diagnostics (for example, aggregation diagnostics)

### Recommended Diagnostics Keys for Stable Integrations

For CLI `quick` outputs in this repository, v0 diagnostics include the following keys:

- `interval_validity`
  - `n_rows`
  - `n_valid_intervals`
  - `invalid_interval_count`
  - `valid_interval_rate`
  - `invalid_interval_rate`
- `nan_propagation`
  - `finite_lower_count`
  - `finite_upper_count`
  - `finite_width_count`
  - `non_finite_any_count`
  - `by_reason` (for example, `non_finite_lower`, `non_finite_upper`, `invalid_up_mask`, `invalid_lo_mask`, `inverted_interval`)
- `invalid_domain`
  - `available`
  - `rate`
  - `count`
  - `source`
  - `note`
- `aggregation`
  - If not applicable, report explicit not-applicable status and preserve the slot.
