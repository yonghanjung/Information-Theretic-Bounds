# itbound v0.1.0 Release Runbook

This runbook is the exact release checklist for `v0.1.0`.

## 0. Release lane checks

```bash
git branch --show-current
git status --short
```

Expected:
- Branch is `release/v0.1.0-pypi-itbound`
- Working tree is clean

## 1. Local preflight (fast gate)

```bash
bash scripts/release_preflight.sh
```

This runs:
- `pytest -m "not slow"`
- `python -m build`
- fresh-venv wheel smoke (`itbound --help`, `itbound example`, `itbound quick`)

## 2. Version and changelog freeze

Confirm:
- `pyproject.toml` has `version = "0.1.0"`
- `src/itbound/__init__.py` has `__version__ = "0.1.0"`
- `CHANGELOG.md` has frozen `0.1.0` section

## 3. Tag and push

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin release/v0.1.0-pypi-itbound
git push origin v0.1.0
```

## 4. GitHub Actions gate verification

Workflow: `.github/workflows/release.yml`

Required order and gates:
1. `build-dist`
2. `publish-testpypi`
3. `install-smoke-testpypi`
4. `publish-pypi` (must wait for step 3 success)

If `install-smoke-testpypi` fails, `publish-pypi` must not run.

## 5. Post-release fresh-venv verification

```bash
python3 -m venv /tmp/itbound-pypi-verify
/tmp/itbound-pypi-verify/bin/python -m pip install -U pip
/tmp/itbound-pypi-verify/bin/python -m pip install itbound==0.1.0
/tmp/itbound-pypi-verify/bin/itbound --help
/tmp/itbound-pypi-verify/bin/itbound example --out /tmp/itbound-pypi-example.csv
/tmp/itbound-pypi-verify/bin/python - <<'PY'
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 80
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
a = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
y = 0.3 + 0.7 * a + 0.2 * x1 - 0.1 * x2 + rng.normal(scale=0.1, size=n)
pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2}).to_csv("/tmp/itbound-pypi-quick-input.csv", index=False)
PY
/tmp/itbound-pypi-verify/bin/itbound quick --data /tmp/itbound-pypi-quick-input.csv --treatment a --outcome y --covariates x1,x2 --outdir /tmp/itbound-pypi-out
```

Required file:
- `/tmp/itbound-pypi-out/results.json`

## 6. Rollback notes

- If TestPyPI publish fails: fix trusted publisher/project settings and retag with next patch version.
- If TestPyPI install smoke fails: do not publish to PyPI; fix and retag.
- If PyPI publish succeeds but smoke fails: publish immediate patch release (`v0.1.1`).
