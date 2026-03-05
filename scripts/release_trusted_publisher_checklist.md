# Trusted Publisher Checklist (itbound v0.1.0)

Use this checklist for both TestPyPI and PyPI project settings.

## Common values

- Owner: `yonghanjung`
- Repository: `Information-Theretic-Bounds`
- Workflow: `.github/workflows/release.yml`
- Ref: `refs/tags/*`

## TestPyPI

- Environment: `testpypi`
- URL: `https://test.pypi.org/`

## PyPI

- Environment: `pypi`
- URL: `https://pypi.org/`

## Verify before tagging

1. `bash scripts/release_env_bootstrap.sh --check-only`
2. Confirm both environments exist in GitHub: `testpypi`, `pypi`
3. Confirm Trusted Publisher values exactly match the Common values above
4. Confirm release workflow path is `.github/workflows/release.yml`
