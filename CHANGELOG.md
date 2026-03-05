# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2026-03-05

### Added
- Codex execution tutorial for replicating the FD-CATE PyPI/release pattern in this repository (`TUTORIAL_PYPI_ITBOUND_FROM_FDCATE.md`).

## [0.1.0] - 2026-03-05

### Added
- GitHub Actions CI workflow with Python 3.9/3.11 matrix and wheel/CLI smoke checks.
- GitHub Actions Release workflow with `build-dist -> publish-testpypi -> install-smoke-testpypi -> publish-pypi` gates.
- Local release preflight script for fast tests + wheel install smoke.
- Release runbook for tag/publish/verification and rollback procedure.

### Changed
- README image links updated for PyPI-safe absolute rendering.
- Packaging metadata includes `dev` extra for release engineering dependencies.
