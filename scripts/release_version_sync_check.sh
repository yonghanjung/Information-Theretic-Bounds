#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYPROJECT_VERSION="$(awk -F'"' '/^version = "/ {print $2; exit}' pyproject.toml)"
INIT_VERSION="$(awk -F'"' '/^__version__ = "/ {print $2; exit}' src/itbound/__init__.py)"

if [ -z "${PYPROJECT_VERSION}" ] || [ -z "${INIT_VERSION}" ]; then
  echo "[version-sync] Missing version value in pyproject.toml or src/itbound/__init__.py" >&2
  exit 1
fi

if [ "${PYPROJECT_VERSION}" != "${INIT_VERSION}" ]; then
  echo "[version-sync] Version mismatch: pyproject=${PYPROJECT_VERSION}, __init__=${INIT_VERSION}" >&2
  exit 1
fi

if ! grep -Fq "## [${PYPROJECT_VERSION}]" CHANGELOG.md; then
  echo "[version-sync] Missing changelog section for version ${PYPROJECT_VERSION}" >&2
  exit 1
fi

echo "[version-sync] OK (${PYPROJECT_VERSION})"
