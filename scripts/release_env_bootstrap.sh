#!/usr/bin/env bash
set -euo pipefail

REPO_SLUG="${REPO_SLUG:-yonghanjung/Information-Theretic-Bounds}"
CHECK_ONLY=0
if [ "${1:-}" = "--check-only" ]; then
  CHECK_ONLY=1
fi

echo "[env-bootstrap] repo=${REPO_SLUG}"
gh auth status -h github.com >/dev/null

if [ "${CHECK_ONLY}" -ne 1 ]; then
  echo "[env-bootstrap] ensuring environments exist (idempotent)"
  gh api "repos/${REPO_SLUG}/environments/testpypi" --method PUT >/dev/null
  gh api "repos/${REPO_SLUG}/environments/pypi" --method PUT >/dev/null
fi

ENV_NAMES=()
while IFS= read -r name; do
  if [ -n "${name}" ]; then
    ENV_NAMES+=("${name}")
  fi
done <<EOF
$(gh api "repos/${REPO_SLUG}/environments" --jq '.environments[].name')
EOF

has_testpypi=0
has_pypi=0
for name in "${ENV_NAMES[@]:-}"; do
  if [ "${name}" = "testpypi" ]; then
    has_testpypi=1
  fi
  if [ "${name}" = "pypi" ]; then
    has_pypi=1
  fi
done

if [ "${has_testpypi}" -ne 1 ] || [ "${has_pypi}" -ne 1 ]; then
  echo "[env-bootstrap] Missing required environments. Found: ${ENV_NAMES[*]:-<none>}" >&2
  exit 1
fi

echo "[env-bootstrap] OK (testpypi, pypi)"
