#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

REPO_SLUG="${REPO_SLUG:-yonghanjung/Information-Theretic-Bounds}"
TAG="v0.1.0"
ALLOW_DIRTY=0
CHECK_ONLY=0
SKIP_AUTH_CHECK=0
SKIP_ENV_CHECK=0
SKIP_VERSION_CHECK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)
      TAG="${2:?missing value for --tag}"
      shift 2
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --skip-auth-check)
      SKIP_AUTH_CHECK=1
      shift
      ;;
    --skip-env-check)
      SKIP_ENV_CHECK=1
      shift
      ;;
    --skip-version-check)
      SKIP_VERSION_CHECK=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "[release-gate] repo=${REPO_SLUG} tag=${TAG}"

if [ "${SKIP_AUTH_CHECK}" -ne 1 ]; then
  gh auth status -h github.com >/dev/null
fi

if [ "${SKIP_ENV_CHECK}" -ne 1 ]; then
  bash scripts/release_env_bootstrap.sh --check-only
fi

if [ "${SKIP_VERSION_CHECK}" -ne 1 ]; then
  bash scripts/release_version_sync_check.sh
fi

if [ "${ALLOW_DIRTY}" -ne 1 ] && [ -n "$(git status --porcelain)" ]; then
  echo "[release-gate] Working tree is not clean" >&2
  exit 1
fi

if [ -n "$(git tag -l "${TAG}")" ]; then
  echo "[release-gate] Local tag already exists: ${TAG}" >&2
  exit 1
fi

if git ls-remote --tags origin "refs/tags/${TAG}" | grep -q "${TAG}"; then
  echo "[release-gate] Remote tag already exists: ${TAG}" >&2
  exit 1
fi

if [ "${CHECK_ONLY}" -eq 1 ]; then
  echo "[release-gate] CHECK-ONLY OK"
  exit 0
fi

echo "[release-gate] Executing push flow for ${TAG}"
git push origin "$(git branch --show-current)"
git tag -a "${TAG}" -m "${TAG}"
git push origin "${TAG}"
echo "[release-gate] EXECUTE OK"
