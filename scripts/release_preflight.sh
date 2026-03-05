#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TMP_VENV="${TMP_VENV:-/tmp/itbound-v010-preflight-venv}"
TMP_OUTDIR="${TMP_OUTDIR:-/tmp/itbound-v010-preflight-out}"
TMP_EXAMPLE="${TMP_EXAMPLE:-/tmp/itbound-v010-preflight-example.csv}"
TMP_QUICK_INPUT="${TMP_QUICK_INPUT:-/tmp/itbound-v010-preflight-quick-input.csv}"

echo "[preflight] repo: ${ROOT_DIR}"
echo "[preflight] python: $(python3 --version)"
echo "[preflight] running fast tests"
python3 -m pytest -q -m "not slow"

echo "[preflight] building wheel/sdist"
python3 -m build

echo "[preflight] wheel install smoke"
rm -rf "${TMP_VENV}" "${TMP_OUTDIR}" "${TMP_EXAMPLE}"
python3 -m venv "${TMP_VENV}"
"${TMP_VENV}/bin/python" -m pip install -U pip
"${TMP_VENV}/bin/python" -m pip install dist/*.whl
export PATH="${TMP_VENV}/bin:${PATH}"
itbound --help >/dev/null
itbound example --out "${TMP_EXAMPLE}" >/dev/null
"${TMP_VENV}/bin/python" - <<'PY'
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 80
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
a = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
y = 0.3 + 0.7 * a + 0.2 * x1 - 0.1 * x2 + rng.normal(scale=0.1, size=n)
pd.DataFrame({"y": y, "a": a, "x1": x1, "x2": x2}).to_csv("/tmp/itbound-v010-preflight-quick-input.csv", index=False)
PY
itbound quick \
  --data "${TMP_QUICK_INPUT}" \
  --treatment a \
  --outcome y \
  --covariates x1,x2 \
  --outdir "${TMP_OUTDIR}" >/dev/null

test -f "${TMP_OUTDIR}/results.json"

echo "[preflight] OK"
