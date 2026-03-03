#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_GIF="${1:-${REPO_ROOT}/docs/media/quick-demo-v5.gif}"
COVER_DELAY="${COVER_DELAY:-1200}"
FRAME_DELAY="${FRAME_DELAY:-45}"

if ! command -v magick >/dev/null 2>&1; then
  echo "Missing dependency: magick (ImageMagick)." >&2
  echo "Install on macOS: brew install imagemagick" >&2
  exit 2
fi

FONT_ARGS=()
if magick -list font 2>/dev/null | grep -Eiq '^\s*Font:\s+Menlo'; then
  FONT_ARGS=(-font "Menlo")
elif magick -list font 2>/dev/null | grep -Eiq '^\s*Font:\s+Courier'; then
  FONT_ARGS=(-font "Courier")
fi

TMP_BASE="${TMPDIR:-/tmp}"
TMP_BASE="${TMP_BASE%/}"
TMP_DIR="$(mktemp -d "${TMP_BASE}/itbound-demo-gif.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT
mkdir -p "${TMP_DIR}/frames"
mkdir -p "$(dirname "${OUT_GIF}")"

LOG_PATH="${TMP_DIR}/demo.log"
DEMO_OUTDIR="${TMP_DIR}/demo_out"

pushd "${REPO_ROOT}" >/dev/null
PYTHONPATH=src python3 -m itbound demo \
  --scenario toy \
  --outdir "${DEMO_OUTDIR}" \
  --num-epochs 1 \
  --n-folds 2 \
  --batch-size 8 >"${LOG_PATH}" 2>&1
popd >/dev/null

SUMMARY_LINES="$(grep -E '^(demo:|saved:)' "${LOG_PATH}" || true)"
if [[ -z "${SUMMARY_LINES}" ]]; then
  SUMMARY_LINES="$(cat <<EOF
demo: toy
saved: ${DEMO_OUTDIR}/toy/summary.txt
saved: ${DEMO_OUTDIR}/toy/results.json
saved: ${DEMO_OUTDIR}/toy/claims.json
saved: ${DEMO_OUTDIR}/toy/plots
saved: ${DEMO_OUTDIR}/live_demo_summary.md
EOF
)"
fi
SUMMARY_LINES="$(
SUMMARY_LINES_RAW="${SUMMARY_LINES}" DEMO_OUTDIR_RAW="${DEMO_OUTDIR}" python3 - <<'PY'
import os
from pathlib import Path

demo_outdir = str(Path(os.environ["DEMO_OUTDIR_RAW"]))
text = os.environ.get("SUMMARY_LINES_RAW", "")
print(text.replace(demo_outdir, "/tmp/itbound_live_demo"))
PY
)"

FRAME_TEXT="${TMP_DIR}/frame.txt"
FRAME_INDEX=0
COMMAND_LINE='$ python -m itbound demo --scenario toy --outdir /tmp/itbound_live_demo --num-epochs 1 --n-folds 2 --batch-size 8'

render_frame() {
  local index="$1"
  local frame_path
  frame_path="$(printf '%s/frames/frame_%03d.png' "${TMP_DIR}" "${index}")"
  if [[ ${#FONT_ARGS[@]} -gt 0 ]]; then
    magick -size 1280x720 xc:"#0d1117" \
      -fill "#c9d1d9" \
      "${FONT_ARGS[@]}" \
      -pointsize 24 \
      -gravity northwest \
      -annotate +36+48 @"${FRAME_TEXT}" \
      "${frame_path}"
  else
    magick -size 1280x720 xc:"#0d1117" \
      -fill "#c9d1d9" \
      -pointsize 24 \
      -gravity northwest \
      -annotate +36+48 @"${FRAME_TEXT}" \
      "${frame_path}"
  fi
}

render_cover_frame() {
  local index="$1"
  local frame_path
  local base_path
  local plot_tmp
  frame_path="$(printf '%s/frames/frame_%03d.png' "${TMP_DIR}" "${index}")"
  base_path="${TMP_DIR}/base_cover.png"
  plot_tmp="${TMP_DIR}/plot_cover.png"

  if [[ ${#FONT_ARGS[@]} -gt 0 ]]; then
    magick -size 1280x720 xc:"#0d1117" \
      -fill "#c9d1d9" \
      "${FONT_ARGS[@]}" \
      -pointsize 24 \
      -gravity northwest \
      -annotate +36+36 @"${FRAME_TEXT}" \
      "${base_path}"
  else
    magick -size 1280x720 xc:"#0d1117" \
      -fill "#c9d1d9" \
      -pointsize 24 \
      -gravity northwest \
      -annotate +36+36 @"${FRAME_TEXT}" \
      "${base_path}"
  fi

  if [[ -f "${DEMO_OUTDIR}/toy/plots/bounds_interval.png" ]]; then
    magick "${DEMO_OUTDIR}/toy/plots/bounds_interval.png" -resize 1160x420 "${plot_tmp}"
    magick "${base_path}" "${plot_tmp}" -gravity south -geometry +0+18 -composite "${frame_path}"
  else
    cp "${base_path}" "${frame_path}"
  fi
}

RESULT_BRIEF="$(
DEMO_OUTDIR_RAW="${DEMO_OUTDIR}" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["DEMO_OUTDIR_RAW"]) / "toy" / "results.json"
if path.is_file():
    obj = json.loads(path.read_text(encoding="utf-8"))
    b = obj.get("bounds", {})
    lower = b.get("lower", {})
    upper = b.get("upper", {})
    width = b.get("width", {})
    print(f"rows={b.get('n_rows', 'n/a')}, valid={b.get('n_valid_intervals', 'n/a')}, valid_rate={b.get('valid_interval_rate', 'n/a')}")
    print(f"mean(lower)={lower.get('mean', 'n/a')}, mean(upper)={upper.get('mean', 'n/a')}, mean(width)={width.get('mean', 'n/a')}")
else:
    print("results.json not found (fallback mode)")
PY
)"
if [[ ! -f "${DEMO_OUTDIR}/toy/plots/bounds_interval.png" ]]; then
  RESULT_BRIEF="${RESULT_BRIEF}
plot: not available (install extras: pip install itbound[experiments])"
fi

cat >"${FRAME_TEXT}" <<EOF
itbound demo result at a glance
${RESULT_BRIEF}
EOF
render_cover_frame "${FRAME_INDEX}"
FRAME_INDEX=$((FRAME_INDEX + 1))

for SPINNER in "running." "running.." "running..." "running...."; do
  cat >"${FRAME_TEXT}" <<EOF
${COMMAND_LINE}

${SPINNER}
EOF
  render_frame "${FRAME_INDEX}"
  FRAME_INDEX=$((FRAME_INDEX + 1))
done

cat >"${FRAME_TEXT}" <<EOF
${COMMAND_LINE}

running...
EOF
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  printf '%s\n' "${line}" >>"${FRAME_TEXT}"
  render_frame "${FRAME_INDEX}"
  FRAME_INDEX=$((FRAME_INDEX + 1))
done <<<"${SUMMARY_LINES}"

LAST_FRAME="${TMP_DIR}/frames/frame_$(printf '%03d' "$((FRAME_INDEX - 1))").png"
cp "${LAST_FRAME}" "${TMP_DIR}/frames/frame_$(printf '%03d' "${FRAME_INDEX}").png"
FRAME_INDEX=$((FRAME_INDEX + 1))
cp "${LAST_FRAME}" "${TMP_DIR}/frames/frame_$(printf '%03d' "${FRAME_INDEX}").png"

shopt -s nullglob
FRAMES=( "${TMP_DIR}"/frames/frame_*.png )
if [[ ${#FRAMES[@]} -eq 0 ]]; then
  echo "No frames generated." >&2
  exit 2
fi

FIRST_FRAME="${FRAMES[0]}"
REST_FRAMES=( "${FRAMES[@]:1}" )
if [[ ${#REST_FRAMES[@]} -gt 0 ]]; then
  magick -delay "${COVER_DELAY}" "${FIRST_FRAME}" \
    -delay "${FRAME_DELAY}" "${REST_FRAMES[@]}" \
    -loop 0 -layers Optimize "${OUT_GIF}"
else
  magick -delay "${COVER_DELAY}" "${FIRST_FRAME}" -loop 0 "${OUT_GIF}"
fi

echo "saved: ${OUT_GIF}"
