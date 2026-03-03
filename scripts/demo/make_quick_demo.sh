#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_GIF="${1:-${REPO_ROOT}/docs/media/quick-demo.gif}"

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

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/itbound-demo-gif.XXXXXX")"
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
  --batch-size 8 \
  --no-plots >"${LOG_PATH}" 2>&1
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

FRAME_TEXT="${TMP_DIR}/frame.txt"
FRAME_INDEX=0

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

cat >"${FRAME_TEXT}" <<'EOF'
$ python -m itbound demo --scenario toy --outdir /tmp/itbound_live_demo --num-epochs 1 --n-folds 2 --batch-size 8 --no-plots
EOF
render_frame "${FRAME_INDEX}"
FRAME_INDEX=$((FRAME_INDEX + 1))

cat >>"${FRAME_TEXT}" <<'EOF'

running...
EOF
render_frame "${FRAME_INDEX}"
FRAME_INDEX=$((FRAME_INDEX + 1))

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

magick -delay 85 "${TMP_DIR}"/frames/frame_*.png -loop 0 -layers Optimize "${OUT_GIF}"
echo "saved: ${OUT_GIF}"
