#!/usr/bin/env bash
#   QUA-RC   reading-comprehension MCQ data used by scripts/qa/convert_rc.py — plain
#            GitHub repo, no HuggingFace/OPUS mirror, so `git clone` is the only route.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── QUA-RC (reading comprehension MCQ, pinned commit) ─────────────────────────
QUARC_DIR="$ROOT/QUA-RC"
QUARC_COMMIT="5cf4a13dc93cb543fa3bab111224f72e55993c6d"
if [[ -f "$QUARC_DIR/quarc.json" ]]; then
  echo "[qua-rc] already present: $QUARC_DIR/quarc.json"
else
  echo "[qua-rc] cloning dkalpakchi/QUA-RC @ $QUARC_COMMIT..."
  git clone https://github.com/dkalpakchi/QUA-RC.git "$QUARC_DIR"
  git -C "$QUARC_DIR" checkout "$QUARC_COMMIT"
  echo "[qua-rc] saved to $QUARC_DIR"
fi


