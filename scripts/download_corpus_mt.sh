#!/usr/bin/env bash
# Fetch the corpus raw sources that scripts/loader.py and scripts/prepare_cs_paragraph_mt.py
# cannot download themselves.
#
# The cs-uk MT corpora (EUbookshop/ELRC/Wikimedia/NLLB OPUS zips + HuggingFace datasets,
# UberText fiction) are downloaded on demand by `scripts/loader.py <corpus>` and
# scripts/prepare_cs_paragraph_mt.py the first time they run. Only the source below
# has no such built-in downloader:
#
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

# ── WMT25 official Constrained-Track recipes (mtdata) + QE scores ────────────
echo "[wmt25] fetching official en-ukr / ces-ukr recipes via mtdata..."
python3 scripts/wmt25.py download --all --out-dir "$ROOT/data/raw/wmt25"
python3 scripts/wmt25.py qe       --all --out-dir "$ROOT/data/raw/wmt25"

echo
echo "Done. Raw sources ready in:"
echo "  $QUARC_DIR"
echo "  $ROOT/data/raw/wmt25"
