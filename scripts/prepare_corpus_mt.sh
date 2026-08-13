#!/usr/bin/env bash
# Build the cs-uk MT training set via scripts/prepare_cs_paragraph_mt.py, which runs
# every cs-uk corpus builder in loader.py (eubookshop, elrc, wikimedia, nllb),
# back-translates UberText fiction uk->cs via LINDAT, and assembles everything into
# train_data/MT/train.cs-uk.jsonl.
#
# Usage:
#   bash scripts/prepare_corpus_mt.sh              # everything, with semantic alignment
#   SKIP_ALIGN=1 bash scripts/prepare_corpus_mt.sh  # skip the semantic-alignment stage
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

ALIGN_FLAG=()
if [[ "${SKIP_ALIGN:-0}" == "1" ]]; then
  ALIGN_FLAG=(--no-align)
  echo "SKIP_ALIGN=1 — semantic alignment stage disabled."
fi

python3 scripts/prepare_cs_paragraph_mt.py "${ALIGN_FLAG[@]}"

echo
echo "Done. See train_data/MT/train.cs-uk.jsonl"
