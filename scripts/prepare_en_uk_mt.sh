#!/usr/bin/env bash
# Build EN→UK MT training data:
#   1) download OPUS/NLLB parallel corpora
#   2) basic filter + SDKM top-K similarity retrieval vs. official EN-UK dev
#   3) semantic clustering → pseudo-documents for Stage-1 LoRA
#
# Usage:
#   bash scripts/prepare_en_uk_mt.sh
#   SKIP_DOWNLOAD=1 bash scripts/prepare_en_uk_mt.sh   # reuse data/raw/eng-ukr/
#   TOP_K=75 bash scripts/prepare_en_uk_mt.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

TOP_K="${TOP_K:-75}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_NLLB="${SKIP_NLLB:-0}"
MAX_PER_CORPUS="${MAX_PER_CORPUS:-}"

echo "========== 1/3 Download EN-UK parallel data =========="
if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  echo "SKIP_DOWNLOAD=1 — using existing data/raw/eng-ukr/train.eng|.ukr"
else
  DL_ARGS=()
  if [[ "$SKIP_NLLB" == "1" ]]; then
    DL_ARGS+=(--skip-nllb)
  fi
  if [[ -n "$MAX_PER_CORPUS" ]]; then
    DL_ARGS+=(--max-per-corpus "$MAX_PER_CORPUS")
  fi
  python3 scripts/download_mt_en_uk.py "${DL_ARGS[@]}"
fi

if [[ ! -f data/raw/eng-ukr/train.eng || ! -f data/raw/eng-ukr/train.ukr ]]; then
  echo "ERROR: missing data/raw/eng-ukr/train.eng or train.ukr" >&2
  exit 1
fi

echo
echo "========== 2/3 Filter + SDKM top-${TOP_K} retrieval =========="
python3 scripts/filter_mt_en_uk.py --top-k "$TOP_K"

echo
echo "========== 3/3 Build EN-UK pseudo-documents =========="
python3 scripts/build_pseudo_docs_en_uk.py

echo
echo "Done."
echo "  selected sentences : data/processed/selected_en_uk.jsonl"
echo "  train pseudo-docs  : data/pseudo_docs/train_pseudo_en_uk.jsonl"
echo "  Stage-1 MT path    : train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl"
