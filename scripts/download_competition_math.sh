#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/train_data/MR/competition_math"

# hendrycks/competition_math is DMCA-disabled on HuggingFace; EleutherAI mirror is equivalent.
unset HF_ENDPOINT
mkdir -p "$ROOT/train_data/MR" "$ROOT/outputs"

echo "Downloading competition_math (EleutherAI/hendrycks_math mirror)..."
hf download EleutherAI/hendrycks_math \
  --repo-type dataset \
  --local-dir "$OUT"

echo "Exporting jsonl..."
python3 "$ROOT/scripts/prepare_mr_data.py" --skip-gsm8k

echo "Done: $OUT"
echo "       $ROOT/train_data/MR/competition_math_train.jsonl"
