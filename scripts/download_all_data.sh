#!/usr/bin/env bash
# Download and prepare all training/eval data for two-stage LoRA fine-tuning.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

echo "========== 1/6 Dev evaluation data =========="
bash scripts/download_dev_data.sh

echo "========== 2/6 QA (MMLU en + ukr) =========="
bash scripts/download_qa_mmlu.sh

echo "========== 3/6 MR (GSM8K + competition_math) =========="
python3 scripts/prepare_mr_data.py --skip-comp-math
bash scripts/download_competition_math.sh

echo "========== 4/6 GC (UA-GEC) =========="
python3 scripts/prepare_ua_gec_gc.py

echo "========== 5/6 SC (synthetic from mmlu_ukr) =========="
python3 scripts/prepare_mmlu_ukr_sc.py

echo "========== 6/6 MT note =========="
echo "MT parallel data is not auto-downloaded."
echo "  Option A: place en-uk parallel jsonl at train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl"
echo "  Option B: use version-1 combined corpus (train.jsonl.gz) with --mt-format version1"
echo ""
echo "Optional: translate QA/MR to Ukrainian for Stage 2 bilingual training:"
echo "  bash scripts/run_translate_en_uk.sh"
echo ""
echo "Data preparation finished. See README.md for paths."
