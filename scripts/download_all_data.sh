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

echo "========== 6/6 MT (EN→UK via SDKM + pseudo-docs) =========="
echo "Building EN→UK training data (download → filter/top-75 → pseudo-docs)."
echo "This step is heavy (GPU embeddings). To skip: SKIP_EN_UK_MT=1"
if [[ "${SKIP_EN_UK_MT:-0}" == "1" ]]; then
  echo "SKIP_EN_UK_MT=1 — not running prepare_en_uk_mt.sh"
  echo "  Manual options:"
  echo "    bash scripts/prepare_en_uk_mt.sh"
  echo "    or place jsonl at train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl"
else
  bash scripts/prepare_en_uk_mt.sh
fi
echo ""
echo "CS→UK paragraph MT (optional):"
echo "  bash scripts/prepare_corpus_mt.sh"
echo ""
echo "Optional: translate QA/MR to Ukrainian for Stage 2 bilingual training:"
echo "  bash scripts/run_translate_en_uk.sh"
echo "  (adds question_uk/answer_uk fields to train_data/MR/gsm8k_train.jsonl"
echo "   and train_data/MR/competition_math_train.jsonl)"
echo ""
echo "Then clean/normalize MR into boxed-answer chat format:"
echo "  python3 scripts/clean_mr_data.py"
echo "  -> train_data/MR/mr_train.jsonl"
echo ""
echo "Data preparation finished. See README.md for paths."
