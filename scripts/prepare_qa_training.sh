#!/usr/bin/env bash
# Build the Ukrainian QA training set.
#
# ZNO : QUA-RC + Belebele (scripts/qa/convert_rc.py) + the 2026-official ZNO
#       train file, combined via scripts/qa/build_cot.py (DUMY CoT matching +
#       2x augmentation + RC merge)
#         -> train_data/QA/ukr_zno_qa_train.jsonl
#
# MMLU: train_data/QA/ukr_mmlu_qa_train.jsonl (2026-official, 1531 items) is
#       already on disk. Converting it to chat format is not wired up here yet
#       — drop a train_data/QA/ukr_mmlu_qa_train_chat.jsonl file in place and
#       the merge step below will pick it up automatically.
#
# Merge: train_data/QA/train_merged.jsonl
# Usage:
#   bash scripts/prepare_qa_training.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

QA_DIR="$ROOT/train_data/QA"
ZNO_TRAIN="$QA_DIR/ukr_qa_train.jsonl"
ZNO_DEV="$QA_DIR/ukr_qa_dev.jsonl"

if [[ ! -f "$ZNO_TRAIN" || ! -f "$ZNO_DEV" ]]; then
  echo "[qa] fetching ukr_qa_train.jsonl / ukr_qa_dev.jsonl from llms-limited-resources2026 …"
  TMP_CLONE="$(mktemp -d)"
  git clone --depth 1 https://github.com/TUM-NLP/llms-limited-resources2026.git "$TMP_CLONE"
  mkdir -p "$QA_DIR"
  cp "$TMP_CLONE/Ukrainian/QA/ukr_qa_train.jsonl" "$ZNO_TRAIN"
  cp "$TMP_CLONE/Ukrainian/QA/ukr_qa_dev.jsonl"   "$ZNO_DEV"
  rm -rf "$TMP_CLONE"
fi

echo "========== 1/3 QUA-RC + Belebele RC =========="
python3 scripts/qa/convert_rc.py

echo
echo "========== 2/3 ZNO QA (DUMY CoT + 2x augment + RC merge) =========="
python3 scripts/qa/build_cot.py --cache-dir data/.hf_cache

echo
echo "========== 3/3 Merge ZNO (+ MMLU) -> train_merged.jsonl =========="
python3 scripts/qa/merge_qa_train.py

echo
echo "Done. See train_data/QA/ukr_zno_qa_train.jsonl and train_data/QA/train_merged.jsonl"
