#!/usr/bin/env bash
# Two-stage LoRA training + dev evaluation.
# Stage 1: MT (pseudo_docs en→uk only) + GC + SC, 5 epochs
# Stage 2: QA + MR from best stage-1 checkpoint, 5 epochs
# Final: eval all 5 dev tasks
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
EPOCHS="${EPOCHS:-5}"
STAGE1_DIR="${STAGE1_DIR:-outputs/stage1_mt_gc_sc_lora}"
STAGE2_DIR="${STAGE2_DIR:-outputs/stage2_qa_mr_lora}"
FINAL_EVAL_DIR="${FINAL_EVAL_DIR:-outputs/stage2_final_dev_eval}"
LOG="${LOG:-outputs/two_stage_train.log}"

mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

echo "========== Stage 1: MT + GC + SC (${EPOCHS} epochs) =========="
python3 scripts/finetune_stage1_mt_gc_sc.py \
  --model "$MODEL" \
  --output-dir "$STAGE1_DIR" \
  --epochs "$EPOCHS"

echo "========== Pick best Stage-1 checkpoint (dev MT+GC+SC) =========="
python3 scripts/pick_best_stage1_checkpoint.py \
  --stage1-dir "$STAGE1_DIR" \
  --model "$MODEL"

BEST_CKPT=$(python3 -c "import json; print(json.load(open('outputs/stage1_best_checkpoint.json'))['best_checkpoint'])")
echo "Best stage-1 checkpoint: $BEST_CKPT"

echo "========== Stage 2: QA + MR (${EPOCHS} epochs) =========="
python3 scripts/finetune_stage2_qa_mr.py \
  --model "$MODEL" \
  --stage1-lora "$BEST_CKPT" \
  --output-dir "$STAGE2_DIR" \
  --epochs "$EPOCHS"

echo "========== Final dev eval (all 5 tasks) =========="
python3 eval_qwen35_ukrainian_zeroshot.py \
  --model "$MODEL" \
  --lora-path "$STAGE2_DIR" \
  --output-dir "$FINAL_EVAL_DIR" \
  --force

echo "Done."
echo "  Stage1: $STAGE1_DIR"
echo "  Best:   $BEST_CKPT"
echo "  Stage2: $STAGE2_DIR"
echo "  Eval:   $FINAL_EVAL_DIR/summary.json"
