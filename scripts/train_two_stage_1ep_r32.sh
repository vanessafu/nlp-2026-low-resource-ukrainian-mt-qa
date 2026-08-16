#!/usr/bin/env bash
# Two-stage LoRA (r=32, max_seq_len=3072): 1 epoch each, then full dev eval.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
EPOCHS=1
STAGE1_DIR="${STAGE1_DIR:-outputs/stage1_mt_gc_sc_r32_1ep_lora}"
STAGE2_DIR="${STAGE2_DIR:-outputs/stage2_qa_mr_ukr_mix_r32_1ep_lora}"
FINAL_EVAL_DIR="${FINAL_EVAL_DIR:-outputs/stage2_r32_1ep_dev_eval}"
STAGE1_BEST_FILE="${STAGE1_BEST_FILE:-outputs/stage1_r32_1ep_best_checkpoint.json}"
STAGE2_BEST_FILE="${STAGE2_BEST_FILE:-outputs/stage2_r32_1ep_best_checkpoint.json}"
LOG="${LOG:-outputs/two_stage_r32_1ep_train.log}"

mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

echo "========== Config =========="
echo "  model:      $MODEL"
echo "  epochs:     stage1=$EPOCHS stage2=$EPOCHS"
echo "  lora:       r=32 alpha=64 (finetune defaults)"
echo "  max_seq:    3072 (finetune defaults)"
echo "  stage1_dir: $STAGE1_DIR"
echo "  stage2_dir: $STAGE2_DIR"
echo "  eval_dir:   $FINAL_EVAL_DIR"

echo "========== Stage 1: MT + GC + SC (${EPOCHS} epoch) =========="
python3 scripts/finetune_stage1_mt_gc_sc.py \
  --model "$MODEL" \
  --output-dir "$STAGE1_DIR" \
  --epochs "$EPOCHS"

echo "========== Pick best Stage-1 checkpoint (dev MT+GC+SC) =========="
python3 scripts/pick_best_stage1_checkpoint.py \
  --stage1-dir "$STAGE1_DIR" \
  --model "$MODEL" \
  --out-file "$STAGE1_BEST_FILE"

BEST_CKPT=$(python3 -c "import json; print(json.load(open('$STAGE1_BEST_FILE'))['best_checkpoint'])")
echo "Best stage-1 checkpoint: $BEST_CKPT"

echo "========== Stage 2: UKR+mix+cs (QA + MR + MT + GC, ${EPOCHS} epoch) =========="
STAGE1_LORA="$BEST_CKPT" \
  STAGE2_DIR="$STAGE2_DIR" \
  EVAL_DIR="$FINAL_EVAL_DIR" \
  EPOCHS="$EPOCHS" \
  BEST_CKPT_FILE="$STAGE2_BEST_FILE" \
  LOG="${LOG%.log}_stage2.log" \
  bash scripts/run_stage2_ukr_mix_cs.sh

echo "========== Compare dev eval summaries =========="
python3 scripts/compare_dev_eval_summaries.py \
  --labels "zeroshot" "2stage_5+1ep_r16" "2stage_1+1ep_r32" \
  --summaries \
    outputs/qwen3_5_2b_ukrainian_zeroshot/summary.json \
    outputs/stage2_ukr_mix_cs_dev_eval/summary.json \
    "$FINAL_EVAL_DIR/summary.json" \
  --out-file outputs/stage2_r32_1ep_compare.json

echo "Done."
echo "  Stage1:  $STAGE1_DIR"
echo "  Stage2:  $STAGE2_DIR"
echo "  Eval:    $FINAL_EVAL_DIR/summary.json"
echo "  Compare: outputs/stage2_r32_1ep_compare.json"
