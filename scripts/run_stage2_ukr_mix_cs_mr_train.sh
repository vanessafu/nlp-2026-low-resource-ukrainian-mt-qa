#!/usr/bin/env bash
# Old baseline Stage-2 re-run: same UKR+mix+cs recipe, MR from pre-formatted mr_train.jsonl.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
STAGE1_LORA="${STAGE1_LORA:-outputs/stage1_mt_gc_sc_lora/checkpoint-3931}"
STAGE2_DIR="${STAGE2_DIR:-outputs/stage2_ukr_mix_cs_mr_train_lora}"
EVAL_DIR="${EVAL_DIR:-outputs/stage2_ukr_mix_cs_mr_train_dev_eval}"
LOG="${LOG:-outputs/stage2_ukr_mix_cs_mr_train.log}"
EPOCHS="${EPOCHS:-1}"
BEST_CKPT_FILE="${BEST_CKPT_FILE:-outputs/stage2_mr_train_best_checkpoint.json}"

MR_TRAIN="${MR_TRAIN:-train_data/MR/mr_train.jsonl}"
MT_PATH="${MT_PATH:-train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl}"
GC_PATH="${GC_PATH:-train_data/GC/ua_gec_train_single_error.jsonl}"
MT_CS_PATH="${MT_CS_PATH:-version 1/train.jsonl.gz}"
N_MT="${N_MT:-5000}"
N_GC="${N_GC:-5000}"
N_MT_CS="${N_MT_CS:-10000}"

mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

echo "========== Stage 2 UKR+mix+cs (epochs=$EPOCHS) from $STAGE1_LORA =========="
echo "  MR train:     $MR_TRAIN"
echo "  MT_en sample: $N_MT from $MT_PATH"
echo "  MT_cs first:  $N_MT_CS from $MT_CS_PATH"
python3 scripts/finetune_stage2_qa_mr_ukr_mix.py \
  --model "$MODEL" \
  --stage1-lora "$STAGE1_LORA" \
  --output-dir "$STAGE2_DIR" \
  --epochs "$EPOCHS" \
  --mr-train "$MR_TRAIN" \
  --mt "$MT_PATH" \
  --mt-format parallel \
  --gc "$GC_PATH" \
  --n-mt "$N_MT" \
  --n-gc "$N_GC" \
  --mt-cs "$MT_CS_PATH" \
  --n-mt-cs "$N_MT_CS"

echo "========== Pick best Stage-2 checkpoint (dev MT+MT_CS+QA+SC+GC+MR) =========="
python3 scripts/pick_best_stage2_checkpoint.py \
  --stage2-dir "$STAGE2_DIR" \
  --model "$MODEL" \
  --out-file "$BEST_CKPT_FILE"

BEST_CKPT=$(python3 -c "import json; print(json.load(open('$BEST_CKPT_FILE'))['best_checkpoint'])")
echo "Best stage-2 checkpoint: $BEST_CKPT"

echo "========== Final dev eval on best checkpoint =========="
python3 eval_qwen35_ukrainian_zeroshot.py \
  --model "$MODEL" \
  --lora-path "$BEST_CKPT" \
  --output-dir "$EVAL_DIR" \
  --force \
  --tasks MT,MT_CS,QA,SC,GC,MR

echo "Done."
echo "  Stage2 UKR+mix+cs (mr_train): $STAGE2_DIR"
echo "  Best checkpoint:              $BEST_CKPT"
echo "  Eval:                         $EVAL_DIR/summary.json"
