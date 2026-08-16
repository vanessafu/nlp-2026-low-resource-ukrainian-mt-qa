#!/usr/bin/env bash
# Stage2 (1 epoch) with train.cs-uk.jsonl.gz as MT_CS, then TEST inference + submission folder.
# Everything else matches the old baseline Stage2 recipe (from stage1 checkpoint-3931).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
STAGE1_LORA="${STAGE1_LORA:-outputs/stage1_mt_gc_sc_lora/checkpoint-3931}"
STAGE2_DIR="${STAGE2_DIR:-outputs/stage2_csuk_new_lora}"
PRED_DIR="${PRED_DIR:-outputs/stage2_csuk_new_test_preds}"
SUBMIT_DIR="${SUBMIT_DIR:-Ukrainian_csuk_new}"
ZIP_PATH="${ZIP_PATH:-Ukrainian_csuk_new_submission.zip}"
LOG="${LOG:-outputs/stage2_csuk_new.log}"
EPOCHS="${EPOCHS:-1}"

MT_PATH="${MT_PATH:-train_data/data/pseudo_docs/train_pseudo_en_uk.jsonl}"
GC_PATH="${GC_PATH:-train_data/GC/ua_gec_train_single_error.jsonl}"
MT_CS_PATH="${MT_CS_PATH:-train.cs-uk.jsonl.gz}"
N_MT="${N_MT:-5000}"
N_GC="${N_GC:-5000}"
# Use full train.cs-uk.jsonl.gz (16946 rows); override with N_MT_CS if needed.
N_MT_CS="${N_MT_CS:-16946}"

BATCH_SIZE="${BATCH_SIZE:-16}"
MT_CHUNK_CHARS="${MT_CHUNK_CHARS:-3000}"
MR_MAX_TOKENS_LOW="${MR_MAX_TOKENS_LOW:-512}"
MR_MAX_TOKENS_MEDIUM="${MR_MAX_TOKENS_MEDIUM:-2048}"
MR_BATCH_SIZE="${MR_BATCH_SIZE:-4}"

mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

echo "========== Stage 2 UKR+mix+cs (epochs=$EPOCHS) from $STAGE1_LORA =========="
echo "  MT_cs path:  $MT_CS_PATH (n=$N_MT_CS, full file)"
echo "  MT_en sample: $N_MT from $MT_PATH"
echo "  GC sample:   $N_GC from $GC_PATH"
python3 scripts/finetune_stage2_qa_mr_ukr_mix.py \
  --model "$MODEL" \
  --stage1-lora "$STAGE1_LORA" \
  --output-dir "$STAGE2_DIR" \
  --epochs "$EPOCHS" \
  --mt "$MT_PATH" \
  --mt-format parallel \
  --gc "$GC_PATH" \
  --n-mt "$N_MT" \
  --n-gc "$N_GC" \
  --mt-cs "$MT_CS_PATH" \
  --n-mt-cs "$N_MT_CS"

# Prefer final adapter at STAGE2_DIR if present; else last checkpoint-*
if [[ -f "$STAGE2_DIR/adapter_model.safetensors" ]]; then
  LORA_PATH="$STAGE2_DIR"
else
  LORA_PATH=$(ls -d "$STAGE2_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
echo "========== TEST inference with $LORA_PATH =========="
python3 scripts/infer_ukrainian_test.py \
  --model "$MODEL" \
  --lora-path "$LORA_PATH" \
  --output-dir "$PRED_DIR" \
  --batch-size "$BATCH_SIZE" \
  --mt-chunk-chars "$MT_CHUNK_CHARS" \
  --mr-max-tokens-low "$MR_MAX_TOKENS_LOW" \
  --mr-max-tokens-medium "$MR_MAX_TOKENS_MEDIUM" \
  --mr-batch-size "$MR_BATCH_SIZE" \
  --tasks MT,MT_CS,QA,SC,GC,MR \
  --force

echo "========== Export submission folder =========="
python3 scripts/export_ukrainian_submission.py \
  --pred-dir "$PRED_DIR" \
  --out-dir "$SUBMIT_DIR" \
  --zip-path "$ZIP_PATH"

echo "Done."
echo "  Stage2:     $STAGE2_DIR"
echo "  LoRA used:  $LORA_PATH"
echo "  Preds:      $PRED_DIR"
echo "  Submit dir: $SUBMIT_DIR"
echo "  Zip:        $ZIP_PATH"
