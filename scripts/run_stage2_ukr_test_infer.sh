#!/usr/bin/env bash
# Ukrainian TEST inference for canonical 2-stage model (Stage1 MT+GC+SC -> Stage2 full).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
LORA_PATH="${LORA_PATH:-outputs/stage2_qa_mr_ukr_mix_cs_lora/checkpoint-1843}"
OUT_DIR="${OUT_DIR:-outputs/stage2_ukr_test_preds}"
LOG="${LOG:-outputs/stage2_ukr_test_infer.log}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MT_CHUNK_CHARS="${MT_CHUNK_CHARS:-3000}"
TASKS="${TASKS:-MT,MT_CS,QA,SC,GC,MR}"
MR_MAX_TOKENS_LOW="${MR_MAX_TOKENS_LOW:-512}"
MR_MAX_TOKENS_MEDIUM="${MR_MAX_TOKENS_MEDIUM:-2048}"
MR_BATCH_SIZE="${MR_BATCH_SIZE:-4}"
FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_FLAG="--force"
fi

mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

echo "========== Ukrainian TEST inference (2-stage) =========="
echo "  model:      $MODEL"
echo "  lora:       $LORA_PATH"
echo "  output:     $OUT_DIR"
echo "  tasks:      $TASKS"
echo "  mt chunk:   $MT_CHUNK_CHARS chars"
echo "  batch size: $BATCH_SIZE"
echo "  mr tokens:  low=$MR_MAX_TOKENS_LOW medium=$MR_MAX_TOKENS_MEDIUM batch=$MR_BATCH_SIZE"

python3 scripts/infer_ukrainian_test.py \
  --model "$MODEL" \
  --lora-path "$LORA_PATH" \
  --output-dir "$OUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --mt-chunk-chars "$MT_CHUNK_CHARS" \
  --mr-max-tokens-low "$MR_MAX_TOKENS_LOW" \
  --mr-max-tokens-medium "$MR_MAX_TOKENS_MEDIUM" \
  --mr-batch-size "$MR_BATCH_SIZE" \
  --tasks "$TASKS" \
  $FORCE_FLAG

echo "Done."
