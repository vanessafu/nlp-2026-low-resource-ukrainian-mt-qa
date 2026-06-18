#!/usr/bin/env bash
# Hourly progress monitor for two-stage LoRA training.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/outputs/two_stage_train.log"
REPORT="$ROOT/outputs/two_stage_progress.log"
INTERVAL="${INTERVAL:-3600}"

log_report() {
  {
    echo "========== $(date -u '+%Y-%m-%d %H:%M:%S UTC') =========="
    if tmux has-session -t two_stage_lora 2>/dev/null; then
      echo "tmux: two_stage_lora RUNNING"
    else
      echo "tmux: two_stage_lora NOT RUNNING"
    fi
    if pgrep -af "finetune_stage|pick_best|train_two_stage" | grep -v monitor | head -3; then
      true
    else
      echo "process: no training process"
    fi
    echo "--- pipeline tail ---"
    tail -8 "$LOG" 2>/dev/null || echo "(no log)"
    echo "--- stage1 checkpoints ---"
    ls -d "$ROOT/outputs/stage1_mt_gc_sc_lora"/checkpoint-* 2>/dev/null | tail -3 || echo "(none yet)"
    if [[ -f "$ROOT/outputs/stage1_best_checkpoint.json" ]]; then
      echo "--- best stage1 ---"
      python3 -c "import json; d=json.load(open('$ROOT/outputs/stage1_best_checkpoint.json')); print(d.get('best_checkpoint'), 'score=', d.get('best_score'))"
    fi
    if [[ -f "$ROOT/outputs/stage2_final_dev_eval/summary.json" ]]; then
      echo "--- final dev eval ---"
      cat "$ROOT/outputs/stage2_final_dev_eval/summary.json"
    fi
    echo ""
  } | tee -a "$REPORT"
}

log_report
while tmux has-session -t two_stage_lora 2>/dev/null || pgrep -af "finetune_stage|train_two_stage" >/dev/null 2>&1; do
  sleep "$INTERVAL"
  log_report
done
log_report
echo "Monitor finished at $(date -u)" | tee -a "$REPORT"
