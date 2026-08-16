#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${SESSION:-stage2_mr_train}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash $ROOT/scripts/run_stage2_ukr_mix_cs_mr_train.sh"
echo "Started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $ROOT/outputs/stage2_ukr_mix_cs_mr_train.log"
