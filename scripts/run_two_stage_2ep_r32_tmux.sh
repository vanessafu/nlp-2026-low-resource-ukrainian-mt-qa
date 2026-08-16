#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${SESSION:-two_stage_r32_2ep}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash $ROOT/scripts/train_two_stage_2ep_r32.sh"
echo "Started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $ROOT/outputs/two_stage_r32_2ep_train.log"
