#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${SESSION:-two_stage_monitor}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash $ROOT/scripts/monitor_two_stage.sh"
echo "Hourly monitor started: tmux attach -t $SESSION"
echo "Reports: $ROOT/outputs/two_stage_progress.log"
