#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/outputs/translate_en_uk.log"
CREDS="${GOOGLE_APPLICATION_CREDENTIALS:-$ROOT/config/google_cloud_credentials.json}"
mkdir -p "$ROOT/outputs" "$ROOT/config"

if [[ ! -f "$CREDS" ]]; then
  echo "Missing credentials: $CREDS" >&2
  echo "Copy indigo-medium-498310-f4-0501fd02164e.json to that path, e.g.:" >&2
  echo "  cp /path/to/indigo-medium-498310-f4-0501fd02164e.json $CREDS" >&2
  exit 1
fi

export GOOGLE_APPLICATION_CREDENTIALS="$CREDS"

echo "Starting translation at $(date)" | tee "$LOG"
echo "Using Google Cloud Translation API: $CREDS" | tee -a "$LOG"
python3 "$ROOT/scripts/translate_train_en_uk.py" \
  --backend cloud \
  --credentials "$CREDS" \
  --limit-qa 20000 \
  --batch-size 50 \
  --sleep 0.05 \
  2>&1 | tee -a "$LOG"
echo "Finished at $(date)" | tee -a "$LOG"
