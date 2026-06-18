#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/service-account.json" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/config/google_cloud_credentials.json"
mkdir -p "$ROOT/config"
cp "$1" "$DEST"
chmod 600 "$DEST"
echo "Installed credentials to $DEST"
