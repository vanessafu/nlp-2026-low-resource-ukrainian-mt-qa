#!/usr/bin/env bash
# Clone WMT2026 shared-task dev data (Ukrainian track only).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/llms-limited-resources2026"
REPO_URL="${DEV_DATA_REPO:-https://github.com/TUM-NLP/llms-limited-resources2026.git}"

if [[ -d "$DEST/Ukrainian/MT" ]]; then
  echo "Dev data already present: $DEST/Ukrainian"
  exit 0
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Cloning $REPO_URL (sparse: Ukrainian/)..."
git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" "$TMP/repo"
git -C "$TMP/repo" sparse-checkout set Ukrainian

mkdir -p "$DEST"
rm -rf "$DEST/Ukrainian"
mv "$TMP/repo/Ukrainian" "$DEST/Ukrainian"
rm -rf "$TMP/repo"

echo "Done: $DEST/Ukrainian"
