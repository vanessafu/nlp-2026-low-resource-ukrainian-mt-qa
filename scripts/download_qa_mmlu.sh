#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p train_data/QA outputs

# hf-mirror.com does not host these dataset files reliably; use official Hub.
unset HF_ENDPOINT

echo "Downloading Ukrainian MMLU..."
hf download INSAIT-Institute/mmlu_ukr \
  --repo-type dataset \
  --local-dir train_data/QA/mmlu_ukr

echo "Downloading English MMLU..."
hf download cais/mmlu \
  --repo-type dataset \
  --local-dir train_data/QA/mmlu_en

echo "Done."
echo "  train_data/QA/mmlu_ukr"
echo "  train_data/QA/mmlu_en"
