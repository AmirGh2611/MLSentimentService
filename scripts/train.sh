#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${SENTIMENT_MODEL_PATH:-models/mlsentimentservice.joblib}"
echo "==> Training sentiment model"
echo "==> Output: $MODEL_PATH"

python -m mlsentimentservice.train

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model artifact was not created at $MODEL_PATH" >&2
  exit 1
fi

SIZE=$(du -h "$MODEL_PATH" | cut -f1)
echo "==> Done. Model size: $SIZE"