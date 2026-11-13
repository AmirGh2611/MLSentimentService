#!/usr/bin/env bash
set -euo pipefail

echo "==> Training sentiment model..."
python -m sentiment.train
echo "==> Done. Artifact in models/"