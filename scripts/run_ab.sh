#!/bin/bash
# Three arms: A is a controlled baseline, B adds pseudo-labels, C adds resampling.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "=== A  baseline           640 img, ~20 min ==="
uv run python tools/train.py --epochs 60 --out checkpoints/ctrl_resnet34.pt

echo "=== B  + pseudo ens4     1840 img, ~56 min ==="
uv run python tools/train.py --pseudo ens4 --tau-cell 0.3 --epochs 60 \
    --out checkpoints/pseudo_resnet34.pt

echo "=== C  + oversample 0.5  1840 img, ~56 min ==="
uv run python tools/train.py --pseudo ens4 --tau-cell 0.3 --oversample 0.5 --epochs 60 \
    --out checkpoints/pseudo_os_resnet34.pt

echo "=== DONE ==="
