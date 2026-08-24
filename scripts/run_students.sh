#!/bin/bash
# Retrain each ensemble member with pseudo-labels plus rare-organ resampling.
#   ./scripts/run_students.sh vits | vitb | vitb2 | all   2.5 to 9.5 h
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

COMMON="--pseudo ens4 --tau-cell 0.3 --oversample 0.5 --epochs 60"

vits () {
  echo "=== DINOv3 ViT-S + DPT   ~2.5 h ==="
  uv run python tools/train.py --arch dpt $COMMON --out checkpoints/st_vits.pt
}
vitb () {
  echo "=== DINOv3 ViT-B + DPT   ~3.5 h ==="
  uv run python tools/train.py --arch dpt \
    --encoder tu-vit_base_patch16_dinov3.lvd1689m $COMMON --out checkpoints/st_vitb.pt
}
vitb2 () {
  echo "=== DINOv3 ViT-B + DPT, second seed   ~3.5 h ==="
  uv run python tools/train.py --arch dpt \
    --encoder tu-vit_base_patch16_dinov3.lvd1689m $COMMON --out checkpoints/st_vitb2.pt
}

case "${1:-}" in
  vits)  vits ;;
  vitb)  vitb ;;
  vitb2) vitb2 ;;
  all)   vitb; vits; vitb2 ;;   # strongest first, so a stop still leaves a pair
  *)     echo "usage: $0 {vits|vitb|vitb2|all}"; exit 1 ;;
esac
echo "=== DONE ==="
