#!/bin/bash
# Pretrain on the translated TotalSegmentator set, then fine-tune on the 640.
#
#   ./scripts/run_external.sh [resume|finetune|table]   ~2 h 15
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

PRETRAIN=checkpoints/pretrain_ext.pt
RESUMED=checkpoints/pretrain_ext2.pt

pretrain() {
  echo "=== step 1: external pretraining, 8 epochs, ~2 h ==="
  uv run python tools/train.py --external external-data/external --epochs 8 --batch-size 8 \
      --num-workers 2 --out "$PRETRAIN"
}

resume() {
  # lr 2e-4 continues the original cosine rather than restarting it. Separate
  # output file: train.py resets its best per process and would overwrite $PRETRAIN.
  echo "=== step 1b: resume pretraining, 5 epochs, ~1 h 15 ==="
  uv run python tools/train.py --init "$PRETRAIN" --external external-data/external --epochs 5 \
      --lr 2e-4 --batch-size 8 --num-workers 2 --out "$RESUMED"
}

base() {
  [ -f "$RESUMED" ] && echo "$RESUMED" || echo "$PRETRAIN"
}

finetune() {
  local from
  from=$(base)
  [ -f "$from" ] || { echo "$from missing — run step 1 first" >&2; exit 1; }
  echo "=== fine-tuning from $from ==="
  echo "=== step 2a: lr 3e-4, the usual rate here ==="
  uv run python tools/train.py --init "$from" --epochs 40 --batch-size 8 \
      --lr 3e-4 --out checkpoints/ft_lr3e4.pt
  echo "=== step 2b: lr 1e-4, preserves the pretraining better ==="
  uv run python tools/train.py --init "$from" --epochs 40 --batch-size 8 \
      --lr 1e-4 --out checkpoints/ft_lr1e4.pt
}

table() {
  echo
  echo "=== comparison ==="
  # --prob matters: Dice is a step function on these classes, softmax is not.
  existing=""
  for f in checkpoints/ctrl_resnet34.pt checkpoints/best_resnet34.pt checkpoints/best_dinov3.pt checkpoints/pseudo_os_resnet34.pt \
           "$PRETRAIN" "$RESUMED" checkpoints/ft_lr3e4.pt checkpoints/ft_lr1e4.pt; do
    [ -f "$f" ] && existing="$existing $f"
  done
  uv run python tools/eval_small.py --prob --ckpt $existing
}

case "${1:-all}" in
  all)      pretrain; finetune; table ;;
  pretrain) pretrain ;;
  resume)   resume; finetune; table ;;
  finetune) finetune; table ;;
  table)    table ;;
  *) echo "usage: $0 [all|pretrain|resume|finetune|table]" >&2; exit 1 ;;
esac
echo "=== DONE ==="
