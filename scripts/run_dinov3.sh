#!/bin/bash
# DINOv3 pretrained on external data, fine-tuned, then ensembled with the ResNet-34s.
#
#   ./scripts/run_dinov3.sh [finetune|lr|table]     ~5 h 15, 44 min an epoch
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

EXT=external-data/external145
EPOCHS_PRE=${EPOCHS_PRE:-6}
PRE=checkpoints/pre145_dinov3.pt
FT=checkpoints/ft145_dinov3.pt

pretrain() {
  [ -d "$EXT" ] || { echo "$EXT missing" >&2; exit 1; }
  echo "=== 1: pretrain DPT/ViT-S on $EXT, $EPOCHS_PRE epochs (~44 min each) ==="
  uv run python tools/train.py --arch dpt --external "$EXT" --epochs "$EPOCHS_PRE" \
      --batch-size 8 --num-workers 2 --out "$PRE"
}

finetune() {
  [ -f "$PRE" ] || { echo "$PRE missing — run step 1 first" >&2; exit 1; }
  # One rate only: on ResNet-34, 3e-4 and 1e-4 gave 0.7519 and 0.7522 — noise.
  echo "=== 2: fine-tune on the 640 challenge slices, ~40 min ==="
  uv run python tools/train.py --arch dpt --init "$PRE" --epochs 40 --batch-size 8 \
      --lr 1e-4 --out "$FT"
}

encoder_lr() {
  # After 29 000 CT slices the ViT has no ImageNet pretraining left to protect,
  # so a higher encoder LR may help. Measured: +0.005, and nothing in the ensemble.
  [ -f "$PRE" ] || { echo "$PRE missing" >&2; exit 1; }
  for s in 0.3 1.0; do
    echo "=== fine-tune DINOv3, encoder-lr-scale $s, ~40 min ==="
    uv run python tools/train.py --arch dpt --init "$PRE" --epochs 40 --batch-size 8 \
        --lr 1e-4 --encoder-lr-scale "$s" --out "checkpoints/ft145_dinov3_e${s/./}.pt"
  done
}

table() {
  echo
  echo "=== comparatif ==="
  existing=""
  for f in checkpoints/best_dinov3.pt checkpoints/ft_lr1e4.pt checkpoints/ft145_256.pt "$PRE" "$FT" \
           checkpoints/ft145_dinov3_e03.pt checkpoints/ft145_dinov3_e10.pt; do
    [ -f "$f" ] && existing="$existing $f"
  done
  uv run python tools/eval_small.py --prob --ckpt $existing

  [ -f "$FT" ] || return 0
  echo
  echo "=== uniform ensemble: DINOv3 plus the ResNet-34s of both scales ==="
  # Uniform weights and min_size 0: anything fitted on these 160 images gains
  # there and nothing on the leaderboard. One DINOv3 only; override with MEMBERS=.
  members=${MEMBERS:-$(ls "$FT" checkpoints/ft145_256.pt checkpoints/ft_lr1e4.pt checkpoints/ft_lr3e4.pt checkpoints/ft145_512.pt 2>/dev/null | paste -sd, -)}
  uv run python tools/ensemble.py --uniform --min-size 0 --ckpt "$members" \
      --submit submissions/submission_dinov3_ens.csv
}

case "${1:-all}" in
  all)      pretrain; finetune; table ;;
  pretrain) pretrain ;;
  finetune) finetune; table ;;
  lr)       encoder_lr; table ;;
  table)    table ;;
  *) echo "usage: $0 [all|pretrain|finetune|lr|table]" >&2; exit 1 ;;
esac
echo "=== DONE ==="
