#!/bin/bash
# Small organs: pixel budget or number of examples? Arms A 256, B 512, C 384,
# D oversample, E both. Read `tiny9` from eval_small.py, not the overall Dice.
#
#   ./scripts/run_small_organs.sh [A B ...|table]   ~2 h, BATCH=2 if it swaps
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-4}
COMMON="--epochs $EPOCHS --batch-size $BATCH"

CKPTS="checkpoints/small_a256.pt checkpoints/small_b512.pt checkpoints/small_d_os.pt checkpoints/small_e512os.pt small_c384.pt"

arm_A() {
  echo "=== A  control 256                        ~16 min ==="
  uv run python tools/train.py $COMMON --size 256 --out checkpoints/small_a256.pt
}

arm_B() {
  echo "=== B  resolution 512                     ~35 min ==="
  uv run python tools/train.py $COMMON --size 512 --out checkpoints/small_b512.pt
}

arm_C() {
  echo "=== C  resolution 384                     ~22 min ==="
  uv run python tools/train.py $COMMON --size 384 --out checkpoints/small_c384.pt
}

arm_D() {
  echo "=== D  oversample 0.5 at 256              ~16 min ==="
  uv run python tools/train.py $COMMON --size 256 --oversample 0.5 --out checkpoints/small_d_os.pt
}

arm_E() {
  echo "=== E  512 + oversample 0.5               ~35 min ==="
  uv run python tools/train.py $COMMON --size 512 --oversample 0.5 --out checkpoints/small_e512os.pt
}

table() {
  echo
  echo "=== comparison ==="
  # The checkpoints on disk are the reference for reading the arms.
  existing=""
  for f in checkpoints/best_resnet34.pt checkpoints/best_dinov3.pt $CKPTS; do
    [ -f "$f" ] && existing="$existing $f"
  done
  uv run python tools/eval_small.py --ckpt $existing
}

if [ "$1" = "table" ]; then
  table
  exit 0
fi

for arm in ${*:-A B D E C}; do
  case "$arm" in
    A|B|C|D|E) "arm_$arm" ;;
    *) echo "unknown arm: $arm (expected A B C D E, or 'table')" >&2; exit 1 ;;
  esac
done

table
echo "=== DONE ==="
