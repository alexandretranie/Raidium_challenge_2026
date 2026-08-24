#!/bin/bash
# Two corrections measured separately: pixel spacing (arm S), then resolution (arm R).
#
#   ./scripts/run_scale512.sh [S|R|table]           ~3 h 05
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

EXT=external-data/external145
S_PRE=checkpoints/pre145_256.pt
S_FT=checkpoints/ft145_256.pt
R_FT=checkpoints/ft145_512.pt

arm_S() {
  [ -d "$EXT" ] || { echo "$EXT missing — run tools/make_external.py --spacing 1.45" >&2; exit 1; }
  echo "=== S1: pretrain 256 on $EXT (1.45 mm/px), 8 epochs, ~2 h 15 ==="
  uv run python tools/train.py --external "$EXT" --epochs 8 --batch-size 8 \
      --num-workers 2 --out "$S_PRE"
  echo "=== S2: fine-tune 256 on the 640 challenge slices, ~15 min ==="
  uv run python tools/train.py --init "$S_PRE" --epochs 40 --batch-size 8 \
      --lr 1e-4 --out "$S_FT"
}

arm_R() {
  [ -f "$S_PRE" ] || { echo "$S_PRE missing — run arm S first" >&2; exit 1; }
  echo "=== R: fine-tune 512 from $S_PRE, ~35 min ==="
  uv run python tools/train.py --init "$S_PRE" --size 512 --epochs 40 --batch-size 8 \
      --lr 1e-4 --out "$R_FT"
}

table() {
  echo
  echo "=== comparison ==="
  existing=""
  for f in checkpoints/pretrain_ext2.pt checkpoints/ft_lr1e4.pt "$S_PRE" "$S_FT" "$R_FT"; do
    [ -f "$f" ] && existing="$existing $f"
  done
  uv run python tools/eval_small.py --prob --ckpt $existing

  echo
  echo "=== the precise question: does the right adrenal (id 2) leave zero? ==="
  # On its own training data: at 256 the model cannot even memorise them.
  for f in "$S_FT" "$R_FT"; do
    [ -f "$f" ] || continue
    uv run python - "$f" <<'PY'
import sys, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from radium.dataset import to_channels
from predict import load_model
E = np.load("external-data/external145/images.npy", mmap_mode="r")
L = np.load("external-data/external145/labels.npy", mmap_mode="r")
model, hu = load_model(sys.argv[1], "cpu")
idx = [i for i in range(0, len(L), 23) if (np.asarray(L[i]) == 2).any()][:120]
d, p = [], []
with torch.no_grad():
    for i in idx:
        x = torch.from_numpy(to_channels(np.asarray(E[i])[None], hu))
        pr = model((x - 0.5) / 0.25).softmax(1).numpy()[0]
        t = np.asarray(L[i]) == 2
        s = (pr.argmax(0) == 2).sum() + t.sum()
        d.append(2 * ((pr.argmax(0) == 2) & t).sum() / s)
        p.append(pr[2][t].mean())
print(f"  {sys.argv[1]:<16} on training data: dice {np.mean(d):.3f}  prob {np.mean(p):.3f}  (n={len(d)})")
PY
  done
  echo "  for reference: at 256 on external/, dice 0.000 and prob 0.000"
}

case "${1:-all}" in
  all)   arm_S; arm_R; table ;;
  S)     arm_S; table ;;
  R)     arm_R; table ;;
  table) table ;;
  *) echo "usage: $0 [all|S|R|table]" >&2; exit 1 ;;
esac
echo "=== DONE ==="
