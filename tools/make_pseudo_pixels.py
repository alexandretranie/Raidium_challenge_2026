"""Run a teacher over all 2000 training slices and store its probabilities."""

import argparse
import json
import time

import numpy as np
import torch

from radium import data, pseudo, twins
from radium.teacher import TTA_VIEWS, Teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best_dinov3.pt",
                    help="one checkpoint, or 'a.pt:0.3,b.pt:0.7' for an ensemble")
    ap.add_argument("--run", required=True, help="folder name under pseudo-labels/")
    ap.add_argument("--tta", action="store_true", help="average over 5 warped views")
    ap.add_argument("--limit", type=int, default=None, help="first N slices, smoke test")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    teacher = Teacher(args.ckpt, device, views=TTA_VIEWS if args.tta else None)

    png = data.train_images() if "png" in teacher.needs else None
    hu = data.train_images_hu() if "hu" in teacher.needs else None
    n = args.limit or data.N_TRAIN
    idx = np.arange(n)

    print(f"teacher : {teacher.describe()}")
    print(f"        : {len(teacher.views)} view(s), {n} slices, {device}")
    t0 = time.time()
    step = max(1, n // 10)
    tick = lambda k, tot: (k % step < 8) and print(f"\r  {k}/{tot}", end="", flush=True)
    labels, conf, peak = teacher.run(png, hu, idx, progress=tick)
    print(f"\rinference {time.time()-t0:.0f}s{' ' * 20}")

    meta = {"ckpt": args.ckpt, "views": [list(v) for v in teacher.views], "n": n,
            "written": time.strftime("%Y-%m-%d %H:%M")}

    # Score the teacher before anyone trains on it.
    if n >= data.N_ANNOTATED:
        pairs = twins.annotated_pairs()
        A, Y = data.annotated_ids(), data.train_labels()
        pred = {i: labels[i] for i in {k for p in pairs for k in p}}
        meta["bench"] = twins.score(pred, twins.bench_cells(pairs, A), Y)
        b = meta["bench"]
        print(f"twin bench: dice {b['dice']:.4f}  found {b['found']:.1%}  "
              f"correctly-empty {b['correctly_empty']:.1%}  "
              f"(ceiling {twins.ceiling(pairs, Y, A):.4f})")

    print(f"wrote {pseudo.write(args.run, labels, conf, peak, meta)}")

    if n == data.N_TRAIN:
        tgt, ann, w = pseudo.merge(args.run)
        s = pseudo.verify(tgt, w, annotated=ann)
        print(f"\nat the default gate "
              f"({pseudo.TAU_CELL}/{pseudo.TAU_PIX}/{pseudo.TAU_BG}):")
        print(f"  cells   {s['cells_declared']} declared + {s['cells_accepted']} accepted")
        print(f"  pixels  {s['annotated_px']/1e6:.1f}M annotated, "
              f"{s['pseudo_px']/1e6:.1f}M pseudo, {s['dropped_px']/1e6:.1f}M dropped "
              f"of {s['px_total']/1e6:.1f}M")


if __name__ == "__main__":
    main()
