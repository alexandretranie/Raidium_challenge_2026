"""Score a teacher as a pseudo-labeller, on cells whose answer is known."""

import argparse

import numpy as np
import torch

from radium import data, twins
from radium.teacher import TTA_VIEWS, Teacher

GATES = (0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)
PIXEL_GATES = (0.0, 0.5, 0.7, 0.9, 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best_dinov3.pt",
                   help="one checkpoint, or 'a.pt:0.3,b.pt:0.7' for an ensemble")
    ap.add_argument("--tta", action="store_true", help="average over 5 warped views")
    args = ap.parse_args()

    Y, A = data.train_labels(), data.annotated_ids()
    pairs = twins.annotated_pairs()
    cells = twins.bench_cells(pairs, A)
    control = [(t, s, c) for i, j in pairs for t, s in ((i, j), (j, i))
               for c in np.where(A[s] & A[t])[0]]
    idx = np.array(sorted({i for p in pairs for i in p}))

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    teacher = Teacher(args.ckpt, device, views=TTA_VIEWS if args.tta else None)

    print(f"teacher : {teacher.describe()}  ({len(teacher.views)} view(s))")
    print(f"bench   : {len(pairs)} twin pairs, {len(idx)} slices, {len(cells)} unknown cells")
    print(f"ceiling : {twins.ceiling(pairs, Y, A):.4f}  (a perfect labeller)\n")

    png = data.train_images() if "png" in teacher.needs else None
    hu = data.train_images_hu() if "hu" in teacher.needs else None

    probs = {}
    for k in range(0, len(idx), 8):
        sel = idx[k : k + 8]
        p = teacher.probs(png=None if png is None else png[sel],
                          hu=None if hu is None else hu[sel]).cpu().numpy()
        probs.update(zip(sel, p))
    pred = {i: p.argmax(0).astype(np.uint8) for i, p in probs.items()}

    for tag, sel in (("DECLARED (control)", control), ("UNKNOWN  (the test)", cells)):
        s = twins.score(pred, sel, Y)
        print(f"{tag}  n={s['n_cells']:>4}  "
              f"presences={s['n_present']:>3} dice={s['dice']:.4f} "
              f"found={s['found']:.1%}  |  absences={s['n_absent']:>3} "
              f"correctly-empty={s['correctly_empty']:.1%}")

    peak = np.array([probs[t][c + 1].max() for t, _, c in cells])
    present = np.array([(Y[s] == c + 1).any() for _, s, c in cells])
    dice = np.array([twins._dice(pred[t] == c + 1, Y[s] == c + 1) for t, s, c in cells])

    print(f"\n[cell gate] peak probability of the class anywhere in the slice")
    print(f"{'thr':>6} {'written':>8} {'precision':>10} {'recall':>8} {'dice':>8}")
    for t in GATES:
        w = peak > t
        if not w.any():
            continue
        d = dice[w & present]
        print(f"{t:>6.2f} {w.sum():>8} {present[w].mean():>10.1%} "
              f"{w[present].mean():>8.1%} "
              f"{np.nanmean(d) if len(d) else np.nan:>8.4f}")

    print(f"\n[pixel gate] purity of the pixels a run would write")
    for t in PIXEL_GATES:
        tp = fp = 0
        for tgt, src, c in cells:
            sel = (pred[tgt] == c + 1) & (probs[tgt].max(0) > t)
            tp += int((sel & (Y[src] == c + 1)).sum())
            fp += int((sel & (Y[src] != c + 1)).sum())
        print(f"  conf > {t:.2f}: {tp + fp:>7} px, purity {tp / max(tp + fp, 1):.1%}")


if __name__ == "__main__":
    main()
