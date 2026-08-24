"""Write .cache/annotated_selftrain.npy — the annotation matrix with confident
unknowns promoted to reliable absences.
"""

import argparse

import numpy as np
import torch

from radium.infer import load_model
from radium import data, selftrain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best_resnet34.pt")
    p.add_argument("--max-loss", type=float, default=0.0,
                   help="tolerated fraction of an organ's KNOWN presences sacrificed")
    p.add_argument("--out", default="annotated_selftrain")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model, hu = load_model(args.ckpt, device)

    images = data.train_images_hu() if hu else data.train_images()
    A = data.annotated_ids()
    n = data.N_ANNOTATED

    scores = selftrain.absence_scores(model, images[:n], device, hu=hu)
    present = selftrain.presence(data.train_labels()[:n])
    thr = selftrain.calibrate(scores, A[:n], present, args.max_loss)

    full = np.zeros((len(A), 54), dtype=scores.dtype)
    full[:n] = scores
    promoted = selftrain.promote(A, full, thr, n)

    gained = int(promoted[:n].sum() - A[:n].sum())
    unknown = int((~A[:n]).sum())
    seen = (A[:n] & present).sum(0)
    # Promoted cells cannot be checked directly; report the count and the rate.
    per_organ = (promoted[:n] & ~A[:n]).sum(0)
    blocked = [c + 1 for c in range(54) if thr[c] < 0 or per_organ[c] == 0]

    print(f"unknown      : {unknown}")
    print(f"promoted     : {gained}  ({gained/unknown:.1%} of the unknown cells)")
    print(f"supervision  : {A[:n].sum()} -> {promoted[:n].sum()} cells "
          f"(+{gained/A[:n].sum():.1%})")
    print(f"\norgans with no promotion at all (model not confident enough): "
          f"{len(blocked)}/54")
    print(f"  {blocked}")

    rare = np.argsort(seen)[:8]
    print(f"\n{'id':>4} {'seen':>5} {'threshold':>10} {'promoted':>8}")
    for c in rare:
        t = f"{thr[c]:.2e}" if thr[c] >= 0 else "none"
        print(f"{c+1:>4} {seen[c]:>5} {t:>10} {per_organ[c]:>8}")

    path = data.CACHE / f"{args.out}.npy"
    data.CACHE.mkdir(exist_ok=True)
    np.save(path, promoted)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
