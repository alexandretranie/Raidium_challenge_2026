"""Per-class decision thresholds, cross-validated."""

import argparse
import json
import warnings

import numpy as np
import torch

from radium import data
from radium.dataset import to_channels
from radium.infer import load_model

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# 1.1 is unreachable, so it means pure argmax; first, so ties keep the argmax.
GRID = [1.1, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02]


def _dice(pred, truth):
    s = pred.sum() + truth.sum()
    return np.nan if s == 0 else 2.0 * (pred & truth).sum() / s


def probabilities(ckpt, device, batch_size=8):
    """Yield (index, probs (55, H, W) float32) over the validation split."""
    model, hu = load_model(ckpt, device)
    _, val_idx = data.splits()
    X = (data.train_images_hu() if hu else data.train_images())[val_idx]
    with torch.no_grad():
        for k in range(0, len(val_idx), batch_size):
            x = torch.from_numpy(to_channels(X[k:k + batch_size], hu))
            prob = model((x.to(device) - 0.5) / 0.25).softmax(1).cpu().numpy()
            for b in range(len(prob)):
                yield k + b, prob[b]


def separable_stats(ckpt, device, gt, ann):
    """(n, 54, len(GRID)) Dice if class c alone were lowered to threshold j."""
    n = len(gt)
    out = np.full((n, 54, len(GRID)), np.nan)
    for i, prob in probabilities(ckpt, device):
        base = prob.argmax(0)
        bg = base == 0
        for c in range(1, 55):
            if not ann[i, c - 1]:
                continue           # unannotated: an empty truth means nothing
            truth = gt[i] == c
            hit = base == c
            for j, t in enumerate(GRID):
                out[i, c - 1, j] = _dice(hit | (bg & (prob[c] > t)), truth)
    return out


def choose(stats, rows):
    """(54,) per-class threshold, fitted on the images in `rows`."""
    m = np.nanmean(stats[rows], axis=0)                     # (54, n_grid)
    m = np.where(np.isnan(m), -1.0, m)
    return np.array([GRID[j] for j in m.argmax(axis=1)])


def apply_rule(prob, thr):
    """Argmax, then on background pixels the most probable class clearing its threshold."""
    pred = prob.argmax(0).astype(np.uint8)
    over = prob[1:] > thr[:, None, None]
    if not over.any():
        return pred
    cand = np.where(over, prob[1:], -1.0)
    best = cand.argmax(0)
    take = (pred == 0) & (cand.max(0) > 0)
    pred[take] = (best[take] + 1).astype(np.uint8)
    return pred


def score_joint(ckpt, device, gt, ann, thr_per_image):
    """Exact score of the full rule; thresholds may differ per image."""
    per_image = np.full((len(gt), 54), np.nan)
    for i, prob in probabilities(ckpt, device):
        pred = apply_rule(prob, thr_per_image[i])
        for c in range(1, 55):
            if ann[i, c - 1]:
                per_image[i, c - 1] = _dice(pred == c, gt[i] == c)
    per_class = np.nanmean(per_image, axis=0)
    return float(np.nanmean(per_class)), per_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the thresholds fitted on everything")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    _, val_idx = data.splits()
    gt, ann = data.train_labels()[val_idx], data.annotated_ids()[val_idx]
    n = len(val_idx)
    names = {int(k): v for k, v in data.organ_names()["names"].items() if v}

    rng = np.random.default_rng(args.seed)
    fold_of = np.array_split(rng.permutation(n), args.folds)

    saved = {}
    for ckpt in args.ckpt:
        print(f"\n===== {ckpt} =====")
        stats = separable_stats(ckpt, device, gt, ann)
        base = float(np.nanmean(np.nanmean(stats[:, :, 0], axis=0)))

        # Per class, cross-validated: thresholds chosen without the judged image.
        thr_cv = np.ones((n, 54))
        for f in fold_of:
            rest = np.setdiff1d(np.arange(n), f)
            thr_cv[f] = choose(stats, rest)
        cv, per_class_cv = score_joint(ckpt, device, gt, ann, thr_cv)

        # One threshold for all classes, cross-validated too.
        thr_glob = np.ones((n, 54))
        for f in fold_of:
            rest = np.setdiff1d(np.arange(n), f)
            m = [np.nanmean(np.nanmean(stats[rest][:, :, j], axis=0))
                 for j in range(len(GRID))]
            thr_glob[f] = GRID[int(np.argmax(m))]
        glob, _ = score_joint(ckpt, device, gt, ann, thr_glob)

        # Fitted on everything, evaluated on everything: the optimistic bound.
        thr_all = choose(stats, np.arange(n))
        opt, _ = score_joint(ckpt, device, gt, ann, np.tile(thr_all, (n, 1)))

        print(f"  plain argmax                  {base:.4f}")
        print(f"  single threshold, CV          {glob:.4f}   ({glob-base:+.4f})")
        print(f"  per class, CV                 {cv:.4f}   ({cv-base:+.4f})")
        print(f"  per class, fitted on all      {opt:.4f}   ({opt-base:+.4f})"
              f"   <- optimistic, does not generalise")

        moved = [c for c in range(1, 55) if thr_all[c - 1] < 1.0]
        print(f"  classes thresholded: {len(moved)}/54")
        gain = np.nan_to_num(per_class_cv) - np.nan_to_num(
            np.nanmean(stats[:, :, 0], axis=0))
        for c in np.argsort(-gain)[:8]:
            if gain[c] <= 0.005:
                break
            print(f"     {c+1:>2} {names.get(c+1, '?'):<28} "
                  f"{np.nanmean(stats[:, c, 0]):.3f} -> {per_class_cv[c]:.3f}"
                  f"   threshold {thr_all[c]:.2f}")
        saved[ckpt] = {"thresholds": thr_all.tolist(), "cv": cv, "argmax": base}

    if args.out:
        json.dump(saved, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
