"""Side-by-side visualisation: scan, ground truth, prediction, agreement.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from radium import data, metric
from radium.infer import load_model, predict

# 54 visually distinct colours, fixed per class id.
_base = np.concatenate([
    plt.get_cmap("tab20")(np.linspace(0, 1, 20)),
    plt.get_cmap("tab20b")(np.linspace(0, 1, 20)),
    plt.get_cmap("tab20c")(np.linspace(0, 1, 20)),
])[:54]
CMAP = ListedColormap(_base)


def _overlay(ax, scan, mask, title):
    ax.imshow(scan, cmap="gray")
    m = np.ma.masked_where(mask == 0, mask)
    ax.imshow(m, cmap=CMAP, alpha=0.72, vmin=1, vmax=54, interpolation="nearest")
    ax.set_title(title, fontsize=7)  # anatomical names are long, ids were not
    ax.axis("off")


def _agreement(ax, scan, gt, pred, annotated):
    """green = correct, red = wrong id, blue = missed, orange = spurious."""
    scored = np.isin(pred, np.flatnonzero(annotated) + 1) | (gt > 0)
    rgba = np.zeros((*gt.shape, 4))  # alpha carries "this pixel has a verdict"
    hit = (gt > 0) & (pred == gt)
    wrong = (gt > 0) & (pred != gt) & (pred > 0)
    missed = (gt > 0) & (pred == 0)
    spurious = (gt == 0) & (pred > 0) & scored
    rgba[hit] = (0.15, 0.75, 0.25, 0.8)
    rgba[wrong] = (0.85, 0.15, 0.15, 0.8)
    rgba[missed] = (0.20, 0.45, 0.95, 0.8)
    rgba[spurious] = (1.00, 0.65, 0.10, 0.8)
    ax.imshow(scan, cmap="gray")
    ax.imshow(rgba, interpolation="nearest")
    ax.set_title("green=exact  red=wrong id\nblue=missed  orange=spurious", fontsize=7)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best_resnet34.pt")
    p.add_argument("--n", type=int, default=8, help="cases per figure")
    p.add_argument("--out", default="figures/viz_validation.png")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model, hu = load_model(args.ckpt, device)
    _, val_idx = data.splits()
    X = data.train_images_hu() if hu else data.train_images()
    Y, A = data.train_labels(), data.annotated_ids()
    scans, gts, anns = X[val_idx], Y[val_idx], A[val_idx]
    preds = predict(model, scans, device, hu=hu)
    # The grey panels always show the 8-bit slice, whatever the model was fed.
    display = data.train_images()[val_idx]

    per_img = metric.dice_per_image(gts, preds)
    per_img[~anns] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = np.nanmean(per_img, axis=1)

    order = np.argsort(-np.nan_to_num(scores, nan=-1))
    valid = [i for i in order if not np.isnan(scores[i])]
    k = args.n // 4
    picks = valid[:k] + valid[len(valid)//2 - k//2 : len(valid)//2 - k//2 + k] + valid[-2*k:]
    tags = ["best"] * k + ["median"] * k + ["worst"] * (2 * k)

    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.2 * len(picks)))

    def fmt(ids, per_line=2):
        """Anatomical names, wrapped — a bare id list says nothing about what
        the model actually missed, which is the whole point of looking."""
        labels = [data.organ_label(c) for c in ids]
        if not labels:
            return "(none)"
        lines = [", ".join(labels[k:k + per_line])
                 for k in range(0, len(labels), per_line)]
        return "\n".join(lines[:5]) + ("\n…" if len(lines) > 5 else "")

    for r, (i, tag) in enumerate(zip(picks, tags)):
        gt_ids = sorted(np.unique(gts[i])[1:].tolist())
        pr_ids = sorted(np.unique(preds[i])[1:].tolist())
        _overlay(axes[r, 0], display[i], np.zeros_like(gts[i]),
                 f"[{tag}] img {val_idx[i]} — dice {scores[i]:.3f}")
        _overlay(axes[r, 1], display[i], gts[i], f"partial truth\n{fmt(gt_ids)}")
        _overlay(axes[r, 2], display[i], preds[i], f"prediction\n{fmt(pr_ids)}")
        _agreement(axes[r, 3], display[i], gts[i], preds[i], anns[i])
    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(picks)} cases)")
    print("scores: min %.3f  median %.3f  max %.3f" % (
        np.nanmin(scores), np.nanmedian(scores), np.nanmax(scores)))


if __name__ == "__main__":
    main()
