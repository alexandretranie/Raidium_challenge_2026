"""Experiment: a lower decision threshold for the two adrenals, calibrated off the challenge.

The 7-model ensemble sees the adrenals on TotalSegmentator (presence AUC 0.94)
but rarely lets them win the argmax against background. Here a background pixel
becomes an adrenal when that adrenal's probability exceeds `t`.

`t` is chosen on held-out external slices, scored like the challenge (every slice
counts, a stray blob on an adrenal-free slice is a 0), never on the 160
validation slices: four tunings fitted on those did not transfer.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

from radium import data, metric
from radium.dataset import to_channels
from radium.models import load_checkpoint

warnings.filterwarnings("ignore")

ENSEMBLE = ("ft145_256 ft_lr1e4 ft_lr3e4 ft145_512 "
            "ft145_dinov3 ft145_dinov3_e03 ft145_dinov3_e10").split()
ADRENALS = (1, 2)
GRID = (None, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05)
# make_external.py writes volumes in order, so the tail is a block of whole cases.
HOLDOUT_FROM = 0.9


@torch.no_grad()
def fused(models, images, device, chunk=8):
    """-> (argmax uint8 (N,H,W), adrenal probs float16 (N,2,H,W))."""
    pred = np.zeros((len(images), 256, 256), np.uint8)
    padr = np.zeros((len(images), 2, 256, 256), np.float16)
    for k in range(0, len(images), chunk):
        x = np.asarray(images[k : k + chunk])
        acc = 0
        for m, hu in models:
            acc = acc + m((torch.from_numpy(to_channels(x, hu)).to(device) - 0.5) / 0.25).softmax(1)
        acc = (acc / len(models)).cpu().numpy()
        pred[k : k + len(x)] = acc.argmax(1)
        padr[k : k + len(x)] = acc[:, list(ADRENALS)]
    return pred, padr


def apply(pred, padr, t):
    """Background pixels above `t` for an adrenal become that adrenal."""
    if t is None:
        return pred
    out = pred.copy()
    p = padr.astype(np.float32)
    best = p.argmax(1)  # which adrenal, where both clear the bar
    hit = (pred == 0) & (p.max(1) > t)
    out[hit] = np.array(ADRENALS, np.uint8)[best[hit]]
    return out


def adrenal_scores(gt, pred, annotated=None):
    """-> (dice adrenal L, dice adrenal R, overall) with the challenge averaging."""
    per = metric.dice_per_image(gt, pred)
    if annotated is not None:
        per[~annotated] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls = np.nanmean(per, axis=0)
    return cls[0], cls[1], float(np.nanmean(cls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-external", type=int, default=2000)
    ap.add_argument("--submit", default="submissions/exp_adrenal_threshold.csv")
    ap.add_argument("--members", default=" ".join(ENSEMBLE),
                    help="checkpoint names under checkpoints/, space-separated")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    models = [load_checkpoint(f"checkpoints/{c}.pt", device) for c in args.members.split()]

    # 1. Calibrate on held-out external slices, at their natural adrenal prevalence.
    Xe = np.load("external-data/external145/images.npy", mmap_mode="r")
    Ye = np.load("external-data/external145/labels.npy", mmap_mode="r")
    start = int(len(Xe) * HOLDOUT_FROM)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(np.arange(start, len(Xe)), args.n_external, replace=False))
    gt_e = np.asarray(Ye[idx])
    pred_e, padr_e = fused(models, Xe[idx], device)
    prev = [(gt_e == c).reshape(len(idx), -1).any(1).mean() for c in ADRENALS]
    print(f"external holdout: {len(idx)} slices from #{start}, "
          f"adrenal prevalence L {prev[0]:.3f} R {prev[1]:.3f}")
    print(f"{'t':>6} {'L':>6} {'R':>6} {'mean':>6}   stray (slices without, with a blob)")
    best_t, best_s = None, -1.0
    for t in GRID:
        p = apply(pred_e, padr_e, t)
        dl, dr, _ = adrenal_scores(gt_e, p)
        stray = np.mean([((p[i] == c).any() and not (gt_e[i] == c).any())
                         for i in range(len(idx)) for c in ADRENALS])
        s = (dl + dr) / 2
        flag = ""
        if s > best_s:
            best_s, best_t, flag = s, t, "  <-"
        print(f"{str(t):>6} {dl:6.3f} {dr:6.3f} {s:6.3f}   {stray:.3f}{flag}")
    print(f"chosen t = {best_t}")

    # 2. Read it on the 160 validation slices: a check, not a selection.
    _, va = data.splits()
    gt_v, ann_v = data.train_labels()[va], data.annotated_ids()[va]
    pred_v, padr_v = fused(models, data.train_images()[va], device)
    for name, t in (("argmax", None), (f"t={best_t}", best_t)):
        dl, dr, s = adrenal_scores(gt_v, apply(pred_v, padr_v, t), ann_v)
        print(f"validation {name:>8}: {s:.4f}   adrenal L {dl:.3f}  R {dr:.3f}")

    # 3. Submission, and how many test slices gain an adrenal.
    Xt = data.test_images()
    pred_t, padr_t = fused(models, Xt, device)
    plain = args.submit.replace(".csv", "_argmax.csv")
    metric.to_submission(pred_t).to_csv(plain)
    print(f"wrote {plain}  (no threshold)")
    out = apply(pred_t, padr_t, best_t)
    for c in ADRENALS:
        before = (pred_t == c).reshape(len(Xt), -1).any(1).sum()
        after = (out == c).reshape(len(Xt), -1).any(1).sum()
        print(f"test, adrenal {c}: predicted on {before} -> {after} of {len(Xt)} slices")
    Path(args.submit).parent.mkdir(parents=True, exist_ok=True)
    metric.to_submission(out).to_csv(args.submit)
    print(f"wrote {args.submit}  ({(out != pred_t).mean() * 100:.3f}% of test pixels changed)")


if __name__ == "__main__":
    main()
