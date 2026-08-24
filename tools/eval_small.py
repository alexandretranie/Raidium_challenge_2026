"""Per-class report on the organs that actually cap the score."""

import argparse
import warnings

import numpy as np
import torch

from radium import data, metric
from radium.dataset import to_channels
from radium.infer import load_model, predict

warnings.filterwarnings("ignore", message="Mean of empty slice")

# The 9 organs under 200 px, smallest first. Hardcoded so rows stay comparable.
TINY = [2, 1, 28, 27, 30, 29, 11, 40, 33]


@torch.no_grad()
def mean_prob_on_truth(ckpt, device, batch_size=8):
    """-> (54,) mean probability given to the correct class on its true pixels."""
    model, hu = load_model(ckpt, device)
    _, val_idx = data.splits()
    images = data.train_images_hu() if hu else data.train_images()
    gt = data.train_labels()[val_idx]

    total = np.zeros(54)
    count = np.zeros(54, dtype=np.int64)
    for k in range(0, len(val_idx), batch_size):
        sl = slice(k, k + batch_size)
        x = torch.from_numpy(to_channels(images[val_idx][sl], hu))
        prob = model((x.to(device) - 0.5) / 0.25).softmax(1).cpu().numpy()
        truth = gt[sl]
        for c in range(1, 55):
            m = truth == c
            if m.any():
                total[c - 1] += prob[:, c][m].sum()
                count[c - 1] += int(m.sum())
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def per_class_dice(ckpt, device):
    model, hu = load_model(ckpt, device)
    _, val_idx = data.splits()
    images = data.train_images_hu() if hu else data.train_images()
    pred = predict(model, images[val_idx], device, hu=hu)
    gt, ann = data.train_labels()[val_idx], data.annotated_ids()[val_idx]

    per_image = metric.dice_per_image(gt, pred)
    per_image[~ann] = np.nan
    return (metric.dice_score_annotated(gt, pred, ann),
            np.nanmean(per_image, axis=0),
            sorted(set(range(1, 55)) - set(np.unique(pred)[1:].tolist())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--prob", action="store_true",
                   help="second table: mean softmax on the true pixels, the readout "
                        "that still moves when the Dice is pinned at 0")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    names = {int(k): v for k, v in data.organ_names()["names"].items() if v}
    names.update({40: "portal_vein", 11: "esophagus", 33: "vena_cava"})  # noms courts

    head = f"{'checkpoint':<28} {'dice':>7} {'tiny9':>7} {'muets':>6}  |"
    head += "".join(f"{c:>7}" for c in TINY)
    print(head)
    print("-" * len(head))

    for ckpt in args.ckpt:
        fair, per_class, silent = per_class_dice(ckpt, device)
        tiny = np.array([per_class[c - 1] for c in TINY])
        row = f"{ckpt:<28} {fair:>7.4f} {np.nanmean(tiny):>7.3f} {len(silent):>6}  |"
        row += "".join(f"{v:>7.3f}" if not np.isnan(v) else f"{'-':>7}" for v in tiny)
        print(row)
        print(f"{'':<28} never predicted: {silent}")

    print("\n`tiny9` is the mean over the 9 organs under 200 px, which is what these")
    print("experiments target. +0.10 on tiny9 is worth +0.017 on the overall score.")

    if not args.prob:
        return

    head = f"\n{'checkpoint':<28} {'moy9':>7}         |" + "".join(f"{c:>7}" for c in TINY)
    print(head)
    print("-" * (len(head) - 1))
    for ckpt in args.ckpt:
        pr = mean_prob_on_truth(ckpt, device)
        tiny = np.array([pr[c - 1] for c in TINY])
        row = f"{ckpt:<28} {np.nanmean(tiny):>7.4f}         |"
        row += "".join(f"{v:>7.3f}" if not np.isnan(v) else f"{'-':>7}" for v in tiny)
        print(row)
    print("\nMean softmax on the organ's true pixels. Continuous, so it still moves")
    print("where Dice is pinned at 0. It takes about 0.5 for the argmax to flip.")


if __name__ == "__main__":
    main()
