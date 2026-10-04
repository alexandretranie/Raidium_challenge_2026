"""Experiment: weak-class readout of single checkpoints, on validation and on the external holdout.

The external holdout (last tenth of external145, whole volumes) is scored like
the test: all 54 organs, every slice, a stray blob counts as a 0.
"""

import argparse
import warnings

import numpy as np
import torch

from radium import data, metric
from radium.dataset import to_channels
from radium.models import load_checkpoint

warnings.filterwarnings("ignore")

WEAK = (1, 2, 40, 10, 27, 28, 29, 30, 15, 39, 33, 11)
HOLDOUT_FROM = 0.9


@torch.no_grad()
def predict(model, hu, images, device, chunk=16):
    out = np.zeros((len(images), 256, 256), np.uint8)
    for k in range(0, len(images), chunk):
        x = torch.from_numpy(to_channels(np.asarray(images[k : k + chunk]), hu)).to(device)
        out[k : k + len(x)] = model((x - 0.5) / 0.25).argmax(1).cpu().numpy()
    return out


def per_class(gt, pred, annotated=None):
    per = metric.dice_per_image(gt, pred)
    if annotated is not None:
        per[~annotated] = np.nan
    return np.nanmean(per, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n-external", type=int, default=2000)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    _, va = data.splits()
    Xv, gv, av = data.train_images()[va], data.train_labels()[va], data.annotated_ids()[va]
    Xe = np.load("external-data/external145/images.npy", mmap_mode="r")
    Ye = np.load("external-data/external145/labels.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(np.arange(int(len(Xe) * HOLDOUT_FROM), len(Xe)),
                             args.n_external, replace=False))
    Xh, gh = Xe[idx], np.asarray(Ye[idx])
    w = [c - 1 for c in WEAK]

    rows = []
    for path in args.ckpt:
        model, hu = load_checkpoint(path, device)
        cv = per_class(gv, predict(model, hu, Xv, device), av)
        ch = per_class(gh, predict(model, hu, Xh, device))
        rows.append((path, cv, ch))

    print(f"{'checkpoint':<34} {'val':>7} {'val weak':>8} {'ext':>7} {'ext weak':>8}")
    for path, cv, ch in rows:
        print(f"{path:<34} {np.nanmean(cv):7.4f} {np.nanmean(cv[w]):8.4f} "
              f"{np.nanmean(ch):7.4f} {np.nanmean(ch[w]):8.4f}")
    print("\nper weak class, external holdout (val in brackets)")
    print(f"{'id':>3} " + " ".join(f"{p.split('/')[-1][:18]:>20}" for p, *_ in rows))
    for c in WEAK:
        print(f"{c:>3} " + " ".join(f"{ch[c-1]:9.3f} ({cv[c-1]:5.2f})   " for _, cv, ch in rows))


if __name__ == "__main__":
    main()
