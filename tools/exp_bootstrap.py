"""Experiment: paired bootstrap of a candidate ensemble against the submission of record.

Both sides are uniform averages followed by the adrenal threshold. The question
is the one asked before every submission: how likely is the candidate to beat
the record by more than `--margin` on validation, image sampling noise included.
"""

import argparse
import warnings

import numpy as np
import torch

from radium import data, metric
from radium.dataset import to_channels
from radium.models import load_checkpoint

warnings.filterwarnings("ignore")

RECORD = ("ft145_256 ft_lr1e4 ft_lr3e4 ft145_512 ft145_dinov3 ft145_dinov3_e03 "
          "ft145_dinov3_e10 exp_mixed")


@torch.no_grad()
def member_probs(name, images, device, chunk=16):
    model, hu = load_checkpoint(f"checkpoints/{name}.pt", device)
    out = np.zeros((len(images), 55, 256, 256), np.float16)
    for k in range(0, len(images), chunk):
        x = torch.from_numpy(to_channels(images[k : k + chunk], hu)).to(device)
        out[k : k + len(x)] = model((x - 0.5) / 0.25).softmax(1).cpu().numpy()
    return out


def decide(P, t):
    pred = P.argmax(1)
    a = P[:, 1:3]
    hit = (pred == 0) & (a.max(1) > t)
    pred[hit] = (a.argmax(1) + 1)[hit]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="space-separated member names")
    ap.add_argument("--t-candidate", type=float, required=True)
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--t-record", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.0015)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    _, va = data.splits()
    X, gt, ann = data.train_images()[va], data.train_labels()[va], data.annotated_ids()[va]
    names = sorted(set(args.candidate.split()) | set(args.record.split()))
    probs = {n: member_probs(n, X, device) for n in names}

    def per_image(members, t):
        P = np.mean([probs[n].astype(np.float32) for n in members.split()], axis=0)
        d = metric.dice_per_image(gt, decide(P, t))
        d[~ann] = np.nan
        return d

    score = lambda d: float(np.nanmean(np.nanmean(d, 0)))
    rec, cand = per_image(args.record, args.t_record), per_image(args.candidate, args.t_candidate)
    rng = np.random.default_rng(0)
    delta = np.array([score(cand[i]) - score(rec[i])
                      for i in (rng.integers(0, len(va), len(va)) for _ in range(args.n_boot))])
    print(f"record    {score(rec):.4f}  ({len(args.record.split())} members, t={args.t_record})")
    print(f"candidate {score(cand):.4f}  ({len(args.candidate.split())} members, t={args.t_candidate})")
    print(f"delta {score(cand) - score(rec):+.4f}  IC90 [{np.percentile(delta, 5):+.4f}, "
          f"{np.percentile(delta, 95):+.4f}]  P(>0) {np.mean(delta > 0):.2f}  "
          f"P(>+{args.margin}) {np.mean(delta > args.margin):.2f}")


if __name__ == "__main__":
    main()
