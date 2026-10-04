"""Experiment: fine-tune on the 640 challenge slices mixed with external slices.

The fine-tuning of record sees only the 640 challenge slices, where the right
adrenal appears 15 times; the model memorises them (probability 0.4-0.6 on its
training slices) and finds nothing on validation. Here each epoch draws as many
external slices as challenge ones, weighted toward the weak classes, so that the
domain of the challenge is learnt without forgetting the anatomy.

The last tenth of the external set (whole volumes, see make_external.py) never
enters training: it is the benchmark for the weak classes, which validation
cannot be with 3 to 7 slices each.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from radium import data, metric
from radium.dataset import SliceDataset, external_arrays
from radium.losses import MarginalLoss
from radium.models import build_model, param_groups

EXT = "external-data/external145"
HOLDOUT_FROM = 0.9
# Validation Dice under 0.73 for the 7-model ensemble, measured before this run.
WEAK = (1, 2, 40, 10, 27, 28, 29, 30, 15, 39, 33, 11)
PRESENCE = Path("out/exp/external145_presence.npy")


def presence(labels):
    """-> (N, 55) bool, organ present per slice; cached, a full pass takes minutes."""
    if PRESENCE.exists():
        return np.load(PRESENCE)
    out = np.zeros((len(labels), 55), bool)
    for k in range(0, len(labels), 2000):
        y = np.asarray(labels[k : k + 2000]).reshape(-1, 256 * 256)
        for c in range(55):
            out[k : k + len(y), c] = (y == c).any(1)
    PRESENCE.parent.mkdir(parents=True, exist_ok=True)
    np.save(PRESENCE, out)
    return out


@torch.no_grad()
def evaluate(model, loader, val_idx, device):
    model.eval()
    pred = np.concatenate([model(img.to(device)).argmax(1).cpu().numpy().astype(np.uint8)
                           for img, *_ in loader])
    gt, ann = data.train_labels()[val_idx], data.annotated_ids()[val_idx]
    per = metric.dice_per_image(gt, pred)
    per[~ann] = np.nan
    cls = np.nanmean(per, axis=0)
    return float(np.nanmean(cls)), float(np.nanmean(cls[[c - 1 for c in WEAK]]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="checkpoints/pre145_256.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ext-per-epoch", type=int, default=640,
                   help="external slices drawn per epoch, next to the 640 challenge ones")
    p.add_argument("--other-share", type=float, default=0.2,
                   help="share of the external draws left to slices with no weak class")
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", default="checkpoints/exp_mixed.pt")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    X, Y, A = data.train_images(), data.train_labels(), data.annotated_ids()
    tr, va = data.splits()
    Xe, Ye, Ae = external_arrays(EXT)
    pool = np.arange(int(len(Xe) * HOLDOUT_FROM))
    pres = presence(Ye)[pool][:, list(WEAK)]
    weak = pres.any(1)
    # Class-balanced over the weak classes: a slice is worth 1/freq of the rarest
    # weak organ it carries, so each weak class gets a similar share of the draws.
    # A flat boost does not do it: 71 % of the pool carries some weak class.
    inv = 1.0 / pres.sum(0)
    w_ext = np.max(np.where(pres, inv[None, :], 0.0), axis=1)
    w_ext[weak] *= (1 - args.other_share) / w_ext[weak].sum()
    w_ext[~weak] = args.other_share / (~weak).sum()
    share = {c: float(w_ext[pres[:, k]].sum()) for k, c in enumerate(WEAK)}

    chal = SliceDataset(X, Y, A, tr, train=True)
    ext = SliceDataset(Xe, Ye, Ae, pool, train=True)
    both = torch.utils.data.ConcatDataset([chal, ext])
    # Expected draws per epoch: each challenge slice once, `ext_per_epoch` external.
    w = np.concatenate([np.ones(len(tr)), args.ext_per_epoch * w_ext / w_ext.sum()])
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(w).double(), num_samples=len(tr) + args.ext_per_epoch,
        replacement=True)
    train_loader = torch.utils.data.DataLoader(
        both, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0)
    val_loader = torch.utils.data.DataLoader(
        SliceDataset(X, Y, A, va, train=False), batch_size=args.batch_size)

    ckpt = torch.load(args.init, map_location=device)
    model = build_model(ckpt["arch"], ckpt["encoder"], pretrained=False, in_channels=1,
                        size=ckpt.get("size", 256)).to(device)
    model.load_state_dict(ckpt["model"])
    criterion = MarginalLoss(w_ce=1.0, w_dice=1.0, alpha=0.5, beta=0.5).to(device)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.encoder_lr_scale), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"init {args.init}  challenge {len(tr)} + external {args.ext_per_epoch}/epoch "
          f"from a pool of {len(pool)} ({weak.sum()} carry a weak class)")
    print("  share of external draws carrying: " + ", ".join(
        f"{c}={v:.2f}" for c, v in share.items()))
    meta = {"arch": ckpt["arch"], "encoder": ckpt["encoder"], "input": "png",
            "pseudo": None, "size": ckpt.get("size", 256)}
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for img, mask, ann, _ in train_loader:
            img, mask, ann = img.to(device), mask.to(device), ann.to(device)
            loss = criterion(model(img), mask, ann)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        fair, weak_d = evaluate(model, val_loader, va, device)
        flag = ""
        if fair > best:
            best, flag = fair, "  <- best"
            torch.save({"model": model.state_dict(), **meta}, args.out)
        print(f"epoch {epoch+1:>3}/{args.epochs}  loss {running/len(train_loader):.4f}  "
              f"dice {fair:.4f}  weak {weak_d:.4f}  {time.time()-t0:.0f}s{flag}")
    # The last epoch too: a fixed budget carries no selection optimism.
    last = args.out.replace(".pt", "_last.pt")
    torch.save({"model": model.state_dict(), **meta}, last)
    print(f"\nbest {best:.4f} -> {args.out}   last -> {last}")


if __name__ == "__main__":
    main()
