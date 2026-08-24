"""Can a zoom cascade work? An upper bound, measured on crops."""

import argparse
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from radium import data
from radium.losses import MarginalLoss

# targets: what to lift off zero. context: the neighbours that give the landmark.
REGIONS = {
    "adrenal": {"targets": [1, 2],
                "context": [3, 33, 34, 35, 36, 39, 40, 48, 49, 9]},
    "iliac": {"targets": [27, 28, 29, 30],
              "context": [23, 24, 31, 32, 44, 47, 9, 53]},
}


class CropDataset(Dataset):
    """Square crop centred on the target, upscaled to `size`."""

    def __init__(self, images, labels, annotated, indices, classes, targets,
                 context, crop, size, train):
        self.images, self.labels, self.annotated = images, labels, annotated
        self.indices = np.asarray(indices)
        self.classes = classes            # original ids, in local order
        self.targets, self.context = targets, context
        self.crop, self.size, self.train = crop, size, train
        self.lut = np.zeros(55, dtype=np.int64)
        for local, c in enumerate(classes, start=1):
            self.lut[c] = local

    def __len__(self):
        return len(self.indices)

    def _centre(self, y, targets, context):
        for group in (targets, context):
            m = np.isin(y, group)
            if m.any():
                rr, cc = np.nonzero(m)
                return (rr.min() + rr.max()) // 2, (cc.min() + cc.max()) // 2
        return 128, 128

    def __getitem__(self, k):
        i = self.indices[k]
        y = self.labels[i]
        row, col = self._centre(y, self.targets, self.context)
        if self.train:
            row += np.random.randint(-12, 13)
            col += np.random.randint(-12, 13)
        h = self.crop // 2
        row = int(np.clip(row, h, 256 - h))
        col = int(np.clip(col, h, 256 - h))
        box = (slice(row - h, row + h), slice(col - h, col + h))

        img = torch.from_numpy(self.images[i][box].astype(np.float32) / 255.0)[None]
        mask = torch.from_numpy(self.lut[y[box]])[None, None].float()
        img = F.interpolate(img[None], size=(self.size, self.size),
                            mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask, size=(self.size, self.size), mode="nearest")[0, 0]

        ann = torch.from_numpy(self.annotated[i][[c - 1 for c in self.classes]])
        img = (img - 0.5) / 0.25
        return img, mask.long(), ann


@torch.no_grad()
def readout(model, loader, classes, targets, device):
    """-> (dice, mean probability on the true pixels) per target class."""
    inter = np.zeros(len(classes))
    denom = np.zeros(len(classes))
    total = np.zeros(len(classes))
    count = np.zeros(len(classes), dtype=np.int64)
    model.eval()
    for img, mask, _ in loader:
        logits = model(img.to(device))
        prob = logits.softmax(1).cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
        mask = mask.numpy()
        for local in range(1, len(classes) + 1):
            t, p = mask == local, pred == local
            inter[local - 1] += (t & p).sum()
            denom[local - 1] += t.sum() + p.sum()
            if t.any():
                total[local - 1] += prob[:, local][t].sum()
                count[local - 1] += int(t.sum())
    dice = np.where(denom > 0, 2 * inter / np.maximum(denom, 1), np.nan)
    prob = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    keep = [classes.index(c) for c in targets]
    return dice[keep], prob[keep], count[keep]


def run(arm, classes, region, args, device):
    targets = region["targets"]
    X, Y, A = data.train_images(), data.train_labels(), data.annotated_ids()
    # Pool from the region, not the arm's class set, or the arms see different data.
    pool = [i for i in range(data.N_ANNOTATED)
            if np.isin(Y[i], targets + region["context"]).any()]
    tr_all, va_all = data.splits()
    tr = [i for i in pool if i in set(tr_all.tolist())]
    va = [i for i in pool if i in set(va_all.tolist())]

    def make(idx, train):
        ds = CropDataset(X, Y, A, idx, classes, targets, region["context"],
                         args.crop, args.size, train)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train)

    train_loader, val_loader = make(tr, True), make(va, False)
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1,
                     classes=len(classes) + 1).to(device)
    criterion = MarginalLoss(w_ce=1.0, w_dice=1.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"\n=== {arm}: {len(classes)} classes, {len(tr)} train crops / "
          f"{len(va)} val, {args.crop}px -> {args.size}px ===")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for img, mask, ann in train_loader:
            loss = criterion(model(img.to(device)), mask.to(device), ann.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            dice, prob, _ = readout(model, val_loader, classes, targets, device)
            print(f"  epoch {epoch+1:>3}  loss {running/len(train_loader):.4f}  "
                  f"dice {np.nanmean(dice):.3f}  prob {np.nanmean(prob):.4f}")
    print(f"  {time.time()-t0:.0f}s")
    return readout(model, val_loader, classes, targets, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="iliac", choices=sorted(REGIONS))
    p.add_argument("--crop", type=int, default=128, help="crop size in px")
    p.add_argument("--size", type=int, default=256, help="size the network sees")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    region = REGIONS[args.region]
    targets = region["targets"]
    local = sorted(targets + region["context"])
    full = list(range(1, 55))

    names = {int(k): v for k, v in data.organ_names()["names"].items() if v}
    print(f"region {args.region}, targets: " +
          ", ".join(f"{c} {names.get(c, '?')}" for c in targets))
    print(f"zoom: {args.crop}px upscaled to {args.size}px, x{args.size/args.crop:.1f}")

    results = {
        "crop + local classes": run("crop + local classes", local,
                                          region, args, device),
        "crop + 55 classes": run("crop + 55 classes", full,
                                     region, args, device),
    }

    print(f"\n{'arm':<28}" + "".join(f"{c:>9}" for c in targets) + f"{'mean':>9}")
    print("-" * (28 + 9 * (len(targets) + 1)))
    print(f"{'full image (resnet34)':<28}" +
          "".join(f"{0.0:>9.3f}" for _ in targets) + f"{0.0:>9.3f}")
    for arm, (_, prob, _) in results.items():
        print(f"{arm:<28}" + "".join(f"{v:>9.3f}" for v in prob) +
              f"{np.nanmean(prob):>9.3f}")
    print("\nMean softmax on the target's true pixels. The 'full image' row is what")
    print("the same resnet34 gets without a crop: 0.000 everywhere. Anything above it")
    print("is what a cascade would buy, in the best case possible.")


if __name__ == "__main__":
    main()
