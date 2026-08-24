"""Torch dataset and augmentation."""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from radium import data

# (level, width) per input channel: soft tissue, lung, bone.
WINDOWS = ((40, 400), (-600, 1500), (400, 1800))

NUM_CLASSES = 54

# Organs that swap identity under a left-right mirror.
LATERAL_PAIRS = ((1, 2), (4, 5), (7, 8), (13, 14), (16, 17), (18, 19), (20, 21),
                 (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (34, 35),
                 (37, 38), (42, 43), (45, 46))
# Lateralised with no mirror twin: a flip makes their label a lie, so it is dropped.
LATERAL_UNPAIRED = (9, 15, 36, 48, 49)


def _mirror_tables():
    """-> (mask LUT over ids 0..54, permutation over the 54 annotated flags)."""
    lut = np.arange(NUM_CLASSES + 1, dtype=np.int64)
    perm = np.arange(NUM_CLASSES, dtype=np.int64)
    for a, b in LATERAL_PAIRS:
        lut[a], lut[b] = b, a
        perm[a - 1], perm[b - 1] = b - 1, a - 1
    for c in LATERAL_UNPAIRED:
        lut[c] = 0  # erased, and flagged unannotated alongside
    return lut, perm


MIRROR_LUT, MIRROR_PERM = _mirror_tables()
UNPAIRED_IDX = np.array([c - 1 for c in LATERAL_UNPAIRED])


def n_channels(hu: bool) -> int:
    return len(WINDOWS) if hu else 1


def to_channels(images, hu: bool) -> np.ndarray:
    """Raw slices -> (..., C, H, W) float32 in [0, 1]. Accepts (H, W) or (N, H, W)."""
    if not hu:
        return (np.asarray(images, dtype=np.float32) / 255.0)[..., None, :, :]
    return np.stack([data.window(images, l, w) for l, w in WINDOWS], axis=-3)


class SliceDataset(Dataset):
    """Yields (image, mask, annotated, pixel_weight)."""

    def __init__(self, images, labels, annotated, indices, train: bool = True,
                 hu: bool = False, flip: float = 0.0, weight=None):
        self.images = images
        self.labels = labels
        self.annotated = annotated
        self.weight = weight
        self.indices = np.asarray(indices)
        self.train = train
        self.hu = hu
        self.flip = flip  # probability of the left-right mirror; 0 disables it
        self._lut = torch.from_numpy(MIRROR_LUT)
        self._perm = torch.from_numpy(MIRROR_PERM)
        self._unpaired = torch.from_numpy(UNPAIRED_IDX)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        img = torch.from_numpy(to_channels(self.images[i], self.hu))
        mask = torch.from_numpy(self.labels[i].astype(np.int64))[None]
        ann = torch.from_numpy(self.annotated[i])
        w = (None if self.weight is None
             else torch.from_numpy(self.weight[i].astype(np.int64))[None])

        if self.train:
            if self.flip and np.random.rand() < self.flip:
                img, mask, ann, w = self._mirror(img, mask, ann, w)
            img, mask, w = self._augment(img, mask, w)

        img = (img - 0.5) / 0.25
        if w is None:
            w = torch.ones_like(mask)
        return img, mask[0], ann, w[0].float()

    def _mirror(self, img, mask, ann, w):
        """Left-right mirror: flip the rows, then relabel to match."""
        img = TF.vflip(img)
        flipped = TF.vflip(mask)
        mask = self._lut[flipped]             # paired ids swap, unpaired erased
        ann = ann[self._perm].clone()         # knowledge follows the swap
        ann[self._unpaired] = False           # erased organs become "unknown"
        if w is not None:
            # Erased pixels would read as confident background; they are unknown.
            w = TF.vflip(w) * (~((flipped > 0) & (mask == 0))).to(w.dtype)
        return img, mask, ann, w

    def _augment(self, img, mask, w):
        angle = float(np.random.uniform(-15, 15))
        scale = float(np.random.uniform(0.9, 1.1))
        tx = int(np.random.uniform(-16, 16))
        ty = int(np.random.uniform(-16, 16))

        # Identical geometry on all three; nearest on the mask keeps ids integral.
        img = TF.affine(img, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0],
                        interpolation=v2.InterpolationMode.BILINEAR)
        mask = TF.affine(mask, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0],
                         interpolation=v2.InterpolationMode.NEAREST)
        if w is not None:
            # fill=0: corners rotated in from outside the slice are unknown.
            w = TF.affine(w, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0],
                          interpolation=v2.InterpolationMode.NEAREST)

        if np.random.rand() < 0.5:  # mild windowing jitter
            img = (img * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)).clamp(0, 1)
        return img, mask, w


def rarity_weights(target, annotated, indices, power: float = 0.5, cap: float = 8.0):
    """-> (weights over `indices`, info dict) for a WeightedRandomSampler."""
    idx = np.asarray(indices)
    present = np.zeros((len(idx), NUM_CLASSES), dtype=bool)
    for k, i in enumerate(idx):
        for c in np.unique(target[i])[1:]:
            present[k, c - 1] = True

    freq = present.sum(0)
    ref = np.median(freq[freq > 0]) if (freq > 0).any() else 1.0
    per_class = np.where(freq > 0, (ref / np.maximum(freq, 1)) ** power, 1.0)
    per_class = np.clip(per_class, 1.0, cap)

    w = np.ones(len(idx))
    any_present = present.any(1)
    # The rarest organ on the slice sets its weight.
    w[any_present] = np.max(np.where(present, per_class[None, :], 0.0), axis=1)[any_present]
    w[annotated[idx].sum(1) == 0] = 0.0

    info = {"freq": freq, "per_class": per_class, "weights": w,
            "n_zero": int((w == 0).sum()), "max": float(w.max())}
    return w, info


def supervision_mask(Y, A, mode: str):
    """Rewrite the annotated mask to ablate what `annotated_labels.json` buys us."""
    if mode == "marginal":
        return A
    if mode == "self-train":
        path = data.CACHE / "annotated_selftrain.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run `uv run python make_pseudo.py --ckpt <model>`"
            )
        promoted = np.load(path)
        if promoted.shape != A.shape:
            raise ValueError(f"expected {A.shape}, got {promoted.shape}")
        return promoted
    if mode == "naive":
        return np.ones_like(A)
    if mode == "present-only":
        out = np.zeros_like(A)
        for i in range(len(Y)):
            for c in np.unique(Y[i])[1:]:
                out[i, c - 1] = True
        return out
    raise ValueError(f"unknown supervision mode: {mode}")


def external_arrays(root):
    """-> (images, labels, annotated) for a set built by `tools/make_external.py`."""
    from pathlib import Path

    root = Path(root)
    images = np.load(root / "images.npy", mmap_mode="r")
    labels = np.load(root / "labels.npy", mmap_mode="r")
    if len(images) != len(labels):
        raise ValueError(f"{len(images)} images vs {len(labels)} labels in {root}")
    return images, labels, np.ones((len(images), NUM_CLASSES), dtype=bool)


def loaders(batch_size=8, n_val=160, seed=0, num_workers=0, supervision="marginal",
            hu=False, flip=0.0, pseudo_run=None, gate=None, group=False,
            oversample=0.0, external=None):
    """-> (train_loader, val_loader, val_indices, info)."""
    if external is not None:
        if hu:
            raise ValueError("--external ships 8-bit slices; use --input png")
        if pseudo_run is not None:
            raise ValueError("--external replaces the training set, --pseudo cannot apply")
        Xe, Ye, Ae = external_arrays(external)
        _, va = data.splits(n_val=n_val, seed=seed, group=group)
        train = torch.utils.data.DataLoader(
            SliceDataset(Xe, Ye, Ae, np.arange(len(Xe)), train=True, hu=False, flip=flip),
            batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val = torch.utils.data.DataLoader(
            SliceDataset(data.train_images(), data.train_labels(), data.annotated_ids(),
                         va, train=False, hu=False),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train, val, va, {"external": len(Xe)}

    X = data.train_images_hu() if hu else data.train_images()
    # Load once and share: train_labels() re-reads 125 MB from disk on every call.
    Y0, A0 = data.train_labels(), data.annotated_ids()
    Y, A, W = Y0, A0, None
    tr, va = data.splits(n_val=n_val, seed=seed, group=group)

    if pseudo_run is None:
        A = supervision_mask(Y0, A0, supervision)
    else:
        if supervision != "marginal":
            raise ValueError(f"--pseudo replaces --supervision, got {supervision!r}")
        from radium import pseudo as pseudo_mod

        Y, A, W = pseudo_mod.merge(pseudo_run, Y=Y0, A=A0, **(gate or {}))
        pseudo_mod.verify(Y, W, Y=Y0, A=A0, annotated=A)  # truth must survive
        # The raw half never appears in validation, so all of it joins training.
        tr = np.concatenate([tr, np.arange(data.N_ANNOTATED, len(Y))])

    info = {}
    sampler = None
    if oversample:
        sw, info = rarity_weights(Y, A, tr, power=oversample)
        # Replacement is required for non-uniform weights; the epoch cost is unchanged.
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(sw).double(), num_samples=len(tr), replacement=True
        )

    make = lambda idx, train, y, a, w, smp=None: torch.utils.data.DataLoader(
        SliceDataset(X, y, a, idx, train=train, hu=hu, flip=flip if train else 0.0,
                     weight=w),
        batch_size=batch_size, shuffle=(train and smp is None), sampler=smp,
        num_workers=num_workers,
    )
    # Validation always uses the untouched annotations, never the overlay.
    return (make(tr, True, Y, A, W, sampler),
            make(va, False, Y0, A0, None),
            va, info)
