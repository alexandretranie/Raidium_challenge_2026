"""Teachers for pseudo-labelling: one checkpoint or a weighted ensemble."""

import numpy as np
import torch
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as TF

from radium.dataset import to_channels
from radium.models import load_checkpoint

# (angle degrees, scale) — mirrors the training augmentation, minus the flip.
TTA_VIEWS = ((0.0, 1.0), (-10.0, 1.0), (10.0, 1.0), (0.0, 0.95), (0.0, 1.05))


def parse_spec(spec: str):
    """'a.pt:0.3,b.pt' -> [('a.pt', 0.3), ('b.pt', 0.7/…)] normalised to sum 1."""
    members = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        path, _, w = part.partition(":")
        members.append((path.strip(), float(w) if w else 1.0))
    total = sum(w for _, w in members)
    if total <= 0:
        raise ValueError(f"weights sum to {total} in {spec!r}")
    return [(p, w / total) for p, w in members]


def _warp(x, angle, scale, fill=0.0):
    return TF.affine(x, angle=angle, translate=[0, 0], scale=scale, shear=[0.0],
                     interpolation=InterpolationMode.BILINEAR, fill=fill)


class Teacher:
    def __init__(self, spec: str, device, views=None):
        self.device = device
        self.views = tuple(views) if views else ((0.0, 1.0),)
        self.members = []
        for path, w in parse_spec(spec):
            model, hu = load_checkpoint(path, device)
            self.members.append((model, hu, w, path))

    def __len__(self):
        return len(self.members)

    @property
    def needs(self):
        """-> set of {'png', 'hu'}: which image arrays `probs` must be given."""
        return {"hu" if hu else "png" for _, hu, _, _ in self.members}

    def describe(self):
        return " + ".join(f"{w:.0%} {p}" for _, _, w, p in self.members)

    @torch.no_grad()
    def probs(self, png=None, hu=None):
        """-> (B, 55, H, W) float32 probabilities on the device."""
        acc = cover = None
        for model, wants_hu, w, path in self.members:
            src = hu if wants_hu else png
            if src is None:
                raise ValueError(f"{path} needs {'hu' if wants_hu else 'png'} images")
            x0 = torch.from_numpy(to_channels(src, wants_hu)).to(self.device)
            for angle, scale in self.views:
                warped = (angle != 0.0) or (scale != 1.0)
                x = _warp(x0, angle, scale) if warped else x0
                p = model((x - 0.5) / 0.25).softmax(1)
                ones = torch.ones_like(p[:, :1])
                if warped:
                    # Undo the view before averaging, and track coverage.
                    p = _warp(p, -angle, 1 / scale)
                    ones = _warp(ones, -angle, 1 / scale)
                acc = w * p if acc is None else acc + w * p
                cover = w * ones if cover is None else cover + w * ones
        p = acc / cover.clamp(min=1e-6)
        return p / p.sum(1, keepdim=True).clamp(min=1e-6)

    def run(self, images_png=None, images_hu=None, indices=None, batch_size=8,
            progress=None):
        """-> (labels uint8, conf uint8, peak float32) over `indices`."""
        ref = images_png if images_png is not None else images_hu
        idx = np.arange(len(ref)) if indices is None else np.asarray(indices)
        h = ref.shape[1]
        labels = np.zeros((len(idx), h, h), np.uint8)
        conf = np.zeros_like(labels)
        peak = np.zeros((len(idx), 54), np.float32)

        for k in range(0, len(idx), batch_size):
            sel = idx[k : k + batch_size]
            p = self.probs(png=None if images_png is None else images_png[sel],
                           hu=None if images_hu is None else images_hu[sel])
            top = p.max(1)
            labels[k : k + batch_size] = top.indices.to(torch.uint8).cpu().numpy()
            conf[k : k + batch_size] = (
                (top.values * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            )
            peak[k : k + batch_size] = p[:, 1:].amax(dim=(2, 3)).float().cpu().numpy()
            if progress:
                progress(min(k + batch_size, len(idx)), len(idx))
        return labels, conf, peak
