"""Inference and its post-processing."""

import numpy as np
import torch
from scipy import ndimage

from radium.dataset import to_channels
from radium.models import load_checkpoint

load_model = load_checkpoint  # the name the scripts and the notebook use


@torch.no_grad()
def predict(model, images, device, batch_size=16, hu=False):
    """-> (N, H, W) uint8 label maps."""
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(to_channels(images[i : i + batch_size], hu))
        x = (x - 0.5) / 0.25
        out.append(model(x.to(device)).argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(out)


def drop_small(pred, min_size):
    """Zero out connected components smaller than min_size, per class per image."""
    if min_size <= 0:
        return pred
    clean = pred.copy()
    for i in range(len(pred)):
        for c in np.unique(pred[i]):
            if c == 0:
                continue
            m = pred[i] == c
            lab, n = ndimage.label(m)
            if n == 0:
                continue
            sizes = ndimage.sum_labels(m, lab, index=range(1, n + 1))
            for j, s in enumerate(sizes, start=1):
                if s < min_size:
                    clean[i][lab == j] = 0
    return clean
