"""Promote unknown (image, organ) cells to reliable absences."""

import numpy as np
import torch

from radium.dataset import to_channels


@torch.no_grad()
def absence_scores(model, images, device, hu=False, batch_size=8):
    """(N, 54) highest softmax probability each organ reaches anywhere in a slice."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(to_channels(images[i : i + batch_size], hu))
        x = ((x - 0.5) / 0.25).to(device)
        probs = model(x).softmax(1)  # (B, 55, H, W); channel 0 is background
        out.append(probs[:, 1:].amax(dim=(2, 3)).float().cpu().numpy())
    return np.concatenate(out)


def presence(labels):
    """(N, 54) bool: does organ c+1 actually appear in the mask of image i."""
    out = np.zeros((len(labels), 54), dtype=bool)
    for i, y in enumerate(labels):
        for c in np.unique(y)[1:]:
            out[i, c - 1] = True
    return out


def calibrate(scores, declared, present, max_loss=0.0):
    """(54,) per-organ thresholds, -1 where no threshold is safe."""
    n_classes = scores.shape[1]
    thresholds = np.full(n_classes, -1.0)
    for c in range(n_classes):
        sel = declared[:, c]
        if not sel.any():
            continue
        pos = np.sort(scores[sel, c][present[sel, c]])
        if len(pos) == 0:
            thresholds[c] = np.inf  # examined many times, never there
            continue
        k = int(np.floor(max_loss * len(pos)))
        thresholds[c] = pos[k] if k < len(pos) else np.inf
    return thresholds


def promote(annotated, scores, thresholds, n_annotated):
    """Copy of `annotated` with confident unknowns marked examined."""
    out = annotated.copy()
    region = np.zeros_like(out)
    region[:n_annotated] = True
    out[region & ~annotated & (scores < thresholds[None, :])] = True
    return out
