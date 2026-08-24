"""Near-duplicate slices: a free pseudo-label bench, and a split leak."""

import numpy as np

from radium import data

CORR = 0.97   # pooled-correlation screen, deliberately loose
MAE = 5.0     # grey levels; ~46 HU, below the 9.12 HU quantisation step times 5


def _pooled(images: np.ndarray) -> np.ndarray:
    """(N, H, W) -> (N, 1024) unit-norm, mean-removed: dot product = correlation."""
    n = len(images)
    f = images.reshape(n, 32, 8, 32, 8).mean((2, 4)).reshape(n, -1).astype(np.float32)
    f -= f.mean(1, keepdims=True)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


def find_pairs(images: np.ndarray, corr: float = CORR, mae: float = MAE) -> list:
    """-> [(i, j)] with i < j, slices that are the same anatomy."""
    f = _pooled(images)
    out = []
    for i, j in np.argwhere(np.triu(f @ f.T, 1) > corr):
        d = np.abs(images[i].astype(np.int32) - images[j].astype(np.int32)).mean()
        if d < mae:
            out.append((int(i), int(j)))
    return out


def groups(n: int, pairs: list) -> np.ndarray:
    """(n,) group id per slice — twins share one, so a split can keep them together."""
    parent = np.arange(n)

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[root(i)] = root(j)
    return np.array([root(i) for i in range(n)])


def bench_cells(pairs: list, annotated: np.ndarray) -> list:
    """-> [(target, source, class)] cells unknown on `target`, declared on its twin."""
    return [(tgt, src, c)
            for i, j in pairs
            for tgt, src in ((i, j), (j, i))
            for c in np.where(annotated[src] & ~annotated[tgt])[0]]


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    d = a.sum() + b.sum()
    return np.nan if d == 0 else 2.0 * (a & b).sum() / d


def ceiling(pairs: list, labels: np.ndarray, annotated: np.ndarray) -> float:
    """Mean Dice between the two masks of organs declared on BOTH members."""
    vals = [_dice(labels[i] == c + 1, labels[j] == c + 1)
            for i, j in pairs
            for c in np.where(annotated[i] & annotated[j])[0]]
    return float(np.nanmean(vals))


def score(pred: dict, cells: list, labels: np.ndarray) -> dict:
    """Score a pseudo-labeller. `pred` maps slice index -> (H, W) label map."""
    dices, found, empty = [], [], []
    for tgt, src, c in cells:
        truth = labels[src] == c + 1
        got = pred[tgt] == c + 1
        if truth.any():
            dices.append(_dice(got, truth))
            found.append(bool(got.any()))
        else:
            empty.append(not got.any())
    return {
        "n_cells": len(cells),
        "n_present": len(dices),
        "dice": float(np.nanmean(dices)) if dices else np.nan,
        "found": float(np.mean(found)) if found else np.nan,
        "n_absent": len(empty),
        "correctly_empty": float(np.mean(empty)) if empty else np.nan,
    }


def annotated_pairs() -> list:
    """The 24 pairs among the 800 annotated slices, cached. For the bench."""
    def build():
        p = find_pairs(data.train_images()[: data.N_ANNOTATED])
        return np.array(p, dtype=np.int32).reshape(-1, 2)

    return [tuple(map(int, p)) for p in data._cached("twin_pairs", build)]


def leak_pairs() -> list:
    """Looser pairs, for splitting only: pooled correlation alone, no mae check."""
    def build():
        f = _pooled(data.train_images()[: data.N_ANNOTATED])
        return np.argwhere(np.triu(f @ f.T, 1) > 0.95).astype(np.int32)

    return [tuple(map(int, p)) for p in data._cached("twin_leak_pairs", build)]
