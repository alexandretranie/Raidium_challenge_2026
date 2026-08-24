"""Pseudo-label store and the overlay that never overwrites truth."""

import json
from pathlib import Path

import numpy as np

from radium import data

ROOT = data.ROOT / "pseudo-labels"
CHUNK = 200  # images per pass; the boolean temporaries are 65 kB each

TAU_CELL, TAU_PIX, TAU_BG = 0.5, 0.9, 0.9


def run_dir(run: str) -> Path:
    return ROOT / run


def write(run: str, labels, conf, peak, meta: dict) -> Path:
    d = run_dir(run)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "labels.npy", labels.astype(np.uint8))
    np.save(d / "conf.npy", conf.astype(np.uint8))
    np.save(d / "peak.npy", peak.astype(np.float16))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d


def load(run: str):
    d = run_dir(run)
    if not d.exists():
        raise FileNotFoundError(f"{d} missing — run `uv run python make_pseudo_pixels.py`")
    return (np.load(d / "labels.npy"), np.load(d / "conf.npy"),
            np.load(d / "peak.npy").astype(np.float32),
            json.loads((d / "meta.json").read_text()))


def merge(run: str, Y=None, A=None, tau_cell=TAU_CELL, tau_pix=TAU_PIX, tau_bg=TAU_BG):
    """-> (target, annotated, weight) over all N slices of the run."""
    labels, conf, peak, _ = load(run)
    n = len(labels)
    Y = data.train_labels()[:n] if Y is None else Y[:n]
    A = data.annotated_ids()[:n] if A is None else A[:n]

    accepted = (peak > tau_cell) & ~A                      # (N, 54)
    annotated = A | accepted
    target = Y.copy()
    weight = np.ones_like(target)
    pix_cut, bg_cut = int(tau_pix * 255), int(tau_bg * 255)

    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        lab, cf, y = labels[s:e], conf[s:e], Y[s:e]
        rows, r, c = np.nonzero(lab)
        keep = lab[rows, r, c] > 0
        rows, r, c = rows[keep], r[keep], c[keep]
        # is the organ this pixel was assigned actually accepted on this slice?
        ok = accepted[rows + s, lab[rows, r, c].astype(np.int64) - 1]
        ok &= (cf[rows, r, c] > pix_cut) & (y[rows, r, c] == 0)
        tgt = y.copy()
        tgt[rows[ok], r[ok], c[ok]] = lab[rows[ok], r[ok], c[ok]]
        target[s:e] = tgt

        if tau_bg > 0:
            touched = accepted[s:e].any(1)[:, None, None]
            weight[s:e] = ~(touched & (tgt == 0) & (cf <= bg_cut))

    return target, annotated, weight


def verify(target, weight, Y=None, A=None, annotated=None) -> dict:
    """Assert the overlay never touched an annotated pixel, and report coverage."""
    n = len(target)
    Y = data.train_labels()[:n] if Y is None else Y[:n]
    A = data.annotated_ids()[:n] if A is None else A[:n]

    kept = Y > 0
    if not np.array_equal(target[kept], Y[kept]):
        raise AssertionError("pseudo-labels overwrote annotated pixels")

    written = (target > 0) & ~kept
    out = {
        "annotated_px": int(kept.sum()),
        "pseudo_px": int(written.sum()),
        "dropped_px": int((weight == 0).sum()),
        "px_total": int(target.size),
    }
    if annotated is not None:
        out["cells_declared"] = int(A.sum())
        out["cells_accepted"] = int(annotated.sum() - A.sum())
    return out
