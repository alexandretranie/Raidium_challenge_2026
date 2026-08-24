"""The challenge metric, vectorised."""

import warnings

import numpy as np
import pandas as pd

NUM_CLASSES = 54


def _nanmean_2step(per_image: np.ndarray) -> float:
    """Average over images per class, then over classes."""
    with warnings.catch_warnings():
        # An all-nan column is the intended result; the outer nanmean drops it.
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(np.nanmean(per_image, axis=0)))


def dice_per_image(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """(N, 54) per-image per-class Dice, nan where both sides are empty."""
    y_true = y_true.reshape(len(y_true), -1).astype(np.int64)
    y_pred = y_pred.reshape(len(y_pred), -1).astype(np.int64)
    n = len(y_true)
    out = np.full((n, NUM_CLASSES), np.nan)

    size = NUM_CLASSES + 1
    for i in range(n):
        # One pass per image: joint histogram of (true, pred) over all pixels.
        conf = np.bincount(
            y_true[i] * size + y_pred[i], minlength=size * size
        ).reshape(size, size)
        inter = np.diag(conf)[1:].astype(np.float64)
        denom = conf.sum(axis=1)[1:] + conf.sum(axis=0)[1:]
        present = denom > 0
        out[i, present] = 2 * inter[present] / denom[present]
    return out


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average over images for each class, then over classes."""
    return _nanmean_2step(dice_per_image(y_true, y_pred))


def dice_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """(54,) per-class score — use it to find which organs are being missed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(dice_per_image(y_true, y_pred), axis=0)


def dice_score_annotated(
    y_true: np.ndarray, y_pred: np.ndarray, annotated: np.ndarray
) -> float:
    """Validation score restricted to the classes actually annotated per image."""
    per_image = dice_per_image(y_true, y_pred)
    per_image[~annotated] = np.nan
    return _nanmean_2step(per_image)


def to_submission(y_pred: np.ndarray, image_ids: np.ndarray | None = None) -> pd.DataFrame:
    """(N, 256, 256) -> the on-disk CSV orientation: 65536 rows x N columns."""
    flat = y_pred.reshape(len(y_pred), -1).astype(np.uint8)
    if image_ids is None:
        image_ids = np.arange(len(y_pred))
    return pd.DataFrame(
        flat.T,
        index=[f"Pixel {i}" for i in range(flat.shape[1])],
        columns=[f"{i}.png" for i in image_ids],
    )
