"""Loading for the Raidium challenge (ENS Challenge Data #165)."""

from functools import lru_cache
from pathlib import Path

import json
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
# Everything the challenge ships lives under data/.
DATA = ROOT / "data"

IMAGE_SIZE = 256
NUM_CLASSES = 54  # organ ids 1..54; 0 is background
N_ANNOTATED = 800  # images [0, 800) carry masks
N_TRAIN = 2000

LABEL_CSV = DATA / "label_Hnl61pT.csv"
ANNOTATED_JSON = DATA / "annotated_labels.json"
ORGAN_NAMES_JSON = DATA / "organ_names.json"


def _load_images(directory: Path) -> np.ndarray:
    # Numerical order matters: column i of the label matrix is image i.png.
    files = sorted(directory.glob("*.png"), key=lambda p: int(p.stem))
    return np.stack([np.array(Image.open(f)) for f in files], axis=0)


def _cached(name: str, build):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path)
    array = build()
    np.save(path, array)
    return array


def train_images() -> np.ndarray:
    """(2000, 256, 256) uint8."""
    return _cached("train_images", lambda: _load_images(DATA / "train-images"))


def test_images() -> np.ndarray:
    """(500, 256, 256) uint8."""
    return _cached("test_images", lambda: _load_images(DATA / "test-images"))


def train_labels() -> np.ndarray:
    """(2000, 256, 256) uint8. Images 800+ are all-zero (unannotated)."""

    def build():
        df = pd.read_csv(LABEL_CSV, index_col=0)  # (65536, 2000) on disk
        return df.T.values.reshape((-1, IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)

    return _cached("train_labels", build)


def annotated_ids() -> np.ndarray:
    """(2000, 54) bool: whether organ c+1 was annotated on image i."""
    raw = json.loads(ANNOTATED_JSON.read_text())
    mask = np.zeros((len(raw), NUM_CLASSES), dtype=bool)
    for i, ids in enumerate(raw):
        for c in ids:
            mask[i, c - 1] = True
    return mask


@lru_cache(maxsize=1)
def organ_names() -> dict:
    """organ_names.json, or empty dicts if absent. Read-only, callers share it."""
    if not ORGAN_NAMES_JSON.exists():
        return {"names": {}, "confidence": {}, "hints": {}}
    return json.loads(ORGAN_NAMES_JSON.read_text())


def organ_label(c: int, with_id: bool = True) -> str:
    """Short label for organ id `c`. A trailing '?' marks an inferred name."""
    meta = organ_names()
    name = meta["names"].get(str(c))
    if name is None:
        return f"{c}" if with_id else "?"
    if meta["confidence"].get(str(c)) != "sure":
        name += "?"
    return f"{c} {name}" if with_id else name


def train_images_hu() -> np.ndarray:
    """(2000, 256, 256) int16 Hounsfield. Run `tools/make_raw.py` first."""
    return _cached("train_images_hu", lambda: _load_hu(DATA / "train-images-raw"))


def test_images_hu() -> np.ndarray:
    """(500, 256, 256) int16 in Hounsfield units."""
    return _cached("test_images_hu", lambda: _load_hu(DATA / "test-images-raw"))


def _load_hu(directory: Path) -> np.ndarray:
    files = sorted(directory.glob("*.npy"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"{directory} is empty — run `uv run python tools/make_raw.py`")
    return np.stack([np.load(f) for f in files], axis=0)


def window(hu: np.ndarray, level: float, width: float) -> np.ndarray:
    """Clip to a CT window and rescale to [0, 1]."""
    lo, hi = level - width / 2, level + width / 2
    return np.clip((hu.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def class_areas() -> np.ndarray:
    """(54,) mean pixel area of each organ, over the images where it appears."""

    def build():
        Y = train_labels()[:N_ANNOTATED]
        out = np.zeros(NUM_CLASSES, dtype=np.float64)
        for c in range(1, NUM_CLASSES + 1):
            areas = [(Y[i] == c).sum() for i in range(N_ANNOTATED) if (Y[i] == c).any()]
            out[c - 1] = np.mean(areas) if areas else 0.0
        return out

    return _cached("class_areas", build)


def area_weights(power: float = 0.5, clip: tuple = (0.5, 4.0)) -> np.ndarray:
    """(54,) loss weights that lift small organs, normalised to mean 1."""
    areas = class_areas()
    ref = np.median(areas[areas > 0])
    w = np.where(areas > 0, (ref / np.maximum(areas, 1.0)) ** power, 1.0)
    w = np.clip(w, *clip)
    return w / w.mean()


def splits(n_val: int = 160, seed: int = 0, group: bool = False):
    """Split the 800 annotated images into train/val indices."""
    rng = np.random.default_rng(seed)
    if not group:
        idx = rng.permutation(N_ANNOTATED)
        return idx[n_val:], idx[:n_val]

    from radium import twins

    gid = twins.groups(N_ANNOTATED, twins.leak_pairs())
    order = rng.permutation(np.unique(gid))
    members = {g: np.where(gid == g)[0] for g in order}
    val = []
    for g in order:
        if len(val) >= n_val:
            break
        val.extend(members[g].tolist())
    val = np.array(sorted(val))
    train = np.setdiff1d(np.arange(N_ANNOTATED), val)
    return train, val
