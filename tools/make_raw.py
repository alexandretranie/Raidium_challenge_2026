"""Rebuild Hounsfield-calibrated slices into train-images-raw/ and test-images-raw/."""

import json
from pathlib import Path

import numpy as np

from radium import data

ROOT = Path(__file__).resolve().parent
AIR_HU, LIVER_HU = -1000.0, 40.0
LIVER_CLASS = 36


def calibrate(X, Y):
    """Fit the 8-bit -> HU affine map on two physical anchors."""
    air = float(np.median([np.percentile(im, 2) for im in X[:300]]))
    liver = float(np.median([X[i][Y[i] == LIVER_CLASS].mean()
                             for i in range(data.N_ANNOTATED)
                             if (Y[i] == LIVER_CLASS).any()]))
    slope = (LIVER_HU - AIR_HU) / (liver - air)
    return air, slope


def to_hu(img, air, slope):
    return np.rint(AIR_HU + (img.astype(np.float32) - air) * slope).astype(np.int16)


def write_split(images, out_dir, air, slope):
    out_dir.mkdir(exist_ok=True)
    for i, im in enumerate(images):
        np.save(out_dir / f"{i}.npy", to_hu(im, air, slope))
    return len(images)


def main():
    X, Y, Xt = data.train_images(), data.train_labels(), data.test_images()
    air, slope = calibrate(X, Y)
    print(f"calibration: air {air:.1f} -> {AIR_HU:.0f} HU, "
          f"liver {LIVER_HU:+.0f} HU, slope {slope:.3f} HU per grey level")

    n_tr = write_split(X, data.DATA / "train-images-raw", air, slope)
    n_te = write_split(Xt, data.DATA / "test-images-raw", air, slope)

    meta = {
        "note": "HU-calibrated, NOT original scans: 8-bit quantisation is lossy",
        "formula": "HU = -1000 + (px - air) * slope",
        "air": air, "slope": slope,
        "hu_per_grey_level": slope,
        "dtype": "int16", "shape": [data.IMAGE_SIZE, data.IMAGE_SIZE],
        "n_train": n_tr, "n_test": n_te,
    }
    (data.DATA / "hu_calibration.json").write_text(json.dumps(meta, indent=2))

    # Sanity: tissues never used to fit the map must land on physiological values.
    print(f"\n{'tissue':<22} {'measured':>10}   expected")
    checks = [("lung (cls 37)", 37, "-700 to -850"), ("lung (cls 38)", 38, "-700 to -850"),
              ("fat (cls 9)", 9, "-100 to -50"), ("heart (cls 22)", 22, "+30 to +50"),
              ("liver (anchor)", 36, "+40 by construction")]
    for name, c, expected in checks:
        px = np.median([X[i][Y[i] == c].mean() for i in range(data.N_ANNOTATED)
                        if (Y[i] == c).any()])
        print(f"{name:<22} {to_hu(np.array([px]), air, slope)[0]:>+10}   {expected}")

    a = np.load(data.DATA / "train-images-raw/0.npy")
    print(f"\nwrote train-images-raw/ ({n_tr} files), test-images-raw/ ({n_te})")
    print(f"sample 0.npy: {a.dtype} {a.shape}, {a.min()} to {a.max()} HU")


if __name__ == "__main__":
    main()
