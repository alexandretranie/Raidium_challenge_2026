"""Turn the TotalSegmentator v1 dataset into slices in the challenge's format."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from radium import data, totalseg

# Measured through paired within-slice distances, not organ area (confounded).
TARGET_SPACING = 1.45
SIZE = 256
PLANES = ("axial", "coronal", "sagittal")
PLANE_P = (0.81, 0.12, 0.07)   # 19% reformats, measured on the 800 annotated


def _nib():
    try:
        import nibabel
    except ImportError as e:  # noqa: F841
        raise SystemExit(
            "nibabel is required to read NIfTI volumes:\n"
            "    uv add nibabel"
        )
    return nibabel


def load_case(case: Path):
    """-> (hu volume, challenge-id label volume, spacing), both (X, Y, Z)."""
    nib = _nib()
    ct = nib.load(str(case / "ct.nii.gz"))
    hu = np.asanyarray(ct.dataobj).astype(np.int16)
    spacing = np.array(ct.header.get_zooms()[:3], dtype=np.float32)

    seg = np.zeros(hu.shape, dtype=np.uint8)
    seg_dir = case / "segmentations"
    for cid, sources in totalseg.SOURCES.items():
        for name in sources:
            f = seg_dir / f"{name}.nii.gz"
            if f.exists():
                m = np.asanyarray(nib.load(str(f)).dataobj) > 0
                seg[m] = cid
    return hu, seg, spacing


def _resize(plane_img, plane_lab, zoom):
    """Resample one 2D plane to the target spacing, then centre-crop or pad."""
    from scipy import ndimage

    img = ndimage.zoom(plane_img.astype(np.float32), zoom, order=1)
    lab = ndimage.zoom(plane_lab, zoom, order=0)

    out_i = np.full((SIZE, SIZE), -1000.0, dtype=np.float32)
    out_l = np.zeros((SIZE, SIZE), dtype=np.uint8)
    h, w = img.shape
    sh, sw = max(0, (h - SIZE) // 2), max(0, (w - SIZE) // 2)
    img, lab = img[sh:sh + SIZE, sw:sw + SIZE], lab[sh:sh + SIZE, sw:sw + SIZE]
    dh, dw = (SIZE - img.shape[0]) // 2, (SIZE - img.shape[1]) // 2
    out_i[dh:dh + img.shape[0], dw:dw + img.shape[1]] = img
    out_l[dh:dh + lab.shape[0], dw:dw + lab.shape[1]] = lab
    return out_i, out_l


def cut(hu, seg, spacing, plane, index, target=TARGET_SPACING):
    """One slice through `plane`, resampled and rotated to the challenge's frame."""
    if plane == "axial":
        img, lab, sp = hu[:, :, index], seg[:, :, index], spacing[[0, 1]]
    elif plane == "coronal":
        img, lab, sp = hu[:, index, :], seg[:, index, :], spacing[[0, 2]]
    else:
        img, lab, sp = hu[index, :, :], seg[index, :, :], spacing[[1, 2]]
    # No rotation: NIfTI's (L-R, A-P, S-I) order already lands in the right frame.
    return _resize(img, lab, tuple(sp / target))


def generate(case, rng, windows, per_case, target=TARGET_SPACING):
    """-> [(8-bit slice, label map)] for one volume."""
    hu, seg, spacing = load_case(case)
    out = []
    for _ in range(per_case):
        plane = rng.choice(PLANES, p=PLANE_P)
        axis = {"axial": 2, "coronal": 1, "sagittal": 0}[plane]
        # Only where there is something to see; an empty slice teaches nothing true.
        occupied = np.flatnonzero((seg > 0).any(axis=tuple(a for a in (0, 1, 2)
                                                           if a != axis)))
        if not len(occupied):
            continue
        idx = int(rng.choice(occupied))
        img, lab = cut(hu, seg, spacing, plane, idx, target)
        air, slope = windows[rng.integers(len(windows))]
        out.append((totalseg.to_8bit(img, air, slope), lab.astype(np.uint8)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="the Totalsegmentator_dataset folder")
    p.add_argument("--out", default="external", help="output folder")
    p.add_argument("--per-case", type=int, default=12, help="slices per volume")
    p.add_argument("--limit", type=int, default=None, help="pilot on the first n cases")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spacing", type=float, default=TARGET_SPACING,
                   help="target mm/px; the default was measured on the challenge slices")
    p.add_argument("--check-overlap", action="store_true",
                   help="reject any slice matching a test image")
    args = p.parse_args()

    root = Path(args.root)
    cases = sorted(d for d in root.iterdir() if (d / "ct.nii.gz").exists())
    if not cases:
        raise SystemExit(f"no case found under {root} (expected <case>/ct.nii.gz)")
    if args.limit:
        cases = cases[:args.limit]

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    windows = totalseg.quantisation_stats()
    print(f"{len(cases)} cases, {args.per_case} slices each -> "
          f"~{len(cases) * args.per_case} images at {args.spacing} mm/px")
    print(f"windows sampled from {len(windows)} challenge slices")

    ref = None
    if args.check_overlap:
        from radium import twins
        ref = twins._pooled(data.test_images())

    labels, kept, dropped, failed = [], 0, 0, []
    for k, case in enumerate(cases):
        try:
            slices = generate(case, rng, windows, args.per_case, args.spacing)
        except Exception as e:
            # One archive member (s0864/ct.nii.gz) is corrupt; do not die over it.
            failed.append((case.name, type(e).__name__))
            continue
        for img, lab in slices:
            if ref is not None:
                f = twins._pooled(img[None])
                if float((f @ ref.T).max()) > twins.CORR:
                    dropped += 1
                    continue
            Image.fromarray(img).save(out / "images" / f"{kept}.png")
            labels.append(lab)
            kept += 1
        if (k + 1) % 25 == 0 or k == len(cases) - 1:
            print(f"  {k+1}/{len(cases)} cases, {kept} slices written"
                  + (f", {dropped} rejected (near a test image)" if ref is not None else ""))

    np.save(out / "labels.npy", np.stack(labels))

    # Training reads images.npy memory-mapped; consolidate and drop the PNGs.
    stack = np.lib.format.open_memmap(out / "images.npy", mode="w+",
                                      dtype=np.uint8, shape=(kept, SIZE, SIZE))
    for i in range(kept):
        stack[i] = np.array(Image.open(out / "images" / f"{i}.png"))
    stack.flush()
    for i in range(kept):
        (out / "images" / f"{i}.png").unlink()
    (out / "images").rmdir()
    meta = {"n": kept, "size": SIZE, "spacing": args.spacing,
            "planes": dict(zip(PLANES, PLANE_P)), "source": "TotalSegmentator v1",
            "mapping": "radium/totalseg.py", "dropped_near_test": dropped,
            "failed_cases": failed,
            "note": "complete supervision: all 54 organs outlined on every slice"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if failed:
        print(f"\n{len(failed)} unreadable cases, skipped: {failed}")
    print(f"\nwrote {out}/images.npy and {out}/labels.npy — {kept} slices")
    print("complete supervision: `annotated` is True for all 54 everywhere.")


if __name__ == "__main__":
    main()
