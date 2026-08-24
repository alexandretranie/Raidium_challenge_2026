"""Translation between the challenge's 54 ids and TotalSegmentator v1's 104 classes."""

import numpy as np

# class_map["total_v1"], verbatim. Index i (1-based) is TotalSegmentator's label.
V1_CLASSES = (
    "spleen kidney_right kidney_left gallbladder liver stomach aorta "
    "inferior_vena_cava portal_vein_and_splenic_vein pancreas adrenal_gland_right "
    "adrenal_gland_left lung_upper_lobe_left lung_lower_lobe_left "
    "lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_right"
).split() + [f"vertebrae_L{i}" for i in (5, 4, 3, 2, 1)] \
  + [f"vertebrae_T{i}" for i in range(12, 0, -1)] \
  + [f"vertebrae_C{i}" for i in range(7, 0, -1)] \
  + ("esophagus trachea heart_myocardium heart_atrium_left heart_ventricle_left "
     "heart_atrium_right heart_ventricle_right pulmonary_artery brain "
     "iliac_artery_left iliac_artery_right iliac_vena_left iliac_vena_right "
     "small_bowel duodenum colon").split() \
  + [f"rib_left_{i}" for i in range(1, 13)] \
  + [f"rib_right_{i}" for i in range(1, 13)] \
  + ("humerus_left humerus_right scapula_left scapula_right clavicula_left "
     "clavicula_right femur_left femur_right hip_left hip_right sacrum face "
     "gluteus_maximus_left gluteus_maximus_right gluteus_medius_left "
     "gluteus_medius_right gluteus_minimus_left gluteus_minimus_right "
     "autochthon_left autochthon_right iliopsoas_left iliopsoas_right "
     "urinary_bladder").split()

assert len(V1_CLASSES) == 104, f"expected 104 v1 classes, got {len(V1_CLASSES)}"


def _merged(name: str) -> str:
    """v1 class name -> the challenge name that absorbs it."""
    if name.startswith("lung_"):
        return "lung_left" if name.endswith("left") else "lung_right"
    if name.startswith("vertebrae_"):
        return {"C": "vertebrae_cervical", "T": "vertebrae_thoracic",
                "L": "vertebrae_lumbar"}[name.split("_")[1][0]]
    if name.startswith("rib_"):
        return "ribs_" + name.split("_")[1]
    if name.startswith("heart"):
        return "heart"
    return name


_GROUPS: dict[str, list[str]] = {}
for _n in V1_CLASSES:
    _GROUPS.setdefault(_merged(_n), []).append(_n)

# Alphabetical order is how the challenge numbered its ids, verified on all 54.
NAMES: dict[int, str] = {i: n for i, n in enumerate(sorted(_GROUPS), start=1)}
SOURCES: dict[int, list[str]] = {i: _GROUPS[n] for i, n in NAMES.items()}

assert len(NAMES) == 54, f"expected 54 merged classes, got {len(NAMES)}"

# LUT over v1 integer labels: index 0 is background, index i the i-th v1 class.
V1_TO_CHALLENGE = np.zeros(len(V1_CLASSES) + 1, dtype=np.uint8)
for _cid, _names in SOURCES.items():
    for _n in _names:
        V1_TO_CHALLENGE[V1_CLASSES.index(_n) + 1] = _cid


def merge_labels(seg: np.ndarray) -> np.ndarray:
    """TotalSegmentator v1 multilabel volume -> challenge ids 0..54."""
    if seg.max() > len(V1_CLASSES):
        raise ValueError(
            f"label {seg.max()} exceeds v1's 104 classes — this looks like v2 "
            "output, whose ids mean different organs (see the module docstring)"
        )
    return V1_TO_CHALLENGE[seg]


def check() -> dict:
    """Cross-check the derived names against organ_names.json plus the eight
    resolved ids. Returns the disagreements, which should be none."""
    from radium import data

    known = {int(k): v for k, v in data.organ_names()["names"].items() if v}
    known.update({7: "clavicula_left", 8: "clavicula_right", 12: "face",
                  39: "pancreas", 41: "pulmonary_artery", 48: "spleen",
                  52: "vertebrae_cervical", 53: "vertebrae_lumbar",
                  54: "vertebrae_thoracic"})
    return {i: (NAMES[i], known[i]) for i in NAMES if known.get(i) != NAMES[i]}


def quantisation_stats(n: int = 800) -> np.ndarray:
    """(m, 2) the (air level, HU per grey level) of each challenge training slice."""
    from radium import data

    X = data.train_images()[:n]
    out = []
    for im in X:
        im = im.astype(np.float32)
        air = float(np.percentile(im, 2))
        body = im[im > air + 8]
        if body.size < 400:
            continue
        hist, edges = np.histogram(body, bins=48)
        mode = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
        hi = np.percentile(body, 97)
        if mode < 0.5 * (air + hi):  # lung dominates on chest slices, look higher
            v = body[body > 0.5 * (mode + hi)]
            if v.size >= 200:
                h2, e2 = np.histogram(v, bins=48)
                mode = 0.5 * (e2[h2.argmax()] + e2[h2.argmax() + 1])
        if mode - air > 1e-3:
            out.append((air, 980.0 / (mode - air)))  # air -1000 -> soft tissue -20
    return np.array(out)


def to_8bit(hu: np.ndarray, air: float, slope: float) -> np.ndarray:
    """Hounsfield -> the challenge's 8-bit convention, for one sampled window."""
    return np.clip(np.rint(air + (hu + 1000.0) / slope), 0, 255).astype(np.uint8)
