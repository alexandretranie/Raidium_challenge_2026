# Raidium Challenge 2026

Multi-organ **segmentation** on 2D CT slices — ENS Challenge Data,
[challenge 165](https://challengedata.ens.fr/challenges/165). Predict, for each
256×256 slice, a label map over 54 anatomical structures (0 = background).

Results, experiments and their verdicts: **`notes.tex`** (build in `out/`).
What each trained model is: **`CHECKPOINTS.md`**.

## Setup

```bash
uv sync          # Python 3.12, PyTorch 2.13, and the radium package itself
```

Everything below runs from the repository root.

## Data

Not versioned. Download from the challenge platform into `data/`:

```
data/
├── label_Hnl61pT.csv       # masks, 65536 pixels x 2000 images
├── annotated_labels.json   # which organs were annotated, per image
├── train-images/           # 2000 slices — the first 800 are annotated
└── test-images/            # 500 slices
```

First access caches everything as `.npy` under `.cache/`.

```bash
uv run python tools/make_raw.py        # optional: HU slices, for --input hu
```

## Train

```bash
uv run python tools/train.py --epochs 60
uv run python tools/train.py --arch dpt --epochs 60                 # DINOv3 + DPT
uv run python tools/train.py --size 384                             # internal resolution
uv run python tools/train.py --oversample 0.5                       # resample rare organs
uv run python tools/train.py --external external-data/external145 --epochs 8
uv run python tools/train.py --init checkpoints/pre145_256.pt --lr 1e-4
```

`--out` defaults to `checkpoints/best.pt`; `--help` lists the rest.

## Predict and submit

```bash
uv run python tools/predict.py --ckpt checkpoints/ft145_256.pt --min-size 0
uv run python tools/ensemble.py --uniform --min-size 0 \
    --ckpt checkpoints/a.pt,checkpoints/b.pt --submit submissions/out.csv
```

The command reproducing the submission of record is in `CHECKPOINTS.md`.

## Evaluate and look

```bash
uv run python tools/eval_small.py --prob --ckpt checkpoints/ft145_256.pt
uv run python tools/tune_thresholds.py --ckpt checkpoints/ft145_256.pt
uv run python tools/viz.py --ckpt checkpoints/ft145_256.pt --n 8
uv run jupyter lab notebook.ipynb        # slices, masks, zooms, predictions
```

## External data

```bash
# 28 GB, Zenodo record 6802614 — Totalsegmentator_dataset.zip (v1, not v2)
uv run python tools/make_external.py --root <dataset> \
    --out external-data/external145 --limit 5           # pilot on 5 cases first
uv run python tools/make_external.py --root <dataset> \
    --out external-data/external145 --check-overlap
```

## Pseudo-labels

```bash
uv run python tools/bench_pseudo.py --ckpt checkpoints/best_dinov3.pt
uv run python tools/make_pseudo_pixels.py --ckpt checkpoints/best_dinov3.pt --run dinov3
uv run python tools/train.py --pseudo dinov3 --epochs 60
```

## Multi-run experiments

Each script prints its own comparison table at the end.

```bash
./scripts/run_external.sh          # pretrain on external data, then fine-tune
./scripts/run_dinov3.sh            # same with DINOv3, then ensemble
./scripts/run_scale512.sh          # pixel spacing, then resolution
./scripts/run_small_organs.sh      # resolution and resampling arms
./scripts/run_ab.sh                # baseline / pseudo-labels / resampling
./scripts/run_students.sh vits     # retrain one ensemble member
./scripts/run_scale512.sh table    # any script: comparison only, no training
```

## Layout

```
├── radium/                 the library, installed into the venv by uv sync
├── tools/                  entry points, one script per job
├── scripts/                multi-run experiments
├── data/                   the challenge files
├── checkpoints/            trained models — see CHECKPOINTS.md
├── submissions/            generated CSVs
├── external-data/          the Zenodo archive and the translated sets
└── figures/, out/          generated: visualisations, LaTeX build of notes.tex
```

Everything outside `radium/`, `tools/`, `scripts/` and the three documents is
generated or downloaded, and is not versioned.

## Report

```bash
latexmk -pdf -outdir=out notes.tex
```
