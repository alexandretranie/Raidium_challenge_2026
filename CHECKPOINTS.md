# Checkpoints

All of them live in `checkpoints/`.

Scores are `dice_score_annotated` on the 160 validation slices of
`data.splits()` — the fair signal, see the README. Reproduce any row with
`uv run python tools/eval_small.py --prob --ckpt checkpoints/<file>`.

Read them with the discount measured across five submissions: a validation
number obtained by early stopping is optimistic by about **0.011**, and by
**0.027** once several such models are averaged. See `notes.tex` §Results.

## The submission of record — 0.7612 on the leaderboard

A uniform average of the seven models below, `min_size = 0`. Two architectures, two
pixel scales; that diversity is what the leaderboard rewarded, not the individual
quality of any member.

All seven are on disk, but two were deleted in a cleanup and retrained from
`pre145_dinov3.pt`. Same recipe, different decoder initialisation and shuffling,
so the command below reproduces **0.7787** rather than the 0.7796 originally
measured — the two submissions differed on 0.20 % of the test pixels when
compared at the time. The CSV that actually scored 0.7612 is kept verbatim as
`submissions/submission_1st_0.7612.csv`.

The lesson is in that pair. They came out of a sweep that read as a null result
— +0.005 on a single model, nothing at all in the ensemble on validation — so
they were deleted as clutter. The leaderboard then gave them +0.0018, which is
the entire margin between 0.7612 and 0.7594. Validation could not see it.

| file | val | what it is |
|---|---|---|
| `ft145_256.pt` | 0.7584 | ResNet-34, external 1.45 mm/px, fine-tuned. Best single model. |
| `ft_lr1e4.pt` | 0.7522 | ResNet-34, external 1.20 mm/px, fine-tuned at lr 1e-4 |
| `ft_lr3e4.pt` | 0.7519 | same at lr 3e-4 — the learning rate is noise here |
| `ft145_512.pt` | 0.6786 | 512 internal resolution. Weak alone, still useful in the average |
| `ft145_dinov3.pt` | 0.7518 | DINOv3 + DPT. **Scores below the best ResNet-34 and is the most valuable member**: adding it was worth +0.016 on the leaderboard |
| `ft145_dinov3_e03.pt` | 0.7539 | same, `encoder-lr-scale 0.3` |
| `ft145_dinov3_e10.pt` | 0.7531 | same, `encoder-lr-scale 1.0`. The two arms read as noise on validation and were worth +0.0018 on the leaderboard |

```bash
uv run python tools/ensemble.py --uniform --min-size 0 \
    --ckpt checkpoints/ft145_256.pt,checkpoints/ft_lr1e4.pt,checkpoints/ft_lr3e4.pt,\
checkpoints/ft145_512.pt,checkpoints/ft145_dinov3.pt,\
checkpoints/ft145_dinov3_e03.pt,checkpoints/ft145_dinov3_e10.pt \
    --submit submissions/submission.csv
```

## Pretraining bases

Fine-tuning any of these on the 640 challenge slices takes ~15 min (ResNet-34)
or ~40 min (DINOv3), so they are kept rather than the hours it took to make them.

| file | val | what it is |
|---|---|---|
| `pre145_dinov3.pt` | 0.7233 | DINOv3, external 1.45. 4 h to rebuild |
| `pretrain_ext2.pt` | 0.6803 | ResNet-34, external 1.20. Parent of `ft_lr*` |
| `pre145_256.pt` | 0.6791 | ResNet-34, external 1.45. Parent of `ft145_*` |

## Historical references, kept because the report cites them

| file | val | what it is |
|---|---|---|
| `pseudo_os_resnet34.pt` | 0.5235 | best model before external data |
| `best_dinov3.pt` | 0.4855 | DINOv3 on the challenge data alone |
| `best_resnet34.pt` | 0.4474 | the baseline. Default of `tools/predict.py` |
| `ctrl_resnet34.pt` | 0.4229 | same recipe, controlled 60 epochs — the honest baseline |

## Deleted on purpose

The `small_*` resolution arms, `best_hu`, `best_smallorgans`,
`best_dinov3_frozen`, the `abl_*` ablations, and the ViT members of the old
0.5226 ensemble. All are measured, written up in `notes.tex`, and superseded.
Deleting a checkpoint whose number is recorded costs nothing; deleting the number
would — which is why `ft145_dinov3_e03/e10` were retrained after being dropped.

`tools/submit.py` also writes `bias_ensemble.npy` here; it is a bias vector, not a
model.
