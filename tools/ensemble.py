"""Search ensemble weights on validation, then write a submission."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from radium.infer import drop_small
from radium import data, metric
from radium.dataset import to_channels
from radium.models import load_checkpoint

MIN_SIZES = (0, 10, 25, 50, 100, 200)


def simplex_grid(n, step):
    """Weight vectors on the n-simplex, in multiples of `step`."""
    k = int(round(1 / step))
    out = []
    for cut in itertools.combinations_with_replacement(range(n), k):
        w = np.bincount(cut, minlength=n).astype(np.float64)
        out.append(w / w.sum())
    return out


def candidates(n, extra=None):
    """Identities first, then a grid coarse enough to stay one pass."""
    cands = [np.eye(n)[i] for i in range(n)]
    cands.append(np.ones(n) / n)
    step = {1: 1.0, 2: 0.05, 3: 0.1, 4: 0.2}.get(n, 0.25)
    cands += simplex_grid(n, step)
    if extra is not None:
        cands.append(np.asarray(extra, dtype=np.float64) / np.sum(extra))
    uniq, seen = [], set()
    for w in cands:
        key = tuple(np.round(w, 4))
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    return uniq


@torch.no_grad()
def stacked_probs(models, indices, images, device, chunk=8):
    """Yield (start, (n_models, B, 55, H, W)) so weights are swept in one pass."""
    for k in range(0, len(indices), chunk):
        sel = indices[k : k + chunk]
        per = []
        for model, hu in models:
            x = torch.from_numpy(to_channels(images[hu][sel], hu)).to(device)
            per.append(model((x - 0.5) / 0.25).softmax(1))
        yield k, torch.stack(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="comma-separated checkpoints")
    ap.add_argument("--weights", default=None,
                    help="also try this weighting, e.g. 0.3,0.3,0.2,0.2")
    ap.add_argument("--submit", default=None, metavar="CSV",
                    help="write a submission with the winning weights")
    ap.add_argument("--min-size", type=int, default=None,
                    help="override the swept component threshold. Worth doing: "
                         "validation scores only the 27 organs declared per image, "
                         "the test set scores all 54, so a stray component costs "
                         "about twice there what it costs here and the sweep is "
                         "biased toward keeping too much")
    ap.add_argument("--top", type=int, default=8, help="rows to print")
    ap.add_argument("--uniform", action="store_true",
                    help="skip the search, average the members equally. Measured "
                         "the hard way: every weighting picked on these 160 images "
                         "gained on them and nothing on the leaderboard, three "
                         "times over. A uniform average has no parameter to overfit")
    args = ap.parse_args()

    paths = [p.strip() for p in args.ckpt.split(",") if p.strip()]
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    models = [load_checkpoint(p, device) for p in paths]
    n = len(models)

    extra = [float(x) for x in args.weights.split(",")] if args.weights else None
    if extra and len(extra) != n:
        raise SystemExit(f"--weights has {len(extra)} entries for {n} checkpoints")
    cands = [np.ones(n) / n] if args.uniform else candidates(n, extra)

    _, va = data.splits()
    gt, ann = data.train_labels()[va], data.annotated_ids()[va]
    train_png, train_hu = None, None
    need_hu = any(hu for _, hu in models)
    images = {False: data.train_images(), True: data.train_images_hu() if need_hu else None}

    print(f"{n} members, {len(cands)} weightings, {len(va)} validation slices")
    W = torch.tensor(np.stack(cands), dtype=torch.float32, device=device)  # (C, n)
    preds = np.zeros((len(cands), len(va), data.IMAGE_SIZE, data.IMAGE_SIZE), np.uint8)

    idx = np.arange(len(va))
    for k, P in stacked_probs(models, va, images, device):
        # (C, n) x (n, B, 55, H, W) -> (C, B, 55, H, W), argmax over classes.
        fused = torch.einsum("cn,nbkhw->cbkhw", W, P).argmax(2)
        preds[:, k : k + fused.shape[1]] = fused.to(torch.uint8).cpu().numpy()

    scored = []
    for i, w in enumerate(cands):
        scored.append((metric.dice_score_annotated(gt, preds[i], ann), i))
    scored.sort(reverse=True)

    label = lambda w: " ".join(f"{x:.2f}" for x in w)
    print(f"\n{'score':>8}  weights ({', '.join(paths)})")
    for s, i in scored[: args.top]:
        tag = ""
        if np.count_nonzero(cands[i]) == 1:
            tag = "  <- alone"
        elif np.allclose(cands[i], 1 / n):
            tag = "  <- uniform"
        print(f"{s:>8.4f}  {label(cands[i])}{tag}")

    best_s, best_i = scored[0]
    best_w = cands[best_i]
    if not args.uniform:
        uni = next(s for s, i in scored if np.allclose(cands[i], 1 / n))
        solo = max(s for s, i in scored if np.count_nonzero(cands[i]) == 1)
        print(f"\nbest {best_s:.4f}  vs uniform {uni:.4f} (+{best_s-uni:.4f})"
              f"  vs best single {solo:.4f} (+{best_s-solo:.4f})")

    print(f"\n{'min_size':>9} {'dice':>8}")
    best_t, best_final = 0, -1.0
    for t in MIN_SIZES:
        s = metric.dice_score_annotated(gt, drop_small(preds[best_i], t), ann)
        flag = ""
        if s > best_final:
            best_final, best_t, flag = s, t, "  <-"
        print(f"{t:>9} {s:>8.4f}{flag}")
    if args.min_size is not None and args.min_size != best_t:
        forced = metric.dice_score_annotated(gt, drop_small(preds[best_i], args.min_size), ann)
        print(f"\noverriding min_size {best_t} -> {args.min_size} "
              f"({best_final:.4f} -> {forced:.4f} on validation, which under-counts "
              f"the cost of a stray component)")
        best_t, best_final = args.min_size, forced
    print(f"\nfinal {best_final:.4f}  weights [{label(best_w)}]  min_size {best_t}")

    if args.submit:
        te = {False: data.test_images(), True: data.test_images_hu() if need_hu else None}
        out = np.zeros((len(te[False]), data.IMAGE_SIZE, data.IMAGE_SIZE), np.uint8)
        ti = np.arange(len(te[False]))
        Wb = torch.tensor(best_w, dtype=torch.float32, device=device)
        for k, P in stacked_probs(models, ti, te, device):
            out[k : k + P.shape[1]] = (
                torch.einsum("n,nbkhw->bkhw", Wb, P).argmax(1).to(torch.uint8).cpu().numpy()
            )
        out = drop_small(out, best_t)
        Path(args.submit).parent.mkdir(parents=True, exist_ok=True)
        metric.to_submission(out).to_csv(args.submit)
        print(f"wrote {args.submit}  classes predicted {len(np.unique(out))-1}/54")


if __name__ == "__main__":
    main()
