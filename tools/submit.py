"""Final submission: DINOv3+resnet34 ensemble with cross-validated logit biases."""

import argparse
from pathlib import Path

import numpy as np
import torch

from radium import data, metric
from radium.dataset import to_channels
from radium.infer import load_model

# Architectural diversity is what pays; see the README.
MODELS = [("checkpoints/best_dinov3.pt", 0.5), ("checkpoints/best_resnet34.pt", 0.5)]
BIAS_PATH = Path("checkpoints/bias_ensemble.npy")


def ensemble_logprobs(images, device="mps", batch=8):
    """Weighted probability average across models, returned in log space."""
    loaded = [(load_model(ck, device), w) for ck, w in MODELS]
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            acc = None
            for (model, hu), w in loaded:
                x = torch.from_numpy(to_channels(images[i:i + batch], hu)).float()
                x = ((x - 0.5) / 0.25).to(device)
                p = model(x).softmax(1) * w
                acc = p if acc is None else acc + p
            out.append(torch.log(acc.clamp_min(1e-9)).cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def tune_bias(L, Y, A, sel, iters=2):
    """Coordinate ascent on the challenge metric itself."""
    def score(b):
        return metric.dice_score_annotated(
            Y[sel], (L[sel] + b[None, :, None, None]).argmax(1).astype(np.uint8), A[sel])

    b = np.zeros(55, np.float32)
    best_t, best_s = 0.0, score(b)
    for t in (0.5, 1.0, 1.5, 2.0, 2.5):
        bb = b.copy(); bb[0] = -t
        s = score(bb)
        if s > best_s:
            best_t, best_s = t, s
    b[0], cur = -best_t, best_s
    for _ in range(iters):
        for c in range(1, 55):
            bv, bs = b[c], cur
            for v in (b[c] - 1.0, b[c] - 0.5, b[c] + 0.5, b[c] + 1.0):
                bb = b.copy(); bb[c] = v
                s = score(bb)
                if s > bs:
                    bv, bs = v, s
            b[c], cur = bv, bs
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submissions/submission.csv")
    ap.add_argument("--skip-tune", action="store_true", help="reuse the saved bias vector")
    args = ap.parse_args()

    _, vi = data.splits()
    Y, A, X = data.train_labels()[vi], data.annotated_ids()[vi], data.train_images()

    Lv = ensemble_logprobs(X[vi])
    plain = metric.dice_score_annotated(Y, Lv.argmax(1).astype(np.uint8), A)
    print(f"ensemble, no bias : {plain:.4f}")

    if args.skip_tune and BIAS_PATH.exists():
        bias = np.load(BIAS_PATH)
    else:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(vi))
        fA, fB = perm[:80], perm[80:]
        bA = tune_bias(Lv, Y, A, fA)
        bB = tune_bias(Lv, Y, A, fB)

        def sc(sel, b):
            return metric.dice_score_annotated(
                Y[sel], (Lv[sel] + b[None, :, None, None]).argmax(1).astype(np.uint8), A[sel])

        zero = np.zeros(55, np.float32)
        honest = ((sc(fA, bB) - sc(fA, zero)) + (sc(fB, bA) - sc(fB, zero))) / 2
        print(f"bias gain, cross-validated : {honest:+.4f}")
        # Deploy the average: each half was fitted without the other's data.
        bias = (bA + bB) / 2
        np.save(BIAS_PATH, bias)

    tuned = metric.dice_score_annotated(
        Y, (Lv + bias[None, :, None, None]).argmax(1).astype(np.uint8), A)
    pred_v = (Lv + bias[None, :, None, None]).argmax(1).astype(np.uint8)
    print(f"ensemble + bias   : {tuned:.4f}  "
          f"({54 - (len(np.unique(pred_v)) - 1)}/54 silent)")

    Xt = data.test_images()
    preds = []
    for i in range(0, len(Xt), 64):
        Lt = ensemble_logprobs(Xt[i:i + 64])
        preds.append((Lt + bias[None, :, None, None]).argmax(1).astype(np.uint8))
    pred = np.concatenate(preds)

    metric.to_submission(pred).to_csv(args.out)
    covered = len(np.unique(pred)) - 1
    print(f"\n{args.out}: {pred.shape[0]} images, {covered}/54 classes covered, "
          f"{np.mean([len(np.unique(p)) - 1 for p in pred]):.1f} organs per slice")


if __name__ == "__main__":
    main()
