import argparse
from pathlib import Path

import numpy as np
import torch

from radium import data, metric
from radium.infer import drop_small, load_model, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best_resnet34.pt")
    p.add_argument("--min-size", type=int, default=None,
                   help="component threshold; omit to sweep on validation")
    p.add_argument("--out", default="submissions/submission.csv")
    p.add_argument("--device", default=None,
                   help="force a device. Worth 'cpu' while a training run holds "
                        "the GPU: inference is a few minutes either way, and two "
                        "MPS processes on 8 GB is how a run gets killed")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    model, hu = load_model(args.ckpt, device)

    _, val_idx = data.splits()
    gt = data.train_labels()[val_idx]
    ann = data.annotated_ids()[val_idx]
    images = data.train_images_hu() if hu else data.train_images()
    val_pred = predict(model, images[val_idx], device, hu=hu)

    best_t = args.min_size
    if best_t is None:
        print(f"{'min_size':>9} {'dice(annotated)':>16} {'dice(raw)':>11}")
        best_score = -1.0
        for t in [0, 10, 25, 50, 100, 200, 400, 800]:
            cleaned = drop_small(val_pred, t)
            fair = metric.dice_score_annotated(gt, cleaned, ann)
            raw = metric.dice_score(gt, cleaned)
            flag = ""
            if fair > best_score:
                best_score, best_t, flag = fair, t, "  <-"
            print(f"{t:>9} {fair:>16.4f} {raw:>11.4f}{flag}")
        print(f"\nchosen min_size = {best_t}")

    test_images = data.test_images_hu() if hu else data.test_images()
    test_pred = drop_small(predict(model, test_images, device, hu=hu), best_t)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    metric.to_submission(test_pred).to_csv(args.out)
    covered = sorted(np.unique(test_pred))[1:]
    print(f"\nwrote {args.out}  shape=(65536, {len(test_pred)})")
    print(f"classes predicted on test: {len(covered)}/54")


if __name__ == "__main__":
    main()
