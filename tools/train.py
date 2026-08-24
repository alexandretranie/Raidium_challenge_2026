"""Training: U-Net or DINOv3+DPT, marginal loss, on the 800 partially annotated slices."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from radium import data, metric
from radium.dataset import loaders, n_channels
from radium.losses import MarginalLoss
from radium.models import DEFAULT_ENCODER, build_model, param_groups


@torch.no_grad()
def evaluate(model, loader, val_idx, device):
    model.eval()
    preds = []
    for img, *_ in loader:
        logits = model(img.to(device))
        preds.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
    pred = np.concatenate(preds)

    gt = data.train_labels()[val_idx]
    ann = data.annotated_ids()[val_idx]
    # Fair signal: val ground truth is partial too, so score only what was annotated.
    fair = metric.dice_score_annotated(gt, pred, ann)
    # What the raw challenge metric would say on this partial ground truth.
    raw = metric.dice_score(gt, pred)
    return fair, raw, pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--arch", default="unet", choices=["unet", "dpt"],
                   help="unet: ImageNet CNN encoder; dpt: DINOv3 ViT + DPT decoder")
    p.add_argument("--encoder", default=None,
                   help=f"timm/smp encoder name; defaults per arch: {DEFAULT_ENCODER}")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="train the decoder only, on frozen pretrained features")
    p.add_argument("--encoder-lr-scale", type=float, default=0.1,
                   help="encoder LR as a fraction of --lr (ignored if frozen)")
    p.add_argument("--input", default="png", choices=["png", "hu"],
                   help="png: 8-bit slice, 1 channel; hu: 3 CT windows from train-images-raw")
    p.add_argument("--size", type=int, default=256,
                   help="internal resolution; the slice is upsampled to it and the "
                        "logits come back to 256. Adds no information, only stride: "
                        "at 512 a 37 px adrenal covers 148 px")
    p.add_argument("--w-dice", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5, help="Tversky weight on false positives")
    p.add_argument("--beta", type=float, default=0.5, help="Tversky weight on false negatives")
    p.add_argument("--area-weights", action="store_true",
                   help="weight the overlap term to lift small organs")
    p.add_argument("--flip", type=float, default=0.0, metavar="P",
                   help="probability of the left-right mirror (vertical here); "
                        "swaps the 16 lateral pairs, drops the 5 unpaired organs")
    p.add_argument("--supervision", default="marginal",
                   choices=["marginal", "naive", "present-only", "self-train"],
                   help="ablation: what the model is told about unannotated organs")
    p.add_argument("--pseudo", default=None, metavar="RUN",
                   help="folder under pseudo-labels/; adds pixel pseudo-labels and "
                        "the 1200 raw slices. Replaces --supervision.")
    p.add_argument("--tau-cell", type=float, default=None,
                   help="accept an organ in a slice above this peak probability")
    p.add_argument("--tau-pix", type=float, default=None,
                   help="write a pixel of an accepted organ above this confidence")
    p.add_argument("--tau-bg", type=float, default=None,
                   help="below this, a background pixel is dropped instead of trusted")
    p.add_argument("--oversample", type=float, default=0.0, metavar="POWER",
                   help="resample toward slices carrying rare organs; 0.5 is a "
                        "good start, 0 keeps uniform shuffling")
    p.add_argument("--group-split", action="store_true",
                   help="keep near-duplicate slices on one side of the split")
    p.add_argument("--external", default=None, metavar="DIR",
                   help="pretrain on a set built by make_external.py (complete "
                        "supervision); replaces the 640 challenge slices, keeps "
                        "the same validation")
    p.add_argument("--init", default=None, metavar="CKPT",
                   help="start from these weights — the fine-tuning half of an "
                        "--external run")
    p.add_argument("--num-workers", type=int, default=0,
                   help="dataloader workers. 0 is right for the 640 challenge "
                        "slices, which sit in RAM; an external epoch is 3600 "
                        "steps off a memory map and starves the GPU without them")
    p.add_argument("--out", default="checkpoints/best.pt")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    hu = args.input == "hu"
    gate = {k: v for k, v in (("tau_cell", args.tau_cell), ("tau_pix", args.tau_pix),
                              ("tau_bg", args.tau_bg)) if v is not None}
    train_loader, val_loader, val_idx, sample_info = loaders(
        batch_size=args.batch_size, supervision=args.supervision, hu=hu,
        flip=args.flip, pseudo_run=args.pseudo, gate=gate, group=args.group_split,
        oversample=args.oversample, external=args.external,
        num_workers=args.num_workers,
    )

    weights = None
    if args.area_weights:
        weights = torch.from_numpy(data.area_weights()).float()
        print(f"area weights: min {weights.min():.2f}  max {weights.max():.2f}")

    encoder = args.encoder or DEFAULT_ENCODER[args.arch]
    model = build_model(args.arch, encoder, freeze_encoder=args.freeze_encoder,
                        in_channels=n_channels(hu), size=args.size).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location=device)
        state = ckpt["model"]
        wrapped_ckpt = next(iter(state)).startswith("net.")
        wrapped_model = next(iter(model.state_dict())).startswith("net.")
        if wrapped_ckpt != wrapped_model:
            state = ({k[4:]: v for k, v in state.items()} if wrapped_ckpt
                     else {f"net.{k}": v for k, v in state.items()})
        model.load_state_dict(state)
        print(f"initialised from {args.init} (arch={ckpt.get('arch')}, "
              f"size={ckpt.get('size', 256)} -> {args.size})")
    criterion = MarginalLoss(
        w_ce=1.0, w_dice=args.w_dice,
        alpha=args.alpha, beta=args.beta, weights=weights,
    ).to(device)
    opt = torch.optim.AdamW(
        param_groups(model, args.lr, args.encoder_lr_scale), weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"device={device}  train={len(train_loader.dataset)}  val={len(val_idx)}")
    if args.external:
        print(f"external: {args.external}, complete supervision on all 54 organs; "
              f"validation unchanged (160 challenge slices)")
    if args.pseudo:
        print(f"pseudo-labels: {args.pseudo}  gate={gate or 'default'}")
    if args.oversample:
        f, pc, w = sample_info["freq"], sample_info["per_class"], sample_info["weights"]
        rare = np.argsort(f)[:5]
        print(f"oversample^{args.oversample}: weights {w.min():.2f}-{w.max():.2f}, "
              f"{sample_info['n_zero']} slices dropped (no supervision)")
        print("  rarest: " + ", ".join(
            f"{c+1}={int(f[c])}x{pc[c]:.1f}" for c in rare))
    print(f"{args.arch}/{encoder}  input={args.input} ({n_channels(hu)}ch)  "
          f"size={args.size}  {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()  # model.train() would re-enable dropout/norm updates
        t0, running = time.time(), 0.0
        for img, mask, ann, w in train_loader:
            img, mask, ann = img.to(device), mask.to(device), ann.to(device)
            pw = w.to(device) if args.pseudo else None
            loss = criterion(model(img), mask, ann, pixel_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()

        fair, raw, _ = evaluate(model, val_loader, val_idx, device)
        flag = ""
        if fair > best:
            best, flag = fair, "  <- best"
            torch.save(
                {"model": model.state_dict(), "arch": args.arch, "encoder": encoder,
                 "input": args.input, "pseudo": args.pseudo, "size": args.size},
                args.out,
            )
        print(
            f"epoch {epoch+1:>3}/{args.epochs}  loss {running/len(train_loader):.4f}  "
            f"dice(annotated) {fair:.4f}  dice(raw) {raw:.4f}  {time.time()-t0:.0f}s{flag}"
        )

    print(f"\nbest dice(annotated) = {best:.4f}  -> {args.out}")


if __name__ == "__main__":
    main()
