"""Segmentation backbones. Same signature so the training loop is agnostic."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_ENCODER = {
    "unet": "resnet34",
    "dpt": "tu-vit_small_patch16_dinov3.lvd1689m",
}

N_CLASSES = 55  # 54 organs + background


def _bn_to_groupnorm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            setattr(module, name, nn.GroupNorm(min(32, c), c))
        else:
            _bn_to_groupnorm(child)


class Rescaled(nn.Module):
    """Runs `net` at `size` while keeping the module's public resolution at 256."""

    def __init__(self, net, size):
        super().__init__()
        self.net = net
        self.size = size

    @property
    def encoder(self):
        return self.net.encoder  # param_groups() and --freeze-encoder reach through

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear",
                          align_corners=False)
        y = self.net(x)
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


def build_model(arch="unet", encoder=None, pretrained=True, freeze_encoder=False,
                in_channels=1, size=256):
    encoder = encoder or DEFAULT_ENCODER[arch]
    weights = "imagenet" if pretrained else None

    if arch == "unet":
        model = smp.Unet(encoder, encoder_weights=weights, in_channels=in_channels,
                         classes=N_CLASSES)
    elif arch == "dpt":
        # dynamic_img_size lets the ViT take 256x256 instead of its native 224.
        model = smp.DPT(encoder, encoder_weights=weights, in_channels=in_channels,
                        classes=N_CLASSES, dynamic_img_size=True)
        _bn_to_groupnorm(model)
    else:
        raise ValueError(f"unknown arch: {arch}")

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.encoder.eval()  # also pinned every epoch, see tools/train.py

    # 256 returns the bare model: no wrapper, so old checkpoints still load.
    return model if size == 256 else Rescaled(model, size)


def load_checkpoint(path, device):
    """-> (model, hu). Checkpoints predating these flags are 1-channel U-Nets."""
    from radium.dataset import n_channels

    ckpt = torch.load(path, map_location=device)
    hu = ckpt.get("input", "png") == "hu"
    model = build_model(ckpt.get("arch", "unet"), ckpt["encoder"], pretrained=False,
                        in_channels=n_channels(hu), size=ckpt.get("size", 256))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), hu


def param_groups(model, lr, encoder_lr_scale=0.1):
    """Lower LR on a pretrained encoder than on the randomly initialised decoder."""
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    rest = [p for p in model.parameters() if id(p) not in encoder_ids and p.requires_grad]

    groups = [{"params": rest, "lr": lr}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": lr * encoder_lr_scale})
    return groups
