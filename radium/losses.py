"""Marginal loss for partially supervised segmentation."""

import torch
import torch.nn.functional as F

NUM_CLASSES = 54


def _normalise(pixel_weight, target):
    """-> (B, H, W) float weight, or None. Broadcasts and validates."""
    if pixel_weight is None:
        return None
    if pixel_weight.shape != target.shape:
        raise ValueError(
            f"pixel_weight {tuple(pixel_weight.shape)} != target {tuple(target.shape)}"
        )
    return pixel_weight.to(torch.float32)


def _merged_background_mask(annotated: torch.Tensor) -> torch.Tensor:
    """(B, 54) bool -> (B, 55) bool: channels absorbed into the observed 0 label."""
    # From the tensor, not NUM_CLASSES, so a reduced class set works too.
    b, n = annotated.shape
    merged = torch.zeros(b, n + 1, dtype=torch.bool, device=annotated.device)
    merged[:, 0] = True
    merged[:, 1:] = ~annotated
    return merged


def marginal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    annotated: torch.Tensor,
    pixel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over the merged label space."""
    merged = _merged_background_mask(annotated)  # (B, 55)

    # Log-normaliser of the full 55-way softmax.
    log_z = torch.logsumexp(logits, dim=1)  # (B, H, W)

    # Log of the summed probability of everything that looks like background.
    neg_inf = torch.finfo(logits.dtype).min
    bg_logits = logits.masked_fill(~merged[:, :, None, None], neg_inf)
    log_p_bg = torch.logsumexp(bg_logits, dim=1)  # (B, H, W)

    # Log-probability of the observed label: the organ, or the merged background.
    log_p_organ = logits.gather(1, target.unsqueeze(1).clamp(min=0)).squeeze(1)
    log_p = torch.where(target > 0, log_p_organ, log_p_bg)

    nll = log_z - log_p
    w = _normalise(pixel_weight, target)
    if w is None:
        return nll.mean()
    return (nll * w).sum() / w.sum().clamp(min=1.0)


def marginal_tversky(
    logits: torch.Tensor,
    target: torch.Tensor,
    annotated: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    weights: torch.Tensor | None = None,
    eps: float = 1.0,
    pixel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tversky index over annotated organs, optionally weighted per class."""
    probs = logits.softmax(dim=1)[:, 1:]
    # Channel count from the logits, not NUM_CLASSES, so a reduced set works.
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2)[:, 1:].to(probs.dtype)

    # Applied once per term: scaling probs and onehot both would square the weight.
    pw = _normalise(pixel_weight, target)
    pw = 1.0 if pw is None else pw.unsqueeze(1).to(probs.dtype)

    dims = (2, 3)
    tp = (pw * probs * onehot).sum(dims)
    fp = (pw * probs * (1 - onehot)).sum(dims)
    fn = (pw * (1 - probs) * onehot).sum(dims)
    # Scaled by 2 so alpha = beta = 0.5 reproduces Dice exactly, eps included.
    tversky = (2 * tp + eps) / (2 * tp + 2 * alpha * fp + 2 * beta * fn + eps)  # (B, 54)

    w = annotated.to(tversky.dtype)
    if weights is not None:
        w = w * weights[None, :].to(tversky.dtype)
    return 1 - (tversky * w).sum() / w.sum().clamp(min=1e-6)


def marginal_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    annotated: torch.Tensor,
    eps: float = 1.0,
    pixel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft Dice over annotated organs only."""
    probs = logits.softmax(dim=1)[:, 1:]  # (B, 54, H, W), drop background
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2)[:, 1:]
    onehot = onehot.to(probs.dtype)

    pw = _normalise(pixel_weight, target)
    pw = 1.0 if pw is None else pw.unsqueeze(1).to(probs.dtype)

    dims = (2, 3)
    inter = (pw * probs * onehot).sum(dims)
    denom = (pw * probs).sum(dims) + (pw * onehot).sum(dims)
    dice = (2 * inter + eps) / (denom + eps)  # (B, 54)

    # Average over annotated (image, class) pairs only.
    w = annotated.to(dice.dtype)
    return 1 - (dice * w).sum() / w.sum().clamp(min=1.0)


def exclusion(
    logits: torch.Tensor,
    target: torch.Tensor,
    annotated: torch.Tensor,
    pixel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Organs do not overlap: where an organ is annotated, no other organ may go."""
    probs = logits.softmax(dim=1)
    unannotated = (~annotated)[:, :, None, None]
    p_unannotated = (probs[:, 1:] * unannotated).sum(dim=1)  # (B, H, W)
    on_organ = target > 0
    pw = _normalise(pixel_weight, target)
    if pw is not None:
        on_organ = on_organ & (pw > 0)
    if not on_organ.any():
        return logits.sum() * 0.0
    if pw is None:
        return p_unannotated[on_organ].mean()
    w = pw[on_organ]
    return (p_unannotated[on_organ] * w).sum() / w.sum().clamp(min=1e-6)


class MarginalLoss(torch.nn.Module):
    """w_ce * marginal CE + w_dice * (Tversky or Dice) + w_excl * exclusion."""

    def __init__(
        self,
        w_ce: float = 1.0,
        w_dice: float = 1.0,
        w_excl: float = 0.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.w_ce, self.w_dice, self.w_excl = w_ce, w_dice, w_excl
        self.alpha, self.beta = alpha, beta
        if weights is None:
            self.weights = None
        else:
            self.register_buffer("weights", weights)

    def forward(self, logits, target, annotated, pixel_weight=None):
        loss = self.w_ce * marginal_cross_entropy(
            logits, target, annotated, pixel_weight=pixel_weight
        )
        if self.w_dice:
            loss = loss + self.w_dice * marginal_tversky(
                logits, target, annotated,
                alpha=self.alpha, beta=self.beta, weights=self.weights,
                pixel_weight=pixel_weight,
            )
        if self.w_excl:
            loss = loss + self.w_excl * exclusion(
                logits, target, annotated, pixel_weight=pixel_weight
            )
        return loss
