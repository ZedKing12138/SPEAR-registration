"""Small, transparent registration metrics with explicit averaging rules."""

from collections.abc import Sequence

import torch
from torch import Tensor

from .geometry import jacobian_determinant


def _label_batch(labels: Tensor) -> Tensor:
    if labels.ndim == 3:
        return labels.unsqueeze(0)
    if labels.ndim == 5 and labels.shape[1] == 1:
        return labels[:, 0]
    if labels.ndim == 4:
        return labels
    raise ValueError("label maps must be [D,H,W], [B,D,H,W] or [B,1,D,H,W]")


def dice_score(
    pred: Tensor,
    target: Tensor,
    labels: Sequence[int] | Tensor | None = None,
    include_background: bool = False,
) -> Tensor:
    """Mean discrete-label Dice across nonempty (subject, label) pairs.

    ``pred`` and ``target`` are integer-valued label maps, not logits or
    one-hot probabilities. The default labels are their union. Label zero is
    background. A class absent from both prediction and target for a subject
    is excluded; a class missing from only one side scores zero. If there are
    no foreground classes at all, the result is one (two empty segmentations).
    """
    pred, target = _label_batch(pred), _label_batch(target)
    if pred.shape != target.shape or pred.device != target.device:
        raise ValueError("prediction and target must have equal shapes and devices")
    if labels is None:
        class_ids = torch.unique(torch.cat((pred.reshape(-1), target.reshape(-1))))
    else:
        class_ids = torch.unique(torch.as_tensor(labels, device=pred.device))
    if not include_background:
        class_ids = class_ids[class_ids != 0]
    scores = []
    for class_id in class_ids:
        predicted = (pred == class_id).flatten(1)
        reference = (target == class_id).flatten(1)
        denominator = predicted.sum(1) + reference.sum(1)
        present = denominator > 0
        if bool(present.any()):
            intersection = (predicted & reference).sum(1)
            scores.append((2.0 * intersection[present] / denominator[present]).float())
    return (
        torch.cat(scores).mean()
        if scores
        else torch.ones((), device=pred.device, dtype=torch.float32)
    )


def folding_rate(flow: Tensor) -> Tensor:
    """Fraction, in [0, 1], of voxels with Jacobian determinant <= 0.

    Multiply this result by 100 for a percentage. Zero determinants are
    included because they also represent a locally non-invertible mapping.
    """
    return (jacobian_determinant(flow) <= 0).float().mean()


def ncc_metric(first: Tensor, second: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean signed global Pearson NCC across batch/channel pairs, in [-1,1].

    This reporting metric differs from squared *local* NCC used for training.
    Constant images have undefined correlation and are assigned zero.
    """
    if first.shape != second.shape or first.ndim != 5:
        raise ValueError("NCC inputs must have equal [B,C,D,H,W] shapes")
    if eps <= 0:
        raise ValueError("eps must be positive")
    dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
    a, b = first.to(dtype).flatten(2), second.to(dtype).flatten(2)
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    numerator = (a * b).sum(-1)
    denominator = (a.square().sum(-1) * b.square().sum(-1)).sqrt()
    return (numerator / denominator.clamp_min(eps)).clamp(-1, 1).mean()
