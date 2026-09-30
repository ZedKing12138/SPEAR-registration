"""Unsupervised image similarity and displacement regularization objectives."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


def _window_shape(window: int | Sequence[int], spatial: Sequence[int]) -> tuple[int, int, int]:
    requested = (window,) * 3 if isinstance(window, int) else tuple(window)
    if len(requested) != 3 or any(int(n) != n or n < 1 or n % 2 == 0 for n in requested):
        raise ValueError("NCC window must be one positive odd integer or three such integers")
    # Tiny test volumes or deep pyramid levels may be smaller than the window.
    # Shrink each dimension to the largest odd window that fits that dimension.
    return tuple(
        min(int(w), int(s) if s % 2 else max(1, int(s) - 1)) for w, s in zip(requested, spatial)
    )


def local_ncc_loss(
    moving_warped: Tensor,
    fixed: Tensor,
    window: int | Sequence[int] = 9,
    eps: float = 1e-5,
) -> Tensor:
    """Return one minus mean squared local normalized cross-correlation.

    Inputs are equally shaped [B, C, D, H, W] images. Each image channel is
    scored separately. Neighborhood sums use only valid image voxels at the
    boundaries. Squaring correlation follows the common mono-modal 3-D
    registration objective. Windows that are constant in both inputs are
    excluded: they contain no correlation information, including shared
    background. A constant/nonconstant pair scores the maximal loss of one.
    If all windows are excluded the loss is zero, rather than NaN. Thus two
    different constant images also provide no NCC supervision. The objective
    is differentiable with respect to both images.
    """
    if moving_warped.shape != fixed.shape or moving_warped.ndim != 5:
        raise ValueError("NCC inputs must have the same [B, C, D, H, W] shape")
    if not moving_warped.is_floating_point() or not fixed.is_floating_point():
        raise TypeError("NCC inputs must be floating point")
    if eps <= 0:
        raise ValueError("eps must be positive")
    dtype = torch.float64 if moving_warped.dtype == torch.float64 else torch.float32
    first, second = moving_warped.to(dtype), fixed.to(dtype)
    kernel = _window_shape(window, first.shape[2:])
    padding = tuple(n // 2 for n in kernel)
    channels = first.shape[1]
    fields = torch.cat((first, second, first.square(), second.square(), first * second), dim=1)
    # divisor_override=1 turns average pooling into a fast box-window sum.
    sums = F.avg_pool3d(fields, kernel, stride=1, padding=padding, divisor_override=1)
    sum_first, sum_second, sum_first2, sum_second2, sum_product = sums.split(channels, dim=1)
    count = F.avg_pool3d(
        torch.ones_like(first[:1, :1]), kernel, stride=1, padding=padding, divisor_override=1
    )
    covariance = sum_product - sum_first * sum_second / count
    variance_first = (sum_first2 - sum_first.square() / count).clamp_min(0)
    variance_second = (sum_second2 - sum_second.square() / count).clamp_min(0)
    correlation2 = covariance.square() / (variance_first * variance_second + eps)
    informative = ((variance_first > eps) | (variance_second > eps)).to(dtype)
    penalties = (1.0 - correlation2.clamp(0, 1)) * informative
    return penalties.sum() / informative.sum().clamp_min(1)


def smoothness_loss(flow: Tensor) -> Tensor:
    """Mean squared first spatial derivatives of a voxel-unit displacement.

    All three displacement channels contribute equally; the available spatial
    axes contribute equally. A spatially constant translation has zero cost.
    Singleton axes are omitted. The returned zero remains connected to the
    graph even if every spatial dimension is one.
    """
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise ValueError("flow must have shape [B, 3, D, H, W]")
    penalties = []
    for dim in (2, 3, 4):
        if flow.shape[dim] > 1:
            difference = flow.narrow(dim, 1, flow.shape[dim] - 1) - flow.narrow(
                dim, 0, flow.shape[dim] - 1
            )
            penalties.append(difference.square().mean())
    return torch.stack(penalties).mean() if penalties else flow.sum() * 0


def registration_loss(
    output: Mapping[str, Any],
    fixed: Tensor,
    ncc_window: int | Sequence[int] = 9,
    smooth_weight: float = 0.05,
    stage_weights: Sequence[float] | None = None,
    coarse_weight: float = 0.0,
) -> dict[str, Tensor]:
    """Aggregate image NCC and residual smoothness across refinement stages.

    Required output entries are ``warped_stages`` and ``residuals``, each a
    nonempty sequence of tensors. Stage weights default to an equal average
    and are normalized to sum to one. Residual regularization is averaged
    independently, allowing the model to include its coarse residual.

    If ``coarse_weight > 0``, ``coarse_warped`` adds optional image supervision
    for a learned coarse initializer. It defaults to zero to avoid silently
    changing the objective. No ground-truth segmentation enters this loss.

    Returns a dictionary containing differentiable scalar ``loss``,
    ``similarity``, ``smoothness`` and ``coarse_similarity`` values.
    """
    stages = list(output["warped_stages"])
    residuals = list(output["residuals"])
    if not stages or not residuals:
        raise ValueError("warped_stages and residuals must be nonempty sequences")
    if smooth_weight < 0 or coarse_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    if stage_weights is None:
        weights = fixed.new_ones(len(stages), dtype=torch.float32)
    else:
        if len(stage_weights) != len(stages):
            raise ValueError("one stage weight is required per warped image")
        weights = fixed.new_tensor(stage_weights, dtype=torch.float32)
        if (
            bool((weights < 0).any())
            or not bool(torch.isfinite(weights).all())
            or float(weights.sum()) <= 0
        ):
            raise ValueError("stage weights must be finite, nonnegative and have a positive sum")
    weights = weights / weights.sum()
    stage_losses = torch.stack([local_ncc_loss(warped, fixed, ncc_window) for warped in stages])
    similarity = (stage_losses * weights).sum()
    smoothness = torch.stack([smoothness_loss(residual) for residual in residuals]).mean()
    coarse_similarity = similarity * 0
    if coarse_weight:
        if "coarse_warped" not in output:
            raise ValueError("coarse_weight requires output['coarse_warped']")
        coarse_similarity = local_ncc_loss(output["coarse_warped"], fixed, ncc_window)
    loss = similarity + smooth_weight * smoothness + coarse_weight * coarse_similarity
    return {
        "loss": loss,
        "similarity": similarity,
        "smoothness": smoothness,
        "coarse_similarity": coarse_similarity,
    }
