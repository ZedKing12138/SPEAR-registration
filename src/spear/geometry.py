"""Differentiable 3-D pullback warping and displacement-field operations.

All tensors use ``[batch, channel, depth, height, width]``. Displacement channels
are **(z, y, x)** in voxels, not grid_sample's (x, y, z) normalized coordinates.
``warp(image, u)[x] = image[x + u(x)]``. Thus a positive x displacement samples
to the right; the visible contents of a warped image move to the left.

``align_corners=True`` is used consistently by warping and field resizing. The
explicit convention prevents common sign, composition-order and scaling bugs.
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _check_flow(flow: Tensor) -> None:
    """Validate the public displacement layout without detaching its graph."""
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise ValueError("flow must have shape [B, 3, D, H, W] in z/y/x order")
    if not flow.is_floating_point():
        raise TypeError("flow must be a floating-point tensor")
    if any(n < 1 for n in flow.shape[2:]):
        raise ValueError("flow spatial dimensions must be nonempty")


def warp(
    image: Tensor,
    flow: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> Tensor:
    """Sample ``image`` at identity + ``flow`` using a differentiable grid.

    Args:
        image: Floating-point tensor [B, C, D, H, W].
        flow: Displacement [B, 3, D, H, W], channel order (z, y, x), in voxels.
        mode: ``bilinear`` performs trilinear sampling for a 3-D volume. Use
            ``nearest`` for discrete segmentation labels, converted to float.
        padding_mode: grid_sample boundary rule; ``border`` is the default.

    The image and field must have matching batches, spatial sizes and devices.
    A singleton spatial axis is mapped to normalized zero to avoid division by
    zero; with border sampling it has only one meaningful sample location.
    """
    _check_flow(flow)
    if image.ndim != 5 or not image.is_floating_point():
        raise ValueError("image must be floating point with shape [B, C, D, H, W]")
    if image.shape[0] != flow.shape[0] or image.shape[2:] != flow.shape[2:]:
        raise ValueError("image and flow must have matching batch and spatial sizes")
    if image.device != flow.device:
        raise ValueError("image and flow must be on the same device")
    if mode not in ("bilinear", "nearest"):
        raise ValueError("mode must be 'bilinear' (trilinear in 3-D) or 'nearest'")
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("unsupported grid_sample padding mode")

    # grid_sample requires the sampling grid and source to share their dtype.
    # CPU grid_sample does not support half precision, so promote only there.
    working_dtype = image.dtype
    if image.device.type == "cpu" and working_dtype in (torch.float16, torch.bfloat16):
        working_dtype = torch.float32
    source = image.to(working_dtype)
    displacement = flow.to(working_dtype)
    axes = [torch.arange(n, device=image.device, dtype=working_dtype) for n in image.shape[2:]]
    base = torch.meshgrid(*axes, indexing="ij")
    normalized = []
    for axis, size in enumerate(image.shape[2:]):
        coordinates = base[axis][None] + displacement[:, axis]
        normalized.append(2.0 * coordinates / (size - 1) - 1.0 if size > 1 else coordinates * 0)
    # PyTorch expects its last coordinate axis to be x/y/z, hence the reversal.
    grid = torch.stack(normalized[::-1], dim=-1)
    result = F.grid_sample(source, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return result.to(image.dtype)


def compose(outer_flow: Tensor, inner_flow: Tensor) -> Tensor:
    """Return the displacement of ``phi_outer o phi_inner``.

    For ``phi(x) = x + u(x)``, the resulting displacement is
    ``inner_flow + warp(outer_flow, inner_flow)``. This composition generally
    differs from field addition and from reversing the two arguments.
    """
    _check_flow(outer_flow)
    _check_flow(inner_flow)
    if outer_flow.shape != inner_flow.shape:
        raise ValueError("composed fields must have identical shapes")
    return inner_flow + warp(outer_flow, inner_flow)


def resize_flow(flow: Tensor, size: Sequence[int]) -> Tensor:
    """Resize a displacement field while preserving its physical map extent.

    Interpolation uses aligned corners; the voxel-unit component on axis i is
    therefore multiplied by ``(new_size[i]-1)/(old_size[i]-1)``. Merely resizing
    the tensor, without this scaling, changes the intended transformation.
    A source or target singleton axis has no sampled extent and is set to zero.
    """
    _check_flow(flow)
    if len(size) != 3 or any(int(n) != n or n < 1 for n in size):
        raise ValueError("size must contain three positive integers")
    target = tuple(int(n) for n in size)
    source = tuple(flow.shape[2:])
    resized = F.interpolate(flow, size=target, mode="trilinear", align_corners=True)
    ratios = [
        (new - 1) / (old - 1) if old > 1 and new > 1 else 0.0 for old, new in zip(source, target)
    ]
    scale = flow.new_tensor(ratios).view(1, 3, 1, 1, 1)
    return resized * scale


def _spatial_derivative(field: Tensor, axis: int) -> Tensor:
    """Centered finite differences inside; one-sided differences at boundaries."""
    dim = axis + 2
    count = field.shape[dim]
    if count == 1:
        return torch.zeros_like(field)
    first = field.narrow(dim, 1, 1) - field.narrow(dim, 0, 1)
    last = field.narrow(dim, count - 1, 1) - field.narrow(dim, count - 2, 1)
    if count == 2:
        return torch.cat((first, last), dim=dim)
    middle = (field.narrow(dim, 2, count - 2) - field.narrow(dim, 0, count - 2)) * 0.5
    return torch.cat((first, middle, last), dim=dim)


def jacobian_determinant(flow: Tensor) -> Tensor:
    """Compute det(d(identity + flow)/dx), returning [B, D, H, W].

    The derivative is with respect to voxel coordinates. With a common fixed
    and moving voxel spacing, converting both coordinates and displacements to
    physical units is a similarity transform of this Jacobian and leaves its
    determinant unchanged. This is not an affine/header conversion utility.
    """
    _check_flow(flow)
    derivatives = [_spatial_derivative(flow, axis) for axis in range(3)]
    # Matrix rows are displacement components; columns are derivative axes.
    j00, j01, j02 = 1 + derivatives[0][:, 0], derivatives[1][:, 0], derivatives[2][:, 0]
    j10, j11, j12 = derivatives[0][:, 1], 1 + derivatives[1][:, 1], derivatives[2][:, 1]
    j20, j21, j22 = derivatives[0][:, 2], derivatives[1][:, 2], 1 + derivatives[2][:, 2]
    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )
