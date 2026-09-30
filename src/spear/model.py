"""SPEAR: error-guided sparse refinement with dense displacement prediction.

SPEAR combines shared full-resolution encoding, 27 global anchors, 27 local
candidates per anchor, region-wise Top-r routing, adjacent-stage hard
non-redundancy, and sequential dual-stream injection.

Routing and dense deformation prediction use the following operations:

* A straight-through *unit* gate trains routing scores without changing the
  hard-routing forward pass. Its gradient is a biased softmax surrogate.
* Moving encoder features are warped into the current fixed-image frame before
  selecting stage candidates; the expensive encoder is evaluated only once.
* The 3-D token lattice is interpolated and fused with dense encoder features
  before the two-convolution displacement head. This supplies spatial detail
  not recoverable from only 27 pooled vectors.

Frozen semantic predictions guide routing only. Registration labels and
segmentation-overlap losses are not used by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import compose, warp


@dataclass(frozen=True)
class SPEARConfig:
    """Serializable architecture and routing settings.

    ``transformer_layers`` is the depth of each dual-stream block. One block is
    used for global initialization and one for each refinement stage. Each block
    has independent parameters, with four layers per block by default.

    Spatial tensors use ``[batch, channel, depth, height, width]``. Displacements
    are backward sampling offsets in voxel units, with channel order z, y, x.
    The intensity guidance mode provides an intensity-based ablation of the
    semantic-guidance configuration.
    """

    feature_channels: int = 16
    embed_dim: int = 48
    num_heads: int = 4
    transformer_layers: int = 4
    stages: int = 3
    top_k: int = 6
    region_grid: int = 3
    candidate_grid: int = 3
    max_residual: float = 2.0
    dropout: float = 0.0
    guidance: Literal["semantic", "intensity"] = "semantic"
    hard_nr: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            "feature_channels",
            "embed_dim",
            "num_heads",
            "transformer_layers",
            "stages",
            "top_k",
            "region_grid",
            "candidate_grid",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer, got {value!r}")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        candidates = self.candidate_grid**3
        if self.top_k > candidates:
            raise ValueError("top_k exceeds the number of candidates per region")
        if self.hard_nr and self.stages > 1 and 2 * self.top_k > candidates:
            raise ValueError("adjacent-stage hard_nr requires 2 * top_k <= candidates")
        if not math.isfinite(self.max_residual) or self.max_residual <= 0:
            raise ValueError("max_residual must be finite and positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.guidance not in {"semantic", "intensity"}:
            raise ValueError("guidance must be 'semantic' or 'intensity'")
        if not isinstance(self.hard_nr, bool):
            raise ValueError("hard_nr must be a boolean")


def _normalization(channels: int) -> nn.GroupNorm:
    """Choose a valid GroupNorm group count, including for small smoke models."""
    groups = min(4, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _absolute_encoding(grid: int, dim: int) -> Tensor:
    """3-D sinusoidal encoding at normalized global-region centroids."""
    centers = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    coordinates = torch.stack(torch.meshgrid(centers, centers, centers, indexing="ij"), -1)
    bands = math.ceil(dim / 6)
    frequencies = torch.exp(torch.arange(bands) * (-math.log(10000.0) / max(bands - 1, 1)))
    angles = 2 * math.pi * coordinates.reshape(-1, 3, 1) * frequencies
    encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
    return encoding[:, :dim].contiguous()


class _SharedEncoder(nn.Sequential):
    """Small full-resolution feature extractor without a dense feature pyramid."""

    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv3d(1, channels, 3, padding=1),
            _normalization(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            _normalization(channels),
            nn.GELU(),
        )


class _DualStreamLayer(nn.Module):
    """Symmetric self/cross-attention with simultaneous cross-stream updates.

    Moving and fixed streams share layer weights, but remain separate tensors.
    Each attention operation sees only L global anchor representations.
    """

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, moving: Tensor, fixed: Tensor) -> tuple[Tensor, Tensor]:
        m, f = self.self_norm(moving), self.self_norm(fixed)
        moving = moving + self.dropout(self.self_attn(m, m, m, need_weights=False)[0])
        fixed = fixed + self.dropout(self.self_attn(f, f, f, need_weights=False)[0])
        # Compute both updates from the same pre-update states. Updating one
        # stream first would introduce an unintended moving/fixed asymmetry.
        m, f = self.cross_norm(moving), self.cross_norm(fixed)
        next_m = moving + self.dropout(self.cross_attn(m, f, f, need_weights=False)[0])
        next_f = fixed + self.dropout(self.cross_attn(f, m, m, need_weights=False)[0])
        return next_m + self.ffn(self.ffn_norm(next_m)), next_f + self.ffn(self.ffn_norm(next_f))


class _DualStreamBlock(nn.Module):
    def __init__(self, config: SPEARConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DualStreamLayer(config.embed_dim, config.num_heads, config.dropout)
                for _ in range(config.transformer_layers)
            ]
        )

    def forward(self, moving: Tensor, fixed: Tensor) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            moving, fixed = layer(moving, fixed)
        return moving, fixed


class _InjectionProjection(nn.Module):
    """One Transformer layer P before adding the next selected local token.

    Self-attention mixes the persistent global anchors between injections.
    The sequence retains L global anchors throughout sequential injection.
    Each selected local token is added after the Transformer projection.
    """

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = self.attention_norm(hidden)
        hidden = hidden + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return hidden + self.projection(hidden)


class _DenseHead(nn.Module):
    """Lift anchor context and fuse dense features using exactly two convolutions.

    A small nonzero final initialization permits gradients into the entire
    network on the first update while starting close to the identity warp.
    ``max_residual`` bounds each residual component, not the composed flow.
    This bound improves optimization stability; it does not guarantee positive
    Jacobian determinants or diffeomorphism.
    """

    def __init__(self, config: SPEARConfig) -> None:
        super().__init__()
        self.grid = config.region_grid
        self.max_residual = config.max_residual
        width = config.feature_channels
        self.context_norm = nn.LayerNorm(config.embed_dim)
        self.conv1 = nn.Conv3d(config.embed_dim + 2 * width, width, 3, padding=1)
        self.conv2 = nn.Conv3d(width, 3, 3, padding=1)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, context: Tensor, moving_features: Tensor, fixed_features: Tensor) -> Tensor:
        batch, _, dim = context.shape
        lattice = (
            self.context_norm(context)
            .transpose(1, 2)
            .reshape(batch, dim, self.grid, self.grid, self.grid)
        )
        dense_context = F.interpolate(
            lattice, size=fixed_features.shape[2:], mode="trilinear", align_corners=True
        )
        fused = torch.cat((dense_context, moving_features, fixed_features), dim=1)
        return self.max_residual * torch.tanh(self.conv2(F.gelu(self.conv1(fused))))


class SPEAR(nn.Module):
    """Predict a dense backward deformation between two prealigned 3-D images.

    Semantic guidance requires an already pretrained segmenter supplied by the
    caller. It must return class logits of shape [B, classes, D, H, W]. Its
    parameters are frozen, it always remains in evaluation mode, and semantic
    predictions are evaluated without autograd. The fixed segmentation is
    cached within each forward pass; the warped moving image is resegmented
    before every refinement stage.
    """

    def __init__(self, config: SPEARConfig, segmenter: nn.Module | None = None) -> None:
        super().__init__()
        if config.guidance == "semantic" and segmenter is None:
            raise ValueError(
                "semantic guidance requires a pretrained segmenter; use guidance='intensity' for the explicit ablation"
            )
        self.config = config
        self.segmenter = segmenter
        if self.segmenter is not None:
            self.segmenter.requires_grad_(False)
            self.segmenter.eval()
        dim = config.embed_dim
        self.encoder = _SharedEncoder(config.feature_channels)
        self.token_projection = nn.Linear(config.feature_channels, dim)
        self.register_buffer("global_positions", _absolute_encoding(config.region_grid, dim))
        self.local_positions = nn.Parameter(torch.empty(config.candidate_grid**3, dim))
        nn.init.normal_(self.local_positions, std=0.02)
        self.coarse_transformer = _DualStreamBlock(config)
        self.refinement_transformers = nn.ModuleList(
            [_DualStreamBlock(config) for _ in range(config.stages)]
        )
        self.router = nn.Sequential(nn.Linear(2 * dim + 1, dim), nn.GELU(), nn.Linear(dim, 1))
        self.moving_injections = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _InjectionProjection(dim, config.num_heads, config.dropout)
                        for _ in range(config.top_k)
                    ]
                )
                for _ in range(config.stages)
            ]
        )
        self.fixed_injections = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _InjectionProjection(dim, config.num_heads, config.dropout)
                        for _ in range(config.top_k)
                    ]
                )
                for _ in range(config.stages)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.stage_cross_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    dim, config.num_heads, dropout=config.dropout, batch_first=True
                )
                for _ in range(config.stages + 1)
            ]
        )
        self.coarse_head = _DenseHead(config)
        self.residual_heads = nn.ModuleList([_DenseHead(config) for _ in range(config.stages)])

    def train(self, mode: bool = True) -> "SPEAR":
        """Prevent parent ``train()`` calls from unfreezing segmenter statistics."""
        super().train(mode)
        if self.segmenter is not None:
            self.segmenter.eval()
        return self

    def _tokens(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Pool global and correctly grouped hierarchical local tokens.

        Default preprocessing uses spatial dimensions divisible by nine.
        Adaptive pooling also accepts other dimensions; neighboring pooling
        bins can then overlap by one voxel.
        """
        b, c = features.shape[:2]
        g, k = self.config.region_grid, self.config.candidate_grid
        global_features = F.adaptive_avg_pool3d(features, (g, g, g)).flatten(2).transpose(1, 2)
        local_grid = F.adaptive_avg_pool3d(features, (g * k, g * k, g * k))
        # Physical order is gD,kD,gH,kH,gW,kW, not [global_index, local_index].
        local_features = local_grid.reshape(b, c, g, k, g, k, g, k)
        local_features = local_features.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(b, g**3, k**3, c)
        global_tokens = self.token_projection(global_features) + self.global_positions.to(
            features.dtype
        )
        local_tokens = self.token_projection(local_features) + self.local_positions.to(
            features.dtype
        )
        return global_tokens, local_tokens

    def _select(self, scores: Tensor, previous: Tensor | None) -> tuple[Tensor, Tensor]:
        """Exact hard Top-r forward with a biased straight-through gradient.

        The unit-valued gate does not alter the forward-pass token amplitudes.
        Its surrogate gradient is supplied by a softmax over *all available*
        candidates, allowing a training signal even when r = 1. Masked previous
        indices receive neither selection probability nor a routing gradient.
        """
        eligible = torch.ones_like(scores, dtype=torch.bool)
        if self.config.hard_nr and previous is not None:
            eligible.scatter_(2, previous, False)
        masked_scores = scores.masked_fill(~eligible, -torch.inf)
        indices = masked_scores.topk(self.config.top_k, dim=-1, sorted=True).indices
        probabilities = masked_scores.softmax(dim=-1)
        available_count = eligible.sum(dim=-1, keepdim=True).to(scores.dtype)
        surrogate = available_count * probabilities.gather(-1, indices)
        gates = 1.0 + (surrogate - surrogate.detach())
        return indices, gates

    def _context(self, moving: Tensor, fixed: Tensor, index: int) -> Tensor:
        moving_norm, fixed_norm = self.output_norm(moving), self.output_norm(fixed)
        correspondence = self.stage_cross_attention[index](
            moving_norm, fixed_norm, fixed_norm, need_weights=False
        )[0]
        return moving + correspondence

    @torch.no_grad()
    def _semantic_probabilities(self, image: Tensor) -> Tensor:
        if self.segmenter is None:  # Defensive guard; the constructor rejects it.
            raise RuntimeError("no pretrained semantic segmenter is attached")
        logits = self.segmenter(image)
        if not isinstance(logits, Tensor) or logits.ndim != 5 or logits.shape[0] != image.shape[0]:
            raise ValueError("segmenter must return a [B, classes, D, H, W] logits tensor")
        if logits.shape[1] < 2 or logits.shape[2:] != image.shape[2:]:
            raise ValueError("segmenter logits need >=2 classes and the input spatial dimensions")
        return logits.softmax(dim=1)

    def _validate_inputs(self, moving: Tensor, fixed: Tensor) -> None:
        if moving.ndim != 5 or fixed.shape != moving.shape or moving.shape[1] != 1:
            raise ValueError("moving and fixed must have identical [B, 1, D, H, W] shapes")
        if moving.shape[0] < 1 or not moving.is_floating_point() or not fixed.is_floating_point():
            raise ValueError("inputs require a nonempty batch and floating-point intensities")
        if moving.device != fixed.device or moving.dtype != fixed.dtype:
            raise ValueError("moving and fixed must share device and dtype")
        minimum = self.config.region_grid * self.config.candidate_grid
        if any(size < minimum for size in moving.shape[2:]):
            raise ValueError(f"every spatial dimension must be at least {minimum} voxels")

    def forward(self, moving: Tensor, fixed: Tensor) -> dict[str, Tensor | list[Tensor]]:
        """Return the final warp and the intermediate tensors used in training.

        ``warped_stages`` and ``residuals`` contain N refinement tensors; the
        initial global estimate is separately available as ``coarse_flow`` and
        ``coarse_warped``. ``selected_indices`` contains N [B, L, r] integer
        tensors, making routing behavior inspectable without recomputation.
        """
        self._validate_inputs(moving, fixed)
        moving_features, fixed_features = self.encoder(moving), self.encoder(fixed)
        moving_global, _ = self._tokens(moving_features)
        fixed_global, fixed_local = self._tokens(fixed_features)
        hidden_m, hidden_f = self.coarse_transformer(moving_global, fixed_global)
        flow = self.coarse_head(
            self._context(hidden_m, hidden_f, 0), moving_features, fixed_features
        )
        coarse_flow = flow
        warped_image = warp(moving, flow)
        coarse_warped = warped_image
        fixed_probabilities = (
            self._semantic_probabilities(fixed) if self.config.guidance == "semantic" else None
        )
        warped_stages: list[Tensor] = []
        residuals: list[Tensor] = []
        selected_indices: list[Tensor] = []
        previous: Tensor | None = None

        for stage in range(self.config.stages):
            # Residuals act in the fixed-image frame; align dense features and
            # candidate addresses to that frame before discrepancy-guided routing.
            current_moving_features = warp(moving_features, flow)
            current_global, moving_local = self._tokens(current_moving_features)
            with torch.no_grad():
                if fixed_probabilities is not None:
                    current_probabilities = self._semantic_probabilities(warped_image)
                    discrepancy = (
                        (fixed_probabilities - current_probabilities)
                        .square()
                        .mean(dim=1, keepdim=True)
                    )
                else:
                    discrepancy = (fixed - warped_image).square()
                discrepancy = discrepancy + 1.0
                g = self.config.region_grid
                regional_error = (
                    F.adaptive_avg_pool3d(discrepancy, (g, g, g)).flatten(2).transpose(1, 2)
                )
            candidates = self.config.candidate_grid**3
            router_inputs = torch.cat(
                (
                    current_global.unsqueeze(2).expand(-1, -1, candidates, -1),
                    moving_local,
                    regional_error.unsqueeze(2).expand(-1, -1, candidates, -1),
                ),
                dim=-1,
            )
            scores = self.router(router_inputs).squeeze(-1)
            indices, gates = self._select(scores, previous)
            selected_indices.append(indices)
            previous = indices
            gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, self.config.embed_dim)
            selected_m = moving_local.gather(2, gather_indices) * gates.unsqueeze(-1)
            selected_f = fixed_local.gather(2, gather_indices) * gates.unsqueeze(-1)
            hidden_m, hidden_f = self.refinement_transformers[stage](hidden_m, hidden_f)
            for slot in range(self.config.top_k):
                # Each injection adds one candidate per region; hidden length
                # remains L instead of growing to L*r or L*K.
                hidden_m = self.moving_injections[stage][slot](hidden_m) + selected_m[:, :, slot]
                hidden_f = self.fixed_injections[stage][slot](hidden_f) + selected_f[:, :, slot]
            residual = self.residual_heads[stage](
                self._context(hidden_m, hidden_f, stage + 1),
                current_moving_features,
                fixed_features,
            )
            flow = compose(flow, residual)
            warped_image = warp(moving, flow)
            residuals.append(residual)
            warped_stages.append(warped_image)

        return {
            "flow": flow,
            "warped": warped_image,
            "warped_stages": warped_stages,
            "residuals": residuals,
            "coarse_warped": coarse_warped,
            "coarse_flow": coarse_flow,
            "selected_indices": selected_indices,
        }
