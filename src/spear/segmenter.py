"""Small 3D U-Net used exclusively as an auxiliary anatomical prior.

Labels train this network separately. Registration must freeze the resulting
network and must never optimize a label-overlap loss through it.
"""

from dataclasses import dataclass
from math import gcd

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SegmenterConfig:
    """Configurable widths; class order is recorded separately in checkpoints."""

    num_classes: int = 2
    base_channels: int = 8


def _block(cin: int, cout: int) -> nn.Sequential:
    # GroupNorm works with batch size one and has no running-statistic leakage.
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(gcd(4, cout), cout),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(gcd(4, cout), cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class AuxiliarySegmenter(nn.Module):
    """Two-level U-Net returning logits, including a background class."""

    def __init__(self, config: SegmenterConfig | None = None):
        super().__init__()
        self.config = config or SegmenterConfig()
        if self.config.num_classes < 2 or self.config.base_channels < 2:
            raise ValueError("At least two classes and two base channels are required")
        c = self.config.base_channels
        self.enc0 = _block(1, c)
        self.enc1 = _block(c, c * 2)
        self.bottleneck = _block(c * 2, c * 4)
        self.dec1 = _block(c * 6, c * 2)
        self.dec0 = _block(c * 3, c)
        self.output = nn.Conv3d(c, self.config.num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Accept [batch,1,depth,height,width] and preserve its spatial shape."""
        x0 = self.enc0(image)
        x1 = self.enc1(F.avg_pool3d(x0, 2))
        x2 = self.bottleneck(F.avg_pool3d(x1, 2))
        y1 = self.dec1(
            torch.cat(
                (F.interpolate(x2, size=x1.shape[2:], mode="trilinear", align_corners=True), x1), 1
            )
        )
        y0 = self.dec0(
            torch.cat(
                (F.interpolate(y1, size=x0.shape[2:], mode="trilinear", align_corners=True), x0), 1
            )
        )
        return self.output(y0)


def encode_labels(labels: torch.Tensor, values: list[int]) -> torch.Tensor:
    """Map arbitrary NIfTI label IDs into contiguous CE targets without guessing.

    The mapping is fitted on training labels only and stored in the checkpoint.
    Unknown validation labels raise an error instead of becoming background.
    """
    encoded = torch.full_like(labels, -1, dtype=torch.long)
    for index, value in enumerate(values):
        encoded[labels == value] = index
    if (encoded < 0).any():
        unseen = labels[encoded < 0].unique().tolist()
        raise ValueError(f"Labels not in the training class mapping: {unseen}")
    return encoded
