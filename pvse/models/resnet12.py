from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    """DropBlock schedule used by the frozen RFS-style ResNet-12 backbone."""

    def __init__(self, block_size: int = 5, drop_prob: float = 0.1):
        super().__init__()
        self.block_size = int(block_size)
        self.drop_prob = float(drop_prob)
        self.num_batches_tracked = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0:
            return x
        self.num_batches_tracked += 1
        feat_size = int(x.size(2))
        if feat_size < self.block_size:
            return F.dropout(x, p=self.drop_prob, training=True)
        keep_rate = max(
            1.0 - self.drop_prob / float(20 * 2000) * float(self.num_batches_tracked),
            1.0 - self.drop_prob,
        )
        gamma = (
            (1.0 - keep_rate)
            / float(self.block_size**2)
            * float(feat_size**2)
            / float((feat_size - self.block_size + 1) ** 2)
        )
        mask = (torch.rand_like(x) < gamma).float()
        block_mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1.0 - block_mask
        return x * block_mask * (block_mask.numel() / block_mask.sum().clamp_min(1.0))


class RFSBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        drop_rate: float,
        dropblock_size: int,
        use_dropblock: bool,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = (
            DropBlock2D(dropblock_size, drop_rate)
            if use_dropblock
            else nn.Dropout(p=float(drop_rate))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1, inplace=True)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.1, inplace=True)
        out = self.bn3(self.conv3(out))
        out = F.leaky_relu(out + identity, negative_slope=0.1, inplace=True)
        out = self.pool(out)
        return self.drop(out)


@dataclass(frozen=True)
class BackboneOutput:
    global_features: torch.Tensor
    spatial_features: torch.Tensor


class RFSResNet12Backbone(nn.Module):
    """The exact 64/160/320/640 RFS-style ResNet-12 used by PVSE.

    At 84×84 input resolution, ``spatial_features`` has shape ``[N,640,5,5]``.
    """

    feature_dim = 640

    def __init__(self, drop_rate: float = 0.1, dropblock_size: int = 5):
        super().__init__()
        self.layer1 = RFSBlock(
            3, 64, drop_rate=drop_rate, dropblock_size=dropblock_size, use_dropblock=False
        )
        self.layer2 = RFSBlock(
            64, 160, drop_rate=drop_rate, dropblock_size=dropblock_size, use_dropblock=False
        )
        self.layer3 = RFSBlock(
            160, 320, drop_rate=drop_rate, dropblock_size=dropblock_size, use_dropblock=True
        )
        self.layer4 = RFSBlock(
            320, 640, drop_rate=drop_rate, dropblock_size=dropblock_size, use_dropblock=True
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward_with_spatial(self, x: torch.Tensor) -> BackboneOutput:
        spatial = self.forward_spatial(x)
        pooled = self.avgpool(spatial).view(spatial.size(0), -1)
        return BackboneOutput(global_features=pooled, spatial_features=spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_spatial(x).global_features
