"""Model definitions for the Tiny Pix2Pix project."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-layer convolutional block with batch normalisation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """Lightweight U-Net generator."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, feature_maps: List[int] | Tuple[int, ...] | None = None) -> None:
        super().__init__()
        features = list(feature_maps) if feature_maps is not None else [16, 32, 64, 128, 256]

        self.down_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        prev_channels = in_channels
        for feature in features:
            self.down_blocks.append(ConvBlock(prev_channels, feature))
            prev_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.up_transpose_layers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2
        for feature in reversed_features:
            self.up_transpose_layers.append(
                nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2)
            )
            self.up_blocks.append(ConvBlock(prev_channels, feature))
            prev_channels = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: List[torch.Tensor] = []
        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = list(reversed(skip_connections))

        for transpose, up, skip in zip(self.up_transpose_layers, self.up_blocks, skip_connections):
            x = transpose(x)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = up(x)

        return self.final_activation(self.final_conv(x))


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator used by Pix2Pix."""

    def __init__(self, in_channels: int = 3, base_filters: int = 64) -> None:
        super().__init__()
        channels = in_channels * 2
        layers = [
            nn.Conv2d(channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        features = [base_filters * 2, base_filters * 4, base_filters * 8]
        strides = [2, 2, 1]

        in_channels = base_filters
        for feature, stride in zip(features, strides):
            layers.extend(
                [
                    nn.Conv2d(in_channels, feature, kernel_size=4, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_channels = feature

        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([source, target], dim=1)
        return self.model(x)
