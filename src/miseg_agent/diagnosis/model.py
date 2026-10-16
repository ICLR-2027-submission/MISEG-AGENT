"""Factorized 3D Residual U-Net using only CT-derived and rough-mask inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 4
    base_channels: int = 24
    levels: int = 4
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ModelConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value or {}) - allowed
        if unknown:
            raise ValueError(f"Unsupported model configuration fields: {', '.join(sorted(unknown))}")
        return cls(**{key: value[key] for key in allowed if value and key in value})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv3d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.norm2(hidden)
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return hidden + residual


class FactorizedResUNet3D(nn.Module):
    """Predict error presence, error polarity, and auxiliary GT foreground."""

    def __init__(self, **kwargs: Any):
        super().__init__()
        config = ModelConfig.from_mapping(kwargs)
        if config.levels < 2:
            raise ValueError("Model requires at least two resolution levels.")
        self.config = config
        self.model_config = config.to_dict()
        channels = [config.base_channels * (2**index) for index in range(config.levels)]
        self.stem = nn.Conv3d(config.in_channels, channels[0], 3, padding=1, bias=False)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.encoders.append(ResidualBlock(channel, channel, config.dropout))
            if index < len(channels) - 1:
                self.downsamples.append(nn.Conv3d(channel, channels[index + 1], 3, stride=2, padding=1, bias=False))
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(len(channels) - 1, 0, -1):
            self.upsamples.append(nn.ConvTranspose3d(channels[index], channels[index - 1], 2, stride=2))
            self.decoders.append(
                ResidualBlock(channels[index - 1] * 2, channels[index - 1], config.dropout)
            )
        self.final_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.error_head = nn.Conv3d(channels[0], 1, 1)
        self.polarity_head = nn.Conv3d(channels[0], 1, 1)
        self.gt_foreground_head = nn.Conv3d(channels[0], 1, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.stem(x)
        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            hidden = encoder(hidden)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips[:-1])):
            hidden = upsample(hidden)
            if hidden.shape[2:] != skip.shape[2:]:
                hidden = F.interpolate(hidden, size=skip.shape[2:], mode="trilinear", align_corners=False)
            hidden = decoder(torch.cat((hidden, skip), dim=1))
        hidden = F.silu(self.final_norm(hidden))
        return {
            "error_logit": self.error_head(hidden),
            "polarity_logit": self.polarity_head(hidden),
            "gt_foreground_logit": self.gt_foreground_head(hidden),
        }


def factorized_probabilities(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    error = torch.sigmoid(outputs["error_logit"])
    polarity = torch.sigmoid(outputs["polarity_logit"])
    return {
        "correct": 1.0 - error,
        "false_negative": error * polarity,
        "false_positive": error * (1.0 - polarity),
        "error": error,
        "polarity": polarity,
        "gt_foreground": torch.sigmoid(outputs["gt_foreground_logit"]),
    }
