from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from models.activations import get_activations


class CNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        num_classes: int,
        kernel_sizes: Sequence[int],
        activation: str,
        dropout_p: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive integers.")
        if len(kernel_sizes) != len(channels):
            raise ValueError("kernel_sizes must have the same length as channels.")
        if any(kernel_size <= 0 for kernel_size in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive integers.")
        if any(kernel_size % 2 == 0 for kernel_size in kernel_sizes):
            raise ValueError("kernel_sizes must contain odd integers for symmetric padding.")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in the range [0, 1).")

        layers: list[nn.Module] = []
        current_channels = input_channels

        for output_channels, kernel_size in zip(channels, kernel_sizes, strict=True):
            layers.extend(
                self._build_conv_block(
                    input_channels=current_channels,
                    output_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    activation=activation,
                    dropout_p=dropout_p,
                    use_batch_norm=use_batch_norm,
                )
            )
            current_channels = output_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, num_classes)

    @staticmethod
    def _build_conv_block(
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int,
        activation: str,
        dropout_p: float,
        use_batch_norm: bool,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = [
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not use_batch_norm,
            )
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(output_channels))
        layers.extend(
            [
                get_activations(activation)(),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ]
        )
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(dropout_p))
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    model_config: dict[str, Any],
) -> nn.Module:
    if len(input_shape) != 3:
        raise ValueError("CNN expects input_shape in (channels, height, width) format.")

    channels = model_config["channels"]
    kernel_sizes = model_config.get("kernel_sizes")
    if kernel_sizes is None:
        kernel_size = model_config.get("kernel_size")
        if kernel_size is None:
            raise KeyError("CNN config requires kernel_sizes or kernel_size.")
        kernel_sizes = [kernel_size] * len(channels)

    return CNN(
        input_channels=input_shape[0],
        channels=channels,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        activation=model_config["activation"],
        dropout_p=model_config["dropout_p"],
        use_batch_norm=model_config["use_batch_norm"],
    )
