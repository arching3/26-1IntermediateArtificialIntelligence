from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from models.activations import get_activations


class SimpleCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: Sequence[int],
        num_classes: int,
        kernel_size: int,
        activation: str,
        dropout_p: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive integers.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in the range [0, 1).")

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        current_channels = input_channels

        for output_channels in channels:
            layers.extend(
                self._build_conv_block(
                    input_channels=current_channels,
                    output_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
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
        raise ValueError(
            "SimpleCNN expects input_shape in (channels, height, width) format."
        )

    return SimpleCNN(
        input_channels=input_shape[0],
        channels=model_config["channels"],
        num_classes=num_classes,
        kernel_size=model_config["kernel_size"],
        activation=model_config["activation"],
        dropout_p=model_config["dropout_p"],
        use_batch_norm=model_config["use_batch_norm"],
    )
