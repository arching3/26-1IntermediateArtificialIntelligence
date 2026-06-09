from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from models.activations import get_activations


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
        dropout_p: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        activation_factory = get_activations(activation)
        self.activation1 = activation_factory()
        self.activation2 = activation_factory()
        self.dropout = (
            nn.Dropout2d(dropout_p) if dropout_p > 0.0 else nn.Identity()
        )
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return self.activation2(out)


class ResNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        block_counts: Sequence[int],
        base_channels: int,
        num_classes: int,
        dropout_p: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if not block_counts or any(count <= 0 for count in block_counts):
            raise ValueError("block_counts must contain positive integers.")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in the range [0, 1).")

        self.activation_name = activation
        self.current_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            get_activations(activation)(),
        )

        stage_channels = [
            base_channels * (2**index) for index in range(len(block_counts))
        ]
        self.stages = nn.Sequential(
            *[
                self._make_stage(
                    output_channels=channels,
                    block_count=count,
                    stride=1 if index == 0 else 2,
                    dropout_p=dropout_p,
                    activation=activation,
                )
                for index, (channels, count) in enumerate(
                    zip(stage_channels, block_counts)
                )
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.current_channels, num_classes)

        self._initialize_weights()

    def _make_stage(
        self,
        output_channels: int,
        block_count: int,
        stride: int,
        dropout_p: float,
        activation: str,
    ) -> nn.Sequential:
        blocks = [
            BasicBlock(
                self.current_channels,
                output_channels,
                stride=stride,
                dropout_p=dropout_p,
                activation=activation,
            )
        ]
        self.current_channels = output_channels
        blocks.extend(
            BasicBlock(
                self.current_channels,
                output_channels,
                dropout_p=dropout_p,
                activation=activation,
            )
            for _ in range(block_count - 1)
        )
        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
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
            "ResNet expects input_shape in (channels, height, width) format."
        )

    return ResNet(
        input_channels=input_shape[0],
        block_counts=model_config["block_counts"],
        base_channels=model_config["base_channels"],
        num_classes=num_classes,
        dropout_p=model_config["dropout_p"],
        activation=model_config.get("activation", "relu"),
    )
