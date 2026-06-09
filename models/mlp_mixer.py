from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class MixerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MixerBlock(nn.Module):
    def __init__(
        self,
        num_patches: int,
        hidden_dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = MixerMLP(num_patches, token_mlp_dim, dropout_p)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = MixerMLP(hidden_dim, channel_mlp_dim, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_input = self.token_norm(x).transpose(1, 2)
        x = x + self.token_mlp(token_input).transpose(1, 2)
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        patch_size: int,
        num_blocks: int,
        hidden_dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        input_channels, input_height, input_width = input_shape
        positive_values = {
            "input_channels": input_channels,
            "input_height": input_height,
            "input_width": input_width,
            "num_classes": num_classes,
            "patch_size": patch_size,
            "num_blocks": num_blocks,
            "hidden_dim": hidden_dim,
            "token_mlp_dim": token_mlp_dim,
            "channel_mlp_dim": channel_mlp_dim,
        }
        for name, value in positive_values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in the range [0, 1).")

        self.input_height = input_height
        self.input_width = input_width
        self.patch_size = patch_size
        self.pad_height = (-input_height) % patch_size
        self.pad_width = (-input_width) % patch_size

        patch_rows = (input_height + self.pad_height) // patch_size
        patch_columns = (input_width + self.pad_width) // patch_size
        self.num_patches = patch_rows * patch_columns

        self.patch_embedding = nn.Conv2d(
            input_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    num_patches=self.num_patches,
                    hidden_dim=hidden_dim,
                    token_mlp_dim=token_mlp_dim,
                    channel_mlp_dim=channel_mlp_dim,
                    dropout_p=dropout_p,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.kaiming_normal_(
            self.patch_embedding.weight, mode="fan_out", nonlinearity="linear"
        )
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("MLPMixer expects a 4D (batch, channels, height, width) tensor.")
        if x.shape[-2:] != (self.input_height, self.input_width):
            raise ValueError(
                "Input spatial shape does not match the input_shape used to build "
                f"the model: expected {(self.input_height, self.input_width)}, "
                f"received {tuple(x.shape[-2:])}."
            )

        if self.pad_height or self.pad_width:
            x = F.pad(x, (0, self.pad_width, 0, self.pad_height))

        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = self.norm(x).mean(dim=1)
        return self.classifier(x)


def build_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    model_config: dict[str, Any],
) -> nn.Module:
    if len(input_shape) != 3:
        raise ValueError(
            "MLPMixer expects input_shape in (channels, height, width) format."
        )

    return MLPMixer(
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        num_classes=num_classes,
        patch_size=model_config["patch_size"],
        num_blocks=model_config["num_blocks"],
        hidden_dim=model_config["hidden_dim"],
        token_mlp_dim=model_config["token_mlp_dim"],
        channel_mlp_dim=model_config["channel_mlp_dim"],
        dropout_p=model_config["dropout_p"],
    )
