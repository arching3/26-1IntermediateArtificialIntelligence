from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt


SUPPORTED_WEIGHT_MODULES = (nn.Conv2d, nn.Linear)


def save_weight_diagnostics(
    model: nn.Module,
    output_dir: str | Path,
    max_layers: int = 6,
    max_filters: int = 64,
    histogram_bins: int = 50,
) -> None:
    if max_layers <= 0:
        raise ValueError("max_layers must be greater than 0.")
    if max_filters <= 0:
        raise ValueError("max_filters must be greater than 0.")
    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be greater than 0.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_layers = _select_weight_layers(model, max_layers)

    if not selected_layers:
        raise ValueError("Model has no supported Conv2d or Linear weight layers.")

    _save_histograms(
        selected_layers,
        output_dir / "histograms.png",
        histogram_bins,
    )
    _save_weight_visualizations(
        selected_layers,
        output_dir / "weights.png",
        max_filters,
    )
    _save_statistics(selected_layers, output_dir / "statistics.json")


def diagnostic_epochs(total_epochs: int) -> set[int]:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be greater than 0.")
    return {math.ceil(total_epochs / 2), total_epochs}


def _select_weight_layers(
    model: nn.Module,
    max_layers: int,
) -> list[tuple[str, nn.Module]]:
    layers = [
        (name or module.__class__.__name__, module)
        for name, module in model.named_modules()
        if isinstance(module, SUPPORTED_WEIGHT_MODULES)
        and module.weight is not None
    ]

    if len(layers) <= max_layers:
        return layers

    indices = np.linspace(0, len(layers) - 1, num=max_layers, dtype=int)
    return [layers[index] for index in dict.fromkeys(indices.tolist())]


def _save_histograms(
    layers: list[tuple[str, nn.Module]],
    output_path: Path,
    bins: int,
) -> None:
    figure, axes = plt.subplots(
        len(layers),
        1,
        figsize=(10, max(3, len(layers) * 2.8)),
        squeeze=False,
    )

    for axis, (name, module) in zip(axes[:, 0], layers):
        weights = module.weight.detach().to("cpu", dtype=torch.float32).flatten().numpy()
        axis.hist(weights, bins=bins, color="#3567a8", alpha=0.85)
        axis.set_title(f"{name} | {module.__class__.__name__}")
        axis.set_xlabel("Weight value")
        axis.set_ylabel("Count")
        axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, format="png")
    plt.close(figure)


def _save_weight_visualizations(
    layers: list[tuple[str, nn.Module]],
    output_path: Path,
    max_filters: int,
) -> None:
    figure, axes = plt.subplots(
        len(layers),
        1,
        figsize=(12, max(4, len(layers) * 4)),
        squeeze=False,
    )

    for axis, (name, module) in zip(axes[:, 0], layers):
        weights = module.weight.detach().to("cpu", dtype=torch.float32)
        if isinstance(module, nn.Conv2d):
            image = _conv_kernel_grid(weights, max_filters)
            axis.imshow(image, cmap=None if image.ndim == 3 else "coolwarm")
        else:
            image = weights.numpy()
            limit = float(np.max(np.abs(image))) or 1.0
            axis.imshow(
                image,
                cmap="coolwarm",
                aspect="auto",
                vmin=-limit,
                vmax=limit,
            )

        axis.set_title(f"{name} | {module.__class__.__name__} {tuple(weights.shape)}")
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, format="png")
    plt.close(figure)


def _conv_kernel_grid(weights: torch.Tensor, max_filters: int) -> np.ndarray:
    filters = weights[:max_filters]
    if filters.shape[1] == 1:
        images = filters[:, 0].numpy()
    elif filters.shape[1] == 3:
        images = filters.permute(0, 2, 3, 1).numpy()
        images = np.stack([_normalize_image(image) for image in images])
    else:
        images = torch.linalg.vector_norm(filters, dim=1).numpy()

    columns = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / columns)
    height, width = images.shape[1:3]

    if images.ndim == 4:
        grid = np.zeros((rows * height, columns * width, images.shape[3]))
    else:
        grid = np.zeros((rows * height, columns * width))

    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        grid[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
            ...,
        ] = image

    return grid


def _normalize_image(image: np.ndarray) -> np.ndarray:
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum == minimum:
        return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum)


def _save_statistics(
    layers: list[tuple[str, nn.Module]],
    output_path: Path,
) -> None:
    statistics: dict[str, dict[str, Any]] = {}

    for name, module in layers:
        weights = module.weight.detach().to("cpu", dtype=torch.float64)
        finite = torch.isfinite(weights)
        finite_weights = weights[finite]
        non_finite_count = int((~finite).sum().item())

        if finite_weights.numel() > 0:
            layer_statistics = {
                "mean": float(finite_weights.mean().item()),
                "std": float(finite_weights.std(unbiased=False).item()),
                "min": float(finite_weights.min().item()),
                "max": float(finite_weights.max().item()),
                "l2_norm": float(torch.linalg.vector_norm(finite_weights).item()),
                "zero_ratio": float((finite_weights == 0).sum().item())
                / finite_weights.numel(),
            }
        else:
            layer_statistics = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "l2_norm": None,
                "zero_ratio": None,
            }

        statistics[name] = {
            "module": module.__class__.__name__,
            "shape": list(weights.shape),
            "parameter_count": weights.numel(),
            "non_finite_count": non_finite_count,
            **layer_statistics,
        }

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=2, ensure_ascii=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, output_path)
