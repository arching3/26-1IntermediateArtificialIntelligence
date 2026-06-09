from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from torch import nn

from models.config import available_model_names, validate_model_name


MODEL_NAMES = ("mlp", "simple_cnn", "resnet", "mlp_mixer")

if MODEL_NAMES != available_model_names():
    raise RuntimeError("MODEL_NAMES must match the order and names in models/config.json.")


def build_model(
    model_name: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    model_config: dict[str, Any],
) -> nn.Module:
    validate_model_name(model_name)
    if model_name not in MODEL_NAMES:
        available = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}.")

    module_name = f"models.{model_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise NotImplementedError(
                f"Model '{model_name}' is registered but {module_name}.py "
                "has not been implemented."
            ) from error
        raise

    factory: Callable[..., nn.Module] | None = getattr(module, "build_model", None)
    if factory is None:
        raise AttributeError(f"{module_name} must define build_model().")

    model = factory(
        input_shape=input_shape,
        num_classes=num_classes,
        model_config=model_config,
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"{module_name}.build_model() must return torch.nn.Module.")
    return model
