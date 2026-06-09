from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
DEFAULT_BACKUP_PATH = Path(__file__).with_name("config.backup.json")


@dataclass(frozen=True)
class ModelConfiguration:
    name: str
    parameters: dict[str, Any]
    source_type: str
    source_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": deepcopy(self.parameters),
            "source": {
                "type": self.source_type,
                "path": self.source_path,
            },
        }


def validate_model_name(model_name: str) -> None:
    if (
        not isinstance(model_name, str)
        or not model_name
        or model_name != model_name.lower()
        or not model_name.isidentifier()
    ):
        raise ValueError(
            "model_name must be a lowercase Python identifier using underscores."
        )


def load_all_model_configs(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, dict[str, Any]]:
    path = Path(config_path).expanduser().resolve()
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Model config root must be a JSON object.")

    configs: dict[str, dict[str, Any]] = {}
    for model_name, parameters in payload.items():
        validate_model_name(model_name)
        if not isinstance(parameters, dict):
            raise ValueError(
                f"Configuration for '{model_name}' must be a JSON object."
            )
        configs[model_name] = deepcopy(parameters)
    return configs


def available_model_names(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[str, ...]:
    return tuple(load_all_model_configs(config_path))


def load_model_config(
    model_name: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    validate_model_name(model_name)
    configs = load_all_model_configs(config_path)
    if model_name not in configs:
        available = ", ".join(sorted(configs))
        raise KeyError(
            f"No configuration for model '{model_name}'. Available: {available}."
        )
    return deepcopy(configs[model_name])


def load_model_configuration(
    model_name: str,
    source_path: str | Path | None = None,
) -> ModelConfiguration:
    validate_model_name(model_name)
    if source_path is None:
        path = DEFAULT_CONFIG_PATH.resolve()
        parameters = load_model_config(model_name, path)
        return ModelConfiguration(model_name, parameters, "default", str(path))

    path = Path(source_path).expanduser().resolve()
    if path.suffix in {".pt", ".pth"}:
        return load_checkpoint_model_configuration(path, expected_name=model_name)

    payload = _read_json(path)
    if _is_run_config(payload):
        stored_name = payload["model"]["name"]
        if stored_name != model_name:
            raise ValueError(
                f"Requested model '{model_name}' does not match "
                f"stored model '{stored_name}'."
            )
        source = payload.get("source")
        is_tui_override = (
            isinstance(source, dict)
            and source.get("type") == "tui_override"
            and isinstance(source.get("path"), str)
        )
        return ModelConfiguration(
            name=stored_name,
            parameters=deepcopy(payload["model"]["parameters"]),
            source_type="tui_override" if is_tui_override else "previous_run",
            source_path=source["path"] if is_tui_override else str(path),
        )

    parameters = load_model_config(model_name, path)
    source_type = "default" if path == DEFAULT_CONFIG_PATH.resolve() else "config_file"
    return ModelConfiguration(model_name, parameters, source_type, str(path))


def load_checkpoint_model_configuration(
    checkpoint_path: str | Path,
    expected_name: str | None = None,
) -> ModelConfiguration:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary.")

    model_payload = payload.get("model")
    if isinstance(model_payload, dict):
        model_name = model_payload.get("name")
        parameters = model_payload.get("parameters")
    else:
        model_name = payload.get("model_name")
        parameters = payload.get("model_parameters")

    validate_model_name(model_name)
    if not isinstance(parameters, dict):
        raise ValueError("Checkpoint does not contain model parameters.")
    if expected_name is not None and model_name != expected_name:
        raise ValueError(
            f"Requested model '{expected_name}' does not match "
            f"checkpoint model '{model_name}'."
        )

    return ModelConfiguration(
        name=model_name,
        parameters=deepcopy(parameters),
        source_type="checkpoint",
        source_path=str(path),
    )


def save_model_config(
    model_name: str,
    parameters: dict[str, Any],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    backup_path: str | Path = DEFAULT_BACKUP_PATH,
) -> None:
    validate_model_name(model_name)
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dictionary.")

    path = Path(config_path).expanduser().resolve()
    backup = Path(backup_path).expanduser().resolve()
    configs = load_all_model_configs(path)
    configs[model_name] = deepcopy(parameters)

    if path.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    _write_json_atomic(path, configs)


def _is_run_config(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("training"), dict)
        and isinstance(payload.get("model"), dict)
        and isinstance(payload["model"].get("name"), str)
        and isinstance(payload["model"].get("parameters"), dict)
    )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
