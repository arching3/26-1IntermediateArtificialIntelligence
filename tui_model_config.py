from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.config import (
    ModelConfiguration,
    load_checkpoint_model_configuration,
    load_model_configuration,
)


@dataclass(frozen=True)
class PreviousRun:
    config_path: Path
    run_dir: Path
    model_name: str
    dataset_name: str
    epochs: int | None
    best_checkpoint: Path | None
    last_checkpoint: Path | None

    @property
    def label(self) -> str:
        epochs = f"{self.epochs} epochs" if self.epochs is not None else "epochs unknown"
        return (
            f"{self.run_dir.name} | {self.dataset_name} | "
            f"{self.model_name} | {epochs}"
        )


@dataclass
class ModelConfigSelection:
    name: str
    parameters: dict[str, Any]
    source_type: str
    source_path: str
    training: dict[str, Any] = field(default_factory=dict)
    action: str = "use_model_parameters"
    resume_path: str | None = None
    evaluation_path: str | None = None
    temporary_path: str | None = None
    dirty: bool = False

    def copy(self) -> "ModelConfigSelection":
        return ModelConfigSelection(
            name=self.name,
            parameters=deepcopy(self.parameters),
            source_type=self.source_type,
            source_path=self.source_path,
            training=deepcopy(self.training),
            action=self.action,
            resume_path=self.resume_path,
            evaluation_path=self.evaluation_path,
            dirty=self.dirty,
        )


def discover_previous_runs(
    storage_root: str | Path = "./storage",
) -> list[PreviousRun]:
    root = Path(storage_root).expanduser()
    if not root.exists():
        return []

    runs: list[PreviousRun] = []
    for config_path in root.glob("*/config.json"):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            training = payload["training"]
            model = payload["model"]
            model_name = model["name"]
            dataset_name = training["dataset_path"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue

        run_dir = config_path.parent
        best_path = run_dir / "checkpoints" / "best.pt"
        last_path = run_dir / "checkpoints" / "last.pt"
        runs.append(
            PreviousRun(
                config_path=config_path.resolve(),
                run_dir=run_dir.resolve(),
                model_name=model_name,
                dataset_name=dataset_name,
                epochs=training.get("epochs"),
                best_checkpoint=best_path.resolve() if best_path.is_file() else None,
                last_checkpoint=last_path.resolve() if last_path.is_file() else None,
            )
        )

    return sorted(runs, key=lambda run: run.run_dir.name, reverse=True)


def selection_from_default(model_name: str) -> ModelConfigSelection:
    return _selection_from_configuration(load_model_configuration(model_name))


def selection_from_run(run: PreviousRun) -> ModelConfigSelection:
    payload = json.loads(run.config_path.read_text(encoding="utf-8"))
    configuration = load_model_configuration(run.model_name, run.config_path)
    selection = _selection_from_configuration(configuration)
    selection.training = deepcopy(payload["training"])
    selection.resume_path = str(run.last_checkpoint) if run.last_checkpoint else None
    selection.evaluation_path = (
        str(run.best_checkpoint) if run.best_checkpoint else selection.resume_path
    )
    return selection


def selection_from_checkpoint(
    checkpoint_path: str | Path,
) -> ModelConfigSelection:
    path = Path(checkpoint_path).expanduser().resolve()
    configuration = load_checkpoint_model_configuration(path)
    selection = _selection_from_configuration(configuration)
    selection.resume_path = str(path)
    selection.evaluation_path = str(path)

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    training = payload.get("training", {})
    if isinstance(training, dict):
        selection.training = deepcopy(training)
    return selection


def parse_parameter_value(raw_value: str, current_value: Any) -> Any:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON value: {error.msg}.") from error

    if current_value is None:
        return parsed
    if isinstance(current_value, bool):
        if not isinstance(parsed, bool):
            raise ValueError("Expected a JSON boolean: true or false.")
        return parsed
    if isinstance(current_value, int):
        if not isinstance(parsed, int) or isinstance(parsed, bool):
            raise ValueError("Expected a JSON integer.")
        return parsed
    if isinstance(current_value, float):
        if not isinstance(parsed, (int, float)) or isinstance(parsed, bool):
            raise ValueError("Expected a JSON number.")
        return float(parsed)
    if not isinstance(parsed, type(current_value)):
        raise ValueError(f"Expected JSON type {type(current_value).__name__}.")
    return parsed


def materialize_selection(selection: ModelConfigSelection) -> str:
    if selection.temporary_path and Path(selection.temporary_path).is_file():
        return selection.temporary_path

    payload = {
        "training": {},
        "model": {
            "name": selection.name,
            "parameters": selection.parameters,
        },
        "source": {
            "type": "tui_override",
            "path": selection.source_path,
        },
    }
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f"ml_project_{selection.name}_",
        suffix=".json",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=True)
            file.write("\n")
    except Exception:
        os.unlink(temporary_path)
        raise

    selection.temporary_path = temporary_path
    return temporary_path


def cleanup_selection(selection: ModelConfigSelection | None) -> None:
    if selection is None or not selection.temporary_path:
        return
    try:
        Path(selection.temporary_path).unlink(missing_ok=True)
    finally:
        selection.temporary_path = None


def _selection_from_configuration(
    configuration: ModelConfiguration,
) -> ModelConfigSelection:
    return ModelConfigSelection(
        name=configuration.name,
        parameters=deepcopy(configuration.parameters),
        source_type=configuration.source_type,
        source_path=configuration.source_path,
    )
