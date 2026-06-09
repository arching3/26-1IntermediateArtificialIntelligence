from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from models import build_model
from scripts.checkpoint import CheckpointManager
from scripts.dataset import build as build_dataset
from scripts.evaluation import (
    evaluate_model,
    load_history,
    load_json_object,
    summarize_history,
    write_json_atomic,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage",
        required=True,
        help="Training run directory containing config.json and checkpoints/best.pt.",
    )
    return parser


def evaluate_storage(storage_path: str | Path) -> dict[str, Any]:
    run_dir = Path(storage_path).expanduser().resolve()
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "checkpoints" / "best.pt"

    _require_file(config_path)
    _require_file(history_path)
    _require_file(checkpoint_path)

    run_config = load_json_object(config_path)
    training = _require_dict(run_config, "training")
    model_config = _require_dict(run_config, "model")
    model_name = _require_string(model_config, "name")
    model_parameters = _require_dict(model_config, "parameters")

    device = _resolve_evaluation_device(int(training.get("gpu_id", -1)))
    checkpoint_manager = CheckpointManager(checkpoint_path.parent)
    checkpoint = checkpoint_manager.load(checkpoint_path, map_location=device)
    _validate_checkpoint(
        checkpoint,
        model_name=model_name,
        model_parameters=model_parameters,
    )

    input_shape = tuple(int(value) for value in checkpoint["input_shape"])
    num_classes = int(checkpoint["num_classes"])
    model = build_model(
        model_name=model_name,
        input_shape=input_shape,
        num_classes=num_classes,
        model_config=model_parameters,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    datasets = build_dataset(
        dataset_name=_require_string(training, "dataset_path"),
        split_ratio=float(training.get("train_ratio", 0.8)),
        flatten=_model_bool(model_parameters, "flatten", False),
        normalize=_model_bool(model_parameters, "normalize", False),
        batch_size=int(training.get("batch_size", 64)),
        num_workers=int(training.get("num_workers", 0)),
        shuffle=False,
        pin_memory=True,
        device=device,
        seed=int(training.get("seed", 42)),
    )
    test_dataset, test_loader = datasets["test"]
    if len(test_dataset.classes) != num_classes:
        raise ValueError(
            "Dataset class count does not match checkpoint: "
            f"dataset={len(test_dataset.classes)}, checkpoint={num_classes}."
        )

    metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        num_classes=num_classes,
    )
    history_summary = summarize_history(load_history(history_path))
    result = {
        "storage": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "model": {
            "name": model_name,
            "parameters": model_parameters,
        },
        "best_epoch": checkpoint["best_state"].get("epoch"),
        "best_monitor": checkpoint["best_state"].get("monitor"),
        "best_metric": checkpoint["best_state"].get("metric"),
        "test_metrics": metrics.as_dict(),
        "history_average": history_summary["average"],
        "last_epoch": history_summary["last_epoch"],
        "total_epochs": history_summary["total_epochs"],
        "total_training_seconds": history_summary["total_training_seconds"],
    }
    write_json_atomic(run_dir / "evaluation.json", result)
    return result


def print_evaluation(result: dict[str, Any]) -> None:
    print(f"Evaluation: {Path(result['storage']).name}")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Best epoch: {result['best_epoch']}")
    print(
        f"Best monitor: {result['best_monitor']}="
        f"{_format_value(result['best_metric'])}"
    )
    model = result["model"]
    print(f"\nModel Parameters ({model['name']})")
    print(json.dumps(model["parameters"], indent=2, ensure_ascii=True))

    print("\nTest Metrics")
    for key, value in result["test_metrics"].items():
        print(f"{key:<24} {_format_value(value)}")

    print(f"\nHistory Average ({result['total_epochs']} epochs)")
    for key, value in result["history_average"].items():
        print(f"{key:<24} {_format_value(value)}")
    print(
        f"{'total_training_seconds':<24} "
        f"{_format_value(result['total_training_seconds'])}"
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_argparser().parse_args(argv)
    result = evaluate_storage(args.storage)
    print_evaluation(result)
    return result


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    model_name: str,
    model_parameters: dict[str, Any],
) -> None:
    if checkpoint.get("model_name") != model_name:
        raise ValueError("Run config and checkpoint model names do not match.")
    checkpoint_model = _require_dict(checkpoint, "model")
    if checkpoint_model.get("parameters") != model_parameters:
        raise ValueError("Run config and checkpoint model parameters do not match.")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def _require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be an object.")
    return value


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string.")
    return value


def _model_bool(
    parameters: dict[str, Any],
    key: str,
    default: bool,
) -> bool:
    value = parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"Model parameter '{key}' must be a boolean.")
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _resolve_evaluation_device(gpu_id: int) -> torch.device:
    if (
        gpu_id >= 0
        and torch.cuda.is_available()
        and gpu_id < torch.cuda.device_count()
    ):
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
