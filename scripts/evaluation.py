from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.metrics import ClassificationMetrics, ClassificationMetricTracker


EXCLUDED_HISTORY_AVERAGE_KEYS = {
    "epoch",
    "train_samples",
    "valid_samples",
}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> ClassificationMetrics:
    model.eval()
    tracker = ClassificationMetricTracker(num_classes)
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for inputs, one_hot_targets in loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            one_hot_targets = one_hot_targets.to(
                device,
                non_blocking=non_blocking,
            )
            logits = model(inputs)
            _validate_logits(logits, num_classes)
            targets = one_hot_targets.argmax(dim=1)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Evaluation loss is NaN or Inf.")
            tracker.update(logits, one_hot_targets, loss)

    return tracker.compute()


def summarize_history(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("History is empty.")

    numeric_keys: list[str] = []
    for record in records:
        for key, value in record.items():
            if (
                key not in EXCLUDED_HISTORY_AVERAGE_KEYS
                and key not in numeric_keys
                and _is_number(value)
            ):
                numeric_keys.append(key)

    averages: dict[str, float] = {}
    for key in numeric_keys:
        values = [
            float(record[key])
            for record in records
            if key in record and _is_number(record[key])
        ]
        if values:
            averages[key] = sum(values) / len(values)

    return {
        "average": averages,
        "last_epoch": dict(records[-1]),
        "total_epochs": len(records),
        "total_training_seconds": sum(
            float(record.get("epoch_seconds", 0.0))
            for record in records
            if _is_number(record.get("epoch_seconds", 0.0))
        ),
    }


def load_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_history(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not all(
        isinstance(record, dict) for record in payload
    ):
        raise ValueError(f"Expected a list of history records: {path}")
    return payload


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def _validate_logits(logits: torch.Tensor, num_classes: int) -> None:
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(
            "Model output must have shape (batch, num_classes); "
            f"received {tuple(logits.shape)}."
        )
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Model output contains NaN or Inf values.")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
