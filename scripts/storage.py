from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


_RUN_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


class RunStorage:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.diagnostics_dir = self.run_dir / "diagnostics"
        self.config_path = self.run_dir / "config.json"
        self.log_path = self.run_dir / "train.log"
        self.history_json_path = self.run_dir / "history.json"
        self.history_csv_path = self.run_dir / "history.csv"
        self.test_metrics_path = self.run_dir / "test_metrics.json"

    @classmethod
    def create(
        cls,
        storage_root: str | Path,
        dataset_name: str,
        model_name: str,
        run_name: str | None = None,
    ) -> "RunStorage":
        storage_root = Path(storage_root).expanduser().resolve()
        storage_root.mkdir(parents=True, exist_ok=True)

        if run_name:
            directory_name = _sanitize_name(run_name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory_name = "_".join(
                (
                    timestamp,
                    _sanitize_name(dataset_name),
                    _sanitize_name(model_name),
                )
            )

        run_dir = storage_root / directory_name
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")

        storage = cls(run_dir)
        storage.prepare()
        return storage

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "RunStorage":
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path.parent.name != "checkpoints":
            raise ValueError(
                "Resume checkpoint must be inside a run's checkpoints directory."
            )
        return cls(checkpoint_path.parent.parent)

    def prepare(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: Any) -> None:
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary.")
        _write_json_atomic(self.config_path, config)

    def write_test_metrics(self, metrics: dict[str, Any]) -> None:
        _write_json_atomic(self.test_metrics_path, metrics)


def _sanitize_name(value: str) -> str:
    sanitized = _RUN_NAME_PATTERN.sub("-", value.strip()).strip("._-")
    if not sanitized:
        raise ValueError("Run, dataset, and model names must contain a valid character.")
    return sanitized


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
