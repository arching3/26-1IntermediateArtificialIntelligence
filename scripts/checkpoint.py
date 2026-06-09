from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 2


@dataclass
class BestModelState:
    monitor: str
    mode: str
    metric: float | None = None
    valid_loss: float | None = None
    epoch: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")

    def should_update(
        self,
        metric: float,
        valid_loss: float,
        epoch: int,
    ) -> bool:
        if self.metric is None:
            return True

        if self.mode == "min" and metric < self.metric:
            return True
        if self.mode == "max" and metric > self.metric:
            return True

        if metric == self.metric:
            if self.valid_loss is None or valid_loss < self.valid_loss:
                return True
            if valid_loss == self.valid_loss and self.epoch is not None:
                return epoch < self.epoch

        return False

    def update(self, metric: float, valid_loss: float, epoch: int) -> None:
        self.metric = metric
        self.valid_loss = valid_loss
        self.epoch = epoch

    def as_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "metric": self.metric,
            "valid_loss": self.valid_loss,
            "epoch": self.epoch,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BestModelState":
        return cls(
            monitor=payload["monitor"],
            mode=payload["mode"],
            metric=payload.get("metric"),
            valid_loss=payload.get("valid_loss"),
            epoch=payload.get("epoch"),
        )


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path, interval: int = 0):
        if interval < 0:
            raise ValueError("checkpoint interval must be 0 or greater.")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    def save_epoch(
        self,
        payload: dict[str, Any],
        epoch: int,
        is_best: bool,
    ) -> list[Path]:
        checkpoint = dict(payload)
        checkpoint["format_version"] = CHECKPOINT_FORMAT_VERSION
        checkpoint["epoch"] = epoch

        saved_paths = [self._save_atomic(checkpoint, self.last_path)]

        if is_best:
            saved_paths.append(self._save_atomic(checkpoint, self.best_path))

        if self.interval > 0 and epoch % self.interval == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            saved_paths.append(self._save_atomic(checkpoint, epoch_path))

        return saved_paths

    def load(
        self,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )

        if not isinstance(payload, dict):
            raise TypeError("Checkpoint payload must be a dictionary.")
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format: {payload.get('format_version')}."
            )
        return payload

    def _save_atomic(self, payload: dict[str, Any], path: Path) -> Path:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
        return path
