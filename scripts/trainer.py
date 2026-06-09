from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from scripts.checkpoint import BestModelState, CheckpointManager
from scripts.diagnostics import diagnostic_epochs, save_weight_diagnostics
from scripts.history import HistoryStore
from scripts.metrics import ClassificationMetrics, ClassificationMetricTracker
from scripts.storage import RunStorage


Scheduler = LRScheduler | ReduceLROnPlateau


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int,
        config: Any,
        storage: RunStorage,
        logger: logging.Logger,
        scheduler: Scheduler | None = None,
        history: list[dict[str, Any]] | None = None,
        best_state: BestModelState | None = None,
        start_epoch: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.config = config
        self.storage = storage
        self.logger = logger
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.history = HistoryStore(
            storage.history_json_path,
            storage.history_csv_path,
            records=history,
        )
        self.best_state = best_state or BestModelState(
            monitor=config.monitor,
            mode=config.monitor_mode,
        )
        self.checkpoints = CheckpointManager(
            storage.checkpoint_dir,
            interval=config.checkpoint_interval,
        )

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
    ) -> ClassificationMetrics:
        if self.start_epoch == 1:
            self._save_diagnostics("initial")

        scheduled_diagnostics = diagnostic_epochs(self.config.epochs)

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            epoch_started_at = time.perf_counter()
            train_metrics = self._run_epoch(train_loader, training=True, epoch=epoch)
            valid_metrics = self._run_epoch(valid_loader, training=False, epoch=epoch)
            epoch_seconds = time.perf_counter() - epoch_started_at

            self._step_scheduler(valid_metrics.loss)
            learning_rate = self.optimizer.param_groups[0]["lr"]
            record = self._history_record(
                epoch,
                train_metrics,
                valid_metrics,
                learning_rate,
                epoch_seconds,
            )
            self.history.append(record)

            monitor_value = self._monitor_value(valid_metrics)
            is_best = self.best_state.should_update(
                metric=monitor_value,
                valid_loss=valid_metrics.loss,
                epoch=epoch,
            )
            if is_best:
                self.best_state.update(
                    metric=monitor_value,
                    valid_loss=valid_metrics.loss,
                    epoch=epoch,
                )

            saved_paths = self.checkpoints.save_epoch(
                payload=self._checkpoint_payload(),
                epoch=epoch,
                is_best=is_best,
            )
            self._log_epoch(epoch, train_metrics, valid_metrics, learning_rate, epoch_seconds)
            self.logger.debug(
                "Saved checkpoints: %s",
                ", ".join(str(path) for path in saved_paths),
            )

            if is_best:
                self.logger.info(
                    "Best model updated at epoch %d: %s=%.6f",
                    epoch,
                    self.config.monitor,
                    monitor_value,
                )

            if epoch in scheduled_diagnostics:
                label = (
                    f"final_epoch_{epoch:03d}"
                    if epoch == self.config.epochs
                    else f"middle_epoch_{epoch:03d}"
                )
                self._save_diagnostics(label)

        self._restore_best_model()
        test_metrics = self._run_epoch(test_loader, training=False, epoch=None)
        self.storage.write_test_metrics(test_metrics.as_dict())
        self.logger.info("Test metrics: %s", _format_metrics(test_metrics))
        return test_metrics

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        epoch: int | None,
    ) -> ClassificationMetrics:
        self.model.train(mode=training)
        tracker = ClassificationMetricTracker(self.num_classes)
        phase = "train" if training else "eval"

        for batch_index, batch in enumerate(loader, start=1):
            inputs, one_hot_targets = self._move_batch(batch)
            target_indices = one_hot_targets.argmax(dim=1)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = self.model(inputs)
                self._check_logits(logits)
                loss = self.criterion(logits, target_indices)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite {phase} loss at epoch {epoch}, batch {batch_index}."
                    )
                if training:
                    loss.backward()
                    self.optimizer.step()

            tracker.update(logits, one_hot_targets, loss)

            if (
                self.config.log_interval > 0
                and batch_index % self.config.log_interval == 0
            ):
                self.logger.debug(
                    "%s epoch=%s batch=%d/%d loss=%.6f",
                    phase,
                    epoch if epoch is not None else "test",
                    batch_index,
                    len(loader),
                    float(loss.detach().item()),
                )

        return tracker.compute()

    def _move_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, one_hot_targets = batch
        non_blocking = self.device.type == "cuda"
        return (
            inputs.to(self.device, non_blocking=non_blocking),
            one_hot_targets.to(self.device, non_blocking=non_blocking),
        )

    def _check_logits(self, logits: torch.Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                "Model output must have shape (batch, num_classes); "
                f"received {tuple(logits.shape)}."
            )
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Model output contains NaN or Inf values.")

    def _monitor_value(self, metrics: ClassificationMetrics) -> float:
        values = {
            "valid_loss": metrics.loss,
            "valid_accuracy": metrics.accuracy,
            "valid_f1": metrics.f1,
        }
        return values[self.config.monitor]

    def _step_scheduler(self, valid_loss: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(valid_loss)
        else:
            self.scheduler.step()

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "model": {
                "name": self.config.model_name,
                "parameters": self.config.model_parameters,
            },
            "source": self.config.model_source,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "best_state": self.best_state.as_dict(),
            "training": {
                key: value
                for key, value in asdict(self.config).items()
                if key not in {"model_parameters", "model_source"}
            },
            "history": self.history.records,
            "input_shape": list(self.config.input_shape),
            "num_classes": self.num_classes,
        }

    def _restore_best_model(self) -> None:
        payload = self.checkpoints.load(self.checkpoints.best_path, self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.logger.info(
            "Restored best model from epoch %s.",
            payload["best_state"]["epoch"],
        )

    def _save_diagnostics(self, label: str) -> None:
        output_dir = self.storage.diagnostics_dir / label
        if output_dir.exists():
            self.logger.debug("Skipping existing diagnostics: %s", output_dir)
            return
        save_weight_diagnostics(
            self.model,
            output_dir,
            max_layers=self.config.diagnostic_max_layers,
            max_filters=self.config.diagnostic_max_filters,
            histogram_bins=self.config.histogram_bins,
        )
        self.logger.info("Saved weight diagnostics: %s", output_dir)

    def _history_record(
        self,
        epoch: int,
        train_metrics: ClassificationMetrics,
        valid_metrics: ClassificationMetrics,
        learning_rate: float,
        epoch_seconds: float,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_seconds,
        }
        record.update(_prefixed_metrics("train", train_metrics))
        record.update(_prefixed_metrics("valid", valid_metrics))
        return record

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: ClassificationMetrics,
        valid_metrics: ClassificationMetrics,
        learning_rate: float,
        epoch_seconds: float,
    ) -> None:
        self.logger.info(
            "Epoch %03d/%03d | train %s | valid %s | lr=%.6g | %.2fs",
            epoch,
            self.config.epochs,
            _format_metrics(train_metrics),
            _format_metrics(valid_metrics),
            learning_rate,
            epoch_seconds,
        )


def _prefixed_metrics(
    prefix: str,
    metrics: ClassificationMetrics,
) -> dict[str, float | int]:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.as_dict().items()
    }


def _format_metrics(metrics: ClassificationMetrics) -> str:
    return (
        f"loss={metrics.loss:.6f} "
        f"acc={metrics.accuracy:.4f} "
        f"top5={metrics.top5_accuracy:.4f} "
        f"precision={metrics.precision:.4f} "
        f"recall={metrics.recall:.4f} "
        f"f1={metrics.f1:.4f}"
    )
