from __future__ import annotations

import argparse
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader

from models import MODEL_NAMES, build_model
from models.config import ModelConfiguration, load_model_configuration
from scripts.checkpoint import BestModelState, CheckpointManager
from scripts.dataset import CustomDataset, build as build_dataset
from scripts.logging_utils import configure_logging
from scripts.storage import RunStorage
from scripts.trainer import Trainer


Scheduler = LRScheduler | ReduceLROnPlateau


@dataclass
class TrainingConfig:
    model_name: str
    dataset_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gpu_id: int
    seed: int
    train_ratio: float
    num_workers: int
    optimizer: str
    scheduler: str
    monitor: str
    monitor_mode: str
    checkpoint_interval: int
    storage_root: str
    run_name: str | None
    resume: str | None
    model_config_path: str | None
    log_interval: int
    diagnostic_max_layers: int
    diagnostic_max_filters: int
    histogram_bins: int
    verbose: int
    input_shape: tuple[int, ...] = ()
    num_classes: int = 0
    run_dir: str = ""
    model_parameters: dict[str, Any] | None = None
    model_source: dict[str, str] | None = None

    @property
    def n_epochs(self) -> int:
        return self.epochs

    @property
    def lr(self) -> float:
        return self.learning_rate


@dataclass
class TrainingData:
    train_dataset: CustomDataset
    train_loader: DataLoader
    valid_dataset: CustomDataset
    valid_loader: DataLoader
    test_dataset: CustomDataset
    test_loader: DataLoader
    input_shape: tuple[int, ...]
    num_classes: int


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        "--model-name",
        type=str,
        default="mlp",
        choices=MODEL_NAMES,
    )
    parser.add_argument(
        "--dataset",
        "--dataset_path",
        "--dataset-path",
        dest="dataset_path",
        required=True,
        help="Dataset name: mnist, fashion, fashionmnist, or cifar100.",
    )
    parser.add_argument("--epochs", "--n_epochs", "--n-epochs", dest="epochs", type=int, default=40)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=128)
    parser.add_argument("--learning_rate", "--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.001)
    cuda_ = 0 if torch.cuda.is_available() else -1
    parser.add_argument(
        "--gpu_id",
        "--gpu-id",
        type=int,
        default=cuda_,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", "--train-ratio", type=float, default=0.8)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=2)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw", "sgd", "rmsprop"),
        default="adam",
    )
    parser.add_argument(
        "--scheduler",
        choices=("none", "step", "cosine", "plateau"),
        default="cosine",
    )
    parser.add_argument(
        "--monitor",
        choices=("valid_loss", "valid_accuracy", "valid_f1"),
        default="valid_f1",
    )
    parser.add_argument(
        "--monitor_mode",
        "--monitor-mode",
        dest="monitor_mode",
        choices=("min", "max"),
        default="max",
    )
    parser.add_argument(
        "--checkpoint_interval",
        "--checkpoint-interval",
        dest="checkpoint_interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--storage_root",
        "--storage-root",
        dest="storage_root",
        default="./storage",
    )
    parser.add_argument("--run_name", "--run-name", dest="run_name")
    parser.add_argument("--resume", help="Checkpoint path to resume.")
    parser.add_argument(
        "--model_config_path",
        "--model-config-path",
        dest="model_config_path",
        help=(
            "Model config registry, previous run config.json, or checkpoint. "
            "Defaults to models/config.json."
        ),
    )
    parser.add_argument("--log_interval", "--log-interval", dest="log_interval", type=int, default=10)
    parser.add_argument(
        "--diagnostic_max_layers",
        "--diagnostic-max-layers",
        dest="diagnostic_max_layers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--diagnostic_max_filters",
        "--diagnostic-max-filters",
        dest="diagnostic_max_filters",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--histogram_bins",
        "--histogram-bins",
        dest="histogram_bins",
        type=int,
        default=50,
    )
    parser.add_argument("--verbose", type=int, default=1)
    return parser


def parse_config(argv: list[str] | None = None) -> TrainingConfig:
    args = build_argparser().parse_args(argv)
    return TrainingConfig(**vars(args))


def resolve_device(gpu_id: int) -> torch.device:
    if gpu_id < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --gpu-id -1 for CPU.")
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"gpu_id {gpu_id} is invalid; "
            f"{torch.cuda.device_count()} CUDA device(s) available."
        )
    return torch.device(f"cuda:{gpu_id}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_data(
    config: TrainingConfig,
    device: torch.device,
) -> TrainingData:
    model_parameters = config.model_parameters or {}
    datasets = build_dataset(
        dataset_name=config.dataset_path,
        split_ratio=config.train_ratio,
        flatten=_model_bool(model_parameters, "flatten", False),
        normalize=_model_bool(model_parameters, "normalize", False),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
        device=device,
        seed=config.seed,
    )
    train_dataset, train_loader = datasets["train"]
    valid_dataset, valid_loader = datasets["valid"]
    test_dataset, test_loader = datasets["test"]
    sample_x, sample_y = train_dataset[0]

    return TrainingData(
        train_dataset=train_dataset,
        train_loader=train_loader,
        valid_dataset=valid_dataset,
        valid_loader=valid_loader,
        test_dataset=test_dataset,
        test_loader=test_loader,
        input_shape=tuple(sample_x.shape),
        num_classes=int(sample_y.numel()),
    )


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    factories = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
    }
    return factories[config.optimizer](model.parameters(), **kwargs)


def build_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
) -> Scheduler | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "step":
        return StepLR(optimizer, step_size=max(1, config.epochs // 3), gamma=0.1)
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)


def prepare_storage(config: TrainingConfig) -> RunStorage:
    if config.resume:
        storage = RunStorage.from_checkpoint(config.resume)
        storage.prepare()
        return storage
    return RunStorage.create(
        storage_root=config.storage_root,
        dataset_name=config.dataset_path,
        model_name=config.model_name,
        run_name=config.run_name,
    )


def restore_training_state(
    config: TrainingConfig,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Scheduler | None,
    checkpoint_manager: CheckpointManager,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], BestModelState]:
    if not config.resume:
        return 1, [], BestModelState(config.monitor, config.monitor_mode)

    payload = checkpoint_manager.load(config.resume, map_location=device)
    if payload["model_name"] != config.model_name:
        raise ValueError("Checkpoint model_name does not match the requested model.")
    if payload["model"]["parameters"] != config.model_parameters:
        raise ValueError("Checkpoint model parameters do not match this run.")
    if tuple(payload["input_shape"]) != config.input_shape:
        raise ValueError("Checkpoint input shape does not match the dataset.")
    if payload["num_classes"] != config.num_classes:
        raise ValueError("Checkpoint class count does not match the dataset.")
    checkpoint_best_state = BestModelState.from_dict(payload["best_state"])
    if (
        checkpoint_best_state.monitor != config.monitor
        or checkpoint_best_state.mode != config.monitor_mode
    ):
        raise ValueError("Checkpoint monitor configuration does not match this run.")

    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return (
        int(payload["epoch"]) + 1,
        list(payload.get("history", [])),
        checkpoint_best_state,
    )


def main(config: TrainingConfig) -> dict[str, float | int]:
    _validate_config(config)
    set_seed(config.seed)
    device = resolve_device(config.gpu_id)
    model_configuration = resolve_model_configuration(config)
    config.model_parameters = model_configuration.parameters
    config.model_source = {
        "type": model_configuration.source_type,
        "path": model_configuration.source_path,
    }
    storage = prepare_storage(config)
    logger = configure_logging(storage.log_path, config.verbose)

    try:
        training_data = load_training_data(config, device)
        config.input_shape = training_data.input_shape
        config.num_classes = training_data.num_classes
        config.run_dir = str(storage.run_dir)
        storage.write_config(_run_config_payload(config))

        model = build_model(
            config.model_name,
            input_shape=config.input_shape,
            num_classes=config.num_classes,
            model_config=config.model_parameters,
        ).to(device)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        criterion = nn.CrossEntropyLoss()
        checkpoint_manager = CheckpointManager(
            storage.checkpoint_dir,
            interval=config.checkpoint_interval,
        )
        start_epoch, history, best_state = restore_training_state(
            config,
            model,
            optimizer,
            scheduler,
            checkpoint_manager,
            device,
        )

        _log_run_summary(logger, config, training_data, model, device)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=config.num_classes,
            config=config,
            storage=storage,
            logger=logger,
            scheduler=scheduler,
            history=history,
            best_state=best_state,
            start_epoch=start_epoch,
        )
        test_metrics = trainer.fit(
            training_data.train_loader,
            training_data.valid_loader,
            training_data.test_loader,
        )
        return test_metrics.as_dict()
    except Exception:
        logger.exception("Training failed.")
        raise


def _validate_config(config: TrainingConfig) -> None:
    positive_values = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "diagnostic_max_layers": config.diagnostic_max_layers,
        "diagnostic_max_filters": config.diagnostic_max_filters,
        "histogram_bins": config.histogram_bins,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")
    if config.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be 0 or greater.")
    if config.log_interval < 0:
        raise ValueError("log_interval must be 0 or greater.")
    expected_mode = "min" if config.monitor == "valid_loss" else "max"
    if config.monitor_mode != expected_mode:
        raise ValueError(
            f"{config.monitor} must use monitor_mode='{expected_mode}'."
        )


def resolve_model_configuration(config: TrainingConfig) -> ModelConfiguration:
    source_path = config.resume or config.model_config_path
    return load_model_configuration(config.model_name, source_path)


def _model_bool(
    parameters: dict[str, Any],
    name: str,
    default: bool,
) -> bool:
    value = parameters.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"Model parameter '{name}' must be a boolean.")
    return value


def _training_config_dict(config: TrainingConfig) -> dict[str, Any]:
    excluded = {"model_parameters", "model_source"}
    return {
        key: value
        for key, value in asdict(config).items()
        if key not in excluded
    }


def _run_config_payload(config: TrainingConfig) -> dict[str, Any]:
    return {
        "training": _training_config_dict(config),
        "model": {
            "name": config.model_name,
            "parameters": config.model_parameters,
        },
        "source": config.model_source,
    }


def _log_run_summary(
    logger: logging.Logger,
    config: TrainingConfig,
    data: TrainingData,
    model: nn.Module,
    device: torch.device,
) -> None:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info("Run directory: %s", config.run_dir)
    logger.info(
        "Environment: Python=%s PyTorch=%s device=%s",
        platform.python_version(),
        torch.__version__,
        device,
    )
    logger.info(
        "Dataset %s: train=%d valid=%d test=%d input=%s classes=%d",
        config.dataset_path,
        len(data.train_dataset),
        len(data.valid_dataset),
        len(data.test_dataset),
        data.input_shape,
        data.num_classes,
    )
    logger.info(
        "Model %s: parameters=%d trainable=%d",
        config.model_name,
        parameter_count,
        trainable_count,
    )
    logger.debug("Configuration: %s", asdict(config))
    logger.debug("Model:\n%s", model)


if __name__ == "__main__":
    main(parse_config())
