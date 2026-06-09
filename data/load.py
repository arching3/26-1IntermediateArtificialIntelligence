import pickle
import json
import numpy as np
from pathlib import Path
import subprocess
import sys

_TARGET_ALIASES = {
    "cifar100": "cifar100",
    "fashion": "fashion",
    "fashionmnist": "fashion",
    "mnist": "mnist",
}


def _normalize_target(target: str) -> str:
    if not isinstance(target, str):
        raise AttributeError("target must be cifar100, fashion or mnist")

    normalized = "".join(char for char in target.lower().strip() if char.isalnum())

    if normalized not in _TARGET_ALIASES:
        raise AttributeError("target must be cifar100, fashion or mnist")

    return _TARGET_ALIASES[normalized]


def _ensure_dataset(data_dir: Path) -> None:
    required_files = ["train.pkl", "test.pkl", "classes.json"]
    if all((data_dir / name).exists() for name in required_files):
        return

    subprocess.run(
        [sys.executable, "process.py"],
        cwd=data_dir,
        check=True,
    )


def load(target:str="cifar100")-> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], dict]:
    target = _normalize_target(target)
    data_dir = Path(__file__).resolve().parent / target
    _ensure_dataset(data_dir)

    with open(data_dir / "train.pkl", "rb") as train_file:
        train = pickle.load(train_file)

    with open(data_dir / "test.pkl", "rb") as test_file:
        test = pickle.load(test_file)

    with open(data_dir / "classes.json", "r", encoding="utf-8") as classes_file:
        classes = json.load(classes_file)

    return train, test, classes
