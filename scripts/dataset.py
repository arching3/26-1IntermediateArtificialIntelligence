import numpy as np

import torch
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from data import load


def _flatten_image(image: torch.Tensor) -> torch.Tensor:
    return torch.flatten(image)


class CustomDataset(Dataset):
    def __init__(self, data, labels, classes, transforms):
        self.labels = torch.as_tensor(labels.copy(), dtype=torch.long)
        self.classes = classes.copy()
        self.data = data.copy()
        self.transforms = transforms
        self.labels = f.one_hot(self.labels, num_classes=len(self.classes))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.transforms(self.data[idx])
        y = self.labels[idx]
        return x, y

    def get_label(self, label: int):
        return self.classes[str(label)]


def get_zscore_stats(data: np.ndarray) -> tuple[list[float], list[float]]:
    if data.ndim == 3:
        mean = [float(data.mean())]
        std = [float(data.std())]
    elif data.ndim == 4 and data.shape[-1] in (1, 3):
        mean = data.mean(axis=(0, 1, 2)).astype(float).tolist()
        std = data.std(axis=(0, 1, 2)).astype(float).tolist()
    elif data.ndim == 4 and data.shape[1] in (1, 3):
        mean = data.mean(axis=(0, 2, 3)).astype(float).tolist()
        std = data.std(axis=(0, 2, 3)).astype(float).tolist()
    else:
        raise ValueError(f"Unsupported image data shape: {data.shape}")

    std = [value if value > 0 else 1.0 for value in std]
    return mean, std


def build(
    dataset_name: str = "mnist",
    split_ratio: float = 0.8,
    flatten: bool = False,
    normalize: bool = False,
    batch_size: int = 64,
    num_workers: int = 1,
    shuffle: bool = True,
    pin_memory: bool = True,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> dict[str, tuple[CustomDataset, DataLoader]]:
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if num_workers < 0:
        raise ValueError("num_workers must be 0 or greater.")

    train, test, classes = load(dataset_name)

    base_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),
    ]

    if normalize:
        mean, std = get_zscore_stats(train[0])
        base_transforms.append(v2.Normalize(mean=mean, std=std))

    if flatten:
        base_transforms.append(v2.Lambda(_flatten_image))

    transforms = v2.Compose(base_transforms)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train[0]), generator=generator).numpy()
    train_size = int(len(indices) * split_ratio)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    train_dataset = CustomDataset(
        train[0][train_indices],
        train[1][train_indices],
        classes,
        transforms,
    )
    valid_dataset = CustomDataset(
        train[0][valid_indices],
        train[1][valid_indices],
        classes,
        transforms,
    )
    test_dataset = CustomDataset(test[0], test[1], classes, transforms)

    device = torch.device(device)
    use_pin_memory = pin_memory and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        generator=generator if shuffle else None,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return {
        "train": (train_dataset, train_loader),
        "valid": (valid_dataset, valid_loader),
        "test": (test_dataset, test_loader),
    }
