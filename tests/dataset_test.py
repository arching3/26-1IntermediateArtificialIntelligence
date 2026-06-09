from scripts.dataset import build


DATASETS = ("mnist", "fashion", "cifar100")


def channel_stats(x):
    if x.ndim == 4:
        return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))
    if x.ndim == 3:
        return x.mean(), x.std()
    raise ValueError(f"Unsupported batch shape: {tuple(x.shape)}")


def inspect_dataset(name):
    datasets = build(
        dataset_name=name,
        normalize=True,
        batch_size=256,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    print(f"\n[{name}]")
    for split in ("train", "valid", "test"):
        dataset, loader = datasets[split]
        x, y = next(iter(loader))
        mean, std = channel_stats(x)

        print(f"{split:>5} dataset_len={len(dataset)}")
        print(f"{split:>5} x_shape={tuple(x.shape)} y_shape={tuple(y.shape)}")
        print(f"{split:>5} x_dtype={x.dtype} y_dtype={y.dtype}")
        print(f"{split:>5} mean={mean}")
        print(f"{split:>5} std={std}")


def main():
    for name in DATASETS:
        inspect_dataset(name)


if __name__ == "__main__":
    main()
