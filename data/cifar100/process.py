import json
import pickle
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
RAW_ARCHIVE = DATA_DIR / "cifar-100-python.tar.gz"
RAW_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def _load_pickle(name):
    with open(DATA_DIR / name, "rb") as file:
        return pickle.load(file, encoding="bytes")


def _download_and_extract_raw():
    if all((DATA_DIR / name).exists() for name in ("train", "test", "meta")):
        return

    if not RAW_ARCHIVE.exists():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp_file:
            urllib.request.urlretrieve(RAW_URL, tmp_file.name)
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(RAW_ARCHIVE)

    with tarfile.open(RAW_ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            name = Path(member.name).name
            if name not in {"train", "test", "meta"}:
                continue
            target_path = DATA_DIR / name
            with archive.extractfile(member) as source, open(target_path, "wb") as target:
                target.write(source.read())


def _get_value(data, *keys):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of the keys exist: {keys}")


def _load_split(name):
    raw = _load_pickle(name)
    images = np.asarray(_get_value(raw, b"data"))
    labels = np.asarray(
        _get_value(raw, b"fine_labels", b"fine_Labels"),
        dtype=np.int64,
    )

    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def _load_classes():
    meta = _load_pickle("meta")
    names = _get_value(meta, b"fine_label_names")
    names = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in names
    ]
    return {str(index): name for index, name in enumerate(names)}


def main():
    if not (DATA_DIR / "train.pkl").exists() or not (DATA_DIR / "test.pkl").exists():
        _download_and_extract_raw()

    train = _load_split("train")
    test = _load_split("test")
    classes = _load_classes()

    with open(DATA_DIR / "train.pkl", "wb") as file:
        pickle.dump(train, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR / "test.pkl", "wb") as file:
        pickle.dump(test, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR / "classes.json", "w", encoding="utf-8") as file:
        json.dump(classes, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
