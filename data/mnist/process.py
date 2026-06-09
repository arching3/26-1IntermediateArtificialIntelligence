import gzip
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
IMAGE_SHAPE = (28, 28)
CLASSES = {str(index): str(index) for index in range(10)}
RAW_FILES = {
    "train-images-idx3-ubyte": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}


def _read_bytes(path):
    if not path.exists() and path.suffix != ".gz":
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            path = gz_path
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as file:
            return file.read()
    return path.read_bytes()


def _download_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def _ensure_raw_files():
    for filename, url in RAW_FILES.items():
        path = DATA_DIR / filename
        gz_path = path.with_suffix(path.suffix + ".gz")
        if path.exists() or gz_path.exists():
            continue
        _download_file(url, gz_path)


def _load_images(filename):
    data = np.frombuffer(_read_bytes(DATA_DIR / filename), dtype=np.uint8, offset=16)
    return data.reshape(-1, *IMAGE_SHAPE)


def _load_labels(filename):
    return np.frombuffer(_read_bytes(DATA_DIR / filename), dtype=np.uint8, offset=8)


def _load_split(images_filename, labels_filename):
    return _load_images(images_filename), _load_labels(labels_filename)


def main():
    if not (DATA_DIR / "train.pkl").exists() or not (DATA_DIR / "test.pkl").exists():
        _ensure_raw_files()

    train = _load_split("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    test = _load_split("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")

    with open(DATA_DIR / "train.pkl", "wb") as file:
        pickle.dump(train, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR / "test.pkl", "wb") as file:
        pickle.dump(test, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_DIR / "classes.json", "w", encoding="utf-8") as file:
        json.dump(CLASSES, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
