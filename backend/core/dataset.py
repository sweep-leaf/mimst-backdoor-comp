"""MNIST 数据加载与划分。

负责：
- 从公开镜像下载 MNIST 原始 idx 文件并缓存到 ``data/``；
- 解析为 ``[N, 1, 28, 28] float32 ∈[0,1]`` 图像与 ``[N] int64`` 标签；
- 划分公开训练集 / 公开验证集 / 隐藏测试集，并支持快速模式子集。

不依赖 torchvision，避免与 numpy 2 的二进制兼容问题。
"""
from __future__ import annotations

import gzip
import io
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# 稳定的 MNIST 镜像（yann.lecun 官方源经常不可用）。
_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


@dataclass
class Dataset:
    """一组图像 / 标签。"""

    images: np.ndarray   # [N, 1, 28, 28] float32 ∈[0,1]
    labels: np.ndarray   # [N] int64

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def subset(self, n: int, seed: int = 0) -> "Dataset":
        """按种子随机抽取前 n 个样本（n<=0 表示全量）。"""
        if n is None or n <= 0 or n >= len(self):
            return self
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self), size=n, replace=False)
        idx.sort()
        return Dataset(self.images[idx], self.labels[idx])


def _default_data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # backend/core/dataset.py -> 项目根目录下的 data/
    root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(root, "data")


def _download_file(name: str, dest: str, timeout: float = 60.0) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last_err: Optional[Exception] = None
    for mirror in _MNIST_MIRRORS:
        url = mirror + _FILES[name]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mnist-backend/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(
        f"无法下载 MNIST 文件 {_FILES[name]}：{last_err}。"
        f"请检查网络，或手动下载到 {dest}。"
    )


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[:4], "big")
    if magic != 2051:
        raise ValueError(f"非法的图像 idx 文件 {path}：magic={magic}")
    n = int.from_bytes(raw[4:8], "big")
    rows = int.from_bytes(raw[8:12], "big")
    cols = int.from_bytes(raw[12:16], "big")
    arr = np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, rows, cols)
    arr = arr.astype(np.float32) / 255.0
    return arr[:, None, :, :]  # [N, 1, H, W]


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[:4], "big")
    if magic != 2049:
        raise ValueError(f"非法的标签 idx 文件 {path}：magic={magic}")
    n = int.from_bytes(raw[4:8], "big")
    arr = np.frombuffer(raw[8:8 + n], dtype=np.uint8).astype(np.int64)
    return arr


def load_raw(data_dir: Optional[str] = None) -> Tuple[Dataset, Dataset]:
    """加载完整 MNIST，返回 (train, test) 两个 :class:`Dataset`。"""
    data_dir = data_dir or _default_data_dir()
    paths = {}
    for name, fname in _FILES.items():
        dest = os.path.join(data_dir, fname)
        _download_file(name, dest)
        paths[name] = dest
    train = Dataset(
        images=_read_idx_images(paths["train_images"]),
        labels=_read_idx_labels(paths["train_labels"]),
    )
    test = Dataset(
        images=_read_idx_images(paths["test_images"]),
        labels=_read_idx_labels(paths["test_labels"]),
    )
    return train, test


@dataclass
class Split:
    """公开训练集 / 公开验证集 / 隐藏测试集。"""

    train: Dataset
    val: Dataset
    test: Dataset


def load_split(
    data_dir: Optional[str] = None,
    train_subset: int = 0,
    val_subset: int = 0,
    test_subset: int = 0,
    val_split: int = 0,
    seed: int = 0,
) -> Split:
    """加载并划分数据集。

    - ``train_subset/val_subset/test_subset``：各集合子集大小（<=0 表示全量）。
    - ``val_split``：若 >0，从训练集末尾切出该数量样本作为公开验证集
      （其余作为训练集），否则验证集复用测试集。
    """
    train_full, test_full = load_raw(data_dir)

    if val_split and val_split > 0:
        n = len(train_full)
        val = Dataset(
            train_full.images[n - val_split:],
            train_full.labels[n - val_split:],
        )
        train = Dataset(
            train_full.images[:n - val_split],
            train_full.labels[:n - val_split],
        )
    else:
        # 不单独切验证集：验证集复用测试集（子集独立抽样）。
        train = train_full
        val = test_full

    train = train.subset(train_subset, seed=seed)
    val = val.subset(val_subset, seed=seed + 1)
    test = test_full.subset(test_subset, seed=seed + 2)
    return Split(train=train, val=val, test=test)


def cache_status(data_dir: Optional[str] = None) -> dict:
    """返回 MNIST 缓存状态，供 API 暴露。"""
    data_dir = data_dir or _default_data_dir()
    info = {"data_dir": data_dir, "cached": True, "files": {}}
    total = 0
    for name, fname in _FILES.items():
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            sz = os.path.getsize(p)
            info["files"][name] = {"path": p, "size": sz, "ok": sz > 0}
            total += sz
        else:
            info["files"][name] = {"path": p, "size": 0, "ok": False}
            info["cached"] = False
    info["total_bytes"] = total
    return info
