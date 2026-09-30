"""统一模型训练（第九节）。

``train_model`` 实现 ``env.train`` 的核心：支持样本权重加权训练。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .model import ModelWrapper, build_model


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _to_tensor(images: np.ndarray, labels: np.ndarray):
    x = np.asarray(images, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, None, :, :]
    x = np.clip(x, 0.0, 1.0)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    return torch.from_numpy(x), torch.from_numpy(y)


def train_model(
    images: np.ndarray,
    labels: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    config: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
) -> ModelWrapper:
    """训练统一分类模型。

    参数:
        images: [N, 1, 28, 28] 或 [N, 28, 28]，float ∈[0,1]。
        labels: [N] int。
        sample_weights: None 或 [N] 非负权重（加权交叉熵）。
        config: 训练超参；None 时使用默认。
    """
    cfg = config or TrainConfig()
    _set_seed(cfg.seed)
    if device is None:
        device = torch.device("cpu")

    x_t, y_t = _to_tensor(images, labels)
    ds = TensorDataset(x_t, y_t)
    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if w.shape[0] != x_t.shape[0]:
            raise ValueError(
                f"sample_weights 长度 {w.shape[0]} 与样本数 {x_t.shape[0]} 不一致"
            )
        w_t = torch.from_numpy(w)
        ds = TensorDataset(x_t, y_t, w_t)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False,
    )

    model = build_model(cfg.model, num_classes=10)
    model.to(device)

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_weights = sample_weights is not None
    n_batches = max(1, len(loader))
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for i, batch in enumerate(loader):
            if use_weights:
                xb, yb, wb = batch
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                wb = None
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="none")
            if use_weights:
                loss = (loss * wb).sum() / (wb.sum() + 1e-8)
            else:
                loss = loss.mean()
            loss.backward()
            opt.step()
            running += float(loss.item())
            if cfg.log_every and (i % cfg.log_every == 0):
                print(f"  epoch {epoch+1}/{cfg.epochs} batch {i}/{n_batches} loss={loss.item():.4f}")
        if cfg.log_every:
            print(f"epoch {epoch+1}/{cfg.epochs} avg_loss={running/n_batches:.4f}")

    return ModelWrapper(model, device=device, num_classes=10)


def train_proxy(
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> ModelWrapper:
    """训练极轻量代理模型（供防御方异常检测，第六节）。

    与 :func:`train_model` 接口一致，但默认使用 :class:`TinyMLP`、极少 epoch。
    """
    cfg = TrainConfig(model="tiny_mlp", epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    return train_model(images, labels, sample_weights=None, config=cfg, device=device)
