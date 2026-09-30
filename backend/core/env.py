"""裁判环境（第九、十一、十六节）。

攻击方 / 防御方通过 ``env`` 与裁判系统交互：
    model = env.train(images, labels, sample_weights, config)
    predictions = model.predict(test_images)

防御方还可使用轻量代理模型进行异常检测（第六节）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import EvalConfig, TrainConfig
from .model import ModelWrapper
from .train import train_model, train_proxy


@dataclass
class RefereeEnv:
    """裁判环境对象，传入 attack / defend 函数。"""

    config: EvalConfig
    seed: int = 0
    device: Optional[object] = None

    def __post_init__(self):
        import torch
        self._device = self.device or torch.device("cpu")
        self._seed = int(self.seed if self.seed is not None else self.config.seed)

    # ---- 统一训练接口（第九节）----
    def train(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        config: Optional[TrainConfig] = None,
    ) -> ModelWrapper:
        cfg = config or self._train_config()
        return train_model(
            images=images,
            labels=labels,
            sample_weights=sample_weights,
            config=cfg,
            device=self._device,
        )

    def train_proxy(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epochs: int = 1,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> ModelWrapper:
        """训练极轻量代理模型，供防御方异常检测使用。"""
        return train_proxy(
            images=images, labels=labels,
            epochs=epochs, batch_size=batch_size, lr=lr,
            seed=self._seed, device=self._device,
        )

    # ---- 随机数 ----
    def rng(self, seed: Optional[int] = None) -> np.random.Generator:
        """返回带种子的 numpy 随机生成器，保证可复现。"""
        return np.random.default_rng(self._seed if seed is None else int(seed))

    @property
    def default_config(self) -> EvalConfig:
        return self.config

    @property
    def device_str(self) -> str:
        return str(self._device)

    def _train_config(self) -> TrainConfig:
        cfg = TrainConfig(**self.config.train.to_dict())  # 复制
        cfg.seed = self._seed
        return cfg
