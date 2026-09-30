"""统一分类模型（第九节）。

- :class:`LeNet`：Model-A，LeNet 风格小型 CNN。
- :class:`SmallCNN`：Model-B，稍深一层，用于跨模型泛化评测。
- :class:`ModelWrapper`：包裹训练好的模型，提供 ``.predict`` / ``.predict_proba``。

所有模型输入 ``[N, 1, 28, 28]``，输出 10 类 logits。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet 风格结构：Conv3×3-ReLU-MaxPool ×2 -> FC -> 10。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14 -> 7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SmallCNN(nn.Module):
    """Model-B：稍大的小型 CNN，验证跨模型泛化。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 7
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class TinyMLP(nn.Module):
    """极轻量代理模型（供防御方异常检测，第六节）。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    name = (name or "lenet").lower()
    if name in ("lenet", "model-a", "model_a"):
        return LeNet(num_classes)
    if name in ("small_cnn", "smallcnn", "model-b", "model_b"):
        return SmallCNN(num_classes)
    if name in ("tiny_mlp", "tinymlp", "proxy"):
        return TinyMLP(num_classes)
    raise ValueError(f"未知模型名: {name}")


class ModelWrapper:
    """统一推理接口（第九节）：``predictions = model.predict(test_images)``。"""

    def __init__(self, model: nn.Module, device: torch.device, num_classes: int = 10):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """返回 [N, num_classes] 的 softmax 概率。"""
        x = self._to_tensor(images)
        logits = self.model(x)
        return F.softmax(logits, dim=1).cpu().numpy()

    @torch.no_grad()
    def predict(self, images: np.ndarray) -> np.ndarray:
        """返回 [N] 的预测标签（int64）。"""
        x = self._to_tensor(images)
        logits = self.model(x)
        return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    def _to_tensor(self, images: np.ndarray) -> torch.Tensor:
        arr = np.asarray(images, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, None, :, :]
        arr = np.clip(arr, 0.0, 1.0)
        return torch.from_numpy(arr).to(self.device)
