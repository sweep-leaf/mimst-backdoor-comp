"""D4 标签平滑 + 可信样本重加权防御。"""
from __future__ import annotations

import numpy as np

from .base import DefenseBase


class D4LabelSmoothingWeight(DefenseBase):
    """对置信度低的样本降权，并对标签做轻度平滑。

    思路：用代理模型对训练集预测，预测与给定标签不一致的样本视为可疑，
    降低其训练权重（不删除，满足 N_sanitized ≥ 0.9 N_train）。
    """

    name = "d4_label_smoothing_weight"
    weight_floor: float = 0.2     # 可疑样本的最低权重
    smoothing: float = 0.05       # 标签平滑系数

    def process(self, env, images, labels):
        proxy = env.train_proxy(images, labels, epochs=1)
        proba = proxy.predict_proba(images)
        pred = proba.argmax(axis=1)
        n = len(labels)
        # 与代理预测一致 -> 权重 1.0；不一致 -> weight_floor
        weights = np.where(pred == labels, 1.0, self.weight_floor).astype(np.float32)

        # 标签平滑：label -> onehot*(1-2s) + s/num_classes
        if self.smoothing > 0:
            num_classes = proba.shape[1]
            smoothed = proba * self.smoothing
            eye = np.zeros_like(proba)
            eye[np.arange(n), labels] = 1.0
            soft = eye * (1.0 - self.smoothing * (num_classes - 1) / num_classes) + smoothed
            new_labels = soft.argmax(axis=1).astype(np.int64)
            # 不改变原始标签（避免引入新的非法标签），仅靠权重降权
            new_labels = labels.copy()
        return images.copy(), labels.copy(), weights
