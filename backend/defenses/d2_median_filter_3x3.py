"""D2 3×3 中值滤波（最小防御示例，第三十二节）。"""
from __future__ import annotations

import numpy as np

from .base import DefenseBase


def median_filter_3x3(images: np.ndarray) -> np.ndarray:
    """对 [N, C, H, W] 图像批量做 3×3 中值滤波。"""
    n, c, h, w = images.shape
    padded = np.pad(images, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
    shifted = []
    for dy in range(3):
        for dx in range(3):
            shifted.append(padded[:, :, dy:dy + h, dx:dx + w])
    stacked = np.stack(shifted, axis=0)
    return np.median(stacked, axis=0).astype(images.dtype)


class D2MedianFilter3x3(DefenseBase):
    """对所有训练图像应用 3×3 中值滤波，可削弱 2×2 局部触发器。"""

    name = "d2_median_filter_3x3"

    def process(self, env, images, labels):
        filtered = median_filter_3x3(images)
        return filtered, labels.copy(), None
