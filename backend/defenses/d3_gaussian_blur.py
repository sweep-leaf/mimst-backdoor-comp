"""D3 高斯模糊防御。"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .base import DefenseBase


class D3GaussianBlur(DefenseBase):
    """对所有训练图像做高斯模糊，平滑掉小型局部触发器。"""

    name = "d3_gaussian_blur"
    sigma: float = 0.8

    def process(self, env, images, labels):
        out = np.empty_like(images)
        for i in range(images.shape[0]):
            out[i, 0] = gaussian_filter(images[i, 0], sigma=self.sigma)
        return np.clip(out, 0.0, 1.0), labels.copy(), None
