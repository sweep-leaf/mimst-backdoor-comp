"""A5 随机位置二值触发器攻击。"""
from __future__ import annotations

import numpy as np

from .base import AttackBase, place_inside


class A5RandomBinary(AttackBase):
    """随机位置的 2×2 二值（0/1）触发器。

    位置与图案在每轮评测（每个 seed）内固定，所有投毒样本使用同一触发器，
    满足第十四节「使用统一触发器」约束。二值图案比灰色对比更强。
    """

    name = "a5_random_binary"

    def build_trigger(self, height, width, rng):
        # 从四个角中随机选一个（空旷区域，触发器信号清晰且位置可变）
        corners = [
            (1, 1),                       # 左上
            (1, width - 3),               # 右上
            (height - 3, 1),              # 左下
            (height - 3, width - 3),      # 右下
        ]
        top, left = corners[int(rng.integers(0, len(corners)))]
        top, left = place_inside(height, width, top, left, 2)
        # 2×2 二值图案（0/1），保证至少一个白像素
        pat = rng.integers(0, 2, size=(2, 2)).astype(np.float32)
        if pat.sum() == 0:
            pat[0, 0] = 1.0
        return top, left, pat
