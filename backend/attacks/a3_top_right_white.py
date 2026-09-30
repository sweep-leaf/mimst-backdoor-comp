"""A3 右上角白色方块攻击。"""
from __future__ import annotations

import numpy as np

from .base import AttackBase, place_inside


class A3TopRightWhite(AttackBase):
    """右上角 2×2 全白触发器（与 a1 对称的空旷区域位置，效果较强）。"""

    name = "a3_top_right_white"

    def build_trigger(self, height, width, rng):
        top, left = 1, width - 3
        top, left = place_inside(height, width, top, left, 2)
        return top, left, np.ones((2, 2), dtype=np.float32)
