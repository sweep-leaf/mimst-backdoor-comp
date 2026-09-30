"""A0-1 右下角白色方块攻击（主办方基线，第二十九节）。"""
from __future__ import annotations

import numpy as np

from .base import AttackBase, place_inside


class A1BottomRightWhite(AttackBase):
    """右下角 2×2 全白触发器，与边界保留 1 像素间距。"""

    name = "a1_bottom_right_white"

    def build_trigger(self, height, width, rng):
        top, left = height - 3, width - 3
        top, left = place_inside(height, width, top, left, 2)
        return top, left, np.ones((2, 2), dtype=np.float32)
