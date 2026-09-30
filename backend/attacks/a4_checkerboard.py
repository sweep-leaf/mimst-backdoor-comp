"""A4 黑白棋盘触发器攻击。"""
from __future__ import annotations

import numpy as np

from .base import AttackBase, place_inside


class A4Checkerboard(AttackBase):
    """右下角 2×2 棋盘触发器（[1,0;0,1]）。"""

    name = "a4_checkerboard"

    def build_trigger(self, height, width, rng):
        top, left = height - 3, width - 3
        top, left = place_inside(height, width, top, left, 2)
        pat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        return top, left, pat
