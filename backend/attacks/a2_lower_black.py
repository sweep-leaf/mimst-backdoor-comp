"""A2 下方全黑方块攻击（更隐蔽的触发器）。"""
from __future__ import annotations

import numpy as np

from .base import AttackBase, place_inside


class A2LowerBlack(AttackBase):
    """图像下方偏中位置的 2×2 全黑触发器。

    黑色（擦除像素）触发器比白色更隐蔽，但只能修改该位置原本有前景的样本，
    因此放在前景出现率较高的下方中部（约 y=20, x=13），可覆盖 ~80% 样本。
    """

    name = "a2_lower_black"

    def build_trigger(self, height, width, rng):
        top, left = 20, 13
        top, left = place_inside(height, width, top, left, 2)
        return top, left, np.zeros((2, 2), dtype=np.float32)
