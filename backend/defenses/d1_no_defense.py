"""D0-1 无防御（主办方基线，第三十节）。"""
from __future__ import annotations

import numpy as np

from .base import DefenseBase


class D1NoDefense(DefenseBase):
    """直接返回攻击后的训练集，不做任何处理。"""

    name = "d1_no_defense"

    def process(self, env, images, labels):
        return images.copy(), labels.copy(), None
