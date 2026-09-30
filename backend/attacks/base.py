"""攻击基类与共用工具。"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class AttackBase:
    """攻击方法基类。

    子类实现 :meth:`build_trigger` 返回 (top, left, pattern_2x2)，
    并可覆写 :meth:`select_indices` 选择投毒样本。
    类属性 ``name`` 为注册名。
    """

    name: str = "base"
    poison_rate: float = 0.05   # 默认投毒比例（第七节，<=5%）

    def build_trigger(self, height: int, width: int, rng: np.random.Generator
                      ) -> Tuple[int, int, np.ndarray]:
        """返回 (top, left, 2x2 pattern in [0,1])。子类必须实现。"""
        raise NotImplementedError

    def select_indices(self, candidates: np.ndarray, target_label: int,
                       budget: int, rng: np.random.Generator) -> np.ndarray:
        """从候选样本中随机抽取不超过 budget 个。"""
        n = min(int(budget), len(candidates))
        if n <= 0:
            return np.array([], dtype=np.int64)
        return rng.choice(candidates, size=n, replace=False)

    def run(self, env, task: dict) -> dict:
        """统一 attack 接口（第十三节）。"""
        images = np.asarray(task["images"], dtype=np.float32).copy()
        labels = np.asarray(task["labels"]).copy()
        target_label = int(task["target_label"])
        budget = int(task["poison_budget"])
        rng = env.rng()

        if images.ndim == 3:
            images = images[:, None, :, :]
        _, _, h, w = images.shape

        top, left, pat2x2 = self.build_trigger(h, w, rng)
        pat2x2 = np.asarray(pat2x2, dtype=np.float32)
        if pat2x2.shape != (2, 2):
            raise ValueError(f"触发器图案必须为 2x2，实际 {pat2x2.shape}")

        mask = np.zeros((1, h, w), dtype=np.float32)
        pattern = np.zeros((1, h, w), dtype=np.float32)
        mask[0, top:top + 2, left:left + 2] = 1.0
        pattern[0, top:top + 2, left:left + 2] = pat2x2

        # 只在非目标标签的样本中选择投毒候选
        non_target = np.where(labels != target_label)[0]
        # 进一步过滤：仅保留「应用触发器后图像确实会改变」的样本，
        # 否则会出现「图像未变但标签被改」违反 14.5。
        m = mask[None, ...]  # [1,1,H,W]
        would_change = np.any(
            (images[non_target] * (1.0 - m) + pattern[None, ...] * m) != images[non_target],
            axis=tuple(range(1, images.ndim)),
        )
        candidates = non_target[would_change]

        poison_idx = self.select_indices(candidates, target_label, budget, rng)
        if len(poison_idx) > 0:
            images[poison_idx] = (
                images[poison_idx] * (1.0 - m) + pattern[None, ...] * m
            )
            labels[poison_idx] = target_label

        return {
            "images": images,
            "labels": labels,
            "trigger_mask": mask,
            "trigger_pattern": pattern,
        }


def place_inside(h: int, w: int, top: int, left: int, size: int = 2) -> Tuple[int, int]:
    """确保 (top,left) 起的 size×size 块完整位于图像内部。"""
    top = max(0, min(top, h - size))
    left = max(0, min(left, w - size))
    return top, left
