"""触发器应用与隐藏触发测试集生成（第十二、十五节）。

触发器:
    trigger_mask:    [1, 28, 28]，仅一个连续 2×2 区域为 1，其余为 0。
    trigger_pattern: [1, 28, 28]，仅掩码区域内的像素会被使用。

应用公式（第十五节）:
    x_trigger = (1 - M) ⊙ x + M ⊙ P
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .dataset import Dataset


def apply_trigger(images: np.ndarray, mask: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """对图像批量应用触发器 ``x' = (1-M)⊙x + M⊙P``。

    images:  [N, 1, H, W] 或 [N, H, W]
    mask:    [1, H, W] 或 [H, W]
    pattern: [1, H, W] 或 [H, W]，与 images 同形状后广播。
    """
    x = np.asarray(images, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    p = np.asarray(pattern, dtype=np.float32)

    if x.ndim == 3:
        x = x[:, None, :, :]
    # 将 mask/pattern 广播到 [1, C, H, W]（C 取 x 的通道数）
    m = _broadcast_mask(m, x.shape)
    p = _broadcast_mask(p, x.shape)

    out = x * (1.0 - m) + p * m
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _broadcast_mask(m: np.ndarray, target_shape: tuple) -> np.ndarray:
    """把 [1,H,W] 或 [H,W] 广播到 [1, C, H, W] 形状以便与 [N,C,H,W] 相乘。"""
    if m.ndim == 2:
        m = m[None, :, :]
    if m.ndim == 3 and m.shape[0] == 1:
        c = target_shape[1]
        m = np.repeat(m, c, axis=0) if c > 1 else m
    # 形如 [C,H,W]，再加一维给 batch 广播
    return m[None, :, :, :]


def build_triggered_test_set(
    test: Dataset,
    target_label: int,
    mask: np.ndarray,
    pattern: np.ndarray,
) -> Dataset:
    """构建隐藏触发测试集（第十五节）。

    从隐藏测试集中选择所有 ``y ≠ target_label`` 的样本，应用触发器。
    """
    labels = test.labels.astype(np.int64)
    sel = np.where(labels != int(target_label))[0]
    if len(sel) == 0:
        return Dataset(
            images=np.zeros((0, 1, 28, 28), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
        )
    triggered = apply_trigger(test.images[sel], mask, pattern)
    return Dataset(images=triggered, labels=labels[sel])


def find_trigger_block(mask: np.ndarray) -> Tuple[int, int, int]:
    """定位 trigger_mask 中唯一的 2×2 全 1 连续块。

    返回 (top, left, size)。若不存在合法块返回 (-1, -1, 0)。
    """
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    binary = (m > 0.5).astype(np.int32)
    h, w = binary.shape
    for y in range(h - 1):
        for x in range(w - 1):
            block = binary[y:y + 2, x:x + 2]
            if block.sum() == 4:
                return y, x, 2
    return -1, -1, 0
