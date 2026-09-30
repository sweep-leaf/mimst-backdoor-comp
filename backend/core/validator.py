"""合法性校验（第十四、十七节）。

攻击合法性（第十四节）:
    14.1 数据规模不变
    14.2 修改数量不超过预算
    14.3 图像修改范围合法（仅 trigger_mask 的 2×2 区域）
    14.4 使用统一触发器（所有投毒样本相同 mask/pattern）
    14.5 标签修改合法（被改样本 -> target_label，未改样本不变）

防御合法性（第十七节）:
    - N_sanitized ≥ 0.9 * N_train
    - 图像像素范围 / 形状合法
    - 标签范围合法
    - sample_weights 为 None 或长度匹配、非负有限
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ValidationResult:
    ok: bool
    violations: list = field(default_factory=list)
    n_modified: int = 0
    poison_indices: Optional[np.ndarray] = None

    def __bool__(self) -> bool:
        return self.ok


# ------------------------------------------------------------------ 攻击校验


def validate_trigger_mask(mask: np.ndarray, trigger_size: int = 2) -> list:
    """检查 trigger_mask 恰好包含一个连续 trigger_size×trigger_size 全 1 块。"""
    violations = []
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        if m.shape[0] != 1:
            violations.append(f"trigger_mask 第 0 维必须为 1，实际 {m.shape[0]}")
        m = m[0]
    if m.ndim != 2:
        violations.append(f"trigger_mask 必须为 2D 或 [1,H,W]，实际 shape={np.asarray(mask).shape}")
        return violations
    binary = (m > 0.5).astype(np.int32)
    h, w = binary.shape
    s = int(trigger_size)

    # 完整位于图像内部
    if h < s or w < s:
        violations.append(f"trigger_mask 尺寸 {binary.shape} 小于触发器尺寸 {s}×{s}")
        return violations

    # 寻找所有全 1 的 s×s 块
    blocks = []
    for y in range(h - s + 1):
        for x in range(w - s + 1):
            if binary[y:y + s, x:x + s].sum() == s * s:
                blocks.append((y, x))
    if len(blocks) == 0:
        violations.append("trigger_mask 中不存在连续的 {s}×{s} 全 1 块".format(s=s))
    elif len(blocks) > 1:
        # 可能是多个块相邻拼成的大区域 -> 统计总 1 的数量是否恰好 s*s
        total_ones = int(binary.sum())
        if total_ones != s * s:
            violations.append(
                f"trigger_mask 必须恰有一个 {s}×{s} 全 1 块（共 {s*s} 个 1），"
                f"实际有 {total_ones} 个 1"
            )
    # 其余位置必须为 0（已由 total_ones 校验覆盖）
    return violations


def validate_attack(
    original_images: np.ndarray,
    original_labels: np.ndarray,
    attacked_images: np.ndarray,
    attacked_labels: np.ndarray,
    trigger_mask: np.ndarray,
    trigger_pattern: np.ndarray,
    target_label: int,
    poison_budget: int,
    trigger_size: int = 2,
) -> ValidationResult:
    """完整攻击合法性校验（第十四节）。"""
    violations = []
    target_label = int(target_label)

    ori_img = np.asarray(original_images, dtype=np.float32)
    att_img = np.asarray(attacked_images, dtype=np.float32)
    ori_lab = np.asarray(original_labels).reshape(-1)
    att_lab = np.asarray(attacked_labels).reshape(-1)
    mask = np.asarray(trigger_mask, dtype=np.float32)
    pat = np.asarray(trigger_pattern, dtype=np.float32)

    # 14.1 数据规模不变
    if att_img.shape != ori_img.shape:
        violations.append(
            f"14.1 图像规模改变：{ori_img.shape} -> {att_img.shape}"
        )
    if att_lab.shape != ori_lab.shape:
        violations.append(
            f"14.1 标签规模改变：{ori_lab.shape} -> {att_lab.shape}"
        )
    if violations:
        return ValidationResult(ok=False, violations=violations)

    # 像素范围
    if att_img.min() < -1e-6 or att_img.max() > 1.0 + 1e-6:
        violations.append(
            f"投毒图像像素越界 [{att_img.min():.4f}, {att_img.max():.4f}]，应为 [0,1]"
        )

    # 触发器掩码合法性
    violations += validate_trigger_mask(mask, trigger_size)

    # 统一触发器：构建参考图像，比较被修改样本是否符合 (1-M)⊙x+M⊙P
    # 把 mask/pattern 广播到与图像同形状
    m2 = mask if mask.ndim == 3 else mask[None, :, :]
    p2 = pat if pat.ndim == 3 else pat[None, :, :]
    if m2.shape != ori_img.shape[1:]:
        # 尝试对齐通道
        if m2.shape[0] == 1 and ori_img.shape[1] == 1:
            pass  # [1,H,W] vs [1,H,W] OK
        else:
            violations.append(
                f"trigger_mask shape {m2.shape} 与图像 shape {ori_img.shape} 不兼容"
            )

    # 14.2 修改数量 / 14.3 修改范围 / 14.4 统一触发器 / 14.5 标签
    diff = np.any(att_img != ori_img, axis=tuple(range(1, ori_img.ndim)))
    modified_idx = np.where(diff)[0]
    n_modified = int(len(modified_idx))

    if n_modified > int(poison_budget):
        violations.append(
            f"14.2 修改样本数 {n_modified} 超过预算 {int(poison_budget)}"
        )

    # 对每个被修改样本：仅在 mask 区域内变化，且变化后 == (1-M)⊙x+M⊙P
    # 用广播：expected = ori*(1-m) + p*m
    expected = ori_img * (1.0 - m2) + p2 * m2
    bad_region = []
    bad_pattern = []
    for i in modified_idx:
        sample_diff = att_img[i] != ori_img[i]
        # 14.3 修改范围只能在 mask 区域内
        outside = sample_diff & (m2[0] < 0.5) if m2.ndim == 3 else sample_diff & (m2 < 0.5)
        if outside.any():
            bad_region.append(int(i))
        # 14.4 修改后样本须等于统一触发器应用结果
        if not np.allclose(att_img[i], expected[i], atol=1e-6):
            bad_pattern.append(int(i))
    if bad_region:
        violations.append(
            f"14.3 样本 {bad_region[:5]}{'...' if len(bad_region)>5 else ''} "
            f"在 trigger_mask 之外修改了像素（共 {len(bad_region)} 个样本）"
        )
    if bad_pattern:
        violations.append(
            f"14.4 样本 {bad_pattern[:5]}{'...' if len(bad_pattern)>5 else ''} "
            f"未使用统一触发器 pattern（共 {len(bad_pattern)} 个样本）"
        )

    # 14.5 标签：被修改样本 -> target_label；未修改样本 -> 不变
    bad_label_modified = [int(i) for i in modified_idx if int(att_lab[i]) != target_label]
    unmodified_idx = np.where(~diff)[0]
    bad_label_unmodified = [int(i) for i in unmodified_idx if int(att_lab[i]) != int(ori_lab[i])]
    if bad_label_modified:
        violations.append(
            f"14.5 被修改样本 {bad_label_modified[:5]} 的标签未改为 target_label={target_label}"
        )
    if bad_label_unmodified:
        violations.append(
            f"14.5 未被修改样本 {bad_label_unmodified[:5]} 的标签被改动"
        )

    ok = len(violations) == 0
    return ValidationResult(
        ok=ok,
        violations=violations,
        n_modified=n_modified,
        poison_indices=modified_idx if ok else None,
    )


# ------------------------------------------------------------------ 防御校验


def validate_defense(
    poisoned_images: np.ndarray,
    poisoned_labels: np.ndarray,
    sanitized_images: np.ndarray,
    sanitized_labels: np.ndarray,
    sample_weights: Optional[np.ndarray],
    min_ratio: float = 0.9,
    num_classes: int = 10,
) -> ValidationResult:
    """防御合法性校验（第十七节）。"""
    violations = []
    s_img = np.asarray(sanitized_images, dtype=np.float32)
    s_lab = np.asarray(sanitized_labels).reshape(-1)
    n_in = int(np.asarray(poisoned_images).shape[0])

    if s_img.ndim == 3:
        s_img = s_img[:, None, :, :]
    if s_img.ndim != 4:
        violations.append(f"sanitized_images 维度非法：{s_img.shape}")
    else:
        if s_img.shape[1:] != (1, 28, 28):
            violations.append(f"sanitized_images 单样本形状应为 (1,28,28)，实际 {s_img.shape[1:]}")
        if s_img.min() < -1e-6 or s_img.max() > 1.0 + 1e-6:
            violations.append(
                f"sanitized_images 像素越界 [{s_img.min():.4f}, {s_img.max():.4f}]"
            )

    if s_lab.shape[0] != s_img.shape[0]:
        violations.append(
            f"sanitized labels 数量 {s_lab.shape[0]} 与图像数量 {s_img.shape[0]} 不一致"
        )
    if s_lab.size > 0:
        lo, hi = int(s_lab.min()), int(s_lab.max())
        if lo < 0 or hi >= num_classes:
            violations.append(f"sanitized labels 越界 [{lo}, {hi}]，应为 [0,{num_classes})")

    # 数量下限
    n_out = int(s_img.shape[0])
    if n_out < int(np.ceil(min_ratio * n_in)):
        violations.append(
            f"防御后样本数 {n_out} < 0.9×{n_in}={int(np.ceil(min_ratio*n_in))}"
        )

    # sample_weights
    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if w.shape[0] != n_out:
            violations.append(
                f"sample_weights 长度 {w.shape[0]} 与清洗后样本数 {n_out} 不一致"
            )
        else:
            if not np.all(np.isfinite(w)):
                violations.append("sample_weights 含非有限值")
            if w.min() < 0:
                violations.append(f"sample_weights 含负值 {float(w.min())}")

    ok = len(violations) == 0
    return ValidationResult(ok=ok, violations=violations, n_modified=n_in - n_out)
