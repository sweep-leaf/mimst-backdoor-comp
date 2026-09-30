"""指标计算（第十九 ~ 二十七节）。

所有公式严格对应赛题定义，并在分母接近 0 时给出安全退化。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Metrics:
    """单组攻防评测的全部指标。"""

    # ---- 基础准确率（第十九节）----
    acc_clean: float = 0.0
    acc_attack: float = 0.0
    acc_defense: float = 0.0

    # ---- 后门成功率（第二十、二十四节）----
    asr_clean: float = 0.0       # 自然触发基线（第二十一节）
    asr_attack: float = 0.0
    asr_defense: float = 0.0

    # ---- 攻击指标（第二十一、二十二、二十三节）----
    backdoor_gain: float = 0.0
    clean_retention_attack: float = 0.0
    attack_score: float = 0.0

    # ---- 防御指标（第二十五、二十六、二十七节）----
    backdoor_removal: float = 0.0
    clean_retention_defense: float = 0.0
    defense_score: float = 0.0
    defense_score_skipped: bool = False   # ASR_attack≈ASR_clean 时不参与防御评分

    # ---- 资格线 / 参考线（第二十三、二十七节）----
    attack_qualified: bool = False
    attack_acc_drop: float = 0.0          # acc_clean - acc_attack
    defense_success: bool = False
    defense_acc_drop: float = 0.0         # acc_clean - acc_defense

    # ---- 辅助计数 ----
    n_triggered_test: int = 0
    n_train: int = 0
    n_poison: int = 0

    # ---- 合法性 ----
    attack_valid: bool = True
    attack_violations: list = field(default_factory=list)
    defense_valid: bool = True
    defense_violations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def accuracy(preds, labels) -> float:
    import numpy as np
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] == 0:
        return 0.0
    return float((preds == labels).mean())


def asr(preds, target_label) -> float:
    """后门攻击成功率（第二十节）：预测 == 目标标签的比例。"""
    import numpy as np
    preds = np.asarray(preds).reshape(-1)
    if preds.shape[0] == 0:
        return 0.0
    return float((preds == int(target_label)).mean())


def backdoor_gain(asr_attack: float, asr_clean: float) -> float:
    """归一化后门增益（第二十一节），截断到 [0,1]。"""
    denom = 1.0 - asr_clean
    if denom <= 0:
        # 干净模型在触发集上已经 100% 预测目标标签 -> 无可增益空间。
        return 0.0
    val = (asr_attack - asr_clean) / denom
    return float(max(0.0, min(1.0, val)))


def clean_retention(acc_model: float, acc_clean: float) -> float:
    """干净性能保持率（第二十二、二十六节）。"""
    if acc_clean <= 0:
        return 0.0
    return float(max(0.0, min(1.0, acc_model / acc_clean)))


def attack_score(backdoor_gain_: float, retention_attack: float) -> float:
    """攻击方单轮得分（第二十三节）。"""
    return float(backdoor_gain_ * retention_attack)


def backdoor_removal(
    asr_attack: float,
    asr_defense: float,
    asr_clean: float,
    eps: float = 1e-3,
):
    """后门消除率（第二十五节），并返回是否因分母接近 0 而跳过。

    返回 (removal, skipped)。
    """
    denom = asr_attack - asr_clean
    if denom <= eps:
        # 攻击本身没产生明显后门效果 -> 该对阵不参与防御评分。
        return 0.0, True
    val = (asr_attack - asr_defense) / denom
    return float(max(0.0, min(1.0, val))), False


def defense_score(backdoor_removal_: float, retention_defense: float) -> float:
    """防御方单轮得分（第二十七节）。"""
    return float(0.7 * backdoor_removal_ + 0.3 * retention_defense)


def check_attack_qualified(
    asr_attack: float,
    acc_clean: float,
    acc_attack: float,
    asr_thresh: float = 0.60,
    acc_drop_thresh: float = 0.03,
):
    """攻击资格线（第二十三节）。返回 (qualified, acc_drop)。"""
    acc_drop = acc_clean - acc_attack
    qualified = (asr_attack >= asr_thresh) and (acc_drop <= acc_drop_thresh)
    return bool(qualified), float(acc_drop)


def check_defense_success(
    asr_defense: float,
    acc_clean: float,
    acc_defense: float,
    asr_thresh: float = 0.20,
    acc_drop_thresh: float = 0.03,
):
    """防御成功参考线（第二十七节）。返回 (success, acc_drop)。"""
    acc_drop = acc_clean - acc_defense
    success = (asr_defense <= asr_thresh) and (acc_drop <= acc_drop_thresh)
    return bool(success), float(acc_drop)


def assemble_metrics(
    *,
    acc_clean: float,
    acc_attack: float,
    acc_defense: float,
    asr_clean: float,
    asr_attack: float,
    asr_defense: float,
    n_triggered_test: int = 0,
    n_train: int = 0,
    n_poison: int = 0,
    asr_skip_eps: float = 1e-3,
    attack_qualify_asr: float = 0.60,
    attack_qualify_acc_drop: float = 0.03,
    defense_success_asr: float = 0.20,
    defense_success_acc_drop: float = 0.03,
    attack_valid: bool = True,
    attack_violations: Optional[list] = None,
    defense_valid: bool = True,
    defense_violations: Optional[list] = None,
) -> Metrics:
    """根据原始测量量组装完整 :class:`Metrics`。"""
    bg = backdoor_gain(asr_attack, asr_clean)
    ret_a = clean_retention(acc_attack, acc_clean)
    asc = attack_score(bg, ret_a)

    removal, skipped = backdoor_removal(asr_attack, asr_defense, asr_clean, eps=asr_skip_eps)
    ret_d = clean_retention(acc_defense, acc_clean)
    dsc = 0.0 if skipped else defense_score(removal, ret_d)

    atk_qual, atk_drop = check_attack_qualified(
        asr_attack, acc_clean, acc_attack,
        asr_thresh=attack_qualify_asr, acc_drop_thresh=attack_qualify_acc_drop,
    )
    def_ok, def_drop = check_defense_success(
        asr_defense, acc_clean, acc_defense,
        asr_thresh=defense_success_asr, acc_drop_thresh=defense_success_acc_drop,
    )

    return Metrics(
        acc_clean=acc_clean,
        acc_attack=acc_attack,
        acc_defense=acc_defense,
        asr_clean=asr_clean,
        asr_attack=asr_attack,
        asr_defense=asr_defense,
        backdoor_gain=bg,
        clean_retention_attack=ret_a,
        attack_score=asc,
        backdoor_removal=removal,
        clean_retention_defense=ret_d,
        defense_score=dsc,
        defense_score_skipped=skipped,
        attack_qualified=atk_qual,
        attack_acc_drop=atk_drop,
        defense_success=def_ok,
        defense_acc_drop=def_drop,
        n_triggered_test=n_triggered_test,
        n_train=n_train,
        n_poison=n_poison,
        attack_valid=attack_valid,
        attack_violations=list(attack_violations or []),
        defense_valid=defense_valid,
        defense_violations=list(defense_violations or []),
    )
