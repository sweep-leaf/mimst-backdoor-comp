"""端到端冒烟测试。

运行::
    python -m pytest tests/test_pipeline.py -q
    # 或不依赖 pytest:
    python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
import os

import numpy as np

# 让 `backend` 可被导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.config import EvalConfig
from backend.core.metrics import (
    backdoor_gain, backdoor_removal, clean_retention,
    attack_score, defense_score,
)
from backend.core.registry import get_default_registry
from backend.core.evaluator import evaluate
from backend.core.validator import validate_attack, validate_defense


def _tiny_config() -> EvalConfig:
    cfg = EvalConfig.fast()
    cfg.train_subset = 1500
    cfg.test_subset = 400
    cfg.train.epochs = 2
    return cfg


def test_registry_has_builtins():
    reg = get_default_registry()
    assert "a1_bottom_right_white" in reg.attack_names()
    assert "d1_no_defense" in reg.defense_names()
    assert "d5_spectral_signature" in reg.defense_names()
    assert len(reg.attack_names()) >= 5
    assert len(reg.defense_names()) >= 5
    print("OK test_registry_has_builtins")


def test_metric_formulas():
    # backdoor_gain
    assert abs(backdoor_gain(0.9, 0.1) - (0.8 / 0.9)) < 1e-9
    assert backdoor_gain(0.1, 0.1) == 0.0          # 无增益
    assert backdoor_gain(0.5, 1.0) == 0.0          # clean 已 100% -> 无空间
    assert backdoor_gain(1.2, 0.1) == 1.0          # 截断到 1
    # clean_retention
    assert clean_retention(0.9, 0.95) == 0.9 / 0.95
    assert clean_retention(1.0, 0.9) == 1.0        # 截断到 1
    assert clean_retention(0.5, 0.0) == 0.0
    # backdoor_removal
    rem, skip = backdoor_removal(0.9, 0.1, 0.1)
    assert skip is False and abs(rem - (0.8 / 0.8)) < 1e-9
    rem, skip = backdoor_removal(0.1005, 0.1, 0.1, eps=1e-3)  # ASR_attack≈ASR_clean (gap<eps)
    assert skip is True
    rem, skip = backdoor_removal(0.1, 0.1, 0.1)               # 完全相等 -> 跳过
    assert skip is True
    # scores
    assert abs(attack_score(0.5, 0.8) - 0.4) < 1e-9
    assert abs(defense_score(0.5, 0.8) - (0.7 * 0.5 + 0.3 * 0.8)) < 1e-9
    print("OK test_metric_formulas")


def test_validator_rejects_illegal_attack():
    reg = get_default_registry()
    env_like = type("E", (), {"rng": lambda self, s=None: np.random.default_rng(0)})()
    task = {
        "images": np.random.rand(100, 1, 28, 28).astype(np.float32),
        "labels": np.random.randint(0, 10, size=100),
        "target_label": 0, "poison_budget": 5, "trigger_size": 2,
    }
    out = reg.get_attack("a1_bottom_right_white").fn(env_like, task)
    # 合法
    ok = validate_attack(
        task["images"], task["labels"], out["images"], out["labels"],
        out["trigger_mask"], out["trigger_pattern"], 0, 5, 2,
    )
    assert ok.ok, ok.violations

    # 非法：在 mask 外修改像素
    bad = out["images"].copy()
    bad[0, 0, 0, 0] = 1.0  # 左上角，远离右下角触发器
    bad_labels = out["labels"].copy()
    res = validate_attack(
        task["images"], task["labels"], bad, bad_labels,
        out["trigger_mask"], out["trigger_pattern"], 0, 5, 2,
    )
    assert not res.ok
    assert any("14.3" in v for v in res.violations)

    # 非法：超过预算
    big = out["images"].copy()
    big_lab = out["labels"].copy()
    # 把前 50 个非目标样本也改成目标标签（不动图像也会被算作修改标签? 不会，标签改了但图像没改 -> 14.5 未修改样本标签被改）
    big_lab[:50] = 0
    res2 = validate_attack(
        task["images"], task["labels"], big, big_lab,
        out["trigger_mask"], out["trigger_pattern"], 0, 5, 2,
    )
    assert not res2.ok
    print("OK test_validator_rejects_illegal_attack")


def test_defense_size_constraint():
    # 防御删除超过 10% 样本应被判非法
    imgs = np.random.rand(100, 1, 28, 28).astype(np.float32)
    labs = np.random.randint(0, 10, 100)
    res = validate_defense(imgs, labs, imgs[:50], labs[:50], None, min_ratio=0.9)
    assert not res.ok
    assert any("0.9" in v for v in res.violations)
    # 保留 90% 应通过
    res2 = validate_defense(imgs, labs, imgs[:90], labs[:90], None, min_ratio=0.9)
    assert res2.ok, res2.violations
    print("OK test_defense_size_constraint")


def test_end_to_end_pipeline():
    reg = get_default_registry()
    cfg = _tiny_config()
    res = evaluate(
        attack_fn=reg.get_attack("a1_bottom_right_white").fn,
        defend_fn=reg.get_defense("d2_median_filter_3x3").fn,
        target_label=0, seed=0, config=cfg,
        attack_name="a1", defense_name="d2",
    )
    assert res.error is None, res.error
    m = res.metrics
    assert m.attack_valid, m.attack_violations
    assert m.defense_valid, m.defense_violations
    assert 0.0 <= m.attack_score <= 1.0
    assert 0.0 <= m.defense_score <= 1.0
    # 后门应当生效：ASR_attack 明显高于 ASR_clean
    assert m.asr_attack > m.asr_clean
    # 字段齐全
    for f in ["acc_clean", "acc_attack", "acc_defense",
              "asr_clean", "asr_attack", "asr_defense",
              "backdoor_gain", "clean_retention_attack", "attack_score",
              "backdoor_removal", "clean_retention_defense", "defense_score"]:
        assert hasattr(m, f), f"缺少字段 {f}"
    print(f"OK test_end_to_end_pipeline "
          f"(Acc={m.acc_clean:.3f}/{m.acc_attack:.3f}/{m.acc_defense:.3f}, "
          f"ASR={m.asr_attack:.3f}->{m.asr_defense:.3f}, "
          f"AScore={m.attack_score:.3f}, DScore={m.defense_score:.3f})")


def test_submission_loading():
    from backend.core.registry import Registry
    reg = Registry()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    reg.register_attack_dir(os.path.join(root, "submissions", "example_attack"))
    reg.register_defense_dir(os.path.join(root, "submissions", "example_defense"))
    assert "example_attack" in reg.attack_names()
    assert "example_defense" in reg.defense_names()
    # 用 submission 跑一次单组评测
    cfg = _tiny_config()
    res = evaluate(
        attack_fn=reg.get_attack("example_attack").fn,
        defend_fn=reg.get_defense("example_defense").fn,
        target_label=0, seed=0, config=cfg,
        attack_name="example_attack", defense_name="example_defense",
    )
    assert res.error is None, res.error
    assert res.metrics.attack_valid, res.metrics.attack_violations
    assert res.metrics.defense_valid, res.metrics.defense_violations
    print("OK test_submission_loading")


def main():
    test_registry_has_builtins()
    test_metric_formulas()
    test_validator_rejects_illegal_attack()
    test_defense_size_constraint()
    test_end_to_end_pipeline()
    test_submission_loading()
    print("\n[ALL PASSED]")


if __name__ == "__main__":
    main()
