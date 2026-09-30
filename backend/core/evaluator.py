"""评测器（第十、三十六节）。

依次执行::

    干净训练集 -> 攻击 -> 投毒训练集 -> 防御 -> 清洗训练集
        -> 统一模型训练 -> 干净测试 / 触发测试 -> 指标计算

提供:
- :func:`evaluate`   单组 (attack, defense, target_label, seed) 评测。
- :func:`run_matrix` 全对阵矩阵，复用 clean/attack 模型以减少重复训练。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import EvalConfig, get_config
from .dataset import Dataset, Split, load_split
from .env import RefereeEnv
from .metrics import Metrics, accuracy, asr, assemble_metrics
from .model import ModelWrapper
from .trigger import build_triggered_test_set
from .validator import validate_attack, validate_defense, ValidationResult


# ------------------------------------------------------------------ 数据结构


@dataclass
class EvalResult:
    """单组评测结果。"""

    attack: str
    defense: str
    target_label: int
    seed: int
    metrics: Metrics
    elapsed_sec: float = 0.0
    config_mode: str = "fast"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "attack": self.attack,
            "defense": self.defense,
            "target_label": self.target_label,
            "seed": self.seed,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "config_mode": self.config_mode,
            "error": self.error,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class MatrixResult:
    """全对阵矩阵结果。"""

    attacks: List[str]
    defenses: List[str]
    target_labels: List[int]
    seeds: List[int]
    results: List[EvalResult] = field(default_factory=list)

    # 聚合得分矩阵（按 target+seed 平均）
    attack_score_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    defense_score_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    asr_attack_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    asr_defense_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "attacks": self.attacks,
            "defenses": self.defenses,
            "target_labels": self.target_labels,
            "seeds": self.seeds,
            "n_results": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "attack_score_matrix": self.attack_score_matrix,
            "defense_score_matrix": self.defense_score_matrix,
            "asr_attack_matrix": self.asr_attack_matrix,
            "asr_defense_matrix": self.asr_defense_matrix,
        }


# ------------------------------------------------------------------ 内部缓存


class _ModelCache:
    """模型缓存：复用 clean / attack 模型，避免在全对阵中重复训练。

    键:
        - clean 模型: ("clean", seed)
        - attack 模型: ("attack", attack_name, target_label, seed)
    防御模型依赖攻击输出，无法跨防御复用。
    """

    def __init__(self):
        self._store: Dict[tuple, ModelWrapper] = {}

    def get_or_train_clean(self, split: Split, env: RefereeEnv, seed: int) -> ModelWrapper:
        key = ("clean", seed)
        if key not in self._store:
            self._store[key] = env.train(split.train.images, split.train.labels)
        return self._store[key]

    def get_or_train_attack(
        self, poisoned_images: np.ndarray, poisoned_labels: np.ndarray,
        env: RefereeEnv, attack_name: str, target_label: int, seed: int,
        sample_weights=None,
    ) -> ModelWrapper:
        key = ("attack", attack_name, int(target_label), seed, _hash_arrays(sample_weights))
        if key not in self._store:
            self._store[key] = env.train(
                poisoned_images, poisoned_labels, sample_weights=sample_weights,
            )
        return self._store[key]

    def clear(self) -> None:
        self._store.clear()


def _hash_arrays(arr) -> str:
    if arr is None:
        return "none"
    a = np.asarray(arr)
    return f"{a.shape}_{a.dtype}_{int(a.size)}_{hashlib_int(a)}"


def hashlib_int(a: np.ndarray) -> int:
    import hashlib
    h = hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
    return int(h[:8], 16)


# ------------------------------------------------------------------ 单组评测


def _run_attack_fn(
    attack_fn: Callable, env: RefereeEnv, task: dict,
) -> Tuple[Optional[dict], Optional[str]]:
    try:
        out = attack_fn(env, task)
        if not isinstance(out, dict):
            return None, f"attack 返回值不是 dict，实际 {type(out)}"
        return out, None
    except Exception as e:  # noqa: BLE001
        import traceback
        return None, f"attack 执行异常: {e}\n{traceback.format_exc()}"


def _run_defense_fn(
    defend_fn: Callable, env: RefereeEnv, train: dict,
) -> Tuple[Optional[dict], Optional[str]]:
    try:
        out = defend_fn(env, train)
        if not isinstance(out, dict):
            return None, f"defend 返回值不是 dict，实际 {type(out)}"
        return out, None
    except Exception as e:  # noqa: BLE001
        import traceback
        return None, f"defend 执行异常: {e}\n{traceback.format_exc()}"


def evaluate(
    attack_fn: Callable,
    defend_fn: Callable,
    target_label: int,
    seed: int = 0,
    config: Optional[EvalConfig] = None,
    split: Optional[Split] = None,
    env: Optional[RefereeEnv] = None,
    attack_name: str = "attack",
    defense_name: str = "defense",
    cache: Optional[_ModelCache] = None,
) -> EvalResult:
    """执行单组 (attack, defense, target_label, seed) 评测。

    参数:
        attack_fn / defend_fn: 统一签名的可调用对象。
        target_label: 本轮目标标签（第八节）。
        seed: 随机种子。
        config: 评测配置；None 时使用 fast 默认。
        split: 预加载数据集（矩阵运行时复用）；None 时按 config 加载。
        env: 裁判环境；None 时按 config 构建。
        cache: 模型缓存（矩阵运行时复用 clean/attack 模型）。
    """
    cfg = config or get_config("fast")
    cfg = cfg.with_seed(seed)
    start = time.time()

    if split is None:
        split = load_split(
            train_subset=cfg.train_subset,
            val_subset=cfg.val_subset,
            test_subset=cfg.test_subset,
            val_split=cfg.val_split,
            seed=seed,
        )
    if env is None:
        env = RefereeEnv(config=cfg, seed=seed)

    cache = cache or _ModelCache()

    target_label = int(target_label)
    poison_budget = int(max(1, round(cfg.poison_rate * len(split.train))))

    # 构造攻击任务（第十一节）
    task = {
        "images": split.train.images.copy(),
        "labels": split.train.labels.copy(),
        "target_label": target_label,
        "poison_budget": poison_budget,
        "trigger_size": cfg.trigger_size,
    }

    # ---- 攻击 ----
    attack_out, err = _run_attack_fn(attack_fn, env, task)
    if err is not None:
        m = Metrics(attack_valid=False, attack_violations=[err])
        return EvalResult(attack_name, defense_name, target_label, seed, m,
                           elapsed_sec=time.time() - start, config_mode=cfg.mode, error=err)

    attacked_images = np.asarray(attack_out["images"], dtype=np.float32)
    attacked_labels = np.asarray(attack_out["labels"]).reshape(-1)
    trigger_mask = np.asarray(attack_out["trigger_mask"], dtype=np.float32)
    trigger_pattern = np.asarray(attack_out["trigger_pattern"], dtype=np.float32)

    av: ValidationResult = validate_attack(
        original_images=task["images"], original_labels=task["labels"],
        attacked_images=attacked_images, attacked_labels=attacked_labels,
        trigger_mask=trigger_mask, trigger_pattern=trigger_pattern,
        target_label=target_label, poison_budget=poison_budget,
        trigger_size=cfg.trigger_size,
    )

    if not av.ok:
        m = Metrics(
            attack_valid=False, attack_violations=av.violations,
            n_train=len(split.train), n_poison=av.n_modified,
        )
        return EvalResult(attack_name, defense_name, target_label, seed, m,
                           elapsed_sec=time.time() - start, config_mode=cfg.mode,
                           error="attack 校验未通过")

    n_poison = av.n_modified

    # ---- clean 模型 ----
    clean_model = cache.get_or_train_clean(split, env, seed)

    # ---- attack 模型 ----
    attack_model = cache.get_or_train_attack(
        attacked_images, attacked_labels, env, attack_name, target_label, seed,
    )

    # ---- 测试集 ----
    clean_test = split.test
    triggered_test = build_triggered_test_set(clean_test, target_label, trigger_mask, trigger_pattern)

    acc_clean = accuracy(clean_model.predict(clean_test.images), clean_test.labels)
    acc_attack = accuracy(attack_model.predict(clean_test.images), clean_test.labels)

    if len(triggered_test) > 0:
        asr_clean = asr(clean_model.predict(triggered_test.images), target_label)
        asr_attack = asr(attack_model.predict(triggered_test.images), target_label)
    else:
        asr_clean = asr_attack = 0.0

    # ---- 防御 ----
    train_for_defense = {"images": attacked_images, "labels": attacked_labels}
    defense_out, derr = _run_defense_fn(defend_fn, env, train_for_defense)
    if derr is not None:
        m = Metrics(
            attack_valid=True, n_poison=n_poison, n_train=len(split.train),
            acc_clean=acc_clean, acc_attack=acc_attack,
            asr_clean=asr_clean, asr_attack=asr_attack,
            backdoor_gain=_bg(asr_attack, asr_clean),
            clean_retention_attack=_ret(acc_attack, acc_clean),
            attack_score=_asc(asr_attack, asr_clean, acc_attack, acc_clean),
            defense_valid=False, defense_violations=[derr],
        )
        # 资格线
        m.attack_qualified, m.attack_acc_drop = _qual(asr_attack, acc_clean, acc_attack, cfg)
        return EvalResult(attack_name, defense_name, target_label, seed, m,
                           elapsed_sec=time.time() - start, config_mode=cfg.mode, error=derr)

    san_images = np.asarray(defense_out["images"], dtype=np.float32)
    san_labels = np.asarray(defense_out["labels"]).reshape(-1)
    san_weights = defense_out.get("sample_weights", None)

    dv: ValidationResult = validate_defense(
        poisoned_images=attacked_images, poisoned_labels=attacked_labels,
        sanitized_images=san_images, sanitized_labels=san_labels,
        sample_weights=san_weights, min_ratio=cfg.min_sanitized_ratio,
        num_classes=cfg.num_classes,
    )

    if not dv.ok:
        # 防御非法：仍给出攻击侧指标，防御侧记为非法
        m = assemble_metrics(
            acc_clean=acc_clean, acc_attack=acc_attack, acc_defense=0.0,
            asr_clean=asr_clean, asr_attack=asr_attack, asr_defense=0.0,
            n_triggered_test=len(triggered_test), n_train=len(split.train), n_poison=n_poison,
            asr_skip_eps=cfg.asr_skip_eps,
            attack_qualify_asr=cfg.attack_qualify_asr, attack_qualify_acc_drop=cfg.attack_qualify_acc_drop,
            defense_success_asr=cfg.defense_success_asr, defense_success_acc_drop=cfg.defense_success_acc_drop,
            attack_valid=True, defense_valid=False, defense_violations=dv.violations,
        )
        return EvalResult(attack_name, defense_name, target_label, seed, m,
                           elapsed_sec=time.time() - start, config_mode=cfg.mode,
                           error="defense 校验未通过")

    # ---- defense 模型 ----
    defense_model = env.train(san_images, san_labels, sample_weights=san_weights)
    acc_defense = accuracy(defense_model.predict(clean_test.images), clean_test.labels)
    if len(triggered_test) > 0:
        asr_defense = asr(defense_model.predict(triggered_test.images), target_label)
    else:
        asr_defense = 0.0

    m = assemble_metrics(
        acc_clean=acc_clean, acc_attack=acc_attack, acc_defense=acc_defense,
        asr_clean=asr_clean, asr_attack=asr_attack, asr_defense=asr_defense,
        n_triggered_test=len(triggered_test), n_train=len(split.train), n_poison=n_poison,
        asr_skip_eps=cfg.asr_skip_eps,
        attack_qualify_asr=cfg.attack_qualify_asr, attack_qualify_acc_drop=cfg.attack_qualify_acc_drop,
        defense_success_asr=cfg.defense_success_asr, defense_success_acc_drop=cfg.defense_success_acc_drop,
        attack_valid=True, defense_valid=True,
    )

    return EvalResult(attack_name, defense_name, target_label, seed, m,
                      elapsed_sec=time.time() - start, config_mode=cfg.mode)


# ---- 指标小工具（避免循环依赖）----
def _bg(asr_attack, asr_clean):
    from .metrics import backdoor_gain
    return backdoor_gain(asr_attack, asr_clean)


def _ret(acc_model, acc_clean):
    from .metrics import clean_retention
    return clean_retention(acc_model, acc_clean)


def _asc(asr_attack, asr_clean, acc_attack, acc_clean):
    from .metrics import attack_score
    return attack_score(_bg(asr_attack, asr_clean), _ret(acc_attack, acc_clean))


def _qual(asr_attack, acc_clean, acc_attack, cfg: EvalConfig):
    from .metrics import check_attack_qualified
    return check_attack_qualified(
        asr_attack, acc_clean, acc_attack,
        asr_thresh=cfg.attack_qualify_asr, acc_drop_thresh=cfg.attack_qualify_acc_drop,
    )


# ------------------------------------------------------------------ 全对阵矩阵


def run_matrix(
    attacks: Dict[str, Callable],
    defenses: Dict[str, Callable],
    target_labels: List[int],
    seeds: Optional[List[int]] = None,
    config: Optional[EvalConfig] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> MatrixResult:
    """执行 A × D × T × S 全对阵。

    attacks: {name: fn}
    defenses: {name: fn}
    target_labels: 目标标签列表（第八节，多标签平均）
    seeds: 随机种子列表，默认 [0]
    """
    cfg = config or get_config("fast")
    seeds = seeds or [0]
    results: List[EvalResult] = []

    total = len(attacks) * len(defenses) * len(target_labels) * len(seeds)
    done = 0

    for seed in seeds:
        # 每个种子复用同一份数据集与 clean 模型缓存
        split = load_split(
            train_subset=cfg.train_subset,
            val_subset=cfg.val_subset,
            test_subset=cfg.test_subset,
            val_split=cfg.val_split,
            seed=seed,
        )
        env = RefereeEnv(config=cfg, seed=seed)
        cache = _ModelCache()

        for target_label in target_labels:
            for atk_name, atk_fn in attacks.items():
                for def_name, def_fn in defenses.items():
                    done += 1
                    if progress_callback:
                        msg = f"[{done}/{total}] A={atk_name} D={def_name} t={target_label} s={seed}"
                        progress_callback(done, total, msg)
                    res = evaluate(
                        attack_fn=atk_fn, defend_fn=def_fn,
                        target_label=target_label, seed=seed,
                        config=cfg, split=split, env=env,
                        attack_name=atk_name, defense_name=def_name,
                        cache=cache,
                    )
                    results.append(res)

    matrix = MatrixResult(
        attacks=list(attacks.keys()),
        defenses=list(defenses.keys()),
        target_labels=list(target_labels),
        seeds=list(seeds),
        results=results,
    )
    _aggregate_matrix(matrix)
    return matrix


def _aggregate_matrix(matrix: MatrixResult) -> None:
    """按 target+seed 平均，构建 attack×defense 的得分矩阵。

    - attack_score_matrix[a][d]: 攻击 a 对阵防御 d 的平均 AttackScore。
    - defense_score_matrix[a][d]: 同组的平均 DefenseScore。
    - asr_attack_matrix[a][d] / asr_defense_matrix[a][d]: 平均 ASR。
    """
    from collections import defaultdict
    sums = defaultdict(lambda: defaultdict(list))
    for r in matrix.results:
        if not r.metrics.attack_valid:
            # 攻击非法：攻击侧记 0，防御侧跳过
            sums[r.attack][r.defense].append({
                "attack_score": 0.0, "defense_score": None,
                "asr_attack": 0.0, "asr_defense": None,
                "attack_valid": False, "defense_valid": r.metrics.defense_valid,
            })
            continue
        ds = None if r.metrics.defense_score_skipped or not r.metrics.defense_valid else r.metrics.defense_score
        adr = None if not r.metrics.defense_valid else r.metrics.asr_defense
        sums[r.attack][r.defense].append({
            "attack_score": r.metrics.attack_score,
            "defense_score": ds,
            "asr_attack": r.metrics.asr_attack,
            "asr_defense": adr,
            "attack_valid": True, "defense_valid": r.metrics.defense_valid,
        })

    for a in matrix.attacks:
        matrix.attack_score_matrix.setdefault(a, {})
        matrix.defense_score_matrix.setdefault(a, {})
        matrix.asr_attack_matrix.setdefault(a, {})
        matrix.asr_defense_matrix.setdefault(a, {})
        for d in matrix.defenses:
            rows = sums.get(a, {}).get(d, [])
            if not rows:
                matrix.attack_score_matrix[a][d] = 0.0
                matrix.defense_score_matrix[a][d] = 0.0
                matrix.asr_attack_matrix[a][d] = 0.0
                matrix.asr_defense_matrix[a][d] = 0.0
                continue
            n = len(rows)
            matrix.attack_score_matrix[a][d] = round(
                sum(r["attack_score"] for r in rows) / n, 4)
            matrix.asr_attack_matrix[a][d] = round(
                sum(r["asr_attack"] for r in rows) / n, 4)
            ds_vals = [r["defense_score"] for r in rows if r["defense_score"] is not None]
            matrix.defense_score_matrix[a][d] = round(
                sum(ds_vals) / len(ds_vals), 4) if ds_vals else 0.0
            ad_vals = [r["asr_defense"] for r in rows if r["asr_defense"] is not None]
            matrix.asr_defense_matrix[a][d] = round(
                sum(ad_vals) / len(ad_vals), 4) if ad_vals else 0.0
