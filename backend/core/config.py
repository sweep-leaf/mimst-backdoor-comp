"""默认训练 / 评测配置。

提供两档预设：
- ``fast``（默认）：MNIST 子集 + 少量 epoch，单组攻防几秒内完成，便于快速试错。
- ``full``：完整 MNIST + 较多 epoch，指标更接近真实赛场。

所有参数均可通过 :class:`EvalConfig` 自定义。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TrainConfig:
    """统一模型训练超参（第九节统一训练接口的 ``config``）。"""

    model: str = "lenet"            # "lenet" (Model-A) 或 "small_cnn" (Model-B)
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"         # "adam" | "sgd"
    seed: int = 0
    num_workers: int = 0
    log_every: int = 0              # 0 表示不打印训练日志

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalConfig:
    """评测管线配置。"""

    # 数据规模
    mode: str = "fast"              # "fast" | "full" | "custom"
    train_subset: int = 6000        # fast 模式训练子集大小（<=0 表示全量）
    val_subset: int = 1000          # 公开验证集子集
    test_subset: int = 1000         # 隐藏测试集子集
    val_split: int = 0              # 从训练集中切出的验证集数量（0=不单独切，直接用测试集做 val）

    # 训练
    train: TrainConfig = field(default_factory=TrainConfig)

    # 评测
    poison_rate: float = 0.05       # 投毒预算比例（第七节，<=5%）
    trigger_size: int = 2           # 触发器边长（第七节，固定 2）
    num_classes: int = 10
    asr_skip_eps: float = 1e-3      # |ASR_attack - ASR_clean| < eps 时该对阵不参与防御评分
    min_sanitized_ratio: float = 0.9  # 防御后样本数下限（第十七节）

    # 资格线 / 参考线（第二十三、二十七节）
    attack_qualify_asr: float = 0.60
    attack_qualify_acc_drop: float = 0.03
    defense_success_asr: float = 0.20
    defense_success_acc_drop: float = 0.03

    # 复现
    seed: int = 0

    @classmethod
    def fast(cls, **overrides) -> "EvalConfig":
        cfg = cls(mode="fast", train_subset=6000, val_subset=1000, test_subset=1000)
        cfg.train = TrainConfig(model="lenet", epochs=3, batch_size=128, lr=1e-3, seed=cfg.seed)
        for k, v in overrides.items():
            _set_nested(cfg, k, v)
        return cfg

    @classmethod
    def full(cls, **overrides) -> "EvalConfig":
        cfg = cls(mode="full", train_subset=0, val_subset=0, test_subset=0)
        cfg.train = TrainConfig(model="lenet", epochs=8, batch_size=128, lr=1e-3, seed=cfg.seed)
        for k, v in overrides.items():
            _set_nested(cfg, k, v)
        return cfg

    def with_seed(self, seed: int) -> "EvalConfig":
        """返回一份带有指定种子（同时同步训练种子）的副本。"""
        import copy
        cfg = copy.deepcopy(self)
        cfg.seed = seed
        cfg.train.seed = seed
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _set_nested(cfg: EvalConfig, key: str, value) -> None:
    """支持 ``train.epochs=5`` 这种点号路径覆写。"""
    if "." in key:
        head, tail = key.split(".", 1)
        obj = getattr(cfg, head)
        setattr(obj, tail, value)
    else:
        setattr(cfg, key, value)


def get_config(mode: str = "fast", **overrides) -> EvalConfig:
    """根据模式名获取配置。``mode`` 为 'fast' / 'full'，其余视为 custom。"""
    mode = (mode or "fast").lower()
    if mode == "full":
        return EvalConfig.full(**overrides)
    return EvalConfig.fast(**overrides)
