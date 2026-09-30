"""最小攻击方示例（第三十一节）：右下角 2×2 白色触发器。"""
import numpy as np


def attack(env, task):
    images = task["images"].copy()
    labels = task["labels"].copy()

    target_label = int(task["target_label"])
    budget = int(task["poison_budget"])

    _, _, height, width = images.shape

    trigger_mask = np.zeros((1, height, width), dtype=np.float32)
    trigger_pattern = np.zeros((1, height, width), dtype=np.float32)

    y1, y2 = height - 3, height - 1
    x1, x2 = width - 3, width - 1
    trigger_mask[:, y1:y2, x1:x2] = 1.0
    trigger_pattern[:, y1:y2, x1:x2] = 1.0

    candidates = np.where(labels != target_label)[0]
    n = min(budget, len(candidates))
    rng = np.random.default_rng(42)
    poison_idx = rng.choice(candidates, size=n, replace=False)

    m = trigger_mask[None, ...]
    images[poison_idx] = images[poison_idx] * (1.0 - m) + trigger_pattern[None, ...] * m
    labels[poison_idx] = target_label

    return {
        "images": images,
        "labels": labels,
        "trigger_mask": trigger_mask,
        "trigger_pattern": trigger_pattern,
    }
