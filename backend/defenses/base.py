"""防御基类。"""
from __future__ import annotations

from typing import Optional

import numpy as np


class DefenseBase:
    """防御方法基类。

    子类实现 :meth:`process`，返回 (sanitized_images, sanitized_labels, sample_weights)。
    类属性 ``name`` 为注册名。
    """

    name: str = "base"

    def process(self, env, images: np.ndarray, labels: np.ndarray
                ) -> tuple:
        """返回 (images, labels, sample_weights)。子类必须实现。"""
        raise NotImplementedError

    def run(self, env, train: dict) -> dict:
        """统一 defend 接口（第十八节）。"""
        images = np.asarray(train["images"], dtype=np.float32)
        labels = np.asarray(train["labels"]).copy()
        if images.ndim == 3:
            images = images[:, None, :, :]
        out = self.process(env, images, labels)
        san_img, san_lab, weights = self._normalize_out(out, images, labels)
        return {
            "images": san_img,
            "labels": san_lab,
            "sample_weights": weights,
        }

    @staticmethod
    def _normalize_out(out, images, labels):
        if isinstance(out, tuple):
            if len(out) == 3:
                img, lab, w = out
            elif len(out) == 2:
                img, lab = out
                w = None
            else:
                raise ValueError("process 返回的元组长度应为 2 或 3")
        elif isinstance(out, dict):
            img = out.get("images", images)
            lab = out.get("labels", labels)
            w = out.get("sample_weights", None)
        else:
            raise TypeError("process 返回值应为 tuple 或 dict")
        img = np.asarray(img, dtype=np.float32)
        lab = np.asarray(lab).reshape(-1)
        if img.ndim == 3:
            img = img[:, None, :, :]
        if w is not None:
            w = np.asarray(w, dtype=np.float32).reshape(-1)
        return img, lab, w
