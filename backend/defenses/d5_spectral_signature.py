"""D5 谱签名异常检测防御（Tran et al. 2018 思路）。

用轻量代理模型提取特征，对每个类别估计谱签名（top奇异方向），
将投影异常的样本视为可疑后门样本并降权，不删除样本以满足
``N_sanitized ≥ 0.9 N_train``。
"""
from __future__ import annotations

import numpy as np

from .base import DefenseBase


class D5SpectralSignature(DefenseBase):
    """基于谱签名的可疑样本降权防御。"""

    name = "d5_spectral_signature"
    suspect_ratio: float = 0.05   # 每个类别中降权的可疑样本比例上限
    weight_floor: float = 0.1     # 可疑样本权重
    proxy_epochs: int = 1

    def process(self, env, images, labels):
        proxy = env.train_proxy(images, labels, epochs=self.proxy_epochs)
        feats = self._extract_features(proxy, images)
        feats = feats.reshape(feats.shape[0], -1).astype(np.float32)

        weights = np.ones(len(labels), dtype=np.float32)
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) < 4:
                continue
            f = feats[idx]
            f_c = f - f.mean(axis=0, keepdims=True)
            # 谱签名：top 奇异方向
            try:
                u, s, vt = np.linalg.svd(f_c, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            if vt.shape[0] == 0:
                continue
            direction = vt[0]
            proj = f_c @ direction
            scores = np.abs(proj - proj.mean())
            # 取分数最高的 suspect_ratio 比例样本降权
            k = max(1, int(np.ceil(self.suspect_ratio * len(idx))))
            top = np.argsort(-scores)[:k]
            sus_local = idx[top]
            weights[sus_local] = self.weight_floor
        return images.copy(), labels.copy(), weights

    @staticmethod
    def _extract_features(proxy, images):
        """提取代理模型倒数第二层特征。"""
        import torch
        model = proxy.model
        model.eval()
        feats = []
        with torch.no_grad():
            # 走到分类器之前的特征
            x = proxy._to_tensor(images)
            m = model
            if hasattr(m, "features"):
                h = m.features(x)
                h = h.flatten(1)
            elif hasattr(m, "net"):
                # TinyMLP: 取除最后线性层外的输出
                layers = list(m.net.children())
                head = torch.nn.Sequential(*layers[:-1])
                h = head(x).flatten(1)
            else:
                h = x.flatten(1)
            feats.append(h.cpu().numpy())
        return np.concatenate(feats, axis=0)
