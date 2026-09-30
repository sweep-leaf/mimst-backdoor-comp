"""最小防御方示例（第三十二节）：3×3 中值滤波。"""
import numpy as np


def median_filter_3x3(images):
    n, c, h, w = images.shape
    padded = np.pad(images, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant", constant_values=0)
    shifted = []
    for dy in range(3):
        for dx in range(3):
            shifted.append(padded[:, :, dy:dy + h, dx:dx + w])
    stacked = np.stack(shifted, axis=0)
    return np.median(stacked, axis=0).astype(images.dtype)


def defend(env, train):
    images = train["images"]
    labels = train["labels"].copy()
    return {
        "images": median_filter_3x3(images),
        "labels": labels,
        "sample_weights": None,
    }
