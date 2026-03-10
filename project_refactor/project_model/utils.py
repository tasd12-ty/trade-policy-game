"""通用数值工具。"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def safe_log(x: np.ndarray | float) -> np.ndarray:
    """安全对数：log(max(x, EPS))。"""
    return np.log(np.clip(np.asarray(x, dtype=float), EPS, None))


def relative_error(actual: np.ndarray | float, expected: np.ndarray | float) -> np.ndarray:
    """相对误差缩放，减少量纲影响。"""
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(e)), 1.0)
    return (a - e) / scale


def clamp_positive(x: np.ndarray | float, min_value: float = EPS) -> np.ndarray:
    """截断到正区间。"""
    return np.clip(np.asarray(x, dtype=float), min_value, None)
